"""
processing/ocr_corrector.py — Post-correction OCR couche 2 : ByT5
=================================================================
Deuxième couche de la pipeline post-OCR.

Usage :
    python processing/ocr_corrector.py --input data/processed/gallica/gallica_bpt6k123_clean.txt
    python processing/ocr_corrector.py --batch --source gallica
    python processing/ocr_corrector.py --batch --force
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR   = Path("data/processed")
BYT5_MODEL_NAME = os.getenv("BYT5_MODEL", "google/byt5-base")
OCR_THRESHOLD   = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.85"))

CHUNK_SIZE       = 512
CHUNK_STRIDE     = 50
BATCH_SIZE       = 8
MAX_LENGTH_RATIO = 1.3

# ── Chargement du modèle (singleton) ─────────────────────────────────────────

_model     = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        logger.info(f"Chargement du modèle ByT5 : {BYT5_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(BYT5_MODEL_NAME)
        _model     = T5ForConditionalGeneration.from_pretrained(BYT5_MODEL_NAME)
        _model.eval()
        logger.success(f"Modèle ByT5 chargé ✓")
        return _model, _tokenizer
    except ImportError:
        logger.error("transformers non installé. Exécuter : pip install transformers")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Impossible de charger ByT5 : {e}")
        sys.exit(1)

# ── Découpage en chunks ───────────────────────────────────────────────────────

def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = max(start + 1, cut - stride)
    return [c for c in chunks if c.strip()]


def _reassemble_chunks(original_chunks: list[str], corrected_chunks: list[str]) -> str:
    if not corrected_chunks:
        return ""
    if len(corrected_chunks) == 1:
        return corrected_chunks[0]
    result = corrected_chunks[0]
    for i in range(1, len(corrected_chunks)):
        chunk   = corrected_chunks[i]
        overlap = min(CHUNK_STRIDE * 2, len(result), len(chunk))
        tail    = result[-overlap:]
        head    = chunk[:overlap]
        join_pos = 0
        for length in range(overlap, 2, -1):
            if tail.endswith(head[:length]):
                join_pos = length
                break
        result += chunk[join_pos:]
    return result

# ── Correction ByT5 ───────────────────────────────────────────────────────────

def _correct_chunks_batch(chunks: list[str]) -> list[str]:
    import torch
    model, tokenizer = _load_model()
    corrected = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch   = chunks[i : i + BATCH_SIZE]
        inputs  = [f"fix ocr: {chunk}" for chunk in batch]
        encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        with torch.no_grad():
            max_new_tokens = int(max(len(enc) for enc in encoded["input_ids"]) * MAX_LENGTH_RATIO)
            outputs = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrected.extend(decoded)

    return corrected


def _sanity_check(original: str, corrected: str) -> str:
    if not corrected or not corrected.strip():
        return original
    ratio = len(corrected) / max(len(original), 1)
    if ratio < 0.6:
        return original
    english_markers = {"the", "and", "that", "this", "with", "from", "have", "been"}
    french_markers  = {"le", "la", "les", "de", "du", "et", "en", "un", "une", "que"}
    words    = set(corrected.lower().split()[:50])
    en_score = len(words & english_markers)
    fr_score = len(words & french_markers)
    if en_score > fr_score and en_score >= 3:
        return original
    return corrected


def correct_text(text: str) -> tuple[str, float]:
    from processing.clean import compute_ocr_quality

    chunks           = _split_into_chunks(text)
    corrected_chunks = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Correction ByT5", unit="batch"):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            batch_corrected = _correct_chunks_batch(batch)
        except Exception as e:
            logger.warning(f"Erreur batch {i} — conservation de l'original : {e}")
            batch_corrected = batch
        for orig, corr in zip(batch, batch_corrected):
            corrected_chunks.append(_sanity_check(orig, corr))

    corrected_text = _reassemble_chunks(chunks, corrected_chunks)
    score_after    = compute_ocr_quality(corrected_text)
    return corrected_text, score_after

# ── Sauvegarde et base de données ─────────────────────────────────────────────

def _save_corrected(text: str, book_id: str, source: str) -> Path:
    out_dir  = PROCESSED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{book_id}_corrected.txt"
    filepath.write_text(text, encoding="utf-8")
    return filepath


def _update_book_corrected(
    conn: psycopg2.extensions.connection,
    book_id: str,
    processed_path: str,
    score_after: float,
) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE books SET
            processed_path    = %s,
            ocr_quality_after = %s,
            ocr_corrected     = 1
        WHERE book_id = %s
    """, (processed_path, score_after, book_id))
    cur.close()


def _get_books_to_process(
    conn: psycopg2.extensions.connection,
    source_filter: str | None,
    force: bool,
) -> list[dict]:
    query  = """
        SELECT book_id, processed_path, ocr_quality_after, source
        FROM books
        WHERE processed_path IS NOT NULL
          AND ocr_corrected = 0
    """
    params = []
    if not force:
        query += " AND (ocr_quality_after IS NULL OR ocr_quality_after < %s)"
        params.append(OCR_THRESHOLD)
    if source_filter:
        query += " AND source = %s"
        params.append(source_filter)

    cur  = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    return [{"book_id": r[0], "path": r[1], "ocr_score": r[2], "source": r[3]} for r in rows]

# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_file(
    input_path: Path,
    conn: psycopg2.extensions.connection | None = None,
    update_db: bool = True,
    force: bool = False,
) -> dict:
    stem    = input_path.stem
    book_id = re.sub(r"_clean$", "", stem)
    source  = input_path.parent.name

    if not force and conn:
        cur = conn.cursor()
        cur.execute("SELECT ocr_quality_after, ocr_corrected FROM books WHERE book_id = %s", (book_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            score, already_corrected = row
            if already_corrected:
                logger.debug(f"{book_id} déjà corrigé — ignoré")
                return {"book_id": book_id, "status": "already_corrected"}
            if score is not None and score >= OCR_THRESHOLD:
                logger.info(f"{book_id} score OCR ({score:.3f}) >= seuil — correction non nécessaire")
                return {"book_id": book_id, "status": "skipped_good_quality", "score": score}

    logger.info(f"Correction ByT5 : {book_id}")

    text = input_path.read_text(encoding="utf-8", errors="replace")
    corrected_text, score_after = correct_text(text)
    output_path = _save_corrected(corrected_text, book_id, source)

    if update_db and conn:
        try:
            _update_book_corrected(conn, book_id, str(output_path), score_after)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur mise à jour DB pour {book_id} : {e}")

    logger.success(f"✓ {book_id} — score OCR après correction : {score_after:.3f}")
    return {"book_id": book_id, "status": "ok", "score_after": score_after, "output_path": str(output_path)}

# ── Mode batch ────────────────────────────────────────────────────────────────

def process_batch(
    source_filter: str | None = None,
    force: bool = False,
    update_db: bool = True,
) -> list[dict]:
    if not verify_db():
        logger.error("Base non initialisée.")
        sys.exit(1)

    conn  = get_connection()
    books = _get_books_to_process(conn, source_filter, force)

    if not books:
        logger.info(f"Aucun livre à corriger (seuil={OCR_THRESHOLD})")
        conn.close()
        return []

    logger.info(f"{len(books)} livres à corriger")
    _load_model()

    results = []
    for book in books:
        path = Path(book["path"])
        if not path.exists():
            logger.warning(f"Fichier introuvable : {path}")
            results.append({"book_id": book["book_id"], "status": "file_not_found"})
            continue
        try:
            result = process_file(path, conn=conn, update_db=update_db, force=force)
            results.append(result)
        except Exception as e:
            logger.error(f"Erreur sur {book['book_id']} : {e}")
            results.append({"book_id": book["book_id"], "status": "error", "error": str(e)})

    conn.close()

    ok      = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if "skipped" in r.get("status", ""))
    errors  = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Batch ByT5 terminé : {ok} OK / {skipped} ignorés / {errors} erreurs")
    return results

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-correction OCR couche 2 : ByT5.")
    parser.add_argument("--input",  default=None)
    parser.add_argument("--batch",  action="store_true")
    parser.add_argument("--source", default=None, choices=["gallica", "gutenberg", "wikisource"])
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--no-db",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.batch:
        process_batch(source_filter=args.source, force=args.force, update_db=not args.no_db)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Fichier introuvable : {input_path}")
            sys.exit(1)
        conn = None
        if not args.no_db:
            if not verify_db():
                logger.error("Base non initialisée.")
                sys.exit(1)
            conn = get_connection()
        result = process_file(input_path, conn=conn, update_db=not args.no_db, force=args.force)
        if conn:
            conn.close()
        logger.info(f"Résultat : {result}")
    else:
        logger.error("Spécifier --input <fichier> ou --batch")
        sys.exit(1)

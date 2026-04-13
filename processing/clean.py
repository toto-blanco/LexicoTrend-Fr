"""
processing/clean.py — Post-correction OCR couche 1 : nettoyage regex
=====================================================================
Première couche de la pipeline post-OCR.

Usage :
    python processing/clean.py --input data/raw/gallica/gallica_bpt6k123.txt
    python processing/clean.py --input data/raw/ --batch
    python processing/clean.py --input data/raw/ --batch --source gutenberg
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DATA_DIR  = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MIN_TEXT_LENGTH = 10_000

# ── Patterns de nettoyage ─────────────────────────────────────────────────────

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\ufeff]", re.UNICODE)
_PAGE_NUMBERS = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_RUNNING_HEADERS = re.compile(
    r"^.{0,60}(gallica\.bnf\.fr|bibliothèque nationale de france|"
    r"project gutenberg|gutenberg\.org|"
    r"bibliothèque de la pléiade|"
    r"reproduction numérique|numérisé par la bnf|"
    r"tous droits réservés).{0,60}$",
    re.IGNORECASE | re.MULTILINE,
)
_OCR_DASHES          = re.compile(r"[-‒–—―]{2,}")
_APOSTROPHE_VARIANTS = re.compile(r"[ʼʻ\u2018`´ʹ]")
_QUOTE_OPEN_VARIANTS  = re.compile(r"[‹❮❝〈⟨]")
_QUOTE_CLOSE_VARIANTS = re.compile(r"[›❯❞〉⟩]")
_SPACE_BEFORE_PUNCT   = re.compile(r" +([,;:!?.])")
_MULTI_SPACES         = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINES       = re.compile(r"\n{3,}")
_SHORT_ISOLATED_LINE  = re.compile(r"(?<=\n)(.{1,3})\n(?=[A-ZÀÂÇÉÈÊËÎÏÔÙÛÜ])", re.UNICODE)

# ── Fonctions de nettoyage ────────────────────────────────────────────────────

def remove_control_chars(text: str) -> str:
    return _CTRL_CHARS.sub("", text)

def remove_page_numbers(text: str) -> str:
    return _PAGE_NUMBERS.sub("", text)

def remove_running_headers(text: str) -> str:
    return _RUNNING_HEADERS.sub("", text)

def normalize_dashes(text: str) -> str:
    return _OCR_DASHES.sub("—", text)

def normalize_quotes(text: str) -> str:
    text = _APOSTROPHE_VARIANTS.sub("'", text)
    text = _QUOTE_OPEN_VARIANTS.sub("«", text)
    text = _QUOTE_CLOSE_VARIANTS.sub("»", text)
    return text

def normalize_spaces(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text

def remove_short_isolated_lines(text: str) -> str:
    return _SHORT_ISOLATED_LINE.sub(r"", text)


def compute_ocr_quality(text: str) -> float:
    """Score de qualité OCR (0–1) basé sur le ratio de tokens suspects."""
    tokens = text.split()
    if not tokens:
        return 0.0

    suspicious = 0
    for token in tokens:
        if len(token) <= 2:
            continue
        non_alpha = len(re.sub(r"[a-zA-ZÀ-ÿ'\-]", "", token))
        if non_alpha > len(token) * 0.5:
            suspicious += 1
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", token.lower()):
            suspicious += 1
            continue
        if re.search(r"[a-zA-ZÀ-ÿ]\d|\d[a-zA-ZÀ-ÿ]", token):
            suspicious += 1

    quality = 1.0 - (suspicious / len(tokens))
    return round(max(0.0, min(1.0, quality)), 4)


def clean_text(raw_text: str) -> tuple[str, float, float]:
    """
    Applique la pipeline complète de nettoyage regex (couche 1).

    Returns:
        Tuple (texte_nettoyé, score_ocr_avant, score_ocr_après).
    """
    score_before = compute_ocr_quality(raw_text)

    text = raw_text
    text = remove_control_chars(text)
    text = remove_running_headers(text)
    text = remove_page_numbers(text)
    text = normalize_dashes(text)
    text = normalize_quotes(text)
    text = remove_short_isolated_lines(text)
    text = normalize_spaces(text)
    text = text.strip()

    score_after = compute_ocr_quality(text)
    return text, score_before, score_after

# ── Sauvegarde et base de données ─────────────────────────────────────────────

def _save_processed(text: str, book_id: str, source: str) -> Path:
    out_dir = PROCESSED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{book_id}_clean.txt"
    filepath.write_text(text, encoding="utf-8")
    return filepath


def _update_book_clean(
    conn: psycopg2.extensions.connection,
    book_id: str,
    processed_path: str,
    score_before: float,
    score_after: float,
) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE books SET
            processed_path     = %s,
            ocr_quality_before = %s,
            ocr_quality_after  = %s
        WHERE book_id = %s
    """, (processed_path, score_before, score_after, book_id))
    cur.close()

# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_file(
    input_path: Path,
    conn: psycopg2.extensions.connection | None = None,
    update_db: bool = True,
) -> dict:
    book_id = input_path.stem
    source  = input_path.parent.name

    logger.info(f"Nettoyage : {book_id}")

    raw_text      = input_path.read_text(encoding="utf-8", errors="replace")
    length_before = len(raw_text.split())

    cleaned_text, score_before, score_after = clean_text(raw_text)
    length_after = len(cleaned_text.split())

    if len(cleaned_text) < MIN_TEXT_LENGTH:
        logger.warning(f"{book_id} trop court après nettoyage ({len(cleaned_text):,} chars) — ignoré")
        return {"book_id": book_id, "status": "skipped_too_short",
                "score_before": score_before, "score_after": score_after}

    output_path = _save_processed(cleaned_text, book_id, source)

    if update_db and conn:
        try:
            _update_book_clean(conn, book_id, str(output_path), score_before, score_after)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur mise à jour DB pour {book_id} : {e}")

    improvement = score_after - score_before
    logger.success(
        f"✓ {book_id} — {length_before:,} → {length_after:,} mots — "
        f"OCR: {score_before:.3f} → {score_after:.3f} "
        f"({'+'if improvement >= 0 else ''}{improvement:.3f})"
    )

    return {
        "book_id":       book_id,
        "status":        "ok",
        "score_before":  score_before,
        "score_after":   score_after,
        "output_path":   str(output_path),
        "length_before": length_before,
        "length_after":  length_after,
    }

# ── Mode batch ────────────────────────────────────────────────────────────────

def process_batch(
    input_dir: Path,
    source_filter: str | None = None,
    update_db: bool = True,
) -> list[dict]:
    if update_db:
        if not verify_db():
            logger.error("Base non initialisée. Exécuter : python init_db.py")
            sys.exit(1)
        conn = get_connection()
    else:
        conn = None

    pattern = f"**/{source_filter}/*.txt" if source_filter else "**/*.txt"
    files   = list(input_dir.glob(pattern))

    if not files:
        logger.warning(f"Aucun fichier .txt trouvé dans {input_dir}")
        return []

    logger.info(f"{len(files)} fichiers à traiter")

    results = []
    for file_path in sorted(files):
        try:
            result = process_file(file_path, conn=conn, update_db=update_db)
            results.append(result)
        except Exception as e:
            logger.error(f"Erreur sur {file_path.name} : {e}")
            results.append({"book_id": file_path.stem, "status": "error", "error": str(e)})

    if conn:
        conn.close()

    ok      = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped_too_short")
    errors  = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Batch terminé : {ok} OK / {skipped} ignorés / {errors} erreurs")
    return results

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-correction OCR couche 1 : nettoyage regex.")
    parser.add_argument("--input",   required=True)
    parser.add_argument("--batch",   action="store_true")
    parser.add_argument("--source",  default=None, choices=["gallica", "gutenberg", "wikisource"])
    parser.add_argument("--no-db",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            logger.error(f"--batch nécessite un répertoire : {input_path}")
            sys.exit(1)
        process_batch(input_dir=input_path, source_filter=args.source, update_db=not args.no_db)
    else:
        if not input_path.is_file():
            logger.error(f"Fichier introuvable : {input_path}")
            sys.exit(1)
        conn = None
        if not args.no_db:
            if not verify_db():
                logger.error("Base non initialisée.")
                sys.exit(1)
            conn = get_connection()
        result = process_file(input_path, conn=conn, update_db=not args.no_db)
        if conn:
            conn.close()
        logger.info(f"Résultat : {result}")

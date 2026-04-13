"""
processing/metrics.py — Calcul des métriques de richesse lexicale
=================================================================
Troisième étape de la pipeline de traitement.

Usage :
    python processing/metrics.py --input data/processed/gallica/gallica_bpt6k123_corrected.txt
    python processing/metrics.py --batch
    python processing/metrics.py --batch --source gallica
    python processing/metrics.py --book-id gallica_bpt6k123
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

SPACY_MODEL        = os.getenv("SPACY_MODEL", "fr_core_news_md")
PROCESSED_DIR      = Path("data/processed")
MIN_TOKENS_FOR_MTLD = 200
MTLD_THRESHOLD     = 0.72

# ── Chargement spaCy (singleton) ──────────────────────────────────────────────

_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        logger.info(f"Chargement du modèle spaCy : {SPACY_MODEL}")
        _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        _nlp.max_length = 3_000_000
        logger.success(f"Modèle spaCy chargé ✓")
        return _nlp
    except OSError:
        logger.error(f"Modèle '{SPACY_MODEL}' non trouvé. Exécuter : python -m spacy download {SPACY_MODEL}")
        sys.exit(1)
    except ImportError:
        logger.error("spaCy non installé.")
        sys.exit(1)

# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(text: str, lemmatize: bool = False) -> list[str]:
    nlp    = _load_spacy()
    doc    = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_alpha or len(token.text) < 2:
            continue
        tokens.append(token.lemma_.lower() if lemmatize else token.text.lower())
    return tokens

# ── Calcul des métriques ──────────────────────────────────────────────────────

def compute_metrics(tokens: list[str]) -> dict:
    try:
        from lexicalrichness import LexicalRichness
    except ImportError:
        logger.error("lexicalrichness non installé.")
        sys.exit(1)

    n_tokens = len(tokens)
    n_types  = len(set(tokens))

    result = {
        "word_count":   n_tokens,
        "unique_words": n_types,
        "ttr":          None,
        "mtld":         None,
        "hdd":          None,
        "mtld_ma_bid":  None,
    }

    if n_tokens == 0:
        return result

    result["ttr"] = round(n_types / n_tokens, 6)

    if n_tokens < MIN_TOKENS_FOR_MTLD:
        logger.warning(f"Corpus trop court pour MTLD ({n_tokens} tokens) — TTR uniquement")
        return result

    try:
        lex = LexicalRichness(" ".join(tokens))
        try:
            result["mtld"] = round(lex.mtld(threshold=MTLD_THRESHOLD), 4)
        except Exception as e:
            logger.warning(f"MTLD non calculable : {e}")
        try:
            draws = min(42, n_tokens - 1)
            result["hdd"] = round(lex.hdd(draws=draws), 6)
        except Exception as e:
            logger.warning(f"HD-D non calculable : {e}")
        try:
            if hasattr(lex, "mtld_ma_bid"):
                result["mtld_ma_bid"] = round(lex.mtld_ma_bid(threshold=MTLD_THRESHOLD), 4)
            elif hasattr(lex, "mtld_ma_wrap"):
                result["mtld_ma_bid"] = round(lex.mtld_ma_wrap(threshold=MTLD_THRESHOLD), 4)
        except Exception as e:
            logger.debug(f"MTLD_MA_BID non calculable : {e}")
    except Exception as e:
        logger.error(f"Erreur LexicalRichness : {e}")

    return result


def count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+(?:\s+[A-ZÀÂÇÉÈÊËÎÏÔÙÛÜ]|\s*$)", text)
    return max(1, len([s for s in sentences if s.strip()]))

# ── Base de données ───────────────────────────────────────────────────────────

def _update_book_metrics(
    conn: psycopg2.extensions.connection,
    book_id: str,
    metrics: dict,
    sentence_count: int,
) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE books SET
            word_count     = %s,
            unique_words   = %s,
            sentence_count = %s,
            ttr            = %s,
            mtld           = %s,
            hdd            = %s,
            mtld_ma_bid    = %s,
            metrics_at     = %s
        WHERE book_id = %s
    """, (
        metrics["word_count"],
        metrics["unique_words"],
        sentence_count,
        metrics["ttr"],
        metrics["mtld"],
        metrics["hdd"],
        metrics["mtld_ma_bid"],
        datetime.now(timezone.utc).isoformat(),
        book_id,
    ))
    cur.close()


def _get_books_to_process(
    conn: psycopg2.extensions.connection,
    source_filter: str | None,
    force: bool,
) -> list[dict]:
    query  = "SELECT book_id, processed_path, source FROM books WHERE processed_path IS NOT NULL"
    params = []
    if not force:
        query += " AND metrics_at IS NULL"
    if source_filter:
        query += " AND source = %s"
        params.append(source_filter)

    cur  = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    return [{"book_id": r[0], "path": r[1], "source": r[2]} for r in rows]

# ── Traitement d'un fichier ───────────────────────────────────────────────────

def process_file(
    input_path: Path,
    conn: psycopg2.extensions.connection | None = None,
    update_db: bool = True,
    lemmatize: bool = False,
) -> dict:
    stem    = input_path.stem
    book_id = stem.replace("_corrected_l3", "").replace("_corrected", "").replace("_clean", "")

    logger.info(f"Calcul métriques : {book_id}")

    text = input_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        logger.warning(f"{book_id} : fichier vide — ignoré")
        return {"book_id": book_id, "status": "empty_file"}

    tokens = tokenize(text, lemmatize=lemmatize)
    if len(tokens) < 50:
        logger.warning(f"{book_id} : trop peu de tokens ({len(tokens)}) — ignoré")
        return {"book_id": book_id, "status": "too_few_tokens", "token_count": len(tokens)}

    metrics        = compute_metrics(tokens)
    sentence_count = count_sentences(text)

    if update_db and conn:
        try:
            _update_book_metrics(conn, book_id, metrics, sentence_count)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur mise à jour DB pour {book_id} : {e}")

    logger.success(
        f"✓ {book_id} — {metrics['word_count']:,} tokens / "
        f"{metrics['unique_words']:,} types / TTR={metrics['ttr']:.4f} / MTLD={metrics['mtld']}"
    )
    return {"book_id": book_id, "status": "ok", "metrics": metrics, "sentences": sentence_count}

# ── Mode batch ────────────────────────────────────────────────────────────────

def process_batch(
    source_filter: str | None = None,
    force: bool = False,
    update_db: bool = True,
    lemmatize: bool = False,
) -> list[dict]:
    if not verify_db():
        logger.error("Base non initialisée.")
        sys.exit(1)

    conn  = get_connection()
    books = _get_books_to_process(conn, source_filter, force)

    if not books:
        logger.info("Aucun livre à traiter")
        conn.close()
        return []

    logger.info(f"{len(books)} livres à traiter")
    _load_spacy()

    results = []
    for book in tqdm(books, desc="Calcul métriques", unit="livre"):
        path = Path(book["path"])
        if not path.exists():
            logger.warning(f"Fichier introuvable : {path}")
            results.append({"book_id": book["book_id"], "status": "file_not_found"})
            continue
        try:
            result = process_file(path, conn=conn, update_db=update_db, lemmatize=lemmatize)
            results.append(result)
        except Exception as e:
            logger.error(f"Erreur sur {book['book_id']} : {e}")
            results.append({"book_id": book["book_id"], "status": "error", "error": str(e)})

    conn.close()

    ok     = sum(1 for r in results if r.get("status") == "ok")
    errors = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Batch métriques terminé : {ok} OK / {errors} erreurs")

    mtld_values = [
        r["metrics"]["mtld"]
        for r in results
        if r.get("status") == "ok" and r["metrics"].get("mtld") is not None
    ]
    if mtld_values:
        import statistics
        logger.info(
            f"MTLD — min: {min(mtld_values):.1f} / "
            f"médiane: {statistics.median(mtld_values):.1f} / "
            f"max: {max(mtld_values):.1f}"
        )

    return results

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcul des métriques de richesse lexicale.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",    help="Fichier texte à traiter")
    group.add_argument("--batch",    action="store_true")
    group.add_argument("--book-id",  help="Traiter un livre par book_id")
    parser.add_argument("--source",  default=None, choices=["gallica", "gutenberg", "wikisource"])
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--no-db",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch:
        process_batch(
            source_filter=args.source,
            force=args.force,
            update_db=not args.no_db,
            lemmatize=args.lemmatize,
        )
    elif args.book_id:
        if not verify_db():
            sys.exit(1)
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT processed_path FROM books WHERE book_id = %s", (args.book_id,))
        row = cur.fetchone()
        cur.close()
        if not row or not row[0]:
            logger.error(f"book_id '{args.book_id}' introuvable ou sans fichier traité")
            conn.close()
            sys.exit(1)
        result = process_file(Path(row[0]), conn=conn, update_db=not args.no_db, lemmatize=args.lemmatize)
        conn.close()
        logger.info(f"Résultat : {result}")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Fichier introuvable : {input_path}")
            sys.exit(1)
        conn = None
        if not args.no_db:
            if not verify_db():
                sys.exit(1)
            conn = get_connection()
        result = process_file(input_path, conn=conn, update_db=not args.no_db, lemmatize=args.lemmatize)
        if conn:
            conn.close()
        logger.info(f"Résultat : {result}")

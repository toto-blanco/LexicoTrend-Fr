"""
scraping/gutenberg.py — Collecte des textes Project Gutenberg
=============================================================
Récupère les textes français du domaine public depuis Project Gutenberg
via l'API GutendexAPI (https://gutendex.com) — pas de clé requise.

Usage :
    python scraping/gutenberg.py
    python scraping/gutenberg.py --max-books 10 --decade 1880
    python scraping/gutenberg.py --dry-run
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GUTENDEX_BASE_URL = "https://gutendex.com/books"
RAW_DATA_DIR      = Path("data/raw/gutenberg")
USER_AGENT        = os.getenv("GUTENBERG_USER_AGENT", "LexicoTrend-Research/1.0")
REQUEST_DELAY     = float(os.getenv("GUTENBERG_REQUEST_DELAY", "2.0"))

YEAR_MIN = 1850
YEAR_MAX = 1980

GUTENBERG_HEADER_END_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "*END*THE SMALL PRINT",
]
GUTENBERG_FOOTER_START_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
]

# ── Helpers réseau ────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain",
    })
    return session


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _get_json(session: requests.Session, url: str, params: dict = None) -> dict:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.content.decode("latin-1")

# ── Extraction métadonnées ────────────────────────────────────────────────────

def _extract_year(book: dict) -> int | None:
    if book.get("copyright_date"):
        try:
            return int(str(book["copyright_date"])[:4])
        except (ValueError, TypeError):
            pass
    for subject in book.get("subjects", []):
        match = re.search(r"\b(1[89]\d{2})\b", subject)
        if match:
            year = int(match.group(1))
            if YEAR_MIN <= year <= YEAR_MAX:
                return year
    for shelf in book.get("bookshelves", []):
        match = re.search(r"\b(1[89]\d{2})\b", shelf)
        if match:
            year = int(match.group(1))
            if YEAR_MIN <= year <= YEAR_MAX:
                return year
    return None


def _extract_txt_url(book: dict) -> str | None:
    formats = book.get("formats", {})
    for mime in ["text/plain; charset=utf-8", "text/plain; charset=us-ascii"]:
        url = formats.get(mime)
        if url and not url.endswith(".zip"):
            return url
    url = formats.get("text/plain")
    if url and not url.endswith(".zip"):
        return url
    return None


def _infer_genre(book: dict) -> str:
    text = " ".join(book.get("bookshelves", []) + book.get("subjects", [])).lower()
    if any(k in text for k in ["naturalism", "naturalisme", "realism", "réalisme"]):
        return "naturaliste"
    if any(k in text for k in ["historical", "histoire", "historique"]):
        return "historique"
    if any(k in text for k in ["adventure", "aventure", "popular", "populaire"]):
        return "populaire"
    return "roman_realiste"

# ── Nettoyage boilerplate Gutenberg ──────────────────────────────────────────

def strip_gutenberg_boilerplate(text: str) -> str:
    for marker in GUTENBERG_HEADER_END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            end_of_line = text.find("\n", idx)
            text = text[end_of_line + 1:] if end_of_line != -1 else text[idx + len(marker):]
            break
    for marker in GUTENBERG_FOOTER_START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

# ── Sauvegarde locale ─────────────────────────────────────────────────────────

def _save_locally(text: str, book_id: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DATA_DIR / f"{book_id}.txt"
    filepath.write_text(text, encoding="utf-8")
    return filepath

# ── Base de données ───────────────────────────────────────────────────────────

def _book_exists(conn: psycopg2.extensions.connection, book_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM books WHERE book_id = %s", (book_id,))
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def _insert_book(conn: psycopg2.extensions.connection, book_data: dict) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO books (
            book_id, title, author, year, decade, genre,
            source, langue, raw_path, collected_at
        ) VALUES (
            %(book_id)s, %(title)s, %(author)s, %(year)s, %(decade)s, %(genre)s,
            %(source)s, %(langue)s, %(raw_path)s, %(collected_at)s
        )
        ON CONFLICT (book_id) DO UPDATE SET
            raw_path     = EXCLUDED.raw_path,
            collected_at = EXCLUDED.collected_at
    """, book_data)
    cur.close()

# ── Collecte principale ───────────────────────────────────────────────────────

def fetch_gutenberg_books(
    session: requests.Session,
    max_books: int = 100,
    decade_filter: int | None = None,
) -> list[dict]:
    books  = []
    url    = GUTENDEX_BASE_URL
    params = {"languages": "fr", "topic": "fiction", "mime_type": "text/plain"}
    page   = 1

    while url and len(books) < max_books:
        logger.info(f"GutendexAPI — page {page} ({len(books)}/{max_books} livres)")
        try:
            data = _get_json(session, url, params=params if page == 1 else None)
        except requests.RequestException as e:
            logger.error(f"Erreur API Gutendex page {page} : {e}")
            break

        for book in data.get("results", []):
            year = _extract_year(book)
            if year is None or not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            if decade_filter and (year // 10) * 10 != decade_filter:
                continue
            if not _extract_txt_url(book):
                continue
            book["_year_extracted"] = year
            books.append(book)
            if len(books) >= max_books:
                break

        url = data.get("next")
        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info(f"{len(books)} livres trouvés")
    return books


def collect(
    max_books: int = 100,
    decade_filter: int | None = None,
    dry_run: bool = False,
) -> int:
    if not verify_db():
        logger.error("Base de données non initialisée. Exécuter : python init_db.py")
        sys.exit(1)

    session = _make_session()
    conn    = get_connection()
    run_id  = None

    if not dry_run:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pipeline_runs (run_type, started_at, status)
            VALUES ('collect', %s, 'running')
            RETURNING run_id
        """, (datetime.now(timezone.utc).isoformat(),))
        run_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

    success_count = 0

    try:
        books = fetch_gutenberg_books(session, max_books=max_books, decade_filter=decade_filter)

        for book in books:
            gutenberg_id = book["id"]
            book_id      = f"gutenberg_{gutenberg_id}"
            year         = book["_year_extracted"]
            decade       = (year // 10) * 10

            authors     = book.get("authors", [])
            author_name = authors[0]["name"] if authors else "Inconnu"
            if "," in author_name:
                parts       = [p.strip() for p in author_name.split(",", 1)]
                author_name = f"{parts[1]} {parts[0]}"

            title   = book.get("title", "Sans titre")
            genre   = _infer_genre(book)
            txt_url = _extract_txt_url(book)

            logger.info(f"{'[DRY-RUN] ' if dry_run else ''}→ {title} ({author_name}, {year})")

            if dry_run:
                continue

            if _book_exists(conn, book_id):
                logger.debug(f"Déjà en base : {book_id} — ignoré")
                continue

            try:
                raw_text = _get_text(session, txt_url)
            except requests.RequestException as e:
                logger.warning(f"Impossible de télécharger {book_id} : {e}")
                continue

            clean_text = strip_gutenberg_boilerplate(raw_text)
            if len(clean_text) < 5000:
                logger.warning(f"{book_id} trop court ({len(clean_text)} chars) — ignoré")
                continue

            local_path = _save_locally(clean_text, book_id)

            try:
                _insert_book(conn, {
                    "book_id":      book_id,
                    "title":        title,
                    "author":       author_name,
                    "year":         year,
                    "decade":       decade,
                    "genre":        genre,
                    "source":       "gutenberg",
                    "langue":       "fr",
                    "raw_path":     str(local_path),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur insertion {book_id} : {e}")
                continue

            success_count += 1
            logger.success(f"✓ {book_id} sauvegardé ({len(clean_text):,} chars)")
            time.sleep(REQUEST_DELAY)

        if not dry_run and run_id:
            cur = conn.cursor()
            cur.execute("""
                UPDATE pipeline_runs
                SET finished_at = %s, status = 'success', books_processed = %s
                WHERE run_id = %s
            """, (datetime.now(timezone.utc).isoformat(), success_count, run_id))
            conn.commit()
            cur.close()

    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        if not dry_run and run_id:
            cur = conn.cursor()
            cur.execute("""
                UPDATE pipeline_runs
                SET finished_at = %s, status = 'error', error_message = %s
                WHERE run_id = %s
            """, (datetime.now(timezone.utc).isoformat(), str(e), run_id))
            conn.commit()
            cur.close()
        raise
    finally:
        conn.close()

    logger.info(f"Collecte terminée : {success_count} livres collectés")
    return success_count

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collecte les textes français depuis Project Gutenberg.")
    parser.add_argument("--max-books", type=int, default=100)
    parser.add_argument("--decade",    type=int, default=None)
    parser.add_argument("--dry-run",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = collect(max_books=args.max_books, decade_filter=args.decade, dry_run=args.dry_run)
    if not args.dry_run:
        logger.info(f"Résultat final : {n} livres ajoutés à la base")

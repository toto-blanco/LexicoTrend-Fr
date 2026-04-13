"""
scraping/gallica.py — Collecte des textes BnF Gallica
======================================================
Récupère les textes français du domaine public depuis l'API Gallica (BnF)
via le protocole SRU (Search/Retrieve via URL) et l'API IIIF pour les textes.

Stratégie :
    1. Interroger l'API SRU Gallica avec filtres langue=fre, type=monographie
    2. Filtrer sur les œuvres avec date de publication connue (1850–1980)
    3. Récupérer le texte plein via l'API Gallica OCR (mode texte)
    4. Sauvegarder localement ET enregistrer en PostgreSQL

API Gallica :
    SRU  : https://gallica.bnf.fr/SRU?operation=searchRetrieve&...
    OCR  : https://gallica.bnf.fr/ark:/12148/{ark}/f1.texteBrut
    Meta : https://gallica.bnf.fr/ark:/12148/{ark}.dc

Usage :
    python scraping/gallica.py
    python scraping/gallica.py --max-books 10 --decade 1880
    python scraping/gallica.py --dry-run
    python scraping/gallica.py --ark ark:/12148/bpt6k9735557n
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import psycopg2
import requests
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GALLICA_SRU_URL  = "https://gallica.bnf.fr/SRU"
GALLICA_ARK_BASE = "https://gallica.bnf.fr/ark:/12148"
GALLICA_DC_NS    = "http://purl.org/dc/elements/1.1/"
GALLICA_SRU_NS   = "http://www.loc.gov/zing/srw/"

RAW_DATA_DIR  = Path("data/raw/gallica")
USER_AGENT    = os.getenv("GALLICA_USER_AGENT", "LexicoTrend-Research/1.0")
REQUEST_DELAY = float(os.getenv("GALLICA_REQUEST_DELAY", "1.5"))

YEAR_MIN     = 1850
YEAR_MAX     = 1980
SRU_PAGE_SIZE = 50
MIN_TEXT_LENGTH = 10_000

# ── Helpers réseau ────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/xml, text/plain, text/html",
    })
    return session


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _get_xml(session: requests.Session, url: str, params: dict = None) -> ET.Element:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return ET.fromstring(response.content)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.content.decode("latin-1", errors="replace")

# ── Requête SRU ───────────────────────────────────────────────────────────────

def _build_sru_query(year_min: int, year_max: int, decade: int | None = None) -> str:
    if decade:
        y_min, y_max = decade, decade + 9
    else:
        y_min, y_max = year_min, year_max

    return (
        f'dc.language all "fre" '
        f'and dc.type all "monographie" '
        f'and gallica.type all "texte" '
        f'and dc.date >= "{y_min}" '
        f'and dc.date <= "{y_max}"'
    )


def search_sru(
    session: requests.Session,
    query: str,
    start_record: int = 1,
    max_records: int = SRU_PAGE_SIZE,
) -> tuple[list[dict], int]:
    params = {
        "operation":      "searchRetrieve",
        "version":        "1.2",
        "query":          query,
        "startRecord":    start_record,
        "maximumRecords": max_records,
        "collapsing":     "false",
    }

    try:
        root = _get_xml(session, GALLICA_SRU_URL, params=params)
    except (requests.RequestException, ET.ParseError) as e:
        logger.error(f"Erreur SRU (start={start_record}) : {e}")
        return [], 0

    ns = {"srw": GALLICA_SRU_NS}
    total_elem = root.find(".//srw:numberOfRecords", ns)
    total = int(total_elem.text) if total_elem is not None else 0

    records = []
    for record in root.findall(".//srw:record", ns):
        data = _parse_dc_record(record)
        if data:
            records.append(data)

    return records, total

# ── Parsing Dublin Core ───────────────────────────────────────────────────────

def _parse_dc_record(record_elem: ET.Element) -> dict | None:
    ns_dc  = {"dc":  GALLICA_DC_NS}
    ns_srw = {"srw": GALLICA_SRU_NS}

    record_data = record_elem.find(".//srw:recordData", ns_srw)
    if record_data is None:
        return None

    def get_dc(tag: str) -> str | None:
        elem = record_data.find(f"dc:{tag}", ns_dc)
        return elem.text.strip() if elem is not None and elem.text else None

    def get_dc_all(tag: str) -> list[str]:
        return [e.text.strip() for e in record_data.findall(f"dc:{tag}", ns_dc) if e.text]

    identifier = get_dc("identifier")
    if not identifier:
        return None
    ark = _extract_ark(identifier)
    if not ark:
        return None

    title = get_dc("title")
    if not title:
        return None
    creator  = get_dc("creator") or "Inconnu"
    date_str = get_dc("date")
    year     = _parse_year(date_str)
    if year is None:
        return None

    return {
        "ark":      ark,
        "title":    title,
        "author":   _normalize_author(creator),
        "year":     year,
        "decade":   (year // 10) * 10,
        "subjects": get_dc_all("subject"),
        "types":    get_dc_all("type"),
    }


def _extract_ark(identifier: str) -> str | None:
    match = re.search(r"ark:/12148/([\w]+)", identifier)
    if match:
        return match.group(1)
    match = re.search(r"(bpt6k[\w]+|btv1b[\w]+|cb[\w]+)", identifier)
    if match:
        return match.group(1)
    return None


def _parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    match = re.search(r"\b(1[89]\d{2})\b", date_str)
    if match:
        year = int(match.group(1))
        if YEAR_MIN <= year <= YEAR_MAX:
            return year
    return None


def _normalize_author(raw: str) -> str:
    raw = re.sub(r"\s*\(\d{4}.*?\)", "", raw).strip()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            return f"{parts[1]} {parts[0]}"
    return raw


def _infer_genre(record: dict) -> str:
    text = " ".join(record.get("subjects", []) + record.get("types", [])).lower()
    if any(k in text for k in ["naturalisme", "naturalism", "zola", "maupassant"]):
        return "naturaliste"
    if any(k in text for k in ["roman historique", "histoire de france", "historical"]):
        return "historique"
    if any(k in text for k in ["roman populaire", "feuilleton", "aventure", "policier"]):
        return "populaire"
    return "roman_realiste"

# ── Téléchargement texte ──────────────────────────────────────────────────────

def fetch_full_text(session: requests.Session, ark: str) -> str | None:
    all_pages = []
    page = 1
    consecutive_errors = 0

    while consecutive_errors < 3:
        url = f"{GALLICA_ARK_BASE}/{ark}/f{page}.texteBrut"
        try:
            text = _get_text(session, url)
            if not text or len(text.strip()) < 10:
                break
            if "<html" in text.lower() and "erreur" in text.lower():
                break
            all_pages.append(text.strip())
            consecutive_errors = 0
            page += 1
            time.sleep(REQUEST_DELAY / 2)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                break
            consecutive_errors += 1
            time.sleep(REQUEST_DELAY)
        except requests.RequestException:
            consecutive_errors += 1
            time.sleep(REQUEST_DELAY)

    if not all_pages:
        return None
    return "\n\n".join(all_pages)

# ── Nettoyage texte Gallica ───────────────────────────────────────────────────

_OCR_ARTIFACTS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", re.UNICODE
)
_METADATA_LINES = re.compile(
    r"^(gallica\.bnf\.fr|bibliothèque nationale|bnf|source gallica|"
    r"reproduction numérique|numérisé par|imprimé par|paris,?\s+\d{4})",
    re.IGNORECASE | re.MULTILINE,
)


def clean_gallica_text(raw_text: str) -> str:
    text = _OCR_ARTIFACTS.sub("", raw_text)
    text = _METADATA_LINES.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [ln for ln in text.splitlines() if len(ln.strip()) >= 3]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_ocr_quality(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    suspicious_count = 0
    for token in tokens:
        non_alpha = re.sub(r"[a-zA-ZÀ-ÿ'\-]", "", token)
        if len(non_alpha) > len(token) * 0.5 and len(token) > 2:
            suspicious_count += 1
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", token.lower()):
            suspicious_count += 1
    quality = 1.0 - (suspicious_count / len(tokens))
    return round(max(0.0, min(1.0, quality)), 4)

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


def _insert_book(conn: psycopg2.extensions.connection, data: dict) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO books (
            book_id, title, author, year, decade, genre,
            source, langue, raw_path, ocr_quality_before, collected_at
        ) VALUES (
            %(book_id)s, %(title)s, %(author)s, %(year)s, %(decade)s, %(genre)s,
            %(source)s, %(langue)s, %(raw_path)s, %(ocr_quality_before)s, %(collected_at)s
        )
        ON CONFLICT (book_id) DO UPDATE SET
            raw_path           = EXCLUDED.raw_path,
            ocr_quality_before = EXCLUDED.ocr_quality_before,
            collected_at       = EXCLUDED.collected_at
    """, data)
    cur.close()

# ── Pipeline principale ───────────────────────────────────────────────────────

def collect_one(
    session: requests.Session,
    conn: psycopg2.extensions.connection,
    record: dict,
    dry_run: bool = False,
) -> bool:
    ark     = record["ark"]
    book_id = f"gallica_{ark}"

    logger.info(
        f"{'[DRY-RUN] ' if dry_run else ''}→ {record['title']} "
        f"({record['author']}, {record['year']}) — ark:{ark}"
    )

    if dry_run:
        return True

    if _book_exists(conn, book_id):
        logger.debug(f"Déjà en base : {book_id} — ignoré")
        return False

    raw_text = fetch_full_text(session, ark)
    if not raw_text:
        logger.warning(f"Texte inaccessible pour {book_id} — ignoré")
        return False

    clean_text = clean_gallica_text(raw_text)
    if len(clean_text) < MIN_TEXT_LENGTH:
        logger.warning(f"{book_id} trop court après nettoyage ({len(clean_text):,} chars) — ignoré")
        return False

    ocr_score  = compute_ocr_quality(clean_text)
    local_path = _save_locally(clean_text, book_id)

    try:
        _insert_book(conn, {
            "book_id":            book_id,
            "title":              record["title"],
            "author":             record["author"],
            "year":               record["year"],
            "decade":             record["decade"],
            "genre":              _infer_genre(record),
            "source":             "gallica",
            "langue":             "fr",
            "raw_path":           str(local_path),
            "ocr_quality_before": ocr_score,
            "collected_at":       datetime.now(timezone.utc).isoformat(),
        })
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur insertion {book_id} : {e}")
        return False

    logger.success(f"✓ {book_id} — {len(clean_text):,} chars — OCR score: {ocr_score:.3f}")
    return True


def collect(
    max_books: int = 100,
    decade_filter: int | None = None,
    dry_run: bool = False,
    ark_direct: str | None = None,
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
        # ── Mode ARK direct ──────────────────────────────────────────────────
        if ark_direct:
            ark = _extract_ark(ark_direct) or ark_direct
            dc_url = f"{GALLICA_ARK_BASE}/{ark}.dc"
            try:
                root   = _get_xml(session, dc_url)
                ns     = {"dc": GALLICA_DC_NS}
                title  = root.findtext("dc:title",   namespaces=ns) or "Sans titre"
                author = root.findtext("dc:creator", namespaces=ns) or "Inconnu"
                date   = root.findtext("dc:date",    namespaces=ns)
                year   = _parse_year(date)
                if year is None:
                    logger.error(f"Impossible d'extraire l'année depuis {dc_url}")
                    return 0
                record = {
                    "ark":      ark,
                    "title":    title,
                    "author":   _normalize_author(author),
                    "year":     year,
                    "decade":   (year // 10) * 10,
                    "subjects": [],
                    "types":    [],
                }
                if collect_one(session, conn, record, dry_run):
                    success_count += 1
            except (requests.RequestException, ET.ParseError) as e:
                logger.error(f"Erreur récupération métadonnées {ark} : {e}")
            return success_count

        # ── Mode recherche SRU ───────────────────────────────────────────────
        query        = _build_sru_query(YEAR_MIN, YEAR_MAX, decade=decade_filter)
        start_record = 1
        total_found  = None

        while success_count < max_books:
            records, total = search_sru(session, query, start_record=start_record)

            if total_found is None:
                total_found = total
                logger.info(f"Total résultats SRU : {total_found}")

            if not records:
                break

            for record in records:
                if success_count >= max_books:
                    break
                if collect_one(session, conn, record, dry_run):
                    success_count += 1
                time.sleep(REQUEST_DELAY)

            start_record += SRU_PAGE_SIZE
            if start_record > (total_found or 0):
                break
            time.sleep(REQUEST_DELAY)

        # Finaliser le run
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

    logger.info(f"Collecte Gallica terminée : {success_count} livres collectés")
    return success_count

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collecte les textes français depuis Gallica (BnF).")
    parser.add_argument("--max-books", type=int, default=100)
    parser.add_argument("--decade",    type=int, default=None)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--ark",       default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = collect(
        max_books=args.max_books,
        decade_filter=args.decade,
        dry_run=args.dry_run,
        ark_direct=args.ark,
    )
    if not args.dry_run:
        logger.info(f"Résultat final : {n} livres ajoutés à la base")

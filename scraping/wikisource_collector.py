"""
scraping/wikisource_collector.py — Collecte depuis Wikisource FR
================================================================
Télécharge les textes depuis Wikisource via l'API MediaWiki,
insère les métadonnées en base PostgreSQL et sauvegarde les fichiers.

Usage :
    python scraping/wikisource_collector.py
    python scraping/wikisource_collector.py --dry-run
"""

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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DATA_DIR  = Path("data/raw/wikisource")
REQUEST_DELAY = 10.0
WIKISOURCE_API = "https://fr.wikisource.org/w/api.php"

# ── Corpus cible ──────────────────────────────────────────────────────────────
# Uniquement les œuvres confirmées disponibles sur Wikisource FR

CORPUS = [
    # 1860s
    {
        "book_id":  "wikisource_therese_raquin",
        "title":    "Thérèse Raquin",
        "author":   "Émile Zola",
        "year":     1867,
        "genre":    "naturaliste",
        "page":     "Thérèse_Raquin",
    },
    # 1880s
    {
        "book_id":  "wikisource_bel_ami",
        "title":    "Bel-Ami",
        "author":   "Guy de Maupassant",
        "year":     1885,
        "genre":    "roman_realiste",
        "page":     "Bel-Ami",
    },
    {
        "book_id":  "wikisource_germinal",
        "title":    "Germinal",
        "author":   "Émile Zola",
        "year":     1885,
        "genre":    "naturaliste",
        "page":     "Germinal",
    },
    {
        "book_id":  "wikisource_une_vie",
        "title":    "Une vie",
        "author":   "Guy de Maupassant",
        "year":     1883,
        "genre":    "roman_realiste",
        "page":     "Une_vie",
    },
    # 1890s
    {
        "book_id":  "wikisource_bete_humaine",
        "title":    "La Bête humaine",
        "author":   "Émile Zola",
        "year":     1890,
        "genre":    "naturaliste",
        "page":     "La_Bête_humaine",
    },
    {
        "book_id":  "wikisource_fort_comme_la_mort",
        "title":    "Fort comme la mort",
        "author":   "Guy de Maupassant",
        "year":     1889,
        "genre":    "roman_realiste",
        "page":     "Fort_comme_la_mort",
    },
    {
        "book_id":  "wikisource_pierre_et_jean",
        "title":    "Pierre et Jean",
        "author":   "Guy de Maupassant",
        "year":     1887,
        "genre":    "roman_realiste",
        "page":     "Pierre_et_Jean",
    },
    # 1910s
    {
        "book_id":  "wikisource_grand_meaulnes",
        "title":    "Le Grand Meaulnes",
        "author":   "Alain-Fournier",
        "year":     1913,
        "genre":    "roman_realiste",
        "page":     "Le_Grand_Meaulnes",
    },
    {
        "book_id":  "wikisource_symphonie_pastorale",
        "title":    "La Symphonie pastorale",
        "author":   "André Gide",
        "year":     1919,
        "genre":    "roman_realiste",
        "page":     "La_Symphonie_pastorale",
    },
    {
        "book_id":  "wikisource_porte_etroite",
        "title":    "La Porte étroite",
        "author":   "André Gide",
        "year":     1909,
        "genre":    "roman_realiste",
        "page":     "La_Porte_étroite",
    },
    # 1920s
    {
        "book_id":  "wikisource_la_garconne",
        "title":    "La Garçonne",
        "author":   "Victor Margueritte",
        "year":     1922,
        "genre":    "roman_realiste",
        "page":     "La_Garçonne",
    },
    {
        "book_id":  "wikisource_ble_en_herbe",
        "title":    "Le Blé en herbe",
        "author":   "Colette",
        "year":     1923,
        "genre":    "roman_realiste",
        "page":     "Le_Blé_en_herbe",
    },
]

# ── Téléchargement via API MediaWiki ─────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "LexicoTrend-Research/1.0 (data analysis project)",
        "Accept": "application/json",
    })
    return session


WS_EXPORT_URL = "https://ws-export.wmcloud.org/"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=30, max=120),
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    reraise=True,
)
def fetch_wikisource_text(session: requests.Session, page_title: str) -> str | None:
    """
    Récupère le texte complet d'une œuvre Wikisource via ws-export.
    ws-export compile toutes les sous-pages DjVu en un seul fichier texte.
    """
    try:
        time.sleep(5)  # délai pré-requête pour respecter le rate limit ws-export
        response = session.get(
            WS_EXPORT_URL,
            params={
                "format": "txt",
                "lang":   "fr",
                "page":   page_title,
            },
            timeout=120,
        )

        if response.status_code == 429:
            logger.warning(f"Rate limit ws-export pour {page_title} — attente 60s")
            time.sleep(60)
            response = session.get(
                WS_EXPORT_URL,
                params={"format": "txt", "lang": "fr", "page": page_title},
                timeout=120,
            )

        if response.status_code == 429:
            logger.error(f"Rate limit persistant pour {page_title} — livre ignoré")
            return None

        response.raise_for_status()

        text = response.text
        if not text or len(text.strip()) < 1000:
            logger.warning(f"Export trop court pour : {page_title} ({len(text)} chars)")
            return None

        text = _strip_wsexport_header(text)
        return text.strip()

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            logger.error(f"Rate limit définitif pour {page_title} — ignoré")
            return None
        logger.error(f"Erreur HTTP pour {page_title} : {e}")
        return None
    except requests.RequestException as e:
        logger.error(f"Erreur réseau pour {page_title} : {e}")
        raise


def _strip_wsexport_header(text: str) -> str:
    """
    Supprime l'en-tête généré par ws-export :
    titre, auteur, éditeur, date d'export, table des matières.
    """
    # Chercher la fin de la table des matières (marquée par "* * *")
    marker = "* * *"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):]
    # Fallback : supprimer les 20 premières lignes (en-tête typique)
    lines = text.splitlines()
    return "\n".join(lines[20:])


def _clean_wikitext(wikitext: str) -> str:
    """
    Nettoie le markup MediaWiki pour obtenir du texte brut.
    Supprime templates, liens, balises, etc.
    """
    text = wikitext

    # Supprimer les templates {{...}}
    while "{{" in text:
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Supprimer les balises HTML et MediaWiki
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    # Liens internes [[texte|affichage]] → affichage
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # Liens externes [url texte] → texte
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Titres MediaWiki == Titre == → Titre
    text = re.sub(r"={2,6}\s*(.+?)\s*={2,6}", r"\1", text)

    # Gras et italique
    text = re.sub(r"'{2,3}", "", text)

    # Commentaires HTML
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Catégories et fichiers
    text = re.sub(r"\[\[(?:Catégorie|Category|Fichier|File|Image):[^\]]+\]\]", "", text)

    # Nettoyer les espaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

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


def _insert_book(conn: psycopg2.extensions.connection, data: dict) -> None:
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
    """, data)
    cur.close()

# ── Pipeline principale ───────────────────────────────────────────────────────

def collect(dry_run: bool = False) -> int:
    if not verify_db():
        logger.error("Base non initialisée.")
        sys.exit(1)

    session = _make_session()
    conn    = get_connection() if not dry_run else None
    success = 0

    for entry in CORPUS:
        book_id = entry["book_id"]
        title   = entry["title"]
        author  = entry["author"]
        year    = entry["year"]
        decade  = (year // 10) * 10

        logger.info(f"{'[DRY-RUN] ' if dry_run else ''}→ {title} ({author}, {year})")

        if not dry_run and conn and _book_exists(conn, book_id):
            logger.debug(f"Déjà en base : {book_id} — ignoré")
            continue

        text = fetch_wikisource_text(session, entry["page"])
        if not text or len(text) < 5000:
            logger.warning(f"{book_id} : texte trop court ou vide ({len(text) if text else 0} chars) — ignoré")
            time.sleep(REQUEST_DELAY)
            continue

        logger.info(f"  {len(text):,} chars récupérés")

        if dry_run:
            time.sleep(REQUEST_DELAY)
            continue

        local_path = _save_locally(text, book_id)

        try:
            _insert_book(conn, {
                "book_id":      book_id,
                "title":        title,
                "author":       author,
                "year":         year,
                "decade":       decade,
                "genre":        entry["genre"],
                "source":       "wikisource",
                "langue":       "fr",
                "raw_path":     str(local_path),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })
            conn.commit()
            success += 1
            logger.success(f"✓ {book_id} sauvegardé ({len(text):,} chars)")
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur insertion {book_id} : {e}")

        time.sleep(REQUEST_DELAY)

    if conn:
        conn.close()

    logger.info(f"Collecte Wikisource terminée : {success}/{len(CORPUS)} livres collectés")
    return success

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collecte depuis Wikisource FR.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    collect(dry_run=args.dry_run)

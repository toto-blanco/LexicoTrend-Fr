"""
init_db.py — Initialisation du schéma PostgreSQL LexicoTrend FR
===============================================================
À exécuter une seule fois avant tout autre script.
Crée les tables books, anomalies et pipeline_runs si elles n'existent pas.

Usage :
    python init_db.py
    python init_db.py --reset   # ⚠️  supprime et recrée toutes les tables
"""

import argparse
import os
import sys

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Config connexion ──────────────────────────────────────────────────────────

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "lexicotrend")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── Schéma ────────────────────────────────────────────────────────────────────

SCHEMA_BOOKS = """
CREATE TABLE IF NOT EXISTS books (
    -- Identification
    book_id             TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    author              TEXT NOT NULL,
    year                INTEGER NOT NULL,
    decade              INTEGER NOT NULL,
    genre               TEXT,
    source              TEXT NOT NULL,
    langue              TEXT NOT NULL DEFAULT 'fr',

    -- Chemins fichiers
    raw_path            TEXT,
    processed_path      TEXT,

    -- Qualité OCR
    ocr_quality_before  FLOAT,
    ocr_quality_after   FLOAT,
    ocr_corrected       INTEGER NOT NULL DEFAULT 0,

    -- Métriques lexicales brutes
    word_count          INTEGER,
    unique_words        INTEGER,
    sentence_count      INTEGER,

    -- Métriques de richesse lexicale
    ttr                 FLOAT,
    mtld                FLOAT,
    hdd                 FLOAT,
    mtld_ma_bid         FLOAT,

    -- Enrichissement Claude API
    claude_interpretation TEXT,
    is_outlier          INTEGER NOT NULL DEFAULT 0,

    -- Métadonnées de traitement
    collected_at        TEXT,
    processed_at        TEXT,
    metrics_at          TEXT,
    enriched_at         TEXT,

    -- Contraintes
    CHECK (year >= 1800 AND year <= 2030),
    CHECK (decade = (year / 10) * 10),
    CHECK (ocr_quality_before IS NULL OR (ocr_quality_before >= 0 AND ocr_quality_before <= 1)),
    CHECK (ocr_quality_after  IS NULL OR (ocr_quality_after  >= 0 AND ocr_quality_after  <= 1)),
    CHECK (source IN ('gallica', 'gutenberg', 'wikisource')),
    CHECK (ocr_corrected IN (0, 1)),
    CHECK (is_outlier IN (0, 1))
);
"""

SCHEMA_ANOMALIES = """
CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id          SERIAL PRIMARY KEY,
    book_id             TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,

    -- Contexte statistique au moment de la détection
    mtld_value          FLOAT NOT NULL,
    decade_median_mtld  FLOAT NOT NULL,
    decade_mean_mtld    FLOAT NOT NULL,
    decade_std_mtld     FLOAT NOT NULL,
    deviation_pct       FLOAT NOT NULL,
    direction           TEXT NOT NULL,

    -- Interprétation Claude API
    prompt_used         TEXT,
    interpretation      TEXT,
    model_used          TEXT,
    tokens_used         INTEGER,

    -- Métadonnées
    detected_at         TEXT NOT NULL,
    CHECK (direction IN ('above', 'below'))
);
"""

SCHEMA_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              SERIAL PRIMARY KEY,
    run_type            TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'running',
    books_processed     INTEGER DEFAULT 0,
    error_message       TEXT,
    CHECK (run_type IN ('collect', 'process', 'enrich', 'analyze')),
    CHECK (status IN ('running', 'success', 'error'))
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_books_decade   ON books (decade);",
    "CREATE INDEX IF NOT EXISTS idx_books_genre    ON books (genre);",
    "CREATE INDEX IF NOT EXISTS idx_books_source   ON books (source);",
    "CREATE INDEX IF NOT EXISTS idx_books_outlier  ON books (is_outlier);",
    "CREATE INDEX IF NOT EXISTS idx_books_mtld     ON books (mtld);",
    "CREATE INDEX IF NOT EXISTS idx_anomalies_book ON anomalies (book_id);",
]

# ── Connexion ─────────────────────────────────────────────────────────────────

def get_connection() -> psycopg2.extensions.connection:
    """
    Retourne une connexion psycopg2 vers PostgreSQL.
    autocommit=False — chaque script gère ses transactions explicitement
    via conn.commit() / conn.rollback().
    Les paramètres sont lus depuis les variables d'environnement (.env).
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    conn.autocommit = False
    return conn


def verify_db() -> bool:
    """
    Vérifie que les tables requises existent dans PostgreSQL.
    Utilise information_schema (standard SQL) — compatible tous SGBD.

    Returns:
        True si toutes les tables requises existent, False sinon.
    """
    required_tables = {"books", "anomalies", "pipeline_runs"}
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type   = 'BASE TABLE';
        """)
        existing = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()

        missing = required_tables - existing
        if missing:
            logger.error(f"Tables manquantes : {missing}")
            logger.error("Exécuter : python init_db.py")
            return False
        return True

    except psycopg2.OperationalError as e:
        logger.error(f"Impossible de se connecter à PostgreSQL : {e}")
        return False


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(reset: bool = False) -> None:
    """
    Initialise le schéma PostgreSQL.

    Args:
        reset: Si True, supprime et recrée toutes les tables. ⚠️ Destructif.
    """
    conn = get_connection()
    cur  = conn.cursor()

    try:
        if reset:
            logger.warning("--reset activé : suppression des tables existantes")
            cur.execute("DROP TABLE IF EXISTS anomalies CASCADE;")
            cur.execute("DROP TABLE IF EXISTS pipeline_runs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS books CASCADE;")
            logger.warning("Tables supprimées")

        logger.info("Création des tables...")
        cur.execute(SCHEMA_BOOKS)
        cur.execute(SCHEMA_ANOMALIES)
        cur.execute(SCHEMA_PIPELINE_RUNS)

        logger.info("Création des index...")
        for idx_sql in INDEXES:
            cur.execute(idx_sql)

        conn.commit()
        logger.success("Base de données initialisée avec succès ✓")
        _print_summary(cur)

    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur initialisation : {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def _print_summary(cur: psycopg2.extensions.cursor) -> None:
    """Affiche un résumé des tables créées."""
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    cur.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY indexname;
    """)
    indexes = [r[0] for r in cur.fetchall()]

    logger.info(f"Tables  : {', '.join(tables)}")
    logger.info(f"Index   : {', '.join(indexes)}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise le schéma PostgreSQL LexicoTrend FR."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  Supprime et recrée toutes les tables (perte de données)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        confirm = input(
            "⚠️  --reset va supprimer toutes les données. Confirmer ? [oui/N] : "
        )
        if confirm.strip().lower() != "oui":
            logger.info("Annulé.")
            sys.exit(0)

    init_db(reset=args.reset)

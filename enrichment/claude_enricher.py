"""
enrichment/claude_enricher.py — Enrichissement via Claude API
=============================================================
Deux responsabilités :
1. POST-CORRECTION OCR COUCHE 3 (Haiku) — passages ambigus après ByT5
2. INTERPRÉTATION DES ANOMALIES STATISTIQUES (Sonnet) — outliers MTLD

Usage :
    python enrichment/claude_enricher.py --task ocr
    python enrichment/claude_enricher.py --task anomalies
    python enrichment/claude_enricher.py --task all
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import psycopg2
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path("data/processed")
MODEL_OCR      = os.getenv("ANTHROPIC_MODEL_OCR",     "claude-haiku-4-5-20251001")
MODEL_ANALYSIS = os.getenv("ANTHROPIC_MODEL_ANALYSIS", "claude-sonnet-4-6")
OCR_THRESHOLD  = float(os.getenv("OCR_QUALITY_THRESHOLD", "0.85"))

OCR_CHUNK_CHARS   = 1500
API_DELAY_SECONDS = 1.5

# ── Client Anthropic ──────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY absent du fichier .env")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    reraise=True,
)
def _call_api(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 1024,
) -> anthropic.types.Message:
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — POST-CORRECTION OCR COUCHE 3
# ══════════════════════════════════════════════════════════════════════════════

OCR_SYSTEM_PROMPT = """Tu es un expert en correction de textes OCR de livres français du XIXe siècle.
Tu reçois un extrait de texte issu d'un scan OCR qui peut contenir des erreurs de reconnaissance.

Tes règles absolues :
1. Corrige UNIQUEMENT les erreurs OCR évidentes (caractères mal reconnus, mots fragmentés, espaces parasites dans les mots)
2. CONSERVE intégralement le vocabulaire archaïque légitime : céans, icelle, nonobstant, iceluy, ouïr, point (négation), maint, ains, etc.
3. CONSERVE la syntaxe et le style de l'époque — ne modernise pas
4. CONSERVE la ponctuation originale sauf si elle est clairement un artefact OCR
5. Si un passage est ambigu (impossible de distinguer archaïsme et erreur), CONSERVE l'original
6. Réponds UNIQUEMENT avec le texte corrigé, sans explication ni commentaire"""


def _correct_passage_with_claude(client: anthropic.Anthropic, passage: str) -> tuple[str, int]:
    user_content = f"Corrige les erreurs OCR dans ce texte :\n\n{passage}"
    try:
        response  = _call_api(
            client=client, model=MODEL_OCR, system=OCR_SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=int(len(passage.split()) * 1.5) + 100,
        )
        corrected = response.content[0].text.strip()
        tokens    = response.usage.input_tokens + response.usage.output_tokens
        corrected = _sanity_check_ocr(passage, corrected)
        return corrected, tokens
    except anthropic.AuthenticationError as e:
        logger.error(f"Clé API invalide : {e}")
        sys.exit(1)
    except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
        logger.warning(f"Erreur API transitoire : {e}")
        return passage, 0
    except Exception as e:
        logger.warning(f"Erreur inattendue API OCR : {e}")
        return passage, 0


def _sanity_check_ocr(original: str, corrected: str) -> str:
    if not corrected or not corrected.strip():
        return original
    ratio = len(corrected) / max(len(original), 1)
    if ratio < 0.6 or ratio > 2.0:
        return original
    english_markers = {"the", "and", "that", "this", "with", "from", "have", "been"}
    french_markers  = {"le", "la", "les", "de", "du", "et", "en", "un", "une", "que"}
    words = set(corrected.lower().split()[:30])
    if len(words & english_markers) > len(words & french_markers) and len(words & english_markers) >= 3:
        return original
    return corrected


def correct_book_ocr_layer3(
    book_id: str,
    text_path: Path,
    conn: psycopg2.extensions.connection,
    client: anthropic.Anthropic,
) -> dict:
    from processing.clean import compute_ocr_quality

    logger.info(f"OCR couche 3 (Claude Haiku) : {book_id}")

    text     = text_path.read_text(encoding="utf-8", errors="replace")
    passages = _split_into_ocr_passages(text)

    corrected_passages = []
    total_tokens       = 0
    passages_corrected = 0

    for passage in passages:
        score = compute_ocr_quality(passage)
        if score >= OCR_THRESHOLD:
            corrected_passages.append(passage)
        else:
            corrected, tokens = _correct_passage_with_claude(client, passage)
            corrected_passages.append(corrected)
            total_tokens += tokens
            passages_corrected += 1
            time.sleep(API_DELAY_SECONDS)

    corrected_text = "\n\n".join(corrected_passages)
    final_score    = compute_ocr_quality(corrected_text)

    source   = text_path.parent.name
    out_dir  = PROCESSED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{book_id}_corrected_l3.txt"
    out_path.write_text(corrected_text, encoding="utf-8")

    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE books SET
                processed_path    = %s,
                ocr_quality_after = %s,
                enriched_at       = %s
            WHERE book_id = %s
        """, (str(out_path), final_score, datetime.now(timezone.utc).isoformat(), book_id))
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur mise à jour DB pour {book_id} : {e}")

    logger.success(
        f"✓ {book_id} — {passages_corrected}/{len(passages)} passages corrigés — "
        f"score final: {final_score:.3f} — tokens: {total_tokens:,}"
    )
    return {
        "book_id":            book_id,
        "status":             "ok",
        "passages_total":     len(passages),
        "passages_corrected": passages_corrected,
        "final_ocr_score":    final_score,
        "tokens_used":        total_tokens,
    }


def _split_into_ocr_passages(text: str, size: int = OCR_CHUNK_CHARS) -> list[str]:
    paragraphs  = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages    = []
    current     = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > size and current:
            passages.append("\n\n".join(current))
            current     = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        passages.append("\n\n".join(current))
    return passages

# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — INTERPRÉTATION DES ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_SYSTEM_PROMPT = """Tu es un expert en littérature française du XIXe et XXe siècle
et en stylistique computationnelle.

Tu reçois des informations sur un roman best-seller français dont la richesse lexicale
(mesurée par le score MTLD) est statistiquement inhabituelle par rapport aux autres romans
de la même décennie.

Ta tâche : formuler une hypothèse stylistique, littéraire ou historique qui expliquerait
cet écart. Sois précis et factuel. Cite si possible des éléments biographiques ou des
caractéristiques stylistiques connues de l'auteur.

Réponds en 2-3 phrases maximum, en français, sans introduire ta réponse par des formules
comme "Il est possible que" ou "On peut supposer que". Va directement à l'hypothèse."""


def _build_anomaly_prompt(book: dict, stats: dict) -> str:
    direction = "supérieure" if stats["direction"] == "above" else "inférieure"
    sign      = "+" if stats["deviation_pct"] > 0 else ""
    return (
        f"Roman : « {book['title']} » de {book['author']} "
        f"({book['year']}, genre : {book['genre'] or 'inconnu'})\n\n"
        f"Score MTLD : {stats['mtld_value']:.1f}\n"
        f"Médiane de la décennie {book['decade']}s : {stats['decade_median_mtld']:.1f}\n"
        f"Écart : {sign}{stats['deviation_pct']:.1f}% ({direction} à la médiane)\n\n"
        f"Génère une hypothèse stylistique ou historique expliquant cet écart."
    )


def interpret_anomaly(
    client: anthropic.Anthropic,
    conn: psycopg2.extensions.connection,
    book: dict,
    stats: dict,
) -> dict:
    prompt = _build_anomaly_prompt(book, stats)
    try:
        response       = _call_api(client=client, model=MODEL_ANALYSIS,
                                   system=ANOMALY_SYSTEM_PROMPT, user_content=prompt, max_tokens=300)
        interpretation = response.content[0].text.strip()
        tokens_used    = response.usage.input_tokens + response.usage.output_tokens
    except anthropic.AuthenticationError as e:
        logger.error(f"Clé API invalide : {e}")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Erreur API interprétation {book['book_id']} : {e}")
        return {"book_id": book["book_id"], "status": "api_error", "error": str(e)}

    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE books SET
                claude_interpretation = %s,
                is_outlier            = 1,
                enriched_at           = %s
            WHERE book_id = %s
        """, (interpretation, datetime.now(timezone.utc).isoformat(), book["book_id"]))

        cur.execute("""
            INSERT INTO anomalies (
                book_id, mtld_value, decade_median_mtld, decade_mean_mtld,
                decade_std_mtld, deviation_pct, direction,
                prompt_used, interpretation, model_used, tokens_used, detected_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            book["book_id"],
            stats["mtld_value"],
            stats["decade_median_mtld"],
            stats["decade_mean_mtld"],
            stats["decade_std_mtld"],
            stats["deviation_pct"],
            stats["direction"],
            prompt,
            interpretation,
            MODEL_ANALYSIS,
            tokens_used,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur DB pour {book['book_id']} : {e}")

    logger.success(f"✓ {book['book_id']} — interprétation générée ({tokens_used} tokens)")
    logger.debug(f"Interprétation : {interpretation[:120]}…")
    time.sleep(API_DELAY_SECONDS)

    return {"book_id": book["book_id"], "status": "ok",
            "tokens_used": tokens_used, "interpretation": interpretation}


def _get_outliers(conn: psycopg2.extensions.connection) -> list[dict]:
    """
    Récupère les outliers sans interprétation.

    Note migration : la médiane SQLite était calculée via une sous-requête
    OFFSET — remplacée ici par PERCENTILE_CONT (fonction native PostgreSQL).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            b.book_id,
            b.title,
            b.author,
            b.year,
            b.decade,
            b.genre,
            b.mtld,
            AVG(b2.mtld)                                      AS decade_mean,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.mtld) AS decade_median,
            STDDEV_POP(b2.mtld)                               AS decade_std
        FROM books b
        JOIN books b2 ON b2.decade = b.decade AND b2.mtld IS NOT NULL
        WHERE b.is_outlier = 1
          AND b.mtld IS NOT NULL
          AND b.claude_interpretation IS NULL
        GROUP BY b.book_id, b.title, b.author, b.year, b.decade, b.genre, b.mtld
    """)
    rows = cur.fetchall()
    cur.close()

    results = []
    for row in rows:
        (book_id, title, author, year, decade, genre,
         mtld, mean, median, std) = row

        if std is None or std == 0:
            continue

        deviation_pct = ((mtld - median) / median * 100) if median else 0
        direction     = "above" if mtld > median else "below"

        results.append({
            "book_id":            book_id,
            "title":              title,
            "author":             author,
            "year":               year,
            "decade":             decade,
            "genre":              genre,
            "mtld_value":         mtld,
            "decade_mean_mtld":   float(mean),
            "decade_median_mtld": float(median),
            "decade_std_mtld":    float(std),
            "deviation_pct":      deviation_pct,
            "direction":          direction,
        })
    return results

# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — NORMALISATION DU CORPUS (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

CORPUS_SYSTEM_PROMPT = """Tu es un expert en bibliographie française.
Tu reçois une liste brute de titres/auteurs de romans français.
Retourne UNIQUEMENT un tableau JSON valide (sans markdown, sans explication) avec cette structure :
[
  {
    "title": "Titre normalisé",
    "author": "Prénom Nom",
    "year": 1882,
    "decade": 1880,
    "genre": "roman_realiste|naturaliste|historique|populaire|autre",
    "source_hint": "gallica|gutenberg|wikisource|inconnu"
  }
]
Règles :
- Corriger les fautes de frappe dans les titres et noms
- Normaliser les noms "Nom, Prénom" → "Prénom Nom"
- Inférer le genre depuis tes connaissances de l'auteur et de l'époque
- L'année doit être celle de la PREMIÈRE PUBLICATION, pas d'une réédition
- Si une information est inconnue, mettre null"""


def normalize_corpus_list(raw_list: list[str]) -> list[dict]:
    import json
    client     = _get_client()
    results    = []
    batch_size = 20

    for i in range(0, len(raw_list), batch_size):
        batch    = raw_list[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {item}" for j, item in enumerate(batch))
        try:
            response = _call_api(
                client=client, model=MODEL_ANALYSIS, system=CORPUS_SYSTEM_PROMPT,
                user_content=f"Normalise cette liste :\n\n{numbered}", max_tokens=2000,
            )
            raw_json = response.content[0].text.strip()
            raw_json = re.sub(r"```(?:json)?\s*", "", raw_json).strip("`").strip()
            parsed   = json.loads(raw_json)
            results.extend(parsed)
            logger.success(f"Batch {i//batch_size + 1} normalisé ({len(parsed)} entrées)")
            time.sleep(API_DELAY_SECONDS)
        except Exception as e:
            logger.error(f"Erreur batch {i//batch_size + 1} : {e}")

    return results

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINES PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

def run_ocr_layer3(source_filter: str | None = None) -> list[dict]:
    if not verify_db():
        logger.error("Base non initialisée.")
        sys.exit(1)

    conn   = get_connection()
    client = _get_client()

    cur = conn.cursor()
    query  = """
        SELECT book_id, processed_path, ocr_quality_after, source
        FROM books
        WHERE processed_path IS NOT NULL
          AND ocr_quality_after < %s
          AND ocr_corrected = 1
    """
    params = [OCR_THRESHOLD]
    if source_filter:
        query += " AND source = %s"
        params.append(source_filter)
    cur.execute(query, params)
    rows  = cur.fetchall()
    cur.close()

    books = [{"book_id": r[0], "path": r[1], "score": r[2], "source": r[3]} for r in rows]

    if not books:
        logger.info(f"Aucun livre nécessitant la couche 3 OCR (seuil={OCR_THRESHOLD})")
        conn.close()
        return []

    logger.info(f"{len(books)} livres à corriger via Claude Haiku")
    results = []
    for book in books:
        path = Path(book["path"])
        if not path.exists():
            logger.warning(f"Fichier introuvable : {path}")
            results.append({"book_id": book["book_id"], "status": "file_not_found"})
            continue
        try:
            result = correct_book_ocr_layer3(book_id=book["book_id"],
                                              text_path=path, conn=conn, client=client)
            results.append(result)
        except Exception as e:
            logger.error(f"Erreur sur {book['book_id']} : {e}")
            results.append({"book_id": book["book_id"], "status": "error", "error": str(e)})

    conn.close()
    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info(f"OCR couche 3 terminée : {ok}/{len(books)} livres traités")
    return results


def run_anomalies(conn: psycopg2.extensions.connection | None = None) -> list[dict]:
    close_conn = conn is None
    if conn is None:
        if not verify_db():
            logger.error("Base non initialisée.")
            sys.exit(1)
        conn = get_connection()

    client   = _get_client()
    outliers = _get_outliers(conn)

    if not outliers:
        logger.info("Aucun outlier sans interprétation trouvé")
        if close_conn:
            conn.close()
        return []

    logger.info(f"{len(outliers)} outliers à interpréter")
    results = []
    for item in outliers:
        book  = {k: item[k] for k in ["book_id", "title", "author", "year", "decade", "genre"]}
        stats = {k: item[k] for k in [
            "mtld_value", "decade_mean_mtld", "decade_median_mtld",
            "decade_std_mtld", "deviation_pct", "direction"
        ]}
        result = interpret_anomaly(client, conn, book, stats)
        results.append(result)

    if close_conn:
        conn.close()

    ok           = sum(1 for r in results if r.get("status") == "ok")
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    logger.info(f"Interprétations terminées : {ok}/{len(outliers)} OK — {total_tokens:,} tokens")
    return results

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrichissement Claude API.")
    parser.add_argument("--task",   choices=["ocr", "anomalies", "all"], default="all")
    parser.add_argument("--source", default=None, choices=["gallica", "gutenberg", "wikisource"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task in ("ocr", "all"):
        logger.info("── Tâche : Post-correction OCR couche 3 (Claude Haiku) ──")
        run_ocr_layer3(source_filter=args.source)
    if args.task in ("anomalies", "all"):
        logger.info("── Tâche : Interprétation des anomalies (Claude Sonnet) ──")
        run_anomalies()

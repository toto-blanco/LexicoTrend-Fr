"""
ml/analysis.py — Analyse statistique et Machine Learning
=========================================================
Répond aux trois hypothèses de recherche.

Usage :
    python ml/analysis.py
    python ml/analysis.py --output data/processed/analysis_results.json
    python ml/analysis.py --min-books 5
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from loguru import logger
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("data/processed")
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTLIER_SIGMA_THRESHOLD = 2.0
MIN_BOOKS_PER_DECADE    = 3
KMEANS_RANGE            = range(2, 7)

# ── Chargement des données ────────────────────────────────────────────────────

def load_data(conn: psycopg2.extensions.connection, min_books: int = MIN_BOOKS_PER_DECADE) -> pd.DataFrame:
    """
    Charge les données depuis PostgreSQL.
    pd.read_sql_query est compatible psycopg2 sans modification.
    """
    query = """
        SELECT
            book_id, title, author, year, decade, genre, source,
            word_count, unique_words, sentence_count,
            ttr, mtld, hdd, mtld_ma_bid,
            ocr_quality_after
        FROM books
        WHERE mtld IS NOT NULL
        ORDER BY year
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        logger.error("Aucun livre avec MTLD calculé.")
        return df

    logger.info(f"{len(df)} livres avec MTLD disponibles")

    decade_counts  = df.groupby("decade")["book_id"].count()
    valid_decades  = decade_counts[decade_counts >= min_books].index
    excluded       = decade_counts[decade_counts < min_books]

    if not excluded.empty:
        logger.warning(
            f"Décennies exclues (< {min_books} livres) : "
            + ", ".join(f"{d}s ({excluded[d]})" for d in excluded.index)
        )

    df = df[df["decade"].isin(valid_decades)].copy()
    logger.info(f"{len(df)} livres retenus après filtrage")
    df["genre"] = df["genre"].fillna("inconnu")
    return df

# ── Détection des outliers ────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    decade_stats = df.groupby("decade")["mtld"].agg(
        decade_mean="mean", decade_median="median", decade_std="std",
    ).reset_index()

    df = df.merge(decade_stats, on="decade", how="left")
    df["decade_std"] = df["decade_std"].replace(0, np.nan)
    df["z_score"]       = (df["mtld"] - df["decade_mean"]) / df["decade_std"].fillna(1)
    df["deviation_pct"] = ((df["mtld"] - df["decade_median"]) / df["decade_median"] * 100).round(2)
    df["is_outlier"]    = (df["z_score"].abs() >= OUTLIER_SIGMA_THRESHOLD).astype(int)

    logger.info(f"Outliers détectés : {df['is_outlier'].sum()} livres (seuil = ±{OUTLIER_SIGMA_THRESHOLD}σ)")
    return df


def mark_outliers_in_db(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    outlier_ids = df[df["is_outlier"] == 1]["book_id"].tolist()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE books SET is_outlier = 0")
        if outlier_ids:
            cur.execute(
                "UPDATE books SET is_outlier = 1 WHERE book_id = ANY(%s)",
                (outlier_ids,)
            )
        conn.commit()
        cur.close()
        logger.info(f"{len(outlier_ids)} outliers marqués dans la base")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur marquage outliers : {e}")

# ── Modèle 1 — Régression OLS (H1) ───────────────────────────────────────────

def run_ols_regression(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    logger.info("Régression OLS : année → MTLD")

    X      = sm.add_constant(df["year"].values)
    y      = df["mtld"].values
    model  = sm.OLS(y, X).fit()
    slope  = model.params[1]
    pvalue = model.pvalues[1]
    r2     = model.rsquared
    conf   = model.conf_int(alpha=0.05)

    pre_war  = df[df["year"] <= 1945]["mtld"]
    post_war = df[df["year"] >  1945]["mtld"]
    h1_result = None
    if len(pre_war) >= 3 and len(post_war) >= 3:
        t_stat, t_pvalue = scipy_stats.ttest_ind(pre_war, post_war, equal_var=False)
        h1_result = {
            "pre_war_mean":  round(float(pre_war.mean()), 2),
            "post_war_mean": round(float(post_war.mean()), 2),
            "t_statistic":   round(float(t_stat), 4),
            "p_value":       round(float(t_pvalue), 6),
            "significant":   t_pvalue < 0.05,
            "direction":     "decrease" if pre_war.mean() > post_war.mean() else "increase",
        }

    result = {
        "model":          "OLS_year_to_mtld",
        "n_observations": len(df),
        "slope":          round(float(slope), 6),
        "intercept":      round(float(model.params[0]), 4),
        "r_squared":      round(float(r2), 6),
        "p_value":        round(float(pvalue), 6),
        "conf_int_95":    [round(float(conf[1, 0]), 6), round(float(conf[1, 1]), 6)],
        "significant":    bool(pvalue < 0.05),
        "h1_pre_post_war_test": h1_result,
    }

    if result["significant"]:
        direction = "diminue" if slope < 0 else "augmente"
        logger.success(f"H1 — Tendance significative (p={pvalue:.4f}) : MTLD {direction} (R²={r2:.4f})")
    else:
        logger.info(f"H1 — Aucune tendance significative (p={pvalue:.4f}, R²={r2:.4f})")

    return result

# ── Modèle 2 — Test H2 : variance intra-décennie ─────────────────────────────

def test_variance_h2(df: pd.DataFrame) -> dict:
    logger.info("Test H2 : variance intra-décennie avant/après 1920")

    pre_1920  = df[df["decade"] <  1920]["mtld"].dropna()
    post_1920 = df[df["decade"] >= 1920]["mtld"].dropna()

    result = {
        "model":            "levene_variance_test",
        "pre_1920_n":       int(len(pre_1920)),
        "pre_1920_std":     round(float(pre_1920.std()), 4) if len(pre_1920) >= 2 else None,
        "post_1920_n":      int(len(post_1920)),
        "post_1920_std":    round(float(post_1920.std()), 4) if len(post_1920) >= 2 else None,
        "levene_statistic": None,
        "levene_p_value":   None,
        "significant":      None,
        "h2_supported":     None,
    }

    if len(pre_1920) >= 3 and len(post_1920) >= 3:
        levene_stat, levene_p = scipy_stats.levene(pre_1920, post_1920)
        result["levene_statistic"] = round(float(levene_stat), 4)
        result["levene_p_value"]   = round(float(levene_p), 6)
        result["significant"]      = bool(levene_p < 0.05)
        result["h2_supported"]     = bool(levene_p < 0.05 and pre_1920.std() > post_1920.std())

        if result["h2_supported"]:
            logger.success(f"H2 SUPPORTÉE — Variance avant 1920 ({pre_1920.std():.2f}) > après ({post_1920.std():.2f})")
        else:
            logger.info(f"H2 — résultat : {result['h2_supported']} (p={levene_p:.4f})")
    else:
        logger.warning("H2 — Données insuffisantes")

    decade_variance = (
        df.groupby("decade")["mtld"]
        .agg(["std", "count"])
        .rename(columns={"std": "mtld_std", "count": "n_books"})
        .reset_index()
    )
    result["variance_by_decade"] = decade_variance.to_dict("records")
    return result

# ── Modèle 3 — Clustering KMeans ─────────────────────────────────────────────

def run_kmeans_clustering(df: pd.DataFrame) -> dict:
    logger.info("Clustering KMeans sur métriques lexicales")

    feature_cols = ["mtld", "ttr", "word_count", "decade"]
    df_feat      = df[feature_cols].copy()

    imputer  = SimpleImputer(strategy="median")
    X        = imputer.fit_transform(df_feat)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = 2; best_score = -1; scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in KMEANS_RANGE:
            if k >= len(df):
                break
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X_scaled, labels)
            scores[k] = round(float(sil), 4)
            if sil > best_score:
                best_score = sil; best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km_final.fit_predict(X_scaled)

    cluster_profiles = (
        df.groupby("cluster")["mtld"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mtld_mean", "std": "mtld_std", "count": "n_books"})
        .round(4).reset_index().to_dict("records")
    )
    genre_overlap = _compute_overlap(df, "cluster", "genre")
    df["period"]  = pd.cut(df["decade"], bins=[1849, 1900, 1945, 1985],
                           labels=["1850-1900", "1900-1945", "1945-1980"])
    period_overlap = _compute_overlap(df, "cluster", "period")

    result = {
        "model":             "kmeans",
        "k_optimal":         int(best_k),
        "silhouette_scores": scores,
        "best_silhouette":   round(float(best_score), 4),
        "cluster_profiles":  cluster_profiles,
        "genre_overlap":     genre_overlap,
        "period_overlap":    period_overlap,
        "book_clusters":     df[["book_id", "cluster", "mtld", "decade", "genre"]].to_dict("records"),
    }
    logger.success(f"KMeans — k={best_k} optimal (silhouette={best_score:.4f})")
    return result


def _compute_overlap(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    try:
        ct = pd.crosstab(df[col_a], df[col_b], normalize="index")
        return ct.round(3).to_dict()
    except Exception:
        return {}

# ── Modèle 4 — Random Forest + Feature Importance (H3) ───────────────────────

def run_feature_importance(df: pd.DataFrame) -> dict:
    logger.info("Random Forest : feature importance pour H3")

    df_model = df[["mtld", "year", "decade", "genre", "source", "word_count"]].copy()
    df_model = df_model.dropna(subset=["mtld"])

    le_genre  = LabelEncoder()
    le_source = LabelEncoder()
    df_model["genre_enc"]  = le_genre.fit_transform(df_model["genre"].fillna("inconnu"))
    df_model["source_enc"] = le_source.fit_transform(df_model["source"].fillna("inconnu"))

    feature_names = ["year", "decade", "genre_enc", "source_enc", "word_count"]
    X = df_model[feature_names].values
    y = df_model["mtld"].values

    if len(y) < 10:
        logger.warning(f"Données insuffisantes pour Random Forest ({len(y)} livres)")
        return {"model": "random_forest", "error": "insufficient_data"}

    imputer = SimpleImputer(strategy="median")
    X       = imputer.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_features="sqrt")
    rf.fit(X, y)

    importances = dict(zip(feature_names, rf.feature_importances_.round(6).tolist()))
    importances_readable = {
        "year":       importances["year"],
        "decade":     importances["decade"],
        "genre":      importances["genre_enc"],
        "source":     importances["source_enc"],
        "word_count": importances["word_count"],
    }

    h3_supported = importances_readable["genre"] > importances_readable["decade"]

    rf_oob = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                    max_features="sqrt", oob_score=True)
    rf_oob.fit(X, y)
    oob_score = round(float(rf_oob.oob_score_), 4)

    result = {
        "model":               "random_forest",
        "n_estimators":        200,
        "n_observations":      len(y),
        "oob_r_squared":       oob_score,
        "feature_importances": importances_readable,
        "h3_genre_vs_decade": {
            "genre_importance":  round(importances_readable["genre"], 6),
            "decade_importance": round(importances_readable["decade"], 6),
            "h3_supported":      bool(h3_supported),
        },
        "genre_mapping":  dict(zip(le_genre.classes_.tolist(), le_genre.transform(le_genre.classes_).tolist())),
        "source_mapping": dict(zip(le_source.classes_.tolist(), le_source.transform(le_source.classes_).tolist())),
    }

    if h3_supported:
        logger.success(f"H3 SUPPORTÉE — genre ({importances_readable['genre']:.4f}) > décennie ({importances_readable['decade']:.4f})")
    else:
        logger.info(f"H3 INFIRMÉE — décennie ({importances_readable['decade']:.4f}) >= genre ({importances_readable['genre']:.4f})")

    return result

# ── Pipeline principale ───────────────────────────────────────────────────────

def run_analysis(
    conn: psycopg2.extensions.connection,
    min_books: int = MIN_BOOKS_PER_DECADE,
    output_path: Path | None = None,
) -> dict:
    df = load_data(conn, min_books=min_books)
    if df.empty:
        return {"error": "no_data"}

    df = detect_outliers(df)
    mark_outliers_in_db(conn, df)

    decade_stats = (
        df.groupby("decade")["mtld"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .round(4).reset_index()
        .rename(columns={"count": "n_books"})
        .to_dict("records")
    )

    ols_result = run_ols_regression(df)
    h2_result  = test_variance_h2(df)
    km_result  = run_kmeans_clustering(df)
    rf_result  = run_feature_importance(df)

    h1_supported = ols_result.get("significant") and ols_result.get("slope", 0) < 0
    h2_supported = h2_result.get("h2_supported")
    h3_supported = rf_result.get("h3_genre_vs_decade", {}).get("h3_supported")

    hypotheses_summary = {
        "H1_mtld_decreases_post_war": {
            "supported": h1_supported,
            "p_value":   ols_result.get("p_value"),
            "slope":     ols_result.get("slope"),
        },
        "H2_higher_variance_pre_1920": {
            "supported": h2_supported,
            "p_value":   h2_result.get("levene_p_value"),
        },
        "H3_genre_better_predictor": {
            "supported":         h3_supported,
            "genre_importance":  rf_result.get("h3_genre_vs_decade", {}).get("genre_importance"),
            "decade_importance": rf_result.get("h3_genre_vs_decade", {}).get("decade_importance"),
        },
    }

    logger.info("═" * 50)
    for h, res in hypotheses_summary.items():
        status = "✅ SUPPORTÉE" if res.get("supported") else (
            "❌ INFIRMÉE" if res.get("supported") is False else "⚠️ INDÉTERMINÉE"
        )
        logger.info(f"{h} : {status}")
    logger.info("═" * 50)

    results = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "n_books_analyzed":   len(df),
        "n_outliers":         int(df["is_outlier"].sum()),
        "decade_stats":       decade_stats,
        "hypotheses_summary": hypotheses_summary,
        "ols_regression":     ols_result,
        "variance_test_h2":   h2_result,
        "kmeans_clustering":  km_result,
        "random_forest_h3":   rf_result,
        "outlier_books":      df[df["is_outlier"] == 1][
            ["book_id", "title", "author", "year", "decade", "mtld", "decade_median", "deviation_pct"]
        ].to_dict("records"),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.success(f"Résultats sauvegardés : {output_path}")

    return results

# ── Entrypoint ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse ML : OLS, KMeans, Random Forest.")
    parser.add_argument("--output",    default=str(OUTPUT_DIR / "analysis_results.json"))
    parser.add_argument("--min-books", type=int, default=MIN_BOOKS_PER_DECADE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not verify_db():
        logger.error("Base non initialisée.")
        sys.exit(1)

    conn = get_connection()
    try:
        results = run_analysis(conn=conn, min_books=args.min_books, output_path=Path(args.output))
        if "error" in results:
            logger.error(f"Analyse impossible : {results['error']}")
            sys.exit(1)
    finally:
        conn.close()

"""
dashboard/app.py — Dashboard LexicoTrend FR
============================================
Interface Streamlit en 3 vues :
    Vue 1 — Tendance temporelle
    Vue 2 — Top / Flop lexical
    Vue 3 — Fiche œuvre

Usage :
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from init_db import get_connection, verify_db

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

ANALYSIS_JSON = Path("data/processed/analysis_results.json")

# ── Palette & style ───────────────────────────────────────────────────────────

PALETTE = {
    "primary":  "#1E3A5F",
    "accent":   "#E8853D",
    "light":    "#F5EFE6",
    "success":  "#2E7D5E",
    "warning":  "#C0392B",
    "neutral":  "#6B7280",
    "bg":       "#0F1B2D",
    "card":     "#162235",
}

DECADE_COLOR_SCALE = px.colors.sequential.Blues

# ── Streamlit config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LexicoTrend FR",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
        background-color: #0F1B2D;
        color: #E8E0D5;
    }
    h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif; color: #F5EFE6; }
    .stMetric {
        background: #162235; border-radius: 8px; padding: 1rem;
        border-left: 3px solid #E8853D;
    }
    .stMetric label { color: #9CA3AF !important; font-size: 0.8rem !important; }
    div[data-testid="stSidebar"] { background: #0A1424; border-right: 1px solid #1E3A5F; }
    .hypothesis-card { background: #162235; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; border-left: 4px solid; }
    .hyp-true  { border-color: #2E7D5E; }
    .hyp-false { border-color: #C0392B; }
    .hyp-null  { border-color: #6B7280; }
    .interpretation-box {
        background: #1A2E48; border-radius: 8px; padding: 1rem 1.2rem;
        border-left: 4px solid #E8853D; font-style: italic; color: #D4C5B0; margin-top: 1rem;
    }
    .ocr-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des données ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_books() -> pd.DataFrame:
    if not verify_db():
        return pd.DataFrame()
    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT
            book_id, title, author, year, decade, genre, source,
            word_count, unique_words, sentence_count,
            ttr, mtld, hdd, mtld_ma_bid,
            ocr_quality_before, ocr_quality_after, ocr_corrected,
            claude_interpretation, is_outlier,
            collected_at, metrics_at
        FROM books
        WHERE mtld IS NOT NULL
        ORDER BY year
    """, conn)
    conn.close()
    df["genre"]       = df["genre"].fillna("inconnu")
    df["decade_label"] = df["decade"].astype(str) + "s"
    return df


@st.cache_data(ttl=300)
def load_analysis() -> dict:
    if ANALYSIS_JSON.exists():
        with open(ANALYSIS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_decade_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["decade", "decade_label"])["mtld"]
        .agg(mean="mean", median="median", std="std", count="count", min="min", max="max")
        .reset_index()
        .sort_values("decade")
    )

# ── Composants réutilisables ──────────────────────────────────────────────────

def _ocr_badge(score: float | None) -> str:
    if score is None:
        return '<span class="ocr-badge" style="background:#2a3a4a;color:#6B7280">N/A</span>'
    color = "#2E7D5E" if score >= 0.90 else ("#E8853D" if score >= 0.80 else "#C0392B")
    label = "Bon" if score >= 0.90 else ("Moyen" if score >= 0.80 else "Dégradé")
    return f'<span class="ocr-badge" style="background:{color}22;color:{color}">{label} {score:.2f}</span>'


def _hypothesis_card(label: str, description: str, supported: bool | None, detail: str = "") -> None:
    css_class = "hyp-true" if supported else ("hyp-false" if supported is False else "hyp-null")
    icon      = "✅" if supported else ("❌" if supported is False else "⚠️")
    status    = "Supportée" if supported else ("Infirmée" if supported is False else "Indéterminée")
    st.markdown(f"""
    <div class="hypothesis-card {css_class}">
        <strong style="font-family:'Playfair Display',serif;font-size:1rem">{label}</strong>
        <span style="float:right;font-size:1.1rem">{icon} <em style="font-size:0.8rem;color:#9CA3AF">{status}</em></span>
        <p style="margin:0.3rem 0 0;color:#B0A898;font-size:0.88rem">{description}</p>
        {"<p style='margin:0.4rem 0 0;color:#6B7280;font-size:0.8rem'>" + detail + "</p>" if detail else ""}
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem">
            <h1 style="font-size:1.6rem;margin:0;color:#F5EFE6">📚 LexicoTrend FR</h1>
            <p style="color:#6B7280;font-size:0.8rem;margin:0.2rem 0 0">Richesse lexicale · 1850–1980</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        vue = st.radio(
            "Navigation",
            ["📈 Tendance temporelle", "🏆 Top / Flop lexical", "📖 Fiche œuvre"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Filtres**")

        if df.empty:
            return vue, [], []

        genres_dispo = sorted(df["genre"].unique().tolist())
        genres_sel   = st.multiselect("Genres",  options=genres_dispo,  default=genres_dispo)
        sources_dispo = sorted(df["source"].unique().tolist())
        sources_sel   = st.multiselect("Sources", options=sources_dispo, default=sources_dispo)

        st.divider()
        st.metric("Livres analysés",    len(df))
        st.metric("Décennies couvertes", df["decade"].nunique())
        if not df["mtld"].empty:
            st.metric("MTLD médian", f"{df['mtld'].median():.1f}")

        st.divider()
        st.markdown(
            '<p style="color:#3B5275;font-size:0.72rem;text-align:center">'
            'Jedha Bootcamp · Portfolio Data Analyst<br>'
            'Pipeline : Gallica + Gutenberg → spaCy + ByT5 → Claude API'
            '</p>',
            unsafe_allow_html=True,
        )

    return vue, genres_sel, sources_sel

# ── Vue 1 — Tendance temporelle ───────────────────────────────────────────────

def render_vue_tendance(df: pd.DataFrame, analysis: dict) -> None:
    st.markdown("## Évolution de la richesse lexicale")
    st.markdown('<p style="color:#9CA3AF;margin-top:-0.5rem">Score MTLD par décennie · 1850–1980</p>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    decade_stats = load_decade_stats(df)
    col1, col2   = st.columns([3, 1])

    with col1:
        fig = go.Figure()
        for _, row in decade_stats.iterrows():
            decade_df = df[df["decade"] == row["decade"]]
            fig.add_trace(go.Box(
                y=decade_df["mtld"], name=row["decade_label"], boxmean=True,
                marker_color=PALETTE["primary"], line_color=PALETTE["accent"],
                fillcolor="rgba(30,58,95,0.27)", showlegend=False,
                hovertemplate="<b>%{x}</b><br>MTLD: %{y:.1f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=decade_stats["decade_label"], y=decade_stats["median"],
            mode="lines+markers", name="Médiane MTLD",
            line=dict(color=PALETTE["accent"], width=2.5),
            marker=dict(size=8, color=PALETTE["accent"]),
            hovertemplate="<b>%{x}</b><br>Médiane MTLD : %{y:.1f}<extra></extra>",
        ))
        
        fig.update_layout(
            plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
            font=dict(color="#E8E0D5", family="Source Sans 3"),
            xaxis=dict(title="Décennie", gridcolor="#1E3A5F", linecolor="#1E3A5F", tickfont=dict(size=11)),
            yaxis=dict(title="Score MTLD", gridcolor="#1E3A5F", linecolor="#1E3A5F"),
            hoverlabel=dict(bgcolor=PALETTE["card"], font_color="#E8E0D5"),
            margin=dict(l=10, r=10, t=20, b=10), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Statistiques par décennie**")
        display_df = decade_stats[["decade_label", "median", "std", "count"]].rename(columns={
            "decade_label": "Décennie", "median": "Médiane", "std": "Écart-type", "count": "N livres",
        })
        display_df["Médiane"]    = display_df["Médiane"].round(1)
        display_df["Écart-type"] = display_df["Écart-type"].round(1)
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=420)

    st.divider()
    st.markdown("### Résultats des hypothèses de recherche")

    hyp  = analysis.get("hypotheses_summary", {})
    h1   = hyp.get("H1_mtld_decreases_post_war", {})
    h2   = hyp.get("H2_higher_variance_pre_1920", {})
    h3   = hyp.get("H3_genre_better_predictor", {})
    cols = st.columns(3)

    with cols[0]:
        p1 = h1.get("p_value")
        _hypothesis_card("H1 — Déclin post-guerre",
                         "Le MTLD moyen diminue-t-il significativement après 1945 ?",
                         h1.get("supported"), f"p = {p1:.4f}" if p1 is not None else "Non calculé")
    with cols[1]:
        p2 = h2.get("p_value")
        _hypothesis_card("H2 — Variance avant 1920",
                         "La variance intra-décennie est-elle plus élevée avant 1920 ?",
                         h2.get("supported"), f"Levene p = {p2:.4f}" if p2 is not None else "Non calculé")
    with cols[2]:
        gi = h3.get("genre_importance"); di = h3.get("decade_importance")
        _hypothesis_card("H3 — Genre vs Décennie",
                         "Le genre prédit-il mieux le MTLD que la décennie ?",
                         h3.get("supported"),
                         f"Genre : {gi:.4f} vs Décennie : {di:.4f}" if gi and di else "Non calculé")

    if not hyp:
        st.info("Exécuter `python ml/analysis.py` pour générer les résultats.", icon="ℹ️")

    st.divider()
    st.markdown("### Carte thermique MTLD — Décennie × Genre")
    pivot = df.pivot_table(values="mtld", index="genre", columns="decade_label", aggfunc="median")
    if not pivot.empty:
        fig_heat = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto", labels=dict(color="MTLD médian"))
        fig_heat.update_layout(
            plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
            font=dict(color="#E8E0D5", family="Source Sans 3"),
            coloraxis_colorbar=dict(tickfont=dict(color="#E8E0D5")),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            xaxis=dict(tickfont=dict(size=10), title=""),
            yaxis=dict(tickfont=dict(size=10), title=""),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Vue 2 — Top / Flop lexical ────────────────────────────────────────────────

def render_vue_palmaresle(df: pd.DataFrame) -> None:
    st.markdown("## Palmarès lexical")
    st.markdown('<p style="color:#9CA3AF;margin-top:-0.5rem">Œuvres les plus et moins riches — score MTLD</p>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    n_display    = st.slider("Nombre d'œuvres à afficher", 5, 20, 10)
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown(f"### 🏆 Top {n_display} — plus riches")
        top_df = df.nlargest(n_display, "mtld")[["title", "author", "year", "genre", "mtld", "is_outlier"]].reset_index(drop=True)
        _render_ranking_chart(top_df, color=PALETTE["success"], ascending=False)

    with col_bot:
        st.markdown(f"### 📉 Bottom {n_display} — moins riches")
        bot_df = df.nsmallest(n_display, "mtld")[["title", "author", "year", "genre", "mtld", "is_outlier"]].reset_index(drop=True)
        _render_ranking_chart(bot_df, color=PALETTE["warning"], ascending=True)

    st.divider()
    st.markdown("### Distribution MTLD × Année")

    fig_scatter = px.scatter(
        df, x="year", y="mtld", color="genre", size="word_count", size_max=20,
        hover_data={"title": True, "author": True, "year": True, "genre": True,
                    "mtld": ":.1f", "word_count": ":,.0f"},
        color_discrete_map={"naturaliste": "#4A90D9", "historique": "#E8853D",
                             "populaire": "#2E7D5E", "roman_realiste": "#9B59B6", "inconnu": "#6B7280"},
        labels={"year": "Année", "mtld": "Score MTLD", "genre": "Genre"},
    )

    outliers = df[df["is_outlier"] == 1]
    if not outliers.empty:
        fig_scatter.add_trace(go.Scatter(
            x=outliers["year"], y=outliers["mtld"], mode="markers", name="Outliers",
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=PALETTE["accent"], width=2), symbol="circle-open"),
            hovertemplate="<b>%{customdata[0]}</b><extra>Outlier</extra>",
            customdata=outliers[["title"]].values,
        ))

    fig_scatter.update_layout(
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
        font=dict(color="#E8E0D5", family="Source Sans 3"),
        xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F"),
        legend=dict(bgcolor=PALETTE["card"], bordercolor="#1E3A5F", borderwidth=1),
        hoverlabel=dict(bgcolor=PALETTE["card"], font_color="#E8E0D5"),
        margin=dict(l=10, r=10, t=20, b=10), height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def _render_ranking_chart(rank_df: pd.DataFrame, color: str, ascending: bool) -> None:
    rank_df       = rank_df.copy()
    rank_df["label"] = rank_df["title"].str[:30] + " (" + rank_df["year"].astype(str) + ")"
    if not ascending:
        rank_df = rank_df.sort_values("mtld", ascending=True)

    fig = go.Figure(go.Bar(
        x=rank_df["mtld"], y=rank_df["label"], orientation="h",
        marker_color=f"{color}CC", marker_line=dict(color=color, width=1),
        hovertemplate="<b>%{y}</b><br>MTLD : %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
        font=dict(color="#E8E0D5", family="Source Sans 3", size=11),
        xaxis=dict(gridcolor="#1E3A5F", title="Score MTLD"),
        yaxis=dict(gridcolor="#1E3A5F", title=""),
        margin=dict(l=10, r=10, t=10, b=10), height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Vue 3 — Fiche œuvre ───────────────────────────────────────────────────────

def render_vue_fiche(df: pd.DataFrame) -> None:
    st.markdown("## Fiche œuvre")
    st.markdown('<p style="color:#9CA3AF;margin-top:-0.5rem">Métriques détaillées et interprétation Claude API</p>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    col_sel, col_filter = st.columns([3, 1])
    with col_filter:
        decade_filter = st.selectbox("Filtrer par décennie", ["Toutes"] + sorted(df["decade_label"].unique().tolist()))
    with col_sel:
        df_filtered  = df if decade_filter == "Toutes" else df[df["decade_label"] == decade_filter]
        book_options = [
            f"{row['title']} — {row['author']} ({row['year']})"
            for _, row in df_filtered.sort_values("year").iterrows()
        ]
        if not book_options:
            st.info("Aucun livre pour ce filtre.")
            return
        selected_label = st.selectbox("Sélectionner une œuvre", book_options)

    selected_idx = book_options.index(selected_label)
    book         = df_filtered.sort_values("year").iloc[selected_idx]

    st.divider()

    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown(f"<h2 style='font-family:Playfair Display,serif;margin:0'>{book['title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#9CA3AF;margin:0.2rem 0'>{book['author']} · {book['year']} · <em>{book['genre']}</em></p>", unsafe_allow_html=True)
    with col_badge:
        if book["is_outlier"]:
            st.markdown(
                '<div style="background:#E8853D22;border:1px solid #E8853D;border-radius:8px;'
                'padding:0.5rem;text-align:center;margin-top:0.5rem">'
                '<span style="color:#E8853D;font-size:0.9rem">⚡ Outlier statistique</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    cols = st.columns(4)
    metrics = [
        ("Score MTLD",         f"{book['mtld']:.1f}"        if pd.notna(book['mtld'])        else "N/A", "Métrique principale"),
        ("Type-Token Ratio",   f"{book['ttr']:.4f}"         if pd.notna(book['ttr'])          else "N/A", "Ratio types/tokens"),
        ("Mots totaux",        f"{int(book['word_count']):,}" if pd.notna(book['word_count'])  else "N/A", "Tokens analysés"),
        ("Vocabulaire distinct", f"{int(book['unique_words']):,}" if pd.notna(book['unique_words']) else "N/A", "Types uniques"),
    ]
    for col, (label, value, help_txt) in zip(cols, metrics):
        with col:
            st.metric(label, value, help=help_txt)

    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("HD-D",                f"{book['hdd']:.4f}"       if pd.notna(book.get('hdd'))        else "N/A")
    with cols2[1]:
        st.metric("MTLD Bidirectionnel", f"{book['mtld_ma_bid']:.1f}" if pd.notna(book.get('mtld_ma_bid')) else "N/A")
    with cols2[2]:
        sc = book.get("sentence_count")
        st.metric("Phrases", f"{int(sc):,}" if pd.notna(sc) and sc else "N/A")
    with cols2[3]:
        st.metric("Source", str(book["source"]).capitalize())

    st.divider()
    st.markdown("**Qualité OCR**")
    ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
    with ocr_col1:
        st.markdown(f"Avant correction : {_ocr_badge(book.get('ocr_quality_before'))}", unsafe_allow_html=True)
    with ocr_col2:
        st.markdown(f"Après correction : {_ocr_badge(book.get('ocr_quality_after'))}", unsafe_allow_html=True)
    with ocr_col3:
        corrected = book.get("ocr_corrected", 0)
        label = "✅ ByT5 appliqué" if corrected else "⬜ Non corrigé (score suffisant)"
        st.markdown(f"<span style='color:#9CA3AF;font-size:0.9rem'>{label}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Position dans la décennie**")
    decade_df = df[df["decade"] == book["decade"]]["mtld"].dropna()
    if not decade_df.empty:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=decade_df, name=f"Décennie {int(book['decade'])}s",
            marker_color=f"{PALETTE['primary']}88", nbinsx=12,
            hovertemplate="MTLD : %{x:.1f}<br>N : %{y}<extra></extra>",
        ))
        fig_dist.add_vline(
            x=book["mtld"], line_color=PALETTE["accent"], line_width=2.5,
            annotation_text=book["title"][:25], annotation_font_color=PALETTE["accent"],
            annotation_position="top right",
        )
        fig_dist.update_layout(
            plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
            font=dict(color="#E8E0D5", family="Source Sans 3"),
            xaxis=dict(gridcolor="#1E3A5F", title="Score MTLD"),
            yaxis=dict(gridcolor="#1E3A5F", title="Nombre d'œuvres"),
            margin=dict(l=10, r=10, t=20, b=10), height=250, showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        percentile = (decade_df < book["mtld"]).mean() * 100
        st.markdown(
            f"<p style='color:#9CA3AF;font-size:0.85rem;text-align:center'>"
            f"Ce livre se situe au <strong style='color:#E8853D'>{percentile:.0f}e percentile</strong> "
            f"de sa décennie ({len(decade_df)} œuvres comparées)</p>",
            unsafe_allow_html=True,
        )

    interpretation = book.get("claude_interpretation")
    if interpretation and str(interpretation).strip():
        st.divider()
        st.markdown("**Interprétation Claude API**")
        model_name = os.getenv("ANTHROPIC_MODEL_ANALYSIS", "claude-sonnet-4-6")
        st.markdown(
            f'<div class="interpretation-box">'
            f'<p style="margin:0">💬 {interpretation}</p>'
            f'<p style="margin:0.5rem 0 0;color:#4B6A8A;font-size:0.75rem">Généré par Claude Sonnet · Modèle : {model_name}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif book["is_outlier"]:
        st.info("Outlier sans interprétation. Exécuter `python enrichment/claude_enricher.py --task anomalies`", icon="ℹ️")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not verify_db():
        st.error("Base de données non initialisée ou inaccessible.\n\nExécuter : `python init_db.py`", icon="🚨")
        st.stop()

    df       = load_books()
    analysis = load_analysis()

    vue, genres_sel, sources_sel = render_sidebar(df)

    if not df.empty and genres_sel and sources_sel:
        df = df[df["genre"].isin(genres_sel) & df["source"].isin(sources_sel)]

    if "Tendance" in vue:
        render_vue_tendance(df, analysis)
    elif "Top" in vue:
        render_vue_palmaresle(df)
    elif "Fiche" in vue:
        render_vue_fiche(df)


if __name__ == "__main__":
    main()

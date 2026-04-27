# LexicoTrend FR 📚

> **La richesse lexicale des romans best-sellers français a-t-elle évolué depuis le XIXe siècle ?**
> Pipeline NLP end-to-end sur un corpus de 59 œuvres du domaine public (1850–1940).

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-ByT5-FFD21E?logo=huggingface&logoColor=black)
![Claude API](https://img.shields.io/badge/Claude_API-Anthropic-D4A017)
![Docker](https://img.shields.io/badge/Docker-ARM64-2496ED?logo=docker&logoColor=white)
![n8n](https://img.shields.io/badge/Orchestration-n8n-EA4B71)
![Raspberry Pi](https://img.shields.io/badge/Infra-Raspberry_Pi_5-A22846?logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-En%20production-green)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

---

![Dashboard LexicoTrend — Vue tendance temporelle](docs/dashboard_tendance.png)

---

## 🎯 Problème résolu

Il n'existe pas de dataset sur l'évolution stylistique des romans populaires français.
Ce projet le construit de zéro : collecte automatisée depuis Gallica (BnF) et Project Gutenberg,
post-correction OCR en 3 couches, métriques de diversité lexicale robustes (MTLD),
modélisation statistique, et dashboard interactif hébergé sur Raspberry Pi 5.

**Hypothèses testées :**
- H1 : Le MTLD moyen diminue après la Seconde Guerre mondiale (simplification stylistique)
- H2 : La variance intra-décennie est plus élevée avant 1920 (hétérogénéité des styles)
- H3 : Le genre littéraire est un meilleur prédicteur du MTLD que la décennie

---

## 🔍 Résultats clés

Sur un corpus de 56 romans français (1850–1940), analysés via une pipeline NLP complète :

- **H1 ✅ Supportée** — Le MTLD diminue significativement avec le temps (p=0.0015, R²=0.17) — tendance robuste confirmée sur 9 décennies
- **H2 ❌ Infirmée** — La variance intra-décennie avant/après 1920 n'est pas significativement différente (p=0.86) — résultat inversé par rapport au petit corpus initial, ce qui souligne l'importance de la taille du corpus
- **H3 ❌ Infirmée** — La décennie (importance=0.18) prédit mieux le MTLD que le genre (importance=0.09) — l'époque compte davantage que le registre littéraire

---

## 🚀 Démo

Le dashboard tourne en local sur Raspberry Pi 5 (réseau domestique).

```bash
# Cloner et lancer localement
git clone https://github.com/toto-blanco/LexicoTrend-Fr.git
cd LexicoTrend-Fr
cp .env.example .env  # remplir avec vos credentials
docker build -t lexicotrend:latest .
docker run -d -p 8501:8501 lexicotrend:latest
# → http://localhost:8501
```

---

## 🏗️ Architecture

```
n8n (scheduler — Raspberry Pi 5)
  ├── Workflow collecte  : cron hebdo → Gallica + Gutenberg → PostgreSQL
  └── Workflow trigger   : nouveau fichier .txt → pipeline complète

Pipeline Python (conteneur Docker ARM64)
  ├── scraping/          API SRU Gallica + GutendexAPI
  ├── processing/        Regex → ByT5 post-OCR → spaCy → MTLD/TTR
  ├── enrichment/        Claude API (OCR couche 3 + interprétation outliers)
  ├── ml/                OLS (H1) · Levene (H2) · KMeans · Random Forest (H3)
  └── dashboard/         Streamlit 3 vues — dark theme éditorial

Stockage
  ├── PostgreSQL          métriques + métadonnées + interprétations Claude
  └── Nextcloud (HDD)     archive textes bruts (.txt UTF-8)
```

---

## 🛠️ Stack technique

| Phase | Outil | Justification |
|---|---|---|
| Orchestration | n8n (auto-hébergé) | Scheduler visuel, webhook natif, déjà installé sur le Pi |
| Collecte | requests + BeautifulSoup | APIs publiques Gallica (SRU/CQL) et Gutenberg |
| Post-OCR couche 1 | Python `re` | Artefacts évidents — gratuit, instantané |
| Post-OCR couche 2 | ByT5-base (HuggingFace) | Spécialisé OCR historique, ~65% réduction CER, tourne en CPU |
| Post-OCR couche 3 | Claude API Haiku | Passages ambigus : vieux français vs artefact OCR |
| NLP | spaCy `fr_core_news_md` | Tokenisation FR native, parser/NER désactivés pour perf |
| Métriques | lexicalrichness | TTR, MTLD (McCarthy & Jarvis, 2010), HD-D en 3 lignes |
| ML | statsmodels + scikit-learn | OLS, test de Levene, KMeans, Random Forest avec OOB score |
| LLM analyse | Claude API Sonnet | Interprétation contextuelle des outliers statistiques |
| Dashboard | Streamlit + Plotly | Dark theme éditorial, 3 vues interactives |
| Base de données | PostgreSQL | Conteneur Docker existant sur le Pi, PERCENTILE_CONT natif |
| Stockage brut | Nextcloud (WebDAV) | HDD 4To déjà en place, backup naturel |
| Infra | Raspberry Pi 5 + Docker | Home lab ARM64, traitement batch nocturne |

---

## 💡 Pourquoi ces choix techniques ?

**MTLD plutôt que TTR** : le Type-Token Ratio est biaisé par la longueur du texte — deux romans de longueurs différentes ne sont pas comparables. MTLD (McCarthy & Jarvis, 2010) est validé comme la mesure la plus robuste à ce biais, ce qui est critique ici car les romans du corpus varient entre 50 000 et 400 000 tokens.

**ByT5 avant Claude pour l'OCR** : les tests sur LLMs généralistes en zero-shot révèlent un problème documenté — les erreurs OCR perturbent la détection de langue dans l'espace d'embedding, poussant parfois le modèle à générer du texte corrigé en anglais (Biriuchinskii et al., 2025). ByT5, entraîné spécifiquement sur ce type de bruit, est plus fiable pour le gros du travail. Claude n'intervient qu'en dernier recours sur les passages ambigus.

**PostgreSQL plutôt que SQLite** : un conteneur PostgreSQL était déjà opérationnel sur le Pi. PostgreSQL apporte `PERCENTILE_CONT` (calcul natif de médiane) et `STDDEV_POP`, utiles pour les statistiques par décennie, sans surcoût d'infrastructure.

**Docker sur le Pi** : isolation complète de l'environnement Python (PyTorch, spaCy, ByT5 = ~3 Go de dépendances), redémarrage automatique, et mise à jour via simple `git pull` + rebuild.

---

## ⚡ Installation

### Prérequis

- Python 3.11+ (développement local) ou Docker (déploiement Pi)
- Instance PostgreSQL accessible
- Compte API Anthropic

### Développement local

```bash
git clone https://github.com/toto-blanco/LexicoTrend-Fr.git
cd lexicotrend-fr

python -m venv .venv
source .venv/bin/activate

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_md

cp .env.example .env
# Remplir .env avec vos credentials PostgreSQL et clé Anthropic

python init_db.py
streamlit run dashboard/app.py
```

### Déploiement Docker (Raspberry Pi 5)

```bash
# Sur le Pi via SSH
git clone https://github.com/toto-blanco/LexicoTrend-Fr.git
cd lexicotrend-fr
cp .env.example .env && nano .env

docker build -t lexicotrend:latest .

docker run -d \
  --name lexicotrend \
  --network <reseau-postgres> \
  --restart unless-stopped \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  lexicotrend:latest

docker exec lexicotrend python init_db.py
```

### Lancer la pipeline manuellement

```bash
docker exec lexicotrend python scraping/gutenberg.py --max-books 20
docker exec lexicotrend python scraping/gallica.py --max-books 20
docker exec lexicotrend python processing/clean.py --batch
docker exec lexicotrend python processing/ocr_corrector.py --batch
docker exec lexicotrend python processing/metrics.py --batch
docker exec lexicotrend python ml/analysis.py
```

---

## 📁 Structure du projet

```
lexicotrend-fr/
├── Dockerfile
├── .env.example
├── init_db.py                   ← schéma PostgreSQL
├── requirements.txt
├── scraping/
│   ├── gallica.py               ← API SRU BnF (protocole CQL)
│   ├── gutenberg.py             ← GutendexAPI
│   └── wikisource_collector.py  ← ws-export Wikisource (œuvres manquantes)
├── processing/
│   ├── clean.py                 ← post-OCR couche 1 (regex)
│   ├── ocr_corrector.py         ← post-OCR couche 2 (ByT5)
│   └── metrics.py               ← spaCy + lexicalrichness
├── enrichment/
│   └── claude_enricher.py       ← Claude API (OCR couche 3 + anomalies)
├── ml/
│   └── analysis.py              ← OLS · Levene · KMeans · Random Forest
├── dashboard/
│   └── app.py                   ← Streamlit (3 vues)
├── workflow_collecte.json        ← n8n : cron hebdo
├── workflow_trigger.json         ← n8n : trigger nouveau fichier
├── EDA.ipynb
└── data/
    ├── raw/                      ← textes bruts (non versionnés)
    └── processed/                ← textes traités + analysis_results.json
```

---

## 🔮 Évolutions prévues (V2)

- Extension du corpus sur 1980–2024 (previews légaux, Open Library)
- Comparaison FR vs EN par décennie (corpus Gutenberg anglophone)
- Détection de rupture structurelle avec `ruptures` (statsmodels)
- Onglet "Comparaison internationale" dans le dashboard

---

## 📚 Références académiques

- **Boros et al. (2024)** — *Post-Correction of Historical Text Transcripts with Large Language Models* — LaTeCH-CLfL @ ACL 2024
- **Thomas, Gaizauskas & Lu (2024)** — *Leveraging LLMs for Post-OCR Correction of Historical Newspapers* — LREC-COLING 2024
- **Biriuchinskii, Alrahabi & Roe (2025)** — *Using LLMs for post-OCR correction on historical French texts* — DH2025 (HAL)
- **McCarthy & Jarvis (2010)** — *MTLD, vocd-D, and HD-D: A validation study* — Behavior Research Methods

---

## Licence

CC BY-NC 4.0 — voir [LICENSE](LICENSE)

Usage libre avec attribution, pas d'usage commercial.
Plus d'informations : https://creativecommons.org/licenses/by-nc/4.0/

---

*Projet réalisé dans le cadre d'un portfolio data analyst post-bootcamp Jedha.*

---

## 🤖 Collaboration avec Claude (Anthropic)

Ce projet a été développé en collaboration active avec **Claude Sonnet** (Anthropic).

**Ce que Claude a fait :**
La rédaction de l'ensemble des scripts Python a été déléguée à Claude : scrapers (Gallica SRU, GutendexAPI, Wikisource ws-export), pipeline de traitement (`clean.py`, `ocr_corrector.py`, `metrics.py`), enrichissement LLM (`claude_enricher.py`), analyse ML (`analysis.py`), dashboard Streamlit (`app.py`), schéma PostgreSQL (`init_db.py`), Dockerfile ARM64, et workflows n8n. Claude a également assuré le débogage itératif en temps réel (conflits de dépendances, compilation ARM64, rate limits API, gestion mémoire spaCy sur Pi 5).

**Ce que j'ai fait :**
La direction du projet, les choix de recherche et la formulation des 3 hypothèses (H1/H2/H3), le choix du MTLD comme métrique principale, la sélection du corpus (58 œuvres, 9 décennies), la validation manuelle des résultats statistiques, le déploiement physique sur Raspberry Pi 5, et l'interprétation des résultats. Chaque script produit par Claude a été testé, validé et ajusté selon les résultats réels obtenus en production.

**Pourquoi cette transparence ?**
Utiliser un assistant IA pour accélérer la production de code est une compétence en soi — savoir formuler le bon problème, valider les outputs, et garder la maîtrise analytique. C'est exactement ce qu'un Data Analyst fait avec une équipe technique : piloter, valider, interpréter.

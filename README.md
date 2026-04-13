# LexicoTrend FR 📚

> **La richesse lexicale des romans best-sellers français a-t-elle évolué depuis le XIXe siècle ?**
> Pipeline NLP end-to-end sur un corpus de ~50 œuvres du domaine public (1850–1980).

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?logo=spacy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude_API-Anthropic-D4A017)
![n8n](https://img.shields.io/badge/Orchestration-n8n-EA4B71)
![Raspberry Pi](https://img.shields.io/badge/Infra-Raspberry_Pi_5-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

<!-- Screenshot ou GIF du dashboard à insérer ici une fois déployé -->
<!-- ![Dashboard LexicoTrend](docs/dashboard_preview.gif) -->

---

## Problème résolu

Il n'existe pas de dataset sur l'évolution stylistique des romans populaires français.
Ce projet le construit de zéro : collecte automatisée, post-correction OCR, métriques de
diversité lexicale robustes, modélisation statistique, et dashboard live accessible sans
préparation.

**Hypothèses testées :**
- H1 : le MTLD moyen diminue après la Seconde Guerre mondiale
- H2 : la variance intra-décennie est plus élevée avant 1920
- H3 : le genre littéraire est un meilleur prédicteur du MTLD que la décennie

---

## Architecture

```
n8n (scheduler)
  ├── Workflow 1 : Collecte Gallica / Gutenberg → Nextcloud /data/raw/
  └── Workflow 2 : Trigger pipeline Python sur nouveau fichier

Python pipeline (Raspberry Pi 5 — SSD 500Go)
  ├── scraping/       requests + BeautifulSoup — API Gallica & Gutenberg
  ├── processing/     Nettoyage regex → ByT5 post-OCR → spaCy → MTLD/TTR
  ├── enrichment/     Claude API — normalisation corpus + interprétations
  ├── ml/             Régression OLS + KMeans + Random Forest (scikit-learn)
  └── dashboard/      Streamlit — 3 vues interactives

SQLite (SSD)
  ├── books           métadonnées + métriques lexicales + score OCR
  └── anomalies       interprétations contextuelles générées par Claude

Nextcloud (HDD 4To)  →  archive textes bruts
```

---

## Stack technique

| Phase | Outil | Justification |
|---|---|---|
| Orchestration | n8n (auto-hébergé) | Scheduler visuel, déjà installé sur le Pi |
| Collecte | requests + BeautifulSoup | Suffisant pour APIs publiques Gallica/Gutenberg |
| Post-OCR couche 1 | Python `re` | Artefacts évidents, gratuit, instantané |
| Post-OCR couche 2 | ByT5-base (HuggingFace) | Spécialisé OCR historique, ~65% de réduction CER |
| Post-OCR couche 3 | Claude API Haiku | Passages ambigus uniquement — vieux français vs artefact |
| NLP | spaCy `fr_core_news_md` | Tokenisation FR native, batch processing |
| Métriques | lexicalrichness | TTR, MTLD, HD-D en 3 lignes |
| ML | scikit-learn + statsmodels | Régression OLS, KMeans, Random Forest |
| LLM analyse | Claude API Sonnet | Interprétation contextuelle des outliers |
| Dashboard | Streamlit + Plotly | Déployable sur le Pi, maîtrisé |
| Stockage brut | Nextcloud (WebDAV) | Déjà en place, backup naturel |
| Base de données | SQLite | Léger, sans serveur, versionnable |

---

## Installation

### Prérequis

- Python 3.11+
- Raspberry Pi 5 (ou tout système Linux/macOS)
- Instance Nextcloud accessible
- Compte API Anthropic avec clé

### 1. Cloner le repo

```bash
git clone https://github.com/votre-username/lexicotrend-fr.git
cd lexicotrend-fr
```

### 2. Environnement Python

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

> **Note Raspberry Pi :** installer PyTorch en version CPU uniquement :
> ```bash
> pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. Modèle spaCy français

```bash
python -m spacy download fr_core_news_md
```

### 4. Variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec vos clés API et chemins
```

### 5. Initialiser la base de données

```bash
python init_db.py
```

### 6. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

---

## Utilisation

### Lancer la collecte manuellement

```bash
# Collecter depuis Gutenberg
python scraping/gutenberg.py

# Collecter depuis Gallica
python scraping/gallica.py
```

### Traiter un texte brut

```bash
# Pipeline complète sur un fichier
python processing/clean.py --input data/raw/mon_livre.txt
python processing/ocr_corrector.py --input data/processed/mon_livre_clean.txt
python processing/metrics.py --input data/processed/mon_livre_corrected.txt
```

### Lancer l'analyse ML

```bash
python ml/analysis.py
```

---

## Structure du repo

```
lexicotrend-fr/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── init_db.py                   ← initialisation schéma SQLite
├── scraping/
│   ├── gallica.py               ← API REST BnF
│   └── gutenberg.py             ← dump CSV Project Gutenberg
├── processing/
│   ├── clean.py                 ← nettoyage regex (post-OCR couche 1)
│   ├── ocr_corrector.py         ← ByT5 post-correction (couche 2)
│   └── metrics.py               ← spaCy + lexicalrichness
├── enrichment/
│   └── claude_enricher.py       ← Claude API (OCR couche 3 + anomalies)
├── ml/
│   └── analysis.py              ← régression + clustering + feature importance
├── dashboard/
│   └── app.py                   ← Streamlit (3 vues)
├── n8n/
│   └── workflows/
│       ├── workflow_collecte.json
│       └── workflow_trigger.json
├── data/
│   ├── raw/.gitkeep             ← textes bruts (non versionnés)
│   └── processed/.gitkeep       ← textes traités (non versionnés)
├── logs/.gitkeep
└── notebooks/
    └── EDA.ipynb
```

---

## Insights clés

> *Section à compléter une fois les données collectées et analysées.*

---

## Références académiques

- **Boros et al. (2024)** — *Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study* — LaTeCH-CLfL @ ACL 2024
- **Thomas, Gaizauskas & Lu (2024)** — *Leveraging LLMs for Post-OCR Correction of Historical Newspapers* — LT4HALA @ LREC-COLING 2024
- **Biriuchinskii, Alrahabi & Roe (2025)** — *Using LLMs for post-OCR correction on historical French texts* — DH2025 (HAL)
- **McCarthy & Jarvis (2010)** — *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment* — Behavior Research Methods

---

## Licence

MIT — voir [LICENSE](LICENSE)

---

## Auteur

Projet réalisé dans le cadre d'un portfolio data analyst post-bootcamp Jedha.

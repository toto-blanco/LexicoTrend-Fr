FROM python:3.11-slim

# Dépendances système pour psycopg2 et compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements en premier pour profiter du cache Docker
COPY requirements.txt .

# PyTorch CPU uniquement — version compatible ARM64 Pi 5
RUN pip install --no-cache-dir torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Reste des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Modèle spaCy français
RUN python -m spacy download fr_core_news_md

# Copier tout le code
COPY . .

# Créer les répertoires de données
# (les vrais fichiers arriveront via les volumes Docker)
RUN mkdir -p data/raw/gallica data/raw/gutenberg \
             data/processed/gallica data/processed/gutenberg \
             logs

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

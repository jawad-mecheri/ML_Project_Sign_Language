# Utilise une image de base Python 3.11
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Installe les dépendances système nécessaires pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copie les fichiers nécessaires
# COPY api.py .
# COPY requirements.txt .
# Créer le dossier scripts
RUN mkdir -p scripts

COPY serving/requirements.txt ./
COPY scripts/preprocess_data.py ./scripts/
COPY serving/api.py ./

# Installe les dépendances
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


# Commande pour démarrer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
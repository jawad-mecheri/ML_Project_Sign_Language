# Base image avec Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY retrain_model.py ./  

# Définir la commande d'exécution
CMD ["python", "retrain_model.py"]

# Utiliser une image Python de base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans l'image Docker
COPY project.py ./
COPY requirements.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour le serveur Evidently
EXPOSE 8082

# Commande pour exécuter le projet et lancer l'interface web
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082

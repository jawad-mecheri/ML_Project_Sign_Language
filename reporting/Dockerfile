# Utiliser une image Python officielle (exemple: Python 3.9 slim)
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .


# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'ensemble des fichiers de l'application dans le répertoire de travail
COPY project.py .

# Exposer le port pour l'interface web d'Evidently
EXPOSE 8082

# Commande pour exécuter le projet et lancer l'interface web
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082

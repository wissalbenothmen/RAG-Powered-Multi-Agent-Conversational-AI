# Utiliser une image Python officielle légère avec Python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement requirements.txt en premier pour tirer parti du cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers
COPY . .

# Exposer le port Flask
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
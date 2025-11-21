# ==========================
# Makefile pour pipeline KNN
# ==========================

# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQUIREMENTS := requirements.txt
MODEL_FILE := knn_model.pkl

# ----------------------------
# 1. Installation
# ----------------------------
install: $(VENV)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Environnement virtuel créé"

# ----------------------------
# 2. Préparer les données
# ----------------------------
prepare:
	$(PYTHON) main.py --prepare

# ----------------------------
# 3. Entraîner le modèle KNN
# ----------------------------
train:
	$(PYTHON) main.py --train

# ----------------------------
# 4. Évaluer le modèle
# ----------------------------
evaluate:
	$(PYTHON) main.py --evaluate

# ----------------------------
# 5. Générer des prédictions
# ----------------------------
predict:
	$(PYTHON) main.py --predict

# ----------------------------
# 6. Lancer des tests unitaires
# ----------------------------
test:
	$(PYTHON) test_pipeline.py

# ----------------------------
# 7. Vérification du code (CI)
# ----------------------------
lint:
	flake8 *.py
	black --check *.py
	bandit -r .

format:
	black *.py
	isort *.py

# ----------------------------
# 8. Nettoyage des fichiers générés
# ----------------------------
clean:
	rm -f *.pkl
	rm -f *.csv
	rm -rf __pycache__

# ----------------------------
# 9. Tout exécuter (pipeline complet)
# ----------------------------
all: install prepare train evaluate predict test

# ----------------------------
# 10. Surveiller les fichiers et relancer automatiquement
# ----------------------------
watch:
	@echo "=== Surveiller les fichiers avec watchmedo ==="
	watchmedo shell-command \
        --patterns=".py;.csv" \
        --recursive \
        --command='make all'
# ----------------------------
# 11. Sécurité
# ----------------------------
security:
	bandit -r . -ll


deploy:
	python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

	
.PHONY: install prepare train evaluate predict test lint format clean all watch

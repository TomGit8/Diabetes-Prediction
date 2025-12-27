# 🩺 Diabetes Prediction - MLOps Project

Application web de prédiction du diabète utilisant Machine Learning et déployée sur AWS avec CI/CD automatisé.

## 📋 Description

Ce projet implémente un système complet de prédiction du diabète basé sur le dataset Pima Indians Diabetes. L'application permet aux professionnels de santé d'obtenir une estimation rapide du risque de diabète d'un patient à partir de données médicales simples.

## 🚀 Fonctionnalités

- **Interface web moderne** : Formulaire intuitif avec labels en français et jauge de risque visuelle
- **Prédiction ML en temps réel** : Modèle Random Forest optimisé avec cross-validation
- **Pipeline CI/CD automatisé** : Déploiement continu via GitHub Actions
- **Infrastructure as Code** : Provisionnement AWS avec Terraform
- **Stockage cloud** : Artefacts ML persistés sur Amazon S3

## 🛠️ Technologies

**Backend & ML :**
- Python 3.11
- Flask + Gunicorn
- Scikit-learn, Pandas, NumPy

**DevOps & Cloud :**
- Docker
- Terraform
- GitHub Actions
- AWS (S3, ECR, App Runner)

## 📊 Algorithmes ML Utilisés

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. **Random Forest Classifier** ⭐ (Modèle retenu)
5. Naive Bayes
6. Gradient Boosting

### Méthodes d'évaluation

- Accuracy Score
- ROC AUC Curve
- Cross-Validation (5-fold)
- Confusion Matrix

## 🏗️ Architecture

```
┌─────────────┐
│  Developer  │
└──────┬──────┘
       │ git push
       ▼
┌─────────────────┐
│  GitHub Actions │
│   CI/CD Pipeline│
└────┬───────┬────┘
     │       │
     │       └──────► Amazon ECR (Images Docker)
     │                      │
     ▼                      ▼
Amazon S3          AWS App Runner
(Modèles ML)       (Application Web)
                           │
                           ▼
                    👤 Utilisateurs
```

## 🚀 Déploiement Local (Docker)

### Prérequis
```bash
pip install -r requirements.txt
```

### Lancement

1. **Entraîner le modèle** :
```bash
python model.py
```

2. **Construire l'image Docker** :
```bash
docker build -t diabetes-app .
```

3. **Lancer l'application** :
```bash
docker run -p 8501:8501 diabetes-app
```

4. **Accéder à l'interface** :
Ouvrir http://localhost:8501 dans votre navigateur

## ☁️ Déploiement AWS

### Infrastructure (Terraform)

```bash
cd terraform
terraform init
terraform apply
```

Ressources créées :
- S3 Bucket : `s3-g3mg05`
- ECR Repository : `ecr-g3mg05`
- App Runner Service : `apprunner-g3mg05`
- IAM Role : `AppRunnerECRAccessRole-g3mg05`

### Pipeline CI/CD

Le déploiement est automatique via GitHub Actions :
1. Push sur `main` → Déclenchement du pipeline
2. Tests & entraînement du modèle
3. Upload des artefacts sur S3
4. Build de l'image Docker
5. Push vers Amazon ECR
6. Déploiement automatique sur App Runner

## 🌐 Application en Production

**URL publique** : https://7hsbzsvu65.us-east-1.awsapprunner.com

## 👥 Équipe

**Groupe G3-MG05**
- Tom URBAN
- Ethan SMADJA
- Samuel SIDOUN
- Lucas ARRIESSE

## 📄 License

Projet académique - MLOps 2024

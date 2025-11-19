# Diabetes Prediction Using Machine Learning

## Objective

### Techniques Used

- Data Cleaning
- Data Visualization
- Machine Learning Modeling

### Algortihms Used

1. Logistic Regression
2. Support Vector Machine
3. KNN
4. Random Forest Classifier 
5. Naivye Bayes
6. Gradient Boosting

### Model Evaluation Methods Used

1. Accuracy Score
2. ROC AUC Curve
3. Cross Validation
4. Confusion Matrix

## Guide Lines 

### Packages and Tools Required:
```
Pandas 
Matplotlib
Seaborn
Scikit Learn
Jupyter Notebook
```
### Package Installation
```
pip install numpy
pip install pandas
pip install seaborn
pip install scikit-learn
pip install matplotlib
```
Jupyter Notebook Installation Guide  https://jupyter.org/install

## Déploiement Streamlit (Docker)

1. **Générer les artefacts** : `python model.py` (assure `artifacts/` contient un modèle).
2. **Construire l'image** : `docker build -t diabetes-app .`
3. **Lancer l'app** : `docker run -p 8501:8501 diabetes-app`

L'interface Streamlit est accessible via http://localhost:8501 .

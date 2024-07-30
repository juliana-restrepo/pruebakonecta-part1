# import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('train.csv')

# Define variables
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Crear transformaciones para columnas numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformaciones en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar el preprocesador a los datos de entrenamiento y prueba
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Definir modelos con ajustes en Logistic Regression
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Función para evaluar modelos con manejo de errores
def evaluate_model(model, X_train, X_test, y_train, y_test):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)  # Reducir a 3 para pruebas
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc,
            'CV Score': cv_scores.mean()
        }
    except Exception as e:
        print(f"Error evaluando el modelo {type(model).__name__}: {e}")
        return None

# Evaluar cada modelo
results = {}
for name, model in models.items():
    print(f"Evaluando el modelo: {name}")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Imprimir resultados
for name, metrics in results.items():
    if metrics is not None:
        print(f"\nResultados para {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

# Cargar datos de inferencia
df_inference = pd.read_csv('inference.csv')

# Preprocesar datos de inferencia
X_inference = preprocessor.transform(df_inference)

# Seleccionar el mejor modelo (por ejemplo, Random Forest)
best_model = models['Random Forest']
best_model.fit(X_train, y_train)

# Hacer predicciones
predictions = best_model.predict(X_inference)

# Guardar predicciones en un nuevo archivo CSV
df_inference['Churn'] = predictions
df_inference.to_csv('inference_with_predictions.csv', index=False)




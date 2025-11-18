import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================
# 1. Cargar tu dataset
# ============================
df = pd.read_csv("data/interim/Sociodemograficas_clean.csv", sep=";")

# Definir variables predictoras (solo numéricas) y target
X = df.drop("Consumo", axis=1).select_dtypes(include=["number"])
y = df["Consumo"]

# ============================
# 2. Dividir en train/test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 3. Definir hiperparámetros
# ============================
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",  # en sklearn 1.5+ ya es multinomial por defecto
    "random_state": 8888,
}

# ============================
# 4. Entrenar modelo
# ============================
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# ============================
# 5. Predicciones y métricas
# ============================
# Métrica
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ==========================
# 6. Configurar MLflow
# ==========================
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")  # servidor corriendo en 8080
mlflow.set_experiment("MLflow Quickstart")

# ==========================
# 7. Iniciar Run en MLflow
# ==========================
with mlflow.start_run():
    # Hiperparámetros
    mlflow.log_params(params)

    # Métrica
    mlflow.log_metric("accuracy", accuracy)

    # Firma del modelo
    signature = infer_signature(X_train, lr.predict(X_train))

    # Guardar modelo
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="socio_model",   # ruta en mlflow
        signature=signature,
        input_example=X_train.iloc[:5],   # ejemplo de entrada
        registered_model_name="tracking-quickstart",  # nombre de modelo registrado
    )

    # Tag para recordar
    mlflow.set_tag("Training Info", "Basic LR model for alcohol dataset")

print("✅ Entrenamiento completado. Modelo y métricas guardados en MLflow.")

# ==========================
# 8. Cargar modelo y predecir
# ==========================
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

# Usamos los nombres de columnas de tu dataset
result = pd.DataFrame(X_test, columns=X_test.columns)
result["actual_class"] = y_test.values
result["predicted_class"] = predictions

print("📊 Resultados de predicción (primeras filas):")
print(result.head())
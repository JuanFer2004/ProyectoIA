import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

st.set_page_config(page_title="Predicción Académica - Random Forest", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_or_load_model(df):
    X = df.drop(columns=["Target"])
    y = df["Target"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True),
        "matrix": confusion_matrix(y_test, y_pred),
        "features": X.columns.tolist(),
        "importances": model.feature_importances_,
    }
    return model, encoder, metrics


df = load_data()
model, encoder, metrics = train_or_load_model(df)

st.title("Sistema de predicción académica con Random Forest")
st.markdown(
    "Esta aplicación utiliza un modelo de **Random Forest** para predecir si un estudiante puede quedar en estado "
    "**Dropout**, **Enrolled** o **Graduate**, a partir de variables académicas, económicas y personales del dataset."
)

tab1, tab2, tab3 = st.tabs(["Exploración del dataset", "Modelo", "Predicción individual"])

with tab1:
    st.subheader("Vista general de los datos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Variable objetivo", "Target")

    st.write("Primeros registros:")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Distribución de la variable objetivo")
    st.bar_chart(df["Target"].value_counts())

    st.subheader("Valores faltantes")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Variable", "Valores faltantes"]
    st.dataframe(missing, use_container_width=True)

with tab2:
    st.subheader("Rendimiento del modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("F1 macro", f"{metrics['f1_macro']:.2%}")
    col3.metric("Recall macro", f"{metrics['recall_macro']:.2%}")

    st.subheader("Reporte de clasificación")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Matriz de confusión")
    matrix_df = pd.DataFrame(metrics["matrix"], index=encoder.classes_, columns=encoder.classes_)
    st.dataframe(matrix_df, use_container_width=True)

    st.subheader("Variables más importantes")
    importance_df = pd.DataFrame({
        "Variable": metrics["features"],
        "Importancia": metrics["importances"]
    }).sort_values("Importancia", ascending=False).head(15)
    st.bar_chart(importance_df.set_index("Variable"))

with tab3:
    st.subheader("Ingrese los datos del estudiante")
    st.info("Los campos aparecen con valores promedio o frecuentes del dataset para facilitar una prueba rápida.")

    X = df.drop(columns=["Target"])
    user_values = {}

    cols = st.columns(3)
    for idx, column in enumerate(X.columns):
        series = X[column]
        default_value = float(series.median())
        min_value = float(series.min())
        max_value = float(series.max())

        with cols[idx % 3]:
            if pd.api.types.is_integer_dtype(series):
                user_values[column] = st.number_input(
                    column,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=1
                )
            else:
                user_values[column] = st.number_input(
                    column,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=0.01
                )

    if st.button("Realizar predicción"):
        input_df = pd.DataFrame([user_values])
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = encoder.inverse_transform([prediction_encoded])[0]
        probabilities = model.predict_proba(input_df)[0]

        st.success(f"Resultado predicho: {prediction_label}")

        proba_df = pd.DataFrame({
            "Clase": encoder.classes_,
            "Probabilidad": probabilities
        }).sort_values("Probabilidad", ascending=False)
        st.dataframe(proba_df, use_container_width=True)
        st.bar_chart(proba_df.set_index("Clase"))

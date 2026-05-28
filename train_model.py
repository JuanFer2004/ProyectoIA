import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.txt"


def main():
    df = pd.read_csv(DATA_PATH)

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

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    matrix = confusion_matrix(y_test, y_pred)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("MODELO RANDOM FOREST - PREDICCIÓN DE RESULTADO ACADÉMICO\n")
        f.write("===========================================================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 macro: {f1_macro:.4f}\n")
        f.write(f"Recall macro: {recall_macro:.4f}\n\n")
        f.write("Reporte de clasificación:\n")
        f.write(report)
        f.write("\nMatriz de confusión:\n")
        f.write(str(matrix))

    print("Modelo entrenado correctamente.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

MODEL_PATH = "model.pkl"

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

def train_and_save_model(df, iris, random_state=42, n_estimators=100):
    X = df[[c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]]
    y = df["species"].cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=iris.target_names
        ),
    }

    joblib.dump({"model": clf, "iris": iris}, MODEL_PATH)
    return clf, metrics

def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["iris"]
    return None, None

def main():
    st.set_page_config(page_title="Iris Species Classifier", layout="wide")
    st.title(" Iris Species Classification — Proyecto Final")

    df, iris = load_data()

    left, right = st.columns([1, 2])

    with left:
        st.header("Entrenamiento & Predicción")

        st.subheader("Entrenar modelo")
        if st.button("Entrenar modelo (Random Forest)"):
            with st.spinner("Entrenando modelo..."):
                clf, metrics = train_and_save_model(df, iris)
            st.success("Modelo entrenado y guardado en model.pkl")
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.write("Precision:", f"{metrics['precision_macro']:.4f}")
            st.write("Recall:", f"{metrics['recall_macro']:.4f}")
            st.write("F1 Score:", f"{metrics['f1_macro']:.4f}")
            st.write("Matriz de confusión:")
            st.write(metrics["confusion_matrix"])
            st.write("Reporte de clasificación:")
            st.text(metrics["classification_report"])

        st.subheader("Ingresar medidas de la flor")

        sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
        petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
        petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

        if st.button("Predecir especie"):
            model, iris_loaded = load_model()
            if model is None:
                st.warning("Primero entrena el modelo.")
            else:
                X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                pred_code = model.predict(X_new)[0]
                pred_proba = model.predict_proba(X_new)[0]
                species = iris_loaded.target_names[pred_code]

                st.success(f"Especie predicha: {species}")
                st.write(pd.DataFrame({"species": iris_loaded.target_names, "prob": pred_proba}))

    with right:
        st.header("Visualizaciones")
        fig3d = px.scatter_3d(
            df,
            x="petal_length",
            y="petal_width",
            z="sepal_length",
            color="species",
            title="Dataset Iris (3D)"
        )
        st.plotly_chart(fig3d)

        fig_hist = px.histogram(df, x="sepal_length", color="species")
        st.plotly_chart(fig_hist)

        fig_matrix = px.scatter_matrix(df, dimensions=df.columns[:-1], color="species")
        fig_matrix.update_traces(diagonal_visible=False)
        st.plotly_chart(fig_matrix)

if __name__ == "__main__":
    main()

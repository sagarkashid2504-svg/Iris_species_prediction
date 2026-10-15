import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #171b26 100%);
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1e2533;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #343b4b;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: bold;
}

/* Metric text */
[data-testid="stMetricDelta"] {
    color: #4ade80 !important;
}

/* Input section */
[data-testid="stForm"] {
    background-color: #151a24;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #343b4b;
}

/* General text */
.stMarkdown, label {
    color: #f1f5f9;
}

</style>
""", unsafe_allow_html=True)

st.title("🌸 Iris Species Predictor")
st.caption("Machine Learning classification using flower measurements")
st.write("Enter the sepal and petal measurements to predict the most likely Iris species.")

@st.cache_data
def load_dataset():
    folder = Path(__file__).resolve().parent
    preferred = [
        folder / "Iris.csv",
        folder / "iris.csv",
        folder / "1b4a3f35-5e5e-463a-90ab-ec5362f5e039.csv",
    ]
    csv_path = next((p for p in preferred if p.exists()), None)
    if csv_path is None:
        candidates = [p for p in folder.glob("*.csv") if p.name.lower() not in {"sample.csv"}]
        if not candidates:
            raise FileNotFoundError(
                "Dataset CSV not found. Put your dataset in this folder and name it Iris.csv."
            )
        csv_path = candidates[0]

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    # Support common variations in the Iris dataset's column names.
    required = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
    if not all(c in data.columns for c in required):
        raise ValueError(
            "The CSV does not look like the expected Iris dataset. Required columns: "
            + ", ".join(required)
        )
    data = data[required].copy()
    for col in required[:4]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Species"] = data["Species"].astype(str).str.strip()
    data = data.dropna()
    data = data[data["Species"].ne("")]
    return data, csv_path.name

try:
    data, dataset_name = load_dataset()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
display_names = {
    "SepalLengthCm": "Sepal length (cm)",
    "SepalWidthCm": "Sepal width (cm)",
    "PetalLengthCm": "Petal length (cm)",
    "PetalWidthCm": "Petal width (cm)",
}

X = data[feature_names]
y = data["Species"]

# Use a fixed split so the displayed evaluation metrics remain reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Dataset records", f"{len(data)}")
m2.metric("Iris species", f"{y.nunique()}")
m3.metric("Test accuracy", f"{accuracy:.1%}")

st.divider()
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("🔎 Enter flower measurements")
    with st.form("prediction_form"):
        values = {}
        for feature in feature_names:
            low = float(data[feature].min())
            high = float(data[feature].max())
            default = float(data[feature].median())
            values[feature] = st.number_input(
                display_names[feature],
                min_value=max(0.0, low - 0.5),
                max_value=high + 0.5,
                value=default,
                step=0.1,
                format="%.1f",
            )
        submitted = st.form_submit_button("Predict species", use_container_width=True)

with right:
    st.subheader("🌱 Prediction result")
    if submitted:
        input_df = pd.DataFrame([values], columns=feature_names)
        predicted_species = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_probabilities = pd.DataFrame({
            "Species": model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.success(f"Predicted species: **{predicted_species}**")
        st.write("Prediction confidence by species:")
        st.bar_chart(class_probabilities.set_index("Species"))
        st.caption("Probabilities are estimates from the trained Random Forest model.")
    else:
        st.info("Adjust the measurements, then click **Predict species**.")
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/7/78/Iris_setosa.jpg",
            caption="Example Iris flower",
            use_container_width=True,
        )

with st.expander("📊 Model evaluation"):
    st.write("Model: **Random Forest Classifier**")
    st.write("Train/test split: **80% training / 20% testing**")
    st.text("Classification report")
    report = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.write("Confusion matrix (actual rows × predicted columns)")
    cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
    st.dataframe(
        pd.DataFrame(cm, index=model.classes_, columns=model.classes_),
        use_container_width=True
    )

with st.expander("📁 About this dataset"):
    st.write(f"Dataset file: `{dataset_name}`")
    st.write(f"Clean records used: **{len(data)}**")
    st.write("Input features: sepal length, sepal width, petal length, and petal width (all in cm).")
    st.write("Target: Iris species.")
    st.dataframe(data.head(10), use_container_width=True)

st.caption("Educational project • Predictions depend on the quality and coverage of the dataset.")

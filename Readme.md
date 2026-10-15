# 🌸 Iris Species Prediction — Machine Learning

A simple, interactive **Iris Species Prediction** web app built with Python, Scikit-learn, and Streamlit. It predicts an Iris flower's species from four measurements.

## ✨ Features

- Predicts the Iris species from user-entered flower measurements.
- Uses a **Random Forest Classifier**.
- Displays test accuracy and a classification report.
- Shows class probabilities and a confusion matrix.
- Trains the model automatically when the app starts — **no separate `.pkl` file is required**.

## 📊 Dataset

The dataset contains 150 flower records and these columns:

| Column | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimetres |
| `SepalWidthCm` | Sepal width in centimetres |
| `PetalLengthCm` | Petal length in centimetres |
| `PetalWidthCm` | Petal width in centimetres |
| `Species` | Target species label |

The `Id` column, if present, is not used as a model feature. The app expects the standard Iris CSV column names shown above.

## 🧠 Machine Learning Workflow

1. Load the CSV dataset.
2. Clean missing or invalid values.
3. Select the four flower measurements as input features.
4. Split the data into training (80%) and testing (20%) sets.
5. Train a Random Forest classifier.
6. Evaluate performance on the test set.
7. Use the trained model to predict a species from the app inputs.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## 📂 Project Structure

```text
iris-species-prediction/
├── app.py
├── Iris.csv
├── requirements.txt
└── README.md
```

## 🚀 Run the Project Locally

### 1. Install Python

Install Python 3.10 or newer and make sure Python is available in your terminal.

### 2. Place the files in one folder

Save `app.py`, `requirements.txt`, and the dataset CSV in the same folder. Rename the provided dataset to **`Iris.csv`** (recommended). The app also recognizes the original uploaded filename.

### 3. Open a terminal in the project folder

Create and activate a virtual environment (recommended):

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the app

```bash
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

## 📈 Evaluation

The app reports test accuracy, precision, recall, F1-score, and a confusion matrix. The exact score can vary with the dataset and train/test split. Evaluation results are estimates on the held-out test set, not a guarantee of performance on every new flower.

## 🌐 Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `README.md`, and `Iris.csv` to a GitHub repository.
2. Open Streamlit Community Cloud and connect the repository.
3. Select `app.py` as the main file.
4. Deploy the app.

## ⚠️ Notes

- This is an educational machine learning project.
- Use the standard Iris dataset columns; if your CSV has different column names, update the column names in `app.py`.
- The app trains the model at startup, so a saved model file is not needed.

## 👤 Project Summary

**Project name:** Iris Species Prediction  
**Problem type:** Supervised machine learning — classification  
**Algorithm:** Random Forest Classifier  
**Output:** Predicted Iris species

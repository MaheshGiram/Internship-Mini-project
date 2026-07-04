import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("datasets/alzheimers_disease_data.csv")

# Features
X = df[
    [
        "Age",
        "Gender",
        "Smoking",
        "AlcoholConsumption",
        "PhysicalActivity",
        "SleepQuality",
        "FamilyHistoryAlzheimers",
        "Diabetes",
        "Hypertension",
        "SystolicBP",
        "DiastolicBP",
        "MMSE",
        "MemoryComplaints"
    ]
]

# Target
y = df["Diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/alzheimer_model.pkl")

print("Model Saved Successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load fusion dataset
df = pd.read_csv(
    "datasets/fusion_dataset.csv"
)

# Remove unwanted columns
drop_cols = [
    "PatientID",
    "DoctorInCharge",
    "label"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier()

model.fit(
    X_train,
    y_train
)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Fusion Model Accuracy:", accuracy)

# Save model
joblib.dump(
    model,
    "models/fusion_model.pkl"
)

print("Fusion Model Saved Successfully!")
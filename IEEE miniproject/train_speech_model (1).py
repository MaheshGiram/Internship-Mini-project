import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load speech dataset
df = pd.read_csv("speechdataset/speech_data.csv")

# Features
X = df.drop("label", axis=1)

# Target
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Speech Model Accuracy:", accuracy)

# Save model
joblib.dump(
    model,
    "models/speech_model.pkl"
)

print("Speech Model Saved Successfully!")
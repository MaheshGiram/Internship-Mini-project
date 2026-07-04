import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)

# -------------------------
# Load Fusion Dataset
# -------------------------

df = pd.read_csv("datasets/fusion_dataset.csv")

drop_cols = [
    "PatientID",
    "DoctorInCharge",
    "label"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Load Trained Model
# -------------------------

model = joblib.load("models/fusion_model.pkl")

# -------------------------
# Prediction
# -------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------
# Metrics
# -------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

error_rate = (fp + fn) / (tp + tn + fp + fn)
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

# -------------------------
# Print Results
# -------------------------

print("\n========== MODEL PERFORMANCE ==========\n")

print(f"Accuracy               : {accuracy:.4f}")
print(f"Precision              : {precision:.4f}")
print(f"Recall                 : {recall:.4f}")
print(f"F1 Score               : {f1:.4f}")
print(f"ROC-AUC                : {roc_auc:.4f}")

print(f"\nError Rate             : {error_rate:.4f}")
print(f"False Positive Rate    : {false_positive_rate:.4f}")
print(f"False Negative Rate    : {false_negative_rate:.4f}")

print("\nConfusion Matrix\n")
print(cm)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -------------------------
# Save Evaluation Results
# -------------------------

with open("evaluation_results.txt", "w") as f:

    f.write("MODEL PERFORMANCE\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC AUC : {roc_auc:.4f}\n")

    f.write(f"\nError Rate : {error_rate:.4f}\n")
    f.write(f"False Positive Rate : {false_positive_rate:.4f}\n")
    f.write(f"False Negative Rate : {false_negative_rate:.4f}\n")

# -------------------------
# Save Classification Report
# -------------------------

with open("classification_report.txt", "w") as f:

    f.write(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix Image
# -------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Alzheimer"]
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# ROC Curve
# -------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAll evaluation files generated successfully!")
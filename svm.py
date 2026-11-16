import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Ensure dataset is loaded (using df from previous load)
if 'df' in globals():
    print("Using already loaded dataset...")
else:
    print("Please upload the CSV file first.")
    uploaded = files.upload()
    csv_files = [f for f in uploaded if f.lower().endswith('.csv')]
    df = pd.read_csv(csv_files[0])

# Preprocess
diagnosis = df["diagnosis"].astype(str).str.strip().str.upper()
y = (diagnosis == "M").astype(int)

columns_to_drop = ["diagnosis"]
if "id" in df.columns:
    columns_to_drop.append("id")

X = df.drop(columns=columns_to_drop).copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Define and train ONLY SVM pipeline
# We enable probability=True to allow calculating ROC-AUC
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
])

print("Training SVM model...")
svm_pipeline.fit(X_train, y_train)

# Predictions
y_pred = svm_pipeline.predict(X_test)
y_proba = svm_pipeline.predict_proba(X_test)[:, 1]

# Metrics calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
sensitivity = recall_score(y_test, y_pred, zero_division=0) # Recall
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) else 0
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

# Print metrics summary
print("\n========== SVM PERFORMANCE METRICS ==========")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"ROC-AUC:     {auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Benign", "Malignant"],
    zero_division=0
))

# Plot and Save Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=["Benign", "Malignant"]
)
cm_display.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

# Export metrics to CSV
svm_metrics_df = pd.DataFrame([{
    "Algorithm": "Support Vector Machine (SVM)",
    "Accuracy": accuracy,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "F1-Score": f1,
    "ROC-AUC": auc
}])
svm_metrics_df.to_csv("svm_model_metrics.csv", index=False)
print("\nResults saved. Preparing to download files...")

# Download files
files.download("svm_model_metrics.csv")
files.download("svm_confusion_matrix.png")
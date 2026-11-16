# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# STEP 2: UPLOAD CSV FILE
uploaded = files.upload()

csv_files = [
    filename for filename in uploaded
    if filename.lower().endswith(".csv")
]

if not csv_files:
    raise ValueError("Please upload a CSV file.")

df = pd.read_csv(csv_files[0])

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
display(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

# STEP 3: PREPROCESS THE DATA

# Check target column
if "diagnosis" not in df.columns:
    raise ValueError(
        "Target column 'diagnosis' not found. "
        "Please check the CSV column names."
    )

# Convert diagnosis into binary labels:
# B = 0 (Benign)
# M = 1 (Malignant)

diagnosis = df["diagnosis"].astype(str).str.strip().str.upper()

if not set(diagnosis.unique()).issubset({"B", "M"}):
    raise ValueError(
        f"Unexpected diagnosis labels: {diagnosis.unique()}"
    )

y = (diagnosis == "M").astype(int)

# Remove target and ID columns from input features
columns_to_drop = ["diagnosis"]

if "id" in df.columns:
    columns_to_drop.append("id")

X = df.drop(columns=columns_to_drop).copy()

# Convert predictors to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle missing or non-numeric values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

if X.isnull().any().any():
    raise ValueError(
        "Some feature columns could not be converted to numeric values."
    )

print("\nFeature matrix shape:", X.shape)
print("\nClass distribution:")
print(y.map({0: "Benign", 1: "Malignant"}).value_counts())

# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: DEFINE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ]),

    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# STEP 6: TRAIN MODELS AND CALCULATE PERFORMANCE METRICS

results = []
confusion_matrices = {}
predictions = {}
trained_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    # Train model
    model.fit(X_train, y_train)

    # Predict test labels
    y_pred = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = y_pred

    # Get probability or decision scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) else 0

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    sensitivity = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_test, y_score)

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    confusion_matrices[name] = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by accuracy
results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n========== PERFORMANCE METRICS ==========\n")

display(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Sensitivity": "{:.4f}",
        "Specificity": "{:.4f}",
        "F1-Score": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    })
)

# STEP 7: PLOT CONFUSION MATRICES FOR ALL MODELS

fig, axes = plt.subplots(2, 4, figsize=(18, 10))

for ax, (name, cm) in zip(
    axes.flatten(),
    confusion_matrices.items()
):

    display_matrix = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    display_matrix.plot(
        ax=ax,
        colorbar=False,
        values_format="d"
    )

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

plt.suptitle(
    "Confusion Matrices for Breast Cancer Classification",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# STEP 8: PERFORMANCE COMPARISON GRAPH

metric_columns = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1-Score",
    "ROC-AUC"
]

plot_df = results_df.set_index("Algorithm")[metric_columns]

ax = plot_df.plot(
    kind="bar",
    figsize=(16, 7),
    ylim=(0, 1.05)
)

ax.set_title(
    "Performance Comparison of Machine Learning Algorithms"
)

ax.set_xlabel("Machine Learning Algorithm")
ax.set_ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# STEP 9: PRINT CLASSIFICATION REPORT FOR EACH MODEL

for name, y_pred in predictions.items():

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT:", name)
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

# STEP 10: DISPLAY BEST MODELS BY METRIC

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]

best_sensitivity = results_df.loc[
    results_df["Sensitivity"].idxmax()
]

best_auc = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]

print("\n========== BEST MODEL SUMMARY ==========")

print(
    f"Best Accuracy: {best_accuracy['Algorithm']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Best Sensitivity: {best_sensitivity['Algorithm']} "
    f"({best_sensitivity['Sensitivity']:.4f})"
)

print(
    f"Best ROC-AUC: {best_auc['Algorithm']} "
    f"({best_auc['ROC-AUC']:.4f})"
)

# STEP 11: SAVE METRICS AND GRAPHS

results_df.to_csv(
    "breast_cancer_model_metrics.csv",
    index=False
)

# Save confusion matrix figure
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

for ax, (name, cm) in zip(
    axes.flatten(),
    confusion_matrices.items()
):
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    ).plot(
        ax=ax,
        colorbar=False,
        values_format="d"
    )

    ax.set_title(name, fontsize=10)

plt.tight_layout()
fig.savefig(
    "breast_cancer_confusion_matrices.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# Save performance comparison chart
ax = plot_df.plot(
    kind="bar",
    figsize=(16, 7),
    ylim=(0, 1.05)
)

ax.set_title("Performance Comparison of ML Models")
ax.set_ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    "breast_cancer_performance_comparison.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()

print("\nAll outputs saved successfully.")

# STEP 12: DOWNLOAD OUTPUT FILES

files.download("breast_cancer_model_metrics.csv")
files.download("breast_cancer_confusion_matrices.png")
files.download("breast_cancer_performance_comparison.png")
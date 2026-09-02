import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.shape)
print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.shape)

print(df.describe())

print(df["Class"].value_counts())

print(
    df["Class"]
    .value_counts(normalize=True) * 100
)

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Credit Card Fraud Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")

plt.show()

class_counts = df["Class"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    class_counts,
    labels=["Legitimate", "Fraud"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Fraud vs Legitimate Transactions")

plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Amount"],
    bins=50,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Amount")

plt.show()

plt.figure(figsize=(16, 12))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.show()

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]

scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(
    X[["Amount"]]
)

X["Time"] = scaler.fit_transform(
    X[["Time"]]
)

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print(y_train_smote.value_counts())

plt.figure(figsize=(8, 6))

sns.countplot(
    x=y_train_smote
)

plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.show()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("Model trained successfully!")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(9, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid(True)

plt.show()

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)

plt.figure(figsize=(9, 6))

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.grid(True)

plt.show()

thresholds_to_test = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

threshold_results = []

for threshold in thresholds_to_test:

    y_threshold = (
        y_probability >= threshold
    ).astype(int)

    precision_value = precision_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    recall_value = recall_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    f1_value = f1_score(
        y_test,
        y_threshold,
        zero_division=0
    )

    threshold_results.append(
        {
            "Threshold": threshold,
            "Precision": precision_value,
            "Recall": recall_value,
            "F1 Score": f1_value
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

print(threshold_df)

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 vs Decision Threshold")

plt.legend()
plt.grid(True)

plt.show()

best_row = threshold_df.loc[
    threshold_df["F1 Score"].idxmax()
]

print("Best threshold:")
print(best_row)

best_threshold = best_row["Threshold"]

y_best = (
    y_probability >= best_threshold
).astype(int)

final_precision = precision_score(
    y_test,
    y_best,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_best,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_best,
    zero_division=0
)

print("\nFinal Results")

print("Threshold:", best_threshold)
print("Precision:", final_precision)
print("Recall:", final_recall)
print("F1 Score:", final_f1)
print("ROC-AUC:", roc_auc)

final_cm = confusion_matrix(
    y_test,
    y_best
)

print(final_cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    f"Confusion Matrix - Threshold {best_threshold}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.show()

results = X_test.copy()

results["Actual_Class"] = y_test.values
results["Fraud_Probability"] = y_probability
results["Predicted_Class"] = y_best

results.to_csv(
    "fraud_predictions.csv",
    index=False
)

threshold_df.to_csv(
    "threshold_analysis.csv",
    index=False
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

with open(
    "fraud_detection_report.txt",
    "w"
) as file:

    file.write(
        "CREDIT CARD FRAUD DETECTION REPORT\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write("MODEL: RANDOM FOREST\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

    file.write(
        f"Best Decision Threshold: "
        f"{best_threshold}\n\n"
    )

    file.write(
        f"Final Precision: "
        f"{final_precision:.4f}\n"
    )

    file.write(
        f"Final Recall: "
        f"{final_recall:.4f}\n"
    )

    file.write(
        f"Final F1 Score: "
        f"{final_f1:.4f}\n\n"
    )

    file.write("Threshold Analysis\n")
    file.write("-" * 60 + "\n")
    file.write(
        threshold_df.to_string(index=False)
    )

print("Project 2 completed successfully!")

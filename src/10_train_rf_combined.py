import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Paths
input_csv = Path("/home/lakshminarayanan_evolution/lc25000/data/combined_classic_cnn_features.csv")
model_output = Path("/home/lakshminarayanan_evolution/lc25000/models/rf_model_combined.joblib")
conf_matrix_fig = Path("/home/lakshminarayanan_evolution/lc25000/figures/confusion_matrix_rf_combined.png")

# Load data
df = pd.read_csv(input_csv)
X = df.drop(columns=["filename", "class", "tissue"])
y = df["class"]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, model_output)

# Predict and evaluate
y_pred = rf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix - Random Forest (Combined Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(conf_matrix_fig, dpi=300)


print(f"Model saved to: {model_output}")
print(f"Confusion matrix figure saved to: {conf_matrix_fig}")


from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
input_csv = Path("/home/lakshminarayanan_evolution/lc25000/data/features_classic.csv")
model_out = Path("/home/lakshminarayanan_evolution/lc25000/models/rf_model_classic.joblib")
fig_out = Path("/home/lakshminarayanan_evolution/lc25000/figures/confusion_matrix_rf_classic.png")

# Load data
df = pd.read_csv(input_csv)
X = df.drop(columns=["filename", "class", "tissue"])
y = df["class"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, model_out)

# Evaluate model
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix - Random Forest (Classic Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
fig_out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_out, dpi=300)
plt.close()


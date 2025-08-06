# 09_tsne_combined_features.py

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
input_csv = Path("/home/lakshminarayanan_evolution/lc25000/data/combined_classic_cnn_features.csv")
output_png = Path("/home/lakshminarayanan_evolution/lc25000/figures/tsne_combined_features.png")

# Load combined features
df = pd.read_csv(input_csv)

# Drop non-feature columns
X = df.drop(columns=["filename", "class", "tissue"])
y = df["class"]

# Normalize features
X_scaled = StandardScaler().fit_transform(X)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Prepare DataFrame for plotting
df_vis = pd.DataFrame({
    "tsne_1": X_tsne[:, 0],
    "tsne_2": X_tsne[:, 1],
    "label": y
})

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_vis,
    x="tsne_1",
    y="tsne_2",
    hue="label",
    palette="tab10",
    alpha=0.6
)
plt.title("t-SNE - Combined Feature Space (2D)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_png, dpi=300)

print(f"Saved t-SNE plot to {output_png}")


import os
import matplotlib.pyplot as plt
import cv2

# Base directory
BASE_DIR = "/home/lakshminarayanan_evolution/lc25000/data/images/lung_colon_image_set/"
OUT_PATH = "/home/lakshminarayanan_evolution/lc25000/figures/fig_raw_tissue_examples.png"

# Prepare full list of class directories
class_dirs = []
for group in ["colon_image_sets", "lung_image_sets"]:
    group_dir = os.path.join(BASE_DIR, group)
    for cls in sorted(os.listdir(group_dir)):
        class_path = os.path.join(group_dir, cls)
        if os.path.isdir(class_path):
            class_dirs.append(class_path)

# Plot one image from each class
fig, axes = plt.subplots(1, len(class_dirs), figsize=(18, 4))
for ax, class_path in zip(axes, class_dirs):
    class_name = os.path.basename(class_path)
    image_files = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if image_files:
        img_path = os.path.join(class_path, image_files[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(class_name)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATA_DIR = "/home/lakshminarayanan_evolution/lc25000/data/images/lung_colon_image_set/"
CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
OUTPUT_PATH = "/home/lakshminarayanan_evolution/lc25000/figures/fig_preprocessing_demo.png"

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def find_first_image_by_class(root_dir, class_names):
    found = {cls: None for cls in class_names}
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                for cls in class_names:
                    if cls in subdir and found[cls] is None:
                        full_path = os.path.join(subdir, file)
                        img = cv2.imread(full_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            found[cls] = (cls, img)
    return [found[cls] for cls in class_names if found[cls] is not None]

# Load sample images
samples = find_first_image_by_class(DATA_DIR, CLASS_NAMES)

# Plot and save
fig, axes = plt.subplots(3, len(samples), figsize=(15, 9))

for i, (label, img) in enumerate(samples):
    axes[0, i].imshow(img)
    axes[0, i].set_title(label)
    axes[0, i].axis("off")

    enhanced = enhance_contrast(img)
    axes[1, i].imshow(enhanced)
    axes[1, i].axis("off")

    denoised = reduce_noise(enhanced)
    axes[2, i].imshow(denoised)
    axes[2, i].axis("off")

fig.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)


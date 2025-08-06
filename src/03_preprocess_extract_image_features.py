import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from tqdm import tqdm

# Configuration
DATA_DIR = '/home/lakshminarayanan_evolution/lc25000/data/images/lung_colon_image_set/'
OUTPUT_CSV = '/home/lakshminarayanan_evolution/lc25000/data/features_classic.csv'
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS

# Classic feature extraction functions
def extract_glcm_features(gray):
    glcm = graycomatrix((gray * 255).astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    return contrast, correlation, homogeneity, energy

def extract_lbp_features(gray):
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3), range=(0, LBP_N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def extract_intensity_features(gray):
    mean = np.mean(gray)
    std = np.std(gray)
    entropy = shannon_entropy(gray)
    return mean, std, entropy

def preprocess_image(image):
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised

def extract_features(image):
    gray = rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    features = {}

    contrast, corr, homog, energy = extract_glcm_features(gray)
    features['glcm_contrast'] = contrast
    features['glcm_correlation'] = corr
    features['glcm_homogeneity'] = homog
    features['glcm_energy'] = energy

    lbp_hist = extract_lbp_features(gray)
    for i, val in enumerate(lbp_hist):
        features[f'lbp_{i}'] = val

    mean, std, entropy = extract_intensity_features(gray)
    features['gray_mean'] = mean
    features['gray_std'] = std
    features['gray_entropy'] = entropy

    return features

# Main pipeline
all_features = []

print("Extracting features from images...")
for tissue in sorted(os.listdir(DATA_DIR)):
    tissue_dir = os.path.join(DATA_DIR, tissue)
    if not os.path.isdir(tissue_dir):
        continue
    for cls in sorted(os.listdir(tissue_dir)):
        cls_dir = os.path.join(tissue_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in tqdm(os.listdir(cls_dir), desc=f"{tissue}/{cls}"):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(cls_dir, fname)
            image = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
            processed = preprocess_image(image)
            features = extract_features(processed)
            features['filename'] = fname
            features['tissue'] = tissue
            features['class'] = cls
            all_features.append(features)

# Save to CSV
print(f"Saving extracted features to: {OUTPUT_CSV}")
df = pd.DataFrame(all_features)
df.to_csv(OUTPUT_CSV, index=False)
print("Done.")


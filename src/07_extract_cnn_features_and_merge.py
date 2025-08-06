import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

# Paths
image_dir = "/home/lakshminarayanan_evolution/lc25000/data/images/lung_colon_image_set/"
cnn_features_path="/home/lakshminarayanan_evolution/lc25000/data/features_cnn.csv"
classic_features_path = "/home/lakshminarayanan_evolution/lc25000/data/features_classic.csv"
output_path = "/home/lakshminarayanan_evolution/lc25000/data/combined_classic_cnn_features.csv"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset using ImageFolder (flat structure with 5 folders)
dataset = ImageFolder(image_dir, transform=transform)

# Map index to class name (e.g., 0 -> colon_aca)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18 and remove classifier head
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# Extract CNN features
cnn_features = []
filenames = []

with torch.no_grad():
    for inputs, labels in tqdm(loader, desc="Extracting CNN features"):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        outputs = outputs.cpu().numpy()
        cnn_features.extend(outputs)

        batch_start_idx = len(filenames)
        batch_filenames = [dataset.samples[i + batch_start_idx][0] for i in range(len(inputs))]
        batch_filenames = [os.path.basename(f) for f in batch_filenames]
        filenames.extend(batch_filenames)

# Convert to DataFrame
cnn_df = pd.DataFrame(cnn_features)
cnn_df.columns = [f"cnn_feat_{i}" for i in range(cnn_df.shape[1])]
cnn_df["filename"] = filenames


# Load classic features
classic_df = pd.read_csv(classic_features_path)

# Merge CNN features with class + tissue using filename
cnn_df = pd.merge(cnn_df, classic_df[["filename", "class", "tissue"]], on="filename", how="left")

cnn_df.to_csv(cnn_features_path, index=False)


# Merge on filename
combined_df = pd.merge(classic_df, cnn_df.drop(columns=["class", "tissue"]), on="filename", how="inner")

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined_df.to_csv(output_path, index=False)

print("Saved combined features to:", output_path)


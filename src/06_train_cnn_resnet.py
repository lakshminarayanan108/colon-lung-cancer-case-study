# for ease of use, image containing folders are placed into the same directory (without additional subdirectories) 
# ( 5 folders, 1 for each class ) 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ----- Configurations -----
data_dir = "/home/lakshminarayanan_evolution/lc25000/data/images/lung_colon_image_set"
model_path = "/home/lakshminarayanan_evolution/lc25000/models/cnn_model.h5"
fig_path = "/home/lakshminarayanan_evolution/lc25000/figures/confusion_matrix_cnn.png"
batch_size = 32
epochs = 5  # Increase if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----- Data Preparation -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----- Load Pretrained Model -----
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----- Training Loop -----
print("\nTraining CNN (ResNet18)...")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

# ----- Evaluation -----
print("\nEvaluating on validation set...")
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ----- Confusion Matrix -----
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix - CNN (ResNet18)")
plt.tight_layout()
plt.savefig(fig_path)
plt.close()

# ----- Save Model -----
torch.save(model.state_dict(), model_path)
print(f"\nCNN model saved to {model_path}")
print(f"Confusion matrix saved to {fig_path}")


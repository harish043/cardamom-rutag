import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------
# Config
# --------------------------
VAL_DIR = os.path.join("dataset", "val")
MODEL_PATH = "models/efficientnet_cardamom_best.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# Load Validation Data
# --------------------------
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes

# --------------------------
# Load Trained Model
# --------------------------
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --------------------------
# Inference
# --------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------
# Classification Report
# --------------------------
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
# --------------------------
# Per-Class Accuracy Plot
# --------------------------
cm_diag = np.diag(cm)
class_counts = np.sum(cm, axis=1)
class_accuracy = cm_diag / class_counts

plt.figure(figsize=(8, 5))
sns.barplot(x=class_names, y=class_accuracy)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Per-Class Validation Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("val_acc.png")

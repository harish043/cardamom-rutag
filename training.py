import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --------------------------
# Config
# --------------------------
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
SEED = 42
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Fix randomness for reproducibility
# --------------------------
torch.manual_seed(SEED)
random.seed(SEED)

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
# Load Datasets
# --------------------------
print("[INFO] Loading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print(f"[INFO] Classes: {train_dataset.classes}")

# --------------------------
# Model Setup (with dropout)
# --------------------------
model = models.efficientnet_b0(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = True  # Fine-tune entire model

num_classes = len(train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(DEVICE)

# --------------------------
# Loss, Optimizer, Scheduler
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

# --------------------------
# Training & Validation Loop with Early Stopping
# --------------------------
print(f"\n[INFO] Starting training on {DEVICE}...\n")
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / len(train_dataset)
    

    # ---- Validation ----
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    # ---- Early Stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "efficientnet_cardamom_best.pth")
        print("[INFO] Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

print("\n[INFO] Training complete.")

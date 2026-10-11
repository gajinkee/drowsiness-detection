import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from datetime import datetime

os.makedirs("models", exist_ok=True)
DATA_DIR = "data_faces"
MODEL_PATH = f"models/drowsiness_cnn_{datetime.now():%Y%m%d_%H%M%S}.pth"
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = weights.transforms()

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

if os.path.isdir(train_dir) and os.path.isdir(val_dir):
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Training and validation classes must match.")
else:
    base_ds = datasets.ImageFolder(DATA_DIR)

    if len(base_ds.classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, found: {base_ds.classes}"
        )

    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_base = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    n_train = int(0.8 * len(base_ds))
    n_val = len(base_ds) - n_train

    if n_train == 0 or n_val == 0:
        raise ValueError("Not enough images for a train/validation split.")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        range(len(base_ds)), [n_train, n_val], generator=generator
    )

    train_ds = Subset(train_base, train_subset.indices)
    val_ds = Subset(val_base, val_subset.indices)

    train_ds.classes = base_ds.classes
    train_ds.class_to_idx = base_ds.class_to_idx
    val_ds.classes = base_ds.classes
    val_ds.class_to_idx = base_ds.class_to_idx

if set(train_ds.classes) != {"active", "fatigue"}:
    raise ValueError(
        f"Expected folders named active and fatigue; found {train_ds.classes}"
    )

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    pin_memory=(device.type == "cuda")
)

model = mobilenet_v3_small(weights=weights)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, 2
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_acc = 0.0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss / train_total:.4f} | "
        f"Train acc: {train_acc:.3f} | "
        f"Val loss: {val_loss / val_total:.4f} | "
        f"Val acc: {val_acc:.3f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

torch.save({
    "model_state_dict": best_state,
    "classes": train_ds.classes,
    "class_to_idx": train_ds.class_to_idx
}, MODEL_PATH)

print(f"Best validation accuracy: {best_acc:.3f}")
print(f"Saved model to {MODEL_PATH}")
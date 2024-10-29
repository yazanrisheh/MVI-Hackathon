import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import os
import cv2
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set labels and image size
labels = ['Negative', 'Positive']
img_size = 120

# Seeding function
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# Define test_transform globally for import
test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset class without transforms (we'll apply transforms later)
class StructuralCrackDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.read_image_paths()

    def read_image_paths(self):
        for label in labels:
            path = os.path.join(self.data_dir, label)
            class_num = labels.index(label)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                self.data.append((img_path, class_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Dataset class with transforms
class StructuralCrackDatasetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.base_dataset[self.indices[idx]]
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        image = np.array(resized_arr).reshape(img_size, img_size, 1)
        image = image / 255.0  # Normalize
        image = image.astype(np.float32)
        if self.transform:
            image = Image.fromarray(np.uint8(image * 255).squeeze())
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.tensor(label).long()
        return image, label

# Evaluation function
def evaluate_model(model, data_loader, dataset_name):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Calculating metrics
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # Calculate Intersection over Union (IoU)
    intersection = np.logical_and(np.array(all_labels), np.array(all_predictions))
    union = np.logical_or(np.array(all_labels), np.array(all_predictions))
    iou = np.sum(intersection) / np.sum(union)

    # Print metrics
    print(f"**{dataset_name} Set Metrics**:")
    print("----------------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    print("----------------------------------------------------")
    print()

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{dataset_name} Set Confusion Matrix')
    plt.show()

# Add the training block inside the main function check
if __name__ == "__main__":
    # Load dataset and create train/validation/test splits with a 70/20/10 ratio
    dataset = StructuralCrackDataset(r'test_250')
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(42)
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create datasets with transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    train_dataset = StructuralCrackDatasetWithTransform(dataset, train_subset.indices, transform=train_transform)
    val_dataset = StructuralCrackDatasetWithTransform(dataset, val_subset.indices, transform=test_transform)
    test_dataset = StructuralCrackDatasetWithTransform(dataset, test_subset.indices, transform=test_transform)

    # Define DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define CNN Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(2)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 15 * 15, 256)
            self.relu4 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.batchnorm = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 2)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.flatten(x)
            x = self.batchnorm(self.dropout(self.relu4(self.fc1(x))))
            return self.fc2(x)

    # Initialize model, optimizer, loss function, and scheduler
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training Loop
    for epoch in range(15):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/15], Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Evaluate on train, validation, and test sets
    evaluate_model(model, train_loader, 'Train')
    evaluate_model(model, val_loader, 'Validation')
    evaluate_model(model, test_loader, 'Test')

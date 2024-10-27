import wandb 
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
import shutil
import random
import warnings
from PIL import Image

# Initialize wandb with project name and config parameters
wandb.init(project="structural_crack_detection", config={
    "learning_rate": 1e-3,
    "epochs": 15,
    "batch_size": 128,
    "img_size": 120,
    "step_size": 5,
    "gamma": 0.5
})
config = wandb.config  # Access config for hyperparameters

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set labels and image size
labels = ['Negative', 'Positive']
img_size = config.img_size

# 3. Add seeding
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(42)

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
        # We won't use this directly
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
            # Convert to PIL Image
            image = Image.fromarray(np.uint8(image * 255).squeeze())
            image = self.transform(image)  # Apply transform
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.tensor(label).long()
        return image, label

# 1. Add salt & pepper noise and Gaussian blur
class AddSaltPepperNoise(object):
    def __init__(self, prob=0.05):
        self.prob = prob
    
    def __call__(self, image):
        np_image = np.array(image)
        salt_pepper = np.random.rand(*np_image.shape)
        salt = (salt_pepper < self.prob / 2)
        pepper = (salt_pepper > 1 - self.prob / 2)
        np_image[salt] = 255
        np_image[pepper] = 0
        return Image.fromarray(np.uint8(np_image))

# Training transformations with added noise and blur
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
    transforms.RandomApply([AddSaltPepperNoise(prob=0.05)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor()
])

# Validation and test transformations
test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset and create train/validation/test splits with a 70/20/10 ratio
dataset = StructuralCrackDataset(r'test_250')
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Seeding the data split
generator = torch.Generator()
generator.manual_seed(42)
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Create datasets with transforms
train_dataset = StructuralCrackDatasetWithTransform(dataset, train_subset.indices, transform=train_transform)
val_dataset = StructuralCrackDatasetWithTransform(dataset, val_subset.indices, transform=test_transform)
test_dataset = StructuralCrackDatasetWithTransform(dataset, test_subset.indices, transform=test_transform)

# 2. Check class instances equal number
def get_class_counts(dataset):
    labels_list = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels_list)
    return class_counts

train_class_counts = get_class_counts(train_dataset)
val_class_counts = get_class_counts(val_dataset)
test_class_counts = get_class_counts(test_dataset)

print("Train class counts:", train_class_counts)
print("Validation class counts:", val_class_counts)
print("Test class counts:", test_class_counts)

# WeightedRandomSampler for handling class imbalance in training set
labels_list = [train_dataset[i][1].item() for i in range(len(train_dataset))]
class_counts = np.bincount(labels_list)
class_weights = 1.0 / class_counts
weights = [class_weights[label] for label in labels_list]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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

# 4. Initialize neural network weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# Initialize model, optimizer, loss function, and scheduler
model = CNNModel().to(device)
initialize_weights(model)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# Training Loop with wandb logging
for epoch in range(config.epochs):
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
    wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy})

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
    wandb.log({"Val Loss": val_loss, "Val Accuracy": val_accuracy})
    
    # Adjust learning rate
    scheduler.step()

    print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# Save the model directly using shutil.copy to avoid symlink error
torch.save(model.state_dict(), "model.pth")
shutil.copy("model.pth", os.path.join(wandb.run.dir, "model.pth"))
wandb.finish()

# 2. Check accuracy and confusion matrix of all 3 datasets
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

# Evaluate on train set
evaluate_model(model, train_loader, 'Train')

# Evaluate on validation set
evaluate_model(model, val_loader, 'Validation')

# Evaluate on test set
evaluate_model(model, test_loader, 'Test')
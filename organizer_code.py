import torch
import os
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from main import seed_everything, test_transform

# Define the modified CNNModel to match the original training configuration
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

        # Adjust this to match the original training input size of 224x224
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Adjusted for 224x224 input size
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

# Ensure reproducibility
# seed_everything(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

labels = ['Negative', 'Positive']
img_size = 224  # Set to 224 to match training input size

# Dataset class for loading and preprocessing images
class TestDataset(Dataset):
    def __init__(self, base_dir, max_images=1000):  # Set max_images to 1000
        self.data = []
        self.max_images = max_images
        self.load_images(base_dir)

    def load_images(self, base_dir):
        for label in labels:
            img_folder = os.path.join(base_dir, label, 'Images')
            class_num = labels.index(label)
            img_files = os.listdir(img_folder)[:self.max_images]  # Load up to max_images

            for img_name in img_files:
                img_path = os.path.join(img_folder, img_name)
                self.data.append((img_path, class_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Resize to 224x224
        image = np.array(resized_arr).reshape(img_size, img_size, 1)
        image = image / 255.0  # Normalize
        image = image.astype(np.float32)

        # Apply the test transform from main.py (to tensor)
        image = test_transform(image)
        
        label = torch.tensor(label).long()
        return image, label

# Load the test dataset with 1000 images each

test_dir = r'C:\Users\Asus\Documents\Computer Vision\Concrete\Concrete'
test_dataset = TestDataset(test_dir, max_images=1000)  # 1000 Positive + 1000 Negative
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Data Loaded....")

# Load the pre-trained model
model_path = r'C:\Users\Asus\Documents\Computer Vision\final_model.pth'
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluation function
def evaluate_model(model, data_loader):
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    intersection = np.logical_and(np.array(all_labels), np.array(all_predictions))
    union = np.logical_or(np.array(all_labels), np.array(all_predictions))
    iou = np.sum(intersection) / np.sum(union)

    # Print metrics
    print("Test Set Metrics:")
    print("----------------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    print("----------------------------------------------------")
    print()
# Run evaluation
evaluate_model(model, test_loader)

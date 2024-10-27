import os
import shutil
from image_size import dataset_dir

# Dataset directory
dataset_dir = dataset_dir

# How many images do you want to split
NUMBER_OF_IMAGES = 2

# Create the test directory
test_dir = os.path.join(dataset_dir, f'test_{NUMBER_OF_IMAGES}')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Create the Positive and Negative folders inside the test directory
positive_dir = os.path.join(test_dir, 'Positive')
negative_dir = os.path.join(test_dir, 'Negative')
if not os.path.exists(positive_dir):
    os.makedirs(positive_dir)
if not os.path.exists(negative_dir):
    os.makedirs(negative_dir)

# Extract images from each folder based on NUMBER_OF_IMAGES
for folder in ['Positive', 'Negative']:
    images = [f for f in os.listdir(os.path.join(dataset_dir, folder)) if f.endswith('.jpg')]
    images.sort()
    for i in range(min(NUMBER_OF_IMAGES, len(images))):
        image_file = images[i]
        shutil.copy(os.path.join(dataset_dir, folder, image_file), os.path.join(test_dir, folder))
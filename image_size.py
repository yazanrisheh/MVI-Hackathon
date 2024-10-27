import os
import cv2

# Define the dataset directory
dataset_dir = 'C:\\Users\\Asus\\Documents\\Computer Vision\\dataset'

# Sub-directories to check inside 'dataset'
folders = ['Positive', 'Negative']

def check_image_sizes(dataset_dir, folders):
    sizes = {}
    mismatched_images = []

    # Loop through each folder ('Positive' and 'Negative')
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        
        # Check each image in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            
            # If the image is not read correctly then skip it
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            
            # Get the dimensions of the image (height, width, channels)
            img_size = img.shape
            
            # If this size is not already in the sizes dictionary, add it to know sizes
            if img_size not in sizes:
                sizes[img_size] = []
            
            # Append the image path to the list of images with this size
            sizes[img_size].append(img_path)

    print("\nImage Sizes in the Dataset:")
    for size, img_list in sizes.items():
        print(f"Size {size}: {len(img_list)} images")
        
        # Check mismatches
        if len(sizes) > 1:
            mismatched_images.extend(img_list)

    if len(sizes) == 1:
        print("\nAll images have the same size.")
    else:
        print("\nWarning: Mismatched image sizes found!")
        print("List of mismatched images:")

if __name__ == "__main__":
    check_image_sizes(dataset_dir, folders)


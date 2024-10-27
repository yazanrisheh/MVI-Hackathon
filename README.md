# **Structural Crack Detection Using CNN**

## **Overview**
This project aims to detect structural cracks in buildings and construction sites using a Convolutional Neural Network (CNN) implemented in PyTorch. It uses a dataset containing both positive (crack) and negative (no crack) images, focusing on achieving high precision, recall, F1-score, and IoU metrics.

## **Requirements**

- Python 3.8 or later
- PyTorch version 12.4
- CUDA (if using GPU)


## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yazanrisheh/MVI-Hackathon.git
   cd https://github.com/yazanrisheh/MVI-Hackathon.git
   

2. **Create Virtual env:**
   ```
   python -m venv venv
   venv/Scripts/Activate

3. **Install Packages:**
   ```
   pip install -r requirements.txt

4. **Setup WAND for Visualization:**
   ```
   Create account at wandb.ai
   Log in using your API key with the following command: wandb login <your_api_key>

## Modules

**ain.py:**
```
Main code with pre-processing, augmentation and training model
```

**image_size.py:**
```
Checks the size of images
```

**split_images.py:**
```
splits images based on number you put
to test different samples like 250, 1000, 5000 etc..
```

**organizer_code.py:**
```
Make sure to modify the "base_dir" path in the TestDataset class to point to your test directory.
```

## Run the test

**Testing:**
```
Run organizer_code.py and dont forget to change "base_dir"
```

# Tumor Segmentation for Brain & Breast: Deep Learning Approach

## 📌 Project Overview

This project focuses on the **automatic detection and segmentation of tumors** in medical images using deep learning techniques.  
It targets **brain tumors** from MRI scans and **breast tumors** from ultrasound images. The goal is to assist in medical diagnosis and treatment planning by providing accurate, efficient segmentation, reducing reliance on manual, error-prone methods.

The project leverages **Convolutional Neural Networks (CNNs)**, with a primary focus on the **U-Net++ architecture** combined with a **ResNet34 encoder** pre-trained on ImageNet. It also compares **U-Net**, **U-Net++**, and **DeepLabV3** architectures.

**Technologies:** Python, PyTorch, `segmentation_models.pytorch` library

---

## 🗂 Datasets

Two Kaggle datasets are used:

### 1. Brain Tumor Dataset
- **Source:** [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets)
- **Description:** MRI brain images with corresponding binary masks.
- **Structure:**  
  - `images/` – Original MRI scans  
  - `masks/` – Binary masks for tumor regions  
- **Size:** ~3000+ images  
- **Challenges:** Varied tumor sizes and positions

### 2. Breast Ultrasound Images Dataset
- **Source:** [Kaggle Breast Ultrasound Dataset](https://www.kaggle.com/datasets)
- **Description:** Ultrasound images classified as normal, benign, or malignant, with masks for tumoral cases.
- **Challenges:** High visual noise, low contrast, varied tumor shapes

**Data Split:**  
- Training: 70%  
- Validation: 20%  
- Test: 10%

**Preparation:**  
1. Download and extract datasets into `data/brain/` and `data/breast/`  
2. Run preprocessing:  
```bash
## ⚙️ Methodology

### Preprocessing & Augmentation
- Resize images to **512x512**
- Convert grayscale images to **RGB (3 channels)**
- Apply **Albumentations** for data augmentation: rotations, flips, zooms, translations

### Model Architecture
- **Primary Model:** U-Net++ with **ResNet34 encoder** (pre-trained on ImageNet)
- **Advantages:** Nested skip connections for fine-grained segmentation
- **Comparisons:**  
  - **U-Net:** Standard encoder-decoder with skip connections  
  - **DeepLabV3:** Atrous convolutions for better context

### Training
- **Loss Function:** Hybrid Dice Loss + Binary Cross-Entropy (BCE)  
```python
L_Total = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε) + λ * [-1/N ∑ y_i log(p_i)]
reuirements.txt
torch>=1.10.0
torchvision>=0.11.0
segmentation-models-pytorch>=0.2.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
albumentations>=1.1.0

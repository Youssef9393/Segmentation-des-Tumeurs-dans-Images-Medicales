# Tumor Segmentation for Brain & Breast using Deep Learning

<p align="center">
  <img src="https://github.com/user-attachments/assets/40d17a0b-d044-4d4f-99a7-3aa8bbd528f9" width="900">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Segmentation Models](https://img.shields.io/badge/Segmentation_Models_PyTorch-009688?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-00C853?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</p>

---

# Overview

This repository presents a deep learning framework for automatic tumor segmentation from medical images. The project focuses on two important medical imaging modalities:

- Brain Tumor Segmentation using MRI images
- Breast Tumor Segmentation using Ultrasound images

The objective is to accurately delineate tumor regions, reducing the burden of manual annotation while supporting clinicians in diagnosis and treatment planning.

The proposed approach employs **U-Net++ with a pretrained ResNet34 encoder**, and compares its performance with **U-Net** and **DeepLabV3**.

---

# Features

- Automatic medical image segmentation
- Brain MRI tumor segmentation
- Breast ultrasound tumor segmentation
- Transfer Learning using ImageNet pretrained encoder
- Extensive data augmentation
- Multiple segmentation architectures
- Dice + Binary Cross Entropy loss
- Quantitative evaluation on unseen test data
- Easy training and inference pipeline

---

# Datasets

## Brain Tumor MRI Dataset

**Source:** Kaggle

### Description

- Brain MRI scans
- Binary tumor masks
- Approximately 3,000+ images

### Challenges

- Large variability in tumor size
- Different anatomical locations
- Irregular tumor boundaries

---

## Breast Ultrasound Dataset

**Source:** Kaggle

### Description

Breast ultrasound images with segmentation masks including:

- Normal
- Benign
- Malignant

### Challenges

- Speckle noise
- Low image contrast
- Complex tumor morphology

---

# Data Split

| Dataset | Percentage |
|----------|-----------:|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

---

# Data Preprocessing

The preprocessing pipeline includes:

- Resize images to **512 × 512**
- Convert grayscale images to RGB
- Normalize pixel intensities
- Convert masks into binary format

---

# Data Augmentation

Albumentations is used with:

- Horizontal Flip
- Vertical Flip
- Rotation
- Random Scale
- Shift
- Elastic Transform
- Random Brightness and Contrast

---

# Model Architectures

## U-Net++ (Primary Model)

- ResNet34 encoder pretrained on ImageNet
- Nested skip connections
- Better feature propagation
- Improved segmentation of small tumors

## Baseline Models

### U-Net

- Encoder-decoder architecture
- Skip connections
- Strong medical imaging baseline

### DeepLabV3

- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale context aggregation
- Robust semantic segmentation

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Batch Size | Configurable |
| Learning Rate | 1e-4 |
| Image Size | 512 × 512 |
| Epochs | Configurable |
| Encoder | ResNet34 |
| Pretrained | ImageNet |

---

# Loss Function

The training objective combines Dice Loss and Binary Cross Entropy (BCE):

\[
L = L_{Dice} + \lambda L_{BCE}
\]

where

\[
L_{Dice}=1-\frac{2|P\cap G|+\epsilon}{|P|+|G|+\epsilon}
\]

- **P**: Predicted mask
- **G**: Ground truth mask

---

# Evaluation Metrics

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall
- F1-score
- Pixel Accuracy

---

# Project Structure

```text
Tumor-Segmentation/
│
├── data/
│   ├── brain/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── breast/
│       ├── images/
│       └── masks/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── outputs/
│   ├── predictions/
│   ├── checkpoints/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/yourusername/Tumor-Segmentation.git

cd Tumor-Segmentation

pip install -r requirements.txt
```

---

# Training

```bash
python src/train.py
```

---

# Evaluation

```bash
python src/evaluate.py
```

---

# Inference

```bash
python src/inference.py
```

---

# Requirements

```text
torch>=1.10.0
torchvision>=0.11.0
segmentation-models-pytorch>=0.2.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
albumentations>=1.1.0
```

---

# Main Libraries

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/)
[![Segmentation Models PyTorch](https://img.shields.io/badge/Segmentation_Models_PyTorch-009688?style=for-the-badge)](https://github.com/qubvel-org/segmentation_models.pytorch)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-00C853?style=for-the-badge)](https://albumentations.ai/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

---

# Results

The proposed **U-Net++** model consistently outperforms the baseline architectures.

- Higher Dice Score
- Better IoU
- Improved segmentation of small lesions
- Robust performance on noisy ultrasound images
- Better boundary delineation than U-Net and DeepLabV3

---

# Future Work

- TransUNet
- SwinUNet
- Attention U-Net
- SAM-based segmentation
- MONAI integration
- Mixed Precision Training
- Clinical deployment using FastAPI
- Web interface using React

---

# Author

**Youssef ELJAOUHARY**

Master's Student in Web Intelligence & Data Science

**Research Interests**

- Artificial Intelligence
- Medical AI
- Computer Vision
- Explainable AI
- Deep Learning
- Medical Image Analysis

---

# License

This project is released under the **MIT License**.

---

# Citation

If you use this project in your research, please cite:

```bibtex
@software{eljaouhary2026tumorsegmentation,
  author       = {Youssef ELJAOUHARY},
  title        = {Tumor Segmentation for Brain and Breast Using Deep Learning},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/YOUR_USERNAME/Tumor-Segmentation},
  note         = {Deep Learning framework for brain MRI and breast ultrasound tumor segmentation using U-Net++}
}
```

---

If you find this project useful, consider giving it a star on GitHub.

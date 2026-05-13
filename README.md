# Tumor Segmentation for Brain & Breast: Deep Learning Approach
<img width="929" height="770" alt="image" src="https://github.com/user-attachments/assets/40d17a0b-d044-4d4f-99a7-3aa8bbd528f9" />
## Project Overview

This project focuses on automatic detection and segmentation of tumors in medical images using deep learning techniques.

It t
argets:
- Brain tumors from MRI scans
- Breast tumors from ultrasound images

The objective is to assist medical diagnosis by providing accurate and efficient tumor segmentation, reducing manual annotation effort and improving clinical decision support.

The project is based on convolutional neural networks, mainly using the U-Net++ architecture with a ResNet34 encoder pretrained on ImageNet. A comparative study is also conducted between U-Net, U-Net++, and DeepLabV3.

Technologies used include Python, PyTorch, and segmentation_models.pytorch.

---

## Datasets

### Brain Tumor Dataset

Source: Kaggle Brain Tumor Dataset

Description:
MRI brain images paired with binary segmentation masks.

Structure:
- images/ : MRI scans
- masks/ : tumor segmentation masks

Size: approximately 3000+ images

Challenges:
- Variation in tumor size
- Irregular tumor shapes
- Different tumor locations

---

### Breast Ultrasound Dataset

Source: Kaggle Breast Ultrasound Dataset

Description:
Ultrasound images categorized into normal, benign, and malignant cases, with segmentation masks for tumor regions.

Challenges:
- Low contrast images
- High noise levels
- Irregular tumor boundaries

---

## Data Split

- Training: 70%
- Validation: 20%
- Testing: 10%

---

## Methodology

### Preprocessing

- Resize images to 512x512
- Convert grayscale images to RGB (3 channels)
- Normalize pixel values

---

### Data Augmentation

Using Albumentations:
- Horizontal flipping
- Vertical flipping
- Rotation
- Zoom
- Translation
- Elastic deformation

---

## Model Architecture

### U-Net++ (Primary Model)

- Encoder: ResNet34 pretrained on ImageNet
- Nested skip connections for improved feature fusion
- Enhanced ability to capture fine-grained details

---

### Baseline Models

- U-Net: Standard encoder-decoder architecture with skip connections
- DeepLabV3: Atrous spatial pyramid pooling for contextual information extraction

---

## Training Setup

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: Combined Dice Loss and Binary Cross-Entropy (BCE)

---

## Loss Function

\[
L_{total} = 1 - \frac{2|P \cap G| + \epsilon}{|P| + |G| + \epsilon}
+ \lambda \left(-\frac{1}{N} \sum y_i \log(p_i)\right)
\]

Where:
- P: Predicted mask
- G: Ground truth mask
- Dice loss measures overlap
- BCE improves pixel-wise classification

---

## Requirements

torch>=1.10.0  
torchvision>=0.11.0  
segmentation-models-pytorch>=0.2.0  
numpy>=1.21.0  
opencv-python>=4.5.0  
matplotlib>=3.4.0  
scikit-learn>=1.0.0  
albumentations>=1.1.0  

---

## Project Structure

data/  
├── brain/  
│   ├── images/  
│   ├── masks/  
├── breast/  
│   ├── images/  
│   ├── masks/  

src/  
├── models/  
├── datasets/  
├── utils.py  
├── train.py  
├── evaluate.py  

---

## Results

- U-Net++ provides better segmentation accuracy compared to U-Net and DeepLabV3
- Improved detection of small and irregular tumor regions
- Better robustness on noisy ultrasound images

---

## Future Work

- Integration of transformer-based models (TransUNet, SwinUNet)
- Multi-class segmentation for tumor subtypes
- Deployment using FastAPI and React
- Real-time inference for clinical applications

---

## Technologies

- Python
- PyTorch
- segmentation_models.pytorch
- OpenCV
- Albumentations
- Matplotlib

---

## License

This project is intended for academic and research purposes only.
albumentations>=1.1.0

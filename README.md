# RetinaLens – AI-Based Diabetic Retinopathy Classification

## Overview

RetinaLens is an AI-based diabetic retinopathy (DR) classification project that uses deep learning to classify retinal fundus images into five stages of diabetic retinopathy.

The project uses **MobileNetV2 Transfer Learning** to classify retinal images and **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the image regions that contribute to the model's predictions.

The objective is to explore how deep learning and explainable AI can support automated retinal image analysis.

## Objectives

- Develop a deep learning model for five-class diabetic retinopathy classification.
- Apply image preprocessing and data augmentation.
- Use transfer learning to improve classification performance.
- Evaluate the model using standard classification metrics.
- Implement Grad-CAM to provide visual explanations of model predictions.

## DR Classification Categories

The model classifies retinal images into five categories:

| Class | Description |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Model Architecture:** MobileNetV2
- **Dataset:** RetinaMNIST (224 × 224)
- **Image Processing:** torchvision, OpenCV, PIL
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Evaluation:** Scikit-learn
- **Development Environment:** Google Colab

## Dataset

RetinaLens uses the RetinaMNIST dataset from the MedMNIST collection.

- Dataset: RetinaMNIST
- Image resolution: 224 × 224 pixels
- Image format: RGB
- Training images: 1,080
- Validation images: 120
- Test images: 400
- Total images: 1,600

Dataset source:

- [MedMNIST Official Website](https://medmnist.com/)
- [MedMNIST Dataset Repository](https://zenodo.org/records/10519652)

## Model Architecture

RetinaLens uses MobileNetV2 with ImageNet-pretrained weights.

The classifier is modified to produce five output classes corresponding to the diabetic retinopathy severity levels.

### Training Approach

The model is trained using two phases:

**Phase 1 – Classifier Training**
- Freeze the pretrained feature extraction layers.
- Train the modified classification head.
- Use weighted cross-entropy loss.

**Phase 2 – Fine-Tuning**
- Unfreeze the final feature extraction blocks.
- Fine-tune the model using lower learning rates.
- Monitor validation macro F1-score.
- Apply early stopping to reduce overfitting.

### Image Preprocessing

The preprocessing pipeline includes:

- Resizing images to 224 × 224.
- Random horizontal flipping.
- Random rotation.
- Color jitter augmentation.
- Tensor conversion.
- ImageNet normalization.

## Explainable AI – Grad-CAM

Grad-CAM is integrated to improve the interpretability of the model.

It generates heatmaps highlighting image regions that contribute to the selected classification prediction.

The visualization includes:

1. Original retinal fundus image.
2. Grad-CAM heatmap.
3. Heatmap overlay on the original image.

Grad-CAM provides an approximate explanation of model behavior. It does not perform lesion segmentation or confirm the presence of specific retinal abnormalities.



The model demonstrates stronger classification performance for No DR and Severe DR categories. Moderate DR remains a challenging classification category.


## Installation and Execution

This project is implemented in Google Colab.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RetinaLens
```

### Step 2: Install dependencies

```python
%pip install -q medmnist
```

PyTorch, torchvision, NumPy, Matplotlib and other required libraries should also be available in the Colab environment.

### Step 3: Open the notebook

Open `RetinaLens.ipynb` in Google Colab.

### Step 4: Download the dataset

The notebook loads RetinaMNIST using:

```python
from medmnist import RetinaMNIST

train_dataset = RetinaMNIST(
    split="train",
    download=True,
    size=224
)

val_dataset = RetinaMNIST(
    split="val",
    download=True,
    size=224
)

test_dataset = RetinaMNIST(
    split="test",
    download=True,
    size=224
)
```

### Step 5: Train and evaluate

Execute the notebook cells in sequence to perform preprocessing, model training, validation, testing and Grad-CAM visualization.

### Step 6: Load the trained model

Load the saved best checkpoint to perform predictions and generate Grad-CAM visualizations.

## Limitations

- The dataset contains a relatively small number of images.
- Class imbalance affects classification performance.
- Moderate and Proliferative DR classification require further improvement.
- The model has not been clinically validated.
- Grad-CAM provides visual explanations but does not identify or segment retinal lesions.
- External validation on independent retinal image datasets is required before any real-world clinical application.

## Future Enhancements

- Improve classification performance using additional retinal datasets.
- Experiment with alternative deep learning architectures.
- Implement lesion detection and segmentation.
- Develop a web-based image upload and prediction interface.
- Improve model interpretability and external validation.

## Disclaimer

RetinaLens is an academic machine learning project developed for educational and research purposes. It is not a medical diagnostic system and must not be used as a substitute for professional ophthalmological examination or clinical diagnosis.

## Author

Developed as an academic capstone project.

**Project:** RetinaLens – AI-Based Diabetic Retinopathy Classification Using Retinal Fundus Images

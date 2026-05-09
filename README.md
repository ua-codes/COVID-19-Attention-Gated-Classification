# AGCA Hybrid Model — COVID-19 CT Classification

This repository contains the implementation of the **AGCA Hybrid Model** for COVID-19 detection from CT scans. Developed as part of an **MSAI Semester Project**, this project combines deep learning (CNN) with symbolic logic reasoning to improve classification performance and interpretability across multiple datasets.

## 📌 Project Overview

The core objective is to accurately classify CT scan slices into three categories: **Normal**, **COVID-19**, and **CAP-Pneumonia**. The model uses an attention-guided backbone and an uncertainty-aware fusion mechanism that integrates traditional neural network outputs with symbolic logic based on clinical indicators.

### Key Highlights

* **Hybrid Architecture:** Combines a ResNet-50 backbone with custom Attention Gates and a Symbolic Logic Mapper.
* **Uncertainty-Aware Fusion:** Dynamically weights CNN predictions and symbolic reasoning based on prediction entropy.
* **Robust Training:** Includes fixes for degenerate collapse and handles multi-domain data (DICOM, PNG, NIfTI).
* **Cross-Dataset Evaluation:** Validated on high-quality clinical datasets including COVID-CT-MD and Kaggle Lung CT.

---

## 🏗 Model Architecture

The **AGCA Hybrid Model** consists of three main components:

1. **Attention-Guided Backbone:**
* Utilizes a ResNet-50 feature extractor.
* Features an **Attention Gate** (Conv2d + Sigmoid) placed after Layer 2 to focus on relevant pathological features in the CT slices.


2. **Symbolic Logic Mapper:**
* A dedicated head that predicts five clinical indicators (symptoms/features) from the image features.
* Translates deep features into human-interpretable "indicators."


3. **Uncertainty-Aware Fusion:**
* Calculates the entropy of the CNN classification.
* If the CNN is uncertain (high entropy), the model gives more weight to the **Symbolic Reasoning** module, which applies logical rules (e.g., `If Indicator A and B are present, then COVID-19`) to determine the final class.



---

## 📊 Datasets

The model is trained and validated on a combination of datasets to ensure generalization:

* **COVID-CT-MD:** 43,322 DICOM slices (Normal / COVID-19 / CAP-Pneumonia).
* **Kaggle Lung CT:** 8,439 PNG images (Normal / COVID-19).
* **MosMed (Test Set):** NIfTI files integrated via a custom data loader patch.
* **BMIC (Test Set)** 

---

## 🚀 Key Features & Fixes

* **Degenerate Collapse Prevention:** Utilizes `WeightedRandomSampler` and class-weighted Cross-Entropy loss to handle class imbalance.
* **Multi-Format Loader:** A robust `MedicalDataset` class capable of handling `.dcm`, `.nii.gz`, `.png`, and `.jpg` files.
* **Dynamic Checkpointing:** Automatically saves the best model based on validation accuracy and F1-score.
* **CPU/GPU Auto-Detection:** Scales batch sizes and epochs automatically based on available hardware.

---

## 🛠 Installation & Requirements

To run this notebook, you will need the following libraries:

```bash
pip install torch torchvision pydicom nibabel pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg scikit-learn tqdm seaborn matplotlib

```

### Setup

1. Clone the repository.
2. Ensure your data is organized in the paths specified in the `CFG` (Configuration) cell.
3. (Optional) If using Google Colab, mount your drive to access the datasets.

---

## 💻 Usage

### 1. Configuration

Modify the `CFG` dictionary in the notebook to set your image size, batch size, and learning rate:

```python
CFG = {
    'img_size': 224,
    'batch_size': 32,
    'epochs': 10,
    'lr': 1e-4,
    # ... other params
}

```

### 2. Training

Run the training cell to start the process. The model saves the best weights as `agca_best.pth`.

```python
# The notebook executes the training loop and validation automatically
for epoch in range(1, CFG['epochs'] + 1):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_acc, val_f1, ... = eval_epoch(model, val_loader)

```

### 3. Evaluation

Load the saved checkpoint to evaluate performance on the test set:

```python
checkpoint = torch.load('agca_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# Evaluation generates Accuracy, F1-Score, and Confusion Matrices

```

---

## 📈 Results

The model produces comprehensive evaluation metrics including:

* **Classification Report:** Precision, Recall, and F1-score per class.
* **Confusion Matrix:** Visualization of true vs. predicted labels.
* **ROC Curves:** Multi-class Area Under the Curve (AUC) analysis.
* **Uncertainty Analysis:** Visualization of how the model switches between CNN and Symbolic logic.

---

## 🎓 Credits

Developed as part of the **MSAI Semester Project** project. Special thanks to the researchers and clinical providers who made the COVID-19 CT datasets available for public use.

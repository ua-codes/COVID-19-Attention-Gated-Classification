# COVID-19-Attention-Gated-Classification
```python
import json

# Load the notebook content
with open('Copy_of_COVID19_Final_Model.ipynb', 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

# Extract text from cells to summarize project details
cells = nb_data.get('cells', [])
content_summary = []

for cell in cells:
    if cell['cell_type'] == 'markdown':
        content_summary.append("".join(cell['source']))
    elif cell['cell_type'] == 'code':
        # Just grab the first few lines of code to see imports or main logic
        code_lines = cell['source']
        if code_lines:
            content_summary.append(f"CODE: {''.join(code_lines[:10])}")

# Print a summary to get a better idea
for text in content_summary[:20]: # Look at first 20 cells
    print("---")
    print(text)



```

```text
---
# AGCA Hybrid Model — COVID-19 CT Classification
### MSAI Thesis | Final Version
**Datasets used (confirmed from your Drive):**
- `COVID-CT-MD` → 43,322 DICOM slices (Normal / COVID-19 / CAP-Pneumonia)
- `Kaggle Lung CT` → 8,439 PNG images (Normal / COVID-19)

**Fixes:**
1. ✅ Degenerate collapse — WeightedRandomSampler + class-weighted loss  
2. ✅ Checkpoint mismatch — strict=True, locked architecture  
3. ✅ Paths hardcoded to your confirmed Drive locations  
4. ✅ CPU-safe — auto-detects GPU/CPU, reduced defaults for CPU run  

---
CODE: # CELL 1 — Install dependencies
!pip install -q pydicom nibabel pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
!pip install -q scikit-learn torchvision tqdm seaborn
print("✅ Packages ready.")

---
CODE: # CELL 2 — Imports
import os, glob, random, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,

---
CODE: # CELL 3 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

---
CODE: # ── CELL 4 — Master Configuration (Updated) ──────────────────────
import torch
import os

IS_GPU = torch.cuda.is_available()

CFG = {
    'img_size'    : 224,
    'batch_size'  : 32 if IS_GPU else 8,
    'epochs'      : 10 if IS_GPU else 3,

---
CODE: import os

def check_path(name, path):
    print(f"\n🔍 Checking {name} at: {path}")
    if os.path.exists(path):
        contents = os.listdir(path)
        print(f"   ✅ Path exists. Found {len(contents)} items.")
        print(f"   📁 Top-level folders/files: {contents[:10]}")
        # Check one level deeper in case of nested folders
        for item in contents[:3]:

---
CODE: # ── CELL 4 — Master Configuration (Final Verified Version) ──────────────────────
import torch
import os

IS_GPU = torch.cuda.is_available()

CFG = {
    'img_size'    : 224,
    'batch_size'  : 32 if IS_GPU else 8,
    'epochs'      : 10 if IS_GPU else 3,

---
CODE: # ── CELL 5 — Smart Data Ingestion (Limited to 11,000) ──────────────────────
import glob
import re
import numpy as np
import random # Added for shuffling

def collect_dataset(ds_cfg):
    samples = []
    root, ext, dom_id = ds_cfg['root'], ds_cfg['ext'], ds_cfg['domain_id']


---
CODE: import nibabel as nib
import os
import numpy as np
import cv2

# 1. Function to find and label the NIfTI files from your Drive
def get_mosmed_test_samples(base_path, samples_per_cat=25):
    new_samples = []
    # MosMed standard folders: CT-0 (Normal), CT-1 (Mild COVID)
    # Check your folders: they might be 'CT-0', 'CT-1', etc.

---
CODE: import random

# ... after all_samples += s loop ...

# 1. Shuffle to get a diverse mix of classes and domains
random.seed(42) # For reproducibility
random.shuffle(all_samples)

# 2. Slice to the paper's specific requirement
if len(all_samples) > CFG['max_samples']:

---
CODE: # CELL 6 — Stratified Split + optional downsampling
idx = np.arange(len(all_samples))

# Downsample to max_samples (stratified) for base-paper comparison
if CFG['max_samples'] and len(idx) > CFG['max_samples']:
    _, idx = train_test_split(idx, test_size=CFG['max_samples'],
                              stratify=labels_all, random_state=SEED)
    print(f"Downsampled to {len(idx):,} samples (stratified).")

sub_labels = labels_all[idx]

---
CODE: # CELL 7 — Dataset class (handles DICOM + PNG/JPG safely)
class MedicalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def _load(self, path):
        try:

---
CODE: # CELL 8 — Weighted sampler + DataLoaders (fixes degenerate collapse)
class_counts  = np.bincount(train_labels, minlength=CFG['num_classes']).astype(float)
class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
sample_w = class_weights[train_labels]
sampler  = WeightedRandomSampler(sample_w, len(train_set), replacement=True)

NW = 4   # 0 = stable for DICOM in Colab
train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], sampler=sampler,   num_workers=NW, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=CFG['batch_size'], shuffle=False,     num_workers=NW, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=CFG['batch_size'], shuffle=False,     num_workers=NW, pin_memory=True)

---
CODE: # CELL 9 — AGCA Hybrid Model (single definition — used for train AND load)
class AttentionGuidedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(*list(resnet.children())[:5])
        for p in self.stem.parameters(): p.requires_grad = False
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

---
CODE: # CELL 10 — Loss, optimiser, training + eval loops
cw_tensor     = torch.tensor(class_weights/class_weights.sum(), dtype=torch.float).to(device)
criterion_cls = nn.CrossEntropyLoss(weight=cw_tensor)
criterion_ind = nn.BCELoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG['lr'], weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)

---
CODE: # CELL 11 — Run training (saves best checkpoint automatically)
best_acc = 0.0
history  = {'loss':[], 'val_acc':[], 'val_f1':[]}

print(f"🚀 Training for {CFG['epochs']} epochs on {device}...")
print("─"*55)

for epoch in range(1, CFG['epochs']+1):
    loss = train_epoch(model, train_loader, optimizer)
    acc, f1, _, _, _ = eval_epoch(model, val_loader)

---
CODE: print(vars(test_set).keys())
---
CODE: print(test_set.samples[0])
---
CODE: # CELL 12 — Load best checkpoint (strict=True) & evaluate on test set
ckpt = torch.load(os.path.join(CFG['save_dir'],'agca_best.pth'), map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.eval()
print(f"✅ Loaded best checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

test_acc, test_f1, test_probs, test_labels, test_preds = eval_epoch(model, test_loader)
p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
present = np.unique(np.concatenate([test_labels, test_preds]))
tnames  = [CLASS_NAMES[i] for i in present]

---
CODE: print(CFG['paths'].keys())


```

```python
# Get the full source of the model definition cell
for cell in cells:
    if 'class AttentionGuidedBackbone' in "".join(cell['source']):
        print("".join(cell['source']))
        break


```

```text
# CELL 9 — AGCA Hybrid Model (single definition — used for train AND load)
class AttentionGuidedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(*list(resnet.children())[:5])
        for p in self.stem.parameters(): p.requires_grad = False
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool= resnet.avgpool
        self.attention_gate = nn.Sequential(nn.Conv2d(512,1,1), nn.Sigmoid())

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = x * self.attention_gate(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)


class SymbolicLogicMapper(nn.Module):
    def __init__(self, in_dim=2048, n_indicators=5):
        super().__init__()
        self.indicator_head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_indicators), nn.Sigmoid()
        )
    def forward(self, x): return self.indicator_head(x)


class AGCA_HybridModel(nn.Module):
    def __init__(self, num_classes=3, n_indicators=5):
        super().__init__()
        self.backbone        = AttentionGuidedBackbone()
        self.classifier      = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))
        self.symbolic_mapper = SymbolicLogicMapper(2048, n_indicators)

    def forward(self, x):
        feat       = self.backbone(x)
        logits     = self.classifier(feat)
        cnn_probs  = F.softmax(logits, dim=1)
        indicators = self.symbolic_mapper(feat)
        return logits, cnn_probs, indicators


def batch_symbolic_reasoning(indicators):
    B = indicators.size(0)
    preds = torch.zeros(B, dtype=torch.long, device=indicators.device)
    covid_mask     = (indicators[:,0]>0.5) & (indicators[:,1]>0.5) & (indicators[:,2]<0.4)
    pneumonia_mask = (indicators[:,4]>0.6)
    preds[covid_mask]     = 1
    preds[pneumonia_mask] = 2
    return preds


def uncertainty_aware_fusion(cnn_probs, indicators, n_cls=3):
    if indicators.dim()==1: indicators = indicators.unsqueeze(0)
    if cnn_probs.dim()==1:  cnn_probs  = cnn_probs.unsqueeze(0)
    entropy = -torch.sum(cnn_probs * torch.log(cnn_probs+1e-8), dim=1)
    alpha   = torch.clamp(1.0-(entropy/1.099), 0.05, 1.0).unsqueeze(1)
    sym_idx   = batch_symbolic_reasoning(indicators)
    sym_probs = F.one_hot(sym_idx, num_classes=n_cls).float()
    return alpha*cnn_probs + (1-alpha)*sym_probs, alpha.squeeze(1)


model = AGCA_HybridModel(CFG['num_classes'], 5).to(device)
tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ AGCA Hybrid Model ready on {device} | Trainable params: {tp:,}")



```

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

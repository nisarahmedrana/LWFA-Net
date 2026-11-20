# LFA-Net: A Lightweight Fusion Attention Network for Breast Ultrasound Lesion Segmentation

 **Paper:** *"A Lightweight Fusion Attention Network for Breast Ultrasound Lesion Segmentation"*  
 **Framework:** TensorFlow / Keras &nbsp;|&nbsp; **Language:** Python 3 &nbsp;|&nbsp; **Platform:** Google Colab (GPU)

---

## Overview

This repository contains the official implementation of **LFA-Net**, a lightweight encoder–decoder segmentation network designed for accurate and efficient lesion delineation in breast ultrasound (BUS) and digital mammography images. LFA-Net integrates three synergistic components:

- **Focal Modulation** — channel-wise modulation driven by the contrast between global max and mean pooling responses, sharpening the network's focus on lesion regions.
- **Context Aggregation** — a parallel branch combining local 3×3 convolutions with global context to capture multi-scale spatial information.
- **Vision Mamba-Inspired Block** — a lightweight token-mixing and channel-mixing module with depthwise convolutions and GELU activations, providing long-range dependency modelling at low computational cost.

These blocks are assembled into a **LiteFusion Attention** module placed at the bottleneck of a U-Net-style encoder–decoder backbone with skip connections and an **RA (Residual Attention)** head for the final prediction.

---

## Repository Structure

```
LFA-Net/
│
├── LFA_Net_updated_BUSI_26_12_25.ipynb   # Training & evaluation on BUSI dataset
├── LFA_Net_updated_DMID_22_1_26.ipynb    # Training & evaluation on DMID dataset
│
├── BUSI_Processed/                     # BUSI dataset (augmented, 256×256)
│   ├── train/
│   │   ├── x/                            # Training images
│   │   └── y/                            # Training masks
│   └── test/
│       ├── x/                            # Test images
│       └── y/                            # Test masks
│
├── UDIAT Processed/                          # UDIAT dataset
│   ├── train/
│   │   ├── x/                            # Training images
│   │   └── y/                            # Training masks
│   └── test/
│       ├── x/                            # Test images
│       └── y/                            # Test masks
│
└── README.md
```

## Network Architecture

LFA-Net follows a U-Net encoder–decoder structure:

| Stage | Block | Spatial Resolution |
|---|---|---|
| Encoder E1 | Conv Block | 256 × 256 |
| Encoder E2 | Conv Block + MaxPool | 128 × 128 |
| Encoder E3 | Conv Block + MaxPool | 64 × 64 |
| Encoder E4 | Conv Block + MaxPool | 32 × 32 |
| **Bottleneck** | **LiteFusion Attention** | **16 × 16** |
| Decoder D4 | ConvTranspose + Skip (E4) | 32 × 32 |
| Decoder D3 | ConvTranspose + Skip (E3) | 64 × 64 |
| Decoder D2 | ConvTranspose + Skip (E2) | 128 × 128 |
| Decoder D1 | ConvTranspose + Skip (E1) | 256 × 256 |
| Output | RA Attention + Sigmoid | 256 × 256 |

The **LiteFusion Attention** bottleneck block comprises:
1. Layer Normalisation + 3×3 Conv projection
2. Focal Modulation Context Aggregation (focal_modulation_context_aggregation)
3. 1×1 Conv projection + residual addition
4. Vision Mamba-Inspired token/channel mixing (VisionMambaInspired)

---

## Datasets

### BUSI — Breast Ultrasound Images Dataset

- **Source:** Al-Dhabyani et al. (2020), publicly available at [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Content:** Ultrasound images of normal, benign, and malignant breast lesions with expert-annotated binary masks
- **Pre-processing applied in this work:**
  - Resized to **256 × 256** pixels
  - Offline data augmentation (horizontal flipping)
  - 80 / 20 training / validation split (random seed 42)
  - Separate held-out test set

Expected folder layout after unzipping `BUSI_256x256_Aug.zip`:

```
BUSI_256x256_Aug/
├── train/
│   ├── x/   ← RGB images  (.png / .jpg)
│   └── y/   ← Binary masks (.png)
└── test/
    ├── x/
    └── y/
```

### UDIAT — Digital Mammography Image Dataset

- **Source:** Publicly available on [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/ayush02102001/udiat-segmentation-dataset)) as `UDIAT_Kagle.zip`
- **Content:** Ultrasound images and binary masks
- **Pre-processing applied in this work:**
  - Resized to **256 × 256** pixels
  - 80 / 20 training / validation split (random seed 42)

Expected folder layout after unzipping `DMID_Kaggle.zip`:

```
UDIAT_Processed/
├── images/   ← Mammography images (.png / .jpg)
└── masks/    ← Ground-truth masks (.png)
```

---

## Requirements

The notebooks are designed to run on **Google Colab** with a GPU runtime. Key dependencies:

```
tensorflow >= 2.10
keras
keras-preprocessing
scikit-learn
scikit-image
opencv-python (cv2)
matplotlib
numpy
Pillow
```

Install additional packages at runtime (already included in notebooks):

```bash
pip install keras-preprocessing scikit-image
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/LFA-Net.git
cd LFA-Net
```

### 2. Prepare data

Upload the dataset zip files to your Google Drive:

- `BUSI_Processed.zip` → Google Drive root (for BUSI notebook)
- `UDIAT_Processed.zip` → `MyDrive/Nisar_work/` (for UDIAT notebook)

### 3. Open a notebook in Google Colab

| Notebook | Dataset |
|---|---|
| `LFA_Net_updated_BUSI_26_12_25.ipynb` | BUSI |
| `LFA_Net_updated_UDIAT_22_1_26.ipynb` | UDIAT |

Mount Google Drive when prompted, then run all cells in order.

### 4. Training configuration

Both notebooks share the following defaults:

| Hyperparameter | Value |
|---|---|
| Input size | 256 × 256 × 3 |
| Batch size (BUSI) | 32 |
| Batch size (UDIAT) | 8 |
| Epochs | 150 (with early stopping) |
| Optimiser | Adam (lr = 0.001) |
| Loss | Binary Cross-Entropy + Dice Loss |
| LR scheduler | ReduceLROnPlateau (factor=0.25, patience=5, min_lr=1e-9) |
| Early stopping | patience=11, min_delta=0.001 |
| Random seed | 42 |

Model weights are saved automatically to Google Drive via `ModelCheckpoint` when validation loss improves.

---

## Evaluation Metrics

The following metrics are computed on the test set:

| Metric | Description |
|---|---|
| **Dice Coefficient** | Overlap between prediction and ground truth |
| **IoU / Jaccard Index** | Intersection over Union |
| **Sensitivity (Recall)** | True positive rate |
| **Specificity** | True negative rate |
| **Accuracy** | Pixel-wise accuracy |
| **AUC-ROC** | Area under the receiver operating characteristic curve |
| **Precision–Recall AUC** | Area under the precision–recall curve |

Segmentation outputs and ground-truth masks are also saved as PNG images for qualitative inspection.

---

## Loss Functions

LFA-Net is trained with a composite loss:

```
L_total = BCE(y_true, y_pred) + DiceLoss(y_true, y_pred)
```

Additional loss variants implemented in the notebooks:

- `DiceLoss` — pure Dice loss
- `bcc_Jaccard_dice_loss` — Dice loss + (1 − IoU)
- `Jaccard_coef_loss` — 1 − Jaccard index

---

## Citation

If you use this code or the LFA-Net architecture in your research, please cite:

```bibtex
@article{lfanet2025,
  title   = {A Lightweight Fusion Attention Network for Breast Ultrasound Lesion Segmentation},
  author  = {[Azeem Aslam, Asim Naveed, Nisar Ahmed, Pekka Ruusuvuori]},
  journal = {[Submitted in Elsevier - Array]},
  year    = {2026},
}
```
---

## License

This project is released for academic and research use. Please refer to the `LICENSE` file for full terms.

- The **BUSI** dataset: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images*. Data in Brief, 2020.
- The **DMID** dataset sourced via Kaggle.
- Architecture inspired by focal modulation networks and Vision Mamba.

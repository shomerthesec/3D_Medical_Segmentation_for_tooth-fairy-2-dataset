# 3D Tooth Segmentation with MONAI

Advanced 3D segmentation model for tooth and anatomical structure identification in CT/CBCT scans using the ToothFairy dataset and MONAI framework.

## Project Overview

This project provides a complete pipeline for:
- **Training** a deep learning model (SwinUNETR, SegResNet, or UNet) for 3D tooth segmentation
- **Inference** on new CT/CBCT volumes
- **Visualization** of segmentation results with full volume montages

### Key Features
- Multi-class segmentation: 48 tooth types + 10 anatomical structures (49 classes total)
- Supports multiple 3D architectures (SwinUNETR, SegResNet, UNet)
- Automatic Mixed Precision (AMP) training for efficiency
- TensorBoard logging for training monitoring
- Advanced data augmentation (spatial & intensity)
- Memory-efficient validation with GPU caching

---

## Installation

### Prerequisites
- **Python**: 3.9 or higher
- **GPU**: CUDA-capable GPU recommended (CPU works but slower)
- **Storage**: ~100 GB for full ToothFairy dataset 

### Step 1: Clone/Download the Project
```bash
cd 3d_medical_segmentation
git clone https://github.com/shomerthesec/3D_Medical_Segmentation_for_tooth-fairy-2-dataset.git
```

### Step 2: Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key dependencies** in requirements.txt:
- `torch >= 2.0.0`
- `monai[all] >= 1.3.0`
- `nibabel >= 5.1.0`
- `matplotlib >= 3.8.0`
- `scikit-learn >= 1.3.0`

---

## File Locations & Organization

```
3d_medical_segmentation/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── Training & Inference Scripts:
├── advanced-models-for-3d-segmentation (1).ipynb  # Main training notebook
├── testing_swinUNTER.py                         # Inference script
│
├── Trained Models:
├── best_model_3d_segres.pth                     # Pre-trained SegResNet model
├── best1.pth                                    # Pre-trained SwinUNETR model
│
├── test_sample/                                 # Test dataset
│   └── imagesTr/                                # Test images (.mha or .nii.gz)
│
├── inference_results/                           # Inference outputs
│   ├── plots/                                   # Visualization montages
│   └── pred_*.nii.gz                            # Predicted segmentations
│
└── Utility Scripts:
    ├── convert_from_mha_to_dicom.py             # Format conversion

```

---

## Usage

### Option 1: Training (Jupyter Notebook)

**File:** `advanced-models-for-3d-segmentation (1).ipynb`

This notebook provides a training pipeline. Edit Config class to set hyperparameters:

```python
class Config:
    BASE_DIR = Path("/path/to/tooth-fairy-dataset")
    MODEL_NAME = "SwinUNETR"  # or "SegResNet", "UNet"
    BATCH_SIZE = 2
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3
```

Run:
```bash
jupyter notebook "advanced-models-for-3d-segmentation (1).ipynb"
```

Dataset structure:
```
/path/to/tooth-fairy-dataset/
├── imagesTr/          # Training images (.mha files)
│   ├── image_001.mha
│   ├── image_002.mha
│   └── ...
├── labelsTr/          # Training labels (.mha files)
│   ├── label_001.mha
│   ├── label_002.mha
│   └── ...
└── dataset.json       # Metadata (optional)
```

---

### Option 2: Inference (Python Script)

**File:** `testing_swinUNTER.py`

Runs inference on new images and generates visualization montages.

Edit configuration:
```python
MODEL_PATH = "3d_medical_segmentation/best1.pth"
TEST_IMG_DIR = "3d_medical_segmentation/test_sample/imagesTr"
OUTPUT_DIR = "./inference_results"
```

**To Run:**
```bash
python testing_swinUNTER.py
```

**Input Requirements:**
- Model file (`.pth`) - Trained PyTorch model
- Image directory containing:
  - `.mha` files (MetaImage format) OR
  - `.nii.gz` files (NIfTI format)

**Outputs Generated:**
```
inference_results/
├── plots/
│   ├── image_001.mha_montage.png      # 8x8 grid of all 64 slices
│   ├── image_002.mha_montage.png
│   └── ...
├── pred_image_001.nii.gz              # Predicted segmentation masks
├── pred_image_002.nii.gz
└── ...
```

**Output Format:**
- Each montage PNG shows all axial slices with:
  - Grayscale CT image
  - Color-coded segmentation overlay (49 classes)
  - Legend with class names and IDs
  - Class names: tooth types (e.g., "Upper Right Central Incisor") + anatomical structures

---

## Configuration Reference

### Config Class (Training)
Located in the training notebook, cell 2:

```python
class Config:
    # Paths
    BASE_DIR = Path("/kaggle/input/tooth-fairy-2-dataset/Dataset112_ToothFairy2")
    IMG_DIR = BASE_DIR / "imagesTr"
    LAB_DIR = BASE_DIR / "labelsTr"
    
    # Model Selection
    MODEL_NAME = "SwinUNETR"        # "SegResNet" | "SwinUNETR" | "UNet"
    SPATIAL_SIZE = (64, 64, 64)    # Patch size for training
    
    # Training Hyperparameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    VAL_INTERVAL = 1               # Validate every N epochs
    
    # GPU & Performance
    USE_AMP = True                 # Automatic Mixed Precision
    NUM_WORKERS = 4                # Data loading threads
    
    # Loss & Metrics
    # DiceCELoss: Combines Dice + Cross-Entropy
```

### TestConfig (Inference)
Located in `testing_swinUNTER.py`, line ~30:

```python
class TestConfig:
    SPATIAL_SIZE = (64, 64, 64)
    IN_CHANNELS = 1
    OUT_CHANNELS = 49
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' or 'cpu'
```

---

## Class Labels Reference

### Anatomical Structures (Labels 0-10)
| ID | Name |
|----|------|
| 0 | Background |
| 1-2 | Jawbones (Lower/Upper) |
| 3-4 | Inferior Alveolar Canals (Left/Right) |
| 5-6 | Maxillary Sinuses (Left/Right) |
| 7-10 | Pharynx, Bridge, Crown, Implant |

### Tooth Labels (11-48)
**Quadrant Numbering:**
- **1x**: Upper Right (UR) - e.g., 11=UR Central Incisor
- **2x**: Upper Left (UL) - e.g., 21=UL Central Incisor
- **3x**: Lower Left (LL) - e.g., 31=LL Central Incisor
- **4x**: Lower Right (LR) - e.g., 41=LR Central Incisor

**Tooth Types (by position within quadrant):**
- `1`: Central Incisor
- `2`: Lateral Incisor
- `3`: Canine
- `4`: 1st Premolar
- `5`: 2nd Premolar
- `6`: 1st Molar
- `7`: 2nd Molar
- `8`: 3rd Molar

**Example:** Label 46 = Lower Right (4x) 1st Molar (6)

---

## Training Tips

### Model Selection
- **SwinUNETR** (Recommended): Transformer-based, best accuracy (~20k params less than UNet)
- **SegResNet**: ResNet-based, good balance of speed/accuracy
- **UNet**: Classic, fastest for smaller datasets

### Data Augmentation
The notebook applies aggressive augmentation:
- **Spatial**: Random flips, rotations, affine transforms
- **Intensity**: Gaussian noise, smoothing, scale/shift
- **Cropping**: Positive/negative sampling (2:1 ratio)

### GPU Memory Management
- Reduce `BATCH_SIZE` if out-of-memory errors occur
- `USE_AMP=True` reduces memory by ~50%
- Validation uses direct forward pass (no sliding window) to save memory

### Monitoring Progress
TensorBoard logs are saved to `Config.LOG_DIR`:
```bash
tensorboard --logdir /path/to/logs
```

Then open http://localhost:6006 in browser.

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
- Reduce `BATCH_SIZE` from 2 to 1
- Reduce `SPATIAL_SIZE` from (64,64,64) to (48,48,48)
- Enable `USE_AMP = True`

### Issue: Model Not Found During Inference
**Solution:**
- Ensure path in `testing_swinUNTER.py` line ~X points to correct `.pth` file
- Check file permissions: `ls -la best1.pth`

### Issue: No Images Found
**Solution:**
- Verify `TEST_IMG_DIR` contains `.mha` or `.nii.gz` files
- Check file format: `file image_001.mha`

### Issue: Slow Inference
**Solution:**
- Ensure GPU is being used: Check output from `print(TestConfig.DEVICE)`
- Reduce batch size or enable GPU monitoring with `nvidia-smi`

---

## Pre-Trained Models

Two pre-trained models are provided:

### 1. `best1.pth` (SwinUNETR)
- **Architecture**: SwinUNETR with transformers
- **Input**: Single-channel CT (HU units -1000 to +3000)
- **Output**: 49-class segmentation

### 2. `best_model_3d_segres.pth` (SegResNet)
- **Architecture**: ResNet-based 3D segmentation
- **Input**: Single-channel CT
- **Output**: 49-class segmentation

**To Use Different Model:**
Edit `testing_swinUNTER.py`:
```python
MODEL_PATH = "./best_model_3d_segres.pth"  # Change here
```

---

## Data Format Support

### Input Formats (Training & Inference)
- **`.mha`** (MetaImage) - Primary format used in ToothFairy
- **`.nii.gz`** (NIfTI) - Common medical imaging format
- **`.dcm`** (DICOM) - Conversion script available: `convert_from_mha_to_dicom.py`

### Output Formats (Inference)
- **`.nii.gz`** (NIfTI) - Segmentation masks with original affine
- **`.png`** (PNG) - Visualization montages (8x8 grid of slices)

---

## Quick Start Examples

### Train from Scratch (2 epochs, 1 sample)
```bash
# 1. Edit the notebook Config class:
#    NUM_EPOCHS = 2
#    BASE_DIR = "/path/to/your/dataset"

# 2. Run in Jupyter:
jupyter notebook "advanced-models-for-3d-segmentation (1).ipynb"

# 3. Execute all cells (Ctrl+Shift+Enter or menu)
```

### Infer on Custom Images
```bash
# 1. Place images in a directory:
mkdir -p custom_images
cp your_scans/*.mha custom_images/

# 2. Edit testing_swinUNTER.py:
#    TEST_IMG_DIR = "./custom_images"

# 3. Run:
python testing_swinUNTER.py

# 4. Check results:
ls inference_results/plots/
```

---

## References

### Dataset
[ToothFairy 2 Challenge](https://toothfairy.grand-challenge.org/) - MICCAI 2023

### Framework
- **MONAI**: [Medical Open Network for AI](https://monai.io/)
- **PyTorch**: [PyTorch.org](https://pytorch.org/)

### Models
- **SwinUNETR**: Ze Liu et al., "Swin Transformers for Medical Image Analysis"
- **SegResNet**: Xueying Zhai et al., "Automatic Segmentation of MR Prostate Images with Ensembles of Neural Networks"

---

# 3D SegNet for Multi-Class Liver Segmentation

A deep learning framework for multi-class liver segmentation using 3D SegNet architecture with Double Deep Q-Network (DDQN) refinement.

## 📋 Project Overview

This project implements a hybrid deep learning approach for multi-class segmentation of liver structures in 3D medical images. The framework combines:
- **3D SegNet**: A semantic segmentation architecture for volumetric data
- **DDQN (Double Deep Q-Network)**: Reinforcement learning for iterative segmentation refinement
- **Comprehensive pipeline**: DICOM conversion, label processing, training, and evaluation

The model segments multiple anatomical structures including liver, hepatic vessels, arteries, and veins (up to 9 classes).

## 🏗️ Architecture

### 3D SegNet
- **Encoder**: 5 contracting blocks with max-pooling and index storage
- **Decoder**: 5 expanding blocks with max-unpooling using stored indices
- **Skip connections**: Pooling indices enable precise spatial recovery
- **Base channels**: 32, expanding to 512 at bottleneck

### DDQN Refinement
- **State space**: Concatenated image patches + initial predictions
- **Action space**: 9 actions (no change + 8 class probability adjustments)
- **Experience replay**: 10,000 transition memory buffer
- **Target network**: Updated every 100 steps for stable learning
- **Epsilon-greedy**: Decay from 1.0 to 0.01 for exploration

## 📁 Project Structure

```
├── 1_dicome_to_nii.py      # DICOM to NIfTI conversion
├── 2_label_conversion.py    # Label processing and merging
├── 3_region_of_interest.py  # ROI extraction for liver masks
├── train.py                  # Main training script
├── test.py                   # Model evaluation script
└── README.md                 # This file
```

## 🔧 Requirements

### Core Dependencies
```
python >= 3.8
pytorch >= 1.9.0
monai >= 1.0.0
nibabel >= 3.2.0
SimpleITK >= 2.0.0
numpy >= 1.21.0
tqdm >= 4.62.0
```

### Optional (for visualization)
```
matplotlib >= 3.4.0
pandas >= 1.3.0
```

## 📊 Data Preparation Pipeline

### Step 1: DICOM to NIfTI Conversion
```bash
python 1_dicome_to_nii.py
```
- Converts DICOM series to NIfTI format
- Organizes output in structured directories
- Handles multi-series patient data

### Step 2: Label Processing
```bash
python 2_label_conversion.py
```
- Processes hepatic/artery/vein label files
- Merges multiple structure labels
- Creates unified multi-class segmentation masks
- Classes: background (0), hepatic vessels (1-8)

### Step 3: ROI Extraction
```bash
python 3_region_of_interest.py
```
- Extracts liver region masks
- Filters non-liver tissues
- Prepares data for focused segmentation

## 🚀 Training

### Data Organization
```
Data/Converted/split_data/
├── train_images/           # Training images (NIfTI)
├── train_labels/           # Training labels (NIfTI)
├── test_images/            # Test images
└── test_labels/            # Test labels
```

### Training Configuration
```python
# Model parameters
in_channels = 1
out_channels = 9
base_channels = 32

# Training parameters
max_epochs = 10000
batch_size = 4
roi_size = (96, 96, 64)
initial_lr = 5e-5

# DDQN parameters
ddqn_gamma = 0.99
ddqn_epsilon = 1.0
ddqn_epsilon_min = 0.01
ddqn_epsilon_decay = 0.995
ddqn_memory_size = 10000
```

### Start Training
```bash
python train.py
```

The training script:
- Uses mixed precision training (AMP) for efficiency
- Implements CosineAnnealingLR scheduler
- Logs metrics to CSV file
- Saves best model based on Dice score
- Handles keyboard interrupts gracefully

## 🧪 Testing and Evaluation

### Run Evaluation
```bash
python test.py
```

### Testing Modes
1. **SegNet Only**: Baseline segmentation performance
2. **SegNet + DDQN**: Full refinement pipeline

### Output Metrics
- Overall Dice score (mean ± std)
- Per-class Dice scores
- Individual sample results
- Formatted text reports

## 📈 Results Visualization

The training logger automatically generates:
- SegNet loss curves
- DDQN loss progression
- Dice score improvement
- Epsilon decay
- Learning rate schedule

To plot results:
```python
from train import plot_training_results
plot_training_results("path/to/training_log.txt")
```

## 💾 Model Checkpoints

Models are saved with naming convention:
```
segnet_ddqn_YYYYMMDD_HHMMSS_dice_0.XXXX.pth
```

Checkpoint contains:
- SegNet state dict
- DDQN online network
- DDQN target network
- Optimizer states
- Training metadata

## 🔄 Refinement Process

The DDQN refinement operates on 16³ patches:
1. Initial SegNet prediction
2. State = [image_patch, prediction_patch]
3. DDQN selects optimal refinement action
4. Apply action to adjust class probabilities
5. Compute reward based on Dice improvement
6. Store experience in replay memory
7. Update online network periodically

## ⚙️ Customization

### Modifying Class Labels
Edit `2_label_conversion.py`:
```python
# Change label mapping
pattern = re.compile(r"(hepatic|artery|vein)(\d+)")
# Adjust output labels as needed
```

### Adjusting Network Architecture
In `train.py`, modify:
- `base_channels`: Control model capacity
- `ddqn_hidden_dim`: DDQN feature dimension
- `ddqn_actions`: Number of refinement actions

### Data Augmentation
Training pipeline includes:
- Random cropping (positive/negative sampling)
- Random flips (3D)
- Random 90° rotations
- Random affine transformations

## 🐛 Troubleshooting

### Common Issues

1. **No training data found**
   - Verify data paths in `train.py`
   - Check directory structure matches expectations

2. **NaN loss encountered**
   - Model automatically handles NaN values
   - Training continues up to 100 NaN occurrences

3. **CUDA out of memory**
   - Reduce batch size
   - Decrease ROI size
   - Adjust sw_batch_size in sliding window inference

4. **DICOM conversion errors**
   - Ensure DICOM series are complete
   - Check for corrupted files
   - Verify directory permissions

## 📝 Logging

Training logs are saved to:
```
Data/Converted/Niazi/training_log_YYYYMMDD_HHMMSS.txt
```

Format:
```
epoch,segnet_loss,ddqn_loss,epsilon,avg_improvement,dice_score,learning_rate
```


## 👥 Contributors

- Muhammad Irfan Khan
- Contributors welcome!

## 🙏 Acknowledgments

- MONAI for medical imaging frameworks
- PyTorch for deep learning infrastructure
- SimpleITK and NiBabel for medical image processing


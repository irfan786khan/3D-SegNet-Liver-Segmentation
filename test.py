# test_segnet_ddqn.py
# Testing script for SegNet + DDQN model with custom model path

import os
import glob
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from collections import deque, namedtuple

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, EnsureTyped
)
from monai.data import Dataset
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ---- Settings ----
data_dir = r"E:\linkdoc\vesselGuided\vesselGuided\Data\Converted\split_data"
test_images_dir = os.path.join(data_dir, "test_images")
test_labels_dir = os.path.join(data_dir, "test_labels")

# Model weights path - UPDATE THIS PATH
model_weights_path = r"E:\linkdoc\vesselGuided\vesselGuided\Data\Converted\Niazi\segnet_ddqn_20251009_041224_dice_0.9139.pth"

batch_size = 1
roi_size = (96, 96, 64)
sw_batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DDQN Components (must match training) ----
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DDQN(nn.Module):
    """Double DQN network for segmentation refinement"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDQN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(state_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ---- SegNet Model ----
class SegNet(nn.Module):
    """SegNet architecture for 3D semantic segmentation"""
    def __init__(self, in_channels=1, out_channels=9, base_channels=32):
        super(SegNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        self.enc5 = self._make_encoder_block(base_channels * 8, base_channels * 16)
        
        # Decoder (Expanding Path)
        self.dec5 = self._make_decoder_block(base_channels * 16, base_channels * 8)
        self.dec4 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_decoder_block(base_channels * 2, base_channels)
        self.dec1 = self._make_decoder_block(base_channels, base_channels)
        
        # Final convolution
        self.final_conv = nn.Conv3d(base_channels, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(2, 2)
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block with two convolutional layers"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with two convolutional layers"""
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with pooling indices storage
        x1 = self.enc1(x)
        x1_size = x1.size()
        x1, idx1 = self.pool(x1)
        
        x2 = self.enc2(x1)
        x2_size = x2.size()
        x2, idx2 = self.pool(x2)
        
        x3 = self.enc3(x2)
        x3_size = x3.size()
        x3, idx3 = self.pool(x3)
        
        x4 = self.enc4(x3)
        x4_size = x4.size()
        x4, idx4 = self.pool(x4)
        
        x5 = self.enc5(x4)
        x5_size = x5.size()
        x5, idx5 = self.pool(x5)
        
        # Decoder with unpooling using stored indices
        x5 = self.unpool(x5, idx5, output_size=x5_size)
        x5 = self.dec5(x5)
        
        x4 = self.unpool(x5, idx4, output_size=x4_size)
        x4 = self.dec4(x4)
        
        x3 = self.unpool(x4, idx3, output_size=x3_size)
        x3 = self.dec3(x3)
        
        x2 = self.unpool(x3, idx2, output_size=x2_size)
        x2 = self.dec2(x2)
        
        x1 = self.unpool(x2, idx1, output_size=x1_size)
        x1 = self.dec1(x1)
        
        return self.final_conv(x1)

# ---- SegNet with DDQN Model (for loading) ----
class SegNetWithDDQN(nn.Module):
    """SegNet architecture with DDQN refinement for 3D semantic segmentation"""
    def __init__(self, in_channels=1, out_channels=9, base_channels=32, ddqn_hidden_dim=256):
        super(SegNetWithDDQN, self).__init__()
        
        self.out_channels = out_channels
        self.segnet = SegNet(in_channels, out_channels, base_channels)
        
        # DDQN components
        self.ddqn_online = DDQN(out_channels + 1, 9, ddqn_hidden_dim).to(device)  # 9 actions as in training
        self.ddqn_target = DDQN(out_channels + 1, 9, ddqn_hidden_dim).to(device)
        
    def forward(self, x, training_mode=True, use_refinement=True):
        # Get initial SegNet prediction
        segnet_output = self.segnet(x)
        
        if not use_refinement or not training_mode:
            return segnet_output
        
        return segnet_output
    
    def apply_refinement_action(self, initial_pred, action_map, image):
        """Apply refinement actions to initial predictions"""
        refined_pred = initial_pred.clone()
        
        # Action meanings (matching training):
        # 0: No change
        # 1-8: Increase probability for class 1-8
        
        for action in range(1, 9):  # 9 actions total
            mask = (action_map == action)
            if mask.any():
                # Increase probability for the target class
                refined_pred[:, action-1:action] += 0.1 * mask.float()
        
        # Ensure probabilities are valid
        refined_pred = torch.sigmoid(refined_pred)  # Convert to probabilities
        return refined_pred

# ---- Inference with Refinement ----
def inference_with_refinement(model, input_volume, refinement_steps=3):
    """Perform inference with DDQN refinement during testing"""
    model.eval()
    
    # Get initial prediction
    with torch.no_grad():
        initial_pred = sliding_window_inference(
            input_volume, roi_size, sw_batch_size, 
            lambda x: model(x, training_mode=False, use_refinement=False)
        )
    
    refined_pred = initial_pred.clone()
    
    # Apply refinement steps
    for step in range(refinement_steps):
        batch_size, channels, d, h, w = input_volume.shape
        
        # Sample random patches for refinement
        patch_size = 16
        d_start = random.randint(0, d - patch_size)
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
        
        image_patch = input_volume[:, :, 
                                 d_start:d_start+patch_size,
                                 h_start:h_start+patch_size,
                                 w_start:w_start+patch_size]
        
        pred_patch = refined_pred[:, :,
                                d_start:d_start+patch_size,
                                h_start:h_start+patch_size,
                                w_start:w_start+patch_size]
        
        # Create state and get best action (no exploration during inference)
        state = torch.cat([image_patch, pred_patch], dim=1)
        with torch.no_grad():
            q_values = model.ddqn_online(state)
            action = q_values.max(1)[1].item()
        
        # Apply refinement action
        action_map = torch.full((batch_size, 1, patch_size, patch_size, patch_size), 
                              action, device=device)
        refined_patch = model.apply_refinement_action(pred_patch, action_map, image_patch)
        
        # Update the refined prediction
        refined_pred[:, :,
                   d_start:d_start+patch_size,
                   h_start:h_start+patch_size,
                   w_start:w_start+patch_size] = refined_patch
    
    return refined_pred

def main():
    # Check if test data exists
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    if not os.path.exists(test_labels_dir):
        print(f"Test labels directory not found: {test_labels_dir}")
        return
    
    # Get test files
    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(test_labels_dir, "*.nii.gz")))
    
    if not test_images or not test_labels:
        print("No test files found!")
        return
    
    # Match images and labels
    test_data = []
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        corresponding_label = None
        for label_path in test_labels:
            label_name = os.path.basename(label_path)
            if img_name.replace("image", "label").replace("Image", "Label") in label_name:
                corresponding_label = label_path
                break
            elif img_name == label_name:
                corresponding_label = label_path
                break
        
        if corresponding_label:
            test_data.append({"image": img_path, "label": corresponding_label})
        else:
            print(f"Warning: No matching label found for {img_name}")
    
    if not test_data:
        print("No valid image-label pairs found!")
        return
    
    print(f"Testing on {len(test_data)} samples")
    
    # Test transforms
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"])
    ])
    
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model (use SegNetWithDDQN to load the complete trained model)
    model = SegNetWithDDQN(in_channels=1, out_channels=9, base_channels=32).to(device)
    
    # Check if model exists
    if not os.path.exists(model_weights_path):
        print(f"❌ Model not found: {model_weights_path}")
        models_dir = os.path.dirname(model_weights_path)
        model_files = glob.glob(os.path.join(models_dir, "segnet_ddqn_*.pth"))
        if model_files:
            print("Available models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {os.path.basename(model_file)}")
        return
    
    # Load model
    print(f"Loading model from: {model_weights_path}")
    try:
        # Try to load the complete checkpoint (from training)
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        if 'segnet_state_dict' in checkpoint:
            # This is a training checkpoint
            model.segnet.load_state_dict(checkpoint['segnet_state_dict'])
            if 'ddqn_online_state_dict' in checkpoint:
                model.ddqn_online.load_state_dict(checkpoint['ddqn_online_state_dict'])
                model.ddqn_target.load_state_dict(checkpoint['ddqn_target_state_dict'])
            print("✅ Complete model (SegNet + DDQN) loaded successfully!")
        else:
            # This might be just the SegNet weights
            model.segnet.load_state_dict(checkpoint)
            print("✅ SegNet model loaded successfully!")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying to load with strict=False...")
        try:
            checkpoint = torch.load(model_weights_path, map_location=device)
            if isinstance(checkpoint, dict) and 'segnet_state_dict' in checkpoint:
                model.segnet.load_state_dict(checkpoint['segnet_state_dict'], strict=False)
            else:
                model.segnet.load_state_dict(checkpoint, strict=False)
            print("✅ Model loaded with strict=False!")
        except Exception as e2:
            print(f"❌ Still failed to load model: {e2}")
            return
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
    
    # Testing with different modes
    print("\nTesting modes:")
    print("1. SegNet only (baseline)")
    print("2. SegNet + DDQN refinement")
    
    for mode in [1, 2]:
        print(f"\n{'='*60}")
        if mode == 1:
            print("🧪 MODE 1: SegNet Only (Baseline)")
            use_refinement = False
        else:
            print("🧪 MODE 2: SegNet + DDQN Refinement")
            use_refinement = True
        
        model.eval()
        dice_scores = []
        per_sample_class_scores = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                try:
                    print(f"Processing test sample {i+1}/{len(test_loader)}")
                    
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)
                    
                    print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
                    
                    if use_refinement:
                        # Use refinement inference
                        outputs = inference_with_refinement(model, inputs, refinement_steps=3)
                    else:
                        # Use standard sliding window inference
                        outputs = sliding_window_inference(
                            inputs, 
                            roi_size, 
                            sw_batch_size, 
                            lambda x: model(x, training_mode=False, use_refinement=False),
                            overlap=0.5,
                            mode="gaussian"
                        )
                    
                    # Apply sigmoid and threshold
                    outputs_prob = torch.sigmoid(outputs)
                    outputs_binary = (outputs_prob > 0.5).float()
                    
                    print(f"Output shape: {outputs_binary.shape}")
                    
                    # Calculate dice metric for overall score
                    dice_metric(y_pred=outputs_binary, y=labels)
                    dice_value = dice_metric.aggregate().item()
                    dice_scores.append(dice_value)
                    
                    # Calculate per-class scores for this sample
                    dice_metric_batch(y_pred=outputs_binary, y=labels)
                    class_dice = dice_metric_batch.aggregate()
                    
                    # Store per-class scores for this sample
                    if class_dice is not None:
                        class_scores = class_dice.cpu().numpy()
                        per_sample_class_scores.append(class_scores)
                        
                        print(f"Sample {i+1} Dice: {dice_value:.4f}")
                        if len(class_scores) <= 10:  # Only print if not too many classes
                            print("Per-class Dice scores:")
                            for class_idx, dice_score in enumerate(class_scores):
                                print(f"  Class {class_idx}: {dice_score:.4f}")
                    else:
                        print(f"Sample {i+1} Dice: {dice_value:.4f}")
                    
                    # Reset metrics for next sample
                    dice_metric.reset()
                    dice_metric_batch.reset()
                    
                    print("-" * 30)
                    
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                    continue
        
        # Calculate overall results for this mode
        if dice_scores:
            overall_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            
            print(f"📊 { 'SEGNET + DDQN REFINEMENT' if use_refinement else 'SEGNET ONLY'} RESULTS:")
            print(f"Overall Dice Score: {overall_dice:.4f} ± {std_dice:.4f}")
            print(f"Number of test samples: {len(dice_scores)}")
            
            # Calculate and display average per-class scores across all samples
            if per_sample_class_scores:
                avg_class_scores = np.mean(per_sample_class_scores, axis=0)
                std_class_scores = np.std(per_sample_class_scores, axis=0)
                print("\nAverage per-class Dice scores across all samples:")
                for class_idx, (dice_score, dice_std) in enumerate(zip(avg_class_scores, std_class_scores)):
                    print(f"  Class {class_idx}: {dice_score:.4f} ± {dice_std:.4f}")
            
            # Save results
            results_dir = os.path.join(data_dir, "Niazi", "test_results")
            os.makedirs(results_dir, exist_ok=True)
            
            mode_name = "segnet_only" if mode == 1 else "segnet_ddqn_refinement"
            results_file = os.path.join(results_dir, f"test_results_{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(results_file, 'w') as f:
                f.write(f"Test Results - {mode_name.upper()}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model: {model_weights_path}\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Overall Dice Score: {overall_dice:.4f} ± {std_dice:.4f}\n")
                f.write(f"Number of test samples: {len(dice_scores)}\n\n")
                
                f.write("Individual Sample Results:\n")
                for i, (dice_score, class_scores) in enumerate(zip(dice_scores, per_sample_class_scores)):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  Overall Dice: {dice_score:.4f}\n")
                    if len(class_scores) <= 10:
                        f.write(f"  Per-class Dice scores:\n")
                        for class_idx, class_dice in enumerate(class_scores):
                            f.write(f"    Class {class_idx}: {class_dice:.4f}\n")
                    f.write("\n")
                
                if per_sample_class_scores:
                    avg_class_scores = np.mean(per_sample_class_scores, axis=0)
                    std_class_scores = np.std(per_sample_class_scores, axis=0)
                    f.write("Average per-class Dice scores across all samples:\n")
                    for class_idx, (dice_score, dice_std) in enumerate(zip(avg_class_scores, std_class_scores)):
                        f.write(f"  Class {class_idx}: {dice_score:.4f} ± {dice_std:.4f}\n")
            
            print(f"Results saved to: {results_file}")
        else:
            print("❌ No samples were successfully processed!")

if __name__ == '__main__':
    import random
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()
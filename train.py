# train_segnet_ddqn.py
# SegNet + Double DQN refinement framework for liver multi-class segmentation

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from collections import deque, namedtuple
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandAffined, EnsureTyped
)
from monai.data import CacheDataset
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

import sys
import multiprocessing
# ---- Settings ----
data_dir = r"E:\linkdoc\vesselGuided\vesselGuided\Data\Converted"
split_data_dir = os.path.join(data_dir, "split_data")

# Use split data
train_images_dir = os.path.join(split_data_dir, "train_images")
train_labels_dir = os.path.join(split_data_dir, "train_labels")

dataset_root = "suggestedTrainOrder"
dataset_name = "allSegmentationOrderLiver_3d"
label_name = "allSegmentationOrderLabelsTr_hepatic_1_8Liver_3d"

max_epochs = 10000
batch_size = 4
roi_size = (96, 96, 64)
sw_batch_size = 4
initial_lr = 5e-5

# DDQN parameters
ddqn_gamma = 0.99
ddqn_epsilon = 1.0
ddqn_epsilon_min = 0.01
ddqn_epsilon_decay = 0.995
ddqn_memory_size = 10000
ddqn_batch_size = 32
ddqn_learning_rate = 1e-4
ddqn_target_update = 100
ddqn_actions = 9  # Number of refinement actions per voxel

set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Training Results Logging ----
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("epoch,segnet_loss,ddqn_loss,epsilon,avg_improvement,dice_score,learning_rate\n")
    
    def log_epoch(self, epoch, segnet_loss, ddqn_loss, epsilon, avg_improvement, dice_score, learning_rate):
        """Log epoch results to file"""
        log_line = f"{epoch},{segnet_loss:.6f},{ddqn_loss:.6f},{epsilon:.6f},{avg_improvement:.6f},{dice_score:.6f},{learning_rate:.8f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def get_log_path(self):
        """Get the path to the log file"""
        return self.log_file

# ---- Experience Replay Memory ----
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# ---- DDQN Network ----
class DDQN(nn.Module):
    """Double DQN network for segmentation refinement"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDQN, self).__init__()
        
        # State: [batch, channels, depth, height, width] - using feature maps
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

# ---- SegNet Model with DDQN Integration ----
class SegNetWithDDQN(nn.Module):
    """SegNet architecture with DDQN refinement for 3D semantic segmentation"""
    def __init__(self, in_channels=1, out_channels=9, base_channels=32, ddqn_hidden_dim=256):
        super(SegNetWithDDQN, self).__init__()
        
        self.out_channels = out_channels
        self.segnet = SegNet(in_channels, out_channels, base_channels)
        
        # DDQN components
        self.ddqn_online = DDQN(out_channels + 1, ddqn_actions, ddqn_hidden_dim).to(device)  # +1 for image
        self.ddqn_target = DDQN(out_channels + 1, ddqn_actions, ddqn_hidden_dim).to(device)
        self.ddqn_target.load_state_dict(self.ddqn_online.state_dict())
        self.ddqn_target.eval()
        
        # DDQN optimizer and memory
        self.ddqn_optimizer = torch.optim.Adam(self.ddqn_online.parameters(), lr=ddqn_learning_rate)
        self.memory = ReplayMemory(ddqn_memory_size)
        
        self.ddqn_step = 0
        
    def forward(self, x, training_mode=True, use_refinement=True):
        # Get initial SegNet prediction
        segnet_output = self.segnet(x)
        
        if not use_refinement or not training_mode:
            return segnet_output
        
        return segnet_output
    
    def get_refinement_action(self, state, epsilon):
        """Get refinement action using epsilon-greedy policy"""
        if random.random() <= epsilon:
            return random.randrange(ddqn_actions)
        else:
            with torch.no_grad():
                q_values = self.ddqn_online(state.unsqueeze(0))
                return q_values.max(1)[1].item()
    
    def apply_refinement_action(self, initial_pred, action_map, image):
        """Apply refinement actions to initial predictions"""
        refined_pred = initial_pred.clone()
        
        # Action meanings:
        # 0: No change
        # 1-8: Increase probability for class 1-8
        # Note: Adjust based on your specific needs
        
        for action in range(1, ddqn_actions):
            mask = (action_map == action)
            if mask.any():
                # Increase probability for the target class
                refined_pred[:, action-1:action] += 0.1 * mask.float()
        
        # Ensure probabilities are valid
        refined_pred = torch.sigmoid(refined_pred)  # Convert to probabilities
        return refined_pred
    
    def compute_refinement_reward(self, initial_pred, refined_pred, ground_truth):
        """Compute reward for refinement actions"""
        with torch.no_grad():
            # Dice score improvement
            initial_dice = self.compute_dice(initial_pred, ground_truth)
            refined_dice = self.compute_dice(refined_pred, ground_truth)
            
            improvement = refined_dice - initial_dice
            reward = improvement * 10.0  # Scale reward
            
            # Penalty for wrong refinements
            if improvement < 0:
                reward -= 0.1
            
            return reward, improvement
    
    def compute_dice(self, pred, target):
        """Compute Dice score"""
        smooth = 1e-5
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        target_bin = target.float()
        
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()
    
    def update_ddqn(self):
        """Update DDQN network"""
        if len(self.memory) < ddqn_batch_size:
            return 0.0
        
        transitions = self.memory.sample(ddqn_batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
        
        # Current Q values
        current_q = self.ddqn_online(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Double DQN: online network selects actions, target network evaluates
        next_actions = self.ddqn_online(next_state_batch).max(1)[1].unsqueeze(1)
        next_q = self.ddqn_target(next_state_batch).gather(1, next_actions).squeeze(1)
        
        # Target Q values
        target_q = reward_batch + (ddqn_gamma * next_q * (1 - done_batch.float()))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q.detach())
        
        # Optimize
        self.ddqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ddqn_online.parameters(), 1.0)
        self.ddqn_optimizer.step()
        
        # Update target network
        self.ddqn_step += 1
        if self.ddqn_step % ddqn_target_update == 0:
            self.ddqn_target.load_state_dict(self.ddqn_online.state_dict())
        
        return loss.item()

# ---- Original SegNet Model ----
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
        # Encoder 1
        x1 = self.enc1(x)
        x1_size = x1.size()
        x1, idx1 = self.pool(x1)
        
        # Encoder 2
        x2 = self.enc2(x1)
        x2_size = x2.size()
        x2, idx2 = self.pool(x2)
        
        # Encoder 3
        x3 = self.enc3(x2)
        x3_size = x3.size()
        x3, idx3 = self.pool(x3)
        
        # Encoder 4
        x4 = self.enc4(x3)
        x4_size = x4.size()
        x4, idx4 = self.pool(x4)
        
        # Encoder 5 (bottleneck)
        x5 = self.enc5(x4)
        x5_size = x5.size()
        x5, idx5 = self.pool(x5)
        
        # Decoder with unpooling using stored indices
        # Decoder 5
        x5 = self.unpool(x5, idx5, output_size=x5_size)
        x5 = self.dec5(x5)
        
        # Decoder 4
        x4 = self.unpool(x5, idx4, output_size=x4_size)
        x4 = self.dec4(x4)
        
        # Decoder 3
        x3 = self.unpool(x4, idx3, output_size=x3_size)
        x3 = self.dec3(x3)
        
        # Decoder 2
        x2 = self.unpool(x3, idx2, output_size=x2_size)
        x2 = self.dec2(x2)
        
        # Decoder 1
        x1 = self.unpool(x2, idx1, output_size=x1_size)
        x1 = self.dec1(x1)
        
        return self.final_conv(x1)

# ---- Data ----
def get_train_data():
    """Get training data from split_data directories"""
    # Use the split data directories you defined
    train_images = sorted(glob.glob(os.path.join(train_images_dir, "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_labels_dir, "*.nii.gz")))
    
    print(f"Found {len(train_images)} training images")
    print(f"Found {len(train_labels)} training labels")
    
    # Verify that we have matching pairs
    if len(train_images) != len(train_labels):
        print(f"Warning: Mismatch between images ({len(train_images)}) and labels ({len(train_labels)})")
    
    train_data = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
    return train_data

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=roi_size,
                            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
    EnsureTyped(keys=["image", "label"])
])

def main():
    # Define batch_size locally within main function
    batch_size = 4
    roi_size = (96, 96, 64)
    sw_batch_size = 4
    initial_lr = 5e-5
    
    # Create dataset and dataloader
    train_data = get_train_data()
    
    if len(train_data) == 0:
        print("No training data found! Please check your data paths.")
        print(f"Looking for images in: {train_images_dir}")
        print(f"Looking for labels in: {train_labels_dir}")
        return
    
    print(f"Training with {len(train_data)} samples")
    
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training logger
    models_dir = os.path.join(data_dir, "Niazi")
    os.makedirs(models_dir, exist_ok=True)
    logger = TrainingLogger(models_dir)
    print(f"Training logs will be saved to: {logger.get_log_path()}")

    # Use the SegNet with DDQN model
    model = SegNetWithDDQN(
        in_channels=1,
        out_channels=9,
        base_channels=32,
        ddqn_hidden_dim=256
    ).to(device)

    loss_fn = DiceFocalLoss(
        sigmoid=True, to_onehot_y=True,
        lambda_dice=1.0, lambda_focal=0.0,
        smooth_nr=1e-5, smooth_dr=1e-5
    )
    
    # Optimizer for SegNet only
    segnet_optimizer = torch.optim.Adam(model.segnet.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(segnet_optimizer, T_max=max_epochs, eta_min=1e-6)

    # AMP setup
    if torch.cuda.is_available():
        try:
            scaler = GradScaler(device_type='cuda')
        except TypeError:
            scaler = GradScaler()
        use_amp = True
    else:
        scaler = None
        use_amp = False

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ---- Training ----
    best_dice = -1.0
    nan_counter = 0
    nan_limit = 100
    epsilon = ddqn_epsilon
    
    save_path = ""

    for epoch in range(1, max_epochs + 1):
        try:
            model.train()
            epoch_loss = 0
            epoch_ddqn_loss = 0
            epoch_improvement = 0
            refinement_steps = 0
            
            for batch_data in train_loader:
                if isinstance(batch_data, list):
                    inputs = torch.cat([item["image"] for item in batch_data], dim=0).to(device)
                    labels = torch.cat([item["label"] for item in batch_data], dim=0).to(device)
                else:
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)

                # ---- SegNet Training ----
                segnet_optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs, training_mode=True, use_refinement=False)
                        outputs = torch.clamp(outputs, min=-10, max=10)
                        segnet_loss = loss_fn(outputs, labels)
                else:
                    outputs = model(inputs, training_mode=True, use_refinement=False)
                    outputs = torch.clamp(outputs, min=-10, max=10)
                    segnet_loss = loss_fn(outputs, labels)

                if torch.isnan(segnet_loss):
                    print("NaN loss encountered. Replacing with zero loss.")
                    segnet_loss = torch.tensor(0.0, requires_grad=True).to(device)
                    nan_counter += 1
                    if nan_counter >= nan_limit:
                        print("Training stopped due to repeated NaNs.")
                        return
                
                # Backprop for SegNet
                if scaler is not None:
                    scaler.scale(segnet_loss).backward()
                    scaler.unscale_(segnet_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.segnet.parameters(), max_norm=1.0)
                    scaler.step(segnet_optimizer)
                    scaler.update()
                else:
                    segnet_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.segnet.parameters(), max_norm=1.0)
                    segnet_optimizer.step()

                epoch_loss += segnet_loss.item()
                
                # ---- DDQN Training ----
                with torch.no_grad():
                    # Get initial prediction
                    initial_pred = model(inputs, training_mode=True, use_refinement=False)
                    
                    # Sample random patches for DDQN training
                    batch_size_current, channels, d, h, w = inputs.shape
                    patch_size = 16
                    
                    for i in range(2):  # Sample 2 patches per batch
                        # Random patch coordinates
                        d_start = random.randint(0, d - patch_size)
                        h_start = random.randint(0, h - patch_size)
                        w_start = random.randint(0, w - patch_size)
                        
                        # Extract patches
                        image_patch = inputs[:, :, 
                                           d_start:d_start+patch_size,
                                           h_start:h_start+patch_size,
                                           w_start:w_start+patch_size]
                        
                        pred_patch = initial_pred[:, :,
                                                d_start:d_start+patch_size,
                                                h_start:h_start+patch_size,
                                                w_start:w_start+patch_size]
                        
                        label_patch = labels[:, :,
                                           d_start:d_start+patch_size,
                                           h_start:h_start+patch_size,
                                           w_start:w_start+patch_size]
                        
                        # Create state: concatenate image and prediction (shape: [B, C, D, H, W])
                        state = torch.cat([image_patch, pred_patch], dim=1)

                        # Process each sample in the patch batch separately so each replay entry
                        # corresponds to a single state (avoids tensor shape mismatches later).
                        batch_B = state.size(0)
                        for b_idx in range(batch_B):
                            # single-sample tensors with a leading batch dim of 1
                            single_image = image_patch[b_idx:b_idx+1]
                            single_pred = pred_patch[b_idx:b_idx+1]
                            single_state = state[b_idx]  # pass without batch dim to action selector

                            # Get refinement action for this single sample
                            action = model.get_refinement_action(single_state, epsilon)

                            # Apply action and get refined prediction for this sample
                            action_map = torch.full((1, 1, patch_size, patch_size, patch_size),
                                                  action, device=device)
                            refined_pred_b = model.apply_refinement_action(single_pred, action_map, single_image)

                            # Compute reward for this sample
                            reward, improvement = model.compute_refinement_reward(
                                single_pred, refined_pred_b, label_patch[b_idx:b_idx+1])

                            # Next state (with batch dim = 1)
                            next_state_b = torch.cat([single_image, refined_pred_b], dim=1)

                            # Store single-sample experience
                            model.memory.push(
                                single_state.unsqueeze(0).detach(),
                                torch.tensor([action], device=device),
                                torch.tensor([reward], device=device),
                                next_state_b.detach(),
                                torch.tensor([False], device=device)  # Assuming non-terminal
                            )

                            epoch_improvement += improvement.item()
                            refinement_steps += 1
                
                # Update DDQN
                ddqn_loss = model.update_ddqn()
                epoch_ddqn_loss += ddqn_loss
            
            # Decay epsilon
            epsilon = max(ddqn_epsilon_min, epsilon * ddqn_epsilon_decay)
            
        except KeyboardInterrupt:
            print('\nKeyboard interrupt received — saving model and exiting...')
            try:
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f'Saved model to {save_path}')
                else:
                    emergency_save_path = os.path.join(models_dir, f"segnet_ddqn_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                    torch.save(model.state_dict(), emergency_save_path)
                    print(f'Saved emergency model to {emergency_save_path}')
            except Exception as e:
                print('Failed to save model:', e)
            raise

        epoch_loss /= len(train_loader)
        epoch_ddqn_loss = epoch_ddqn_loss / max(refinement_steps, 1)
        avg_improvement = epoch_improvement / max(refinement_steps, 1)
        
        print(f"Epoch {epoch}/{max_epochs}, SegNet Loss: {epoch_loss:.4f}, "
              f"DDQN Loss: {epoch_ddqn_loss:.4f}, Epsilon: {epsilon:.4f}, "
              f"Avg Improvement: {avg_improvement:.4f}")

        # Step the LR scheduler
        try:
            scheduler.step()
        except Exception:
            pass

        # ---- Validation ----
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch_data in train_loader:
                if isinstance(batch_data, list):
                    inputs = torch.cat([item["image"] for item in batch_data], dim=0).to(device)
                    labels = torch.cat([item["label"] for item in batch_data], dim=0).to(device)
                else:
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)

                # Use SegNet only for validation (no refinement during validation)
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, 
                                                 lambda x: model(x, training_mode=False, use_refinement=False))
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()

                dice_metric(y_pred=outputs, y=labels)

            dice = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"🎯 Epoch {epoch} Dice Score: {dice:.4f}")

        # Get current learning rate
        current_lr = segnet_optimizer.param_groups[0]['lr']
        
        # Log epoch results to file
        logger.log_epoch(
            epoch=epoch,
            segnet_loss=epoch_loss,
            ddqn_loss=epoch_ddqn_loss,
            epsilon=epsilon,
            avg_improvement=avg_improvement,
            dice_score=dice,
            learning_rate=current_lr
        )

        # ---- Save best model ----
        if dice > best_dice:
            best_dice = dice
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"segnet_ddqn_{timestamp}_dice_{dice:.4f}.pth"
            save_path = os.path.join(models_dir, model_filename)
            
            torch.save({
                'segnet_state_dict': model.segnet.state_dict(),
                'ddqn_online_state_dict': model.ddqn_online.state_dict(),
                'ddqn_target_state_dict': model.ddqn_target.state_dict(),
                'ddqn_optimizer_state_dict': model.ddqn_optimizer.state_dict(),
                'epsilon': epsilon,
                'best_dice': best_dice,
                'epoch': epoch
            }, save_path)
            print(f"✅ Best model saved at epoch {epoch} with Dice {best_dice:.4f}")
            print(f"   Model saved as: {save_path}")

    print(f"\nTraining completed! Log file saved at: {logger.get_log_path()}")

# ---- Inference with Refinement ----
def inference_with_refinement(model, input_volume, refinement_steps=3):
    """Perform inference with DDQN refinement during testing"""
    model.eval()
    
    # Get initial prediction
    with torch.no_grad():
        initial_pred = sliding_window_inference(input_volume, roi_size, sw_batch_size, 
                                              lambda x: model(x, training_mode=False, use_refinement=False))
    
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

# ---- Plotting Helper Function ----
def plot_training_results(log_file_path):
    """
    Helper function to plot training results from the log file
    Usage: Call this function after training with the path to your log file
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Read the log file
        df = pd.read_csv(log_file_path)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Results', fontsize=16)
        
        # Plot 1: SegNet Loss
        axes[0, 0].plot(df['epoch'], df['segnet_loss'])
        axes[0, 0].set_title('SegNet Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Plot 2: DDQN Loss
        axes[0, 1].plot(df['epoch'], df['ddqn_loss'])
        axes[0, 1].set_title('DDQN Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Plot 3: Dice Score
        axes[0, 2].plot(df['epoch'], df['dice_score'])
        axes[0, 2].set_title('Dice Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Dice')
        axes[0, 2].grid(True)
        
        # Plot 4: Epsilon
        axes[1, 0].plot(df['epoch'], df['epsilon'])
        axes[1, 0].set_title('Epsilon')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Plot 5: Average Improvement
        axes[1, 1].plot(df['epoch'], df['avg_improvement'])
        axes[1, 1].set_title('Average Improvement')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement')
        axes[1, 1].grid(True)
        
        # Plot 6: Learning Rate
        axes[1, 2].plot(df['epoch'], df['learning_rate'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plot_path = log_file_path.replace('.txt', '_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
        
    except ImportError:
        print("Pandas or matplotlib not available. Install with: pip install pandas matplotlib")
    except Exception as e:
        print(f"Error plotting results: {e}")

if __name__ == '__main__':
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    main()
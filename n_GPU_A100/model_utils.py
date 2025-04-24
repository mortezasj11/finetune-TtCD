# File: n_GPU_A100/model_utils.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import json
import time
from torchmetrics.functional import auroc
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List

# Import DinoVisionTransformer class from dinov2 package for reusability
import sys
sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")

from dinov2.models.vision_transformer import DinoVisionTransformer

class CombinedModel(nn.Module):
    def __init__(
        self,
        base_model,
        chunk_feat_dim=768,
        hidden_dim=1024,
        num_tasks=1,
        num_attn_heads=8,
        num_layers=2,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.base = base_model
        
        # Transformer aggregator for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=chunk_feat_dim,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Final classifier(s)
        self.classifier = nn.Linear(chunk_feat_dim, num_tasks)
        
    def forward(self, x):
        batch_size = 1  # Assume batch_size is always 1 (per-patient processing)
        num_chunks = x.size(0)
        
        # Process all chunks with the base model
        features = []
        for i in range(num_chunks):
            # Get CLS token embedding for this chunk
            with torch.no_grad():  # Frozen backbone by default
                chunk_feat = self.base(x[i].unsqueeze(0))
            
            features.append(chunk_feat)
        
        # Stack features from all chunks [num_chunks, feat_dim]
        features = torch.cat(features, dim=0)
        
        # Apply transformer to aggregate chunk features
        aggregated = self.transformer(features.unsqueeze(0))  # [1, num_chunks, feat_dim]
        
        # Global average pooling across chunks
        pooled = torch.mean(aggregated, dim=1)  # [1, feat_dim]
        
        # Apply the classifier to get outputs for each task
        outputs = self.classifier(pooled)  # [1, num_tasks]
        
        return outputs
    
    def freeze_base_model(self):
        """Freeze all parameters in the base model"""
        for param in self.base.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model"""
        for param in self.base.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n=1):
        """Unfreeze only the last n transformer blocks in the base model"""
        # Freeze everything first
        self.freeze_base_model()
        
        # Then selectively unfreeze the last n blocks
        for i in range(len(self.base.blocks) - n, len(self.base.blocks)):
            for param in self.base.blocks[i].parameters():
                param.requires_grad = True

class VolumeProcessor:
    def __init__(self, chunk_depth=3, out_size=(448, 448), vmin=-1000, vmax=1000, eps=1e-6):
        self.chunk_depth = chunk_depth
        self.out_size = out_size
        self.vmin = vmin
        self.vmax = vmax
        self.eps = eps
    
    def _preprocess_volume(self, nifti_path):
        # Load NIfTI file
        nifti = nib.load(nifti_path)
        volume = nifti.get_fdata()
        
        # Normalize to 0-1 range
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume
    
    def _get_chunks(self, volume):
        # Extract 3-slice chunks with stride 1
        depth, height, width = volume.shape
        
        # Create list of chunks
        chunks = []
        for i in range(0, depth - self.chunk_depth + 1):
            chunk = volume[i:i+self.chunk_depth]  # Get 3 consecutive slices
            
            # Resize if needed
            if (height, width) != self.out_size:
                chunk_resized = np.zeros((self.chunk_depth, *self.out_size))
                for j in range(self.chunk_depth):
                    # Use CV2 or other resize method here
                    # For simplicity, this is a placeholder
                    chunk_resized[j] = chunk[j]  # Replace with actual resize
                chunk = chunk_resized
            
            # Create "RGB" channels by duplicating (3D to pseudo-RGB)
            chunk_rgb = np.stack([chunk] * 3, axis=1)  # [3, 3, H, W]
            chunks.append(chunk_rgb)
        
        return chunks
    
    def process_file(self, nifti_path):
        volume = self._preprocess_volume(nifti_path)
        chunks = self._get_chunks(volume)
        return chunks

class NLSTDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 processor: VolumeProcessor,
                 label_cols: List[str]):
        df = df[df.file_path.notna()].reset_index(drop=True)
        self.df = df
        self.proc = processor
        self.labels = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        nii_path = row.file_path
        #print(f"[Data] Loading volume {idx}: {nii_path}")
        vol = nib.load(nii_path).get_fdata().astype(np.float32)
        H, W, D = vol.shape
        #print(f"[Data] Volume shape: {vol.shape}")
        num_chunks = D // self.proc.chunk_depth

        windows = []
        for i in range(num_chunks):
            arr = vol[:, :, i*self.proc.chunk_depth:(i+1)*self.proc.chunk_depth]
            arr = np.clip(arr, self.proc.vmin, self.proc.vmax)
            arr = (arr - self.proc.vmin) / (self.proc.vmax - self.proc.vmin)
            arr = np.clip(arr, self.proc.eps, 1.0 - self.proc.eps)

            t = torch.from_numpy(arr).permute(2, 0, 1)
            t = F.interpolate(t.unsqueeze(0), size=self.proc.out_size,
                              mode="bilinear", align_corners=False).squeeze(0)
            t = (t - 0.5) / 0.5
            windows.append(t)

        chunks = torch.stack(windows, dim=0)
        labels = row[self.labels].to_numpy(dtype=np.float32)
        mask   = (labels != -1)
        return chunks, labels, mask

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        accum_steps: int = 10,
        print_every: int = 100,
        val_every: int = 500,
        metrics_dir: str = None
    ):
        self.model = model
        self.opt = optimizer
        self.crit = criterion
        self.device = device
        self.accum = accum_steps
        self.print_every = print_every
        self.val_every = val_every
        self.global_step = 1
        
        # Set up metrics logging if directory is provided
        if metrics_dir:
            self.metrics_dir = metrics_dir
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.metrics_filename = os.path.join(
                metrics_dir, 
                f"training_metrics_nGPU_A100_{timestamp}.jsonl"
            )
    
    def _train_step(self, chunks, labels, mask):
        """Training step for one patient (with all chunks)"""
        running_loss = 0.0
        
        # Move chunks to device
        chunks = chunks.to(self.device)  # [N, 3, H, W]
        
        # Process all chunks with the base model
        features = []
        for i in range(chunks.size(0)):
            # Get CLS token embedding for this chunk
            chunk_feat = self.model.base(chunks[i].unsqueeze(0))
            features.append(chunk_feat)
        
        # Stack features from all chunks [num_chunks, feat_dim]
        features = torch.cat(features, dim=0)
        
        # Apply transformer to aggregate chunk features
        aggregated = self.model.transformer(features.unsqueeze(0))  # [1, num_chunks, feat_dim]
        
        # Global average pooling across chunks
        pooled = torch.mean(aggregated, dim=1)  # [1, feat_dim]
        
        # Apply the classifier to get outputs for each task
        logits = self.model.classifier(pooled)  # [1, num_tasks]
        
        # Convert NumPy arrays to PyTorch tensors
        target = torch.tensor(labels, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        # Apply binary cross entropy to each task output
        loss = 0.0
        for i in range(logits.size(1)):
            if mask_tensor[i]:  # Only calculate loss for non-missing values
                task_loss = self.crit(logits[0, i:i+1], target[i:i+1])
                loss += task_loss
                running_loss += task_loss.item()
        
        # Normalize loss by number of active tasks
        active_tasks = mask_tensor.sum().item()
        if active_tasks > 0:
            loss = loss / active_tasks
            
        # Gradient accumulation
        loss = loss / self.accum
        loss.backward()
        
        # Only update weights after accumulating enough gradients
        if (self.global_step) % self.accum == 0:
            self.opt.step()
            self.opt.zero_grad()
        
        if (self.global_step) % self.print_every == 0:
            print(f"Step {self.global_step} | Loss: {loss.item():.6f}")
        
        self.global_step += 1
        return running_loss / max(1, active_tasks)
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for chunks, labels, mask in val_loader:
                chunks = chunks.to(self.device)
                
                # Process all chunks with the base model
                features = []
                for i in range(chunks.size(0)):
                    # Get CLS token embedding for this chunk
                    chunk_feat = self.model.base(chunks[i].unsqueeze(0))
                    features.append(chunk_feat)
                
                # Stack features from all chunks [num_chunks, feat_dim]
                features = torch.cat(features, dim=0)
                
                # Apply transformer to aggregate chunk features
                aggregated = self.model.transformer(features.unsqueeze(0))  # [1, num_chunks, feat_dim]
                
                # Global average pooling across chunks
                pooled = torch.mean(aggregated, dim=1)  # [1, feat_dim]
                
                # Apply the classifier to get outputs for each task
                logits = self.model.classifier(pooled)  # [1, num_tasks]
                
                # Convert to float32 before going to CPU and numpy
                logits = logits.float()
                
                # Store predictions and targets
                all_preds.append(logits.cpu().numpy())
                all_targets.append(labels)
                all_masks.append(mask)
        
        # Calculate metrics for each task using scikit-learn
        metrics = {}
        
        # Combine all predictions and targets
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        masks = np.vstack(all_masks)
        
        # For each task, calculate metrics if data available
        for i in range(preds.shape[1]):
            if i >= masks.shape[1]:
                continue
                
            # Get valid indices for this task
            valid_idx = masks[:, i].astype(bool)
            
            if np.sum(valid_idx) > 0:
                task_preds = preds[valid_idx, i]
                task_targets = targets[valid_idx, i]
                
                # Calculate AUC
                try:
                    auc = roc_auc_score(task_targets, task_preds)
                    metrics[f'auc_task{i}'] = auc
                except:
                    metrics[f'auc_task{i}'] = float('nan')
        
        # Special handling for 6-year cancer prediction
        six_year_idx = 2  # Index 2 corresponds to '6-year-cancer'
        if six_year_idx < masks.shape[1] and np.any(masks[:, six_year_idx]):
            valid_idx = masks[:, six_year_idx].astype(bool)
            six_year_preds = preds[valid_idx, six_year_idx]
            six_year_targets = targets[valid_idx, six_year_idx]
            
            try:
                auc = roc_auc_score(six_year_targets, six_year_preds)
                metrics['auc_6year'] = auc
            except:
                metrics['auc_6year'] = float('nan')
        
        return metrics
    
    def save_metric_update(self, metrics_dict, metrics_type="train"):
        """Save training/validation metrics to JSONL file"""
        if not hasattr(self, 'metrics_filename'):
            return
        
        # Add type to metrics
        metrics_dict["type"] = metrics_type
        
        # Write to file as a single line
        with open(self.metrics_filename, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            epochs: int = 10,
            unfreeze_strategy: str = 'gradual'):
        """Train the model for the specified number of epochs"""
        for epoch in range(epochs):
            # Implement gradual unfreezing strategy
            if unfreeze_strategy == 'gradual':
                if epoch == 0:
                    print("Freezing base model")
                    self.model.freeze_base_model()
                elif epoch == 1:
                    print("Unfreezing last 2 blocks")
                    self.model.unfreeze_last_n_blocks(n=2)
                elif epoch == 2:
                    print("Unfreezing last 4 blocks")
                    self.model.unfreeze_last_n_blocks(n=4)
                elif epoch == 3:
                    print("Unfreezing last 6 blocks")
                    self.model.unfreeze_last_n_blocks(n=6)
                elif epoch >= 4:
                    print("Unfreezing entire base model")
                    self.model.unfreeze_base_model()
            elif unfreeze_strategy == 'none':
                print("Using frozen base model for all epochs")
                self.model.freeze_base_model()
            elif unfreeze_strategy == 'all':
                print("Using unfrozen base model for all epochs")
                self.model.unfreeze_base_model()
            
            # Train loop
            self.model.train()
            epoch_loss = 0.0
            samples_seen = 0
            
            for chunks, labels, mask in train_loader:
                step_loss = self._train_step(chunks, labels, mask)
                epoch_loss += step_loss
                samples_seen += 1
                
                # Save training metrics periodically
                if (self.global_step) % self.print_every == 0 and hasattr(self, 'metrics_filename'):
                    # Create metrics dict
                    metrics_dict = {
                        "iteration": self.global_step,
                        "epoch": epoch,
                        "lr": self.opt.param_groups[0]['lr'],
                        "accumulation_step": (self.global_step - 1) % self.accum,
                        "accum_size": self.accum,
                        "total_loss": step_loss,
                        "acc1": 0.0,  # Would need predictions for accurate metrics
                        "acc3": 0.0,
                        "acc6": 0.0,
                        "pos_count_1yr": "N/A",
                        "pos_count_3yr": "N/A",
                        "pos_count_6yr": "N/A"
                    }
                    
                    # Save metrics
                    self.save_metric_update(metrics_dict, "train")
                
                # Evaluate on validation set periodically
                if val_loader and self.global_step % self.val_every == 0:
                    self.model.eval()
                    metrics = self.evaluate(val_loader)
                    print(f"Validation | ", end="")
                    for k, v in metrics.items():
                        print(f"{k}: {v:.4f} | ", end="")
                    print()
                    
                    # Save validation metrics
                    if hasattr(self, 'metrics_filename'):
                        # Create metrics dict for validation
                        val_metrics = {
                            "iteration": self.global_step,
                            "epoch": epoch,
                            "lr": self.opt.param_groups[0]['lr'],
                            "accumulation_step": (self.global_step - 1) % self.accum,
                            "accum_size": self.accum,
                            "total_loss": 0.0,  # Would need validation loss
                            "acc1": metrics.get("auc_task0", 0.0),
                            "acc3": metrics.get("auc_task1", 0.0),
                            "acc6": metrics.get("auc_task2", 0.0) if "auc_task2" in metrics else metrics.get("auc_6year", 0.0),
                            "pos_count_1yr": "N/A",
                            "pos_count_3yr": "N/A",
                            "pos_count_6yr": "N/A"
                        }
                        
                        # Save validation metrics
                        self.save_metric_update(val_metrics, "validation")
                    
                    self.model.train()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / samples_seen
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.6f}")
            
            # End of epoch validation
            if val_loader:
                self.model.eval()
                metrics = self.evaluate(val_loader)
                print(f"Epoch {epoch+1} Validation | ", end="")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f} | ", end="")
                print()
                self.model.train()

def calculate_class_weights(df, label_cols):
    """Calculate class weights for imbalanced classification"""
    weights = []
    
    for col in label_cols:
        # Count positive and negative samples
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count
        
        # Calculate weight (higher weight for minority class)
        if pos_count > 0 and neg_count > 0:
            weight = neg_count / pos_count  # Weight for positive class
        else:
            weight = 1.0
        
        weights.append(weight)
    
    return torch.tensor(weights)
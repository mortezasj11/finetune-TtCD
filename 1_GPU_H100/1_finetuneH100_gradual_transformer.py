import sys
sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")

import os
import torch
from functools import partial
from dinov2.models.vision_transformer import vit_base, DinoVisionTransformer
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
# Initialize ViT-Base model with patch size 16 and any additional parameters

import torch
from dinov2.models.vision_transformer import vit_base
from dinov2.models.vision_transformer import DinoVisionTransformer

import numpy as np
    
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import yaml
from pathlib import Path
import json
import datetime  # Add this import at the top of the file

os.environ['PYTHONPATH'] = "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader5"
checkpoint_path = "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/output_dir/448_192_all14_17M_P32_B8/eval/training_314999/teacher_checkpoint.pth"
patch_size = 32
n_patch = 14 if patch_size == 32 else 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
print(checkpoint.keys())
teacher_weights = checkpoint["teacher"]
teacher_weights_cleaned = {k.replace("backbone.", ""): v for k, v in teacher_weights.items()}
# Assuming you have imported vit_base and are creating a model instance
model_ct = DinoVisionTransformer(
        img_size=448,
        patch_size=patch_size,
        drop_path_rate=0.0,
        block_chunks=1,
        drop_path_uniform=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=5,
        init_values = 1.0e-05,
    ).to(device)

model_ct.load_state_dict(teacher_weights_cleaned, strict=False)
#model_ct.eval()

import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========= models.py =========

class VolumeProcessor:
    def __init__(self,
                 vmin: float = -500.0,
                 vmax: float = 100.0,
                 eps: float = 1e-5,
                 out_size: Tuple[int,int] = (448, 448),
                 chunk_depth: int = 3):
        self.vmin = vmin
        self.vmax = vmax
        self.eps = eps
        self.out_size = out_size
        self.chunk_depth = chunk_depth

    def process_volume(self, vol: np.ndarray) -> torch.Tensor:
        H, W, D = vol.shape
        n_chunks = D // self.chunk_depth
        chunks = []
        for i in range(n_chunks):
            arr = vol[:, :, i*self.chunk_depth:(i+1)*self.chunk_depth]
            arr = np.clip(arr, self.vmin, self.vmax)
            arr = (arr - self.vmin) / (self.vmax - self.vmin)
            arr = np.clip(arr, self.eps, 1.0 - self.eps)
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
            t = F.interpolate(t, size=self.out_size, mode='bilinear', align_corners=False).squeeze(0)
            t = (t - 0.5) / 0.5
            chunks.append(t)
        return torch.stack(chunks, dim=0) if chunks else torch.empty(0)

class TransformerAggregator(nn.Module):
    def __init__(self,
                 chunk_feat_dim: int,
                 hidden_dim: int,
                 num_tasks: int,
                 num_attn_heads: int = 4,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        # Layer normalization before pooling
        self.norm = nn.LayerNorm(chunk_feat_dim)
        
        # Multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=chunk_feat_dim,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Learnable query for global pooling via attention
        self.query = nn.Parameter(torch.randn(1, 1, chunk_feat_dim))
        
        # Final classifier
        self.fc = nn.Linear(chunk_feat_dim, num_tasks)
    
    def forward(self, chunk_feats: torch.Tensor) -> torch.Tensor:
        # chunk_feats shape: [N, feat_dim] where N = number of chunks
        
        # Apply layer normalization before pooling (addresses the normalization across chunks request)
        chunk_feats = self.norm(chunk_feats)  # shape: [N, feat_dim]
        
        # Add an implicit batch dimension for transformer input
        x = chunk_feats.unsqueeze(0)  # shape: [1, N, feat_dim]
        
        # Pass through transformer encoder
        context = self.transformer_encoder(x)  # shape: [1, N, feat_dim]
        
        # Compute attention weights using query (attentive pooling)
        query = self.query.expand(1, 1, -1)  # shape: [1, 1, feat_dim]
        
        # Compute scaled dot-product attention
        attn_scores = torch.matmul(query, context.transpose(-2, -1)) / (chunk_feats.size(-1) ** 0.5)  # shape: [1, 1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # shape: [1, 1, N]
        
        # Apply attention weights to get weighted representation
        pooled = torch.matmul(attn_weights, context)  # shape: [1, 1, feat_dim]
        pooled = pooled.squeeze(1)  # shape: [1, feat_dim]
        
        # Final prediction
        return self.fc(pooled)  # shape: [1, num_tasks]

class CombinedModel(nn.Module):
    def __init__(self,
                 base_model: nn.Module,
                 chunk_feat_dim: int,
                 hidden_dim: int,
                 num_tasks: int,
                 num_attn_heads: int = 4,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.base = base_model
        self.agg = TransformerAggregator(
            chunk_feat_dim=chunk_feat_dim, 
            hidden_dim=hidden_dim, 
            num_tasks=num_tasks,
            num_attn_heads=num_attn_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        # Freeze base model initially
        self.freeze_base_model()

    def freeze_base_model(self):
        """Freeze all parameters in the base model"""
        for param in self.base.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_blocks(self, n=3):
        """Gradually unfreeze the last n transformer blocks of the base model"""
        # First ensure all base model parameters are frozen
        self.freeze_base_model()
        
        # DinoVisionTransformer structure has blocks as blocks attribute 
        # Unfreeze only the last n blocks
        num_blocks = len(self.base.blocks)
        # Ensure n is not larger than the number of blocks to avoid negative indices
        n = min(n, num_blocks)
        start_idx = max(0, num_blocks - n)
        
        print(f"Unfreezing {n} blocks out of {num_blocks} total blocks (from index {start_idx} to {num_blocks-1})")
        
        for i in range(start_idx, num_blocks):
            for param in self.base.blocks[i].parameters():
                param.requires_grad = True
                
        # Also unfreeze the head if needed
        if hasattr(self.base, 'head'):
            for param in self.base.head.parameters():
                param.requires_grad = True
                
        # Optional: Unfreeze any layer norms at the end
        if hasattr(self.base, 'norm'):
            for param in self.base.norm.parameters():
                param.requires_grad = True
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model"""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, 3, H, W] where N = number of chunks, H=W=448
        chunk_logits = self.base(x)  # shape: [N, feat_dim=768]
        return self.agg(chunk_logits)  # shape: [1, num_tasks=6]

# ========= dataset.py =========

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

# ========= trainer.py =========

class Trainer:
    def __init__(
        self,
        model: CombinedModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        accum_steps: int = 4,
        print_every: int = 500,
        val_every: int = 5000,
        metrics_dir: str = None,
    ):
        self.model = model.to(device)
        self.opt = optimizer
        self.crit = criterion
        self.dev = device
        self.accum = accum_steps
        self.print_every = print_every
        self.val_every = val_every
        self.metrics_dir = metrics_dir
        self.global_step = 0
        
        # Get base learning rate
        self.base_lr = self.opt.param_groups[0]['lr']
        
        # Create metrics directory if it doesn't exist
        if self.metrics_dir:
            Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate a unique filename with timestamp and include 1GPU_A100
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.metrics_filename = f"training_metrics_1GPU_A100_{timestamp}.jsonl"
            
            # Create metrics file with the unique name
            metrics_file = Path(self.metrics_dir) / self.metrics_filename
            with open(metrics_file, 'w') as f:
                f.write('')  # Just create an empty file

    def save_metric_update(self, update_dict):
        """Save a single metric update as a line in a JSONL file"""
        if self.metrics_dir:
            # Print the JSON content
            print(f"Saving metrics to JSON: {json.dumps(update_dict, indent=2)}")
            
            metrics_file = Path(self.metrics_dir) / self.metrics_filename
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(update_dict) + '\n')

    def _train_step(self, chunks, labels, mask):
        chunks = chunks.to(self.dev)  # shape: [N, 3, H, W]
        labels = torch.tensor(labels, device=self.dev)  # shape: [num_tasks=6]
        mask = torch.tensor(mask, device=self.dev)  # shape: [num_tasks=6], boolean mask
        logits = self.model(chunks)  # shape: [1, num_tasks=6]
        valid = mask.nonzero(as_tuple=True)[0]  # shape: [num_valid], indices of valid labels

        loss = 0.0
        acc1 = 0.0
        if valid.numel() > 0:
            # Get valid predictions and labels
            valid_logits = logits[0, valid]  # shape: [num_valid]
            valid_labels = labels[valid]  # shape: [num_valid]
            
            # Calculate loss using valid predictions and labels 
            valid_weights = self.crit.pos_weight[valid]  # Get weights for valid indices
            loss = F.binary_cross_entropy_with_logits(
                valid_logits, 
                valid_labels,
                pos_weight=valid_weights,
                reduction='mean'
            ) / self.accum
            
            loss.backward()
            
            # Calculate accuracy for 1-year cancer prediction if valid
            if mask[0]:
                prob = torch.sigmoid(logits[0, 0])  # scalar
                pred = (prob > 0.5).float()  # scalar
                acc1 = (pred == labels[0]).float().item()  # scalar

        # Return the real loss value (not scaled by accum steps)
        # For logging purposes, we return the actual loss value
        return loss.item() * self.accum, acc1, logits

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            epochs: int = 10,
            unfreeze_strategy: str = 'gradual'):
        """
        Train model with gradual unfreezing strategy
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of epochs
            unfreeze_strategy: Strategy for unfreezing the base model
                - 'none': Keep base model frozen throughout training
                - 'gradual': Gradually unfreeze blocks as training progresses
                - 'full_after_warmup': Completely unfreeze after initial warmup
        """
        self.global_step = 0
        
        for epoch in range(epochs):
            print(f"\n>>> Epoch {epoch+1}/{epochs}")
            
            # Apply unfreezing strategy at the beginning of each epoch
            if unfreeze_strategy == 'gradual':
                if epoch == 0:
                    # Keep base model frozen in first epoch
                    self.model.freeze_base_model()
                    print("Epoch 1: Base model is fully frozen")
                elif epoch == 1:
                    # Unfreeze just the last 2 blocks in the second epoch
                    self.model.unfreeze_last_n_blocks(n=2)
                    print("Epoch 2: Unfreezing last 2 blocks of base model")
                elif epoch >= 2:
                    # Keep the same unfreezing level (last 2 blocks) for all subsequent epochs
                    # Note: We're not unfreezing additional blocks or the entire model
                    print(f"Epoch {epoch+1}: Maintaining last 2 blocks unfrozen")
            elif unfreeze_strategy == 'full_after_warmup' and epoch == 2:
                # Unfreeze the entire base model after 2 epochs of warmup
                self.model.unfreeze_base_model()
                print(f"Epoch {epoch+1}: Base model fully unfrozen after warmup")
                
            self.model.train()
            self.opt.zero_grad()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc3 = 0.0  # Changed from acc4 to acc3
            total_acc6 = 0.0  # Added acc6
            one_count_1yr = 0
            one_count_3yr = 0  # Changed from 4yr to 3yr
            one_count_6yr = 0  # Added 6yr
            
            for step, (chunks, labels, mask) in enumerate(train_loader, 1):
                self.global_step += 1
                # Calculate accumulation step (0-based)
                accum_step = (self.global_step - 1) % self.accum
                
                true_loss, acc1, logits = self._train_step(chunks, labels, mask)
                total_loss += true_loss  # Store the real loss value
                total_acc1 += acc1
                one_count_1yr += int(labels[0] == 1)
                one_count_3yr += int(labels[2] == 1)  # Index 2 is 3-year (0-indexed)
                one_count_6yr += int(labels[5] == 1)  # Index 5 is 6-year (0-indexed)
                
                # 3-year-cancer accuracy (instead of 4-year)
                acc3 = 0.0
                if mask[2]:  # Check 3-year (index 2)
                    prob3 = torch.sigmoid(logits[0, 2])
                    pred3 = (prob3 > 0.5).float()
                    acc3 = (pred3 == labels[2]).float().item()
                total_acc3 += acc3
                
                # 6-year-cancer accuracy
                acc6 = 0.0
                if mask[5]:  # Check 6-year (index 5) 
                    prob6 = torch.sigmoid(logits[0, 5])
                    pred6 = (prob6 > 0.5).float()
                    acc6 = (pred6 == labels[5]).float().item()
                total_acc6 += acc6

                if step % self.accum == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                if step % self.print_every == 0:
                    avg_loss = total_loss / step
                    avg_acc1 = total_acc1 / step
                    avg_acc3 = total_acc3 / step
                    avg_acc6 = total_acc6 / step
                    print(f"[Train] After {step} patients: avg_loss={avg_loss:.4f}, " 
                          f"avg acc1={avg_acc1:.4f}, acc3={avg_acc3:.4f}, acc6={avg_acc6:.4f}, "
                          f"1-year count={one_count_1yr}, 3-year count={one_count_3yr}, 6-year count={one_count_6yr}")
                    
                    # Get current learning rate
                    current_lr = self.opt.param_groups[0]['lr']
                    
                    # Save metric update with the requested fields
                    metric_update = {
                        "iteration": self.global_step,
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "accumulation_step": accum_step,
                        "accum_size": self.accum,
                        "total_loss": avg_loss,
                        "acc1": avg_acc1,
                        "acc3": avg_acc3,  # Changed from acc4 to acc3
                        "acc6": avg_acc6,  # Added acc6
                        "pos_count_1yr": f"{one_count_1yr}-{step-one_count_1yr}",
                        "pos_count_3yr": f"{one_count_3yr}-{step-one_count_3yr}", # Changed from 4yr to 3yr
                        "pos_count_6yr": f"{one_count_6yr}-{step-one_count_6yr}", # Added 6yr
                        "type": "train"
                    }
                    self.save_metric_update(metric_update)

                if val_loader is not None and step % self.val_every == 0:
                    print(f"=== Interim validation at step {step} ===")
                    va_loss, va_acc1, va_acc3, va_acc6, val_pos_count_1yr, val_pos_count_3yr, val_pos_count_6yr = self.evaluate(val_loader)
                    print(f"[Interim Val] loss={va_loss:.4f}, acc1={va_acc1:.4f}, acc3={va_acc3:.4f}, acc6={va_acc6:.4f}")
                    
                    # Save validation metric update
                    val_update = {
                        "iteration": self.global_step,
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "accumulation_step": accum_step,
                        "accum_size": self.accum,
                        "total_loss": va_loss,
                        "acc1": va_acc1,
                        "acc3": va_acc3,  # Changed from acc4 to acc3
                        "acc6": va_acc6,  # Added acc6
                        "pos_count_1yr": val_pos_count_1yr,
                        "pos_count_3yr": val_pos_count_3yr,  # Changed from 4yr to 3yr
                        "pos_count_6yr": val_pos_count_6yr,  # Added 6yr
                        "type": "validation"
                    }
                    self.save_metric_update(val_update)

            if step % self.accum != 0:
                self.opt.step()
                self.opt.zero_grad()

            avg_loss = total_loss / step
            avg_acc1 = total_acc1 / step
            avg_acc3 = total_acc3 / step
            avg_acc6 = total_acc6 / step
            print(f"=== Epoch {epoch+1} complete: avg loss={avg_loss:.4f}, "
                  f"avg acc1={avg_acc1:.4f}, acc3={avg_acc3:.4f}, acc6={avg_acc6:.4f}, "
                  f"1-year count={one_count_1yr}, 3-year count={one_count_3yr}, 6-year count={one_count_6yr} ===")

            logging.info(f"Epoch {epoch+1}/{epochs} — train_loss={avg_loss:.4f}, train_acc1={avg_acc1:.4f}, train_acc3={avg_acc3:.4f}, train_acc6={avg_acc6:.4f}")
            
            # Save end-of-epoch metric update with the requested fields
            current_lr = self.opt.param_groups[0]['lr']
            epoch_update = {
                "iteration": self.global_step,
                "epoch": epoch + 1,
                "lr": current_lr,
                "accumulation_step": -1,  # -1 indicates end of epoch
                "accum_size": self.accum,
                "total_loss": avg_loss,
                "acc1": avg_acc1,
                "acc3": avg_acc3,  # Changed from acc4 to acc3
                "acc6": avg_acc6,  # Added acc6
                "pos_count_1yr": f"{one_count_1yr}-{step-one_count_1yr}",
                "pos_count_3yr": f"{one_count_3yr}-{step-one_count_3yr}", # Changed from 4yr to 3yr
                "pos_count_6yr": f"{one_count_6yr}-{step-one_count_6yr}", # Added 6yr
                "type": "train_epoch_end"
            }
            self.save_metric_update(epoch_update)

            if val_loader is not None:
                print("=== Starting final validation ===")
                va_loss, va_acc1, va_acc3, va_acc6, val_pos_count_1yr, val_pos_count_3yr, val_pos_count_6yr = self.evaluate(val_loader)
                print(f"=== Validation complete: loss={va_loss:.4f}, acc1={va_acc1:.4f}, acc3={va_acc3:.4f}, acc6={va_acc6:.4f} ===")
                logging.info(f"Validation loss={va_loss:.4f}, val_acc1={va_acc1:.4f}, val_acc3={va_acc3:.4f}, val_acc6={va_acc6:.4f}")
                
                # Save end-of-epoch validation metric update
                val_epoch_update = {
                    "iteration": self.global_step,
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "accumulation_step": -1,  # -1 indicates end of epoch
                    "accum_size": self.accum,
                    "total_loss": va_loss,
                    "acc1": va_acc1,
                    "acc3": va_acc3,  # Changed from acc4 to acc3
                    "acc6": va_acc6,  # Added acc6
                    "pos_count_1yr": val_pos_count_1yr,
                    "pos_count_3yr": val_pos_count_3yr,  # Changed from 4yr to 3yr
                    "pos_count_6yr": val_pos_count_6yr,  # Added 6yr
                    "type": "validation_epoch_end"
                }
                self.save_metric_update(val_epoch_update)

    def evaluate(self, loader: DataLoader):
        """
        Evaluate on the entire validation dataset.
        Reports metrics for 1-year, 3-year, and 6-year predictions.
        """
        self.model.eval()
        print("--- Evaluating on the entire validation dataset ---")

        total_loss = 0.0
        total_acc1 = 0.0
        total_acc3 = 0.0  # Changed from acc4 to acc3
        total_acc6 = 0.0  # Added acc6
        one_count_1yr = 0
        zero_count_1yr = 0
        one_count_3yr = 0  # Changed from 4yr to 3yr
        zero_count_3yr = 0
        one_count_6yr = 0  # Added 6yr
        zero_count_6yr = 0

        with torch.no_grad():
            for idx, (chunks, labels, mask) in enumerate(loader, 1):
                chunks = chunks.to(self.dev)
                labels = torch.tensor(labels, device=self.dev)
                mask = torch.tensor(mask, device=self.dev)
                logits = self.model(chunks)

                # 1-year cancer accuracy
                if mask[0]:
                    prob1 = torch.sigmoid(logits[0, 0])
                    pred1 = (prob1 > 0.5).float()
                    acc1 = (pred1 == labels[0]).float().item()
                    total_acc1 += acc1
                    if labels[0].item() == 1:
                        one_count_1yr += 1
                    else:
                        zero_count_1yr += 1
                
                # 3-year cancer accuracy (instead of 4-year)
                if mask[2]:  # Index 2 is 3-year
                    prob3 = torch.sigmoid(logits[0, 2])
                    pred3 = (prob3 > 0.5).float()
                    acc3 = (pred3 == labels[2]).float().item()
                    total_acc3 += acc3
                    if labels[2].item() == 1:
                        one_count_3yr += 1
                    else:
                        zero_count_3yr += 1
                
                # 6-year cancer accuracy
                if mask[5]:  # Index 5 is 6-year
                    prob6 = torch.sigmoid(logits[0, 5])
                    pred6 = (prob6 > 0.5).float()
                    acc6 = (pred6 == labels[5]).float().item()
                    total_acc6 += acc6
                    if labels[5].item() == 1:
                        one_count_6yr += 1
                    else:
                        zero_count_6yr += 1

                # Loss calculation
                valid = mask.nonzero(as_tuple=True)[0]
                if valid.numel():
                    loss = self.crit(logits[0, valid], labels[valid])
                    total_loss += loss.item()

        # Calculate final metrics
        samples = idx
        avg_loss = total_loss / samples
        avg_acc1 = total_acc1 / max(1, (one_count_1yr + zero_count_1yr))
        avg_acc3 = total_acc3 / max(1, (one_count_3yr + zero_count_3yr))
        avg_acc6 = total_acc6 / max(1, (one_count_6yr + zero_count_6yr))
        
        # Format counts as requested (pos-neg)
        pos_count_1yr_str = f"{one_count_1yr}-{zero_count_1yr}"
        pos_count_3yr_str = f"{one_count_3yr}-{zero_count_3yr}"
        pos_count_6yr_str = f"{one_count_6yr}-{zero_count_6yr}"
        
        print(f"--- Validation metrics (total {samples} samples) ---")
        print(f"Loss: {avg_loss:.4f}")
        print(f"1-year accuracy: {avg_acc1:.4f} (pos-neg: {pos_count_1yr_str})")
        print(f"3-year accuracy: {avg_acc3:.4f} (pos-neg: {pos_count_3yr_str})")
        print(f"6-year accuracy: {avg_acc6:.4f} (pos-neg: {pos_count_6yr_str})")
        
        return avg_loss, avg_acc1, avg_acc3, avg_acc6, pos_count_1yr_str, pos_count_3yr_str, pos_count_6yr_str


# ========= run.py =========

def build_base_model(chunk_feat_dim: int, num_tasks: int) -> nn.Module:
    raise NotImplementedError("Please implement or import your chunk-level model.")

def calculate_class_weights(df: pd.DataFrame, label_cols: List[str]) -> torch.Tensor:
    """
    Calculate class weights based on inverse class frequencies.
    Returns weights for positive class (1) for each time point.
    """
    weights = []
    for col in label_cols:
        # Count number of positive samples (1) and total valid samples (0 or 1)
        pos_count = (df[col] == 1).sum()
        total_valid = ((df[col] == 0) | (df[col] == 1)).sum()
        
        # Calculate weight for positive class
        if pos_count > 0 and total_valid > 0:
            weight = total_valid / (2.0 * pos_count)  # 2.0 to balance between classes
        else:
            weight = 1.0  # default weight if no positive samples
            
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)

def main(args):
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(args.csv)
    df = df[df.file_path.notna()].reset_index(drop=True)
    train_df = df[df.split == 'train']
    val_df   = df[df.split == 'val'] if 'val' in df.split.unique() else None

    # Calculate class weights
    class_weights = calculate_class_weights(train_df, args.label_cols)
    print(f"Class weights: {class_weights}")

    processor = VolumeProcessor(
        vmin=args.vmin, vmax=args.vmax,
        eps=args.eps, out_size=(args.size, args.size),
        chunk_depth=args.chunk_depth
    )

    train_ds = NLSTDataset(train_df, processor, args.label_cols)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda b: b[0])
    val_loader = None
    if val_df is not None:
        val_df = val_df[val_df.file_path.notna()].reset_index(drop=True)
        val_ds = NLSTDataset(val_df, processor, args.label_cols)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, collate_fn=lambda b: b[0])

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    base_model = model_ct 
    model = CombinedModel(
        base_model,
        chunk_feat_dim=args.chunk_feat_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(args.label_cols),
        num_attn_heads=args.num_attn_heads,
        num_layers=args.num_transformer_layers,
        dropout_rate=args.dropout
    )
    
    # Create separate parameter groups with different learning rates
    vit_params = []
    aggregator_params = []
    
    # Collect base model parameters (ViT)
    for name, param in model.base.named_parameters():
        if param.requires_grad:
            vit_params.append(param)
            
    # Collect aggregator parameters
    for name, param in model.agg.named_parameters():
        if param.requires_grad:
            aggregator_params.append(param)
    
    # Configure parameter groups with different learning rates
    param_groups = [
        {'params': vit_params, 'lr': args.vit_lr},
        {'params': aggregator_params, 'lr': args.agg_lr}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    # Create metrics directory
    metrics_dir = "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_single_gpu"
    os.makedirs(metrics_dir, exist_ok=True)
    
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        accum_steps=args.accum_steps,
        print_every=1000,
        val_every=10000,
        metrics_dir=metrics_dir
    )
    
    # Print where metrics will be saved
    print(f"Metrics will be saved to: {os.path.join(metrics_dir, trainer.metrics_filename)}")
    
    trainer.fit(train_loader, val_loader, 
                epochs=args.epochs, 
                unfreeze_strategy=args.unfreeze_strategy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv")
    parser.add_argument('--label-cols', nargs='+', default=[
        '1-year-cancer','2-year-cancer','3-year-cancer',
        '4-year-cancer','5-year-cancer','6-year-cancer'
    ])
    parser.add_argument('--vmin', type=float, default=-500.0)
    parser.add_argument('--vmax', type=float, default=100.0)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--size', type=int, default=448)
    parser.add_argument('--chunk-depth', type=int, default=3)
    parser.add_argument('--chunk-feat-dim', type=int, default=768)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--vit-lr', type=float, default=1e-6)
    parser.add_argument('--agg-lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-attn-heads', type=int, default=4)
    parser.add_argument('--num-transformer-layers', type=int, default=2)
    parser.add_argument('--unfreeze-strategy', type=str, default='gradual', 
                      choices=['none', 'gradual', 'full_after_warmup'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda-id', type=int, default=0, help='CUDA device ID (0, 1, 2, or 3)')
    args, _ = parser.parse_known_args()
    
    # Set the specific CUDA device if specified
    if args.device == 'cuda':
        args.device = f'cuda:{args.cuda_id}'
        print(f"Using CUDA device: {args.cuda_id}")
    
    main(args) 
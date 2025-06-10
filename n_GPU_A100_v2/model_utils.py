# File: n_GPU_A100/model_utils.py

import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import json
import time
import subprocess
from torch.utils.data import WeightedRandomSampler
from torchmetrics.functional import auroc
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Callable
import torchvision.transforms as T
import sys
sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")
from dinov2.models.vision_transformer import DinoVisionTransformer

def install_packages():
    commands = [
        ["pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath", "xformers==0.0.18", "submitit","numpy<2.0"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
        ["pip", "install", "mmcv-full==1.5.0"]
    ]
    for i, command in enumerate(commands):
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def prepare_balanced_validation(csv_path):
    """Prepare a balanced validation set from the original data"""
    df = pd.read_csv(csv_path)
    
    # Print original validation set size
    print(f"Original validation set size: {df[df['split']=='val'].shape}")

    val_df = df[df.split == 'val'].copy()
    val_df = val_df[val_df.file_path.notna()].reset_index(drop=True)

    # Get positive and negative samples for 6-year-cancer
    pos_samples = val_df[val_df['6-year-cancer'] == 1]
    neg_samples = val_df[val_df['6-year-cancer'] == 0]

    print("\nBefore balancing:")
    print(f"Positive samples: {len(pos_samples)}")
    print(f"Negative samples: {len(neg_samples)}")

    # Calculate sizes for balanced set
    n_pos = len(pos_samples)
    n_neg = len(neg_samples)
    target_size = min(n_pos, n_neg)

    # Sample equally from positive and negative
    balanced_pos = pos_samples.sample(n=target_size, random_state=42)
    balanced_neg = neg_samples.sample(n=target_size, random_state=42)

    # Combine balanced samples
    balanced_val_df = pd.concat([balanced_pos, balanced_neg]).reset_index(drop=True)

    print("\nAfter balancing:")
    print(f"Balanced validation set shape: {balanced_val_df.shape}")

    # Create new dataframe correctly
    df_new = df.copy()
    # Remove all validation samples
    df_new = df_new[df_new.split != 'val']
    # Add balanced validation samples
    balanced_val_df['split'] = 'val'  # Ensure split column is set
    df_new = pd.concat([df_new, balanced_val_df], ignore_index=True)

    print("\nFinal shapes:")
    print(f"New validation set shape: {df_new[df_new['split']=='val'].shape}")
    print(f"Total dataset shape: {df_new.shape}")

    # Save the balanced validation set
    temp_csv_path = csv_path.replace('.csv', '_balanced.csv')
    df_new.to_csv(temp_csv_path, index=False)
    
    return temp_csv_path

def calculate_auc(predictions, targets, mask=None):
    """Calculate AUC score for binary classification"""
    if mask is not None:
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return float('nan')
        predictions = predictions[valid_indices]
        targets = targets[valid_indices]
    
    # Ensure we have both classes present
    if np.all(targets == 1) or np.all(targets == 0):
        return float('nan')
    
    try:
        return roc_auc_score(targets, predictions)
    except:
        return float('nan')

class MetricsLogger:
    def __init__(self, rank: int, metrics_dir: str = None):
        self.rank = rank
        self.should_save_metrics = (rank == 0 and metrics_dir is not None)
        
        if self.should_save_metrics:
            self.metrics_dir = metrics_dir
            os.makedirs(self.metrics_dir, exist_ok=True)
            
            # Create metrics filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.metrics_filename = os.path.join(
                self.metrics_dir, 
                f"training_metrics_nGPU_DDP_{timestamp}.jsonl"
            )
            
            print(f"Will save metrics to: {self.metrics_filename}")
    
    def log_metrics(self, metrics_dict: Dict[str, Any]):
        """Log metrics to file and console"""
        if not self.should_save_metrics:
            return
            
        # Log to console for both training and validation metrics
        log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                            for k, v in metrics_dict.items()])
        import logging
        logging.info(log_str)
        
        # Always save to file
        try:
            with open(self.metrics_filename, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
        except Exception as e:
            print(f"ERROR saving metrics: {e}")

class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.samples_seen = 0
        
        # Accuracy tracking
        self.correct_1yr = 0
        self.correct_3yr = 0
        self.correct_6yr = 0
        
        # Sample counting
        self.total_1yr_samples = 0
        self.total_3yr_samples = 0
        self.total_6yr_samples = 0
        
        # Positive sample tracking
        self.pos_1yr = 0
        self.pos_3yr = 0
        self.pos_6yr = 0
    
    def update_metrics(self, predictions, targets, mask, loss):
        """Update metrics for a single batch"""
        # Update loss
        self.total_loss += loss
        self.samples_seen += 1
        
        # Update accuracy and counts for each time point
        if mask[0]:  # 1-year
            is_correct = (predictions[0] == targets[0]).float().item()
            self.correct_1yr += is_correct
            self.total_1yr_samples += 1
            self.pos_1yr += int(targets[0].item() == 1)
            
        if mask[1]:  # 3-year
            is_correct = (predictions[1] == targets[1]).float().item()
            self.correct_3yr += is_correct
            self.total_3yr_samples += 1
            self.pos_3yr += int(targets[1].item() == 1)
            
        if mask[2]:  # 6-year
            is_correct = (predictions[2] == targets[2]).float().item()
            self.correct_6yr += is_correct
            self.total_6yr_samples += 1
            self.pos_6yr += int(targets[2].item() == 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return current metrics"""
        metrics = {
            "total_loss": self.total_loss / max(1, self.samples_seen),
            "acc1": self.correct_1yr / max(1, self.total_1yr_samples),
            "acc3": self.correct_3yr / max(1, self.total_3yr_samples),
            "acc6": self.correct_6yr / max(1, self.total_6yr_samples),
            "pos_count_1yr": f"{self.pos_1yr}-{self.total_1yr_samples - self.pos_1yr}",
            "pos_count_3yr": f"{self.pos_3yr}-{self.total_3yr_samples - self.pos_3yr}",
            "pos_count_6yr": f"{self.pos_6yr}-{self.total_6yr_samples - self.pos_6yr}"
        }
        return metrics

class CombinedModel(nn.Module):
    def __init__(
        self,
        base_model,
        chunk_feat_dim: int = 768,
        hidden_dim: int = 1024,
        num_tasks: int = 1,
        num_attn_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        lse_tau: float = 1.0,
    ):
        super().__init__()
        self.base = base_model      # now parameters of the base model are referenced
        self.lse_tau = lse_tau
        self.chunk_feat_dim = chunk_feat_dim + 3   # 768 + 3 = 771

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.chunk_feat_dim,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(self.chunk_feat_dim, num_tasks)

        # ─── shorter _augment_feats ─────────────────────────────────────────────
    def _augment_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.training:                                   # keep eval intact
            return feats
        S, dev = feats.size(0), feats.device
        if torch.rand(1, device=dev) < 0.5:                     # reverse
            feats = feats.flip(0)
        if S > 1 and torch.rand(1, device=dev) < 0.9:           # circular roll
            feats = torch.roll(feats, torch.randint(1, S, (1,), device=dev).item(), 0)
        if torch.rand(1, device=dev) < 0.1:                     # random shuffle
            feats = feats[torch.randperm(S, device=dev)]
        return feats


    #@torch.no_grad()          # remove this decorator if you want the base to train
    def _chunk_embed(self, chunk):
        return self.base(chunk.unsqueeze(0))       # → [1, 768]

    def forward(self, x: torch.Tensor, spacing=(1.0, 1.0, 1.0)):
        """
        x:       [S, C, H, W]  – S chunks for a single patient
        spacing: (dx, dy, dz)  – voxel size in mm (or any scale units)
        """
        S = x.size(0)
        device = x.device
        dtype  = x.dtype
        feats = torch.cat([self._chunk_embed(x[i]) for i in range(S)], dim=0)  # [S, 768]
        spacing_vec = torch.tensor(spacing, dtype=dtype, device=device).expand(S, 3)
        feats = torch.cat([feats, spacing_vec], dim=1)                        # [S, 771]
        feats = self._augment_feats(feats)                    # optional reorder
        encoded = self.transformer(feats.unsqueeze(0))  # [1, S, 771]
        pooled = self.lse_tau * torch.logsumexp(encoded / self.lse_tau, dim=1)  # [1, 771]
        return self.classifier(pooled)       # [1, num_tasks]

    
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
    def __init__(self, chunk_depth=3, out_size=(448, 448), vmin=-1000, vmax=150, eps=0.00005):
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
        volume = (volume - volume.min()) / (volume.max() - volume.min() + self.eps)
        
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

augment_3_channel = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10),           # ±10°
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
])

class NLSTDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 processor: VolumeProcessor,
                 label_cols: List[str],
                 augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        df = df[df.file_path.notna()].reset_index(drop=True)
        self.df = df
        self.proc = processor
        self.labels = label_cols
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int, from_rsrch1: bool = True):
        row = self.df.iloc[idx]  # Shape: pandas.Series (1D row data)
        nii_path = row.file_path  # Shape: str (file path)
        if from_rsrch1:
            old = "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Batchs_Nii/Vols/NLST"
            new = "/rsrch1/ip/msalehjahromi/data/NLST"
            nii_path = nii_path.replace(old, new)
        #print(f"[Data] Loading volume {idx}: {nii_path}")
        nii = nib.load(nii_path)
        vol = nii.get_fdata().astype(np.float32)  # Shape: [H, W, D] 3D volume
        spacing = nii.header.get_zooms()  # Get voxel spacing (dx, dy, dz)
        H, W, D = vol.shape
        #print(f"[Data] Volume shape: {vol.shape}")
        num_chunks = D // self.proc.chunk_depth  # Shape: scalar (number of chunks)

        windows = []  # Will collect chunks
        for i in range(num_chunks):
            arr = vol[:, :, i*self.proc.chunk_depth:(i+1)*self.proc.chunk_depth]  # Shape: [H, W, chunk_depth]
            arr = np.clip(arr, self.proc.vmin, self.proc.vmax)  # Shape: [H, W, chunk_depth]
            arr = (arr - self.proc.vmin) / (self.proc.vmax - self.proc.vmin)  # Shape: [H, W, chunk_depth]
            arr = np.clip(arr, self.proc.eps, 1.0 - self.proc.eps)  # Shape: [H, W, chunk_depth]

            t = torch.from_numpy(arr).permute(2, 0, 1)  # Shape: [chunk_depth, H, W]
            t = F.interpolate(t.unsqueeze(0), size=self.proc.out_size,
                              mode="bilinear", align_corners=False).squeeze(0)  # Shape: [chunk_depth, 448, 448]
            t = (t - 0.5) / 0.5  # Shape: [chunk_depth, 448, 448]
            windows.append(t)  # Add to list

        if self.augment is not None:
            windows = [self.augment(c) for c in windows]
        chunks = torch.stack(windows, dim=0)  # Shape: [num_chunks, chunk_depth, 448, 448]
        labels = row[self.labels].to_numpy(dtype=np.float32)  # Shape: [num_labels] (e.g., [3] for 3 time points)
        mask = (labels != -1)  # Shape: [num_labels] boolean mask
        return chunks, labels, mask, spacing  # Return shapes: [num_chunks, chunk_depth, 448, 448], [num_labels], [num_labels], (dx, dy, dz)

def calculate_class_weights(df, label_cols):
    """Calculate class weights for imbalanced binary labels (only 0 vs 1, ignore -1)."""
    weights = []
    for col in label_cols:
        # Only keep 0/1 entries
        vals = df[col].isin([0, 1])
        pos_count = (df.loc[vals, col] == 1).sum()
        neg_count = (df.loc[vals, col] == 0).sum()
        # If both classes exist, weight positive by neg/pos; else fallback to 1.0
        if pos_count > 0 and neg_count > 0:
            w = neg_count / pos_count
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def print_dataset_statistics(df, label_cols, dataset_name="Dataset"):
    """
    Print detailed statistics about a dataset including class distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        label_cols (List[str]): List of column names for prediction tasks
        dataset_name (str): Name to display for this dataset
    """
    print(f"\n====== {dataset_name} Statistics ======")
    print(f"Total samples: {len(df)}")
    
    # Check if 'split' column exists
    if 'split' in df.columns:
        for split_name in df['split'].unique():
            split_count = (df['split'] == split_name).sum()
            print(f"  '{split_name}' split: {split_count} samples ({100 * split_count / len(df):.1f}%)")
    
    # Calculate class distribution for each time point
    print(f"\nClass distribution in {dataset_name}:")
    for col in label_cols:
        pos = (df[col] == 1).sum()
        neg = (df[col] == 0).sum()
        missing = (df[col] == -1).sum()
        invalid = len(df) - pos - neg - missing
        total_valid = pos + neg
        
        if total_valid > 0:
            pos_ratio = 100 * pos / total_valid
            print(f"  {col}:")
            print(f"    Positive: {pos} ({pos_ratio:.1f}%)")
            print(f"    Negative: {neg} ({100 - pos_ratio:.1f}%)")
            if missing > 0:
                print(f"    Missing: {missing} ({100 * missing / len(df):.1f}% of total)")
            if invalid > 0:
                print(f"    Invalid values: {invalid}")
        else:
            print(f"  {col}: No valid samples")
    
    # Check for file_path if present
    if 'file_path' in df.columns:
        missing_paths = df['file_path'].isna().sum()
        if missing_paths > 0:
            print(f"\nWarning: {missing_paths} samples ({100 * missing_paths / len(df):.1f}%) have missing file paths")
    
    # Volume information if available
    if 'vol_shape' in df.columns:
        print("\nVolume statistics:")
        try:
            # Extract volume shapes and calculate statistics
            shapes = df['vol_shape'].dropna().tolist()
            if shapes:
                print(f"  Number of volumes with shape info: {len(shapes)}")
                # Add any specific volume analysis you need
        except:
            print("  Could not analyze volume shapes")
    
    print("=" * (23 + len(dataset_name)))
    return

def analyze_datasets(csv_path, label_cols):
    """
    Analyze training and validation datasets from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        label_cols (List[str]): List of column names for prediction tasks
    """
    try:
        import pandas as pd
        
        # Read the full dataset
        full_df = pd.read_csv(csv_path)
        print_dataset_statistics(full_df, label_cols, "Full Dataset")
        
        # Analyze training set
        if 'split' in full_df.columns:
            train_df = full_df[full_df['split'] == 'train']
            if len(train_df) > 0:
                print_dataset_statistics(train_df, label_cols, "Training Set")
            
            # Analyze validation set
            val_df = full_df[full_df['split'] == 'val']
            if len(val_df) > 0:
                print_dataset_statistics(val_df, label_cols, "Validation Set")
                
            # Analyze test set if exists
            test_df = full_df[full_df['split'] == 'test']
            if len(test_df) > 0:
                print_dataset_statistics(test_df, label_cols, "Test Set")
        
        # Calculate imbalance ratios for potential class weights
        print("\n===== Class Weight Recommendations =====")
        for col in label_cols:
            pos = (full_df[full_df['split'] == 'train'][col] == 1).sum()
            neg = (full_df[full_df['split'] == 'train'][col] == 0).sum()
            if pos > 0 and neg > 0:
                weight = neg / pos
                print(f"  {col}: Recommended pos_weight = {weight:.2f}")
        print("======================================")
        
    except Exception as e:
        print(f"Error analyzing datasets: {e}")
        import traceback
        traceback.print_exc()

class ModelSaver:
    """
    Utility class to handle model saving and checkpointing.
    
    This class handles:
    1. Saving regular checkpoints at specified intervals
    2. Saving separate components of a model (base model and aggregator)
    3. Saving metadata about the checkpoint
    """
    
    def __init__(self, output_dir, rank=0):
        """
        Initialize the model saver.
        
        Args:
            output_dir (str): Directory to save checkpoints to
            rank (int): Process rank (only rank 0 will save)
        """
        self.rank = rank
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Will save checkpoints to {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, epoch, global_step, metadata=None, is_final=False, save_components=True):
        """
        Save model checkpoint.
        
        Args:
            model: The model to save (should be a DDP wrapped model)
            epoch (int): Current epoch
            global_step (int): Current global step
            metadata (dict, optional): Additional metadata to save
            is_final (bool): Whether this is the final checkpoint
            save_components (bool): Whether to save model components separately
        """
        if self.rank != 0:
            return
        
        # Determine checkpoint filename
        if is_final:
            checkpoint_name = 'model_final.pt'
        else:
            checkpoint_name = f'model_ep{epoch}_it{global_step}.pt'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save the combined model
        torch.save(
            model.module.state_dict(), 
            checkpoint_path
        )
        
        # Save components separately if requested
        if save_components:
            self._save_model_components(model, epoch, global_step, is_final)
        
        # Save metadata
        self._save_metadata(model, epoch, global_step, metadata, is_final)
        
        print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
    def _save_model_components(self, model, epoch, global_step, is_final=False):
        """
        Save base model and aggregator components separately.
        
        Args:
            model: The model to save components of
            epoch (int): Current epoch
            is_final (bool): Whether this is the final checkpoint
        """
        # Extract state dictionaries
        combined_state_dict = model.module.state_dict()
        
        # Create separate state dictionaries
        base_state_dict = {}
        aggregator_state_dict = {}
        
        for name, param in combined_state_dict.items():
            if name.startswith('base.'):
                base_state_dict[name.replace('base.', '')] = param
            else:
                aggregator_state_dict[name] = param
        
        # Determine component checkpoint filenames
        if is_final:
            base_checkpoint_name = 'base_model_final.pt'
            aggregator_checkpoint_name = 'aggregator_final.pt'
        else:
            base_checkpoint_name       = f'base_ep{epoch}_it{global_step}.pt'
            aggregator_checkpoint_name = f'aggregator_ep{epoch}_it{global_step}.pt'
        
        # Save base model
        torch.save(
            base_state_dict, 
            os.path.join(self.checkpoint_dir, base_checkpoint_name)
        )
        
        # Save aggregator
        torch.save(
            aggregator_state_dict, 
            os.path.join(self.checkpoint_dir, aggregator_checkpoint_name)
        )
        
        print(f"Saved model components to:")
        print(f"  Base: {base_checkpoint_name}")
        print(f"  Aggregator: {aggregator_checkpoint_name}")
    
    def _save_metadata(self, model, epoch, global_step, extra_metadata=None, is_final=False):
        """
        Save metadata about the checkpoint.
        
        Args:
            model: The model
            epoch (int): Current epoch
            global_step (int): Current global step
            extra_metadata (dict, optional): Additional metadata to save
            is_final (bool): Whether this is the final checkpoint
        """
        # Create basic metadata
        metadata = {
            'epoch': epoch,
            'global_step': global_step,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'is_final': is_final,
        }
        
        # Add model configuration if available
        if hasattr(model.module, 'chunk_feat_dim'):
            metadata['model_config'] = {
                'chunk_feat_dim': getattr(model.module, 'chunk_feat_dim', 768),
                'hidden_dim': 1024,  # Use default value if attribute not present
                'num_tasks': getattr(model.module, 'classifier').out_features if hasattr(model.module, 'classifier') else 6,
            }
            
            # Add transformer configuration if available
            if hasattr(model.module, 'transformer') and hasattr(model.module.transformer, 'layers'):
                transformer = model.module.transformer
                metadata['model_config'].update({
                    'num_attn_heads': transformer.layers[0].self_attn.num_heads if hasattr(transformer.layers[0], 'self_attn') else 8,
                    'num_layers': len(transformer.layers),
                    'dropout_rate': transformer.layers[0].dropout.p if hasattr(transformer.layers[0], 'dropout') else 0.2,
                })
        
        # Add extra metadata if provided
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Save metadata
        metadata_filename = f'metadata_{"final" if is_final else f"epoch_{epoch}"}.json'
        with open(os.path.join(self.checkpoint_dir, metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

def _latest(path, pat=r"model_epoch_(\d+)\.pt"):
    files = [f for f in os.listdir(path) if re.match(pat, f)]
    if not files: return None
    latest = max(files, key=lambda f: int(re.match(pat, f).group(1)))
    return os.path.join(path, latest), int(re.search(pat, latest).group(1))

def load_latest_checkpoint(model, out_dir, device="cpu", rank=0):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        if rank==0: print(f"[resume] no {ckpt_dir} — fresh run")
        return None
    ckpt_path, epoch = _latest(ckpt_dir) or (None, None)
    if ckpt_path is None:
        if rank==0: print(f"[resume] no model_epoch_*.pt — fresh run")
        return None
    if rank==0: print(f"[resume] Loading checkpoint {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    tgt = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    tgt.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd, strict=False)
    return epoch        # caller can log/decide what to do next

def make_balanced_sampler(dataset, label_col="1-year-cancer"):
    """
    Returns a WeightedRandomSampler that draws 0 / 1 for `label_col`
    with equal chance.  -1 rows are ignored.  Works on a DataFrame
    or on an NLSTDataset (uses its .df).
    """
    df = dataset.df if hasattr(dataset, "df") else dataset           # ①
    msk = df[label_col] != -1                                        # valid rows
    y   = df.loc[msk, label_col].to_numpy()
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    w0, w1 = 1.0 / n0, 1.0 / n1                                      # ②
    weights = np.zeros(len(df), dtype=np.float32)
    weights[msk] = np.where(y == 1, w1, w0)                          # ③
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)

def print_training_dataset_stats(csv_path, label_cols):
    """
    Print basic dataset statistics for training script.
    
    Args:
        csv_path (str): Path to the CSV file
        label_cols (List[str]): List of column names for prediction tasks
    """
    try:
        # Load the full dataframe
        full_df = pd.read_csv(csv_path)
        train_df = full_df[full_df['split'] == 'train']
        val_df_stats = full_df[full_df['split'] == 'val']
        
        print("\n==== Dataset Statistics ====")
        print(f"Total samples: {len(full_df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df_stats)}")
        
        # Print class distribution for each label column
        for col in label_cols:
            if col in train_df.columns:
                # Use basic sum() to count positive samples
                pos_train = sum(train_df[col] == 1)
                pos_val = sum(val_df_stats[col] == 1)
                
                # Calculate percentages
                train_pct = 100 * pos_train / len(train_df) if len(train_df) > 0 else 0
                val_pct = 100 * pos_val / len(val_df_stats) if len(val_df_stats) > 0 else 0
                
                print(f"\n{col}:")
                print(f"  Train positive: {pos_train} ({train_pct:.2f}%)")
                print(f"  Val positive: {pos_val} ({val_pct:.2f}%)")
        print("============================\n")
        
    except Exception as e:
        print(f"Error printing dataset statistics: {e}")

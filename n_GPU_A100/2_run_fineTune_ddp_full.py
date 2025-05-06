# File: n_GPU_A100/2_run_fineTune_ddp_full.py

import subprocess
import os
import logging
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import time
import json

# Import utilities from local model_utils.py file
from model_utils import (
    DinoVisionTransformer, 
    CombinedModel, 
    VolumeProcessor, 
    NLSTDataset, 
    Trainer, 
    calculate_class_weights
)

print("Files in the directory:", os.listdir("/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU"))

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


# metrics_logger.py
import logging
import json
import os
import time
from typing import Dict, Any

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
            
        # Log to console
        log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                             for k, v in metrics_dict.items()])
        logging.info(log_str)
        
        # Save to file
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


# Modified Trainer class that supports DDP with full model copies and gradient accumulation
class DDPFullTrainer:
    def __init__(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        accum_steps: int = 10,
        print_every: int = 100,
        val_every: int = 500,
        rank: int = 0,
        world_size: int = 1,
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
        self.rank = rank
        self.world_size = world_size
        
        # Initialize metrics components
        self.metrics_logger = MetricsLogger(rank, metrics_dir)
        self.metrics_calculator = MetricsCalculator()
    
    def _train_step(self, chunks, labels, mask):
        """Single training step"""
        chunks = chunks.squeeze(1).to(self.device)
        
        # Apply max chunks limit
        max_chunks = 8
        if chunks.size(0) > max_chunks:
            mid_idx = chunks.size(0) // 2
            start_idx = max(0, mid_idx - max_chunks // 2)
            end_idx = min(chunks.size(0), start_idx + max_chunks)
            chunks = chunks[start_idx:end_idx]
        
        # Forward pass
        logits = self.model(chunks)
        
        # Convert to tensors
        target = torch.tensor(labels, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        # Calculate loss
        loss = 0.0
        for i in range(logits.size(1)):
            if mask_tensor[i]:
                task_loss = self.crit(logits[0, i:i+1], target[i:i+1])
                loss += task_loss
        
        # Normalize loss
        active_tasks = mask_tensor.sum().item()
        if active_tasks > 0:
            loss = loss / active_tasks
        
        # Gradient accumulation
        loss = loss / self.accum
        loss.backward()
        
        # Update weights
        if (self.global_step) % self.accum == 0:
            self.opt.step()
            self.opt.zero_grad()
        
        # Update metrics
        predictions = (torch.sigmoid(logits) > 0.5).float()
        self.metrics_calculator.update_metrics(predictions[0], target, mask, loss.item())
        
        # Log metrics if needed
        if self.rank == 0 and (self.global_step) % self.print_every == 0:
            metrics = self.metrics_calculator.get_metrics()
            metrics.update({
                "iteration": self.global_step,
                "epoch": self.current_epoch,
                "lr": self.opt.param_groups[0]['lr'],
                "accumulation_step": (self.global_step - 1) % self.accum,
                "accum_size": self.accum,
                "type": "train"
            })
            self.metrics_logger.log_metrics(metrics)
        
        self.global_step += 1
        return loss.item()
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        running_loss = 0.0
        samples = 0
        
        # For validation metrics accuracy
        total_acc1 = 0.0
        total_acc3 = 0.0
        total_acc6 = 0.0
        one_count_1yr = 0
        zero_count_1yr = 0
        one_count_3yr = 0
        zero_count_3yr = 0
        one_count_6yr = 0
        zero_count_6yr = 0
        
        with torch.no_grad():
            for chunks, labels, mask in val_loader:
                # Process batch as in _train_step
                chunks = chunks.squeeze(1).to(self.device)
                
                # Apply the same chunk limitation as in training
                max_chunks = 8
                if chunks.size(0) > max_chunks:
                    mid_idx = chunks.size(0) // 2
                    start_idx = max(0, mid_idx - max_chunks // 2)
                    end_idx = min(chunks.size(0), start_idx + max_chunks)
                    if end_idx > chunks.size(0):
                        end_idx = chunks.size(0)
                        start_idx = max(0, end_idx - max_chunks)
                    chunks = chunks[start_idx:end_idx]
                
                # Forward pass
                logits = self.model(chunks)
                
                # Convert to float32 before going to CPU and numpy
                logits = logits.float()
                
                # Convert NumPy arrays to PyTorch tensors for accuracy calculation
                target = torch.tensor(labels, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
                
                # 1-year cancer accuracy (index 0)
                if mask_tensor[0]:
                    prob1 = torch.sigmoid(logits[0, 0])
                    pred1 = (prob1 > 0.5).float()
                    acc1 = (pred1 == target[0]).float().item()
                    total_acc1 += acc1
                    if target[0].item() == 1:
                        one_count_1yr += 1
                    else:
                        zero_count_1yr += 1
                
                # 3-year cancer accuracy (index 1)
                if mask_tensor[1]:
                    prob3 = torch.sigmoid(logits[0, 1])
                    pred3 = (prob3 > 0.5).float()
                    acc3 = (pred3 == target[1]).float().item()
                    total_acc3 += acc3
                    if target[1].item() == 1:
                        one_count_3yr += 1
                    else:
                        zero_count_3yr += 1
                
                # 6-year cancer accuracy (index 2)
                if mask_tensor[2]:
                    prob6 = torch.sigmoid(logits[0, 2])
                    pred6 = (prob6 > 0.5).float()
                    acc6 = (pred6 == target[2]).float().item()
                    total_acc6 += acc6
                    if target[2].item() == 1:
                        one_count_6yr += 1
                    else:
                        zero_count_6yr += 1
                
                # Store predictions and targets for AUC calculation
                all_preds.append(logits.cpu().numpy())
                all_targets.append(labels)
                all_masks.append(mask)
                samples += 1
        
        # Calculate metrics
        avg_acc1 = total_acc1 / max(1, (one_count_1yr + zero_count_1yr))
        avg_acc3 = total_acc3 / max(1, (one_count_3yr + zero_count_3yr))
        avg_acc6 = total_acc6 / max(1, (one_count_6yr + zero_count_6yr))
        
        # Format counts as requested
        pos_count_1yr_str = f"{one_count_1yr}-{zero_count_1yr}"
        pos_count_3yr_str = f"{one_count_3yr}-{zero_count_3yr}"
        pos_count_6yr_str = f"{one_count_6yr}-{zero_count_6yr}"
        
        # Calculate metrics for each task using scikit-learn
        metrics = {}
        metrics['acc1'] = avg_acc1
        metrics['acc3'] = avg_acc3
        metrics['acc6'] = avg_acc6
        metrics['pos_count_1yr'] = pos_count_1yr_str
        metrics['pos_count_3yr'] = pos_count_3yr_str
        metrics['pos_count_6yr'] = pos_count_6yr_str
        
        # Save validation metrics in the same format as example
        if self.rank == 0:
            # Same fields as the training metrics
            val_metrics = {
                "iteration": self.global_step,
                "epoch": self.current_epoch,
                "lr": self.opt.param_groups[0]['lr'],
                "accumulation_step": (self.global_step - 1) % self.accum,
                "accum_size": self.accum,
                "total_loss": running_loss / max(1, samples),
                "acc1": avg_acc1,
                "acc3": avg_acc3,
                "acc6": avg_acc6,
                "pos_count_1yr": pos_count_1yr_str,
                "pos_count_3yr": pos_count_3yr_str,
                "pos_count_6yr": pos_count_6yr_str,
                "type": "validation"
            }
            
            # Write to file
            self.metrics_logger.log_metrics(val_metrics)
        
        return metrics
    
    def fit(self, train_loader, train_sampler, val_loader, epochs, unfreeze_strategy):
        self.global_step = 1
        self.current_epoch = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.metrics_calculator.reset()
            
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Handle unfreezing strategy
            self._handle_unfreezing(epoch, unfreeze_strategy)
            
            # Training loop
            self.model.train()
            for step, (chunks, labels, mask) in enumerate(train_loader, 1):
                self._train_step(chunks, labels, mask)
                
                # Validation
                if val_loader and self.rank == 0 and self.global_step % self.val_every == 0:
                    self._run_validation(val_loader)

    def _handle_unfreezing(self, epoch, unfreeze_strategy):
        """Handle unfreezing of model parameters based on strategy and epoch"""
        # Skip handling if rank is not 0 (to avoid duplicate logs)
        if self.rank == 0:
            if unfreeze_strategy == 'all':
                # All parameters already unfrozen in DDP mode
                if epoch == 0:
                    print(f"Epoch {epoch+1}: All parameters already unfrozen (strategy: {unfreeze_strategy})")
            elif unfreeze_strategy == 'none':
                # Parameters should remain frozen - verify but don't change
                if epoch == 0:
                    print(f"Epoch {epoch+1}: Keeping base model parameters frozen (strategy: {unfreeze_strategy})")
            elif unfreeze_strategy == 'gradual':
                # Gradual unfreezing not implemented in DDP mode
                if epoch == 0:
                    print(f"Epoch {epoch+1}: Gradual unfreezing not implemented in DDP mode, all parameters unfrozen")

    def _run_validation(self, val_loader):
        """Run validation and log results"""
        if self.rank == 0:
            print(f"\nRunning validation at step {self.global_step}...")
            self.model.eval()
            metrics = self.evaluate(val_loader)
            self.model.train()
            
            # Log validation metrics
            val_metrics = metrics.copy()
            val_metrics.update({
                "iteration": self.global_step,
                "epoch": self.current_epoch,
                "lr": self.opt.param_groups[0]['lr'],
                "type": "validation"
            })
            self.metrics_logger.log_metrics(val_metrics)
            print(f"Validation complete at step {self.global_step}\n")


def main(args):
    # ------------------------------------------------------------------
    # 1. DDP initialization (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    # convenience so you can still run the script on one GPU
    if world_size == 1:
        print("Single-GPU run – DDP will run on one device")
    
    # Only rank 0 should log preparation information
    if global_rank == 0:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.output, 'training.log'))
            ]
        )
        logging.info(f"Starting training on {world_size} GPUs with full model copies (DDP)")
        
        # Prepare balanced validation dataset if needed
        if args.balance_val:
            csv_path = prepare_balanced_validation(args.csv)
        else:
            csv_path = args.csv
    else:
        # Set up minimal logging for non-zero ranks
        logging.basicConfig(level=logging.WARNING)
        csv_path = args.csv
    
    # Make sure all processes wait until rank 0 has prepared the CSV
    dist.barrier()
    
    # Create the CT volume processor and dataset
    processor = VolumeProcessor(chunk_depth=3, out_size=(448, 448))
    
    # Define the label columns to use for prediction
    label_cols = ['1-year-cancer', '3-year-cancer', '6-year-cancer']
    
    # Create dataset objects for training, validation, and testing
    train_ds = NLSTDataset(
        df=pd.read_csv(csv_path).query("split == 'train'"), 
        processor=processor,
        label_cols=label_cols
    )
    
    val_ds = NLSTDataset(
        df=pd.read_csv(csv_path).query("split == 'val'"),
        processor=processor,
        label_cols=label_cols
    )
    
    # Each rank gets a distinct shard; shuffle is done by the sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=1,                     # still per-patient
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        pin_memory=True,
    )
    
    # Validation dataloader - does not need to be sharded for rank 0
    if global_rank == 0:
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False,
            num_workers=args.num_workers, 
            collate_fn=lambda b: b[0]
        )
    else:
        val_loader = None
    
    # Calculate weights for balanced loss if needed
    if args.class_weights:
        train_df = pd.read_csv(csv_path).query("split == 'train'")
        weights = calculate_class_weights(train_df, label_cols)
        if global_rank == 0:
            logging.info(f"Using class weights: {weights}")
    else:
        weights = None
    
    # Configure device
    device = torch.device(f"cuda:{local_rank}")
    
    # Import model_ct from your module - assuming it's already created as in finetuneA100_gradual_transformer.py
    sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")
    
    from functools import partial
    from dinov2.models.vision_transformer import vit_base, DinoVisionTransformer
    from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
    
    # Load the pretrained model
    checkpoint_path = "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/output_dir/448_192_all14_17M_P32_B8/eval/training_314999/teacher_checkpoint.pth"
    patch_size = 32
    
    # Load the base model first on CPU
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
    )

    # Add explicit initialization of cls_token (this is the problematic part)
    model_ct.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
    torch.nn.init.normal_(model_ct.cls_token, std=0.02)  # Use standard ViT initialization

    # Now load the weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    teacher_weights = checkpoint["teacher"]
    teacher_weights_cleaned = {k.replace("backbone.", ""): v for k, v in teacher_weights.items()}

    # Check if cls_token is in the loaded weights
    if 'cls_token' in teacher_weights_cleaned:
        print(f"Found cls_token in weights with shape: {teacher_weights_cleaned['cls_token'].shape}")
    else:
        print("Warning: cls_token not found in pretrained weights")

    # Load weights with strict=False to allow for missing keys
    model_ct.load_state_dict(teacher_weights_cleaned, strict=False)

    # ---- materialise empty meta tokens ---------------------------------
    def materialize_tokens_(m, dev):
        with torch.no_grad():
            for n in ["cls_token", "register_tokens", "mask_token"]:
                if hasattr(m, n) and getattr(m, n).storage().size() == 0:  # Check if meta/empty
                    print(f"Materializing {n} tensor")
                    real = torch.zeros_like(getattr(m, n), device=dev)
                    torch.nn.init.normal_(real, std=0.02)
                    setattr(m, n, torch.nn.Parameter(real, requires_grad=True))

    # First verify if we have meta tensors
    print(f"Cls token shape after loading: {model_ct.cls_token.shape}, " 
          f"requires_grad: {model_ct.cls_token.requires_grad}, "
          f"storage size: {model_ct.cls_token.storage().size()}")

    # Materialize tokens properly (must come after load_state_dict but before moving to device)
    materialize_tokens_(model_ct, torch.device('cpu'))  # First materialize on CPU

    # Now build the combined model
    model = CombinedModel(
        base_model=model_ct,          # model_ct is not yet on device
        chunk_feat_dim=768,
        hidden_dim=1024,
        num_tasks=len(label_cols),
        num_attn_heads=args.num_attn_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    )

    # Move the model to the device
    model = model.to(device)
    
    # Verify all parameters are on the correct device before DDP wrapping
    if global_rank == 0:
        print(f"Verifying all parameters are on {device}")
    for name, param in model.named_parameters():
        assert param.device == device, f"Parameter {name} is on {param.device}, expected {device}"
    if global_rank == 0:
        print(f"All parameters successfully verified on {device}")

    # Set all parameters to requires_grad=True
    for param in model.parameters():
        param.requires_grad = True

    # Now wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    # Create parameter groups with different learning rates
    # Define our own parameter groups manually after DDP wrapping
    main_lr = args.lr
    base_lr = args.lr * 0.1

    # Option 1: Use named_parameters to separate base and aggregator
    base_params = []
    agg_params = []

    # We need to name the parameters more specifically - Note the module. prefix
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'module.base' in name:
                base_params.append(param)
            else:
                agg_params.append(param)

    # Configure optimizer with parameter groups
    param_groups = [
        {'params': base_params, 'lr': base_lr},  # Lower LR for pretrained DinoVisionTransformer
        {'params': agg_params, 'lr': main_lr}    # Higher LR for aggregator
    ]

    # Log the number of parameters in each group
    if global_rank == 0:
        logging.info(f"DinoVisionTransformer parameters: {len(base_params)} (LR: {base_lr})")
        logging.info(f"Aggregator parameters: {len(agg_params)} (LR: {main_lr})")

    # Create optimizer with parameter groups
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    
    # Configure loss function
    if weights is not None:
        weights = weights.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights, reduction='none')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    # Force unfreeze strategy to "all" to avoid DDP issues
    args.unfreeze_strategy = "all"
    logging.info("Setting unfreeze_strategy to 'all' to avoid DDP synchronization issues")

    # Create the trainer
    trainer = DDPFullTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accum_steps=args.accum_steps,
        print_every=args.print_every,
        val_every=args.val_every,
        rank=global_rank,
        world_size=world_size,
        metrics_dir=args.metrics_dir if global_rank == 0 else None
    )
    
    # Train the model
    trainer.fit(
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        epochs=args.epochs,
        unfreeze_strategy=args.unfreeze_strategy
    )
    
    # Save model checkpoint (only rank 0)
    if global_rank == 0:
        checkpoint_dir = os.path.join(args.output, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the model state dict - note we need to extract the module to save
        torch.save(
            model.module.state_dict(), 
            os.path.join(checkpoint_dir, f'model_final.pt')
        )
        
        logging.info(f"Training completed. Model saved to {checkpoint_dir}")
    
    # Clean shutdown of distributed processes
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Training with DDP (Full Model Copies)")
    parser.add_argument("--csv", type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv", 
                        help="Path to the CSV file containing dataset information")
    parser.add_argument("--balance-val", action="store_true", 
                        help="Whether to balance the validation set")
    parser.add_argument("--class-weights", action="store_true", 
                        help="Whether to use class weights in the loss function")
    parser.add_argument("--accum-steps", type=int, default=10, 
                        help="Number of steps to accumulate gradients over")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                        help="Optimizer to use")
    parser.add_argument("--num-attn-heads", type=int, default=4, 
                        help="Number of attention heads in the transformer aggregator")
    parser.add_argument("--num-layers", type=int, default=2, 
                        help="Number of layers in the transformer aggregator")
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout rate in the transformer aggregator")
    parser.add_argument("--unfreeze-strategy", type=str, default="gradual", choices=["gradual", "none", "all"], 
                        help="Strategy for unfreezing the base model")
    parser.add_argument("--output", type=str, default="./output_ddp_full", 
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--print-every", type=int, default=100, 
                        help="Print training stats every N steps")
    parser.add_argument("--val-every", type=int, default=500, 
                        help="Run validation every N steps")
    parser.add_argument("--metrics-dir", type=str, 
                       default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu",
                       help="Directory to save training metrics")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check if we should install packages (only first time)
    if not os.path.exists(os.path.join(args.output, '.packages_installed')):
        install_packages()
        # Create a marker file to indicate packages have been installed
        with open(os.path.join(args.output, '.packages_installed'), 'w') as f:
            f.write('Packages installed\n')
    
    main(args)
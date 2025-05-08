# File: n_GPU_A100/2_run_fineTune_ddp_full.py

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
    calculate_class_weights,
    # New imports 
    install_packages,
    prepare_balanced_validation,
    calculate_auc,
    MetricsLogger,
    MetricsCalculator,
    augment_3_channel
)

print("Files in the directory:", os.listdir("/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU"))

# Modified Trainer class that supports DDP with full model copies and gradient accumulation
class DDPFullTrainer:
    def __init__(self, args, label_cols):
        self.args = args
        self.label_cols = label_cols
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and training components
        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        self._init_scaler()
        
        # Initialize metrics components
        self.metrics_logger = MetricsLogger(self.args.rank, self.args.metrics_dir if self.args.rank == 0 else None)
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize running accuracy metrics
        self.reset_running_metrics()
        
        # Print configuration
        if self.args.rank == 0:
            print("\nTraining Configuration:")
            print(f"Max chunks per sample: {self.args.max_chunks}")
            print(f"Learning rate: {self.args.lr}")
            print(f"Number of tasks: {len(self.label_cols)}")
            print(f"Validation frequency: {self.args.val_every} steps")
            print(f"Number of epochs: {self.args.epochs}")
            print(f"Warmup steps: {self.args.warmup_steps}")
            print(f"World size: {self.args.world_size}")
            print(f"Device: {self.device}\n")
    
    def _init_model(self):
        """Initialize the model"""
        # Import model_ct from your module
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
            init_values=1.0e-05,
        )

        # Add explicit initialization of cls_token
        model_ct.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        torch.nn.init.normal_(model_ct.cls_token, std=0.02)

        # Load the weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        teacher_weights = checkpoint["teacher"]
        teacher_weights_cleaned = {k.replace("backbone.", ""): v for k, v in teacher_weights.items()}
        model_ct.load_state_dict(teacher_weights_cleaned, strict=False)

        # Materialize tokens
        def materialize_tokens_(m, dev):
            with torch.no_grad():
                for n in ["cls_token", "register_tokens", "mask_token"]:
                    if hasattr(m, n) and getattr(m, n).storage().size() == 0:
                        real = torch.zeros_like(getattr(m, n), device=dev)
                        torch.nn.init.normal_(real, std=0.02)
                        setattr(m, n, torch.nn.Parameter(real, requires_grad=True))

        materialize_tokens_(model_ct, torch.device('cpu'))

        # Build the combined model
        self.model = CombinedModel(
            base_model=model_ct,
            chunk_feat_dim=768,
            hidden_dim=1024,
            num_tasks=len(self.label_cols),
            num_attn_heads=self.args.num_attn_heads,
            num_layers=self.args.num_layers,
            dropout_rate=self.args.dropout
        )

        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set all parameters to requires_grad=True
        for param in self.model.parameters():
            param.requires_grad = True

        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True
        )

    def _init_criterion(self):
        """Initialize the loss function"""
        if self.args.class_weights:
            # Load weights from CSV
            train_df = pd.read_csv(self.args.csv).query("split == 'train'")
            weights = calculate_class_weights(train_df, self.label_cols)
            weights = weights.to(self.device)
            self.crit = torch.nn.BCEWithLogitsLoss(pos_weight=weights, reduction='none')
        else:
            self.crit = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _init_optimizer(self):
        """Initialize the optimizer with parameter groups"""
        # Define learning rates
        main_lr = self.args.lr
        base_lr = self.args.lr * 0.01  # Reduced from 0.1 to 0.01 for base model

        # Separate base and aggregator parameters
        base_params = []
        agg_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'module.base' in name:
                    base_params.append(param)
                else:
                    agg_params.append(param)

        # Configure optimizer with parameter groups
        param_groups = [
            {'params': base_params, 'lr': base_lr},  # Much lower LR for pretrained DinoVisionTransformer
            {'params': agg_params, 'lr': main_lr}    # Higher LR for aggregator
        ]

        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(param_groups)
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    def _init_scaler(self):
        """Initialize the gradient scaler for mixed precision training"""
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_step(self, chunks, labels, mask, spacing):
        """Perform a single training step"""
        chunks = chunks.squeeze(1).to(self.device)
        target = torch.tensor(labels, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        # Apply max_chunks constraint
        if chunks.size(0) > self.args.max_chunks:
            mid_idx = chunks.size(0) // 2
            start_idx = max(0, mid_idx - self.args.max_chunks // 2)
            end_idx = min(chunks.size(0), start_idx + self.args.max_chunks)
            chunks = chunks[start_idx:end_idx]
        
        # Forward pass with gradients
        with torch.cuda.amp.autocast(enabled=True):
            logits = self.model(chunks, spacing)
        
            loss = 0.0
            for j in range(logits.size(1)):
                if mask_tensor[j]:
                    task_loss = self.crit(logits[0, j:j+1], target[j:j+1])
                    loss += task_loss
            
            # Normalize loss by number of tasks
            active_tasks = mask_tensor.sum().item()
            if active_tasks > 0:
                loss = loss / active_tasks
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update learning rate AFTER optimizer step
        self.scheduler.step()
        
        # Update running metrics for accuracy tracking
        with torch.no_grad():
            for j in range(logits.size(1)):
                if mask_tensor[j]:
                    # Convert logits to probability with sigmoid, then threshold
                    probs = torch.sigmoid(logits[0, j])
                    pred = (probs > 0.5).float()
                    
                    # Update running metrics
                    task_key = f'task{j}'
                    is_correct = (pred == target[j]).float().item()
                    if is_correct == 1.0:
                        self.correct_preds[task_key] += 1
                    self.total_preds[task_key] += 1
        
        # Calculate running accuracies
        running_accuracies = {}
        for j in range(len(self.label_cols)):
            task_key = f'task{j}'
            # Only calculate if we have data for this task
            if self.total_preds[task_key] > 0:
                running_acc = self.correct_preds[task_key] / self.total_preds[task_key]
                running_accuracies[f'acc_task{j}_running'] = running_acc
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[1],
            'spacing': spacing,
            **running_accuracies
        }

    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        samples = 0
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for i, (chunks, labels, mask, spacing) in enumerate(val_loader):
                chunks = chunks.squeeze(1).to(self.device)
                
                # Use max_chunks from args
                if chunks.size(0) > self.args.max_chunks:
                    mid_idx = chunks.size(0) // 2
                    start_idx = max(0, mid_idx - self.args.max_chunks // 2)
                    end_idx = min(chunks.size(0), start_idx + self.args.max_chunks)
                    chunks = chunks[start_idx:end_idx]
                
                # Forward pass
                logits = self.model(chunks, spacing)
                
                # Calculate loss
                target = torch.tensor(labels, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
                
                loss = 0.0
                for j in range(logits.size(1)):
                    if mask_tensor[j]:
                        task_loss = self.crit(logits[0, j:j+1], target[j:j+1])
                        loss += task_loss
                
                # Normalize loss
                active_tasks = mask_tensor.sum().item()
                if active_tasks > 0:
                    loss = loss / active_tasks
                    total_loss += loss.item()
                
                # Store predictions and targets
                all_preds.append(logits.cpu().numpy())
                all_targets.append(labels)
                all_masks.append(mask)
                
                samples += 1
        
        # Calculate metrics
        avg_loss = total_loss / max(1, samples)
        
        # Calculate AUC and accuracy for each task
        metrics = {
            'samples_evaluated': samples,
            'avg_loss': avg_loss,
        }
        
        # Combine predictions and calculate metrics
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        masks = np.vstack(all_masks)
        
        for i in range(preds.shape[1]):
            if i >= masks.shape[1]:
                continue
            valid_idx = masks[:, i].astype(bool)
            if np.sum(valid_idx) > 0:
                task_preds = preds[valid_idx, i]
                task_targets = targets[valid_idx, i]
                try:
                    # Calculate AUC using raw logits (this is correct for ROC AUC)
                    auc = roc_auc_score(task_targets, task_preds)
                    metrics[f'auc_task{i}'] = auc
                    
                    # Calculate accuracy with sigmoid transformation and proper thresholding
                    pred_probs = 1 / (1 + np.exp(-task_preds))  # sigmoid in numpy
                    pred_labels = (pred_probs > 0.5).astype(int)
                    acc = np.mean(pred_labels == task_targets)
                    metrics[f'acc_task{i}'] = acc
                except:
                    metrics[f'auc_task{i}'] = float('nan')
                    metrics[f'acc_task{i}'] = float('nan')
        
        print(f"Evaluated {samples} samples")
        print(f"Average loss: {avg_loss:.4f}")
        
        return metrics
    
    def _init_scheduler(self):
        """Initialize the learning rate scheduler"""
        # We'll initialize this in fit() when we have access to the dataloader
        self.scheduler = None

    def fit(self, train_loader, train_sampler, val_loader, epochs, unfreeze_strategy):
        """Train the model"""
        self.global_step = 1
        self.current_epoch = 0
        self.train_loader = train_loader  # Store for scheduler initialization
        
        # Initialize scheduler now that we have access to the dataloader
        total_steps = len(train_loader) * epochs
        warmup_steps = self.args.warmup_steps
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.args.lr * 0.01, self.args.lr],  # Different max_lr for each param group (base, aggregator)
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,  # Percentage of steps for warmup
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=1000.0  # Final lr = initial_lr/1000
        )
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.metrics_calculator.reset()
            
            # Reset running metrics at the start of each epoch
            self.reset_running_metrics()
            
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Handle unfreezing strategy
            self._handle_unfreezing(epoch, unfreeze_strategy)
            
            # Training loop
            self.model.train()
            
            try:
                for step, (chunks, labels, mask, spacing) in enumerate(train_loader, 1):
                    try:
                        # Training step
                        step_metrics = self._train_step(chunks, labels, mask, spacing)
                        
                        # Print training metrics every print_every steps
                        if step_metrics and self.args.rank == 0 and self.global_step % self.args.print_every == 0:
                            print(f"\nStep {self.global_step} (Epoch {epoch+1}):")
                            print(f"Loss: {step_metrics['loss']:.4f}")
                            print(f"Learning rate: {step_metrics['lr']:.6f}")
                            
                            # Print only running averages for all tasks
                            acc_string = "Accuracy: "
                            
                            for i in range(len(self.label_cols)):
                                # Running accuracy (simplified format)
                                running_key = f'acc_task{i}_running'
                                if running_key in step_metrics:
                                    acc_string += f"Task {i}: {step_metrics[running_key]:.2f} | "
                            
                            # Print accuracy string
                            print(acc_string.rstrip(" | "))
                            
                            # Log training metrics to file - only running metrics
                            train_metrics = {
                                "iteration": self.global_step,
                                "epoch": epoch,
                                "loss": step_metrics['loss'],
                                "lr": step_metrics['lr'],
                                "type": "training"
                            }
                            
                            # Add only running accuracies with simplified names
                            for i in range(len(self.label_cols)):
                                running_key = f'acc_task{i}_running'
                                
                                # Add running accuracy if available (with simplified name)
                                if running_key in step_metrics:
                                    # Change from acc_task0_running to just acc_task0
                                    train_metrics[f'acc_task{i}'] = step_metrics[running_key]
                            
                            # Log metrics
                            self.metrics_logger.log_metrics(train_metrics)
                        
                        # Validation - only on rank 0
                        if val_loader and self.args.rank == 0 and self.global_step % self.args.val_every == 0:
                            print(f"\nReached validation point at step {self.global_step} (in epoch {epoch+1})")
                            
                            # Force garbage collection before validation
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Log memory before validation
                            #mem_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                            #mem_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                            #print(f"Memory before validation: {mem_allocated:.2f}MB allocated, {mem_reserved:.2f}MB reserved")
                            
                            # Set a time limit for validation
                            try:
                                self._run_validation(val_loader)
                            except Exception as e:
                                print(f"Validation failed with error: {e}")
                                import traceback
                                traceback.print_exc()
                                
                            # Force garbage collection after validation
                            gc.collect()
                            torch.cuda.empty_cache()
                        
                        self.global_step += 1
                            
                    except Exception as e:
                        print(f"Error in training step at step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Continue with next batch
                        continue
                    
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to continue with next epoch
                continue

    def _handle_unfreezing(self, epoch, unfreeze_strategy):
        """Handle unfreezing of model parameters based on strategy and epoch"""
        # Skip handling if rank is not 0 (to avoid duplicate logs)
        if self.args.rank == 0:
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
        if self.args.rank == 0:
            print(f"\n=== Starting validation at step {self.global_step} ===")
            validation_start_time = time.time()
            
            try:
                # Directly call evaluate method
                metrics = self.evaluate(val_loader)
                
                # Calculate elapsed time
                elapsed = time.time() - validation_start_time
                print(f"Validation completed in {elapsed:.2f} seconds")
                
                # Log validation metrics
                val_metrics = metrics.copy()
                val_metrics.update({
                    "iteration": self.global_step,
                    "epoch": self.current_epoch,
                    "lr": self.scheduler.get_last_lr()[0],
                    "validation_time_seconds": elapsed,
                    "type": "validation"
                })
                self.metrics_logger.log_metrics(val_metrics)
                
            except Exception as e:
                print(f"Error during validation: {e}")
                import traceback
                traceback.print_exc()
            
            # Switch back to training mode
            self.model.train()
            print(f"=== Validation complete at step {self.global_step} ===\n")

    def reset_running_metrics(self):
        """Reset running metrics for a new epoch"""
        # Track correct predictions and total predictions for each task
        self.correct_preds = {f'task{i}': 0 for i in range(len(self.label_cols))}
        self.total_preds = {f'task{i}': 0 for i in range(len(self.label_cols))}


def main(args):
    # ------------------------------------------------------------------
    # 1. DDP initialization (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    # Add rank information to args
    args.rank = global_rank
    args.world_size = world_size
    args.local_rank = local_rank
    
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
    label_cols = ['1-year-cancer', '2-year-cancer', '3-year-cancer', '4-year-cancer', '5-year-cancer', '6-year-cancer']
    
    # Create dataset objects for training, validation, and testing
    train_ds = NLSTDataset(
        df=pd.read_csv(csv_path).query("split == 'train'"), 
        processor=processor,
        label_cols=label_cols,
        augment=augment_3_channel
    )
    
    # Print basic dataset statistics (only on rank 0)
    if global_rank == 0:
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
        # Create validation dataset
        full_df = pd.read_csv(csv_path)
        val_df = full_df[full_df['split'] == 'val'].copy()
        
        # Get total validation count
        total_val_count = len(val_df)
        
        # Limit samples if needed (and if we have more than 100)
        if len(val_df) > 100:
            val_df = val_df.head(100)  # Simply take first 100 to avoid sampling complexity
            print(f"Using first 100 samples from validation set (total: {total_val_count})")
        
        val_ds = NLSTDataset(
            df=val_df,
            processor=processor,
            label_cols=label_cols
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False,
            num_workers=args.num_workers, 
            collate_fn=lambda b: b[0]
        )
        print(f"Created validation loader with {len(val_ds)} samples")
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
        base_model=model_ct,
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
    base_lr = args.lr * 0.01  # Reduced from 0.1 to 0.01 for base model

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
        {'params': base_params, 'lr': base_lr},  # Much lower LR for pretrained DinoVisionTransformer
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
    trainer = DDPFullTrainer(args, label_cols)
    
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

    # Count trainable parameters and print memory usage
    if global_rank == 0:
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n==== MODEL PARAMETERS ====")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Print CUDA memory usage
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        print(f"\n==== GPU MEMORY USAGE ====")
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")
        print(f"Max Allocated: {max_allocated:.2f} MB")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description='DDP Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for learning rate scheduling')
    parser.add_argument('--val_every', type=int, default=100, help='Validation frequency in steps')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default='dino', help='Pretrained model type')
    parser.add_argument('--num_tasks', type=int, default=6, help='Number of classification tasks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dist_url', type=str, default='env://', help='Distributed URL')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    
    # Data parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory in data loader')
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    return parser.parse_args()


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
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduling")
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
    parser.add_argument("--unfreeze-strategy", type=str, default="all", choices=["gradual", "none", "all"], 
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
    parser.add_argument("--max-chunks", type=int, default=66, help="Maximum number of chunks to process per sample")
    
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
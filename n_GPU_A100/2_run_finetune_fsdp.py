
"""
Morteza, drop me a note if you want extra FSDP tuning (e.g. `NO_SHARD`, CPU‑offload, parameter
prefetch) – we can iterate further.
"""

#!/usr/bin/env python
"""
Fully‑Sharded Data‑Parallel (FSDP) fine‑tuning script for DinoV2‑based multi‑task CT‑risk classifier.
Compatible with both PyTorch ≥ 2.1 and legacy 2.0.x CU117 wheels.
"""

import os
import sys
import time
import json
import argparse
import logging
from functools import partial

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ======== FSDP imports =======================================================
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
try:
    from torch.distributed.fsdp.wrap import default_auto_wrap_policy  # PyTorch ≥ 2.1
except (ImportError, AttributeError):
    default_auto_wrap_policy = None
try:
    from torch.distributed.fsdp.sharding_strategy import ShardingStrategy  # new path
except ImportError:
    try:
        from torch.distributed.fsdp import ShardingStrategy  # old path (≤ 2.0.x)
    except ImportError:
        ShardingStrategy = None



from torch.distributed.fsdp import CPUOffload
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")


# ======== Local utilities ====================================================
from model_utils import (
    DinoVisionTransformer,
    CombinedModel,
    VolumeProcessor,
    NLSTDataset,
    calculate_class_weights,
    install_packages,
    prepare_balanced_validation,
    augment_3_channel,
    MetricsLogger,
    MetricsCalculator,
    ModelSaver,
)

print("Files in the directory:", os.listdir("/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU"))

# ---------------------------------------------------------------------------
#   FSDP Trainer
# ---------------------------------------------------------------------------
class FSDPTrainer:
    def __init__(self, args, label_cols):
        self.args = args
        self.label_cols = label_cols
        self.device = torch.device(f"cuda:{args.local_rank}")

        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        self._init_scaler()

        self.metrics_logger = MetricsLogger(args.rank, args.metrics_dir if args.rank == 0 else None)
        self.metrics_calculator = MetricsCalculator()
        self.model_saver = ModelSaver(args.output, args.rank)
        self.reset_running_metrics()

        if args.rank == 0:
            print("\nTraining configuration:")
            for k, v in (
                ("World size", args.world_size),
                ("Device", self.device),
                ("Epochs", args.epochs),
                ("LR", args.lr),
                ("Tasks", len(self.label_cols)),
            ):
                print(f"  {k:14}: {v}")

    # ------------------------------------------------------------------
    # Model / criterion / optimiser
    # ------------------------------------------------------------------
    def _init_model(self):
        sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")
        from dinov2.layers import MemEffAttention, NestedTensorBlock as Block

        ckpt_path = (
            "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/output_dir/"
            "448_192_all23M_p16_win1150_B8_noif/eval/training_1004999/teacher_checkpoint.pth"
        )
        base = DinoVisionTransformer(
            img_size=448,
            patch_size=16,
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
        base.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        torch.nn.init.normal_(base.cls_token, std=0.02)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        base.load_state_dict({k.replace("backbone.", ""): v for k, v in ckpt["teacher"].items()}, strict=False)

        with torch.no_grad():
            for n in ("cls_token", "register_tokens", "mask_token"):
                if hasattr(base, n) and getattr(base, n).storage().size() == 0:
                    t = torch.zeros_like(getattr(base, n))
                    torch.nn.init.normal_(t, std=0.02)
                    setattr(base, n, torch.nn.Parameter(t))

        self.model = CombinedModel(
            base_model=base,
            chunk_feat_dim=768,
            hidden_dim=1024,
            num_tasks=len(self.label_cols),
            num_attn_heads=self.args.num_attn_heads,
            num_layers=self.args.num_layers,
            dropout_rate=self.args.dropout,
        ).to(self.device)

        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        fsdp_kwargs = {"device_id": self.device, "mixed_precision": mp}
        if ShardingStrategy is not None:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
        if default_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = default_auto_wrap_policy
        #fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        self.model = FSDP(self.model, **fsdp_kwargs)

    def _init_criterion(self):
        if self.args.class_weights:
            w = calculate_class_weights(pd.read_csv(self.args.csv).query("split == 'train'"), self.label_cols).to(self.device)
            self.crit = torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="none")
        else:
            self.crit = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _init_optimizer(self):
        main_lr, base_lr = self.args.lr, self.args.lr * 0.01
        base_p, agg_p = [], []
        for n, p in self.model.module.named_parameters():
            if not p.requires_grad:
                continue
            (base_p if "base" in n else agg_p).append(p)
        param_groups = [{"params": base_p, "lr": base_lr}, {"params": agg_p, "lr": main_lr}]
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[self.args.optimizer]
        kwargs = {"weight_decay": self.args.weight_decay} if self.args.optimizer == "adamw" else {}
        if self.args.optimizer == "sgd":
            kwargs["momentum"] = 0.9
        self.optimizer = opt_cls(param_groups, **kwargs)

    def _init_scaler(self):
        self.scaler = torch.cuda.amp.GradScaler()

    # (Training / evaluation loops unchanged – omitted for brevity.)

    def _save_epoch_checkpoint(self, epoch, is_final=False):
        if self.args.rank != 0:
            return
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
            self.model_saver.save_checkpoint(self.model.module, epoch, self.global_step, {"label_cols": self.label_cols, "is_final": is_final}, is_final)

    def reset_running_metrics(self):
        self.correct_preds = {f"task{i}": 0 for i in range(len(self.label_cols))}
        self.total_preds = {f"task{i}": 0 for i in range(len(self.label_cols))}
    # ------------------------------------------------------------------
    # The rest of the trainer (fit, evaluate, etc.) is identical to the
    # original; for clarity and space it is left unchanged.
    # ------------------------------------------------------------------
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
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = self.model(chunks, spacing)
            
            loss = 0.0
            for j in range(logits.size(1)):
                if mask_tensor[j]:
                    task_loss = self.crit(logits[0, j:j+1], target[j:j+1]) 
                    # Mori:the weight in loss is probably not properly applied! 
                    # Maybe later I can change logit where mask is False to 0.0. Actually in training and not here.
                    loss += task_loss
            
            # Normalize loss by number of tasks
            active_tasks = mask_tensor.sum().item()
            if active_tasks > 0:
                loss = loss / active_tasks
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Update weights if we've accumulated enough steps
        if (self.global_step + 1) % self.args.accum_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
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

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate model on validation set using distributed evaluation"""
        self.model.eval()
        local_loss_sum = 0.0
        local_samples = 0
        local_preds = []
        local_targets = []
        local_masks = []
        
        # Process validation samples assigned to this rank
        for i, (chunks, labels, mask, spacing) in enumerate(val_loader):
            chunks = chunks.squeeze(1).to(self.device)
            
            # Apply max_chunks constraint
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
            #print(chunks.shape, logits.shape, target.shape, mask_tensor.shape)#torch.Size([max_chunks, 3, 448, 448]) torch.Size([1, 6]) torch.Size([6]) torch.Size([6])
            #print("logit" ,logits )#logit tensor([[ 1.0563,  0.4275,  0.4167,  0.1464, -0.3098, -1.9752]],device='cuda:3')
            #print("target" ,target)#target tensor([ 0., -1., -1., -1., -1., -1.], device='cuda:3')
            #print("mask_tensor" ,mask_tensor)#mask_tensor tensor([ True, False, False, False, False, False], device='cuda:3')
            loss = 0.0
            for j in range(logits.size(1)):
                if mask_tensor[j]:
                    task_loss = self.crit(logits[0, j:j+1], target[j:j+1])
                    # Mori:the weight in loss is probably not properly applied! 
                    # Maybe later I can change logit where mask is False to 0.0. Actually in training and not here. 
                    loss += task_loss
            
            # Normalize loss
            active_tasks = mask_tensor.sum().item()
            if active_tasks > 0:
                loss = loss / active_tasks
                local_loss_sum += loss.item()
            
            # Store predictions, targets, and masks
            local_preds.append(logits.cpu())
            local_targets.append(torch.tensor(labels, dtype=torch.float32))
            local_masks.append(torch.tensor(mask, dtype=torch.bool))
            
            local_samples += 1
        
        # Convert to tensors for gathering
        if local_preds: #torch.Size([7, 1, 6]) torch.Size([7, 6]) torch.Size([7, 6])
            # Use stack for all tensors to ensure consistent [S, T] shape
            local_preds_tensor = torch.stack(local_preds, dim=0)
            local_targets_tensor = torch.stack(local_targets, dim=0)  
            local_masks_tensor = torch.stack(local_masks, dim=0)
        else:
            # Handle edge case where a rank might not get any validation samples
            num_tasks = len(self.label_cols) 
            local_preds_tensor = torch.zeros((0, num_tasks), dtype=torch.float32)
            local_targets_tensor = torch.zeros((0, num_tasks), dtype=torch.float32)
            local_masks_tensor = torch.zeros((0, num_tasks), dtype=torch.bool)
        #print("## 33 ##", local_preds_tensor.shape, local_targets_tensor.shape, local_masks_tensor.shape)
        ## 33 ## torch.Size([1289, 1, 6]) torch.Size([1289, 6]) torch.Size([1289, 6])
        # Move tensors to device for all_gather
        local_preds_tensor = local_preds_tensor.to(self.device)
        local_targets_tensor = local_targets_tensor.to(self.device)
        local_masks_tensor = local_masks_tensor.to(self.device)
        
        # Compute total loss across all ranks
        total_samples = torch.tensor([local_samples], dtype=torch.float32, device=self.device)
        total_loss_sum = torch.tensor([local_loss_sum], dtype=torch.float32, device=self.device)
        
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_sum.item() / max(1, total_samples.item())
        total_samples = int(total_samples.item())
        
        # Initialize metrics
        metrics = {
            'samples_evaluated': total_samples,
            'avg_loss': avg_loss,
        }
        
        # Get world size for gathering
        world_size = dist.get_world_size()
        
        # Padding tensors to the same size for all_gather
        # First find the maximum size across all ranks
        local_size = torch.tensor([local_preds_tensor.shape[0]], dtype=torch.long, device=self.device)
        all_sizes = [torch.ones(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max([size.item() for size in all_sizes])
        
        # Pad local tensors to max_size
        num_tasks = len(self.label_cols)
        if local_preds_tensor.shape[0] < max_size:
            padding_size = max_size - local_preds_tensor.shape[0]
            # Pad with zeros
            preds_padding = torch.zeros((padding_size, num_tasks), dtype=torch.float32, device=self.device)
            targets_padding = torch.zeros((padding_size, num_tasks), dtype=torch.float32, device=self.device)
            masks_padding = torch.zeros((padding_size, num_tasks), dtype=torch.bool, device=self.device)
            
            # Concatenate original tensor with padding
            local_preds_tensor = torch.cat([local_preds_tensor, preds_padding], dim=0)
            local_targets_tensor = torch.cat([local_targets_tensor, targets_padding], dim=0)
            local_masks_tensor = torch.cat([local_masks_tensor, masks_padding], dim=0)
        
        # Use all_gather for efficient collection on GPU
        gathered_preds = [torch.empty_like(local_preds_tensor) for _ in range(world_size)]
        gathered_targets = [torch.empty_like(local_targets_tensor) for _ in range(world_size)]
        gathered_masks = [torch.empty_like(local_masks_tensor) for _ in range(world_size)]
        
        # Gather data from all ranks
        dist.all_gather(gathered_preds, local_preds_tensor)
        dist.all_gather(gathered_targets, local_targets_tensor)
        dist.all_gather(gathered_masks, local_masks_tensor)
        #print("## 34 ##", gathered_preds[0].shape, gathered_targets[0].shape, gathered_masks[0].shape)
        ## 34 ## torch.Size([1289, 1, 6]) torch.Size([1289, 6]) torch.Size([1289, 6])
        #print("## 34 ##", len(gathered_preds), len(gathered_targets), len(gathered_masks))## 34 ## 4 4 4

        #print("#########")
        #print("## 36 ##", all_sizes, local_size) 
        ## 36 ## [tensor([1289], device='cuda:2'), tensor([1289], device='cuda:2'), tensor([1289], device='cuda:2'), tensor([1289], device='cuda:2')] [tensor([1289], device='cuda:3'), tensor([1289], device='cuda:3'), tensor([1289], device='cuda:3'), tensor([1289], device='cuda:3')] tensor([1289], device='cuda:2')
        #print()
        
        # Only rank 0 computes the actual metrics
        if self.args.rank == 0:
            # Remove padding and convert to numpy for metric calculation
            valid_preds = []
            valid_targets = []
            valid_masks = []
            
            for i, size in enumerate([size.item() for size in all_sizes]):
                if size > 0:
                    valid_preds.append(gathered_preds[i][:size])
                    valid_targets.append(gathered_targets[i][:size])
                    valid_masks.append(gathered_masks[i][:size])
            
            # Combine all gathered data
            #all_preds = torch.cat(valid_preds, dim=0).cpu().numpy()
            all_preds = torch.cat(valid_preds, dim=0).squeeze(1).cpu().numpy()
            all_targets = torch.cat(valid_targets, dim=0).cpu().numpy()
            all_masks = torch.cat(valid_masks, dim=0).cpu().numpy()
            
            for i in range(all_preds.shape[1]):
                valid_idx = all_masks[:, i]
                if valid_idx.sum() == 0:
                    continue

                task_preds   = all_preds[valid_idx, i]
                task_targets = all_targets[valid_idx, i]

                #  keep only entries where target is 0 or 1
                keep = (task_targets == 0) | (task_targets == 1)
                task_preds, task_targets = task_preds[keep], task_targets[keep]

                #  debug: if anything was dropped, print the unique values once
                if self.args.rank == 0 and np.unique(task_targets).size < np.unique(all_targets[valid_idx, i]).size:
                    bad_vals = set(all_targets[valid_idx, i]) - {0, 1}
                    print(f"[WARN] Task {i}: dropped labels {bad_vals}")

                #  need both classes left to compute AUC
                if np.unique(task_targets).size < 2:
                    metrics[f'auc_task{i}'] = np.nan
                    metrics[f'acc_task{i}'] = np.nan
                    continue

                # --- DEBUG BLOCK ----------------------------------------------------
                bad_vals = np.setdiff1d(task_targets, [0, 1])
                if bad_vals.size > 0:
                    # Print once per task per validation
                    print(f"[Rank {self.args.rank}] Task {i} – bad labels detected: {bad_vals}")
                    # Optional: locate the offending rows (expensive, so enable only if needed)
                    bad_rows = np.where(~np.isin(task_targets, [0, 1]))[0]
                    print(f"Bad rows (local indices): {bad_rows[:20]} ...")
                    # You could also log IDs if your dataset returns them.

                    # Drop the invalid entries so metrics still compute
                    keep = np.isin(task_targets, [0, 1])
                    task_preds, task_targets = task_preds[keep], task_targets[keep]
                # --------------------------------------------------------------------
                #  metrics
                auc = roc_auc_score(task_targets, task_preds)
                metrics[f'auc_task{i}'] = auc

                prob = 1 / (1 + np.exp(-task_preds))
                acc  = (prob > 0.5).astype(int).mean()
                metrics[f'acc_task{i}'] = acc
        
        # More efficient metrics broadcasting
        metrics = metrics if self.args.rank == 0 else None
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        metrics = metrics_list[0]
        
        if self.args.rank == 0:
            print(f"Evaluated {total_samples} samples across {world_size} processes")
            print(f"Average loss: {avg_loss:.4f}")
        
        # Synchronize processes
        dist.barrier()
        
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
            
            for step, (chunks, labels, mask, spacing) in enumerate(train_loader, 1):
                # Training step (no try/except)
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
                
                self.global_step += 1
                
                # Validation - all ranks participate
                if val_loader and self.global_step % self.args.val_every == 0:
                    # Only rank 0 prints the validation message
                    if self.args.rank == 0:
                        print(f"\nReached validation point at step {self.global_step} (in epoch {epoch+1})")
                    
                    # Force garbage collection before validation on all ranks
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Run distributed validation
                    self._run_validation(val_loader)
                    
                    # Force garbage collection after validation on all ranks
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Save checkpoint after each epoch
            if self.args.rank == 0:
                checkpoint_metadata = {
                    'label_cols': self.label_cols,
                    'model_config': {
                        'num_attn_heads': self.args.num_attn_heads,
                        'num_layers': self.args.num_layers,
                        'dropout_rate': self.args.dropout
                    },
                    'epoch_metrics': {
                        'epoch': epoch + 1,
                        'total_steps': self.global_step,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    }
                }
                self.model_saver.save_checkpoint(
                    model=self.model,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    metadata=checkpoint_metadata,
                    is_final=False
                )
                print(f"\nSaved checkpoint for epoch {epoch + 1}")
                
        # Save final model
        if self.args.rank == 0:
            checkpoint_metadata = {
                'label_cols': self.label_cols,
                'model_config': {
                    'num_attn_heads': self.args.num_attn_heads,
                    'num_layers': self.args.num_layers,
                    'dropout_rate': self.args.dropout
                },
                'final_metrics': {
                    'total_steps': self.global_step,
                    'epochs_completed': epochs
                }
            }
            self.model_saver.save_checkpoint(
                model=self.model,
                epoch=epochs,
                global_step=self.global_step,
                metadata=checkpoint_metadata,
                is_final=True
            )

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
        """Run validation and log results in a distributed setting"""
        # All ranks participate in validation
        validation_start_time = time.time()
        
        if self.args.rank == 0:
            print(f"\n=== Starting distributed validation at step {self.global_step} ===")
        
        # All ranks call evaluate method (no try/except - errors will propagate)
        metrics = self.evaluate(val_loader)
        
        # Calculate elapsed time
        elapsed = time.time() - validation_start_time
        
        # Only rank 0 logs the results
        if self.args.rank == 0:
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
        
        # Make sure all processes synchronize before continuing
        dist.barrier()
        
        # Switch back to training mode on all ranks
        self.model.train()
        
        if self.args.rank == 0:
            print(f"=== Distributed validation complete at step {self.global_step} ===\n")



# -----------------------------------------------------------------------------
#   CLI helpers
# -----------------------------------------------------------------------------

def init_distributed():
    """Initialise NCCL process‑group and pin CUDA device for this rank."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # <‑‑ FIXED: was torch.cuda.set
    return local_rank, dist.get_rank(), dist.get_world_size()


def build_data(args, rank, world_size, label_cols):
    processor = VolumeProcessor(chunk_depth=3, out_size=(448, 448))
    csv_path = (
        prepare_balanced_validation(args.csv) if args.balance_val and rank == 0 else args.csv
    )
    # Wait for rank0 to finish balancing
    dist.barrier()

    df = pd.read_csv(csv_path)
    train_ds = NLSTDataset(
        df=df.query("split == 'train'"),
        processor=processor,
        label_cols=label_cols,
        augment=augment_3_channel,
    )
    val_ds = NLSTDataset(
        df=df.query("split == 'val'"), processor=processor, label_cols=label_cols
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        pin_memory=True,
    )
    return train_loader, train_sampler, val_loader


def main(args):
    local_rank, global_rank, world_size = init_distributed()
    args.local_rank, args.rank, args.world_size = local_rank, global_rank, world_size

    if global_rank == 0:
        os.makedirs(args.output, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.output, "training.log")),
            ],
        )
        logging.info(f"Starting FSDP training on {world_size} GPUs")

    # Build data loaders
    label_cols = [
        "1-year-cancer",
        "2-year-cancer",
        "3-year-cancer",
        "4-year-cancer",
        "5-year-cancer",
        "6-year-cancer",
    ]
    train_loader, train_sampler, val_loader = build_data(args, global_rank, world_size, label_cols)

    if dist.get_rank() == 0:
        print(torch.cuda.get_device_name())
        print("Total GPU memory:",torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB")#39.3812255859375 GiB
    # Launch trainer
    trainer = FSDPTrainer(args, label_cols)
    trainer.fit(
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        epochs=args.epochs,
        unfreeze_strategy=args.unfreeze_strategy,
    )

    dist.barrier()
    dist.destroy_process_group()


# -----------------------------------------------------------------------------
#   CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU fine‑tuning with FSDP")
    parser.add_argument(
        "--csv",
        type=str,
        default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv",
    )
    parser.add_argument("--balance-val", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default="adamw"
    )
    parser.add_argument("--num-attn-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--unfreeze-strategy", choices=["gradual", "none", "all"], default="all"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu",
    )
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu",
    )
    parser.add_argument("--max-chunks", type=int, default=8)##################################
    args = parser.parse_args()

    # one‑time package install marker
    os.makedirs(args.output, exist_ok=True)
    marker = os.path.join(args.output, ".packages_installed")
    if not os.path.exists(marker):
        install_packages()
        with open(marker, "w") as f:
            f.write("ok\n")

    main(args)

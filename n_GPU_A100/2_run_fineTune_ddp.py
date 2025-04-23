import subprocess
import os
import logging
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext  # for the no_sync helper
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

# Import your model and other components
import sys
import os
import importlib.util

# Define the full path to the module file
module_path = "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/finetuneA100_gradual_transformer.py"

# Check if the file exists
if not os.path.exists(module_path):
    # Try alternative path with a number prefix
    alt_module_path = "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/1_finetuneA100_gradual_transformer.py"
    if os.path.exists(alt_module_path):
        module_path = alt_module_path
    else:
        print(f"ERROR: Could not find module at {module_path} or {alt_module_path}")
        print("Files in directory:", os.listdir("/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU"))
        sys.exit(1)

# Load the module dynamically
spec = importlib.util.spec_from_file_location("transformer_module", module_path)
transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_module)

# Import the required classes and functions from the module
DinoVisionTransformer = transformer_module.DinoVisionTransformer
CombinedModel = transformer_module.CombinedModel
VolumeProcessor = transformer_module.VolumeProcessor
NLSTDataset = transformer_module.NLSTDataset
Trainer = transformer_module.Trainer
calculate_class_weights = transformer_module.calculate_class_weights

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


# Modified Trainer class that supports DDP and gradient accumulation
class DDPTrainer(Trainer):
    def __init__(
        self,
        model: CombinedModel,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        accum_steps: int = 10,
        print_every: int = 100,
        val_every: int = 500,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, optimizer, criterion, device, accum_steps, print_every, val_every)
        self.rank = rank
        self.world_size = world_size
        self.device = device  # Explicitly set device attribute
    
    def _train_step(self, chunks, labels, mask):
        """Modified training step that respects FSDP workflow"""
        running_loss = 0.0
        
        # Remove the extra dimension from chunks
        # Shape from (N, 1, 3, 448, 448) to (N, 3, 448, 448)
        chunks = chunks.squeeze(1)
        
        # Move chunks to device 
        chunks = chunks.to(self.device)  # (N, 3, 448, 448)
        
        # Debug info - only print once
        if self.rank == 0 and self.global_step == 1:
            print(f"Chunks shape after squeeze: {chunks.shape}")
        
        # Forward pass through the entire model (base + aggregator)
        try:
            logits = self.model(chunks)  # Combined forward pass
        except Exception as e:
            # Print diagnostic info and reraise
            print(f"Error in forward pass: {e}")
            print(f"Chunks shape: {chunks.shape}")
            raise

        # Convert NumPy arrays to PyTorch tensors and move to device
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
            
        # -------------------------------------------------------
        # Accumulate gradients locally; only sync on last step
        # -------------------------------------------------------
        accum_step = (self.global_step - 1) % self.accum
        sync_now = (accum_step == self.accum - 1)
        if hasattr(self.model, "no_sync"):
            ctx = nullcontext() if sync_now else self.model.no_sync()
        else:
            ctx = nullcontext()
        with ctx:
            loss.backward()
        
        # Only update weights after accumulating enough gradients
        if (self.global_step) % self.accum == 0:
            self.opt.step()
            self.opt.zero_grad()
        
        # Only log from rank 0
        if self.rank == 0 and (self.global_step) % self.print_every == 0:
            logging.info(f"Step {self.global_step} | Loss: {loss.item():.6f}")
        
        self.global_step += 1
        return running_loss / max(1, active_tasks)
    
    def fit(self, 
            train_loader: DataLoader,
            train_sampler=None,
            val_loader: DataLoader = None,
            epochs: int = 10,
            unfreeze_strategy: str = 'gradual'):
        """Modified fit method with DDP support and epoch-based sampler updates"""
        self.global_step = 1
        
        for epoch in range(epochs):
            # Set the epoch for the data sampler to ensure each process gets different data
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Implement gradual unfreezing strategy
            if unfreeze_strategy == 'gradual':
                if epoch == 0:
                    logging.info("Freezing base model")
                    self.model.freeze_base_model()
                elif epoch == 1:
                    logging.info("Unfreezing last 2 blocks")
                    self.model.unfreeze_last_n_blocks(n=2)
                elif epoch == 2:
                    logging.info("Unfreezing last 4 blocks")
                    self.model.unfreeze_last_n_blocks(n=4)
                elif epoch == 3:
                    logging.info("Unfreezing last 6 blocks")
                    self.model.unfreeze_last_n_blocks(n=6)
                elif epoch >= 4:
                    logging.info("Unfreezing entire base model")
                    self.model.unfreeze_base_model()
            elif unfreeze_strategy == 'none':
                logging.info("Using frozen base model for all epochs")
                self.model.freeze_base_model()
            elif unfreeze_strategy == 'all':
                logging.info("Using unfrozen base model for all epochs")
                self.model.unfreeze_base_model()
            
            # Train loop
            self.model.train()
            epoch_loss = 0.0
            samples_seen = 0
            
            for step, (chunks, labels, mask) in enumerate(train_loader, 1):
                # Move to the right device (already handled in _train_step)
                step_loss = self._train_step(chunks, labels, mask)
                epoch_loss += step_loss
                samples_seen += 1
                
                # Evaluate on validation set periodically
                if val_loader and self.rank == 0 and self.global_step % self.val_every == 0:
                    self.model.eval()
                    metrics = self.evaluate(val_loader)
                    # Log validation metrics
                    log_str = f"Validation | "
                    for metric, value in metrics.items():
                        log_str += f"{metric}: {value:.4f} | "
                    logging.info(log_str)
                    self.model.train()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(1, samples_seen)
            
            # Only log from rank 0
            if self.rank == 0:
                logging.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.6f}")
                
                # End of epoch validation
                if val_loader:
                    self.model.eval()
                    metrics = self.evaluate(val_loader)
                    # Log validation metrics
                    log_str = f"Epoch {epoch+1} Validation | "
                    for metric, value in metrics.items():
                        log_str += f"{metric}: {value:.4f} | "
                    logging.info(log_str)
                    self.model.train()


def main(args):
    # ------------------------------------------------------------------
    # 1. DDP initialization (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    # convenience so you can still run the script on one GPU with python run.py
    # (torchrun will set the env vars for you)
    if world_size == 1:
        print("Single-GPU run – FSDP will act like DataParallel on one device")
    
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
        logging.info(f"Starting training on {world_size} GPUs")
        
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
    
    # Validation dataloader - does not need to be sharded
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers, 
        collate_fn=lambda b: b[0]
    )
    
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

    # Move model_ct to device
    model_ct = model_ct.to(device)

    # Verify cls_token is properly initialized with storage
    print(f"Cls token shape after materialization and device move: {model_ct.cls_token.shape}, " 
          f"requires_grad: {model_ct.cls_token.requires_grad}, "
          f"storage size: {model_ct.cls_token.storage().size()}, "
          f"device: {model_ct.cls_token.device}")

    # Now build the combined model
    model = CombinedModel(
        base_model=model_ct,          # model_ct is already on correct CUDA device
        chunk_feat_dim=768,
        hidden_dim=1024,
        num_tasks=len(label_cols),
        num_attn_heads=args.num_attn_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    )

    # Move the entire combined model to the device (ensures all parameters are on same device)
    model = model.to(device)
    
    # Verify all parameters are on the correct device
    if global_rank == 0:
        print(f"Verifying all parameters are on {device}")
    for name, param in model.named_parameters():
        assert param.device == device, f"Parameter {name} is on {param.device}, expected {device}"
    if global_rank == 0:
        print(f"All parameters successfully verified on {device}")

    # Set all parameters to requires_grad=True
    for param in model.parameters():
        param.requires_grad = True

    # Now it's safe to wrap with FSDP
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16,
                              reduce_dtype=torch.bfloat16,
                              buffer_dtype=torch.bfloat16)
    
    transformer_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            TransformerEncoderLayer,
            TransformerDecoderLayer,
            # Add custom transformer blocks if needed
        }
    )

    model = FSDP(
        model,
        auto_wrap_policy=transformer_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=local_rank,  # Use local_rank instead of device
    )

    # Create parameter groups with different learning rates
    # Define our own parameter groups manually after FSDP wrapping
    main_lr = args.lr
    base_lr = args.lr * 0.1

    # Option 1: Use named_parameters to separate base and aggregator
    base_params = []
    agg_params = []

    # We need to name the parameters more specifically
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'base' in name:
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
    
    # Create the trainer
    trainer = DDPTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accum_steps=args.accum_steps,
        print_every=args.print_every,
        val_every=args.val_every,
        rank=global_rank,
        world_size=world_size
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
        
        # Save the model state dict
        torch.save(
            model.state_dict(), 
            os.path.join(checkpoint_dir, f'model_final.pt')
        )
        
        logging.info(f"Training completed. Model saved to {checkpoint_dir}")
    
    # Clean shutdown of distributed processes
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Training with FSDP")
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
    parser.add_argument("--output", type=str, default="./output", 
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
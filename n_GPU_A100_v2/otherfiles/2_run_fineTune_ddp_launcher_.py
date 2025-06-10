import subprocess
import os
import logging
import argparse
import pandas as pd
import yaml
from pathlib import Path
import socket
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

# Constants for distributed environment
_TORCH_DISTRIBUTED_ENV_VARS = [
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
]

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


def prepare_balanced_validation(csv_path, output_csv_path):
    """Prepare a balanced validation set from the original data"""
    df = pd.read_csv(csv_path)
    
    # Print original validation set size
    print("Original validation set size:", df[df["split"]=="val"].shape)

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
    df_new.to_csv(output_csv_path, index=False)
    return output_csv_path


def _get_available_port():
    """Get a free port on the machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to a free port provided by the OS
    port = sock.getsockname()[1]
    sock.close()
    return port


def _collect_env_vars():
    """Collect existing torch distributed environment variables."""
    return {var: os.environ.get(var) for var in _TORCH_DISTRIBUTED_ENV_VARS if var in os.environ}


def _check_env_variable(name, value):
    """Check if environment variable exists and matches expected value."""
    if name in os.environ and os.environ[name] != value:
        raise ValueError(
            f"Environment variable '{name}' already exists with value '{os.environ[name]}', "
            f"which differs from the expected value '{value}'"
        )


class _TorchDistributedEnvironment:
    def __init__(self):
        self.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self.master_port = int(os.environ.get("MASTER_PORT", _get_available_port()))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

        env_vars = _collect_env_vars()
        if not env_vars:
            raise RuntimeError("Torchrun environment variables are not set.")
        elif len(env_vars) == len(_TORCH_DISTRIBUTED_ENV_VARS):
            self._set_from_preset_env()  # Ensure that environment variables are set correctly
        else:
            collected_env_vars = ", ".join(env_vars.keys())
            raise RuntimeError(f"Partially set environment: {collected_env_vars}")

    def _set_from_preset_env(self):
        # Initialization from preset torchrun environment variables
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                _check_env_variable(k, v)

        os.environ.update(env_vars)
        return self


def main(args):
    # Install required packages if needed
    if args.install_packages:
        install_packages()
    
    # Prepare balanced validation set if needed
    if args.balance_val:
        csv_path = prepare_balanced_validation(
            args.csv, 
            args.csv.replace('.csv', '_balanced.csv')
        )
    else:
        csv_path = args.csv
    
    # Create metrics directory
    metrics_dir = "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu"
    os.makedirs(metrics_dir, exist_ok=True)
        
    # Path to the training script
    script_path = "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/n_GPU_A100_v2/2_run_fineTune_ddp_full_.py"
    
    # Get a random port
    port = _get_available_port()

    # Build torchrun command
    torchrun_command = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={port}",  # Use random port
        script_path,
        "--csv", csv_path,
        "--accum-steps", str(args.accum_steps),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--optimizer", args.optimizer,
        "--num-attn-heads", str(args.num_attn_heads),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--unfreeze-strategy", args.unfreeze_strategy,
        "--output", args.output,
        "--print-every", str(args.print_every),
        "--val-every", str(args.val_every),
        "--metrics-dir", metrics_dir,
        "--warmup-steps", str(args.warmup_steps),
    ]
    
    # Add boolean flags if enabled
    if args.class_weights:
        torchrun_command.append("--class-weights")
    else:
        torchrun_command.append("--no-class-weights")
    
    print(f"Running command: {' '.join(torchrun_command)}")
    
    # Execute the training
    env = _TorchDistributedEnvironment()
    env.export(overwrite=True)  # Force environment variables to be set properly

    subprocess.run(torchrun_command, check=True)

    print(f"Training metrics will be saved to: {metrics_dir}/training_metrics.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Multi-GPU Training with FSDP")
    
    # Data parameters
    parser.add_argument("--csv", type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_.csv", 
                        help="Path to the CSV file containing dataset information")
    parser.add_argument("--balance-val", action="store_true", 
                        help="Whether to balance the validation set")
    parser.add_argument("--class-weights", action="store_false", default=True,
                        help="Whether to use class weights in the loss function (default: True)")
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_true",
                        help="Disable class weights in the loss function")
    
    # Hardware parameters
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use for training")
    parser.add_argument("--num-workers", type=int, default=1, 
                        help="Number of workers for data loading")
    
    # Training parameters
    parser.add_argument("--accum-steps", type=int, default=2000, help="Number of steps to accumulate gradients over")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("--unfreeze-strategy", type=str, default="all", choices=["gradual", "none", "all"], help="Strategy for unfreezing the base model")
    
    # Model parameters
    parser.add_argument("--num-attn-heads", type=int, default=3, help="Number of attention heads in the transformer aggregator")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in the transformer aggregator")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in the transformer aggregator")
    
    # Output parameters
    parser.add_argument("--output", type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu", help="Output directory for logs and checkpoints")
    parser.add_argument("--print-every", type=int, default=50, help="Print training stats every N steps")
    parser.add_argument("--val-every", type=int, default=100, help="Run validation every N steps")
    
    # Setup parameters
    parser.add_argument("--install-packages", action="store_true",help="Whether to install required packages")
    
    # Add metrics directory parameter
    parser.add_argument("--metrics-dir", type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu",help="Directory to save training metrics")
    
    # New parameter for warmup steps
    parser.add_argument("--warmup-steps", type=int, default=50, help="Number of steps for warmup")
    
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    main(args) 
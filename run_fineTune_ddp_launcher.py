import subprocess
import os
import logging
import argparse
import pandas as pd
import yaml
from pathlib import Path

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
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_fineTune_ddp.py")
    
    # Build torchrun command
    torchrun_command = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
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
    ]
    
    # Add boolean flags if enabled
    if args.class_weights:
        torchrun_command.append("--class-weights")
    
    print(f"Running command: {' '.join(torchrun_command)}")
    
    # Execute the training
    subprocess.run(torchrun_command, check=True)

    print(f"Training metrics will be saved to: {metrics_dir}/training_metrics.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Multi-GPU Training with FSDP")
    
    # Data parameters
    parser.add_argument("--csv", type=str, default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv", 
                        help="Path to the CSV file containing dataset information")
    parser.add_argument("--balance-val", action="store_true", 
                        help="Whether to balance the validation set")
    parser.add_argument("--class-weights", action="store_true", 
                        help="Whether to use class weights in the loss function")
    
    # Hardware parameters
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs to use for training")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of workers for data loading")
    
    # Training parameters
    parser.add_argument("--accum-steps", type=int, default=10, 
                        help="Number of steps to accumulate gradients over")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                        help="Optimizer to use")
    parser.add_argument("--unfreeze-strategy", type=str, default="gradual", choices=["gradual", "none", "all"], 
                        help="Strategy for unfreezing the base model")
    
    # Model parameters
    parser.add_argument("--num-attn-heads", type=int, default=4, 
                        help="Number of attention heads in the transformer aggregator")
    parser.add_argument("--num-layers", type=int, default=2, 
                        help="Number of layers in the transformer aggregator")
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout rate in the transformer aggregator")
    
    # Output parameters
    parser.add_argument("--output", type=str, default="./output_ddp", 
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--print-every", type=int, default=100, 
                        help="Print training stats every N steps")
    parser.add_argument("--val-every", type=int, default=500, 
                        help="Run validation every N steps")
    
    # Setup parameters
    parser.add_argument("--install-packages", action="store_true",
                        help="Whether to install required packages")
    
    # Add metrics directory parameter
    parser.add_argument("--metrics-dir", type=str, 
                      default="/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu",
                      help="Directory to save training metrics YAML file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    main(args) 
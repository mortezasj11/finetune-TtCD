import subprocess
#import numpy as np
import os
import logging
import subprocess
import pandas as pd



def install_packages():
    commands = [
        ["pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath", "xformers==0.0.18", "submitit","numpy<2.0"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
        ["pip", "install", "mmcv-full==1.5.0"]
         ]
    for i,command in enumerate(commands):
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #np.save(f"/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/package{i}_installed_in_train.npy", a)


if __name__ == "__main__":
    install_packages()
    # Load and prepare the balanced validation set
    df = pd.read_csv("/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv")

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
    temp_csv_path = "/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_balanced.csv"
    df_new.to_csv(temp_csv_path, index=False)

    # Run the training script with the balanced validation set
    subprocess.run([
        "python3", 
        "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/1_GPU_A100/1_finetuneA100_gradual_transformer.py",
        "--csv", temp_csv_path,
        "--accum-steps", "50",
        "--cuda-id", "0",
    ], check=True)
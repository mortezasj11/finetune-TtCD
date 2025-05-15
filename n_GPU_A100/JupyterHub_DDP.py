import os
import subprocess

# Install required packages first
# pip_commands = [
#     ["pip", "install","-q", "--extra-index-url", "https://download.pytorch.org/whl/cu117", 
#      "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", 
#      "fvcore", "iopath", "xformers==0.0.18", "submitit", "numpy<2.0"],
#     ["pip", "install", "-q",  "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
#     ["pip", "install","-q",  "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
#     ["pip", "install", "-q", "mmsegmentation==0.27.0"],
#     ["pip", "install","-q", "mmcv-full==1.5.0"],
#     ["pip", "install","-q", "nibabel"]
# ]

# for cmd in pip_commands:
#     try:
#         print(cmd)
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install packages with command: {cmd}")
#         print(f"Error: {e}")



# Set required environment variables
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "3"
os.environ["LOCAL_RANK"] = "0"
os.environ["LOCAL_WORLD_SIZE"] = "3"

# Build the command with --install-packages flag removed
command = [
    "python3",
    "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/n_GPU_A100/2_run_fineTune_ddp_launcher.py",
    "--num-gpus", "3",
    "--csv", "/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv",
    "--accum-steps", "200", #200
    "--num-workers", "5",
    "--epochs", "10",
    "--lr", "0.0001",
    "--weight-decay", "0.0001",
    "--optimizer", "adamw",
    "--num-attn-heads", "3",
    "--num-layers", "2",
    "--dropout", "0.3",
    "--unfreeze-strategy", "all",  ##
    "--output", "./output_ddp",
    "--print-every", "2000", #1000
    "--val-every", "24000", #50k
    "--metrics-dir", "/rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/metrics_multi_gpu"
]

# Run the command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    print(f"Command output: {e.output if hasattr(e, 'output') else 'No output available'}")
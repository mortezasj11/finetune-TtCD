# Multi-GPU Fine-Tuning Project

This repository contains code for fine-tuning transformer models for medical image analysis using multiple GPUs with PyTorch's FSDP (Fully Sharded Data Parallel).

## Project Files

| File | Description |
|------|-------------|
| `run_fineTune.py` | Single-GPU fine-tuning launcher |
| `run_fineTune_ddp.py` | Multi-GPU implementation with FSDP |
| `run_fineTune_ddp_launcher.py` | Launcher script for multi-GPU training |
| `run_fineTune_py.ipynb` | Jupyter notebook version of fine-tuning |
| `multiple_GPU.yaml` | Configuration file for multi-GPU setup |
| `finetuneA100_gradual_transformer.py` | Core implementation for transformer fine-tuning |

## Multi-GPU Implementation

We've implemented several optimizations for distributed training:

| Feature | Description | Benefits |
|---------|-------------|----------|
| **DistributedSampler** | Gives each GPU a unique slice of the dataset | Prevents data duplication across GPUs and maintains unbiased statistics |
| **no_sync()** | Skips gradient all-reduce on non-final steps | Eliminates redundant communication during gradient accumulation, improving throughput |
| **Variable batch handling** | Treats each patient as one sample | Allows FSDP/DDP to work with varying chunk counts across samples |
| **Auto-wrap policy** | Only wraps Transformer blocks | Reduces overhead by avoiding unnecessary wrapping of small sub-modules |
| **BFloat16 precision** | Uses BF16 for parameters, gradients and buffers | Reduces memory usage by ~50% while maintaining numerical stability for medical images |
| **FULL_STATE_DICT** | Consolidates checkpoint when saving | Creates a portable checkpoint that can be easily resumed on 1 or more GPUs |

## Running Multi-GPU Training

```bash
# Launch training on 4 GPUs
torchrun --nproc_per_node=4 run_fineTune_ddp.py \
    --csv /path/to/data.csv \
    --accum-steps 4 \
    --output ./output_ddp
    
# Or use the launcher script
python run_fineTune_ddp_launcher.py --num-gpus=4
```

## Server Access

```bash
# Connect to Linux server 
ssh msalehjahromi@10.113.120.155

# Navigate to project directory
cd /rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU
```

## GitHub Repository

This project is hosted at: https://github.com/mortezasj11/finetune-TtCD.git

## Development Workflow

1. Edit files locally using your preferred editor
2. Push changes to GitHub
3. Pull changes on the server and run training
4. Monitor results and iterate

## Git Commands Reference

### Setup & Configuration

```bash
# Clone repository
git clone https://github.com/mortezasj11/finetune-TtCD.git

# Configure Git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Daily Commands

```bash
# Check status and get latest changes
git status
git pull

# Stage, commit and push changes
git add .                           # Add all files
git add filename.py                 # Add specific file
git commit -m "Description of changes"
git push

# View history
git log --oneline
```

### Branch Management

```bash
# Create and use branches
git checkout -b new-feature         # Create new branch
git checkout main                   # Switch to existing branch
git merge new-feature               # Merge branches
git push -u origin new-feature      # Push new branch
```

## Execution Workflows

This project supports two main execution workflows:

#### Single-GPU Training (via JupyterHub)

- one_GPU.yaml   →   run_fineTune.py   →   finetuneA100_gradual_transformer.py

#### Multiple-GPU Training
- multiple_GPU.yaml   →   run_fineTune_ddp_launcher.py   →   run_fineTune_ddp.py

### Json
So 50 iterations would mean 50 individual patient cases have been processed. If your accum_size is 10, then the model weights would have been updated 5 times (after processing iterations 10, 20, 30, 40, and 50).

## K8S commands
- job-runner.sh xxx.yaml

- kubectl delete job -n yn-gpu-workload msalehjahromi-torchrun-ft1
- kubectl delete job -n yn-gpu-workload msalehjahromi-torchrun-ftn

- kubectl apply -f x.yaml


Out of memory, I need to use H100, install other packages!!!!
Also I messed up the 1_ files, I need to have them back from github
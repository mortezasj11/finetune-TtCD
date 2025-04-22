# Multi-GPU Fine-Tuning Project

This repository contains code for fine-tuning machine learning models using multiple GPUs.

## Project Files
- `run_fineTune.py`: Python script to run fine-tuning
- `run_fineTune_py.ipynb`: Jupyter notebook version of the fine-tuning script
- `multiple_GPU.yaml`: Configuration file for multi-GPU setup
- `finetuneA100_gradual_transformer.py`: Implementation of fine-tuning for transformer models on A100 GPUs

## Workflow

### Linux Server Access
```bash
# Connect to Linux server via SSH
ssh msalehjahromi@10.113.120.155

# Navigate to project directory
cd /rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU
```

### GitHub Repository
This project is hosted at: https://github.com/mortezasj11/finetune-TtCD.git

### Essential Git Commands

#### Initial Setup (if needed on a new machine)
```bash
# Clone repository
git clone https://github.com/mortezasj11/finetune-TtCD.git

# Configure Git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Daily Workflow Commands
```bash
# Check repository status
git status

# Pull latest changes from GitHub
git pull

# Add all modified files
git add .

# Add specific files
git add filename.py

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push

# See commit history
git log --oneline
```

#### Branch Management
```bash
# Create and switch to a new branch
git checkout -b new-feature

# Switch to existing branch
git checkout main

# Merge branches
git merge new-feature

# Push new branch to GitHub
git push -u origin new-feature
```

### Development Workflow
1. Edit files locally on Windows using your preferred editor
2. Run code on the Linux server using SSH terminal
3. Use Git to sync changes between environments when needed
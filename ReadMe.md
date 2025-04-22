# Multi-GPU Fine-Tuning Project

This repository contains code for fine-tuning machine learning models using multiple GPUs.

## Project Files
- `run_fineTune.py`: Python script to run fine-tuning
- `run_fineTune_py.ipynb`: Jupyter notebook version of the fine-tuning script
- `multiple_GPU.yaml`: Configuration file for multi-GPU setup
- `finetuneA100_gradual_transformer.py`: Implementation of fine-tuning for transformer models on A100 GPUs

## Setting up GitHub Repository
1. Create a GitHub repository if you don't already have one:
   - Go to GitHub.com and log in
   - Click "+" in the top-right corner and select "New repository"
   - Name your repository (e.g., "multiGPU")
   - Choose visibility (private or public)
   - Click "Create repository"

## Initial Setup on Linux Server
1. Connect to Linux server via SSH:
   ```bash
   ssh msalehjahromi@10.113.120.155
   ```

2. Navigate to your project:
   ```bash
   cd /rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU
   ```

3. Initialize Git and push to GitHub:
   ```bash
   # Initialize Git if not already done
   git init

   # Add your files
   git add .

   # Commit
   git commit -m "Initial commit"

   # Add GitHub repository as remote (replace with your actual username)
   git remote add origin git@github.com:yourusername/multiGPU.git

   # Push to GitHub
   git push -u origin main
   ```

## Working with the Repository
After initial setup, use these commands to manage your code:

1. To push changes:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

2. To pull changes from GitHub:
   ```bash
   git pull
   ```

## Local Development

To work on this project locally:

1. Clone the repository from GitHub:
   ```bash
   git clone git@github.com:yourusername/multiGPU.git
   ```

2. Make your changes in your preferred editor

3. Sync your changes with the remote Linux server through GitHub
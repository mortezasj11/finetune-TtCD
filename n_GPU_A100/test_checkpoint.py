import os
import torch
import pandas as pd
import argparse
import logging
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

# Import utilities from local model_utils.py file
from model_utils import (
    DinoVisionTransformer, 
    CombinedModel, 
    VolumeProcessor, 
    NLSTDataset,
    install_packages
)

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and metadata"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint['metadata']
    model_state = checkpoint['model_state_dict']
    return model_state, metadata

def evaluate_model(model, test_loader, device, label_cols):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for chunks, labels, mask, spacing in test_loader:
            chunks = chunks.squeeze(1).to(device)
            
            # Forward pass
            logits = model(chunks, spacing)
            
            # Store predictions, targets, and masks
            all_preds.append(logits.cpu().numpy())
            all_targets.append(np.array(labels))
            all_masks.append(np.array(mask))
    
    # Combine all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Calculate metrics for each task
    metrics = {}
    for i in range(len(label_cols)):
        valid_idx = all_masks[:, i]
        if valid_idx.sum() == 0:
            continue

        task_preds = all_preds[valid_idx, i]
        task_targets = all_targets[valid_idx, i]

        # Keep only entries where target is 0 or 1
        keep = (task_targets == 0) | (task_targets == 1)
        task_preds = task_preds[keep]
        task_targets = task_targets[keep]

        # Need both classes to compute AUC
        if np.unique(task_targets).size < 2:
            metrics[f'auc_task{i}'] = np.nan
            metrics[f'acc_task{i}'] = np.nan
            continue

        # Calculate AUC
        auc = roc_auc_score(task_targets, task_preds)
        metrics[f'auc_task{i}'] = auc

        # Calculate accuracy
        prob = 1 / (1 + np.exp(-task_preds))  # sigmoid
        pred_labels = (prob > 0.5).astype(int)
        acc = (pred_labels == task_targets.astype(int)).mean()
        metrics[f'acc_task{i}'] = acc

    return metrics

def main():
    parser = argparse.ArgumentParser(description='Test saved model checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--csv', type=str, required=True,
                      help='Path to the CSV file containing dataset information')
    parser.add_argument('--output', type=str, default='test_results',
                      help='Output directory for test results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output, 'test.log'))
        ]
    )

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load checkpoint and metadata
    model_state, metadata = load_checkpoint(args.checkpoint, device)
    label_cols = metadata['label_cols']
    model_config = metadata['model_config']

    # Create model
    model = CombinedModel(
        chunk_feat_dim=768,
        hidden_dim=1024,
        num_tasks=len(label_cols),
        num_attn_heads=model_config['num_attn_heads'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate']
    )

    # Load state dict
    model.load_state_dict(model_state)
    model = model.to(device)

    # Create test dataset and dataloader
    processor = VolumeProcessor(chunk_depth=3, out_size=(448, 448))
    test_df = pd.read_csv(args.csv).query("split == 'test'")
    
    test_ds = NLSTDataset(
        df=test_df,
        processor=processor,
        label_cols=label_cols
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: b[0]
    )

    # Evaluate model
    logging.info("Starting evaluation on test set...")
    metrics = evaluate_model(model, test_loader, device, label_cols)

    # Print and save results
    logging.info("\nTest Results:")
    for task_idx in range(len(label_cols)):
        auc_key = f'auc_task{task_idx}'
        acc_key = f'acc_task{task_idx}'
        if auc_key in metrics:
            logging.info(f"Task {task_idx}:")
            logging.info(f"  AUC: {metrics[auc_key]:.4f}")
            logging.info(f"  Accuracy: {metrics[acc_key]:.4f}")

    # Save results to file
    results_file = os.path.join(args.output, 'test_metrics.json')
    with open(results_file, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    logging.info(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
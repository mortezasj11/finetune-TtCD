# 
import os
import torch
import pandas as pd
import numpy as np
import argparse
import json
from sklearn.metrics import roc_auc_score, accuracy_score
import time
from tqdm import tqdm

# Import utilities from local model_utils.py file
from model_utils import (
    DinoVisionTransformer, 
    CombinedModel, 
    VolumeProcessor, 
    NLSTDataset,
    MetricsCalculator
)

def materialize_tokens_(m, dev):
    """Initialize empty meta tokens with real values"""
    with torch.no_grad():
        for n in ["cls_token", "register_tokens", "mask_token"]:
            if hasattr(m, n) and getattr(m, n).storage().size() == 0:
                real = torch.zeros_like(getattr(m, n), device=dev)
                torch.nn.init.normal_(real, std=0.02)
                setattr(m, n, torch.nn.Parameter(real, requires_grad=True))

def load_combined_model(checkpoint_path, metadata_path=None):
    """Load a trained model from checkpoint"""
    # Import required modules
    from functools import partial
    from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
    
    # Load metadata to get model configuration
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Extract model configuration from metadata
        config = metadata['model_config']
        num_tasks = len(metadata['label_cols'])
        num_attn_heads = config.get('num_attn_heads', 4)
        num_layers = config.get('num_layers', 2)
        dropout_rate = config.get('dropout_rate', 0.3)
        label_cols = metadata['label_cols']
        print(f"Loaded model config from metadata: {config}")
        print(f"Label columns: {label_cols}")
    else:
        # Default configuration if metadata is not available
        print("No metadata found, using default configuration")
        num_tasks = 6  # Default for NLST dataset
        num_attn_heads = 4
        num_layers = 2
        dropout_rate = 0.3
        label_cols = ['1-year-cancer', '2-year-cancer', '3-year-cancer', 
                       '4-year-cancer', '5-year-cancer', '6-year-cancer']
    
    # Create base model
    base_model = DinoVisionTransformer(
        img_size=448,
        patch_size=16,  # Use the same patch size as during training
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
    
    # Initialize cls_token
    base_model.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
    torch.nn.init.normal_(base_model.cls_token, std=0.02)
    
    # Materialize tokens on CPU first
    materialize_tokens_(base_model, torch.device('cpu'))
    
    # Create combined model
    model = CombinedModel(
        base_model=base_model,
        chunk_feat_dim=768,
        hidden_dim=1024,
        num_tasks=num_tasks,
        num_attn_heads=num_attn_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    # Load the state dict from checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model, label_cols

def evaluate_model(model, test_loader, device, label_cols):
    """Evaluate model on test set"""
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    # Processing stats
    total_samples = 0
    total_time = 0
    
    print("Starting evaluation on test set...")
    
    with torch.no_grad():
        for i, (chunks, labels, mask, spacing) in enumerate(tqdm(test_loader)):
            start_time = time.time()
            
            chunks = chunks.squeeze(1).to(device)
            
            # Apply max_chunks constraint
            if chunks.size(0) > args.max_chunks:
                mid_idx = chunks.size(0) // 2
                start_idx = max(0, mid_idx - args.max_chunks // 2)
                end_idx = min(chunks.size(0), start_idx + args.max_chunks)
                chunks = chunks[start_idx:end_idx]
            
            # Forward pass
            logits = model(chunks, spacing)
            
            # Store predictions and targets
            all_preds.append(logits.cpu().numpy())
            all_targets.append(labels)
            all_masks.append(mask)
            
            total_samples += 1
            total_time += time.time() - start_time
    
    avg_time_per_sample = total_time / max(1, total_samples)
    print(f"Evaluated {total_samples} samples")
    print(f"Average processing time per sample: {avg_time_per_sample:.4f} seconds")
    
    # Combine predictions and calculate metrics
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    masks = np.vstack(all_masks)
    
    # Calculate metrics for each task
    metrics = {}
    
    # Overall metrics
    metrics['total_samples'] = total_samples
    metrics['avg_time_per_sample'] = avg_time_per_sample
    
    # Per-task metrics
    for i, col_name in enumerate(label_cols):
        if i >= masks.shape[1]:
            continue
            
        valid_idx = masks[:, i].astype(bool)
        if np.sum(valid_idx) > 0:
            task_preds = preds[valid_idx, i]
            task_targets = targets[valid_idx, i]
            
            try:
                # AUC with raw logits
                auc = roc_auc_score(task_targets, task_preds)
                metrics[f'auc_{col_name}'] = auc
                
                # Accuracy with sigmoid threshold
                pred_probs = 1 / (1 + np.exp(-task_preds))  # sigmoid
                pred_labels = (pred_probs > 0.5).astype(int)
                acc = accuracy_score(task_targets, pred_labels)
                metrics[f'accuracy_{col_name}'] = acc
                
                # Add metrics for task-specific interpretation
                true_pos = np.sum((pred_labels == 1) & (task_targets == 1))
                true_neg = np.sum((pred_labels == 0) & (task_targets == 0))
                false_pos = np.sum((pred_labels == 1) & (task_targets == 0))
                false_neg = np.sum((pred_labels == 0) & (task_targets == 1))
                
                # Sensitivity (recall), specificity, precision
                sensitivity = true_pos / max(1, true_pos + false_neg)
                specificity = true_neg / max(1, true_neg + false_pos)
                precision = true_pos / max(1, true_pos + false_pos)
                
                metrics[f'sensitivity_{col_name}'] = sensitivity
                metrics[f'specificity_{col_name}'] = specificity
                metrics[f'precision_{col_name}'] = precision
                
                # Add counts for context
                metrics[f'positive_samples_{col_name}'] = np.sum(task_targets)
                metrics[f'total_valid_samples_{col_name}'] = np.sum(valid_idx)
                
            except Exception as e:
                print(f"Error calculating metrics for {col_name}: {e}")
                metrics[f'auc_{col_name}'] = float('nan')
                metrics[f'accuracy_{col_name}'] = float('nan')
    
    return metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata if available
    metadata_path = os.path.join(os.path.dirname(args.checkpoint), 
                                 f"metadata_epoch_{args.epoch}.json" if args.epoch else "metadata_final.json")
    
    # Load model
    model, label_cols = load_combined_model(args.checkpoint, metadata_path)
    model = model.to(device)
    print(f"Model loaded from {args.checkpoint}")
    
    # Create dataset
    processor = VolumeProcessor(chunk_depth=3, out_size=(448, 448))
    
    # Load dataframe and filter test set
    full_df = pd.read_csv(args.csv)
    test_df = full_df[full_df['split'] == 'test'].copy()
    
    print(f"Test set size: {len(test_df)} samples")
    
    # Print class distribution
    print("\n==== Test Set Statistics ====")
    for col in label_cols:
        if col in test_df.columns:
            pos_count = sum(test_df[col] == 1)
            pos_pct = 100 * pos_count / len(test_df) if len(test_df) > 0 else 0
            print(f"{col}: {pos_count} positive samples ({pos_pct:.2f}%)")
    print("============================\n")
    
    # Create test dataset
    test_ds = NLSTDataset(
        df=test_df,
        processor=processor,
        label_cols=label_cols
    )
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0]
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, label_cols)
    
    # Print results
    print("\n==== Test Results ====")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Average time per sample: {metrics['avg_time_per_sample']:.4f} seconds")
    
    # Print task-specific metrics
    for col in label_cols:
        print(f"\n{col}:")
        if f'auc_{col}' in metrics:
            print(f"  AUC: {metrics[f'auc_{col}']:.4f}")
            print(f"  Accuracy: {metrics[f'accuracy_{col}']:.4f}")
            print(f"  Sensitivity: {metrics[f'sensitivity_{col}']:.4f}")
            print(f"  Specificity: {metrics[f'specificity_{col}']:.4f}")
            print(f"  Precision: {metrics[f'precision_{col}']:.4f}")
            print(f"  Positive samples: {metrics[f'positive_samples_{col}']} " +
                  f"of {metrics[f'total_valid_samples_{col}']} " +
                  f"({100 * metrics[f'positive_samples_{col}'] / metrics[f'total_valid_samples_{col}']:.2f}%)")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults saved to {results_path}")
    
    # Generate a summary report for easy sharing
    summary_path = os.path.join(args.output_dir, 'test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("==========================================\n")
        f.write("           TEST RESULTS SUMMARY          \n")
        f.write("==========================================\n\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Test set: {args.csv} (split='test')\n")
        f.write(f"Samples evaluated: {metrics['total_samples']}\n")
        f.write(f"Processing time: {metrics['avg_time_per_sample']:.4f} seconds per sample\n\n")
        
        f.write("------------------------------------------\n")
        f.write("RESULTS BY TASK:\n")
        f.write("------------------------------------------\n\n")
        
        for col in label_cols:
            f.write(f"{col}:\n")
            if f'auc_{col}' in metrics:
                f.write(f"  AUC: {metrics[f'auc_{col}']:.4f}\n")
                f.write(f"  Accuracy: {metrics[f'accuracy_{col}']:.4f}\n")
                f.write(f"  Sensitivity: {metrics[f'sensitivity_{col}']:.4f}\n")
                f.write(f"  Specificity: {metrics[f'specificity_{col}']:.4f}\n")
                f.write(f"  Precision: {metrics[f'precision_{col}']:.4f}\n")
                f.write(f"  Class distribution: {metrics[f'positive_samples_{col}']} positive / ")
                f.write(f"{metrics[f'total_valid_samples_{col}']} total ")
                f.write(f"({100 * metrics[f'positive_samples_{col}'] / metrics[f'total_valid_samples_{col}']:.2f}%)\n\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model on the test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g., output_ddp_full/checkpoints/model_final.pt)")
    parser.add_argument("--csv", type=str, 
                        default="/rsrch1/ip/msalehjahromi/codes/FineTune/nlst_event_train_val_test_.csv",
                        help="Path to the CSV file containing dataset information")
    parser.add_argument("--output-dir", type=str, default="./test_results",
                        help="Directory to save test results")
    parser.add_argument("--max-chunks", type=int, default=66,
                        help="Maximum number of chunks to process per sample")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch to evaluate (if not using final model)")
    
    args = parser.parse_args()
    main(args) 
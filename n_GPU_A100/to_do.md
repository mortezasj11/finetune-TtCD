# FineTune Training Roadmap

## Current Training Improvements
- [x] **Differential Learning Rates**
  - Separate learning rates for base model and aggregator
- [x] **Spacing Integration**
  - Added voxel spacing information to feature vectors
  - Transformed model's feature dimension from 768 to 771
- [x] **Data Augmentation**
  - Implemented random flips, rotations, scaling, and Gaussian 
- [ ] **Load the Latest Trained Dinov2**

- [ ] **Class Balancing**
  - Implemented weighted loss for handling imbalanced classes
  - *TODO:* Give more sampling weight to positive cases in data loader
- [ ] **Gradient Aggregation**
  - Sum gradients across all GPUs for better training stability
- [ ] **Attention Loss**
  - Implement additional guidance for attention mechanism
- [ ] **Model Preservation**
  - Save model checkpoints at regular intervals and after training
- [ ] **Final Evaluation**
  - Evaluate on test set after training completes
  - Remove the 100-sample limitation for validation

## Future Research Directions
1. **Nodule-based Approach**
   - Apply nnUNet on cases with nodules
   - Finetune based on nodule attention maps
   - Test model understanding
   - Finetune with NLST loss

2. **Loss Function Research**
   - Explore alternative losses from Sybil
   -[x] Investigate maximum-based data aggregation 

## Key Hyperparameters
- `epochs`: Number of training cycles
- `accum-steps`: Gradient accumulation steps
- `max-chunks`: Maximum number of chunks per sample
- `batch-size`: Samples per forward pass
- `lr`: Learning rate
- `warmup-steps`: Steps for learning rate warmup

# Remove 100 in validation now that it works. 


1. Apply nnUNET on cases that have nodules.
2. Finetune based on nodule attention map only.
3. Test if it has learned.
4. Finetune based on NLST loss. 

# Losses
1. Read other losses in Sybil
2. Also some ~max for data aggregation.




# Main Hyperparams for training:
    epochs
    accum-steps
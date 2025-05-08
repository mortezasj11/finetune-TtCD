# DDP Training Flow Structure

## Core Components

```python
# Initialize distributed environment
dist.init_process_group(backend="nccl")  # Setup multi-GPU communication
local_rank = int(os.environ["LOCAL_RANK"])  # Get GPU ID for this process
world_size = dist.get_world_size()  # Get total number of GPUs

# Create datasets and samplers 
train_ds = NLSTDataset(...)  # Dataset class loads CT volumes and labels
train_sampler = DistributedSampler(...)  # Ensures each GPU processes different data
train_loader = DataLoader(...)  # Provides batches of data to each GPU

# Create and setup model
model = CombinedModel(...)  # Combines ViT backbone with transformer aggregator
model = model.to(device)  # Move model to GPU
model = DDP(model, ...)  # Wrap model for distributed training

# Configure optimization
optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': base_lr},  # Lower LR for backbone
    {'params': agg_params, 'lr': main_lr}    # Higher LR for aggregator
])  # Different learning rates for different parts

# Training coordinator
trainer = DDPFullTrainer(...)  # Manages the training process
trainer.fit(...)  # Runs the training loop
```

## Class Structure

```python
class DDPFullTrainer:
    def __init__(self, model, optimizer, criterion, device, ...):
        # Initializes trainer with model, optimizer, etc.
        # Setup metrics tracking, including running accuracy metrics
        
    def reset_running_metrics(self):
        # Reset running accuracy counters for a new epoch
        
    def _train_step(self, chunks, labels, mask, spacing):
        # Process one batch: forward pass, loss calculation, backprop
        # Track running accuracy metrics for each task
        
    def evaluate(self, val_loader):
        # Run evaluation on validation set
        # Calculate accurate metrics with proper sigmoid thresholding
        
    def _run_validation(self, val_loader):
        # Wrapper for evaluation with timing and logging
        
    def _handle_unfreezing(self, epoch, unfreeze_strategy):
        # Control parameter freezing based on epoch and strategy
        
    def fit(self, train_loader, train_sampler, val_loader, epochs, unfreeze_strategy):
        # Main training loop across epochs
        # Reset running metrics at the start of each epoch
```

## Dataset and Model Classes

```python
class NLSTDataset(Dataset):
    def __init__(self, df, processor, label_cols):
        # Initialize dataset with dataframe, processor, and label columns
        
    def __len__(self):
        # Return number of samples
        
    def __getitem__(self, idx):
        # Process and return one sample: (chunks, labels, mask, spacing)
        # spacing contains voxel dimensions (dx, dy, dz)
```

```python
class CombinedModel(nn.Module):
    def __init__(self, base_model, chunk_feat_dim, hidden_dim, num_tasks, ...):
        # Initialize with ViT backbone and transformer aggregator
        # Extended dimension to incorporate spacing information
        
    def forward(self, x, spacing=[1., 1., 1.]):
        # Process input through backbone and aggregator
        # Incorporate spacing information into feature vectors
        
    def freeze_base_model(self):
        # Disable gradient updates for backbone
        
    def unfreeze_base_model(self):
        # Enable gradient updates for backbone
        
    def unfreeze_last_n_blocks(self, n):
        # Selectively unfreeze last n blocks of backbone
```

# DDP Training Flow with Tensor Shapes

## Initialization and Setup

```python
# Initialize distributed environment
initialize_distributed_process_group()  # Sets up communication between GPUs

# Each GPU process does the following:
rank = get_current_process_rank()            # scalar (0, 1, 2, 3 for 4 GPUs)
local_device = get_local_gpu_device(rank)    # device object (cuda:0, cuda:1, etc)
world_size = get_world_size()                # scalar (4 for 4 GPUs)

# Load data with sharding
dataset = load_dataset()                     # N total samples
# Each GPU gets a different subset of data (~N/world_size samples per GPU)
data_sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=B)  # Each GPU sees B samples per batch

# Create model on local GPU (DinoVisionTransformer + Aggregator in this case)
model = create_model()                       # ~100M parameters
model = model.to(local_device)               # Same ~100M parameters but on GPU memory

# Wrap model with DDP
model = DistributedDataParallel(
    model, 
    device_ids=[local_device],
    output_device=local_device
)
# Now each GPU has a full copy of the model (~100M parameters on each GPU)

# Create optimizer and criterion
optimizer = create_optimizer(model.parameters())  # Optimizes all ~100M parameters
criterion = create_loss_function()                # BCEWithLogitsLoss with class weights
```

## Training Process

```python
# Training function
def train(model, dataloader, optimizer, criterion, epoch, global_step, accum_steps=4):
    # Reset metrics and running accuracy counters
    metrics_calculator.reset()
    reset_running_metrics()
    
    # Set model to training mode
    model.train()
    
    # For each batch in the dataloader
    for batch_idx, (chunks, labels, mask, spacing) in enumerate(dataloader):
        # Process batch - shape: [B, C, H, W] where:
        # B = batch size (often 1 for medical volumes)
        # C = number of slices in volume (variable, often 10-50)
        # H, W = height, width of CT slice (448x448)
        # spacing = physical voxel dimensions (dx, dy, dz) in mm
        chunks = chunks.to(local_device)       # Shape: [1, C, 3, 448, 448]
        target = torch.tensor(labels, device=local_device)  # Shape: [num_tasks] (e.g., [6] for 6 time points)
        mask_tensor = torch.tensor(mask, device=local_device)  # Shape: [num_tasks] (boolean mask)
        spacing_tensor = torch.tensor(spacing, device=local_device)  # Shape: [3] (physical dimensions)
        
        # For large volumes, sample a subset of slices to fit in memory
        if chunks.size(1) > max_chunks:        # If too many slices
            # Center-based selection strategy
            mid_idx = chunks.size(1) // 2
            start_idx = max(0, mid_idx - max_chunks // 2)
            end_idx = min(chunks.size(1), start_idx + max_chunks)
            chunks = chunks[:, start_idx:end_idx]  # Shape: [1, max_chunks, 3, 448, 448]
        
        # Forward pass - different data on each GPU but same model architecture
        # spacing is incorporated into the feature vector
        outputs = model(chunks, spacing)        # Shape: [1, num_tasks]
        
        # Calculate loss
        loss = 0.0
        for i in range(outputs.size(1)):
            if mask_tensor[i]:                 # Only calculate loss for valid tasks
                task_loss = criterion(outputs[0, i:i+1], target[i:i+1])
                loss += task_loss
        
        # Normalize loss by number of active tasks
        active_tasks = mask_tensor.sum().item()
        if active_tasks > 0:
            loss = loss / active_tasks
        
        # Backward pass - gradients calculated locally then automatically synchronized
        loss.backward()                        # Gradients shape matches parameter shapes
        
        # Update metrics
        with torch.no_grad():
            for j in range(outputs.size(1)):
                if mask_tensor[j]:
                    # Calculate whether prediction was correct
                    probs = torch.sigmoid(outputs[0, j])
                    pred = (probs > 0.5).float()
                    is_correct = (pred == target[j]).float().item()
                    
                    # Update running accuracy metrics
                    if is_correct:
                        correct_predictions[f'task{j}'] += 1
                    total_predictions[f'task{j}'] += 1
        
        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate AFTER optimizer step
        
        # Log metrics periodically
        if rank == 0 and (global_step + 1) % print_every == 0:
            # Calculate running accuracies for each task
            running_accuracies = {}
            for j in range(num_tasks):
                if total_predictions[f'task{j}'] > 0:
                    acc = correct_predictions[f'task{j}'] / total_predictions[f'task{j}']
                    running_accuracies[f'acc_task{j}'] = acc
            
            # Log metrics
            metrics = {
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[1],  # Higher LR for aggregator
                **running_accuracies
            }
            log_metrics(metrics, step=global_step, epoch=epoch)
        
        global_step += 1
    
    return global_step
```

## Evaluation Process

```python
# Evaluation function
def evaluate(model, val_loader, criterion):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize tracking variables
    samples = 0
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_masks = []
    
    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for i, (chunks, labels, mask, spacing) in enumerate(val_loader):
            # Process batch
            chunks = chunks.squeeze(1).to(device)
            
            # Apply max_chunks constraint
            if chunks.size(0) > args.max_chunks:
                mid_idx = chunks.size(0) // 2
                start_idx = max(0, mid_idx - args.max_chunks // 2)
                end_idx = min(chunks.size(0), start_idx + args.max_chunks)
                chunks = chunks[start_idx:end_idx]
            
            # Forward pass
            logits = model(chunks, spacing)
            
            # Calculate loss
            target = torch.tensor(labels, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
            
            loss = 0.0
            for j in range(logits.size(1)):
                if mask_tensor[j]:
                    task_loss = criterion(logits[0, j:j+1], target[j:j+1])
                    loss += task_loss
            
            # Normalize loss
            active_tasks = mask_tensor.sum().item()
            if active_tasks > 0:
                loss = loss / active_tasks
                total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(logits.cpu().numpy())
            all_targets.append(labels)
            all_masks.append(mask)
            
            samples += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(1, samples)
    metrics = {'samples_evaluated': samples, 'avg_loss': avg_loss}
    
    # Combine predictions and calculate metrics
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    masks = np.vstack(all_masks)
    
    for i in range(preds.shape[1]):
        valid_idx = masks[:, i].astype(bool)
        if np.sum(valid_idx) > 0:
            task_preds = preds[valid_idx, i]
            task_targets = targets[valid_idx, i]
            
            # Calculate AUC using raw logits
            auc = roc_auc_score(task_targets, task_preds)
            metrics[f'auc_task{i}'] = auc
            
            # Calculate accuracy with proper sigmoid threshold
            pred_probs = 1 / (1 + np.exp(-task_preds))  # sigmoid
            pred_labels = (pred_probs > 0.5).astype(int)
            acc = np.mean(pred_labels == task_targets)
            metrics[f'acc_task{i}'] = acc
    
    return metrics
```

## Full Training Loop

```python
# Training loop
def fit(train_loader, train_sampler, val_loader, epochs, unfreeze_strategy):
    global_step = 0
    
    for epoch in range(epochs):
        # Set epoch for sampler to ensure different data shuffling per epoch
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Reset running metrics for the new epoch
        reset_running_metrics()
        
        # Handle unfreezing strategy based on epoch
        if unfreeze_strategy == 'gradual':
            if epoch == 0:
                freeze_base_model()  # Freeze ViT in first epoch
            elif epoch == 1:
                unfreeze_last_n_blocks(2)  # Unfreeze last 2 blocks
            elif epoch == 2:
                unfreeze_last_n_blocks(4)  # Unfreeze more blocks
            elif epoch >= 3:
                unfreeze_base_model()  # Unfreeze everything
        
        # Train for one epoch
        global_step = train(model, train_loader, optimizer, criterion, epoch, global_step)
        
        # Only rank 0 performs validation
        if rank == 0 and global_step % val_every == 0:
            # Free up memory before validation
            gc.collect()
            torch.cuda.empty_cache()
            
            # Run validation
            val_metrics = evaluate(model, val_loader, criterion)
            
            # Log validation metrics
            log_metrics(val_metrics, step=global_step, epoch=epoch, phase="validation")

# Main execution
for epoch in range(num_epochs):
    data_sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
    fit(train_loader, data_sampler, val_loader, epochs, unfreeze_strategy)
    
# Only rank 0 saves the final model
if rank == 0:
    save_final_model()                       # Model weights (~100M parameters)

# Clean up
destroy_process_group()
```

## Memory Management and Efficiency

1. **Gradient Accumulation**:
   - Effectively increases batch size without increasing memory usage
   - Each GPU accumulates gradients for `accum_steps` batches before updating
   - Helps stabilize training with limited GPU memory

2. **Slice Limitation**:
   - Large medical volumes with many slices are subsampled if they exceed memory limits
   - Center-based chunk selection using configurable `max_chunks` parameter
   - Prevents OOM errors while retaining relevant information

3. **Validation Strategy**:
   - Only performed on rank 0 to avoid redundant computation
   - Limited to 100 samples from validation set for efficiency
   - Proper metrics calculation using sigmoid for proper threshold comparison
   - Memory tracking before and after validation to monitor usage

4. **Model Parameter Groups**:
   - Different learning rates for backbone vs. aggregator
   - Backbone (ViT) uses smaller learning rate (1e-6)
   - Aggregator uses larger learning rate (3e-5)
   - Helps fine-tune pretrained features while training new components

5. **Spacing Information Integration**:
   - Physical voxel dimensions (dx, dy, dz) are now incorporated into the model features
   - Feature vectors are extended from 768 to 771 dimensions to include spacing
   - Allows the model to account for different scanning resolutions

6. **Accurate Metrics Tracking**:
   - Running accuracy metrics maintained throughout training
   - Proper sigmoid transformation and thresholding for binary classification
   - Separate training and validation metrics with AUC calculation
   - Metrics logged to disk for later analysis

This approach maximizes GPU utilization while ensuring stable training across different device configurations and scanning protocols.
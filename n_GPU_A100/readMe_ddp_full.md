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
    # Reset metrics
    metrics_calculator.reset()
    
    # Set model to training mode
    model.train()
    
    # For each batch in the dataloader
    for batch_idx, (chunks, labels, mask) in enumerate(dataloader):
        # Process batch - shape: [B, C, H, W] where:
        # B = batch size (often 1 for medical volumes)
        # C = number of slices in volume (variable, often 10-50)
        # H, W = height, width of CT slice (448x448)
        chunks = chunks.to(local_device)       # Shape: [1, C, 3, 448, 448]
        target = torch.tensor(labels, device=local_device)  # Shape: [num_tasks] (e.g., [3] for 3 time points)
        mask_tensor = torch.tensor(mask, device=local_device)  # Shape: [num_tasks] (boolean mask)
        
        # For large volumes, sample a subset of slices to fit in memory
        if chunks.size(1) > max_chunks:        # If too many slices
            chunks = chunks[:, start:end]      # Shape: [1, max_chunks, 3, 448, 448]
        
        # Forward pass - different data on each GPU but same model architecture
        outputs = model(chunks)                # Shape: [1, num_tasks]
        
        # Calculate loss
        loss = 0.0
        for i in range(outputs.size(1)):
            if mask_tensor[i]:                 # Only calculate loss for valid tasks
                task_loss = criterion(outputs[0, i:i+1], target[i:i+1])
                loss += task_loss
        
        # Normalize loss by number of active tasks and accumulation steps
        active_tasks = mask_tensor.sum().item()
        if active_tasks > 0:
            loss = loss / active_tasks
        
        # Gradient accumulation for larger effective batch size
        loss = loss / accum_steps              # Scale loss by accumulation steps
        
        # Backward pass - gradients calculated locally then automatically synchronized
        loss.backward()                        # Gradients shape matches parameter shapes
        
        # Update metrics
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Shape: [1, num_tasks]
        metrics_calculator.update_metrics(predictions[0], target, mask_tensor, loss.item() * accum_steps)  # Store unscaled loss
        
        # Step optimizer after accumulation steps
        if (global_step + 1) % accum_steps == 0:
            optimizer.step()                   # Updates all ~100M parameters
            optimizer.zero_grad()
        
        # Log metrics periodically
        if rank == 0 and (global_step + 1) % print_every == 0:
            metrics = metrics_calculator.get_metrics()
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
    
    # Initialize metrics
    total_loss = 0.0
    correct_1yr = 0
    total_1yr = 0
    correct_3yr = 0
    total_3yr = 0
    correct_6yr = 0
    total_6yr = 0
    
    # Counts for positive/negative samples
    pos_1yr, neg_1yr = 0, 0
    pos_3yr, neg_3yr = 0, 0
    pos_6yr, neg_6yr = 0, 0
    
    # Collect predictions for AUC calculation
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():  # Disable gradient calculation
        for chunks, labels, mask in val_loader:
            # Process batch same as in training
            chunks = chunks.to(local_device)       # Shape: [1, C, 3, 448, 448]
            target = torch.tensor(labels, device=local_device)  # Shape: [num_tasks]
            mask_tensor = torch.tensor(mask, device=local_device)  # Shape: [num_tasks]
            
            # Handle large volumes
            if chunks.size(1) > max_chunks:
                chunks = chunks[:, start:end]  # Shape: [1, max_chunks, 3, 448, 448]
            
            # Forward pass
            outputs = model(chunks)            # Shape: [1, num_tasks]
            
            # Get binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Shape: [1, num_tasks]
            
            # Calculate accuracy for each time point
            if mask_tensor[0]:  # 1-year
                is_correct = (predictions[0, 0] == target[0]).float().item()
                correct_1yr += is_correct
                total_1yr += 1
                if target[0].item() == 1:
                    pos_1yr += 1
                else:
                    neg_1yr += 1
            
            if mask_tensor[1]:  # 3-year
                is_correct = (predictions[0, 1] == target[1]).float().item()
                correct_3yr += is_correct
                total_3yr += 1
                if target[1].item() == 1:
                    pos_3yr += 1
                else:
                    neg_3yr += 1
            
            if mask_tensor[2]:  # 6-year
                is_correct = (predictions[0, 2] == target[2]).float().item()
                correct_6yr += is_correct
                total_6yr += 1
                if target[2].item() == 1:
                    pos_6yr += 1
                else:
                    neg_6yr += 1
            
            # Store predictions and targets for AUC calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels)
            all_masks.append(mask)
    
    # Calculate metrics
    metrics = {
        "acc1": correct_1yr / max(1, total_1yr),
        "acc3": correct_3yr / max(1, total_3yr),
        "acc6": correct_6yr / max(1, total_6yr),
        "pos_count_1yr": f"{pos_1yr}-{neg_1yr}",
        "pos_count_3yr": f"{pos_3yr}-{neg_3yr}",
        "pos_count_6yr": f"{pos_6yr}-{neg_6yr}",
    }
    
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
        if rank == 0:
            val_metrics = evaluate(model, val_loader, criterion)
            log_metrics(val_metrics, step=global_step, epoch=epoch, phase="validation")
            save_checkpoint(model, optimizer, epoch, global_step, val_metrics)

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
   - Typically centers the sampling window around the middle of the volume
   - Prevents OOM errors while retaining most relevant information

3. **Validation Strategy**:
   - Only performed on rank 0 to avoid redundant computation
   - Small validation subset used to quickly assess model quality
   - AUC and accuracy metrics calculated for each prediction task

4. **Model Parameter Groups**:
   - Different learning rates for backbone vs. aggregator
   - Backbone (ViT) uses smaller learning rate (1e-6)
   - Aggregator uses larger learning rate (3e-5)
   - Helps fine-tune pretrained features while training new components

This approach maximizes GPU utilization while ensuring stable training across different device configurations.
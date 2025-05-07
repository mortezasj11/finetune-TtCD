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
        
    def _train_step(self, chunks, labels, mask):
        # Process one batch: forward pass, loss calculation, backprop
        
    def evaluate(self, val_loader):
        # Run evaluation on validation set
        
    def _run_validation(self, val_loader):
        # Wrapper for evaluation with timing and logging
        
    def _handle_unfreezing(self, epoch, unfreeze_strategy):
        # Control parameter freezing based on epoch and strategy
        
    def fit(self, train_loader, train_sampler, val_loader, epochs, unfreeze_strategy):
        # Main training loop across epochs
```

## Dataset and Model Classes

```python
class NLSTDataset(Dataset):
    def __init__(self, df, processor, label_cols):
        # Initialize dataset with dataframe, processor, and label columns
        
    def __len__(self):
        # Return number of samples
        
    def __getitem__(self, idx):
        # Process and return one sample: (chunks, labels, mask)
```

```python
class CombinedModel(nn.Module):
    def __init__(self, base_model, chunk_feat_dim, hidden_dim, num_tasks, ...):
        # Initialize with ViT backbone and transformer aggregator
        
    def forward(self, x):
        # Process input through backbone and aggregator
        
    def freeze_base_model(self):
        # Disable gradient updates for backbone
        
    def unfreeze_base_model(self):
        # Enable gradient updates for backbone
        
    def unfreeze_last_n_blocks(self, n):
        # Selectively unfreeze last n blocks of backbone


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
    
    # Initialize tracking variables
    samples = 0
    print(f"Starting evaluation on validation set...")
    
    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for i, (chunks, labels, mask) in enumerate(val_loader):
            # Process batch
            try:
                # Move data to device and squeeze batch dimension
                chunks = chunks.squeeze(1).to(device)
                
                # Limit chunks if needed to prevent OOM errors
                max_chunks = 28
                if chunks.size(0) > max_chunks:
                    chunks = chunks[:max_chunks]
                
                # Forward pass
                _ = model(chunks)
                samples += 1
                
                # Print progress
                if i % 10 == 0:
                    print(f"Processed {i+1} validation samples")
            except Exception as e:
                print(f"Error in validation sample {i}: {str(e)}")
                continue
    
    # Return basic metrics
    elapsed = time.time() - start_time
    print(f"Evaluated {samples} samples in {elapsed:.2f}s")
    
    # For now, using placeholder values for metrics
    # This can be extended to calculate actual metrics if needed
    metrics = {
        'samples_evaluated': samples,
        'eval_time_seconds': elapsed,
        'acc1': 0.0,  # Placeholder values 
        'acc3': 0.0,
        'acc6': 0.0
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
        if rank == 0 and global_step % val_every == 0:
            # Free up memory before validation
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log memory usage before validation
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            print(f"Memory before validation: {mem_allocated:.2f}MB allocated, {mem_reserved:.2f}MB reserved")
            
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
   - Typically limits to first 28 slices if volume is too large
   - Prevents OOM errors while retaining relevant information

3. **Validation Strategy**:
   - Only performed on rank 0 to avoid redundant computation
   - Limited to 100 samples from validation set for efficiency
   - Simple forward pass evaluation to check model functionality
   - Memory tracking before and after validation to monitor usage

4. **Model Parameter Groups**:
   - Different learning rates for backbone vs. aggregator
   - Backbone (ViT) uses smaller learning rate (1e-6)
   - Aggregator uses larger learning rate (3e-5)
   - Helps fine-tune pretrained features while training new components

This approach maximizes GPU utilization while ensuring stable training across different device configurations.
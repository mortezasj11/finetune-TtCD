# Kubernetes GPU OOM Troubleshooting

## Pod resource limits
**Problem**: K8s kills the container when it peaks above the GPU memory request.  
**Symptom**: Pod goes OOMKilled even though "steady-state" fits.  
**Solution**: In the pod spec:
```yaml
resources:
  limits:
    nvidia.com/gpu-memory: 40Gi  # k8s >= 1.27
```
Or disable the memory-based eviction rule.

## Container image / driver mismatch
**Problem**: The pod uses a newer CUDA or debug build of PyTorch that keeps extra bookkeeping arenas.  
**Symptom**: Same script, same card, higher idle usage inside pod.  
**Solution**: Build the image with the same CUDA base & PyTorch wheel you use locally, or mount the host's `/usr/local/cuda` instead of the one in the image.

## DataLoader prefetch + large batch
**Problem**: Workers pin-memory → host-to-device transfers allocate staging buffers inside the GPU driver.  
**Symptom**: Memory climbs just before `forward()` is called and falls after `optimizer.step()`.  
**Solution**: Prefetch fewer batches:
```python
DataLoader(prefetch_factor=2, persistent_workers=False)
```

## cuDNN autotune / conv workspace spike
**Problem**: First forward pass asks cuDNN for large scratch buffers, then frees them.  
**Symptom**: One-off spike to almost 40 GB, then drops to steady 25 GB.  
**Solution**: If the spike alone trips OOM:
```python
torch.backends.cudnn.benchmark = False
```
or 
```bash
export CUDNN_WORKSPACE_LIMIT_MB=2048
```

## Mixed precision scaler growth
**Problem**: First few updates use FP32 master weights before they are cast to FP16.  
**Symptom**: Spike during the very first backward pass.  
**Solution**: Keep AMP but set:
```python
torch.cuda.amp.GradScaler(init_scale=2**10, growth_interval=1000)
```
to slow scale growth, or pre-allocate with a dry-run.
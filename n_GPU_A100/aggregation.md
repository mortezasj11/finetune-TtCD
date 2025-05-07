# Feature Aggregation in the Combined Model

## 1. Chunk-wise feature extraction & stacking

Assume `x` has shape `S,C,H,W`, where:
- `S` = number of chunks
- `C×H×W` = per-chunk image dimensions
- `D` = chunk_feat_dim (e.g. 768)

```python
features = []
for i in range(S):
    chunk = x[i].unsqueeze(0)          # → [1, C, H, W]
    with torch.no_grad():              # backbone frozen
        chunk_feat = self.base(chunk)  # → [1, D]
    features.append(chunk_feat)        # list of S tensors, each [1, D]

# Stack along the sequence dimension:
features = torch.cat(features, dim=0)  # → [S, D]
```

After concatenation: `features ∈ ℝ^{S×D}`

## 2. Sequence aggregation with Transformer

We want to treat our S chunk-features as a "sentence" of length S, each token of size D. PyTorch's TransformerEncoder (with batch_first=True) expects input shape [N, S, D], where N is batch size. Here N=1 (one patient):

```python
seq = features.unsqueeze(0)  # → [1, S, D]
agg = self.transformer(seq)  # → [1, S, D]
```

### Inside each Transformer encoder layer

Let:
- `N` = 1 (batch)
- `S` = sequence length (num_chunks)
- `D` = model dimension (chunk_feat_dim)
- `H` = num_attn_heads
- `dₕ` = D/H = head dimension
- `F` = hidden_dim (feed-forward inner size)

#### Multi-head self-attention

**QKV projection:**
- X ∈ ℝ^{N×S×D} → W_qkv · X → ℝ^{N×S×3D} → split into Q,K,V each ℝ^{N×S×D}

**Reshape for heads:**
- Q → ℝ^{N×H×S×dₕ}, same for K and V.

**Scaled dot-product:**
- Attention scores = Q·Kᵀ → ℝ^{N×H×S×S}
- Weights = softmax(scores)
- Context = weights·V → ℝ^{N×H×S×dₕ}

**Concat heads & project:**
- Reshape to ℝ^{N×S×D}, then linear projection back to D.
- Residual & LayerNorm → output MHA_out ∈ ℝ^{N×S×D}

#### Position-wise feed-forward

- FFN In: MHA_out ∈ ℝ^{N×S×D}
- Linear1: D → F → ℝ^{N×S×F}
- Activation (ReLU/GELU), dropout
- Linear2: F → D → ℝ^{N×S×D}
- Residual & LayerNorm → final layer output ∈ ℝ^{N×S×D}

Stacking `num_layers` of these gives you: `agg ∈ ℝ^{1×S×D}`

## 3. Global pooling & classification

```python
# Mean-pool over the chunk (time) dimension:
pooled = agg.mean(dim=1)    # → [1, D]

# Linear classifier to num_tasks:
outputs = self.classifier(pooled)  # → [1, num_tasks]
```

- `pooled ∈ ℝ^{1×D}`
- `classifier weight ∈ ℝ^{num_tasks×D}`
- `outputs ∈ ℝ^{1×num_tasks}`

## Summary of key shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| x | [S, C, H, W] | Input chunks |
| chunk_feat | [1, D] | Per-chunk embedding from backbone |
| features | [S, D] | Stacked chunk embeddings |
| seq | [1, S, D] | Batched sequence to transformer |
| agg | [1, S, D] | Transformer-encoded sequence |
| pooled | [1, D] | Global average over S chunks |
| outputs | [1, num_tasks] | Final predictions for each task |

This design lets the transformer learn relationships across chunks (e.g. spatial regions or slices) before pooling into a single patient-level representation.
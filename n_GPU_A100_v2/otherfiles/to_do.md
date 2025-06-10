# FineTune Training Roadmap

## Current Training Improvements
- [x] **Differential Learning Rates**
  - Separate learning rates for base model and aggregator
- [x] **Spacing Integration**
  - Added voxel spacing information to feature vectors
  - Transformed model's feature dimension from 768 to 771
- [x] **Data Augmentation**
  - Implemented random flips, rotations, scaling, and Gaussian 
- [x] **Load the Latest Trained Dinov2**
  - Tested!

- [ ] **Class Balancing**
  - Implemented weighted loss for handling imbalanced classes
  - *TODO:* Give more sampling weight to positive cases in data loader
- [ ] **Gradient Aggregation**
  - Sum gradients across all GPUs for better training stability
- [ ] **Attention Loss**
  - Implement additional guidance for attention mechanism
- [x] **Model Preservation**
  - Save model checkpoints at regular intervals and after training
  - not tested!
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


## Remove 100 in validation now that it works. 


1. Apply nnUNET on cases that have nodules.
2. Finetune based on nodule attention map only.
3. Test if it has learned.
4. Finetune based on NLST loss. 

## Losses
1. Read other losses in Sybil
2. Also some ~max for data aggregation.


## patch 32 trained: max-chunks: 
    jupyterhub : max_chunk=48 , accum=20  => start memory 19/40 ; on validation 6-8/40!
    K8s        : max_chunk=48 , accum=20  => memory Error !
    K8s        : max_chunk=48 , accum=10  => memory Error !

## So I have no choice than train it on our DGX


## Main Hyperparams for training
- min, max eps    # WRONG on my First Test
- epochs
- accum-steps
- num-workers
- which base to choose patch-size:16 or 32
    *can I run them with the one with patch-size 16 ?
- if overfitting, then what?  

## Train Test
### Key Hyperparameters  
-  **I need to remove 88 on Validation**
- `epochs`: 100
- `accum-steps`: at least 1000 ?
- `max-chunks`: 66 
- `lr`:  I need to increase it based on previous training, not sure
- `warmup-steps`: 5k
- `print-every` : 5000
- `val-every` : 40000


# running on our DGX
- docker run -it --rm --gpus '"device=4,5,6,7"' -p 12344:12344 --name msalehjahromi --shm-size=192G  --user $(id -u):$(id -g) --group-add 1944259512 --cpuset-cpus=49-96 -v /rsrch7/home/ip_rsrch/wulab/:/rsrch7/home/ip_rsrch/wulab -v /rsrch1/ip/msalehjahromi/:/rsrch1/ip/msalehjahromi --name mori_jupyter nnunetv2:msalehjahromi
- cd /rsrch1/ip/msalehjahromi/codes/FineTune/multiGPU/n_GPU_A100
- jupyter notebook --ip 0.0.0.0 --port 12344
- http://1mcprddgx05/
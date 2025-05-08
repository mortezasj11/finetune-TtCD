1. Diff lr for the model and mlp    tik
2. Add the spacing!                 tik
    - remember how CombinedModel works?    tik 
    - why             with torch.no_grad():  # Frozen backbone by default     #removed tik
                chunk_feat = self.base(x[i].unsqueeze(0))
3. Giving more chance for positive to be selected! in data loader.
4. Data augmentation.
5. Attention loss.
6. Saving the model after end.
7. Evaluate on test at the ened.
8. Sums all the GPU gradients ... 

# Remove 100 in validation now that it works. 


1. Apply nnUNET on cases that have nodules.
2. Finetune based on nodule attention map only.
3. Test if it has learned.
4. Finetune based on NLST loss. 

# Losses
1. Read other losses in Sybil
2. Also some ~max for data aggregation.




# Main Hyperparams for t
1. Add the spacing!
2. Giving more chance for positive to be selected! in data loader.
3. Data augmentation.
4. Attention loss.
5. Diff lr for the model and mlp
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
# The configs for training and  validation

We provide the configs of BEVNet, VoxelNet and OccNet

```
BEVNet: projects/configs/bevformer 
VoxelNet: projects/configs/voxelformer
OccNet: projects/configs/hybrid
```

# Train

Train model with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ.py 8  #  bevnet: occupancy
./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_det_occ.py 8  # bevnet: occupancy+detection
./tools/dist_train.sh ./projects/configs/hybrid/hybrid_tiny_occ.py 8  # occnet: occupancy
```

# Validation
Eval model with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/hybrid/hybrid_tiny_occ.py ./path/to/ckpts.pth 8
```
#!/bin/bash

# Enable and edit the following two lines of code in case you're using conda
# source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
# conda activate your_virtual_conda_env

# Enable and edit the following line of code in case you're using virtualenv
# source "/path/to/your/virtualenv/bin/activate"

# Insert the path to DSEC dataset
# DATA_PATH="/path/to/dsec"

# Insert the path to our pretrained weights previous downloaded
# WEIGHTS_PATH="/path/to/pretrained/weights"

# Select your CUDA GPU (Default: 0)
CUDA_DEVICE=0

mkdir -p tmp
mkdir -p results/dsec/retrained/vsh
mkdir -p results/dsec/retrained/bth


# vsh  DSEC

for ER in 'histogram' 'concentration' 'mdes' 'tore' 'voxelgrid' 'ergo12' 'timesurface' 'tencode'
do
    echo "Launching..."
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/inference.py --data_root $DATA_PATH  --checkpoint_path $WEIGHTS_PATH/vsh_raw/$ER.pth --save_root tmp --dataset dsec --raw_mae_threshold 0.5 --guide_method vsh --vsh_patch_size 3 --vsh_maskocc --vsh_uniform_patch --vsh_splatting --vsh_method rnd --vsh_blending 0.5 1>> results/dsec/retrained/vsh/${ER}.txt"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/inference.py --data_root $DATA_PATH  --checkpoint_path $WEIGHTS_PATH/vsh_raw/$ER.pth --save_root tmp --dataset dsec --raw_mae_threshold 0.5 --guide_method vsh --vsh_patch_size 3 --vsh_maskocc --vsh_uniform_patch --vsh_splatting --vsh_method rnd --vsh_blending 0.5 1>> results/dsec/retrained/vsh/${ER}.txt
    echo "Done."
done

# bth DSEC

for ER in 'histogram' 'concentration' 'mdes' 'tore' 'voxelgrid' 'ergo12' 'timesurface' 'tencode'
do
    echo "Launching..."
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/inference.py --data_root $DATA_PATH  --checkpoint_path $WEIGHTS_PATH/bth_raw/$ER.pth --save_root tmp --dataset dsec --raw_mae_threshold 0.5 --guide_method bth --bth_patch_size 3 --bth_maskocc --bth_method h-mdes --bth_n_events 2 --bth_uniform_polarities --bth_splatting none 1>> results/dsec/retrained/bth/${ER}.txt"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/inference.py --data_root $DATA_PATH  --checkpoint_path $WEIGHTS_PATH/bth_raw/$ER.pth --save_root tmp --dataset dsec --raw_mae_threshold 0.5 --guide_method bth --bth_patch_size 3 --bth_maskocc --bth_method h-mdes --bth_n_events 2 --bth_uniform_polarities --bth_splatting none 1>> results/dsec/retrained/bth/${ER}.txt
    echo "Done."
done

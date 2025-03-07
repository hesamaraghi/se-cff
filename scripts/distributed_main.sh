#!/bin/bash

set -x  # Enable command printing for debugging

# GPU and Training Configurations
cuda_idx='0,1'  # GPUs to use
config_path=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/configs/config.yaml
data_root=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/data/NSEK
save_root=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/saved_results_2_15_5M_256_beta_1
save_term=10
num_workers=8
NUM_PROC=2  # Number of processes for distributed training
# checkpoint_path=""  # Default checkpoint path
checkpoint_path=${save_root}/weights/final.pth  # Default checkpoint path

# WandB Configurations
use_wandb=true  # Set to false to disable wandb
wandb_project="se_cff"
wandb_run_name="saved_results_2_15_5M_256_beta_1"
wandb_run_id=""  # Leave empty to start a new run; set to resume

# Check if we should resume training
resume_flag=""
if [ -f "$checkpoint_path" ]; then
    echo "Checkpoint found at $checkpoint_path. Resuming training..."
    resume_flag="--resume --checkpoint-path ${checkpoint_path}"
fi

# Check if WandB should resume a previous run
wandb_args=""
if [ "$use_wandb" = true ]; then
    wandb_args="--use-wandb --wandb-project ${wandb_project} --wandb-run-name ${wandb_run_name}"
    if [ ! -z "$wandb_run_id" ]; then
        wandb_args+=" --wandb-run-id ${wandb_run_id}"  # Resume specific run
    fi
fi

# Launch Distributed Training
CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM src/distributed_main.py --config-path ${config_path} --data-root ${data_root} --save-root ${save_root} --num-workers ${num_workers} --save-term ${save_term} ${resume_flag} ${wandb_args}

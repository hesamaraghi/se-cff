#!/bin/bash

set -x

cuda_idx='0,1'
config_path=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/configs/config.yaml
data_root=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/data/NSEK
save_root=/home/nfs/maraghi/NSEK/Model_zoo/Depth_estimation/se-cff/saved_results_2_100_15M
save_term=10
num_workers=8
NUM_PROC=2

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM src/distributed_main.py --config-path ${config_path} --data-root ${data_root} --save-root ${save_root} --num-workers ${num_workers} --save-term ${save_term}

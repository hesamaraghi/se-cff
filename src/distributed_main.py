import os
import argparse
import torch
import numpy as np
import random
import wandb

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # If using multiple GPUs

# Set NumPy and Python random seeds
np.random.seed(0)
random.seed(0)

from manager import DLManager
from utils.config import get_cfg

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, default='/root/code/configs/config.yaml')
parser.add_argument('--data-root', type=str, default='/root/data/DSEC')
parser.add_argument('--save-root', type=str, default='/root/code/save')

parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--save-term', type=int, default=25)
parser.add_argument('--local-rank', type=int, default=-1)

# Wandb arguments
parser.add_argument('--use-wandb', action='store_true', help="Enable wandb logging")
parser.add_argument('--wandb-project', type=str, default="se_cff", help="Wandb project name")
parser.add_argument('--wandb-run-name', type=str, default=None, help="Wandb run name")
parser.add_argument('--wandb-run-id', type=str, default=None, help="Wandb run ID for resuming")

# Resume training arguments
parser.add_argument('--resume', action='store_true', help="Resume training from the last checkpoint")
parser.add_argument('--checkpoint-path', type=str, default='/root/code/save/latest.pth', help="Checkpoint file to resume training")

args = parser.parse_args()

# Ensure distributed training is enabled
assert int(os.environ['WORLD_SIZE']) >= 1
args.is_distributed = True
args.is_master = args.local_rank == 0
args.device = 'cuda:%d' % args.local_rank

# Set up distributed processing
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()
args.rank = torch.distributed.get_rank()

# ✅ Initialize WandB (if master node)
if args.is_master and args.use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="allow" if args.resume else None,
        sync_tensorboard=True
    )

# Ensure paths exist
assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

# ✅ Load Config
cfg = get_cfg(args.config_path)

# ✅ Initialize Experiment Manager
exp_manager = DLManager(args, cfg)

# ✅ Load Checkpoint if Resuming
if args.resume and os.path.exists(args.checkpoint_path):
    print(f"Resuming training from checkpoint: {args.checkpoint_path}")
    exp_manager.load(args.checkpoint_path)

# ✅ Start Training
exp_manager.train()

# ✅ Run Testing
exp_manager.test()

# ✅ Finish WandB Logging
if args.is_master and args.use_wandb:
    wandb.finish()

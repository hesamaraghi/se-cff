from pathlib import Path  # Correct import

data_dir = Path("data/NSEK")

test_dir = data_dir / "test"
train_dir = data_dir / "train"

DATA_SPLIT = {
    'train': [k.parent.name + '/' + k.name for k in train_dir.iterdir() if k.name.startswith('K')],  # Fixed startswith
    'validation': [],
    'trainval': [],
    'test': [k.parent.name + '/' + k.name  for k in test_dir.iterdir() if k.name.startswith('K')],  # Fixed startswith
    'none': [],
}

# print(DATA_SPLIT)
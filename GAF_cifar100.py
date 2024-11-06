import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import random
import wandb
import os
import argparse

# Make the script callable from the CLI and parse arguments

os.environ["WANDB_API_KEY"] = ""


# List of optimizer types
optimizer_types = ["SGD", "SGD+Nesterov", "SGD+Nesterov+val_plateau", "Adam", "AdamW", "RMSProp"]


parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-100 with various optimizers and GAF.')

# General training parameters
parser.add_argument('--GAF', type=bool, default=True, help='Enable Gradient Agreement Filtering')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
parser.add_argument('--weight_decay_type', type=str, default='l2', choices=['l1', 'l2'], help='Weight decay type')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_val_epochs', type=int, default=2, help='Number of epochs between validation checks')
parser.add_argument('--min_grad', type=float, default=0.0, help='Minimum gradient value for filtering')
parser.add_argument('--epsilon', type=float, default=1e-1, help='Epsilon for gradient agreement filtering')
parser.add_argument('--optimizer', type=str, default='SGD', choices=optimizer_types, help='Optimizer type')
parser.add_argument('--num_batches_to_force_agreement', type=int, default=10, help='Number of batches to force agreement (must be > 1)')
parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
parser.add_argument('--num_samples_per_class_per_batch', type=int, default=1, help='Number of samples per class per batch if we are doing GAF')
parser.add_argument('--label_error_percentage', type=float, default=0, help='Percent of the train labels in the dataset to flip to a wrong label')
parser.add_argument('--cos_distance_thresh', type=float, default=1, help='Angle threshold on the cosine similarity if using that type of agreement filtering')

# Optimizer-specific parameters
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum factor for SGD optimizer')
parser.add_argument('--nesterov', type=bool, default=False, help='Use Nesterov momentum')
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizers')
parser.add_argument('--alpha', type=float, default=0.99, help='Alpha value for RMSProp')
parser.add_argument('--centered', type=bool, default=False, help='Centered RMSProp')
parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')

# Parse arguments
args = parser.parse_args()
config = vars(args)

# Set unused optimizer-specific configs to 'NA'
optimizer = config['optimizer']
all_params = ['momentum', 'nesterov', 'betas', 'eps', 'alpha', 'centered', 'scheduler_patience']

# Define which parameters are used by each optimizer
optimizer_params = {
    'SGD': ['momentum', 'nesterov'],
    'SGD+Nesterov': ['momentum', 'nesterov'],
    'SGD+Nesterov+val_plateau': ['momentum', 'nesterov', 'scheduler_patience'],
    'Adam': ['betas', 'eps'],
    'AdamW': ['betas', 'eps'],
    'RMSProp': ['momentum', 'eps', 'alpha', 'centered'],
}

# Get the list of parameters used by the selected optimizer
used_params = optimizer_params.get(optimizer, [])

# Set unused parameters to 'NA'
for param in all_params:
    if param not in used_params:
        config[param] = 'NA'


# Example CLI commands for each optimizer type:
# For SGD:
# python GAF_cifar100.py --GAF True --learning_rate 0.01 --optimizer SGD --momentum 0.0 --nesterov '' --weight_decay 1e-4 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 3 --num_batches_to_force_agreement 3   --epsilon 1e1 --label_error_percentage 0.05 --cos_distance_thresh 1.0

# For SGD+Nesterov:
# python GAF_cifar100.py --GAF True --learning_rate 0.01 --optimizer "SGD+Nesterov"  --momentum 0.9 --nesterov True --weight_decay 1e-4 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 3   --epsilon 1e1 --label_error_percentage 0.05 --cos_distance_thresh 1.0

# For SGD+Nesterov+val_plateau:
# python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 10 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1

# For Adam:
# python  GAF_cifar100.py --GAF True --optimizer Adam --learning_rate 0.001 --betas 0.9 0.999 --eps 1e-8 --weight_decay 1e-4 --weight_decay_type l2 --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 3 --epsilon 1e1 --label_error_percentage 0.05 --cos_distance_thresh 1.0

# For AdamW:
# python GAF_cifar100.py --GAF True --optimizer AdamW --learning_rate 0.001 --betas 0.9 0.999 --eps 1e-8 --weight_decay 1e-4 --weight_decay_type l1 --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 3 --epsilon 1e1  --label_error_percentage 0.05 --cos_distance_thresh 0.4

# For RMSProp:
# python GAF_cifar100.py --GAF True --optimizer RMSProp --learning_rate 0.01 --alpha 0.99 --eps 1e-8 --weight_decay 1e-4 --momentum 0.0 --centered False --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 3 --label_error_percentage 0.05 --cos_distance_thresh 1.0




# Check device for the model device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_index = random.randint(0, num_gpus - 1)  # Pick a random device index
    device = torch.device(f"cuda:{device_index}")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)




#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 4 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 5 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 6 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 7 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 8 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 9 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1
#6 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 10 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1


#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 4 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 5 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 6 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 7 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 8 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 9 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1
#6 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 10 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1

#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 4 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 5 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 6 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 7 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 8 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 9 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1
#6 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 10 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1













#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.95
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.97
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.99
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1.01
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1.03
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1.05



#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 0.95
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 0.97
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 0.99
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1.01
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1.03
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.05 --cos_distance_thresh 1.05

#0 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 0.95
#1 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 0.97
#2 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 0.99
#3 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1.01
#4 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1.03
#5 python GAF_cifar100.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 3 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.15 --cos_distance_thresh 1.05


# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Create a mapping from class to indices for sampling
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)

# Function to sample IID minibatches for standard training
def sample_iid_mbs(full_dataset, class_indices, batch_size):
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    batch_indices = []
    for cls in class_indices:
        indices = random.sample(class_indices[cls], samples_per_class)
        batch_indices.extend(indices)
    # If batch_size is not divisible by num_classes, fill the rest randomly
    remaining = batch_size - len(batch_indices)
    if remaining > 0:
        all_indices = [idx for idx in range(len(full_dataset))]
        batch_indices.extend(random.sample(all_indices, remaining))
    # Create a Subset
    batch = Subset(full_dataset, batch_indices)
    return batch

def flip_labels(train_dataset, label_error_percentage=0.1, num_classes=100):
    
    num_samples = len(train_dataset.targets)
    num_to_flip = int(label_error_percentage * num_samples)
    all_indices = list(range(num_samples))
    flip_indices = random.sample(all_indices, num_to_flip)

    for idx in flip_indices:
        original_label = train_dataset.targets[idx]
        # Exclude the original label to ensure the label is actually changed
        wrong_labels = list(range(num_classes))
        wrong_labels.remove(original_label)
        new_label = random.choice(wrong_labels)
        train_dataset.targets[idx] = new_label

    return train_dataset
    
def sample_iid_mbs_for_GAF(full_dataset, class_indices, n):
    """
    Samples n independent minibatches, each containing an equal number of samples from each class.
    """
    # Initialize a list to hold indices for each batch
    batch_indices_list = [[] for _ in range(n)]
    for cls in class_indices:
        num_samples_per_class = 1  # Adjust if you want more samples per class per batch
        total_samples_needed = num_samples_per_class * n
        available_indices = class_indices[cls]
        if len(available_indices) < total_samples_needed:
            multiples = (total_samples_needed // len(available_indices)) + 1
            extended_indices = (available_indices * multiples)[:total_samples_needed]
        else:
            extended_indices = random.sample(available_indices, total_samples_needed)
        for i in range(n):
            start_idx = i * num_samples_per_class
            end_idx = start_idx + num_samples_per_class
            batch_indices_list[i].extend(extended_indices[start_idx:end_idx])
    # Create Subsets for each batch
    batches = [Subset(full_dataset, batch_indices) for batch_indices in batch_indices_list]
    return batches

# Gradient Agreement Filtering function (GAF) based on sign agreement
def filter_gradients_sign(G1, G2, epsilon=config['epsilon']):
    filtered_grad = []
    masked = []
    total = []
    for g1, g2 in zip(G1, G2):
        agree = torch.sign(g1) == torch.sign(g2)  # Direction agreement
        similar = torch.abs(g1 - g2) < epsilon    # Magnitude similarity
        big_enough = torch.abs(g1) > config['min_grad']
        mask = agree & similar & big_enough                  # Both conditions satisfied
        filtered_grad.append(mask.float() * (g1 + g2) / 2)  # Average gradients
        masked.append(torch.sum(mask.float()))
        total.append(torch.numel(mask))
    gaf_percentage = (sum(masked) / sum(total)).item() * 100
    return filtered_grad, gaf_percentage

# Gradient Agreement Filtering function (GAF) based on cosine sim
def filter_gradients_cosine_sim(G1, G2, cos_distance_thresh):
    # Flatten G1 and G2 into vectors
    G1_flat = torch.cat([g1.view(-1) for g1 in G1])
    G2_flat = torch.cat([g2.view(-1) for g2 in G2])

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(G1_flat, G2_flat, dim=0)

    # Compute cosine distance
    cos_distance = 1 - cos_sim

    # If cos_distance > cos_distance_thresh, filtered_grad == None
    if cos_distance > cos_distance_thresh:
        filtered_grad = None
    else:
        # Compute average of G1 and G2
        filtered_grad = [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)]

    return filtered_grad, cos_distance.item()


def compute_gradients(b, optimizer, model, criterion, device):
    loader = DataLoader(b, batch_size=len(b), shuffle=False)
    data = next(iter(loader))
    images, labels = data[0].to(device), data[1].to(device)
    # Forward and backward passes
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    # L1 regularization if specified
    if config['weight_decay_type'] == 'l1':
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
    loss.backward()
    G = [p.grad.clone() for p in model.parameters()]
    optimizer.zero_grad()
    return G, loss, labels, outputs

# Initialize the model
model = models.resnet18(num_classes=100)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Handle weight decay and L1 regularization
if config['weight_decay_type'] == 'l1':
    weight_decay = 0.0
    l1_lambda = config['weight_decay']
elif config['weight_decay_type'] == 'l2':
    weight_decay = config['weight_decay']
    l1_lambda = 0.0
else:
    raise ValueError("weight_decay_type must be 'l1' or 'l2'")

# Initialize the optimizer based on the configs
if config['optimizer'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=config['nesterov'])
elif config['optimizer'] == 'SGD+Nesterov':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=True)
elif config['optimizer'] == 'SGD+Nesterov+val_plateau':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['scheduler_patience'])
elif config['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           betas=tuple(config['betas']),
                           eps=config['eps'],
                           weight_decay=weight_decay)
elif config['optimizer'] == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            betas=tuple(config['betas']),
                            eps=config['eps'],
                            weight_decay=weight_decay)
elif config['optimizer'] == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'],
                              alpha=config['alpha'],
                              eps=config['eps'],
                              weight_decay=weight_decay,
                              momentum=config['momentum'],
                              centered=config['centered'])
else:
    raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy_top1 = correct_top1 / total
    return avg_loss, accuracy_top1

# Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

# Set up WandB project and run names
model_name = 'ResNet18'
dataset_name = 'CIFAR100'
project_name = f"{model_name}_{dataset_name}_FLIPPED_LABELS_COSINE_SIM"
name_prefix = 'GAF' if config['GAF'] else 'NO_GAF'
run_name = f"{name_prefix}_opt_{config['optimizer']}_lr_{config['learning_rate']}_bs_{config['batch_size']}"

# Initialize WandB
wandb.init(project=project_name, name=run_name, config=config)
config = wandb.config

# Create checkpoints directory
checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

if config['label_error_percentage'] and config['label_error_percentage']>0:
    if config['label_error_percentage']<1:
        train_dataset = flip_labels(train_dataset, label_error_percentage=config['label_error_percentage'], num_classes=len(class_indices))
    else:
        raise ValueError(f"label_error_percentage needs to be between 0 and 1. Given label_error_percentage={config['label_error_percentage']}")

# Training loop
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    total = 0
    iteration = 0

    # Calculate total iterations per epoch
    if config['GAF']:
        iterations_per_epoch = len(train_dataset) // (len(class_indices) * config['num_samples_per_class_per_batch'])
    else:
        iterations_per_epoch = len(train_dataset) // config['batch_size']

    while iteration < iterations_per_epoch:
        if config['GAF']:
            # Sample minibatches for GAF
            batches = sample_iid_mbs_for_GAF(train_dataset, class_indices, config['num_batches_to_force_agreement'])
            first_batch = batches[0]

            # Map batches to GPUs evenly
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available for parallel processing.")
            batch_gpu_mapping = {}
            for i, batch in enumerate(batches):
                gpu_id = i % num_gpus
                batch_gpu_mapping[i] = gpu_id

            # Compute gradients in parallel
            G_current, loss, labels, outputs = compute_gradients(first_batch, optimizer, model, criterion, device)
            agreed_count = 0
            for i, b in enumerate(batches[1:]):
                G, loss, labels, outputs = compute_gradients(b, optimizer, model, criterion, device)
                # G_current, gaf_percentage = filter_gradients_cosine_sign(G_current, G, epsilon=config['epsilon'])
                G_current_temp, cosine_distance = filter_gradients_cosine_sim(G_current, G, config['cos_distance_thresh'])
                
                if G_current_temp!=None:
                    G_current = G_current_temp
                    agreed_count+=1
                # Log gaf_percentage, iteration, and fuse iter i to wandb
                try:
                    wandb.log({'cosine_distance': cosine_distance, 'iteration': iteration, 'fuse_iter': i})
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                print(f"iteration {iteration}, fuse iter {i}, Gradient Agreement cosine_distance: {cosine_distance:.2f} with thresh {config['cos_distance_thresh']}")
                
            if agreed_count>0:
                # Atleast 1 of the gradients agreed with the first, lets apply the filtered gradients
                with torch.no_grad():
                    for param, grad in zip(model.parameters(), G_current):
                        param.grad = grad
                optimizer.step()
                # Update metrics
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
        else:
            # Sample a minibatch for standard training
            batch = sample_iid_mbs(train_dataset, class_indices, config['batch_size'])
            loader = DataLoader(batch, batch_size=len(batch), shuffle=False)
            data = next(iter(loader))
            images, labels = data[0].to(device), data[1].to(device)
            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # L1 regularization if specified
            if config['weight_decay_type'] == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            # Update metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

        iteration += 1

    # Perform validation every num_val_epochs iterations
    if epoch % config['num_val_epochs'] == 0 and total>0:
        
        train_loss = running_loss / total
        train_accuracy = correct_top1 / total
        val_loss, val_accuracy = evaluate(model, test_loader, device)
        # Log metrics to wandb
        try:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch': epoch,
                'iteration': iteration,
            })
        except Exception as e:
            print(f"Failed to log to wandb: {e}")
        print(f"Epoch [{epoch+1}/{config['epochs']}], Iteration [{iteration}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
        # Reset running metrics
        running_loss = 0.0
        correct_top1 = 0
        total = 0
        # Save the latest checkpoint
        checkpoint_name = f"{run_name}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        try:
            torch.save(model.state_dict(), checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # Adjust learning rate if scheduler is used
        if config['optimizer'] == 'SGD+Nesterov+val_plateau':
            scheduler.step(val_loss)

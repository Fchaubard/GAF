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
parser.add_argument('--num_batches_to_force_agreement', type=int, default=2, help='Number of batches to force agreement (must be > 1)')
parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
parser.add_argument('--num_samples_per_class_per_batch', type=int, default=1, help='Number of samples per class per batch if we are doing GAF')
parser.add_argument('--label_error_percentage', type=float, default=0, help='Percent of the train labels in the dataset to flip to a wrong label')
parser.add_argument('--cos_distance_thresh', type=float, default=1, help='Angle threshold on the cosine similarity if using that type of agreement filtering')
parser.add_argument('--MVA', type=bool, default=False, help='Enable MVA Gradient Agreement Filtering')
parser.add_argument('--MVA_mult', type=float, default=2, help='Multiples of sigma to thresh')
parser.add_argument('--MVA_history_length', type=float, default=200, help='Multiples of sigma to thresh')
parser.add_argument('--dummy', type=bool, default=False, help='if we should use dummy data or not')
parser.add_argument('--cifarn', type=bool, default=True, help='if we should use CIFARN labels or not')


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
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 1 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 2 --cifarn true

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


# CIFARN RUN: num_samples_per_class = [1,2,3,4,5,6,7,8] & cosine_distance_thresh = [0.95, 0.97, 1, 2] + MVA 
# num_samples_per_class = 1
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.95 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.97 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 2 --cifarn True

# num_samples_per_class = 2
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 5 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.95 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 5 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 0.97 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 5 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1 --cifarn True
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 5 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 2 --cifarn True
# ...

# span cos_distance_thresh=[1] MVA_mult=[1, 1.25, 1.5, 1.75, 2, 3] 
# python cifarn.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --scheduler_patience 100 --learning_rate 0.01 --momentum 0.9 --nesterov True --weight_decay 1e-2 --weight_decay_type l2 --num_samples_per_class_per_batch 5 --num_batches_to_force_agreement 2 --epsilon 1e0 --label_error_percentage 0.0 --cos_distance_thresh 1 --cifarn True --MVA True --MVA_history_length=200 --MVA_mult 3



size_of_MVA_history=config["MVA_history_length"]





'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreResNet18(num_classes):
    return ResNet(PreActBlock, [2,2,2,2],num_classes=num_classes)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3],num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


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


if config['dummy']==True:
    print('using dummy data')
    train_dataset = datasets.FakeData(
        size=50000,  # Match CIFAR-100 training set size
        image_size=(3, 32, 32),  # Match CIFAR-100 image size
        num_classes=100,
        transform=transform_train,  # Use the same training transforms
        random_offset=random.randint(0, 1000000)
    )
    # Create fake test data
    test_dataset = datasets.FakeData(
        size=10000,  # Match CIFAR-100 test set size
        image_size=(3, 32, 32),
        num_classes=100,
        transform=transform_test,  # Use the same test transforms
        random_offset=random.randint(0, 1000000)
    )
    
        
elif config['cifarn']==True:
    print('using CIFAR100N data')
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    # Load the noisy labels
    dd = torch.load('/root/CIFAR-100_human.pt')
    noisy_label = dd['noisy_label']
    # Replace the training labels with noisy labels
    train_dataset.targets = noisy_label
    # Load CIFAR-100 test dataset (clean labels)
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

else:
    print('using CIFAR100 data')
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
    
def sample_iid_mbs_for_GAF(full_dataset, class_indices, n, num_samples_per_class_per_batch=1):
    """
    Samples n independent minibatches, each containing an equal number of samples from each class.
    """
    # Initialize a list to hold indices for each batch
    batch_indices_list = [[] for _ in range(n)]
    num_samples_per_class = num_samples_per_class_per_batch  # Adjust if you want more samples per class per batch
    for cls in class_indices:
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
   
    # Use the original batch b
    loader = DataLoader(b, batch_size=len(b), shuffle=False)
    data = next(iter(loader))
    images, labels = data[0].to(device), data[1].to(device)

    # loader = DataLoader(b, batch_size=len(b), shuffle=False)
    # data = next(iter(loader))
    # images, labels = data[0].to(device), data[1].to(device)
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
# model = models.resnet18(num_classes=100)
model = ResNet34(num_classes=100)
model = model.to(device)
print("HERE")
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
model_name = 'ResNet34'
dataset_name = 'CIFAR100N'
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
    cos_distance_history = []

    # Calculate total iterations per epoch
    if config['GAF']:
        iterations_per_epoch = len(train_dataset) // (len(class_indices) * config['num_samples_per_class_per_batch'])
    else:
        iterations_per_epoch = len(train_dataset) // config['batch_size']

    while iteration < iterations_per_epoch:
        if config['GAF']:
            # Sample minibatches for GAF
            batches = sample_iid_mbs_for_GAF(train_dataset, class_indices, config['num_batches_to_force_agreement'], num_samples_per_class_per_batch = config['num_samples_per_class_per_batch'])
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
            G_current, loss, labels, outputs = compute_gradients(first_batch, optimizer, model, criterion, device )
            agreed_count = 0
            for i, b in enumerate(batches[1:]):
                G, loss, labels, outputs = compute_gradients(b, optimizer, model, criterion, device )
                # G_current, gaf_percentage = filter_gradients_cosine_sign(G_current, G, epsilon=config['epsilon'])
                
                if config['MVA']==True and config['cos_distance_thresh']<2:
                    mean_cos_distance = np.mean(cos_distance_history)
                    std_cos_distance = np.std(cos_distance_history)
                    threshold = mean_cos_distance + config['MVA_mult'] * (std_cos_distance)
                    if len(cos_distance_history)<size_of_MVA_history or  threshold>config['cos_distance_thresh']:
                        threshold = config['cos_distance_thresh']
                    G_current_temp, cosine_distance = filter_gradients_cosine_sim(G_current, G, threshold)
                    cos_distance_history.append(cosine_distance)
                    if len(cos_distance_history)>size_of_MVA_history:
                        del cos_distance_history[0]
                else:
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



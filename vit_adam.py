# This code implements ImageNet training with GAF (Gradient Agreement Filtering)

# torchrun --master_port=29502 --nproc_per_node=2 vit_adam.py /tempppp/imagenet --use-gaf --cos-distance-thresh 1  --arch vit_large_patch16_224 --pretrained --lr_warmup 100

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from transformers import get_cosine_schedule_with_warmup


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import timm

# Import wandb for logging
import wandb


os.environ["WANDB_API_KEY"] = ""

# List of available models
model_names = ['vit_large_patch16_224'] + sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# Argument parser for command-line options
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with WandB Logging')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=380, type=int, metavar='N',
                    help='mini-batch size (default: 226)')
parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_warmup', default=100, type=int,
                   help='initial learning rate', dest='lr_warmup')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD optimizer')
parser.add_argument('--wd', '--weight-decay', default=0.05, type=float,
                    metavar='W', help='weight decay (default: 1e-2)',
                    dest='weight_decay')
parser.add_argument('--num_cycles', default=0.5, type=float,
                    help='number of cycles for cosine LR',
                    dest='num_cycles')

parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizers')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend (default: nccl)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--dummy', action='store_true',
                    help='use fake data to benchmark')
parser.add_argument('--cos-distance-thresh', default=1.0, type=float,
                    help='Cosine distance threshold for GAF')
parser.add_argument('--use-gaf', action='store_true',
                    help='Use gradient agreement function with cosine similarity filtering')
parser.add_argument('--iters_till_val', default=0, type=int,
                    help='# of iters until val is performed')


verbose = False
best_acc1 = 0

def main():
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    args.distributed = args.world_size > 1
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    # Set seeds for reproducibility
    if args.seed is not None:
        rank_seed = args.seed + args.rank
        random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        np.random.seed(rank_seed)
        print(f"Rank {args.rank}: Using seed {rank_seed}", flush=True)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda', args.gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.rank = 0
    if args.arch == 'vit_large_patch16_224':
        if args.pretrained:
            print("=> using pre-trained ViT-L/16")
            model = timm.create_model('vit_large_patch16_224', 
                                    pretrained=True, 
                                    img_size=224)  # Directly set img_size in create_model
        else:
            print("=> creating ViT-L/16")
            model = timm.create_model('vit_large_patch16_224', 
                                    pretrained=False,
                                    img_size=224)  # Directly set img_size in create_model
            
        # No need to manually modify patch_embed or pos_embed as timm handles this
    else:
        if args.pretrained:
            print(f"=> using pre-trained model '{args.arch}'")
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print(f"=> creating model '{args.arch}'")
            model = models.__dict__[args.arch]()

    model = model.to(device)

    if args.distributed:
        # Synchronize initial model parameters
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    # if args.arch == 'vit_large_patch16_224':
    #     # Higher weight decay for ViT as per paper
    #     args.weight_decay = 0.1


    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize WandB
    if args.rank == 0:
        wandb.init(project='ImageNet_GAF_Training', config=args)
        wandb_config = wandb.config
    else:
        wandb_config = None

    # Data loading code
    if args.dummy:
        train_size = 224 if args.arch == 'vit_large_patch16_224' else 224
        train_dataset = datasets.FakeData(10000, (3, train_size, train_size), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(500, (3, train_size, train_size), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        if args.arch == 'vit_large_patch16_224':
            train_size = 224
            val_size = 224
            resize_size = 256
        else:
            train_size = 224
            val_size = 224
            resize_size = 256

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(train_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(val_size),
                transforms.ToTensor(),
                normalize,
            ]))


    if args.distributed:
        def get_sampler_seed(epoch):
            return args.seed + epoch * 100 if args.seed is not None else None
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=get_sampler_seed(0)
        )
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args, device, wandb_config)
        return

    total_iterations = len(train_loader) * args.epochs
    iteration = args.start_epoch * len(train_loader)


    # Setup AdamW training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps
    )
    
    # Cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup,
        num_training_steps=total_iterations,
        num_cycles=args.num_cycles
    )

    
    if args.iters_till_val>0:
        iters_till_val = args.iters_till_val
    else:
        if args.world_size>0:
            iters_till_val = len(train_loader) // args.world_size  

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            epoch_seed = get_sampler_seed(epoch)
            train_sampler.set_epoch(epoch)
            if hasattr(train_sampler, 'seed'):
                train_sampler.seed = epoch_seed

        # Train for one epoch
        train_loss, train_acc1, train_acc5, cos_distance = train(
            train_loader, model, criterion, optimizer, scheduler, scaler,
            epoch, args, iteration, total_iterations, device, 
            wandb_config, iters_till_val
        )
        iteration += iters_till_val 

        # Evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(
            val_loader, model, criterion, args, device, wandb_config)

        scheduler.step(val_loss)

        if args.rank == 0:
            message = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy@1': train_acc1,
                'train_accuracy@5': train_acc5,
                'val_loss': val_loss,
                'val_accuracy@1': val_acc1,
                'val_accuracy@5': val_acc5,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'cos_distance': cos_distance
            }
            wandb.log(message)
            print(message, flush=True)

    if args.rank == 0:
        wandb.finish()

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args,
          iteration, total_iterations, device, wandb_config, iters_till_val):
    batch_time = AverageMeter('Train Time', ':6.3f')
    data_time = AverageMeter('Train Data', ':6.3f')
    losses = AverageMeter('Train Loss', ':.4e')
    top1 = AverageMeter('TrainAcc@1', ':6.2f')
    top5 = AverageMeter('TrainAcc@5', ':6.2f')
    progress = ProgressMeter(
        iters_till_val,
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    max_cos_distance = -1
    cos_distance = torch.tensor(0.0, device=device)

    # Pre-allocate buffers for gradient operations
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    local_grad_buffer = torch.zeros(param_count, device=device)
    grad_list_buffer = [torch.zeros(param_count, device=device) for _ in range(args.world_size)]

    for i, (images, target) in enumerate(train_loader):
        if i >= iters_till_val:
            break
        max_cos_distance = 0
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Scaled backward pass
        scaler.scale(loss).backward()

        if args.distributed and args.use_gaf:
            # Unscale gradients once for both GAF and clipping
            scaler.unscale_(optimizer)
            
            # Collect local gradients - vectorized operation
            offset = 0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    size = param.grad.numel()
                    local_grad_buffer[offset:offset + size].copy_(param.grad.view(-1))
                    offset += size

            # Gather gradients from all processes
            for r in range(args.world_size):
                if r == args.rank:
                    grad_list_buffer[r].copy_(local_grad_buffer)
                dist.broadcast(grad_list_buffer[r], src=r)

            # Apply GAF
            agreed_count = 1
            aggregated_grad = local_grad_buffer.clone()
            
            for idx, remote_grad in enumerate(grad_list_buffer):
                if idx == args.rank:
                    continue
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    aggregated_grad.view(1, -1),
                    remote_grad.view(1, -1),
                    dim=1
                )
                cos_distance = 1 - cos_sim
                max_cos_distance = max(max_cos_distance, cos_distance.item())
                
                if cos_distance.item() <= args.cos_distance_thresh:
                    aggregated_grad = (aggregated_grad * agreed_count + remote_grad) / (agreed_count + 1)
                    agreed_count += 1

            # Update gradients based on agreement
            if agreed_count > 1:
                offset = 0
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        numel = param.grad.numel()
                        param.grad.copy_(
                            aggregated_grad[offset:offset + numel].view_as(param.grad)
                        )
                        offset += numel
            else:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.zero_()

            # Apply gradient clipping after GAF
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            # Non-GAF path: just unscale, clip, and step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.rank == 0:
                message = {
                    'train_loss_step': loss.item(),
                    'train_acc1_step': acc1.item(),
                    'train_acc5_step': acc5.item(),
                    'max_cos_distance': max_cos_distance,
                    'cos_distance': cos_distance.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'iteration': iteration + i
                }
                wandb.log(message)
                print(message, flush=True)

    return losses.avg, top1.avg, top5.avg, cos_distance.item()


def get_warmup_lr(base_lr, step, warmup_steps):
    """
    Calculate learning rate for warmup schedule.
    Linearly increases learning rate from 0 to base_lr over warmup_steps.
    
    Args:
        base_lr (float): Target learning rate after warmup
        step (int): Current step
        warmup_steps (int): Number of warmup steps
        
    Returns:
        float: Current learning rate
    """
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step / warmup_steps)

def validate(val_loader, model, criterion, args, device, wandb_config):
    batch_time = AverageMeter('Val_Time', ':6.3f', Summary.AVERAGE)
    losses = AverageMeter('Val_Loss', ':6.2f', Summary.AVERAGE)
    top1 = AverageMeter('Val_Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Val_Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: '
    )

    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Compute output and loss
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return losses.avg, top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save the training model"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics with new value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """All-reduce statistics across processes"""
        if not dist.is_initialized():
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """Summary of statistics"""
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """Displays and updates progress during training"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """Display current progress"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters if meter.val != 0]
        print('\t'.join(entries))

    def display_summary(self):
        """Display summary of progress"""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters if meter.summary()]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """Format batch string"""
        num_digits = len(str(num_batches))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Compute accuracy over top-k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # Check if predictions are correct
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Compute number of correct predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Compute accuracy
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()

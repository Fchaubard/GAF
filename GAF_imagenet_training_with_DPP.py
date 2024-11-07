# This code implements Imagenet training with GAF (Gradient Agreement Filtering)

# You can run with GAF like this:
# torchrun --nproc_per_node=2 main.py /tempppp/imagenet --use-gaf --cos-distance-thresh 0.97

# You can run without GAF like this:
# torchrun --nproc_per_node=2 main.py /tempppp/imagenet --use-gaf --cos-distance-thresh 2

# Or you can run without GAF like this:
# torchrun --nproc_per_node=2 main.py /tempppp/imagenet 


import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
os.environ["WANDB_API_KEY"] = ""

# Import wandb for logging
import wandb

# List of available models in torchvision
model_names = sorted(
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
parser.add_argument('-b', '--batch-size', default=256*8, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD optimizer')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-2)',
                    dest='weight_decay')
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
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend (default: nccl)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training')
parser.add_argument('--dummy', action='store_true',
                    help='use fake data to benchmark')
parser.add_argument('--cos-distance-thresh', default=0.1, type=float,
                    help='Cosine distance threshold for GAF')
parser.add_argument('--use-gaf', action='store_true',
                    help='Use gradient aggregation function with cosine similarity filtering')

best_acc1 = 0

def main():
    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This can slow down your training considerably!')

    # Adjust world size and initialize distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # Get the number of GPUs available
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    # Multiprocessing distributed training
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Single process training
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # Initialize distributed environment
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    # Set up device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print('Using CPU, this will be slow.')

    # Create the model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.to(device)

    # Wrap model with DistributedDataParallel if distributed training
    if args.distributed:
        if args.gpu is not None and torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    # Register the custom communication hook if use_gaf is True
    if args.use_gaf and args.distributed:
        state = {
            'cos_distance_thresh': args.cos_distance_thresh
        }
        model.register_comm_hook(state, cosine_sim_filtering_hook)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Prepare configuration for wandb
    config = {
        'model': args.arch,
        'dataset': 'ImageNet',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': 'SGD',
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'GAF': args.use_gaf,
        'cos_distance_thresh': args.cos_distance_thresh
    }

    model_name = config['model']
    dataset_name = config['dataset']
    project_name = f"{model_name}_{dataset_name}_FLIPPED_LABELS_COSINE_SIM"
    name_prefix = 'GAF' if config['GAF'] else 'NO_GAF'
    run_name = f"{name_prefix}_opt_{config['optimizer']}_lr_{config['learning_rate']}_bs_{config['batch_size']}"

    # Initialize WandB only in the main process
    if (not args.distributed) or (args.distributed and args.rank == 0):
        wandb.init(project=project_name, name=run_name, config=config)
        wandb_config = wandb.config
    else:
        wandb_config = None  # Other processes do not log

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Using dummy data for benchmarking")
        train_dataset = datasets.FakeData(8167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(500, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # Load ImageNet dataset
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    # Create data loaders and samplers
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # Evaluate the model if specified
    if args.evaluate:
        validate(val_loader, model, criterion, args, device, wandb_config)
        return

    total_iterations = len(train_loader) * args.epochs
    iteration = args.start_epoch * len(train_loader)

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_loss, train_accuracy1,train_accuracy5 = train(train_loader, model, criterion, optimizer, epoch, args, iteration, total_iterations, device, wandb_config)

        # Update iteration counter
        iteration += len(train_loader)

        # Evaluate on validation set
        val_loss, val_accuracy1, val_accuracy5 = validate(val_loader, model, criterion, args, device, wandb_config)

        # Step the scheduler
        scheduler.step()

        # Log metrics to wandb only in main process
        if (not args.distributed) or (args.distributed and args.rank == 0):
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy@1': train_accuracy1,
                'train_accuracy@5': train_accuracy5,
                'val_loss': val_loss,
                'val_accuracy@1': val_accuracy1,
                'val_accuracy@5': val_accuracy5,
                'iteration': iteration,
                'total_iterations': total_iterations,
            })

        # Remember best accuracy and save checkpoint
        is_best = val_accuracy1 > best_acc1
        best_acc1 = max(val_accuracy1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best)

    # Finish WandB run in main process
    if (not args.distributed) or (args.distributed and args.rank == 0):
        wandb.finish()

def train(train_loader, model, criterion, optimizer, epoch, args, iteration, total_iterations, device, wandb_config):
    # Meters to track performance metrics
    batch_time = AverageMeter('Train_Time', ':6.3f')
    data_time = AverageMeter('Train_Data', ':6.3f')
    losses = AverageMeter('Train_Loss', ':6.2f')
    top1 = AverageMeter('Train_Acc@1', ':6.2f')
    top5 = AverageMeter('Train_Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    # Switch model to training mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to the specified device
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

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log metrics to wandb only in main process
        if (not args.distributed) or (args.distributed and args.rank == 0):
            if i % args.print_freq == 0:
                wandb.log({
                    'train_loss_step': losses.val,
                    'train_accuracy@1_step': top1.val,
                    'train_accuracy@5_step': top5.val,
                    'iteration_step': iteration + i + 1,
                })

        # Display progress
        if i % args.print_freq == 0:
            progress.display(i)

    # Return average loss and accuracy over the epoch
    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, criterion, args, device, wandb_config):
    # Meters to track performance metrics
    batch_time = AverageMeter('Val_Time', ':6.3f', Summary.AVERAGE)
    losses = AverageMeter('Val_Loss', ':6.2f', Summary.AVERAGE)
    top1 = AverageMeter('Val_Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Val_Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (
            args.distributed and (len(val_loader.sampler) * args.world_size
                                  < len(val_loader.dataset))),
        [batch_time, losses, top1, top5], prefix='Test: '
    )
    

    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # Move data to the specified device
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

            # Display progress
            if i % args.print_freq == 0:
                progress.display(i)

    # Display summary
    progress.display_summary()

    # Return average loss and accuracy over the validation set
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

def cosine_sim_filtering_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Custom communication hook that filters gradients based on cosine similarity.
    """
    # Get the current process group
    group = dist.group.WORLD
    grad_tensor = bucket.buffer()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Initialize a list to gather gradients from all processes
    gathered_gradients = [torch.zeros_like(grad_tensor) for _ in range(world_size)]

    # Asynchronously gather gradients from all processes
    allgather_work = dist.all_gather(gathered_gradients, grad_tensor, group=group, async_op=True)
    fut = allgather_work.get_future()

    # Callback function to process gradients after all_gather completes
    def allgather_complete(fut):
        # Stack gradients into a matrix of shape [world_size, num_elements]
        gradients_matrix = torch.stack([g.view(-1).to(bucket.buffer().device) for g in gathered_gradients])

        # Compute pairwise cosine similarities between gradients
        cos_sim_matrix = torch.nn.functional.cosine_similarity(
            gradients_matrix.unsqueeze(1), gradients_matrix.unsqueeze(0), dim=2
        )
        # Compute cosine distances
        cos_distance_matrix = 1 - cos_sim_matrix

        # Get the maximum cosine distance
        max_cos_distance = cos_distance_matrix.max()
        cos_distance_thresh = state['cos_distance_thresh']

        # Log the maximum cosine distance only from the main process
        if dist.get_rank() == 0:
            wandb.log({
                'cosine_distance': max_cos_distance.item(),
                # 'iteration' can be managed separately if needed
            })

        if max_cos_distance > cos_distance_thresh:
            # If any gradient pair exceeds the threshold, return the original gradient
            return grad_tensor
        else:
            # Average all gradients
            avg_grad = torch.mean(torch.stack(gathered_gradients), dim=0)
            return avg_grad

    # Chain the callback to the future
    return fut.then(allgather_complete)

if __name__ == '__main__':
    main()

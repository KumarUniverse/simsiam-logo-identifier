#######################################################################################
# Final Project
# Course: CSE 586
# Authors: Dylan Knowles and Akash Kumar
# Description: This Python program uses transfer learning to train a SimSiam
# neural network model to identify car brand logos.
# Original SimSiam repo: https://github.com/facebookresearch/simsiam
# Note: This code is intended to be run on a single CPU, a Nvidia GPU or Apple GPU.
#######################################################################################

import argparse
import math
import os
import errno
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Command-line arguments
parser = argparse.ArgumentParser(description='PyTorch SimSiam Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (no default; must be specified)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--set-cp-epoch', action='store_true',
                    help=('Set the starting epoch the same as the ' +
                        'checkpoint epoch (True if included; default: False)'))
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate (default: 0.05)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training (default: None)')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='fix learning rate for the predictor (True if included; default: False)')

def main():
    """Program execution starts here."""
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():  # for Nvidia GPU
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'If using an Nvidia GPU, this will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting from checkpoints.')

    # Call the worker fn and pass in the command-line arguments.
    program_start_time = time.time()
    main_worker(args)
    program_end_time = time.time()
    total_elapsed_time = int(program_end_time - program_start_time)
    total_elapsed_hrs = total_elapsed_time // 60 // 60
    total_elapsed_mins = (total_elapsed_time // 60) % 60
    total_elapsed_secs = total_elapsed_time % 60

    print("\nUnsupervised training of SimSiam network complete.")
    print(f"Total elapsed time: {total_elapsed_hrs} hrs, " +
            f"{total_elapsed_mins} mins and {total_elapsed_secs} secs.\n")

def main_worker(args):
    """Helper function for the main function."""
    # Set the GPU device to use.
    print('Is CUDA available: ', torch.cuda.is_available())
    print('Is MPS available: ', torch.backends.mps.is_available())
    device_str = ("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu")
    gpu_device = torch.device(device_str)

    # Uncomment to force device to be CPU:
    # device_str = "cpu"
    # gpu_device = torch.device("cpu")

    print(f"Using {device_str.upper()} device...\n")

    # Dataset
    data_path = "./datasets/Car_Brand_Logos/"
    train_folder_name = "Train/"
    train_path = os.path.join(data_path, train_folder_name)
    if not os.path.isdir(train_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_path)

    # Load the parameters of the pre-trained model in the saved checkpoint.
    if args.resume is not None:
        if not os.path.isfile(args.resume): # args.resume is the path to the checkpoint
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.resume)
        checkpoint = torch.load(args.resume, map_location=gpu_device)

    if args.resume: # if resuming from checkpoint, use same architecture as before.
        args.arch = checkpoint['arch'] # architecture of the pretrained model (default: 'resnet50')
        # Checkpoints are saved with fix_pred_lr set to True.
        # So when loading checkpoints, fix_pred_lr must be set to True.
        args.fix_pred_lr = True

    # Initialize SimSiam model and optimizer with the loaded parameters.
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, args.pred_dim)
    model = torch.nn.DataParallel(model) # Implement data parallelism at the module level.
    model.to(gpu_device) # Convert model format to make it suitable for current GPU device.
    #print(model) # print model for debugging

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CosineSimilarity(dim=1).to(gpu_device) # use if using single Nvidia GPU
        cudnn.benchmark = True
    else:
        criterion = nn.CosineSimilarity(dim=1).to(gpu_device)

    #init_lr = lr * batch_size / 256 # infer learning rate before changing batch size
    init_lr =  args.lr * 2 # args.lr * 512 / 256 # original batch size was 512
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.set_cp_epoch: # to resume training from the last saved epoch.
            args.start_epoch = checkpoint['epoch']

    # There are 2 main network sections: the encoder and the predictor.
    # By default, all the layers require gradients (requires_grad_=True).
    model_layers = model.module.children()

    # Freeze all the layers of the encoder except the first 2 layers.
    model_layers_list = list(model_layers)
    # print(f"Model_layers_list len: {len(model_layers_list)}")  # 2
    encoder = model_layers_list[0]
    encoder_layers = encoder.children()
    num_encoder_layers = 0  # 10
    num_unfreeze_layers = 2
    for layer in encoder_layers:
        # For debugging.
        # print(f"LAYER: {num_encoder_layers}")
        # print(layer)
        # if layer.requires_grad_:
        #     print(f'This layer requires gradients')
        if num_encoder_layers >= num_unfreeze_layers:
            layer.requires_grad_ = False
        num_encoder_layers += 1
    # print(f"Number of layers in encoder: {num_encoder_layers}\n") # 10 layers

    # Freeze all the layers of the predictor except the last layer.
    predictor = model_layers_list[1]
    predictor_layers = predictor.children()
    num_predictor_layers = 0  # 4
    for layer in predictor_layers:
        # For debugging.
        # print(f"LAYER: {num_predictor_layers}")
        # print(layer)
        # if layer.requires_grad_:
        #     print(f'This layer requires gradients')
        if num_predictor_layers <= 3:
            layer.requires_grad_ = False
        num_predictor_layers += 1
    # print(f"Number of layers in predictor: {num_predictor_layers}") # 4 layers

    #num_layers = num_encoder_layers + num_predictor_layers  # 14 (NOTE)

    # Data loading code
    traindir = os.path.join(args.data, 'Train')  # args.data = ./datasets/Car_Brand_Logos/
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    shuffle_samples = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle_samples,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    is_checkpoint_saved = True  # to control whether checkpoints are saved or not
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device_str)

        if is_checkpoint_saved and ((epoch) % 5 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
            print(f"Checkpoint {epoch} saved.")  # for debugging


def train(train_loader, model, criterion, optimizer, epoch, args, device_str):
    """Train the SimSiam model."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # This should work for a single Nvidia or Apple GPU.
        # print("Converting training images to the correct GPU format.") # for debugging
        if device_str == "cuda":
            images[0] = images[0].cuda(non_blocking=True) # need non-blocking if using pin memory for CUDA
            images[1] = images[1].cuda(non_blocking=True)
        elif device_str == "mps" or device_str == "cpu":
            images[0] = images[0].to(device_str)
            images[1] = images[1].to(device_str)


        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5 # apply stop gradient

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        # perform backprop; compute the gradients of the loss
        optimizer.step()       # update the values of the parameters using the gradients

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the neural network model's parameters to a .tar checkpoint file.
    All checkpoints are stored in the checkpoints/ folder of the current directory.
    """
    filename = os.path.join("./checkpoints/", filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display training progress on the console."""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()

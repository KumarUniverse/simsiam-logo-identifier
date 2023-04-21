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
                    help='path to the checkpoint (default: None)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: None)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training (default: None)')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='fix learning rate for the predictor (True if included; default: False)')

best_acc1 = 0

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
    total_elapsed_hrs = total_elapsed_time / 60 // 60
    total_elapsed_mins = (total_elapsed_time // 60) % 60
    total_elapsed_secs = total_elapsed_time % 60

    print("\nUnsupervised training of SimSiam network complete.")
    print(f"Total elapsed time: {total_elapsed_hrs} hrs, " +
            f"{total_elapsed_mins} mins and {total_elapsed_secs} secs.\n")

def main_worker(args):
    """Helper function for the main function."""
    global best_acc1
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

    # Load the parameters of the pre-trained model in the saved checkpoint.
    if args.pretrained:
        print("=> loading pretrained checkpoint...")
        if not os.path.isfile(args.pretrained): # args.pretrained is the path to the pretrained checkpoint
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location=gpu_device)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        # If resuming from checkpoint, use same architecture as before.
        args.arch = checkpoint['arch'] # 'resnet50' # architecture of the pretrained model (default: 'resnet50')
        # Checkpoints are saved with fix_pred_lr set to True.
        # So when loading checkpoints, fix_pred_lr must be set to True.
        args.fix_pred_lr = True

    # Initialize SimSiam model and optimizer with the loaded parameters.
    print("=> creating model '{}'".format(args.arch))
    # model = simsiam.builder.SimSiam(
    #         models.__dict__[args.arch],
    #         args.dim, args.pred_dim)


    model = models.__dict__[args.arch]()

    model.to(gpu_device) # Convert model format to make it suitable for current GPU device.

     # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # print(model) # prints default resnet50 architecture with 1000 output features in final fc layer.

    # init the fc layer
    model.fc.out_features = 8 # number of car logo classes.
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    init_lr =  args.lr * 2 # args.lr * 512 / 256 # original batch size was 512

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().to(gpu_device) # use if using single Nvidia GPU
        cudnn.benchmark = True
    else:
        criterion = nn.CrossEntropyLoss().to(gpu_device)


     # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias


    #init_lr = lr * batch_size / 256 # infer learning rate before changing batch size

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.pretrained:
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        if args.set_cp_epoch: # to resume training from the last saved epoch.
            args.start_epoch = checkpoint['epoch']

    # optionally resume from a lincls checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=gpu_device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.to(gpu_device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    traindir = os.path.join(args.data, 'Train')  # args.data = ./datasets/Car_Brand_Logos/Train
    valdir = os.path.join(args.data, 'Test') # args.data = ./datasets/Car_Brand_Logos/Test
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

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    is_checkpoint_saved = True  # to control whether checkpoints are saved or not
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device_str)

        acc1 = validate(val_loader, model, criterion, device_str, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_checkpoint_saved and ((epoch+1) % 5 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename='checkpoint_lincls_{:04d}.pth.tar'.format(epoch))
            print(f"Checkpoint {epoch} saved.")  # for debugging

def train(train_loader, model, criterion, optimizer, epoch, args, device_str):
    """Train the SimSiam model."""
    batch_time = AverageMeter('Time (s):', ':6.3f')
    losses = AverageMeter('Loss:', ':.4f')
    top1 = AverageMeter('Acc@1:', ':6.2f')
    top5 = AverageMeter('Acc@5:', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # This should work for a single Nvidia or Apple GPU.
        if device_str == "cuda":
            images = images.cuda(non_blocking=True) # need non-blocking if using pin memory for CUDA
            target = target.cuda(non_blocking=True)
        elif device_str == "mps" or device_str == "cpu":
            images= images.to(device_str)
            target = target.to(device_str)

        # compute output and loss
        output = model(images)
        loss = criterion(output, target)# apply stop gradient

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        # perform backprop; compute the gradients of the loss
        optimizer.step()       # update the values of the parameters using the gradients

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, device_str, args):
    # model = simsiam.builder.SimSiam()
    # model.load_state_dict(torch.load("./checkpoints/checkpoint_0009.pth.tar"))
    # model = torch.nn.DataParallel(model) # Implement data parallelism at the module level.
    # model.to(gpu_device) # Convert model format to make it suitable for current GPU device.
    batch_time = AverageMeter('Time (s):', ':6.3f')
    losses = AverageMeter('Loss:', ':.4e')
    top1 = AverageMeter('Acc@1:', ':6.2f')
    top5 = AverageMeter('Acc@5:', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # This should work for a single Nvidia or Apple GPU.
            if device_str == "cuda":
                images = images.cuda(non_blocking=True) # need non-blocking if using pin memory for CUDA
                target = target.cuda(non_blocking=True)
            elif device_str == "mps" or device_str == "cpu":
                images= images.to(device_str)
                target = target.to(device_str)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # _,predicted = torch.max(output,1)
            # nCorrect += (predicted == target).sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # acc = 100 * nCorrect / nSamples
        # print(f'Accuracy of the model: {acc:.3f} %')

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the neural network model's parameters to a .tar checkpoint file.
    All checkpoints are stored in the checkpoints/ folder of the current directory.
    """
    filename = os.path.join("./cls_checkpoints/", filename)
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
    """Computes and stores the average and current value"""
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

##########################################
# Final Project
# Course: CSE 586
# Authors: Dylan Knowles and Akash Kumar
##########################################

import math
import errno
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

# Set the GPU device to use.
print('Is Pytorch built with MPS: ', torch.backends.mps.is_built()) # for debug
device_str = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
print(f"Using {device_str} device\n")
gpu_device = torch.device(device_str)

# Dataset
data_path = "./datasets/Car_Brand_Logos/"
train_folder_name = "Train/"
train_path = os.path.join(data_path, train_folder_name)
if not os.path.isdir(train_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_path)

# Load the parameters of the pre-trained model in the saved checkpoint.
m_path = "./models/"
pretrained_model_filename = "checkpoint_0099.pth.tar"
model_path = os.path.join(m_path, pretrained_model_filename)
if not os.path.isfile(model_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pretrained_model_filename)
checkpoint = torch.load(model_path)

# Store the hyperparameters into variables
num_workers = 8 # number of data loading workers (default: 32)
epochs = 100 # total number of epochs to run (default: 100)
pretrained_epochs = checkpoint['epoch'] # number of epochs the pretrained model trained for
batch_size = 128 # mini-batch size (default: 512)
lr = 0.05 # initial (base) learning rate (default: 0.05)
momentum = 0.9 # momentum of SGD solver
weight_decay = 1e-4 # weight decay (default: 1e-4)
dim = 2048 # feature dimension (default: 2048)
pred_dim = 512 # hidden dimension of the predictor (default: 512)
fix_pred_lr = True # fix learning rate for the predictor
model_arch = checkpoint['arch'] # architecture of the pretrained model (default: 'resnet50')

# Initialize SimSiam model and optimizer with the loaded parameters.
model = simsiam.builder.SimSiam(
        models.__dict__[model_arch],
        dim, pred_dim)
model.load_state_dict(checkpoint['state_dict'])

if fix_pred_lr:
    optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                    {'params': model.module.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = model.parameters()

#init_lr = lr * batch_size / 256 # infer learning rate before changing batch size
init_lr = lr
optimizer = torch.optim.SGD(optim_params, init_lr, momentum=momentum, weight_decay=weight_decay)
optimizer.load_state_dict(checkpoint['optimizer'])

model.to(gpu_device) # Convert model format to make it suitable for current GPU device.

# Freeze all the layers of the model except the first and last 3 layers.
layers = model.children()
num_layers = len(layers)
freeze_lower_bound = 3
freeze_upper_bound = num_layers - 3
for layer_idx, layer in enumerate(layers):
    if layer_idx > freeze_lower_bound or layer_idx < freeze_upper_bound:
        for param in layer.parameters():
            param.requires_grad = False

print(model) # print model

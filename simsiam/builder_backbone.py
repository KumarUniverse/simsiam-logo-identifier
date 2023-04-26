# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=1000, pred_dim=512):
        """
        dim: feature dimension (default: 1000)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, pretrained=True)

        # # Rename the last layer from 'classifier' to 'fc'
        # print(base_encoder.name)
        # if base_encoder.name == "efficientnet_v2_s":
        #     self.encoder.classifier.name = 'fc'

        # build a 3-layer projector
        if base_encoder.name == "efficientnet_v2_s":
            prev_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.SiLU(inplace=True), # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.SiLU(inplace=True), # second layer
                                            self.encoder.classifier, # (default: resnet50 backbone)
                                            nn.BatchNorm1d(dim, affine=False)) # output layer
            #self.encoder.classifier[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        else:
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            self.encoder.fc, # (default: resnet50 backbone)
                                            nn.BatchNorm1d(dim, affine=False)) # output layer
            self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

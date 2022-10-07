"""PyTorch implementation of Wide-ResNet taken from 
https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/models/cifar10/wide_resnet.py"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DualBN import DualBN2d
from models.noise_layer import noise_Conv2d, noise_Linear



class BasicBlockOAT(nn.Module):
    #SK: removed the last argument FiLM_in_channels=1, as we will not use FiLM based feature transforms in the model.
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use2BN=False):
        super(BasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.bn1 = Norm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = noise_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = Norm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = noise_Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and noise_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

        #self.film1 = FiLM_Layer(channels=in_planes, in_channels=FiLM_in_channels) 
        #self.film2 = FiLM_Layer(channels=out_planes, in_channels=FiLM_in_channels)

    def forward(self, x, _lambda, w_noise=True, idx2BN=None):
        if self.use2BN:
            out = self.bn1(x, idx2BN)
        else:
            # print('x device:', x.get_device())
            # print('bn1 device:', self.bn1.weight.get_device())
            out = self.bn1(x)
        #out = self.film1(out, _lambda)
        out = self.relu1(out)
        if not self.equalInOut:
            sc = self.convShortcut(out, _lambda, w_noise)
        else:
            sc = x
        out = self.conv1(out, _lambda, w_noise)
        
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        #out = self.film2(out, _lambda)
        out = self.relu2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out, _lambda, w_noise)

        out = torch.add(sc, out)
        return out


class WideResNetOAT(nn.Module):
    #Removed FiLM_in_channels=1 argument
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, use2BN=False):
        super(WideResNetOAT, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlockOAT

        self.use2BN = use2BN

        # 1st conv before any network block
        self.conv1 = noise_Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.bundle1 = [block(nChannels[0], nChannels[1], 1, dropRate=dropRate, use2BN=use2BN)]
        for _ in range(1, n):
            self.bundle1.append(block(nChannels[1], nChannels[1], 1, dropRate=dropRate, use2BN=use2BN))
        self.bundle1 = nn.ModuleList(self.bundle1)
        # 2nd block
        self.bundle2 = [block(nChannels[1], nChannels[2], 2, dropRate=dropRate, use2BN=use2BN)]
        for _ in range(1, n):
            self.bundle2.append(block(nChannels[2], nChannels[2], 1, dropRate=dropRate, use2BN=use2BN))
        self.bundle2 = nn.ModuleList(self.bundle2)
        # 3rd block
        self.bundle3 = [block(nChannels[2], nChannels[3], 2, dropRate=dropRate, use2BN=use2BN)]
        for _ in range(1, n):
            self.bundle3.append(block(nChannels[3], nChannels[3], 1, dropRate=dropRate, use2BN=use2BN))
        self.bundle3 = nn.ModuleList(self.bundle3)
        
        # global average pooling and classifier
        if self.use2BN:
            self.bn1 = DualBN2d(nChannels[3])
        else:
            self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        self.nChannels = nChannels[3]
        self.bundles = [self.bundle1, self.bundle2, self.bundle3]


    def forward(self, x, _lambda, w_noise=True, idx2BN=None):
        out = self.conv1(x, _lambda, w_noise)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out, _lambda, w_noise, idx2BN)
        if self.use2BN:
            out = self.bn1(out, idx2BN) 
        else:
            out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 24) # stl10 image shape (96,96)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

#SK: removed the last argument FiLM_in_channels=1, as we will not use FiLM based feature transforms in the model.
def WRN40_2OAT(use2BN=False):
    return WideResNetOAT(depth=40, num_classes=10, widen_factor=2, dropRate=0.3, use2BN=use2BN)


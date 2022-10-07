import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np


class noise_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, pni='channelwise', w_noise=True):
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*0.25,
                                        requires_grad=True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)
        
        self.w_noise = w_noise

    # The w_noise=True for adversarial images and False for clean images and thus the noise_weight will be either
    # noisy or actual weights based on w_noise value, that should pass down from forward pass written in the 
    # model definition file.
    def forward(self, input, w_noise):
        
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        if w_noise:
            noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        else:
            noise_weight = self.weight
        output = F.linear(input, noise_weight, self.bias)
        
        return output 



class noise_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='layerwise', w_noise=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.25,
                                        requires_grad = True)     
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise    

    #_lambda is kept 1.0 for the training phase through out. 
    #Where as for val _lambda can be any value between 0 and 1.
    def forward(self, input, _lambda=1.0, w_noise=True):

        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)
        '''
        #SK: following is to support any random value of _lambda, 0<= _lambda<=1.0 during validation.
        if _lambda == 1.0 or _lambda==0:
            noise_ratioed = noise
        else:
            noise_np = noise.cpu().detach().numpy()
            noise_tmp = np.abs(noise_np)
            delete_frac = 1.0 - _lambda
            delete_percent = delete_frac * 100
            percentile = np.percentile(noise_tmp, delete_percent)
            noise_underTh = noise_tmp < percentile
            noise_np[noise_underTh] = 0
            noise_ratioed = torch.from_numpy(noise_np).cuda() 
        # The w_noise=True for adversarial images and False for clean images and thus the noise_weight will be either
        # noisy or actual weights based on w_noise value, that is passes down from forward pass written in the 
        # model definition file.
        '''
        #'''
        #following is for distributing lambda when the BN_a is used for _lambda > 0.0
        if _lambda < 1.0:
            mod_lambda = (_lambda*1)
        else:
            mod_lambda = (_lambda)*1
        '''
        #following is for distributing lambda when the BN_a is used from halfway line of _lambda i.e. 0.5
        if _lambda < 0.6:
            mod_lambda = (_lambda*2)
        else:
            mod_lambda = (_lambda - 0.5)*2
        '''
        if w_noise:
            noise_weight = self.weight + self.alpha_w * noise * mod_lambda * w_noise
        else:
            noise_weight = self.weight

        '''
        if(self.weight.requires_grad == False):
            print('_lambda:', _lambda)
            print('w_noise:', w_noise)
        '''
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output


class noise_Conv2d_rm(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='layerwise', w_noise=True):
        super(noise_Conv2d_rm, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.25,
                                        requires_grad = True)     
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise

        self.mask_c = (torch.rand(self.weight.size()) > 0.5) + 0.0
        # import pdb; pdb.set_trace()
        self.mask_a = (torch.ones_like(self.weight) - self.mask_c)
        self.mask_c = self.mask_c.cuda()
        self.mask_a = self.mask_a.cuda()

    def forward(self, input, _lambda=1.0, w_noise=True, use_rm=False):

        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)


        #following is for distributing lambda when the BN_a is used for _lambda > 0.0
        if _lambda < 1.0:
            mod_lambda = (_lambda*1)
        else:
            mod_lambda = (_lambda)*1
        '''
        #following is for distributing lambda when the BN_a is used from halfway line of _lambda i.e. 0.5
        if _lambda < 0.6:
            mod_lambda = (_lambda*2)
        else:
            mod_lambda = (_lambda - 0.5)*2
        '''
        if w_noise:
            noise_weight = self.weight + self.alpha_w * noise * mod_lambda * w_noise
        else:
            noise_weight = self.weight

        if use_rm:
            # initilize mask with 50% 1s
            # mask_c = torch.ones_like(noise_weight)
            # mask_c = F.dropout(mask_c, p=0.5)
            # mask_a = torch.ones_like(noise_weight) - mask_c

            if w_noise:
                noise_weight = self.weight * self.mask_a
            else:
                noise_weight = self.weight * self.mask_c

        '''
        if(self.weight.requires_grad == False):
            print('_lambda:', _lambda)
            print('w_noise:', w_noise)
        '''
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output

        
        
                      
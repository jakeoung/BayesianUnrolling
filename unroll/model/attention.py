import torch
import torch.nn as nn
import torch.nn.functional as F

pad_mode='replicate'

# PicaNet https://github.com/Ugness/PiCANet-Implementation/blob/master/network.py
class PAConv(nn.Module):
    def __init__(self, nscale, nf, K=3, opt=0):
        super(PAConv, self).__init__()
        ngp = nscale
        # self.k2 = nn.Conv2d(nf, nf, 1, padding_mode=pad_mode) # 1x1 convolution nf->nf
        KK = K
        dilation=1
        
        if opt == 10: # use dilation mode
            dilation = 2
            pad = (K-1)//2 * dilation
            # 2*pad = dilation*(K-1) 
            # pad = dilation*(K-1) / 2
        else:
            pad = (K-1)//2

        self.k1 = nn.Conv2d(nf, nf, kernel_size=K, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 1x1 convolution nf->nf
        self.k2 = nn.Conv2d(nf, nf, kernel_size=K, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=K, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=K, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 3x3 convolution

        self.act = nn.LeakyReLU()

        
    def forward(self, x):
        x = self.k1(x)
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        
        return out

class PAConvGroup(nn.Module):
    def __init__(self, nscale, nf, K=3, extract_features=False, opt=0):
        super(PAConvGroup, self).__init__()
        ngp = nscale
        dilation=1
        
        if opt == 10: # use dilation mode
            dilation = 2
            pad = (K-1)//2 * dilation
            # 2*pad = dilation*(K-1) 
            # pad = dilation*(K-1) / 2
        else:
            pad = (K-1)//2

        self.k1 = nn.Conv2d(nf*2, nf*2, kernel_size=K, padding=pad, groups=nscale, padding_mode=pad_mode, bias=False, dilation=dilation) # 1x1 convolution nf->nf
        self.k2 = nn.Conv2d(nf*2, nf*2, kernel_size=K, padding=pad, groups=nscale, padding_mode=pad_mode, bias=False, dilation=dilation) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf*2, nf*2, kernel_size=K, groups=nscale, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 3x3 convolution
        self.k4 = nn.Conv2d(nf*2, nf*2, kernel_size=K, groups=nscale, padding=pad, bias=False, padding_mode=pad_mode, dilation=dilation) # 3x3 convolution

        self.extract_features = extract_features

        if extract_features:
            nf2 = nscale
            self.conv0 = nn.Conv2d(nscale*2, nf2*2, groups=ngp*2, kernel_size=K,  bias=False, padding=(K-1)//2, padding_mode="replicate")
            self.conv1 = nn.Conv2d(nf2*2, nf2*2, groups=ngp*2,kernel_size=K, bias=False, padding=(K-1)//2, padding_mode="replicate")
            self.conv2 = nn.Conv2d(nf2*2, nf*2, groups=ngp*2, kernel_size=K, bias=False, padding=(K-1)//2, padding_mode="replicate")

        self.act = nn.LeakyReLU()

    def forward(self, x):
        if self.extract_features:
            x = self.act(self.conv2(self.act(self.conv1(self.act(self.conv0(x))))))

        x = self.k1(x)
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)        
        return out


class SimpleExtractF(nn.Module):
    def __init__(self, nscale, K=3, ngp=None, opt=0, bias=False):
        super(SimpleExtractF, self).__init__()
        nf2 = nscale
        if ngp is None:
            ngp = nscale

        dilation = 1
        if opt == 10: # use dilation mode
            dilation = 2
            pad = (K-1)//2 * dilation
        else:
            pad = (K-1)//2
        
        self.conv0 = nn.Conv2d(nscale, nf2, groups=ngp, kernel_size=K,  bias=bias, padding=pad, padding_mode="replicate", dilation=dilation)
        self.conv1 = nn.Conv2d(nf2, nf2, groups=ngp, kernel_size=K, bias=bias, padding=pad, padding_mode="replicate", dilation=dilation)
        self.conv2 = nn.Conv2d(nf2, nscale, groups=ngp, kernel_size=K, bias=bias, padding=pad, padding_mode="replicate", dilation=dilation)

        self.act = nn.LeakyReLU()
        # self.opt = opt
        # if opt:
        #     self.ca = ChannelAttention(nscale)

    def forward(self, x):
        # if self.opt:
        #     x = self.ca(x)

        x = self.act(self.conv2(self.act(self.conv1(self.act(self.conv0(x))))))
        return x

# 2S x H x W -> S x H x W
class ExtractSqueeze(nn.Module):
    def __init__(self, nf1, nf2, K=3, bias=False):
        super(ExtractSqueeze, self).__init__()
        ngp = 1
        self.conv0 = nn.Conv2d(nf1, nf2, groups=ngp, kernel_size=K, bias=bias, padding=(K-1)//2, padding_mode="replicate")
        self.conv1 = nn.Conv2d(nf2, nf2, groups=ngp, kernel_size=K, bias=bias, padding=(K-1)//2, padding_mode="replicate")
        self.conv2 = nn.Conv2d(nf2, nf2, groups=ngp, kernel_size=K, bias=bias, padding=(K-1)//2, padding_mode="replicate")

        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.conv2(self.act(self.conv1(self.act(self.conv0(x))))))
        return x



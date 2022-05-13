from . import attention
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Softargmax(nn.Module):
    def __init__(self, nscale):
        super(Softargmax, self).__init__()
        
    def forward(self, x, w, tau=0.0):
        # b x c x h w        
        if tau > 0.0:
            onehot = F.gumbel_softmax(w, tau=tau, hard=True, dim=1)
            return torch.sum(x * onehot, dim=1, keepdims=True)
        
        #------------- old
        coord = torch.sum(w * self.ii, dim=1, keepdim=True)
        coord = coord.round().long()
        # coord = torch.sum(w * self.ii, dim=1, keepdim=True).round().long()

        return torch.gather(x, 1, coord)

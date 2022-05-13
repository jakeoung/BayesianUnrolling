from . import attention
from . import softargmax

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Args:
        - nscale : number of scales
        - alpha : scale factor in the softmax in the expansion (rho in the paper)
        - nblock : number of stages (K in the paper)
        - K : kernel size
    """
    def __init__(self, nscale, attention_type="pa", alpha=2.0, nblock=4, tau=10, K=3):
        super(Model, self).__init__()

        self.nblock = nblock
        self.attention_blocks = nn.ModuleList()
        
        ngp = 1
        opt = 0

        bias = False
        if attention_type.find("bias") >= 0:
            bias = True

        nchannel = nscale
        self.nchannel = nchannel

        self.softargmax = softargmax.Softargmax(nchannel)
        for i in range(nblock):
            self.attention_blocks.append(attention.SimpleExtractF(nchannel, K=K, ngp=ngp, opt = opt, bias=bias))
            self.attention_blocks.append(attention.PAConv(nchannel, nchannel, K=K))
            
        if nblock > 1:
            self.expands = nn.ModuleList()
            
            for i in range(nblock-1):
                self.expands.append(attention.SimpleExtractF(nscale, K=K, ngp=1, bias=bias))
                self.expands.append(attention.PAConvGroup(nchannel, nchannel, K=K))                        

        self.sm = nn.Softmax(1) # softmax for the scale dimension
        self.nscale = nscale
        self.attention_type = attention_type

        self.alpha = alpha 
        self.tau = float(tau)

    def forward(self, d1, r1=None, debug=False):
        d = [None] * self.nblock; x = [None] * self.nblock
        if debug:
            w  = [None] * self.nblock;    w2 = [None] * (self.nblock-1)
            fs = [None] * (self.nblock*2 - 1)
            es = [None] * self.nblock
        
        d[0] = d1
        B, H, W = d1.shape[0], d1.shape[2], d1.shape[3]

        if debug:
            d[0] = d[0].cpu()
            
        # for each {i} stage
        for i in range(self.nblock):
            #--------------------------------------------
            #--------------------- squeeze --------------
            # d (multiscale depths) -> x (squeezed depth)
            #--------------------------------------------
            d1_features = self.attention_blocks[i*2](d1)
            
            w1 = self.attention_blocks[i*2+1](d1_features)
            w1 = w1 * d1_features

            x1 = self.softargmax(d1, w1, self.tau)
            x[i] = x1
            
            if debug:
                w1_sm = self.sm(w1)
                w[i] = w1_sm.cpu()
                x[i] = x[i].cpu()
                fs[i*2] = d1_features.reshape(d1.shape).cpu()

                # compute uncertainty
                C_xn = torch.sum(torch.abs(x[i] - d[i]), dim=1, keepdims=True)
                es[i] = C_xn
                # es[i] = (C_xn + 1.0) / ( self.nscale + 3.0 )

            if i == self.nblock-1:
                break
            
            #---------------------------------------
            #--------------------- expansion -------
            #---------------------------------------
            x1_expand = x1.repeat([1, self.nscale*2, 1, 1])
            x1_expand[:, 0:-1:2, :, :] = d1

            d_x = torch.absolute(d1 - x1)
            # d_x = torch.sqrt(d_x)
            x1_features = self.expands[2*i](d_x)
            x1_features_expand = x1_features.repeat_interleave(2, dim=1)

            x1_features_expand[:, 0:-1:2, :, :] = d1_features.reshape(d1.shape) # [B x 2S x H x W]
            
            x1_features_expand = x1_features_expand.reshape(-1, self.nchannel*2, H, W)
            ws = self.expands[2*i+1](x1_features_expand) # [B x 2S x H x W]
            ws = ws.reshape(x1_expand.shape)
            
            e = []
            if debug:
                wbar = w1_sm.clone()

            ws = ws * self.alpha
 
            for ii in range(self.nscale):
                smii = self.sm(ws[:,2*ii:2*ii+2,:,:])

                if debug:
                    wbar[:,ii, :, :] = smii[:,0,:,:].cpu()

                wx = smii * x1_expand[:,2*ii:2*ii+2,:,:]
                e.append(torch.sum(wx, dim=1, keepdim=True))

            d12 = torch.cat(e, dim=1)
            d2 = d12

            # d[i+1] = d2
            if debug:
                fs[i*2+1] = x1_features.cpu()
                w2[i] = wbar.cpu()
                d[i+1] = d2.cpu()
            
            d1 = d2
            
        if debug==False:
            return x, d, None, None
        else:
            return x, d, w, w2, fs, es

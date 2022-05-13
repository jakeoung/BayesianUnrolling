import torch
import torchvision.transforms.functional as TF
import numpy as np
import os
import h5py
from glob import glob

from torch.utils.data import Dataset, DataLoader

def normalize_per_channel(d):
    dd = d.reshape([d.shape[0], -1])
    dmax = torch.max(dd, dim=1, keepdims=True)[0].unsqueeze(2)
    dmin = torch.min(dd, dim=1, keepdims=True)[0].unsqueeze(2)
    return (d - dmin) / (dmax - dmin)

def get_refl_scales(r, nscale=3):
    rr = torch.reshape(r[:,nscale,:,:], [r.shape[0], -1])
    # rr = r.reshape([r.shape[:], -1])
    scales = torch.max(rr, dim=1, keepdims=True)[0]
    return scales.unsqueeze(2).unsqueeze(3)

class TofDataset(Dataset):
    def __init__(self, fname, nscale, istrain=True, pnorm=0, userefl=True):
        """
        Args:
            fname : h5 file name
        """
        with h5py.File(fname) as f:
            self.depths = torch.tensor(np.array(f['depths']))
            self.refls = None
            self.refls_gt = None
            self.userefl = userefl
            if userefl:
                self.refls = torch.tensor(np.array(f['refls']))
            
            self.available_gt = False
            if 'depths_gt' in f.keys():
                self.available_gt = True
                self.depths_gt = torch.tensor(np.array(f['depths_gt']))
                if userefl:
                    self.refls_gt = torch.tensor(np.array(f['refls_gt']))

            self.istrain = istrain

        if len(self.depths.shape) == 3:
            L, H, W = self.depths.shape[0], self.depths.shape[1], self.depths.shape[2]
            self.depths = torch.reshape(self.depths, [-1, L, H, W])
            self.refls = torch.reshape(self.refls, [-1, L, H, W])

        if nscale is not None:
            self.depths = self.depths[:,:nscale,:,:]
            if userefl:
                self.refls = self.refls[:,:nscale,:,:]

                if self.available_gt:
                    self.depths_gt = self.depths_gt[:,:nscale,:,:]
                    if userefl:
                        self.refls_gt = self.refls_gt[:,:nscale,:,:]

        self.nscale = self.depths.shape[1]

        self.pnorm = pnorm
        if istrain:
            self.augment = True

        else:
            self.augment = False

        self.scales = None # used for testing time

    def transform(self, a, b, c, d):
        if np.random.random() > 0.5:
            a = TF.vflip(a)
            c = TF.vflip(c)

            if self.userefl:
                b = TF.vflip(b)
                d = TF.vflip(d)

        if np.random.random() > 0.5:
            a = TF.hflip(a)
            c = TF.hflip(c)

            if self.userefl:
                b = TF.hflip(b)
                d = TF.hflip(d)

        return a,b,c,d

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        d = self.depths[idx,:]
        r_gt = None
        if self.userefl:
            r = self.refls[idx,:]
            if self.pnorm == 1:
                if len(d.shape) == 3:
                    scale = torch.max(r[3,:,:])  # assume the first
                    r /= scale
                else:
                    scales = get_refl_scales(r)
                    r /= scales
        else:
            r = d # temporary

        if self.available_gt is False:
            return d, r, None, None
        
        d_gt = self.depths_gt[idx,:]
        if self.userefl:
            r_gt = self.refls_gt[idx,:]
        else:
            r_gt = d_gt

        d_out = None
        if self.istrain:
            d, r, d_gt, r_gt = self.transform(d, r, d_gt, r_gt)
            return d, r, d_gt, r_gt
        else:
            return d, r, d_gt, r_gt
        
    def get_test(self, idxs, normalize=True):
        # if use_diff_scales == None:
        d = self.depths[idxs,:]
        if self.userefl:
            r = self.refls[idxs,:]
            if normalize:
                scales = get_refl_scales(r)
                r = r / scales
                self.scales = scales
        else:
            r = None

        d_gt = None 
        r_gt = None

        if self.available_gt:
            d_gt = self.depths_gt[idxs,:]
            if self.userefl:
                r_gt = self.refls_gt[idxs,:]
        
        return d, r, d_gt, r_gt
        # else:
        #     d = self.depths[idxs,iscales]
        #     r = self.refls[idxs,iscales]
        #     d_gt = self.depths_gt[idxs,:]
        #     r_gt = self.refls_gt[idxs,:]
        #     return d, r, d_gt, r_gt

    def add_ds(self, ds2):
        self.depths = torch.cat([self.depths, ds2.depths], dim=0)
        self.depths_gt = torch.cat([self.depths_gt, ds2.depths_gt], dim=0)
        if self.userefl:
            self.refls = torch.cat([self.refls, ds2.refls], dim=0)
            self.refls_gt = torch.cat([self.refls_gt, ds2.refls_gt], dim=0)
            
    # def __del__(self):
    #     del self.depths, self.refls, self.depths_gt, self.refls_gt

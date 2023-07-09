import torch
import numpy as np
# import jax.numpy as np

import scipy.ndimage
from jax import lax

def get_IF(F, T, leftT):  #leftT=100
    F[F < 0.01*F.max()] = 0
    h = np.concatenate([  np.zeros(leftT), F, np.zeros(leftT)  ])
    h /= sum(h)

    maxF = np.argmax(h)

    IF = np.zeros([len(h), len(h)])
    
    for i in range(len(h)):
        IF[:,i] = np.roll(h, i - maxF)
        # IF = IF.at[:, i].set(np.roll(h, i - maxF))
        # IF[:,i] = np.roll(h, i - maxF)

    return IF[:T, :T]
    # h = [np.zeros(leftT, 1); F; np.zeros([leftT, 1])]


"discard outliers for middlebury and mpi"
def discard_outliers(disp_, step, thresh=15):
    H, W = disp_.shape
    disp_ = disp_.copy()
    ii, jj = np.where(disp_ <= thresh)
    
    # final, result = lax.scan(funct, ii, jj)

    for it in range(len(ii)):
        idx0, idx1 = ii[it], jj[it]

        vv = []
        for j in range(max(0, idx1-step), min(W-1, idx1+step)):
            for i in range(max(0, idx0-step), min(H-1, idx0+step)):
                vv.append(disp_[i,j])

        # disp_ = disp_[idx0, idx1].set(np.median(vv))
        disp_[idx0, idx1] = np.median(vv)

    return disp_


def shift_h(F, T, shift=False):
    irf = np.zeros(T)
    irf[0:len(F)] = F
    irf = irf / np.sum(irf)
    attack = irf.argmax() + 1

    h = np.roll(irf, -attack)

    return h.copy()

def conv3d_separate(inp, K):
    # inp shape: [4, 1, T, H, W]
    
    # if K < 10:
    conv = torch.nn.functional.conv3d
    w1 = torch.ones(1, 1, K, 1, 1) / K
    w2 = torch.ones(1, 1, 1, K, 1) / K
    w3 = torch.ones(1, 1, 1, 1, K) / K
    w1 = w1.cuda()
    w2 = w2.cuda()
    w3 = w3.cuda()

    if inp.device != torch.device('cpu'):        
        return conv(conv(conv(inp, w1, padding="same"), w2, padding="same"), w3, padding="same")
    
    # for saving memory
    # obsolote code
    else:
        print("saving memory")

        for i in range(inp.shape[0]):
            t1 = conv(inp[i:i+1,:].cuda(), w1, padding="same")
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            t1 = conv(t1, w2, padding="same")
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            inp[i:i+1,:] = conv(t1, w3, padding="same").cpu()
            torch.cuda.empty_cache(); torch.cuda.synchronize()

            # inp[i:i+1,:] = conv(inp[i:i+1,:].cuda(), w2, padding="same").cpu()
            # inp[i:i+1,:] = conv(inp[i:i+1,:].cuda(), w3, padding="same").cpu()

        return inp


def conv3d_separate_cpu(inp, K):
    inp_np = inp.cpu().numpy()
    w = np.ones(K) / K
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=2, mode='constant')
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=3, mode='constant')
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=4, mode='constant')
    return torch.tensor(inp_np)
    
def conv2d(inp, K):
    # inp: [Tx1xHxW]
    if inp.device == torch.device('cpu'):
        inp_np = inp.numpy()
        w = np.ones(K) / K

        inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=2, mode='constant')
        inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=3, mode='constant')
        return torch.tensor(inp_np)

    else:
        w2 = torch.ones(1, 1, K, 1) / K
        inp1 = torch.nn.functional.conv2d(inp, w2.to(inp.device), padding="same")
        w2 = torch.ones(1, 1, 1, K) / K
        return torch.nn.functional.conv2d(inp1, w2.to(inp.device), padding="same")

def compute_refl(d0, attack, trailing, tof_conv, T):
    # d0 shape: [4 x H x W]
    nscale, H, W = d0.shape
    idx_start = d0 - attack
    idx_start[idx_start<0] = 0
    idx_end = d0 + trailing
    idx_end[idx_end > T-1] = T-1
    r0 = torch.zeros(d0.shape)

    for i in range(nscale):
        for iy, ix in np.ndindex((H,W)):
            r0[i, iy, ix] = tof_conv[i,0,idx_start[i, iy,ix]:idx_end[i, iy,ix]+1, iy, ix].sum()

    return r0


def gen_initial_multiscale(tof, h1, nscale_total, KK=[1,3,7,13], attack=None, trailing=None, use_cuda=None):
    if len(h1.shape) != 1:
        assert 0, "IRF should have be of 1D"
    if len(tof.shape) != 3:
        assert 0, "ToF shape should be of [height x width x time bin]"

    brefl = True if attack is not None else False

    if use_cuda == None:
        use_cuda = 1 if torch.cuda.is_available() else 0

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CPU mode is used and it will take several minutes (<= 4 minutes), depending on the data.")

    KK = np.array(KK)
    H, W, T = tof.shape
    
    if type(tof) == np.ndarray:
        tof = torch.tensor(tof)
        h1 = torch.tensor(h1)

    #----------------- 1. cross correlation of histogram with IRF
    # assume that h1 is already shifted
    tof_NxT = tof.reshape(tof.shape[0]*tof.shape[1], tof.shape[2])
    tof_irf = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1))
    tof_Tx1xHxW = tof_irf.transpose(0,1).reshape(T, 1, H, W)

    # tof_Tx1xHxW = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1)).transpose(0,1).reshape(T, 1, H, W)
    # tof_NxT = tof_NxT.cpu()

    del tof_NxT, tof, tof_irf
    if use_cuda:
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    
    # (4, 1, T, H, W)
    tof_conv = torch.zeros(4, 1, T, H, W)
    tof_conv[1, :] = conv2d(tof_Tx1xHxW, KK[1]).transpose(0, 1).reshape(1, 1, -1, H, W)
    tof_conv[2, :] = conv2d(tof_Tx1xHxW, KK[2]).transpose(0, 1).reshape(1, 1, -1, H, W)
    tof_conv[3, :] = conv2d(tof_Tx1xHxW, KK[3]).transpose(0, 1).reshape(1, 1, -1, H, W)
    tof_conv[0, 0, :] = tof_Tx1xHxW.cpu().squeeze(1)

    print("8 additional scales will be generated by separable 3d convolution.")
    #----------------- 3. 3D conv to generate two other sets 
    if use_cuda:
        tof_conv = tof_conv.cuda()
        d0 = torch.argmax(tof_conv, dim=2).cpu().squeeze()
        
        if nscale_total >= 8:
            d1 = torch.argmax(conv3d_separate(tof_conv, KK[-2]), dim=2).cpu().squeeze()

        if nscale_total >= 12:
            d2 = torch.argmax(conv3d_separate(tof_conv, KK[-1]), dim=2).cpu().squeeze()

        del tof_conv

    else:
        d0 = torch.argmax(tof_conv, dim=2).cpu().squeeze()

        if nscale_total >= 8:
            tof_conv2 = conv3d_separate_cpu(tof_conv, KK[-2])
            d1 = torch.argmax(tof_conv2, dim=2).squeeze()

        if nscale_total >= 12:
            tof_conv3 = conv3d_separate_cpu(tof_conv, KK[-1])
            d2 = torch.argmax(tof_conv3, dim=2).squeeze()

        if brefl:
            print("compute reflectivity for d0")
            r0 = compute_refl(d0, attack, trailing, tof_conv, T)
            r1 = compute_refl(d1, attack, trailing, tof_conv2, T)
            r2 = compute_refl(d2, attack, trailing, tof_conv3, T)

        del tof_conv, tof_conv2, tof_conv3

    depths_ = torch.cat([d0, d1, d2], dim=0)
    del d0, d1, d2
    
    # add 1 to be consistent with Julia and normalize wrt. T as a preprocessing
    depths = (depths_ + 1.0) / T  # 1/1024 ~ 1
    
    if brefl == False:
        return depths
    else:
        refls = torch.cat([r0, r1, r2], dim=0)
        return depths, refls

if __name__ == "__main__":
    import time
    tof_cpu = torch.zeros([4, 1, 1024, 500, 500])
    tof = tof_cpu.cuda()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    t0 = time.time()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    print(time.time() - t0)

import torch
import numpy as np
import scipy.ndimage

def shift_h(F, T, shift=False):
    irf = np.zeros(T)
    irf[0:len(F)] = F
    irf = irf / np.sum(irf)
    attack = irf.argmax() + 1

    h = np.roll(irf, -attack)
    return h.copy()

# def conv3d_separate2(inp, K):
#     # inp shape: [4, 1, T, H, W]
#     w3 = torch.ones(1, 1, K, 1, 1) / K
#     inp = torch.nn.functional.conv3d(inp, w3.to(inp.device), padding="same")

#     L, _, T, H, W = inp.shape
#     inp = inp.reshape([L, T, H, W])  # [4 x TH x W]
#     w2 = torch.ones(T, 1, K, 1) / K
#     inp = torch.nn.functional.conv2d(inp, w2.to(inp.device), padding="same", groups=T)
#     w2 = torch.ones(T, 1, 1, K) / K
#     inp = torch.nn.functional.conv2d(inp, w2.to(inp.device), padding="same", groups=T)
    
#     return inp.reshape(L, 1, T, H, W)

def conv3d_separate(inp, K):
    # shape: [4, 1, T, H, W]
    
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
        # # inp is of shape: [4, 1, T, H, W]
        # # inp = torch.nn.functional.conv3d(inp, w1.to(inp.device), padding="same")
        # # inp = torch.nn.functional.conv3d(inp, w2.to(inp.device), padding="same")
        # # return torch.nn.functional.conv3d(inp, w3.to(inp.device), padding="same")
        # return conv(conv(conv(inp, w1.to(inp.device), padding="same"),w2.to(inp.device), padding="same"),w3.to(inp.device), padding="same")

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

        # # conv = torch.nn.functional.conv2d
        # # w1 = torch.ones(1, 1, K, 1) / K
        # w1 = torch.ones(1, 1, K) / K
        # w2 = torch.ones(1, 1, 1, K) / K
        # w2 = w2.cuda()

        # t1 = torch.nn.functional.conv1d(inp.reshape(-1, 1, inp.shape[3]), w1.cuda(), padding="same").reshape(inp.shape)
        # # t1 = torch.nn.functional.conv1d(t1.reshape(-1, 1, inp.shape[3]), w1.cuda(), padding="same").reshape(inp.shape)
        
        # if torch.backends.cudnn.enabled == True or K <= 3:
        #     return torch.nn.functional.conv2d(t1, w2.cuda(), padding="same")
        # else:
        #     out = t1
        #     Thalf = inp.shape[0] // 2
        #     out[:Thalf] = torch.nn.functional.conv2d(out[:Thalf], w2, padding="same")
        #     out[Thalf:] = torch.nn.functional.conv2d(out[Thalf:], w2, padding="same")
        #     return out.cpu()

        # inp1 = torch.nn.functional.conv2d(inp, w2.cuda(), padding="same")
        # return torch.nn.functional.conv2d(inp1, w2.to(inp.device), padding="same")

def gen_initial_multiscale(tof, h1, nscale, KK=[1,3,7,13], use_cuda=None):
    if len(h1.shape) != 1:
        assert 0, "IRF should have be of 1D"
    if len(tof.shape) != 3:
        assert 0, "ToF shape should be of [height x width x time bin]"

    if use_cuda == None:
        use_cuda = 1 if torch.cuda.is_available() else 0

    KK = np.array(KK)
    KKhalf = KK // 2

    H, W, T = tof.shape
        
    if use_cuda == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CPU mode is used and it will take several minutes (<= 4 minutes), depending on the data.")

    # print("before use fft")

    #----------------- 1. cross correlation of histogram with IRF
    tof_NxT = tof.reshape(tof.shape[0]*tof.shape[1], tof.shape[2])
    tof_irf = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1))
    tof_Tx1xHxW = tof_irf.transpose(0,1).reshape(T, 1, H, W)

    # tof_Tx1xHxW = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1)).transpose(0,1).reshape(T, 1, H, W)
    # tof_NxT = tof_NxT.cpu()

    del tof_NxT, tof, tof_irf
    # return torch.zeros([12, H, W]).cuda()
    if use_cuda:
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # print("before spatial sampling")

    # (4, 1, T, H, W)
    if 0:
        tof_conv = torch.zeros(4, 1, T, H, W).to(device)
        tof_conv[1, :] = conv2d(tof_Tx1xHxW, KK[1]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[2, :] = conv2d(tof_Tx1xHxW, KK[2]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[3, :] = conv2d(tof_Tx1xHxW, KK[3]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[0, 0, :] = tof_Tx1xHxW.squeeze(1)
    else:
        tof_conv = torch.zeros(4, 1, T, H, W)
        tof_conv[1, :] = conv2d(tof_Tx1xHxW, KK[1]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[2, :] = conv2d(tof_Tx1xHxW, KK[2]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[3, :] = conv2d(tof_Tx1xHxW, KK[3]).transpose(0, 1).reshape(1, 1, -1, H, W)
        tof_conv[0, 0, :] = tof_Tx1xHxW.cpu().squeeze(1)

    del tof_Tx1xHxW # tof_conv2d
    # print(torch.cuda.memory_summary())
    # del tof_conv; torch.cuda.empty_cache(); torch.cuda.synchronize(); return torch.zeros([12, H, W]).cuda()

    # depths = torch.zeros(nscale, H, W)
    # depths = depths.cuda()
    # depths[0:4, :, :] = torch.argmax(tof_conv, dim=2).squeeze()

    print("4 scales completed. 8 additional scales are generated by separable 3d convolution.")
    #----------------- 3. 3D conv to generate two other sets 
    if use_cuda:
        tof_conv = tof_conv.cuda()
        d0 = torch.argmax(tof_conv, dim=2).cpu().squeeze()
        if nscale >= 8:
            d1 = torch.argmax(conv3d_separate(tof_conv, KK[-2]), dim=2).cpu().squeeze()

        if nscale >= 12:
            d2 = torch.argmax(conv3d_separate(tof_conv, KK[-1]), dim=2).cpu().squeeze()

        del tof_conv

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    else:
        d0 = torch.argmax(tof_conv, dim=2).cpu().squeeze()
        if nscale >= 8:
            d1 = torch.argmax(conv3d_separate_cpu(tof_conv, KK[-2]), dim=2).squeeze()

        if nscale >= 12:
            d2 = torch.argmax(conv3d_separate_cpu(tof_conv, KK[-1]), dim=2).squeeze()

        del tof_conv

    depths = torch.cat([d0, d1, d2], dim=0)
    del d0, d1, d2
    if use_cuda:
        depths = depths.cuda()

    # add 1 to be consistent with Julia and normalize wrt. T as a preprocessing
    return (depths + 1.0) / T
    # return depths

if __name__ == "__main__":
    import time
    tof_cpu = torch.zeros([4, 1, 1024, 500, 500])
    tof = tof_cpu.cuda()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    t0 = time.time()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    print(time.time() - t0)

    # t0 = time.time()
    # a = torch.argmax(conv3d_separate(tof_cpu, 5), dim=2)
    # print(time.time() - t0)
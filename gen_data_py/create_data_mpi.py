# The goal is to generate training data from Middlebury dataset
# Output: depth [nscale x H x W], reflectivity [nscale x K x H x W]
# nscale = number of scales
# K = number of wavelengths
# 21/23 images (436, 1024)

import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import sys 

from gen_initial_multiscale import *
from scipy import io
from PIL import Image

btrain = 1
KK = [1, 3, 7, 13]

ddata = "../data/mpi/"
os.makedirs(f"{ddata}images", exist_ok=True)

# n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
# reversedims(arr) = permutedims(arr, reverse(ntuple(identity, Val(ndims(arr)))))

#------------------------------------------------
# simulation params
#------------------------------------------------
T = 1024
T_max_val = 300

sbr = 64.0
ppp = 64.0

if len(sys.argv) > 1:
    ppp = float(sys.argv[1])
    if len(sys.argv) > 2:
        sbr = float(sys.argv[2])

nscale = 4
nirf = 3
KK = [1, 3, 7, 13]

szratio = 1
step = 6

fname = f"train_T={T}_ppp={ppp}_sbr={sbr}.h5"

H_patch = 256
W_patch = 256
stride_patch = 48

# train_idx = [1,3,4,5,6,7,8,9]

# train_depths = zeros(Float32, ntrain, nscale, 25x, 256)
# train_refls = zeros(Float32, ntrain, nscale, 256, 256)

list_depths = []
list_refls = []
list_depths_gt = []
list_refls_gt = []
idx_train = 0

#
#---------------------------------------
# construct impulse response function
#---------------------------------------
F = io.loadmat("F_real2_100s.mat")["F"]
# F = MAT.matread("F_real2_100s.mat")["F"]
F = F[:,99]
IF = get_IF(F, T, (T - F.shape[0])//2)
IF_mid = IF[:,int(T//2)]

plt.plot(IF_mid)

# prepare different impulse response functions
h1_original = IF_mid.reshape([1,-1])
h1 = shift_h(IF_mid, T)
trailing = np.argmax(h1 < 1e-5) - 1 # right side of IRF
attack = np.argmax(np.flip(h1, 0) < 1e-5) - 1 # left side of IRF
# attack depth trailing

dname = f"{ddata}train/"

import glob

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# seqs = readdir(f"{ddata}training/clean_left/")
seqs = listdir_nohidden(f"{ddata}training/clean_left/")
seqs_removed = ["mountain_1", "temple_3"] # "", "temple_2"

# i, seq = 2, "Art"
for i, seq in enumerate(seqs):
    if seq in seqs_removed:
        continue

    print(i)
    
    # (i<=8) && continue0
    fname_gt = f"{ddata}/training/clean_left/{seq}/{T}_{ppp}_{sbr}.mat"
    np.random.seed(1)

    # disp = Int.(reinterpret(UInt8, load(ddata*"/raw/"*seq*"/disp1.png")))
    disp = np.asarray(Image.open(f"{ddata}/training/disparities_viz/{seq}/frame_0001.png"))
    refl = np.asarray(Image.open(f"{ddata}/training/clean_left/{seq}/frame_0001.png").convert("L"))

    print(seq, disp.shape)
    
    disp = disp[18:-18, :]
    refl = refl[18:-18, :]

    disp0 = np.copy(disp)
    H_ori, W_ori = disp.shape

    # save("$ddata/images/$seq-disp0.png", disp0 ./ maximum(disp0) )

    #------------------------------------------------
    # 1. fill out outliers in disp
    #------------------------------------------------
    disp = discard_outliers(disp, step)
    disp = discard_outliers(disp, step)
    disp = discard_outliers(disp, step)
        
    for ii in range(20):
        noutliers = (disp == 0.0).sum()
        if noutliers > 0:
            print("noutliers:", noutliers)
        else:
            break
        
        disp = discard_outliers(disp, step + (ii+1)*2)
        
        if ii == 13:
            print(seq, "error")
            disp[np.where(disp <= 15)] = 16
            break        

    assert((disp == 0).sum() == 0)

    depth_ = T_max_val - disp  # 0 250 => 50 - 300
    
    # save images to check
    # [ ] TODO: display {disp}
    plt.imsave(f"{ddata}/images/{seq}-depth.png", depth_)

    #------------------------------------------------
    # 2. generate depth images
    #------------------------------------------------
    # for ART, we make the size the same as Reindeer
    depth = depth_
    reflg = refl

    # scale depth to lie 50 ~ 250
    depth_scaled = depth
    depth_quant = depth - 1 # python index starts from 0
    
    reflg = reflg / reflg.mean()
    H, W = reflg.shape
    
    #------------------------------------------------
    # 3. generate ToF data and GT
    #------------------------------------------------
    S = np.zeros([T, H, W])

    for iy, ix in np.ndindex((H, W)):
        S[depth_quant[iy, ix], iy, ix] = reflg[iy, ix]
    
    # convolution in time IF * intensity
    # [T x T] x [N x T]'
    N = H*W

    tt = IF.dot( S.reshape(1024, -1) )
    S_conv = tt.reshape([T, H, W])

    Lev_S = sbr * ppp / ( 1 + sbr)
    Lev_B = ppp - Lev_S

    bg = np.random.poisson(Lev_B/T, (T, H, W)) # 
    p1 = np.random.poisson(S_conv * Lev_S)
    tof_data = p1 + bg

    print("ppp", np.mean(np.sum(tof_data, 0)))
    print("sbr", Lev_S / Lev_B)

    d_gt_n = depth_scaled / T
    r_gt_n = reflg * Lev_S
    # MAT.matwrite(fname_gt, Dict("d_gt_n" => d_gt_n, "r_gt_n"=>r_gt_n, "reflg"=>reflg, "tof_data"=>tof_data); compress=true  )

    #------------------------------------------------
    # 4. estimate the initial depth and intensity
    #------------------------------------------------
    tof_HWT = tof_data.transpose([1,2,0])
    depths, refls = gen_initial_multiscale(tof_HWT, np.flip(h1).copy(), nscale*nirf, KK, attack=attack, trailing=trailing)
    # depths1, refls1 = gen_initial_multiscale(tof_HWT, h1, nscale*nirf, KK, attack=attack, trailing=trailing)

    # break

    #------------------------------------------------
    # 4. normalize reflecitvity, dividing by mean
    # another way would be to divide by max or normalization
    #------------------------------------------------
    r_scale = refls.mean([1,2])
    refls /= r_scale.unsqueeze(1).unsqueeze(1)
    r_gt_n /= r_gt_n.mean()

    # default(yflip=true)
    # plot(heatmap(depths[:,:,1]), heatmap(refls[:,:,1]),heatmap(depths[:,:,2]),heatmap(refls[:,:,2]), heatmap(depths[:,:,3]),heatmap(refls[:,:,3]), heatmap(depths[:,:,4]),heatmap(refls[:,:,4]), layout=(4,2), size=( 1000,1000))
    # savefig("$ddata/images/$seq-multiscale_$fname.png")

    #------------------------------------------------
    # (optional) extract patches
    #------------------------------------------------
    d_gt_n = torch.tensor(d_gt_n)
    r_gt_n = torch.tensor(r_gt_n)

    if btrain:
        for stride in [1, 2]:
            for ww in range(0, W-stride*W_patch+1, stride_patch):
                for hh in range(0, H-stride*H_patch+1, stride_patch):
                    list_depths.append(depths[:nscale*nirf, hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    list_refls.append(refls[:nscale*nirf, hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    list_depths_gt.append(d_gt_n[hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    list_refls_gt.append(r_gt_n[hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])

    # else:
    #     fname = f"test_{seq}_T={T}_ppp={ppp}_sbr={sbr}.h5"
        
    #     with h5py.File(f"{ddata}/{fname}", 'w') as hf:
    #         hf["depths"] = depths
    #         hf["refls"] = refls
    #         hf["depths_gt"] = d_gt_n
    #         hf["refls_gt"] = r_gt_n

    #     with h5py.File(f"{ddata}/{fname}", 'r') as hf:
    #         print(hf["depths"].shape)


#------------------------------------------------
# save training data
#-----------------------------------------------
import h5py
if btrain:
    with h5py.File(f"{ddata}/{fname}", 'w') as hf:
        hf["depths"] = torch.stack(list_depths, 0)
        hf["refls"] = torch.stack(list_refls, 0)
        hf["depths_gt"] = torch.stack(list_depths_gt, 0)
        hf["refls_gt"] = torch.stack(list_refls_gt, 0)

# with h5py.File(f"{ddata}/{fname}", 'r') as hf:
#     print(type(hf["depths"][0]))

import mat73
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
os.sys.path.append("../")
from unroll.model.vanilla import Model
from gen_initial import gen_initial_multiscale, shift_h

# torch.backends.cudnn.enabled = False # necessary to reduce memory

use_cuda = True if torch.cuda.is_available() else False

def main(scene="checkerboard", dout="../result/"):
    fname = f'{scene}.mat'
    fdict = scipy.io.loadmat(fname)
    spad = np.asarray(fdict["spad_processed_data"])[0][0]
    tof = np.array(scipy.sparse.csc_matrix.todense(spad))
    tof = torch.FloatTensor(tof.transpose([1, 0]).reshape([256, 256, -1]).transpose([1, 0, 2]))

    H, W, T = tof.shape

    print("use_fft, h1 is shifted to be used in fft")
    irf_ = mat73.loadmat("irf/irf_real.mat")["h1"]
    irf_[irf_ <= 0.0006] = 0.0
    irf_ /= sum(irf_)
    h1 = torch.FloatTensor(shift_h(irf_, T))

    L = 12
    depths = gen_initial_multiscale(tof, h1, L)
    
    t01 = time.time()
    
    model = Model(L)
    if use_cuda:
        model.cuda()
        model.load_state_dict(torch.load("model_baseline.pth"))
    else:
        model.load_state_dict(torch.load("model_baseline.pth", map_location=torch.device('cpu')))

    out = model(depths.unsqueeze(0))

    depth_final = out[0][-1].cpu().detach().numpy()

    # save the images for debugging
    depths = depths.detach().numpy()

    if os.path.isdir(dout) == False:
        os.makedirs(dout)

    plt.imsave(f"{dout}depth_0.png", depths[0,:,:])
    plt.imsave(f"{dout}depth_3.png", depths[3,:,:])

    to_meter = T * 0.0078 * 0.5
    plt.imshow(depth_final[0,0,:,:]*to_meter); plt.clim(0.5, 2.5); plt.colorbar()
    plt.savefig(f"{dout}depth_final.png"); plt.clf()
    scipy.io.savemat(f"{dout}{scene}.mat", {"depth":depth_final[0,0,:,:]})
    print(f"Results are saved in {dout}")

if __name__ == "__main__":
    # main("elephant")
    main("checkerboard")
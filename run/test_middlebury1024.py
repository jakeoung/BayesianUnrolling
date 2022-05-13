import mat73
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import time
# os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.sys.path.append("../")
from unroll.model.vanilla import Model
from gen_initial import gen_initial_multiscale, shift_h

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False

use_cuda = True if torch.cuda.is_available() else False

def main():
    f = './Art_4.0_1.0.mat'
    fdict = mat73.loadmat(f)
    tof = torch.FloatTensor(fdict["Y"])

    use_halfsize = False
    if use_halfsize:
        tof = tof[0:-1:2, 0:-1:2, :]
    
    H, W, T = tof.shape

    t0 = time.time()

    irf_ = scipy.io.loadmat("irf/irf_middlebury1024.mat")["irf"][0,:]    
    h1 = torch.FloatTensor(shift_h(irf_, T))

    L = 12
    depths = gen_initial_multiscale(tof, h1, L)
    del tof

    with torch.no_grad():
        model = Model(L)
        if use_cuda:
            torch.cuda.empty_cache()
            model.cuda()
            model.load_state_dict(torch.load("model_baseline.pth"))
        else:
            model.load_state_dict(torch.load("model_baseline.pth", map_location=torch.device('cpu')))
        
        out = model(depths.unsqueeze(0), debug=True) # if you don't need to compute uncertainty, debug=False 

    t1 = time.time()
    print("@ elapsed time:", t1 - t0)

    depth_final = out[0][-1].cpu().numpy()

    if os.path.isdir("../result/") == False:
        os.makedirs("../result/")

    to_meter = T * 0.003
    plt.imshow(depth_final[0,0,:,:]*to_meter); plt.clim(0.2, 0.8); plt.colorbar()
    plt.savefig("../result/depth_final.png"); plt.clf()

    scipy.io.savemat(f"../result/depth.mat", {"depth":depth_final[0,0,:,:]*to_meter })
    print("The result is saved in the folder ../result/")
    
    del depths
    # for i in range(model.nblock):
    #     del out[0][0]

    # from compute_uncertainty import compute_uncertainty
    # uncertainty = compute_uncertainty(out)

if __name__ == "__main__":
    main()
    main()

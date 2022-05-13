import numpy as np
import scipy.special
import scipy.io

def compute_uncertainty(out, to_meter=1024*0.003, foutmat=None):
    """
        - out: the raw output of the network with the debug option
        - to_meter: scaling factor to scale the output of network in the unit of meter
    """
    if out[2] == None:
        assert 0, "The input should be obtained by the inference with the debug option"

    ilast = -1
    x = out[0][ilast][0,0,:,:].numpy()
    rr = x
    w = out[2][ilast][0,:,:,:].numpy()
    d = out[1][0][0,:,:,:].numpy()
    # Cx = np.sum(w * np.absolute(d - np.expand_dims(x, 0)), axis=0)

    #------------ uncertainty
    # x, d, w, w2, fs, es, m, rw, r, rw2, rfs, res = out
    beta = 0
    alpha = 0

    w = out[2]
    w2 = out[3]
    d = out[1]

    d0 = d[0].numpy() * to_meter
    d1 = d[1].numpy() * to_meter
    d2 = d[2].numpy() * to_meter
    xi = out[0][-1].numpy() * to_meter
    ibatch=0
    ww = w2[0].numpy()[0,:,:,:]
    ww1 = w2[1].numpy()[0,:,:,:]
    ww2 = w2[2].numpy()[0,:,:,:]

    w2 = w[2].numpy()[0,:,:,:]

    ww_norm = scipy.special.softmax(1-ww, 0)
    C1 = ww_norm * np.absolute(d0[ibatch,:,:,:] - xi[ibatch,0:1,:,:]) / 14
    ww_norm = scipy.special.softmax(1-ww1, 0)
    C2 = ww_norm * np.absolute(d1[ibatch,:,:,:] - xi[ibatch,0:1,:,:]) / 14
    ww_norm = scipy.special.softmax(1-ww2, 0)
    C3 = ww_norm * np.absolute(d2[ibatch,:,:,:] - xi[ibatch,0:1,:,:]) / 14
    epsilon = np.sum(C1 + C2 + C3, axis=0) / 3

    # plt.imshow(epsilon); plt.show()
    if foutmat is not None:
        scipy.io.savemat(foutmat, {"depth":x, "refl":rr, "epsilon":epsilon} )
    
    return epsilon


## example
# eps = compute_uncertainty(out)
# plt.imshow(eps, vmin=0, vmax=0.01)
# plt.title("uncertainty")
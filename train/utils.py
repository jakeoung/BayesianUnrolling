import numpy as np
import matplotlib.pyplot as plt

def set_size(w,h, ax=None): # x, h size in inches
    if ax is None:
        ax=plt.gca()

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def vis_internal(out, writer, epoch, pp, ibatch=0, userefl=False, named_params=None):
    ca = 0.1; cb = 0.4;
    x = out[0]; d = out[1]; w = out[2]; w2 = out[3]; fs = out[4]; es = out[5]
    if len(out) > 6:
        m = out[5]; r = out[6]

    nblock = len(w)
    nscale = w[0].shape[1]
    # out = model(depth.cuda(), debug=True) #training samples
    
    d_test_np = x[nblock-1].numpy()
    # d_test_np = d_test_hat.numpy()
    
    fig=plt.figure(figsize=(15,10)); fig.add_subplot(1,3,1); plt.imshow(d_test_np[ibatch,0,:,:]); plt.colorbar();
    fig.add_subplot(1,3,2); plt.imshow(d_test_np[ibatch+1,0,:,:]); plt.colorbar()
    # fig.add_subplot(1,3,3); plt.imshow(d_test_np[ibatch+2,0,:,:]); plt.colorbar()
    writer.add_figure('{ibatch} final depths', fig, global_step=epoch)

    #------------------ visualize the intermediate depths
    # fig_dep_w = plt.figure(figsize=(26,13))
    # for layer in range(nblock): # 8 9 10 11
    #     ww = w[layer].numpy()
    #     di = d[layer].numpy()
    #     for i in range(nscale):
    #         fig_dep_w.add_subplot(nscale, nblock*2, i*2*nblock+ 1 + layer*2); im=plt.imshow(di[ibatch,i,:,:]); plt.colorbar(im)
    #         if i==0:
    #             plt.title(f"d block{layer+1}")
    #         fig_dep_w.add_subplot(nscale, nblock*2, i*2*nblock+ 1 + layer*2 + 1); plt.imshow(ww[ibatch,i,:,:], vmin=0, vmax=1, interpolation='nearest'); plt.colorbar()
    #         plt.title(f"{ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")
    #         if i==0:
    #             plt.title(f"w block{layer+1}, {ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")

    # writer.add_figure(f'{ibatch} internal expanded depths with attention weights', fig_dep_w, global_step=epoch)

    #---------------- layer by layer
    fig_ll = plt.figure(figsize=(50,13))
    for layer in range(nblock): # 8 9 10 11
        ww = w[layer].numpy()
        di = d[layer].numpy()
        if layer < nblock-1:
            ww2 = w2[layer].numpy()

        for i in range(nscale):
            fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4); im=plt.imshow(di[ibatch,i,:,:], vmin=0, vmax=1, interpolation='nearest');plt.clim(ca, cb); plt.colorbar(im);
            if i==0:
                plt.title(f"d block{layer+1}")
                
            fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4 + 1); plt.imshow(ww[ibatch,i,:,:], vmin=0, vmax=1, interpolation='nearest');plt.colorbar()
            plt.title(f"{ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")
            if i==0:
                plt.title(f"w block{layer+1}, {ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")
            
            if layer < nblock-1:
                fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4 + 3); plt.imshow(ww2[ibatch,i,:,:], vmin=0, vmax=1, interpolation='nearest'); plt.colorbar();
                plt.title(f"{ww2[ibatch,i,:,:].min():.2f} ~ {ww2[ibatch,i,:,:].max():.2f}")
            
        # add x
        xx = x[layer][ibatch,0,:,:] # B x 0 x H x W
        fig_ll.add_subplot(nscale, nblock*4, 0+ 1 + layer*4 + 2); plt.imshow(xx, vmin=0, vmax=1, interpolation='nearest'); plt.clim(ca, cb); plt.colorbar()

        epsilon = es[layer][ibatch,0,:,:] # B x 0 x H x W
        fig_ll.add_subplot(nscale, nblock*4, 1*4*nblock + 1 + layer*4 + 2); plt.imshow(np.log(epsilon+1e-8)); plt.clim(ca, cb); plt.colorbar()

        if layer < nblock-1:
            xx = fs[layer*2+1][ibatch,0,:,:] # B x 0 x H x W
            fig_ll.add_subplot(nscale, nblock*4, 2*4*nblock + 1 + layer*4 + 2); plt.imshow(xx, interpolation='nearest'); plt.clim(ca, cb); plt.colorbar()


    writer.add_figure(f'{ibatch} layer by layer', fig_ll, global_step=epoch)

    if ibatch > 0:
        return 

    #---------------- visulize the intermediate x
    # fig_x = plt.figure(figsize=(15,4))
    # for layer in range(nblock): # 0 1 2 3
    #     x_l = x[layer].numpy()
    #     fig_x.add_subplot(1, nblock+1, layer+1); plt.imshow(x_l[ibatch,0,:,:]); plt.colorbar()
    # # fig_x.add_subplot(1, nblock+1, nblock+1); plt.imshow(d_test_np[ibatch,0,:,:]); plt.colorbar()
    # writer.add_figure('internal squeezed depths', fig_x, global_step=epoch)

    H, W = x[0].shape[2], x[0].shape[3]
    
    #---------------- visualize point cloud? (V: B x N x 3)
    for iblock in range(nblock):
        ddd = x[iblock][0:2,:,:,:] # B x 0 x H x W
        pp[:,:,1] = ddd[:,0,:,:].reshape([-1, H*W])
        writer.add_mesh(f'internal squeezed {iblock} layer', pp, global_step=epoch)

    #---------------- visualize the second attention weights
    # fig_w = plt.figure(figsize=(13,13))
    # for layer in range(nblock-1): # 4 5 6 7
    #     ww = w2[layer].numpy()
    #     for i in range(nscale):
    #         fig_w.add_subplot(nscale, nblock-1, i*(nblock-1)+ 1+layer); im=plt.imshow(ww[ibatch,i,:,:]); plt.colorbar(im)
    #         # fig_w.add_subplot(nscale, nblock, layer+1); im = plt.imshow(max_prob, vmin=0, vmax=1); plt.title("maximum weights"); #plt.colorbar()
    # writer.add_figure('internal sub attention weights wbar', fig_w, global_step=epoch)

    #---------------- visualize d12
    fig_w = plt.figure(figsize=(13,13))
    for layer in range(nblock): # 4 5 6 7
        ww = fs[2*layer].numpy() # d_features
        for i in range(nscale):
            fig_w.add_subplot(nscale, nblock, i*(nblock)+ 1+layer); im=plt.imshow(ww[ibatch,i,:,:]); plt.colorbar(im)
            # fig_w.add_subplot(nscale, nblock, layer+1); im = plt.imshow(max_prob, vmin=0, vmax=1); plt.title("maximum weights"); #plt.colorbar()
    writer.add_figure('d extracted features', fig_w, global_step=epoch)

    #---------------- 
    if named_params is not None:
        for tag, param in named_params:
            # print(tag)
            writer.add_histogram(tag, param.grad.data.cpu().numpy(), global_step=epoch)
                
    if userefl == False:
        return
    
        
    #---------------- reflectivity layer by layer
    x = out[6]; w = out[7]; d = out[8]; w2 = out[9]; fs = out[10]; es = out[11]

    fig_ll = plt.figure(figsize=(50,13))
    for layer in range(nblock): # 8 9 10 11
        ww = w[layer].numpy()
        di = d[layer].numpy()
        if layer < nblock-1:
            ww2 = w2[layer].numpy()

        for i in range(nscale):
            fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4); im=plt.imshow(di[ibatch,i,:,:], interpolation='nearest'); plt.colorbar(im)
            if i==0:
                plt.title(f"r block{layer+1}")
                
            fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4 + 1); plt.imshow(ww[ibatch,i,:,:], vmin=0, vmax=1, interpolation='nearest'); plt.colorbar()
            plt.title(f"{ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")
            if i==0:
                plt.title(f"w block{layer+1}, {ww[ibatch,i,:,:].min():.2f} ~ {ww[ibatch,i,:,:].max():.2f}")
            
            if layer < nblock-1:
                fig_ll.add_subplot(nscale, nblock*4, i*4*nblock+ 1 + layer*4 + 3); plt.imshow(ww2[ibatch,i,:,:], interpolation='nearest'); plt.colorbar()
                plt.title(f"{ww2[ibatch,i,:,:].min():.2f} ~ {ww2[ibatch,i,:,:].max():.2f}")
            
        # add x
        # xx = x[layer][ibatch,0,:,:] # B x 0 x H x W
        # fig_ll.add_subplot(nscale, nblock*4, 0+ 1 + layer*4 + 2); plt.imshow(xx, interpolation='nearest'); plt.colorbar()

        # epsilon = es[layer][ibatch,0,:,:] # B x 0 x H x W
        # fig_ll.add_subplot(nscale, nblock*4, 1*4*nblock + 1 + layer*4 + 2); plt.imshow(np.log(epsilon+1e-8)); plt.colorbar()

        # if layer < nblock-1:
        #     xx = fs[layer*2+1][ibatch,0,:,:] # B x 0 x H x W
            # fig_ll.add_subplot(nscale, nblock*4, 2*4*nblock + 1 + layer*4 + 2); plt.imshow(xx, interpolation='nearest'); plt.colorbar()


    writer.add_figure(f"{ibatch} layer by layer refl", fig_ll, global_step=epoch)

    

    # fig_w = plt.figure(figsize=(15,4))
    # for layer in range(nblock): # 4 5 6 7
    #     ww = out[nblock + layer].numpy()
    #     max_prob = np.max(ww[ibatch,:,:,:], 0)
    #     fig_w.add_subplot(1, nblock, layer+1); im = plt.imshow(max_prob, vmin=0, vmax=1); plt.title("maximum weights"); #plt.colorbar()
    # fig_w.colorbar(im)
    # writer.add_figure('internal maximum weights', fig_w, global_step=epoch)

    #----------------- visualize the kernels
    # if epoch % 500 == 0 and epoch > 0:
    #     kernels_vis = []
    #     for i in range(0, 1):
    #         kernels_vis.append(model.attention_blocks[i].conv0); kernels_vis.append(model.attention_blocks[i].conv1); kernels_vis.append(model.attention_blocks[i].conv2);

    #     for i,convs in enumerate(kernels_vis):
    #         kernels = convs.weight.cpu().detach().clone()
    #         fig_k = vis_kernels(kernels)
    #         writer.add_figure(f"kernels attention block {i}", fig_k, global_step=epoch)        

# https://www.linkedin.com/pulse/custom-function-visualizing-kernel-weights-activations-arun-das/
def vis_kernels(kernels, path=None, cols=None, normalize=True):
    """Visualize weight and activation matrices learned 
    during the optimization process. Works for any size of kernels.
    
    Arguments
    =========
    kernels: Weight or activation matrix. Must be a high dimensional
    Numpy array. Tensors will not work.
    path: Path to save the visualizations.
    cols: TODO: Number of columns (doesn't work completely yet.)
    
    Example
    =======
    kernels = model.conv1.weight.cpu().detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    custom_viz(kernels, 'results/conv1_weights.png', 5)
    """
    
    if normalize:
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()

    N = kernels.shape[0]
    C = kernels.shape[1]

    Tot = N*C

    # If single channel kernel with HxW size,# plot them in a row.# Else, plot image with C number of columns.
    if C>1:
        columns = C
    elif cols==None:
        columns = N
    elif cols:
        columns = cols
    rows = Tot // columns 
    rows += Tot % columns

    pos = range(1,Tot + 1)

    fig = plt.figure(1)
    fig.tight_layout()
    k=0
    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(rows,columns,pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k+1

    set_size(30,30,ax)
    
    return fig
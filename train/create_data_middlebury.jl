# The goal is to generate training data from Middlebury dataset
# Output: depth [nscale x H x W], reflectivity [nscale x K x H x W]
# nscale = number of scales
# K = number of wavelengths
# 9

"""
(seq, size(disp)) = ("Aloe", (555, 641))                                                                                            
(seq, size(disp)) = ("Books", (555, 695))                                                                                           
(seq, size(disp)) = ("Bowling1", (555, 626))                                                                                        
(seq, size(disp)) = ("Dolls", (555, 695))                                                                                           
(seq, size(disp)) = ("Lampshade2", (555, 650))                                                                                      
(seq, size(disp)) = ("Laundry", (555, 671))                                                                                         
(seq, size(disp)) = ("Midd1", (555, 698))                                                                                           
(seq, size(disp)) = ("Moebius", (555, 695))                                                                                        
(seq, size(disp)) = ("Plastic", (555, 635))                                                                                         

(seq, size(disp)) = ("Art", (555, 695))                                                                                             
(seq, size(disp)) = ("Reindeer", (555, 671))
"""

using Random
using FileIO
using Plots
using Images
using ImageFiltering
using DrWatson
using Distributions
using HDF5
using StatsBase
using Statistics
using FFTW
using MAT
using Flux

# using Pkg; Pkg.build("PyPlot")
# import PyPlot
save_train = true
save_test = true

include("gen_hist.jl")
include("gen_multiscale.jl")

ddata = "../data/middlebury/"
if ~isdir(ddata*"images")
    mkdir(ddata*"images")
end

n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
reversedims(arr) = permutedims(arr, reverse(ntuple(identity, Val(ndims(arr)))))

#------------------------------------------------
# simulation params
#------------------------------------------------
T = 1024
T_max_val = 300

show(ARGS)
sbr_inp = 64.0
ppp_inp = 64.0

if length(ARGS) >= 1
    sbr_inp = parse(Float64, ARGS[1])
    if length(ARGS) >= 2
        ppp_inp = parse(Float64, ARGS[2])
    end
end

params = (T = T, ppp = ppp_inp, sbr = sbr_inp)
# params = (T = T, ppp = 2.0, sbr = 0.5)

@show params

# if params.sbr > 4.0
#     save_train = false
# end

nscale = 4
nirf = 3
KK = [1 3 7 13]

szratio = 1
step = 6

fname = savename(params)*".h5"
H_patch = 256
W_patch = 256
stride_patch = 48

train_idx = [1,3,4,5,6,7,8,9]

# train_depths = zeros(Float32, ntrain, nscale, 25x, 256)
# train_refls = zeros(Float32, ntrain, nscale, 256, 256)

list_depths = Array{Float32, 3}[]
list_refls = Array{Float32, 3}[]
list_depths_gt = Array{Float32, 2}[]
list_refls_gt = Array{Float32, 2}[]
idx_train = 0

#---------------------------------------
# construct impulse response function
#---------------------------------------
F = MAT.matread("F_real2_100s.mat")["F"]
F = F[:,100]
IF = get_IF(F, T, 300)
IF_mid = IF[:,Int(T//2)]

# prepare different impulse response functions
h1_original = reshape(IF_mid, 1, :)
h1, a1, t1, _ = shift_h(IF_mid, T)

h2 = imfilter(h1, ones(1, KK[end-1]) ./ KK[end-1], "circular")

h3 = imfilter(h1, ones(1, KK[end]) ./ KK[end], "circular")

a2 = findfirst(x -> x < 1e-5, h2[end:-1:1]) - 2
t2 = findfirst(x -> x < 1e-5, h2[1:end]) - 1

a3 = findfirst(x -> x < 1e-5, h3[end:-1:1]) - 2
t3 = findfirst(x -> x < 1e-5, h3[1:end]) - 1
dname = ddata*"train/"

# prepare 
# i, seq = 2, "Art"
for (i,seq) in enumerate(["Aloe" "Books" "Bowling1" "Dolls" "Laundry" "Midd1" "Moebius" "Plastic"]) #"Reindeer" "Bowling2"])
    
    # (i<=8) && continue0
    fname_gt = ddata*"/raw/"*seq*"/$(params.T)_$(params.ppp)_$(params.sbr).mat"
    Random.seed!(1)

    disp = Int.(reinterpret(UInt8, load(ddata*"/raw/"*seq*"/disp1.png")))
    refl = load(ddata*"/raw/"*seq*"/view1.png")

    @show seq, size(disp)
    # d_out = Bool.(disp .<= 15)
    
    if seq[1:3] == "Mid"
        disp = disp[40:end,:]
        refl = refl[40:end,:]
    end

    disp0 = copy(disp)
    H_ori, W_ori = size(disp)

    #------------------------------------------------
    # 1. fill out outliers in disp
    #------------------------------------------------
    if seq == "Bowling1" 
        discard_outliers!(disp, 3)
        discard_outliers!(disp, 3)
        # discard_outliers!(disp, 3)
        miss_idxs = findall(disp .<= 15)
        disp[miss_idxs] .= 15
    else
        discard_outliers!(disp, step)
        discard_outliers!(disp, step)
        discard_outliers!(disp, step)
    end
    
    miss_idxs = findall(disp .== 0)
    if length(miss_idxs) > 0
        println("$seq missing index")
        discard_outliers!(disp, step)
        if length(findall(disp .<= 15))>0
            println("$seq missing index")
            discard_outliers!(disp, step); discard_outliers!(disp, step)
            discard_outliers!(disp, step); discard_outliers!(disp, step)
            discard_outliers!(disp, step); discard_outliers!(disp, step)
        end
        if length(findall(disp .<= 15))>0
            println("$seq missing index")
            discard_outliers!(disp, step); discard_outliers!(disp, step)
        end
        if length(findall(disp .<= 15))>0
            println("$seq missing index")
            discard_outliers!(disp, step); discard_outliers!(disp, step)
        end
    end
    depth_ = T_max_val .- disp  # 0 250 => 50 - 300
    
    # save images to check
    heatmap(disp, yflip=true);
    savefig("$ddata/images/$seq-disp.png")

    #------------------------------------------------
    # 2. generate depth images
    #------------------------------------------------
    # for ART, we make the size the same as Reindeer
    depth = depth_
    if i in test_idx
        depth = depth[:, end-W_test+1:end]
        refl = refl[:, end-W_test+1:end]
    end

    reflc = channelview(refl)
    reflg = Float32.(Gray.(refl))

    # scale depth to lie 50 ~ 250
    depth_scaled = Float32.(depth)
    depth_quant = depth
    
    reflg ./= mean(reflg)
    H, W = size(reflg)
    
    #------------------------------------------------
    # 3. generate ToF data and GT
    #------------------------------------------------
    # make clean S
    # global S, T
    intensity = reflg
    S = zeros(H, W, T)
    Threads.@threads for n in CartesianIndices((H,W))
        S[n, depth_quant[n]] = intensity[n]
    end
    
    # convolution in time IF * intensity
    # [T x T] x [N x T]'
    N = H*W
    tt = IF * reshape(permutedims(S, (3, 1, 2)), T, :)
    S_conv = reshape(permutedims(tt, (2, 1)), size(S))

    # Lev_S = params.sppp
    # Lev_B = params.bppp
    # sbr = Lev_S / Lev_B

    Lev_S = params.sbr * params.ppp / ( 1 + params.sbr)
    Lev_B = params.ppp - Lev_S

    p_bg = Distributions.Poisson(Lev_B/T)
    bg = rand(p_bg, H, W, T)
    p1 = rand.(Distributions.Poisson.(S_conv * Lev_S))
    tof_data = p1 + bg

    @show "ppp", mean(sum(tof_data, dims=3)) # sbr
    @show "sbr", Lev_S / (Lev_B) # sbr

    # global d_gt_n, depth_scaled, T
    d_gt_n = depth_scaled / T
    r_gt_n = reflg .* Lev_S
    MAT.matwrite(fname_gt, Dict("d_gt_n" => d_gt_n, "r_gt_n"=>r_gt_n, "reflg"=>reflg, "tof_data"=>tof_data); compress=true  )

    #------------------------------------------------
    #------------------------------------------------
    # 4. estimate the initial depth and intensity
    #------------------------------------------------
    #------------------------------------------------
    # Hf, a, t = get_Hf(IF_mid, T)
    tof_NxT = reshape(tof_data, H*W, T)

    Z_F = FourierTools.conv(h1, tof_NxT, [2]);
    tof_conv = reshape(Z_F, H, W, T)
    # plot_hist(tof_data, tof_conv, reflg)
    # savefig("$(ddata)/images/$seq-multiscale_$(fname)-hist1.png")

    # process zero photon pixels
    depths, refls, _, _ = gen_multiscale(tof_conv, nscale, KK, t1, a1)
    # depths, refls, temp = gen_multiscale(tof_conv, nscale, KK, t1, a1)
    
    # default(yflip=true)
    # plot(heatmap(depths[:,:,1]), heatmap(refls[:,:,1]),heatmap(depths[:,:,2]),heatmap(refls[:,:,2]), heatmap(depths[:,:,3]),heatmap(refls[:,:,3]), heatmap(depths[:,:,4]),heatmap(refls[:,:,4]), layout=(4,2), size=( 1000,1000))
    # default()
    # savefig("$(ddata)/images/$seq-multiscale_$fname-1.png")
    
    #------------------------------------------------
    # 7. (optional) different IRF
    #------------------------------------------------
    # generate for subsampled IRF
    # trailing_half = Int(trailing // 2)
    # attack_half = Int(attack // 2)
    if nirf > 1
        K = KK[end-1]        
        tof_conv2 = imfilter(tof_conv, ones(K,1,1)./K)
        tof_conv2 = imfilter(tof_conv2, ones(1,K,1)./K)
        tof_conv2 = imfilter(tof_conv2, ones(1,1,K)./K)
        
        depths2, refls2, _, _ = gen_multiscale(tof_conv2, nscale, KK, t1, a2)

        K = KK[end]
        tof_conv3 = imfilter(tof_conv, ones(K,1,1)./K)
        tof_conv3 = imfilter(tof_conv3, ones(1,K,1)./K)
        tof_conv3 = imfilter(tof_conv3, ones(1,1,K)./K)
        
        depths3, refls3, _, _ = gen_multiscale(tof_conv3, nscale, KK, t3, a3)
        default(yflip=true)
        
        plot(heatmap(depths3[:,:,1]), heatmap(refls3[:,:,1]),heatmap(depths3[:,:,2]),heatmap(refls3[:,:,2]), heatmap(depths3[:,:,3]),heatmap(refls3[:,:,3]), heatmap(depths3[:,:,4]),heatmap(refls3[:,:,4]), layout=(4,2), size=( 1000,1000))

        savefig("$ddata/images/$seq-multiscale_$fname-3.png")
        default()
        # plot_hist(tof_sf4, tof_conv, reflg)
        # savefig("$(ddata)/images/$seq-multiscale_$(fname)-hist3_$(dilation).png")

        @info "cat depths, refls"
        depths = cat(depths, depths2, depths3, dims=3)
        refls = cat(refls, refls2, refls3, dims=3)

        # max_depths = maximum(depths, dims=3)
        # min_depths = minimum(depths, dims=3)
        # @views plot(heatmap(min_depths[:,:,1]), heatmap(max_depths[:,:,1]))
        # savefig("$(ddata)/images/$seq-multiscale_$(fname)-minmax_$(scale_irf).png")
    end

    depths ./= T
    r_scale = maximum(refls[:,:,nscale])
    if i in train_idx
        refls ./= r_scale 
    end
    r_gt_n ./= maximum(r_gt_n)

    default(yflip=true)
    plot(heatmap(depths[:,:,1]), heatmap(refls[:,:,1]),heatmap(depths[:,:,2]),heatmap(refls[:,:,2]), heatmap(depths[:,:,3]),heatmap(refls[:,:,3]), heatmap(depths[:,:,4]),heatmap(refls[:,:,4]), layout=(4,2), size=( 1000,1000))
    savefig("$ddata/images/$seq-multiscale_$fname.png")

    #------------------------------------------------
    # (optional) extract patches
    #------------------------------------------------
    if save_train
        if i in train_idx
            for stride in [1 2]
                for ww=1:stride_patch:W - stride*W_patch+1
                    for hh=1:stride_patch:H - stride*H_patch+1
                        push!(list_depths, depths[hh:stride:hh+stride*H_patch-1, ww:stride:ww+stride*W_patch-1, 1:nscale*nirf])
                        push!(list_refls, refls[hh:stride:hh+stride*H_patch-1, ww:stride:ww+stride*W_patch-1, 1:nscale*nirf])
                        push!(list_depths_gt, d_gt_n[hh:stride:hh+stride*H_patch-1, ww:stride:ww+stride*W_patch-1])
                        push!(list_refls_gt, r_gt_n[hh:stride:hh+stride*H_patch-1, ww:stride:ww+stride*W_patch-1])
                    end
                end
            end
        end
    end
end

#------------------------------------------------
# save training data
#------------------------------------------------
if save_train
    rm("$ddata/train_$fname", force=true)
    rm("$ddata/tof_$fname", force=true)
    # H W C B -> W H C B in julia when saving h5file
    # it will load [B C H W] in python   H W C
    list_depths_pyshape = permutedims(Flux.batch(list_depths), (2, 1, 3, 4))
    list_refls_pyshape = permutedims(Flux.batch(list_refls), (2, 1, 3, 4))
    list_depths_gt = permutedims(reshape(Flux.batch(list_depths_gt), (H_patch,W_patch,1,:)), (2, 1, 3, 4))
    list_refls_gt = permutedims(reshape(Flux.batch(list_refls_gt), (H_patch,W_patch,1,:)), (2, 1, 3, 4))

    h5write("$ddata/train_$fname", "/depths", list_depths_pyshape)
    h5write("$ddata/train_$fname", "/refls", list_refls_pyshape)
    h5write("$ddata/train_$fname", "/depths_gt", list_depths_gt)
    h5write("$ddata/train_$fname", "/refls_gt", list_refls_gt)
end
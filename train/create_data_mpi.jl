# The goal is to generate training data from Middlebury dataset
# Output: depth [nscale x H x W], reflectivity [nscale x K x H x W]
# nscale = number of scales
# K = number of wavelengths
# 21/23 images (436, 1024)

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
using Random

save_train = true

include("gen_hist.jl")
include("gen_multiscale.jl")

ddata = "../data/mpi/"
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
ppp_inp = 64.0
sbr_inp = 64.0

if length(ARGS) >= 1
    sbr_inp = parse(Float64, ARGS[1])
    if length(ARGS) >= 2
        ppp_inp = parse(Float64, ARGS[2])
    end
end


# params = (T = T, ppp = 1.0, sbr = 1.0)
params = (T = T, ppp = ppp_inp, sbr = sbr_inp)

nscale = 4
nirf = 3
sigma1 = 4
sigma2 = 8
KK = [1 3 7 13]

szratio = 1
step = 6

fname = savename(params)*".h5"
H_patch = 256
W_patch = 256
stride_patch = 48

list_depths = Array{Float32, 3}[]
list_refls = Array{Float32, 3}[]
list_depths_gt = Array{Float32, 2}[]
list_refls_gt = Array{Float32, 2}[]

#---------------------------------------
# construct impulse response function
#---------------------------------------
F = MAT.matread("F_real2_100s.mat")["F"]
F = F[:,100]
IF = get_IF(F, T, 300)
IF_mid = IF[:,Int(T//2)]

h1_original = reshape(IF_mid, 1, :)
h1, a1, t1, _ = shift_h(IF_mid, T)

h2 = imfilter(h1, ones(1, KK[end-1]) ./ KK[end-1], "circular")
h3 = imfilter(h1, ones(1, KK[end]) ./ KK[end], "circular")

a2 = findfirst(x -> x < 1e-5, h2[end:-1:1]) - 2
t2 = findfirst(x -> x < 1e-5, h2[1:end]) - 1

a3 = findfirst(x -> x < 1e-5, h3[end:-1:1]) - 2
t3 = findfirst(x -> x < 1e-5, h3[1:end]) - 1

seqs = readdir(ddata*"training/clean_left/")
# i, seq = 1, seqs[1]

seqs_removed = ["mountain_1", "temple_3"] # "", "temple_2"

# i, seq = 4, "ambush_4"
for (i,seq) in enumerate(seqs)
    if seq in seqs_removed
        continue
    end
    Random.seed!(1)
    fname_gt = ddata*"training/clean_left/"*seq*"/$(params.T)_$(params.ppp)_$(params.sbr).mat"
    # if ispath(fname_gt) == false
        disp = Int.(reinterpret(UInt8, load(ddata*"training/disparities_viz/"*seq*"/frame_0001.png")))
        # disp = Float64.(load(ddata*"training/disparities_viz/"*seq*"/frame_0001.png")) #load(ddata*"disparities_viz/"*seq*"/frame_0001.png")
        refl = load(ddata*"training/clean_left/"*seq*"/frame_0001.png")

        # 436 
        disp = disp[19:end-18, :]
        refl = refl[19:end-18, :]

        @show seq, size(disp)
        
        disp0 = copy(disp)
        H_ori, W_ori = size(disp)

        save("$ddata/images/$seq-disp0.png", disp0 ./ maximum(disp0) )

        #------------------------------------------------
        # 1. fill out outliers in disp
        #------------------------------------------------
        discard_outliers!(disp, step)
        discard_outliers!(disp, step)
        discard_outliers!(disp, step)
        
        is_error = false
        for i=1:1000
            noutliers = sum(disp .== 0.0)
            if noutliers > 0
                println("noutliers:", noutliers)
            else
                break
            end
            discard_outliers!(disp, step+i*2)
            # disp = discard_outliers!((disp, step+i*2)
            if i == 14
                is_error = true
                break
            end
        end
        if is_error
            print("$seq error")
            miss_idxs = findall(disp .<= 15)
            disp[miss_idxs] .= 16
            # continue
        end

        @assert sum(disp .== 0.0) == 0
        @show minimum(disp)

        depth_ = T_max_val .- disp
        
        # save images to check
        # save("$ddata/images/$seq-disp.png", n01(disp) )
        # heatmap(depth_, yflip=true);
        # savefig("$ddata/images/$seq-depth.png")

        #------------------------------------------------
        # 2. generate a depth image and downsample it
        #------------------------------------------------
        # H_ori = H_ori - mod(H_ori, 8)
        # W_ori = W_ori - mod(W_ori, 8)
        depth = depth_[1:H_ori, 1:W_ori]
        refl = refl[1:H_ori, 1:W_ori]
        
        reflc = channelview(refl)
        # reflc ./= maximum(reflc)
        reflg = Float32.(Gray.(refl))
        save("$ddata/images/$seq-refl.png", n01(reflg) )

        depth_scaled = depth
        depth_quant = Int.(round.(depth_scaled))
        
        # intensity = n01(reflg)
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

        MAT.matwrite(fname_gt, Dict("d_gt_n" => d_gt_n, "r_gt_n"=>r_gt_n, "reflg"=>reflg, "tof_data"=>tof_data);compress=true )

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

    #------------------------------------------------
    # 7. (optional) different IRF
    #------------------------------------------------
    # generate for subsampled IRF
    # trailing_half = Int(trailing // 2)
    # attack_half = Int(attack // 2)
    if nirf > 1
        # Hf, a2, t2 = get_Hf(IF_mid, T, sigma1)
        # Z_F = real(ifft( fft_tof .* Hf, 2))

        K = KK[end-1]
        
        tof_conv2 = imfilter(tof_conv, ones(K,1,1)./K)
        tof_conv2 = imfilter(tof_conv2, ones(1,K,1)./K)
        tof_conv2 = imfilter(tof_conv2, ones(1,1,K)./K)

        depths2, refls2, _, _ = gen_multiscale(tof_conv2, nscale, KK, t2, a2)
        # plot(heatmap(depths2[:,:,1]), heatmap(refls2[:,:,1]),heatmap(depths2[:,:,2]),heatmap(refls2[:,:,2]), heatmap(depths2[:,:,3]),heatmap(refls2[:,:,3]), heatmap(depths2[:,:,4]),heatmap(refls2[:,:,4]), layout=(4,2), size=( 1000,1000))
        # savefig("$ddata/images/$seq-multiscale2_$fname.png")
        default(yflip=true)
        
        plot(heatmap(depths2[:,:,1]), heatmap(refls2[:,:,1]),heatmap(depths2[:,:,2]),heatmap(refls2[:,:,2]), heatmap(depths2[:,:,3]),heatmap(refls2[:,:,3]), heatmap(depths2[:,:,4]),heatmap(refls2[:,:,4]), layout=(4,2), size=( 1000,1000))
        savefig("$ddata/images/$seq-multiscale_$fname-2.png")
        default()
        
        K = KK[end]
        
        tof_conv3 = imfilter(tof_data, ones(K,1,1)./K)
        tof_conv3 = imfilter(tof_conv3, ones(1,K,1)./K)
        tof_conv3 = imfilter(tof_conv3, ones(1,1,K)./K)
        
        depths3, refls3, _, _ = gen_multiscale(tof_conv3, nscale, KK, t3, a3)
        default(yflip=true)
        
        plot(heatmap(depths3[:,:,1]), heatmap(refls3[:,:,1]),heatmap(depths3[:,:,2]),heatmap(refls3[:,:,2]), heatmap(depths3[:,:,3]),heatmap(refls3[:,:,3]), heatmap(depths3[:,:,4]),heatmap(refls3[:,:,4]), layout=(4,2), size=( 1000,1000))
            
        savefig("$ddata/images/$seq-multiscale_$fname-3.png")
        default()
        
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
    refls ./= r_scale
    r_gt_n ./= maximum(r_gt_n)

    #------------------------------------------------
    # (optional) extract patches
    #------------------------------------------------
    if save_train
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
#------------------------------------------------
# save training data
#------------------------------------------------
    rm("$ddata/train_$fname", force=true)
    rm("$ddata/tof_$fname", force=true)
    # H W C B -> W H C B in julia when saving h5file
    # it will load [B C H W] in python   H W C
    list_depths_pyshape = permutedims(Flux.batch(list_depths), (2, 1, 3, 4))
    list_refls_pyshape = permutedims(Flux.batch(list_refls), (2, 1, 3, 4))
    list_depths_gt = permutedims(reshape(Flux.batch(list_depths_gt), (H_patch,W_patch,1,:)), (2, 1, 3, 4))
    list_refls_gt = permutedims(reshape(Flux.batch(list_refls_gt), (H_patch,W_patch,1,:)), (2, 1, 3, 4))
    # list_depths_out = permutedims(reshape(Flux.batch(list_depths_out), (H_patch,W_patch,1,:)), (2, 1, 3, 4))

    h5write("$ddata/train_$fname", "/depths", list_depths_pyshape)
    h5write("$ddata/train_$fname", "/refls", list_refls_pyshape)
    h5write("$ddata/train_$fname", "/depths_gt", list_depths_gt)
    h5write("$ddata/train_$fname", "/refls_gt", list_refls_gt)
    # h5write("$ddata/train_$fname", "/depths_out", list_depths_out)

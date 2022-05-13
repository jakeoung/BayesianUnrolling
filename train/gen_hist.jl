using ImageFiltering

"discard outliers for middlebury and mpi"
function discard_outliers!(disp_, step, thresh=15)
    H, W = size(disp_)
    miss_idxs = findall(disp_ .<= thresh)
    cnt = 0
    # irand = randperm(length(miss_idxs))
    for idx in miss_idxs
        # idx = miss_idxs[ii]
        vv = []
        for j=max(1,idx[2]-step):min(W,idx[2]+step)
            for i=max(1,idx[1]-step):min(H,idx[1]+step)
                if i == idx[1] && j == idx[2]
                    continue
                end
                push!(vv, disp_[i,j])
            end
        end
        sort!(vv)
        disp_[idx] = vv[floor(Int, (length(vv) +1) /2)]
        cnt += 1
    end
    @show cnt

    return cnt
end


function get_IF(F, T, leftT = 100)
    F[F .< 0.01*maximum(F)] .= 0
    h = [zeros(leftT,1); F; zeros(leftT,1)]
    h ./= sum(h)

    maxF = argmax(h)[1]

    IF = zeros(length(h), length(h))
    Threads.@threads for i=1:length(h)
        IF[:,i] = circshift(h, [i-maxF 0])
    end
    IF = IF[1:T, 1:T]

    return IF
end

function shift_h(F, T, sigma=0)
    # (optional) apply Gaussian smoothing on IRF
    if sigma > 0
        ker = ImageFiltering.Kernel.gaussian((sigma,))
        F = imfilter(F, ker)
    end
    h = zeros(T, 1)
    h[1:min(length(F), T), :] = F
    h ./= sum(h)
    maxF = argmax(h)[1]

    h1 = circshift(h, -maxF)
    h1 = h1[end:-1:1]
    h1 = reshape(h1, 1, :)
    # Hf = reshape(fft(h1), 1, :)

    attack = findfirst(x -> x < 1e-5, h[maxF:-1:1]) - 2
    trailing = findfirst(x -> x < 1e-5, h[maxF:end]) - 2
    return h1, attack, trailing, nothing
end


# function get_Hf(F, T, sigma=0)
#     # (optional) apply Gaussian smoothing on IRF
#     if sigma > 0
#         ker = ImageFiltering.Kernel.gaussian((sigma,))
#         F = imfilter(F, ker)
#     end
#     h = zeros(T, 1)
#     h[1:min(length(F), T), :] = F
#     h ./= sum(h)
#     maxF = argmax(h)[1]

#     h1 = circshift(h, -maxF)
#     h1 = h1[end:-1:1]
#     Hf = reshape(fft(h1), 1, :)

#     attack = findfirst(x -> x < 1e-5, h[maxF:-1:1]) - 2
#     trailing = findfirst(x -> x < 1e-5, h[maxF:end]) - 2
#     return Hf, attack, trailing
# end


"""
    bar(x[, y])

Generate the histogram from the normalized depth and intensity maps.

# Arguments
- depth : normalized depth map
- intensity : normalized intensity map
- ppp : simulation parameter of the total photons per pixel
- sbr : signal-to-background noise ratio
- nosise_type ∈ {nothing, constant}

https://github.com/HWQuantum/HistNet/blob/77617d9993f7e2c6fe7d94bd49e9eb7e2ac2051e/Codes_network/ops_dataset.py
"""
function gen_hist(depth, intensity, ppp, T=15)
    sigma = 0.5174
    depth = (depth .- minimum(depth)) ./ (maximum(depth) - minimum(depth))
    H, W = size(depth)
    precision = 100 #precision per bin 
    array_x = -16 .+ LinRange(0, 32*precision, 32*precision) ./ precision
    array_exp = 1.0 ./ (sqrt(2*pi)*sigma) * exp.(-array_x .^ 2 / sigma.^2 )
    array_bin = zeros(length(array_exp))
    average_intensity = mean(intensity)

    for i = 1:(length(array_exp) - precision)
        array_bin[i] = sum(array_exp[i:i+precision-1])
    end

    tof = zeros(H, W, T)
    for n in CartesianIndices((H,W))
        index_array = precision * (16 - T*depth[n] )
        for index_bin = 1:T
            index_array_bin = index_array + precision * index_bin
            tof[n, index_bin] = array_bin[Int(floor(index_array_bin))+1]
        end
        tof[n,:] .*= intensity[n] * ppp / average_intensity / sum(tof[n,:])
    end

    @show "estimated ppp:,", mean(sum(tof, dims=3))
    return tof
end

function impose_noise(tof, SBR::Real, noise_type="constant")
    H, W, T = size(tof)
    b = zeros(H, W, T)
    if noise_type == "constant" # constant
        for n in CartesianIndices((H,W))
            b_val = sum(tof[n,:]) ./ (3.0*SBR)
            b[n,:] .= b_val
        end
    else

    end

    tof_ambient = tof + b
    tof_noisy = rand.(Distributions.Poisson.(tof_ambient))
    # plot(plot(tof[100,100,:]), plot(tof_noisy[100,100,:], title="SBR=$SBR"))

    return tof_noisy
end
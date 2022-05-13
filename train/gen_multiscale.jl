using ImageFiltering
using FourierTools

function median_filtering(dd, idxs, T, K=3, trailing=10)
    szwindow = Int(floor(K / 2))
    H, W = size(dd)
    out = dd
    for n in idxs
        vals = dd[max(1,n[1]-szwindow):min(H,n[1]+szwindow), max(1,n[2]-szwindow):min(W,n[2]+szwindow)]
        idx_valid = findall((vals .>= trailing) .* (vals .<= T-trailing))
        if isempty(idx_valid) || isempty(vals)
            @info "isnothing(idx_valid) $n $vals"
        else
            out[n] = Int(floor(median(vals[idx_valid])))
        end
        # d_n[n] = Int(floor(median(vals[idxzeros])))
    end
    return out 
end

# smoothing with the guidance of reflectivity
function remove_outliers_from_intensity(dd, rr, th_ratio=0.5, szwindow=11, usemedian=true, debug=false)
    threshold = mean(rr) * th_ratio
    @show threshold
    idx_low = findall(rr .< threshold)
    H, W = size(dd)
    for idx in idx_low
        vv = []
        for j=max(1,idx[2]-szwindow):min(W,idx[2]+szwindow)
            for i=max(1,idx[1]-szwindow):min(H,idx[1]+szwindow)
                if usemedian
                    if i == idx[1] && j == idx[2]
                        continue
                    end
                end
                push!(vv, dd[i,j])
            end
        end
        if usemedian
            dd[idx] = median(vv)
        else
            dd[idx] = mean(vv)
        end
    end 
    if debug
        img_th = rr .< threshold
        return dd, img_th
    else
        return dd
    end
end

function gen_multiscale(tof, nscale=4, KK = [1 3 7 11], trailing=1, attack=1; border="replicate", save_tof=false)
    H, W, T = size(tof)
    
    depths = zeros(Float32, H, W, nscale)
    refls = zeros(Float32, H, W, nscale)
    
    Z_nscale = copy(tof)
    if save_tof
        Z_nscale3 = copy(tof)
    end

    # spatial convolution [1 3 7 11 15 25 31]
    @info "construct multiscale tof"
    for s=1:nscale
        # step 1: downsample
        if s > 1
            K = KK[s]
            # kern = kernelfactors( ( ones(K,1,1)./K, ones(1,K,1)./K, ones(1,1,1) ) )

            # kern = kernelfactors( ( ones(K,1,1)./K, ones(1,K,1)./K, ones(1,1,1) ) )
            # Z_nscale = imfilter(tof, ones(K,1,1)./K, Fill(0.0))
            Z_nscale = imfilter(tof, ones(K,1,1)./K, border)
            Z_nscale = imfilter(Z_nscale, ones(1,K,1)./K, border)
            # Z_nscale = imfilter(tof, kern)
        end

        # step 2: compute depth
        Z_resh = reshape(Z_nscale, H*W, T)
        dd = reshape(argmax(Z_resh, dims=2), H, W)
        d_n = map(x -> x[2], dd)

        # step 3: compute intensity
        for n=CartesianIndices((H, W))
            t_range = max.(1, d_n[n] - trailing : min(T, d_n[n]+attack) )
            refls[n[1],n[2],s] = sum(Z_nscale[n, t_range])
        end

        depths[:,:,s] = d_n
        if save_tof && s == nscale-1
            Z_nscale3 = copy(Z_nscale)
        end
    end

    # if usemedian
    #     @info "apply median filtering for the last depth guided by the reflectivity"
    #     depths[:,:,nscale] = remove_outliers_from_intensity(depths[:,:,nscale], refls[:,:,nscale], 0.5, KK[end])
    # end

    if save_tof
        return depths, refls, Z_nscale3, Z_nscale
    else
        return depths, refls, nothing, nothing
    end
end


# function gen_multiscale3d(tof, nscale, h1, dilation=1)
#     H, W, T = size(tof)
    
#     depths = zeros(Float32, H, W, nscale)
#     refls = zeros(Float32, H, W, nscale)
    
#     K = 3
#     Z_nscale = copy(tof)
#     h2 = copy(h1)

#     # spatial convolution [1 3 7 11 15 25 31]
#     for s=1:nscale
#         @info "construct multiscale s:$s"

#         if s >= 2
#             K = 2*s-1
#             # kern = kernelfactors( ( ones(K,1,1)./K, ones(1,K,1)./K, ones(1,1,1) ) )

#             if dilation == 1
#                 kern = kernelfactors( ( ones(K,1,1)./K, ones(1,K,1)./K, ones(1,1,1) ) )
#                 Z_nscale = imfilter(tof, kern, Fill(0.0))
#             else
#                 K = K*2-1
#                 kernel1 = zeros(K,1,1)
#                 kernel1[1:2:end,:,:] .= 1.0
#                 kernel1 ./= sum(kernel1)
                
#                 kernel2 = zeros(1,K,1)
#                 kernel2[:,1:2:end,:] .= 1.0
#                 kernel2 ./= sum(kernel2)

#                 kern = kernelfactors( ( kernel1, kernel2, ones(1,1,1) ) )
#                 Z_nscale = imfilter(tof, kern, Fill(0.0))
#             end
#             h2 = imfilter(h1, ones(1, K) ./ K, "circular")
#         end

#         # Z_nscale = imfilter(Z_nscale, h2, "circular")
#         attack = findfirst(x -> x < 1e-5, h2[end:-1:1]) - 2
#         trailing = findfirst(x -> x < 1e-5, h2[1:end]) - 1
        
#         # step 2: compute depth
#         Z_resh = reshape(Z_nscale, H*W, T)
#         # Z_resh = imfilter(Z_resh, h2, "circular")
#         Z_resh = FourierTools.conv(h2, Z_resh, [2]);
#         dd = reshape(argmax(Z_resh, dims=2), H, W)
#         d_n = map(x -> x[2], dd)

#         # step 3: compute intensity
#         for n=CartesianIndices((H, W))
#             t_range = max.(1, d_n[n] - trailing : min(T, d_n[n]+attack) )
#             refls[n[1],n[2],s] = sum(Z_nscale[n, t_range])
#         end

#         depths[:,:,s] = d_n
#     end

#     return depths, refls
# end

# use Flux for conv
# tof_HxWx1xB = Float64.(reshape(tof, size(tof,1), size(tof,2), 1, :))
# w = ones(K,K, 1, T) ./ K^2
# Flux.conv(tof_HxWx1xB, w)


function plot_hist(tof_data, tof_conv, reflg)
    plot(tof_conv[1, 1, :], linewidth=2, label="1x1")
    plot!(tof_data[1, 1, :]*0.2, linewidth=2, label="1x1 ori", linestyle=:dot)
    # plot!(tof_conv[100, 100, :], linewidth=2, label="100x100")
    # plot!(tof_data[100,100,:]*0.2, label="100x100 ori", linestyle=:dot)
    plot!(tof_conv[100, 100, :],linewidth=2, label="200x200")
    p1 = plot!(tof_data[100,100,:]*0.2, label="200x200 ori", linestyle=:dot)
    plot(tof_conv[200, 200, :],linewidth=2, label="300x300")
    plot!(tof_data[200,200,:]*0.2, label="300x300 ori", linestyle=:dot)
    # plot!(tof_conv[400, 400, :], linewidth=2,label="400x400")
    # plot!(tof_data[400,400, :]*0.2, label="400x400 ori", linestyle=:dot)
    plot!(tof_conv[300, 300, :], linewidth=2,label="500x500")
    p2 = plot!(tof_data[300, 300, :]*0.2, linewidth=2,label="500x500 ori", linestyle=:dot)
    plot(p1, p2, heatmap(reflg, yflip=true), size=(1000,1000), layout=@layout [a b ; c])
end


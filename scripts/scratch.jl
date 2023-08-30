
using ReverseDiff, Zygote
using ADTypes
using Optimisers

using Accessors
using Base.Iterators
using CUDA
using Distributions
using DistributionsAD
using Flux
using FillArrays
using LogDensityProblems
using LinearAlgebra
using Plots
using Random123
using SimpleUnPack
using StatsFuns
using Optimisers

using AdvancedVI
using HierarchicalBayesZoo

function construct_variational(model)
    n_dims_global = local_dimension(model)
    n_dims_local  = global_dimension(model)
    n_dims        = n_dims_global+n_dims_local
    q             = VILocationScale(
        zeros(Float32, n_dims),
        Diagonal(ones(Float32, n_dims)),
        Normal()
    )
    AmortizedLocationScale(q, n_dims_global, n_dims_local, Int[])
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    CUDA.allowscalar(false)

    use_cuda = false

    #n_obs      = 100
    #n_dims     = 2
    prob      = Volatility()

    n_samples = 10
    advi      = ADVICUDA(prob, n_samples, use_cuda)

    #batchsize = 10
    #advidoubly = Subsampling(advi, batchsize, data_indices)

    d     = LogDensityProblems.dimension(prob)
    q     = VIMeanFieldGaussian(zeros(Float32, d), Diagonal(ones(Float32, d)))
    #q     = VIFullRankGaussian(zeros(Float32, d), Eye{Float32}(d) |> Matrix |> LowerTriangular)
        #construct_variational(prob)
    λ, re = Optimisers.destructure(q)

    n_max_iter = 2*10^4
    q, stats, _ = optimize(
        #advidoubly,
        advi,
        q,
        n_max_iter;
        #callback! = callback!,
        rng       = rng,
        adbackend = ADTypes.AutoZygote(),
        optimizer = Optimisers.Adam(1f-2)
    )
    elbo = [stat.elbo for stat ∈ stats]
    plot(elbo, ylims=quantile(elbo, (0.1, 1.))) |> display
    q
end

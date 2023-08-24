
# using DrWatson
# @quickactivate "HierarchicalBayesZoo"
# include(srcdir("HierarchicalBayesZoo.jl"))

using ReverseDiff, Zygote
using ADTypes
using Optimisers

using Accessors
using Base.Iterators
using CUDA
using Distributions
using DistributionsAD
using Flux
using LogDensityProblems
using LinearAlgebra
using Plots
using Random123
using SimpleUnPack
using StatsFuns

using AdvancedVI
using HierarchicalBayesZoo

struct BlockGaussians{M <: AbstractVector,
                      S <: AbstractVector,
                      I <: AbstractVector{<:Integer}}
    μs::Vector{M}
    Σs::Vector{S}
    block_idx::Vector{I}
    batch_idx::Vector{Int}
end

function RandomBlockGaussians(rng_host, n_blocks, blocksize, use_cuda)
    rng = use_cuda ? CUDA.default_rng() : rng_host

    μs = map(1:n_blocks) do _
        randn(rng, Float32, blocksize)
    end
    Σs = map(1:n_blocks) do _
        Flux.softplus.(randn(rng, Float32, blocksize) .- 2)
    end
    data_idx  = Int32.(1:n_blocks*blocksize)
    block_idx = if use_cuda
        map(idxs -> idxs |> collect |> Flux.gpu, partition(data_idx, blocksize)) 
    else
        map(collect, partition(data_idx, blocksize)) 
    end
    BlockGaussians(μs, Σs, block_idx, collect(1:n_blocks)), block_idx
end

function LogDensityProblems.logdensity(model::BlockGaussians, θ)
    @unpack μs, Σs, block_idx, batch_idx = model
    N = length(block_idx)
    M = length(batch_idx)

    N/M*sum(batch_idx) do idx
        logpdf(TuringDiagMvNormal(μs[idx], sqrt.(Σs[idx])), θ[block_idx[idx]])
    end
end

function LogDensityProblems.dimension(model::BlockGaussians)
    CUDA.@allowscalar model.block_idx |> last |> last
end

function LogDensityProblems.capabilities(::Type{<:BlockGaussians})
    LogDensityProblems.LogDensityOrder{0}()
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)

    CUDA.allowscalar(false)

    use_cuda  = true
    n_blocks  = 5
    blocksize = 1000

    prob, block_idx = RandomBlockGaussians(rng, n_blocks, blocksize, use_cuda)

    update_batch = (prob, batch) -> begin
        @set prob.batch_idx = collect(batch)
    end
    amortize = (q, batch) -> q

    batchsize    = 1 #n_blocks
    n_samples    = 10
    data_indices = 1:n_blocks
    
    obj = DoublyADVI(prob,
                     n_samples;
                     batchsize    = batchsize,
                     update_batch = update_batch,
                     amortize     = amortize,
                     data_indices = data_indices,
                     use_cuda     = use_cuda)

    d    = LogDensityProblems.dimension(prob)
    μ, L = if use_cuda
        CUDA.zeros(Float32, d), Diagonal(CUDA.ones(Float32, d))
    else
        zeros(Float32, d), Diagonal(ones(Float32, d))
    end
    q = VIMeanFieldGaussian(μ, L)

    callback!(; stat, restructure, λ, g) = begin
        @assert eltype(λ) == Float32
        @assert eltype(g) == Float32
        q   = restructure(λ)
        μ   = mean(q)
        Σ   = var(q)

        Δμ² = sum(1:n_blocks) do idx
            sum(abs2, prob.μs[idx] - μ[block_idx[idx]])
        end
        ΔΣ² = sum(1:n_blocks) do idx
            sum(abs2, prob.Σs[idx] - diag(Σ)[block_idx[idx]])
        end
        (wass2 = sqrt(Δμ² + ΔΣ²),)
    end
    
    n_max_iter = 10^3
    q, stats, _ = optimize(
        obj,
        q,
        n_max_iter;
        callback! = callback!,
        rng       = rng,
        adbackend = ADTypes.AutoZygote(),
        optimizer = Optimisers.Adam(1f-3)
    )
    plot!([stat.wass2 for stat ∈ stats])
end

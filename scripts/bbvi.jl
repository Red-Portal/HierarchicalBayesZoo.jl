
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

struct BlockGaussians{
    V <: AbstractVector{<:Real},
    I <: AbstractVector{<:Integer},
}
    μ        ::V
    Σ        ::V
    μs       ::Vector{V}
    Σs       ::Vector{V}
    block_idx::Vector{I}
    batch_idx::Vector{Int}
end

function RandomBlockGaussians(rng_host, n_blocks, blocksize, use_cuda)
    rng = use_cuda ? CUDA.default_rng() : rng_host

    μ = randn(rng, Float32, blocksize)
    Σ = Flux.softplus.(randn(rng, Float32, blocksize))

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
    BlockGaussians(
        μ, Σ, μs, Σs, block_idx, collect(1:n_blocks)
    ), block_idx
end

function LogDensityProblems.logdensity(model::BlockGaussians, θ)
    @unpack μ, Σ, μs, Σs, block_idx, batch_idx = model
    N = length(block_idx)
    M = length(batch_idx)
    d = length(first(μs))

    ℓprior = logpdf(TuringDiagMvNormal(μ, sqrt.(Σ)), θ[1:d])
    ℓlike  = N/M*sum(enumerate(batch_idx)) do (θ_idx, p_idx)
        idx_range = θ_idx*d+1:(θ_idx+1)*d
        logpdf(TuringDiagMvNormal(μs[p_idx], sqrt.(Σs[p_idx])), θ[idx_range])
    end
    ℓprior + ℓlike
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
    n_blocks  = 10
    blocksize = 1000

    update_batch = (prob, batch) -> begin
        @set prob.batch_idx = collect(batch)
    end

    batchsize    = 5 #n_blocks
    n_samples    = 10
    data_indices = 1:n_blocks

    prob, block_idx = RandomBlockGaussians(rng, n_blocks, blocksize, use_cuda)

    advi       = ADVICUDA(prob, n_samples, use_cuda)
    advidoubly = Subsampling(advi, batchsize, update_batch, data_indices)

    q = if use_cuda
        IsoStructuredLocationScale(
            CUDA.zeros(Float32, blocksize), CUDA.zeros(Float32, blocksize), 1f0, n_blocks
        )
    else
        IsoStructuredLocationScale(
            zeros(Float32, blocksize), zeros(Float32, blocksize), 1f0, n_blocks
        )
    end

    λ, re = Optimisers.destructure(q)

    callback!(; stat, restructure, λ, g) = begin
        @assert eltype(λ) == Float32
        @assert eltype(g) == Float32
        q   = restructure(λ)

        Δμ²_loc = sum(1:n_blocks) do idx
            μᵢ = q.m_locals[idx]
            sum(abs2, prob.μs[idx] - μᵢ)
        end

        Δμ²_glo = begin
            μ = q.m_global
            sum(abs2, prob.μ - μ)
        end

        Δμ² = Δμ²_glo + Δμ²_loc

        # ΔΣ² = sum(1:n_blocks) do idx
        #     qᵢ = amortize(q, [idx])
        #     Σᵢ = cov(qᵢ)
        #     sum(abs2, Diagonal(prob.Σs[idx]) - Σᵢ)
        # end
        #(wass2 = sqrt(Δμ² + ΔΣ²),)
        (wass2 = sqrt(Δμ²),)
    end
    
    n_max_iter = 10^4
    q, stats, _ = optimize(
        advidoubly,
        q,
        n_max_iter;
        callback! = callback!,
        rng       = rng,
        adbackend = ADTypes.AutoZygote(),
        optimizer = Optimisers.Adam(1f-3)
    )
    plot!([stat.wass2 for stat ∈ stats])
    #plot!([stat.elbo for stat ∈ stats])
end

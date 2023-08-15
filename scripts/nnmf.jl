
using DrWatson

using Distributions
using CUDA
using SimpleUnPack: @unpack
using SparseArrays
using SpecialFunctions
using BenchmarkTools
using ProgressMeter
using Bijectors
using StatsFuns
using LinearAlgebra
using Zygote
using Flux, Functors

using Plots

include(srcdir("bijectors.jl"))

struct NNMFDirLap{F       <: Real,
                  IntMat  <: AbstractMatrix{<:Integer},
                  RealVec <: AbstractVector{F}}
    α       ::F
    λ₀      ::F
    y       ::IntMat
    K       ::Int
    I       ::Int
    U       ::Int
    b_β     ::ExpBijector
    b_θ     ::SimplexBijector{RealVec}
    likeadj ::F
end

@functor NNMFDirLap

function logdensity(model::NNMFDirLap, λ_β, λ_θ)
    # λ_θ ∈ ℝ^{(K-1) × U}
    # λ_β ∈ ℝ^{I × K}
    #
    # θ ∈ S^{K × U}
    # β ∈ ℝ₊^{I × K}

    @unpack α, λ₀, y, K, I, U, b_β, b_θ, likeadj = model

    β, ℓdetJ_β = forward(b_β, λ_β)
    θ, ℓdetJ_θ = forward(b_θ, λ_θ)

    ℓp_β = mapreduce(βᵢ -> logpdf(Exponential(λ₀), βᵢ), +, β)

    ℓBα  = sum(loggamma, α) - loggamma(K*α)
    ℓp_θ = sum(@. (α - 1)*log(θ)) - U*ℓBα

    λ    = β*θ
    ℓp_y = mapreduce((λᵢ, yᵢ) -> logpdf(Poisson(λᵢ), yᵢ), +, λ, y)

    likeadj*ℓp_y + ℓp_β + ℓp_θ + ℓdetJ_β + ℓdetJ_θ
end

function logdensity_ref(model::NNMFDirLap, λ_β, λ_θ)
    # λ_θ ∈ ℝ^{(K-1) × U}
    # λ_β ∈ ℝ^{I × K}
    #
    # θ ∈ S^{K × U}
    # β ∈ ℝ₊^{I × K}

    @unpack α, λ₀, y, K, I, U, likeadj = model

    p_θ = Dirichlet(K, α)
    p_β = Exponential(λ₀)

    b⁻¹_θ = p_θ |> bijector |> inverse
    b⁻¹_β = p_β |> bijector |> inverse

    β       = map(b⁻¹_β, λ_β)
    ℓdetJ_β = mapreduce(λ_βᵢ -> logabsdetjac(b⁻¹_β, λ_βᵢ), +, λ_β)

    yinit, linit = with_logabsdet_jacobian(b⁻¹_θ, eachcol(λ_θ) |> first)
    ℓdetJ_θ      = sum(linit)

    θ = mapreduce(hcat, eachcol(λ_θ)[2:end]; init=yinit) do λ_θᵢ
        θᵢ, ℓdetJ_θᵢ = with_logabsdet_jacobian(b⁻¹_θ, λ_θᵢ)
        ℓdetJ_θ     += sum(ℓdetJ_θᵢ)
        θᵢ
    end

    λ = β*θ

    ℓp_β = mapreduce(βᵢ -> logpdf(p_β, βᵢ), +, β)
    ℓp_θ = mapreduce(θᵤ -> logpdf(p_θ, θᵤ), +, eachcol(θ))
    ℓp_y = mapreduce((λᵢ, yᵢ) -> logpdf(Poisson(λᵢ), yᵢ), +, λ, y)

    likeadj*ℓp_y + ℓp_β + ℓp_θ + ℓdetJ_β + ℓdetJ_θ
end

function prior_predictive_check(U, I)
    α       = 1.0f0
    λ₀      = 1.0f0
    K       = 5
    likeadj = 1.0f0

    β = rand(Exponential(λ₀), I, K)
    θ = rand(Dirichlet(K, λ₀), U)
    λ = β*θ
    y = convert(Matrix{Int32}, @. rand(Poisson(λ)))

    λ_β = randn(Float32, I, K)
    λ_θ = randn(Float32, K-1, U)

    λ_β_dev = Flux.gpu(λ_β)
    λ_θ_dev = Flux.gpu(λ_θ)

    b_β = ExpBijector()
    b_θ = SimplexBijector(Float32, K)
    
    model     = NNMFDirLap(α, λ₀, y, K, I, U, b_β, b_θ, likeadj)
    model_dev = Flux.gpu(model)

    @assert(
        logdensity(model_dev, λ_β_dev, λ_θ_dev) ≈ logdensity_ref(model, λ_β, λ_θ),
        "$(logdensity(model_dev, λ_β_dev, λ_θ_dev)) $(logdensity_ref(model, λ_β, λ_θ))"
    )

    custom_host_t = @belapsed begin
        #logdensity($model, $λ_β, $λ_θ)
        Zygote.gradient($λ_β) do λ_β′
            logdensity($model, λ_β′, $λ_θ)
        end
    end
    custom_dev_t = @belapsed begin
        #logdensity($model_dev, $λ_β_dev, $λ_θ_dev)
        Zygote.gradient($λ_β_dev) do λ_β′
            logdensity($model_dev, λ_β′, $λ_θ_dev)
        end
    end
    reference_t = @belapsed begin
        #logdensity_ref($model, $λ_β, $λ_θ)
        Zygote.gradient($λ_β) do λ_β′
            logdensity_ref($model, λ_β′, $λ_θ)
        end
    end
    custom_host_t, custom_dev_t, reference_t
end

function benchmark()
    U     = 512
    Is    = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    stats = @showprogress map(Is) do I
        prior_predictive_check(U, I)
    end
    Plots.plot( Is, [stat[1] for stat ∈ stats], label="Custom Host",   yscale=:log10, xscale=:log10) |> display
    Plots.plot!(Is, [stat[2] for stat ∈ stats], label="Custom Device", yscale=:log10, xscale=:log10) |> display
    Plots.plot!(Is, [stat[3] for stat ∈ stats], label="Reference",     yscale=:log10, xscale=:log10) |> display
end

function movielens_dataset()
    # y[1] is the user
    # y[2] is the item
    # y[3] is the rating

    y_entries = readdlm(datadir("datasets", "movielens-100k", "u.data"), Int)[:,1:3]
    #I = 1682
    #U = 943
    sparse(y_entries[:,2], y_entries[:,1], y_entries[:,3])
end


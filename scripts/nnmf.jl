
using DrWatson

using SparseArrays
using BenchmarkTools
using ProgressMeter
using Zygote
using Flux
using Plots

include(srcdir("HierarchicalBayesZoo.jl"))

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

    model     = NNMFDirExp(α, λ₀, y, K, I, U, likeadj)
    model_dev = Flux.gpu(model)

    @assert(
        logdensity(model_dev, λ_β_dev, λ_θ_dev) ≈ logdensity_ref(model, λ_β, λ_θ),
        "$(logdensity(model_dev, λ_β_dev, λ_θ_dev)) $(logdensity_ref(model, λ_β, λ_θ))"
    )

    custom_host_t = @belapsed begin
        logdensity($model, $λ_β, $λ_θ)
    end
    custom_dev_t = @belapsed begin
        logdensity($model_dev, $λ_β_dev, $λ_θ_dev)
    end
    reference_t = @belapsed begin
        logdensity_ref($model, $λ_β, $λ_θ)
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


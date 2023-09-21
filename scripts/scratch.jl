
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
using PDMats
using Random123
using SimpleUnPack
using StatsFuns
using Optimisers, ParameterSchedulers

using AdvancedVI
using HierarchicalBayesZoo

struct Scheduler{T, F} <: Optimisers.AbstractRule
    constructor::F
    schedule::T
end

_get_opt(scheduler::Scheduler, t) = scheduler.constructor(scheduler.schedule(t))

Optimisers.init(o::Scheduler, x::AbstractArray) =
    (t = 1, opt = Optimisers.init(_get_opt(o, 1), x))

function Optimisers.apply!(o::Scheduler, state, x, dx)
    opt = _get_opt(o, state.t)
    new_state, new_dx = Optimisers.apply!(opt, state.opt, x, dx)

    return (t = state.t + 1, opt = new_state), new_dx
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    CUDA.allowscalar(false)

    use_cuda = true
    dev      = use_cuda ? Flux.gpu : Flux.cpu

    prob      = problem(CritLangIRT(0.003)) |> dev
    #prob      = problem(BSSNNMF(0.1)) |> dev
    #prob      = problem(ForeignExchangeVolatility(0.1)) |> dev

    # d    = 10
    # L    = 0.1f0*tril(randn(Float32, d, d))
    # prob = Gaussian(randn(Float32, d), PDMats.PDMat(I + L*L'))

    n_samples = 10
    obj       = ADVICUDA(prob, n_samples, use_cuda)

    #batchsize = 10
    #obj       = Subsampling(obj, batchsize, 1:prob.U)

    @info("", d = LogDensityProblems.dimension(prob))

    #q = HierarchicalBayesZoo.StructuredGaussian(prob) |> dev
    #q = AdvancedVI.VIMeanFieldGaussian(prob) |> dev
    q = AdvancedVI.VIFullRankGaussian(prob) |> dev

    #q     = VIMeanFieldGaussian(zeros(Float32, d), Diagonal(.1f0*ones(Float32, d)))
    #q     = VIFullRankGaussian(zeros(Float32, d), Eye{Float32}(d) |> Matrix |> LowerTriangular)
    #q     = VIMeanFieldGaussian(CUDA.zeros(Float32, d), Diagonal(.1f0*CUDA.ones(Float32, d)))
    #q     = VIFullRankGaussian(CUDA.zeros(Float32, d), 0.1f0*Eye{Float32}(d) |> Matrix |> Flux.gpu |> LowerTriangular)
    λ, re = Optimisers.destructure(q)

    optimizer = Scheduler(Step(λ=1f-2, γ=0.5f0, step_sizes=3*10^3)) do lr
        Optimisers.Adam(lr)
    end
    #optimizer = Optimisers.Adam(1f-2)

    callback!(; stat, g, λ, args...) = begin
        if any(@. isnan(λ) | isinf(λ))
            throw(ErrorException("NaN detected"))
        end
        nothing
    end

    n_max_iter = 10^4
    q, stats, _ = optimize(
        obj,
        q,
        n_max_iter;
        callback! = callback!,
        rng       = rng,
        adbackend = ADTypes.AutoZygote(),
        optimizer = optimizer
    )
    elbo = [stat.elbo for stat ∈ stats]
    plot!(elbo, ylims=quantile(elbo, (0.1, 1.))) |> display
    q
end

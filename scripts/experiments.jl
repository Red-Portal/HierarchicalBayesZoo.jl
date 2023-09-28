
using DrWatson
@quickactivate "HierarchicalBayesZoo"

using HierarchicalBayesZoo
using AdvancedVI

using ADTypes
using Accessors
using Base.Iterators
using CUDA
using DataFrames
using Flux
using Optimisers, ParameterSchedulers
using Plots
using ProgressMeter
using Random, Random123
using SimpleUnPack
using Zygote

function run(config, key)
    SimpleUnPack.@unpack taskname, proportion, familyname, logstepsize, maxiter, n_samples = config

    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, key)
    CUDA.allowscalar(false)

    seed_dev = rand(rng, UInt64)
    rng_dev  = CUDA.RNG(seed_dev)
    use_cuda = true
    dev      = use_cuda ? Flux.gpu : Flux.cpu

    prob = if taskname == :irt
        CritLangIRT(proportion)
    elseif taskname == :poisson
        GermanHealthRobustPoisson(proportion)
    elseif taskname == :volatility
        ForeignExchangeVolatility(proportion)
    end

    prob = problem(rng, prob) |> dev
    obj  = ADVICUDA(prob, n_samples, use_cuda) |> dev

    q = if familyname == :structured
        HierarchicalBayesZoo.StructuredGaussian(prob)
    elseif familyname == :meanfield
        AdvancedVI.VIMeanFieldGaussian(prob)
    elseif familyname == :fullrank
        AdvancedVI.VIFullRankGaussian(prob)
    end |> dev

    λ, re = Optimisers.destructure(q)

    optimizer = Optimisers.Adam(10.0^logstepsize)

    callback!(; stat, λ, args...) = begin
        if any(@. isnan(λ) | isinf(λ))
            throw(ErrorException("NaN detected"))
        end

        if mod(stat.iteration, 100) == 1
            obj′ = @set obj.n_samples = 1024
            (elbo_true=obj′(rng_dev, prob, re(λ)),)
        else
            nothing
        end
    end

    _, stats, _ = optimize(
        obj,
        q,
        maxiter;
        callback!     = callback!,
        rng           = rng,
        adbackend     = ADTypes.AutoZygote(),
        optimizer     = optimizer,
        show_progress = true,
    )
    iter = [stat.iteration for stat ∈ filter(x -> haskey(x,:elbo_true), stats)]
    elbo = [stat.elbo_true for stat ∈ filter(x -> haskey(x,:elbo_true), stats)]
    DataFrame(t = iter, elbo = elbo)
end

function main()
    n_trials = 1

    tasks     = [
        (taskname = :irt,        proportion = 0.005),
        (taskname = :poisson,    proportion = 0.1),
        (taskname = :volatility, proportion = 0.1)
    ]

    families = [
        (familyname = :structured,),
        (familyname = :meanfield,),
        (familyname = :fullrank,),
    ]

    logstepsizes = [(logstepsize = -3,),]
    configs      = [(maxiter = 5*10^4, n_samples = 8),]

    configs = Iterators.product(tasks, families, logstepsizes, configs) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    for config ∈ configs
        DrWatson.produce_or_load(datadir("experiment"), config) do _
            dfs = map(1:n_trials) do key
                run(config, key)
            end
            df = vcat(dfs...)
            for (k, v) ∈ pairs(config)
                df[:,k] .= v
            end
            GC.gc()
            Dict(:data => df, :config => config)
        end
    end
end

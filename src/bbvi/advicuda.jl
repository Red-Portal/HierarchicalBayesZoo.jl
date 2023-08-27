
struct ADVICUDA{P} <: AdvancedVI.AbstractVariationalObjective
    prob     ::P
    n_samples::Int
    use_cuda ::Bool

    function ADVICUDA(prob, n_samples::Int, use_cuda ::Bool = false)
        cap = LogDensityProblems.capabilities(prob)
        if cap === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        new{typeof(prob)}(prob, n_samples, use_cuda)
    end
end

function AdvancedVI.init(rng::Random.AbstractRNG, advi::ADVICUDA, λ, re)
    if advi.use_cuda
        seed = rand(rng, UInt32)
        CUDA.RNG(seed)
    else
        rng
    end
end

function (advi::ADVICUDA)(
    rng::Random.AbstractRNG, prob_batch, q,
)
    n_samples = advi.n_samples
    ηs        = rand(rng, q, n_samples)
    𝔼ℓ = sum(eachcol(ηs)) do ηᵢ
        LogDensityProblems.logdensity(prob_batch, ηᵢ)
    end / n_samples
    ℍ  = entropy(q)
    𝔼ℓ + ℍ
end

function AdvancedVI.estimate_gradient(
    rng         ::Random.AbstractRNG,
    adbackend   ::ADTypes.AbstractADType,
    advi        ::ADVICUDA,
    rng_mc,
    λ           ::AbstractVector{<:Real},
    restructure,
    out         ::DiffResults.MutableDiffResult,
    batch_idx = nothing
)
    f(λ′) = begin
        q_η_x = amortize(restructure(λ′), batch_idx)
        -advi(rng_mc, advi.prob, q_η_x)
    end
    AdvancedVI.value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)
    out, rng_mc, stat
end

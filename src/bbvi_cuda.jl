
struct DoublyADVI{P, B, A} <: AdvancedVI.AbstractVariationalObjective
    prob        ::P
    n_samples   ::Int
    batchsize   ::Int
    update_batch::B
    amortize    ::A
    data_indices::Vector{Int}
    use_cuda    ::Bool

    function DoublyADVI(
        prob,
        n_samples   ::Int;
        batchsize   ::Int                   = 1,
        update_batch                        = (prob, batch) -> prob,
        amortize                            = (   q, batch) -> q,
        data_indices::AbstractVector{<:Int} = 1,
        use_cuda    ::Bool                  = false
    )
        cap = LogDensityProblems.capabilities(prob)
        if cap === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        new{typeof(prob), typeof(update_batch), typeof(amortize)}(
            prob, n_samples, batchsize, update_batch, amortize, data_indices, use_cuda
        )
    end
end

function StatsBase.entropy(q::VILocationScale{L, <:Diagonal, D}) where {L, D}
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*convert(eltype(location), entropy(dist)) + sum(log, diag(scale))
end

function init_batch(rng::Random.AbstractRNG, indices::Vector{Int}, batchsize::Int)
    Iterators.partition(Random.shuffle(rng, indices), batchsize)
end

function AdvancedVI.init(rng::Random.AbstractRNG, advi::DoublyADVI, λ, re)
    iter = init_batch(rng, advi.data_indices, advi.batchsize)
    if advi.use_cuda
        seed = rand(rng, UInt32)
        (iter, CUDA.RNG(seed))
    else
        iter
    end
end

function (advi::DoublyADVI)(
    rng::Random.AbstractRNG,
    prob_batch,
    q::VILocationScale,
)
    @unpack  location, scale = q

    n_dims    = length(location)
    n_samples = advi.n_samples

    us = randn(rng, eltype(q), n_dims, n_samples)
    ηs = scale*us .+ location
    𝔼ℓ = sum(eachcol(ηs)) do ηᵢ
        LogDensityProblems.logdensity(prob_batch, ηᵢ)
    end / n_samples
    ℍ  = entropy(q)
    𝔼ℓ + ℍ
end

function update_objective_state(rng::AbstractRNG, advi::DoublyADVI, obj_state)
    batch_iter, rng_mc = if advi.use_cuda
        batch_iter, rng_cuda = obj_state
    else
        obj_state, rng   
    end

    batch_idx, batch_itr = Iterators.peel(batch_iter)

    if isempty(batch_itr)
        batch_itr = init_batch(rng, advi.data_indices, advi.batchsize)
    end

    obj_state′ = if advi.use_cuda
        (batch_itr, rng_cuda)
    else
        batch_itr
    end

    batch_idx, obj_state′, rng_mc
end

function AdvancedVI.estimate_gradient(
    rng::Random.AbstractRNG,
    adbackend::ADTypes.AbstractADType,
    advi::DoublyADVI,
    obj_state,
    λ::AbstractVector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult
)
    batch_idx, obj_state′, rng_mc = update_objective_state(rng, advi, obj_state)
    prob_batch = advi.update_batch(advi.prob, batch_idx)

    f(λ′) = begin
        q_η   = restructure(λ′)
        q_η_x = advi.amortize(q_η, batch_idx)
        -advi(rng_mc, prob_batch, q_η_x)
    end
    AdvancedVI.value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)
    out, obj_state′, stat
end

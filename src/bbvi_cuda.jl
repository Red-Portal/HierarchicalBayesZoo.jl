
struct DoublyADVI{P, B, A} <: AbstractVariationalObjective
    prob        ::P
    n_samples   ::Int
    batchsize   ::Int
    update_batch::N
    amortize    ::A
    data_indices::Vector{Int}
    use_cuda    ::Bool

    function DoublyADVI(prob,
                        n_samples   ::Int;
                        batchsize   ::Int                   = 1,
                        update_batch                        = (prob, batch) -> prob,
                        amortize                            = (   q, batch) -> q
                        data_indices::AbstractVector{<:Int} = 1,
                        use_cuda    ::Bool                  = false)
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

function init_batch(rng::AbstractRNG, indices::Vector{Int}, batchsize::Int)
    Iterators.partition(suffle(rng, indices), batchsize)
end

function init(rng::AbstractRNG, advi::DoublyADVI)
    iter = init_batch(rng, advi.data_indices, advi.batchsize)
    if advi.use_cuda
        seed = rand(rng, UInt32)
        (iter, CUDA.RNG(seed))
    else
        iter
    end
end

function (advi::DoublyADVI)(
    rng::AbstractRNG,
    prob_batch,
    q_η::LocationScale,
)
    @unpack  location, scale = q

    n_dims    = length(location)
    n_samples = advi.n_samples

    us = randn(rng, n_dims, n_samples)
    ηs = scale*us .+ location
    𝔼ℓ = sum(eachcol(ηs)) do ηᵢ
        logdensity(prob_batch, ηᵢ)
    end / n_samples
    ℍ  = entropy(q_η)
    𝔼ℓ + ℍ
end

function update_objective_state(rng::AbstractRNG, advi::DoublyADVI, obj_state)
    batch_iter, rng_mc = if advi.use_cuda
        batch_iter, rng_cuda = est_state
    else
        only(obj_state), rng   
    end

    batch_idx, batch_itr = Iterators.peel(batch_iter)

    if isempty(batch_iter)
        batch_iter = init_batch(rng, advi.data_indices, advi.batchsize)
    end

    obj_state′ = if advi.use_cuda
        (batch_iter, rng_cuda)
    else
        batch_iter
    end

    batch_idx, obj_state′, rng_cuda
end

function AdvancedVI.estimate_gradient(
    rng::AbstractRNG,
    adbackend::AbstractADType,
    advi::DoublyADVI,
    obj_state,
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult
)
    batch_idx, obj_state′, rng_cuda = update_objective_state(rng, advi, obj_state)
    prob_batch = update_batch(advi.prob, batch_idx)

    f(λ′) = begin
        q_η   = restructure(λ′)
        q_η_x = advi.amortize(q_η, batch_idx)
        -advi(rng_mc, prob_batch, q_η)
    end
    value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)
    out, est_state′, stat
end

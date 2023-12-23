
struct Subsampling{
    B <: Integer,
    O <: AdvancedVI.AbstractVariationalObjective,
    I <: AbstractVector{<:Integer}
} <: AdvancedVI.AbstractVariationalObjective
    batchsize::B
    objective::O
    indices  ::I
end

function init_batch(
    rng      ::Random.AbstractRNG,
    indices  ::AbstractVector{<:Integer},
    batchsize::Integer
)
    shuffled = Random.shuffle(rng, indices)
    batches  = Iterators.partition(shuffled, batchsize)
    enumerate(batches)
end

function AdvancedVI.init(
    rng::Random.AbstractRNG,
    sub::Subsampling,
    λ, re
)
    @unpack batchsize, objective, indices = sub
    epoch     = 1
    sub_state = (epoch, init_batch(rng, indices, batchsize))
    obj_state = AdvancedVI.init(rng, objective, λ, re)
    (sub_state, obj_state)
end

function update_subsampling(rng::Random.AbstractRNG, sub::Subsampling, sub_state)
    epoch, batch_itr         = sub_state
    (step, batch), batch_itr′ = Iterators.peel(batch_itr)
    epoch′, batch_itr′′        = if isempty(batch_itr′)
        epoch+1, init_batch(rng, sub.indices, sub.batchsize)
    else
        epoch, batch_itr′
    end
    logstat = (epoch = epoch, step = step)
    batch, (epoch′, batch_itr′′), logstat
end

function AdvancedVI.estimate_gradient(
    rng    ::Random.AbstractRNG,
    sub    ::Subsampling,
    ad     ::ADTypes.AbstractADType,
    out    ::DiffResults.MutableDiffResult,
    prob,
    λ,
    re,
    state,
)
    objective = sub.objective

    sub_state, obj_state = state
    batch, sub_state′, sub_logstat = update_subsampling(rng, sub, sub_state)

    prob_sub          = subsample_problem(objective.prob, batch)
    q_amort           = amortize(objective.prob, re(λ), batch)
    obj_sub           = @set objective.prob = prob_sub
    λ_amort, re_amort = Optimisers.destructure(q_amort)

    out, obj_state′, obj_logstat = AdvancedVI.estimate_gradient(
        rng, ad, obj_sub, obj_state, λ_amort, re_amort, out
    )
    out, (sub_state′, obj_state′), merge(sub_logstat, obj_logstat)
end

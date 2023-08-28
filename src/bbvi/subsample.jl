
struct Subsampling{
    O         <: AdvancedVI.AbstractVariationalObjective,
    VectorInt <: AbstractVector{<:Integer}
} <: AdvancedVI.AbstractVariationalObjective
    objective   ::O
    batchsize   ::Int
    data_indices::VectorInt
end

function init_batch(
    rng      ::Random.AbstractRNG,
    indices  ::AbstractVector{<:Integer},
    batchsize::Int
)
    shuffled = Random.shuffle(rng, indices)
    batches  = Iterators.partition(shuffled, batchsize)
    enumerate(batches)
end

function AdvancedVI.init(rng::Random.AbstractRNG, sub::Subsampling, λ, re)
    @unpack objective, batchsize, data_indices = sub
    epoch     = 1
    sub_state = (epoch, init_batch(rng, data_indices, batchsize))
    obj_state = AdvancedVI.init(rng, objective, λ, re)
    (sub_state, obj_state)
end

function update_subsampling(rng, sub::Subsampling, sub_state)
    epoch, batch_itr         = sub_state
    (step, batch), batch_itr′ = Iterators.peel(batch_itr)
    epoch′, batch_itr′′        = if isempty(batch_itr′)
        epoch+1, init_batch(rng, sub.data_indices, sub.batchsize)
    else
        epoch, batch_itr′
    end
    stat = (epoch = epoch, step = step)
    batch, (epoch′, batch_itr′′), stat
end

function AdvancedVI.estimate_gradient(
    rng    ::Random.AbstractRNG,
    ad     ::ADTypes.AbstractADType,
    sub    ::Subsampling,
    state,
    λ      ::AbstractVector{<:Real},
    re,
    out    ::DiffResults.MutableDiffResult
)
    objective = sub.objective

    sub_state, obj_state = state
    batch, sub_state′, sub_logstat = update_subsampling(rng, sub, sub_state)

    prob_sub = subsample_problem(objective.prob, batch)
    obj_sub  = @set objective.prob = prob_sub

    out, obj_state′, obj_logstat = AdvancedVI.estimate_gradient(
        rng, ad, obj_sub, obj_state, λ, re, out, batch
    )
    out, (sub_state′, obj_state′), merge(sub_logstat, obj_logstat)
end

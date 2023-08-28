
function StatsBase.entropy(q::VILocationScale{L, <:Diagonal, D}) where {L, D}
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*convert(eltype(location), entropy(dist)) + sum(log, diag(scale))
end

struct AmortizedLocationScale{
    VLS <: VILocationScale,
    VecInt <: AbstractVector{<:Integer}
}
    q             ::VLS
    n_dims_global ::Int
    n_dims_local  ::Int
    amortize_index::VecInt
end

function amortize(
    prob,
    q_x  ::AmortizedLocationScale,
    batch::AbstractVector{<:Integer}
)
    @unpack q, n_dims_global, n_dims_local = q_x
    AmortizedLocationScale(q, n_dims_global, n_dims_local, batch)
end

function StatsBase.entropy(q_x::AmortizedLocationScale) where {L, D}
    entropy(q_x.q)
end

function Distributions.rand(
    rng      ::AbstractRNG,
    q_x      ::AmortizedLocationScale{VILocationScale{L, <:Diagonal, D}},
    n_samples::Integer
)
    @unpack n_dims_global, n_dims_local, amortize_index = q_x
    zs = rand(rng, q_x.q, n_samples)

    z_global  = zs[1:n_n_dims_global, :]
    z_locals  = map(amortize_index) do i
        begin_idx = n_dims_global + (i-1)*n_locals + 1
        end_idx   = n_dims_global + i*n_locals
        zs[begin_idx:end_idx,:]
    end

    vcat(z_locals..., z_global)
end

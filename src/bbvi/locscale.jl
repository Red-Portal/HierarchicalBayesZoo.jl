
function StatsBase.entropy(q::VILocationScale{L, <:Diagonal, D}) where {L, D}
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*convert(eltype(location), entropy(dist)) + sum(log, diag(scale))
end

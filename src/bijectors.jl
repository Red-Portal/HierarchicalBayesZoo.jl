
struct SimplexBijector{Vec <: AbstractVector}
    K::Int
    offset::Vec
end

@functor SimplexBijector

function SimplexBijector(T::Type, K::Int)
    offset = -log.(convert(Vector{T}, range(K-1, 1; step=-1)))
    SimplexBijector(K, offset)
end

SimplexBijector(K::Int) = SimplexBijector(Float64, K)

function forward(b::SimplexBijector, y::AbstractMatrix{F}) where {F <: Real}
    # In : y ∈ ℝ^{(K-1) × N}
    # Out: x ∈ ℝ^{K × N}, where colums are vectors on the K-unit simplex
    N = size(y, 2)

    y_off = y .+ b.offset
    ℓz    = loglogistic.(y_off)
    ℓzm1  = log1mexp.(ℓz)

    # Cumulative product in log-pace
    # - One could use `cumprod` with padding but the ChainRule is not
    #   vectorized, so not GPU-friendly.
    ℓzm1_cumprd_part = cumsum(ℓzm1, dims=1)
    padding          = @ignore_derivatives KernelAbstractions.zeros(
        get_backend(y), F, 1, N)
    ℓzm1_cumprd      = vcat(padding, ℓzm1_cumprd_part[1:end-1,:])
    
    ℓx_1toKm1 = ℓz + ℓzm1_cumprd
    x_1toKm1  = exp.(ℓx_1toKm1)
    x_K       = exp.(log1mexp.(NNlib.logsumexp(ℓx_1toKm1, dims=1)))
    x         = vcat(x_1toKm1, x_K)

    ℓabsdetJ = sum( @. ℓx_1toKm1 + ℓz - y_off )
    x, ℓabsdetJ
end

struct ExpBijector end

function forward(::ExpBijector, y::AbstractArray{<:Real})
    x = exp.(y)
    x, sum(y)
end

struct VecToTril{VecInt <: AbstractVector{<:Integer}}
    d::Int
    chol_idx::VecInt
end

@functor VecToTril

function VecToTril(d::Int, k::Int = 0)
    idx_mat      = reshape(collect(1:d*d), (d,d))
    chol_idx_mat = tril(idx_mat, k)
    chol_idx_vec = chol_idx_mat[chol_idx_mat .!= 0]
    VecToTril(d, chol_idx_vec)
end

function _vectotril(
    d::Int, chol_idx::AbstractVector{<:Integer}, x::AbstractVector
)
    L = similar(x, d, d)
    fill!(L, zero(eltype(x)))
    L[chol_idx] = x
    L
end

function _triltovec(
    chol_idx::AbstractVector{<:Integer}, y::AbstractMatrix
)
    y[chol_idx]
end

function (vectotril::VecToTril)(x::AbstractVector)
    @unpack d, chol_idx = vectotril
    _vectotril(d, chol_idx, x)
end

@adjoint function (vectotril::VecToTril)(x::AbstractVector)
    @unpack d, chol_idx = vectotril
    L = _vectotril(d, chol_idx, x)
    L, Δ -> (nothing, reshape(Δ[chol_idx], length(x)),)
end

function triltovec(vectotril::VecToTril, x::AbstractMatrix)
    @unpack d, chol_idx = vectotril
    _triltovec(chol_idx, x)
end

@adjoint function triltovec(vectotril::VecToTril, y::AbstractMatrix)
    @unpack d, chol_idx = vectotril
    x = _triltovec(chol_idx, y)
    x, Δ -> (nothing, vectotril(Δ),)
end

struct CorrCholBijector{VecInt <: AbstractVector{<:Integer}}
    vectotril1::VecToTril{VecInt}
    vectotril2::VecToTril{VecInt}
end

@functor CorrCholBijector

function CorrCholBijector(d::Int)
    CorrCholBijector(VecToTril(d, -1), VecToTril(d, -2))
end

function forward(b::CorrCholBijector, y::AbstractVector{F}) where {F<:Real}
    @unpack vectotril1, vectotril2 = b

    ϵ    = eps(F)
    log2 = log(2*one(F))

    t = tanh.(y)
    r = vectotril1(t)

    ℓsqrt1mr²_flat = @. y + log2 - log1pexp(2*y)
    ℓsqrt1mr²      = vectotril1(ℓsqrt1mr²_flat)

    # Cumulative product in log-pace
    ℓsqrt1mr²_cumprd = cumsum(ℓsqrt1mr², dims=2)
    sqrt1mr²_cumprd  = exp.(ℓsqrt1mr²_cumprd)

    padding             = @ignore_derivatives KernelAbstractions.ones(
        get_backend(y), F, size(sqrt1mr²_cumprd,1))
    sqrt1mr²_cumprd_pad = hcat(padding, sqrt1mr²_cumprd[:,1:end-1])

    L_dense = ((r + I).*sqrt1mr²_cumprd_pad)
    L       = LowerTriangular(L_dense)
    
    z1m_cumprod           = 1 .- cumsum(L_dense.*L_dense, dims=2)
    z1m_cumprod_tril      = triltovec(vectotril2, z1m_cumprod)
    stick_breaking_logdet = sum(@. log(abs(z1m_cumprod_tril) + ϵ))/2
    tanh_logdet           = -2*sum(@. y + StatsFuns.softplus(-2*y) - log2)
    logabsdetjac          = stick_breaking_logdet + tanh_logdet
    L, logabsdetjac
end


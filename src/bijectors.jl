
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

function forward(b::SimplexBijector, y::Matrix{T}) where {T <: Real}
    # In : y ∈ ℝ^{(K-1) × N}
    # Out: x ∈ ℝ^{K × N}, where colums are vectors on the K-unit simplex

    K = size(y, 1) + 1
    N = size(y, 2)
    ϵ = eps(eltype(y)) 

    y_off = y .+ b.offset
    ℓz    = loglogistic.(y_off)
    ℓzm1  = log1mexp_cuda.(ℓz)

    # Mimicking Tensorflow's cumprod with the "exclusive" option
    ℓzm1_cumprd_part = cumsum(ℓzm1, dims=1)
    ℓzm1_cumprd      = vcat(zeros(eltype(ℓzm1), 1, N), ℓzm1_cumprd_part[1:end-1,:])

    ℓx_1toKm1 = ℓz + ℓzm1_cumprd
    x_1toKm1  = exp.(ℓx_1toKm1)
    x         = vcat(x_1toKm1, 1 .- sum(x_1toKm1, dims=1))

    # `mapreduce` would be more efficient but it currently doesn't work
    # with the CUDA+Zygote combination of doom.
    # See https://github.com/FluxML/Zygote.jl/issues/704
    ℓabsdetJ = sum( @. ℓx_1toKm1 + ℓz - y_off )
    x, ℓabsdetJ
end

function forward(b::SimplexBijector, y::CuMatrix{T}) where {T <: Real}
    # In : y ∈ ℝ^{(K-1) × N}
    # Out: x ∈ ℝ^{K × N}, where colums are vectors on the K-unit simplex

    K = size(y, 1) + 1
    N = size(y, 2)
    ϵ = eps(eltype(y)) 

    y_off = y .+ b.offset
    ℓz    = loglogistic.(y_off)
    ℓzm1  = log1mexp_cuda.(ℓz)

    # Cumulative product in log-pace
    # - One could use `cumprod` with padding but the ChainRule is not
    #   vectorized, so not GPU-friendly.
    ℓzm1_cumprd_part = cumsum(ℓzm1, dims=1)
    zeros_dev        = @ignore_derivatives CUDA.zeros(eltype(ℓzm1), 1, N)
    ℓzm1_cumprd      = vcat(zeros_dev, ℓzm1_cumprd_part[1:end-1,:])
    
    ℓx_1toKm1 = ℓz + ℓzm1_cumprd
    x_1toKm1  = exp.(ℓx_1toKm1)
    x         = vcat(x_1toKm1, 1 .- sum(x_1toKm1, dims=1))

    ℓabsdetJ = sum( @. ℓx_1toKm1 + ℓz - y_off )
    x, ℓabsdetJ
end

struct ExpBijector end

function forward(::ExpBijector, y::AbstractArray{<:Real})
    x = exp.(y)
    x, sum(y)
end

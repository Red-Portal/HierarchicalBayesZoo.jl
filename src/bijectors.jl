
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

    y_off      = y .+ b.offset
    z          = @. clamp(logistic(y_off), ϵ, 1 - ϵ)
    zm1        = 1 .- z

    # Mimicking Tensorflow's cumprod with the "exclusive" option
    zm1_pad        = vcat(ones(eltype(zm1), 1, N), zm1)
    zm1_pad_cumprd = cumprod(zm1_pad, dims=1)
    zm1_cumprd     = zm1_pad_cumprd[1:end-1,:]

    x_1toKm1 = z .* zm1_cumprd 
    x        = vcat(x_1toKm1, 1 .- sum(x_1toKm1, dims=1))

    ℓabsdetJ = sum( @. log(x_1toKm1) + log(z) - y_off )
    x, ℓabsdetJ
end

function forward(b::SimplexBijector, y::CuMatrix{T}) where {T <: Real}
    # In : y ∈ ℝ^{(K-1) × N}
    # Out: x ∈ ℝ^{K × N}, where colums are vectors on the K-unit simplex

    K = size(y, 1) + 1
    N = size(y, 2)
    ϵ = eps(eltype(y)) 

    y_off = y .+ b.offset
    z     = @. clamp(logistic(y_off), ϵ, 1 - ϵ)
    zm1   = 1 .- z

    # Mimicking Tensorflow's cumprod with the "exclusive" option
    zm1_pad        = vcat(CUDA.ones(eltype(zm1), 1, N), zm1)
    zm1_pad_cumprd = cumprod(zm1_pad, dims=1)
    zm1_cumprd     = zm1_pad_cumprd[1:end-1,:]

    x_1toKm1 = z .* zm1_cumprd 
    x        = vcat(x_1toKm1, 1 .- sum(x_1toKm1, dims=1))

    ℓabsdetJ = sum( @. log(x_1toKm1) + log(z) - y_off )
    x, ℓabsdetJ
end

struct ExpBijector end

function forward(::ExpBijector, y::AbstractArray{<:Real})
    x = exp.(y)
    x, sum(y)
end

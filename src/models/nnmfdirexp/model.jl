
struct NNMFDirExp{F       <: Real,
                  Mat     <: AbstractMatrix,
                  RealVec <: AbstractVector{F}}
    α       ::F
    λ₀      ::F
    y       ::Mat
    K       ::Int
    I       ::Int
    U       ::Int
    b_β     ::SimplexBijector{RealVec}
    b_θ     ::ExpBijector
    likeadj ::F
end

function NNMFDirExp(
    α::F, λ₀::F, y::Mat, K::Int, I::Int, U::Int, likeadj = one(F)
) where {F <: Real, Mat <: AbstractMatrix}
    b_θ = ExpBijector()
    b_β = SimplexBijector(F, I)
    NNMFDirExp(α, λ₀, y, K, I, U, b_β, b_θ, likeadj)
end

@functor NNMFDirExp

function logdensity(model::NNMFDirExp, z_β, z_θ)
    # z_β ∈ ℝ^{I × (K-1)}
    # z_θ ∈ ℝ^{K × U}
    #
    # β ∈ S(I)^{K}
    # θ ∈ ℝ₊^{K × U}

    @unpack α, λ₀, y, K, I, U, b_β, b_θ, likeadj = model
    @assert size(z_β) == (I-1, K)
    @assert size(z_θ) == (  K, U)

    ϵ = eps(eltype(z_θ))

    β, ℓdetJ_β = forward(b_β, @. clamp(z_β, -5, 5))
    θ, ℓdetJ_θ = forward(b_θ, z_θ)

    ℓBα  = sum(loggamma, α) - loggamma(I*α)
    ℓp_β = sum(@. (α - 1)*log(β + ϵ)) - K*ℓBα

    ℓp_θ = sum(θᵢ -> logpdf(Exponential(λ₀), θᵢ), θ)


    λ    = β*θ
    # `mapreduce` would be more efficient but it currently doesn't work
    # with the CUDA+Zygote combination of doom.
    # See https://github.com/FluxML/Zygote.jl/issues/704
    ℓp_y  = sum(@. poislogpdf(λ, y))
    likeadj*ℓp_y + ℓp_β + ℓp_θ + ℓdetJ_β + ℓdetJ_θ
end

function LogDensityProblems.capabilities(::Type{<: NNMFDirExp})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(model::NNMFDirExp)
    @unpack K, I, U = model
    K*(I-1) + K*U
end

function LogDensityProblems.logdensity(model::NNMFDirExp, λ_flat)
    @unpack K, I, U = model
    @assert length(λ_flat) == K*(I-1) + K*U
    z_β = reshape(first(λ_flat, (I-1)*K), (I-1, K))
    z_θ = reshape(last( λ_flat,     K*U), (  K, U))
    logdensity(model, z_β, z_θ)
end


struct NNMFDirExp{F       <: Real,
                  IntMat  <: AbstractMatrix{<:Integer},
                  RealVec <: AbstractVector{F}}
    α       ::F
    λ₀      ::F
    y       ::IntMat
    K       ::Int
    I       ::Int
    U       ::Int
    b_β     ::ExpBijector
    b_θ     ::SimplexBijector{RealVec}
    likeadj ::F
end

function NNMFDirExp(
    α::F, λ₀::F, y::IntMat, K::Int, I::Int, U::Int, likeadj = one(F)
) where {F <: Real, IntMat <: AbstractMatrix{<:Integer}}
    b_β = ExpBijector()
    b_θ = SimplexBijector(F, K)
    NNMFDirExp(α, λ₀, y, K, I, U, b_β, b_θ, likeadj)
end

@functor NNMFDirExp

function logdensity(model::NNMFDirExp, λ_β, λ_θ)
    # λ_θ ∈ ℝ^{(K-1) × U}
    # λ_β ∈ ℝ^{I × K}
    #
    # θ ∈ S^{K × U}
    # β ∈ ℝ₊^{I × K}

    @unpack α, λ₀, y, K, I, U, b_β, b_θ, likeadj = model
    @assert size(λ_β) == (I, K)
    @assert size(λ_θ) == ((K-1), U)

    β, ℓdetJ_β = forward(b_β, λ_β)
    θ, ℓdetJ_θ = forward(b_θ, λ_θ)

    ℓp_β = sum(βᵢ -> logpdf(Exponential(λ₀), βᵢ), β)

    ℓBα  = sum(loggamma, α) - loggamma(K*α)
    ℓp_θ = sum(@. (α - 1)*log(θ)) - U*ℓBα

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
    K*I + (K-1)*U
end

function LogDensityProblems.logdensity(model::NNMFDirExp, λ_flat)
    @unpack K, I, U = model
    @assert length(λ_flat) == K*I + (K-1)*U
    λ_β = reshape(first(λ_flat,     I*K), (  I, K))
    λ_θ = reshape(last( λ_flat, (K-1)*U), (K-1, U))
    logdensity(model, λ_β, λ_θ)
end

function logdensity_ref(model::NNMFDirExp, λ_β, λ_θ)
    # λ_θ ∈ ℝ^{(K-1) × U}
    # λ_β ∈ ℝ^{I × K}
    #
    # θ ∈ S^{K × U}
    # β ∈ ℝ₊^{I × K}

    @unpack α, λ₀, y, K, I, U, likeadj = model

    p_θ = Dirichlet(K, α)
    p_β = Exponential(λ₀)

    b⁻¹_θ = p_θ |> bijector |> inverse
    b⁻¹_β = p_β |> bijector |> inverse

    β       = map(b⁻¹_β, λ_β)
    ℓdetJ_β = mapreduce(λ_βᵢ -> logabsdetjac(b⁻¹_β, λ_βᵢ), +, λ_β)

    yinit, linit = with_logabsdet_jacobian(b⁻¹_θ, eachcol(λ_θ) |> first)
    ℓdetJ_θ      = sum(linit)

    θ = mapreduce(hcat, eachcol(λ_θ)[2:end]; init=yinit) do λ_θᵢ
        θᵢ, ℓdetJ_θᵢ = with_logabsdet_jacobian(b⁻¹_θ, λ_θᵢ)
        ℓdetJ_θ     += sum(ℓdetJ_θᵢ)
        θᵢ
    end

    λ = β*θ

    ℓp_β = mapreduce(βᵢ -> logpdf(p_β, βᵢ), +, β)
    ℓp_θ = mapreduce(θᵤ -> logpdf(p_θ, θᵤ), +, eachcol(θ))
    ℓp_y = mapreduce((λᵢ, yᵢ) -> logpdf(Poisson(λᵢ), yᵢ), +, λ, y)

    likeadj*ℓp_y + ℓp_β + ℓp_θ + ℓdetJ_β + ℓdetJ_θ
end

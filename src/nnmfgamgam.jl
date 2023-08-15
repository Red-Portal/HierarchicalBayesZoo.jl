
struct NNMFGamGam{F       <: Real,
                  IntMat  <: AbstractMatrix{<:Integer}}
    α_θ     ::F
    β_θ     ::F
    α_β     ::F
    β_β     ::F
    y       ::IntMat
    K       ::Int
    I       ::Int
    U       ::Int
    likeadj ::F
end

function NNMFGamGam(
    α_θ::F, β_θ::F, α_β::F, β_β::F, # Hyperparameters
    y::IntMat, # Data
    K::Int, I::Int, U::Int, likeadj = one(F)
) where {F <: Real}
    NNMFGamGam(α_θ, β_θ, α_β, β_β, y, K, I, U, likeadj)
end

@functor NNMFGamGam

function logdensity(model::NNMFGamGam, λ_β, λ_θ)
    # λ_θ ∈ ℝ^{K × U}
    # λ_β ∈ ℝ^{I × K}
    #
    # θ ∈ ℝ₊^{K × U}
    # β ∈ ℝ₊^{I × K}

    @unpack α_θ, β_θ, α_β, β_β, y, K, I, U, b_β, b_θ, likeadj = model

    b          = ExpBijector()
    β, ℓdetJ_β = forward(b, λ_β)
    θ, ℓdetJ_θ = forward(b, λ_θ)

    p_βᵢ = Gamma(α_β, 1/β_β)
    p_θᵢ Gamma(α_θ, 1/β_θ)

    ℓp_β = sum(βᵢ -> logpdf(p_βᵢ, βᵢ), β)
    ℓp_θ = sum(θᵢ -> logpdf(p_θᵢ, θᵢ), θ)

    λ    = β*θ
    # `mapreduce` would be more efficient but it currently doesn't work
    # with the CUDA+Zygote combination of doom.
    # See https://github.com/FluxML/Zygote.jl/issues/704
    ℓp_y  = sum(@. poislogpdf(λ, y))

    likeadj*ℓp_y + ℓp_β + ℓp_θ + ℓdetJ_β + ℓdetJ_θ
end

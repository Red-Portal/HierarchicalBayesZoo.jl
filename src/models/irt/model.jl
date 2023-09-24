
struct ItemResponse{
    VecInt  <: AbstractVector, 
    VecBool <: AbstractVector,
    Re
}
    J       ::Int # n_students
    K       ::Int # n_questions
    N       ::Int

    student ::VecInt
    question::VecInt
    correct ::VecBool

    recon_params::Re
end

@functor ItemResponse

struct ItemResponseParam{F<:Real, V<:AbstractVector{F}}
    μ_β   ::V
    η_σ_β ::V
    η_σ_γ ::V 
    η_γ   ::V # ℝ^K = ℝ^{n_questions}
    β     ::V # ℝ^K = ℝ^{n_questions}
    α     ::V # ℝ^J = ℝ^{n_students}
end

@functor ItemResponseParam

function LogDensityProblems.capabilities(::Type{<:ItemResponse})
    LogDensityProblems.LogDensityOrder{0}()
end

function bernlogitlogpdf(logit::Real, y::Bool)
    y ? -log1pexp(-logit) : -log1pexp(logit)
end

function logdensity(model::ItemResponse, param::ItemResponseParam{F,V}) where {F<:Real,V}
    @unpack J, K, N, student, question, correct = model
    @unpack μ_β, η_σ_β, η_σ_γ, α, β, η_γ = param
   
    @assert length(α)   == J
    @assert length(β)   == K
    @assert length(η_γ) == K

    μ_β′, η_σ_β′, η_σ_γ′ = sum(μ_β), sum(η_σ_β), sum(η_σ_γ)

    b⁻¹_sca = bijector(Exponential()) |> inverse
    b⁻¹_vec = ExpBijector()

    σ_β, logabsJ_σ_β = with_logabsdet_jacobian(b⁻¹_sca, η_σ_β′)
    σ_γ, logabsJ_σ_γ = with_logabsdet_jacobian(b⁻¹_sca, η_σ_γ′)
    γ,   logabsJ_γ   = forward(b⁻¹_vec, η_γ)

    #p_μ_β = logpdf(Cauchy{F}(0f0, 5f0), μ_β′)
    #p_σ_β = logpdf(truncated(Cauchy{F}(0f0, 5f0), 0, Inf), σ_β)
    #p_σ_γ = logpdf(truncated(Cauchy{F}(0f0, 5f0), 0, Inf), σ_γ)
    p_μ_β = logpdf(TDist{F}(4f0), μ_β′)
    p_σ_β = logpdf(truncated(TDist{F}(4f0), 0, Inf), σ_β)
    p_σ_γ = logpdf(truncated(TDist{F}(4f0), 0, Inf), σ_γ)

    p_α   = sum(αᵢ -> normlogpdf(αᵢ),         α)
    p_β   = sum(βᵢ -> normlogpdf(0, σ_β, βᵢ), β)

    ℓγ    = log.(γ)
    ℓZ    = log(2f0*π)/2
    p_γ   = sum(@. (ℓγ/σ_γ)^2/-2 - ℓγ - log(σ_γ) - ℓZ) # LogNormal log-pdf

    p_y   = sum(@. bernlogitlogpdf(
        γ[question]*α[student] - (β[question] + μ_β′), correct))

    p_y + p_γ + p_σ_γ + p_σ_β + p_α + p_β + p_μ_β +
        logabsJ_σ_β + logabsJ_σ_γ + logabsJ_γ
end

function LogDensityProblems.logdensity(model::ItemResponse, θ::AbstractVector)
    logdensity(model, model.recon_params(θ))
end

function LogDensityProblems.dimension(model::ItemResponse)
    @unpack J, K, N = model
    J + 2*K + 3
end



"""
# Robust Poisson Regression

## References

4.2.4 Poisson-Lognormal Mixture

Cameron, A. Colin, and Pravin K. Trivedi.
Regression analysis of count data.
Vol. 53. Cambridge university press, 2013.

Wang, Chong, and David M. Blei.
"A general method for robust Bayesian modeling."
Bayesian Analysis (2018): 1163-1191.
"""

struct RobustPoisson{
    Mat <: AbstractMatrix, 
    Vec <: AbstractVector, 
    Re
}
    X::Mat
    y::Vec

    recon_params::Re
end

@functor RobustPoisson

struct RobustPoissonParam{F<:Real, V<:AbstractVector{F}}
    η_σ_α::V
    η_σ_β::V
    η_σ_ϵ::V
    α    ::V
    β    ::V
    ϵ    ::V
end

@functor RobustPoissonParam

function LogDensityProblems.capabilities(::Type{<:RobustPoisson})
    LogDensityProblems.LogDensityOrder{0}()
end

function logpoislogpdf(ℓλ::T, x::T) where {T <: Real}
    val = x*ℓλ - exp(ℓλ) - loggamma(x + 1)
    return x >= 0 ? val : oftype(val, -Inf)
end

function logdensity(model::RobustPoisson, param::RobustPoissonParam{F,V}) where {F<:Real,V}
    @unpack X, y  = model
    @unpack α, β, η_σ_α, η_σ_β, η_σ_ϵ, ϵ = param

    b⁻¹ = bijector(Exponential()) |> inverse

    α′, η_σ_β′, η_σ_α′, η_σ_e′ = sum(α), sum(η_σ_β), sum(η_σ_α), sum(η_σ_ϵ)

    σ_α, logabsJ_σ_α = with_logabsdet_jacobian(b⁻¹, η_σ_α′)
    σ_β, logabsJ_σ_β = with_logabsdet_jacobian(b⁻¹, η_σ_β′)
    σ_ϵ, logabsJ_σ_ϵ = with_logabsdet_jacobian(b⁻¹, η_σ_e′)

    p_σ_α = logpdf(truncated(TDist{F}(4f0), 0, Inf), σ_α)
    p_σ_β = logpdf(truncated(TDist{F}(4f0), 0, Inf), σ_β)
    p_σ_e = logpdf(truncated(TDist{F}(4f0), 0, Inf), σ_ϵ)

    p_α = normlogpdf(0, σ_α, α′)
    p_β = sum(βᵢ -> normlogpdf(0, σ_β, βᵢ), β)
    p_ϵ = sum(ϵᵢ -> normlogpdf(0, σ_ϵ, ϵᵢ), ϵ)

    logits = X*β .+ α′ + ϵ
    p_y    = sum(@. logpoislogpdf(logits, y))

    p_y + p_α + p_β + p_ϵ + p_σ_α + p_σ_β + p_σ_e +
        logabsJ_σ_ϵ + logabsJ_σ_β + logabsJ_σ_α
end

function LogDensityProblems.logdensity(model::RobustPoisson, θ::AbstractVector)
    logdensity(model, model.recon_params(θ))
end

function LogDensityProblems.dimension(model::RobustPoisson)
    @unpack X = model
    size(X,1) + size(X,2) + 4
end


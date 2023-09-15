
struct Volatility{
    Mat       <: AbstractMatrix,
    Re,
    F         <: Real,
}
    x           ::Mat
    recon_params::Re
    likeadj     ::F
    b⁻¹_Σ       ::CorrCholBijector
end

struct VolatilityParam{
    F   <: Real,
    Mat <: AbstractMatrix{F},
    Vec <: AbstractVector{F},
}
    μ    ::Vec
    η_ϕ  ::Vec
    η_τ  ::Vec
    η_L_Σ::Vec
    y    ::Mat
end

@functor VolatilityParam

function LogDensityProblems.capabilities(::Type{<: Volatility})
    LogDensityProblems.LogDensityOrder{0}()
end

function logdensity(model::Volatility, param::VolatilityParam{F,M,V}) where {F,M,V}
    # Multivariate Stochastic Volatility 
    # μ  ~ Cauchy(0, 10)
    # ϕ  ~ Uniform(-1, 1)
    # Q  ~ LKJ()
    # τ  ~ Cauchy₊(0, 10)
    # L  = diag(τ)ᵀ Σ_Q⁻¹ diag(τ)
    # y₁ ~ N(μ, Q) 
    #
    # yₜ ~ N(μ + Φ*(yₜ₋₁ - μ), Q)
    # xₜ ~ N(0, exp(yₜ/2))
    @unpack x, likeadj, b⁻¹_Σ = model
    @unpack y, μ, η_ϕ, η_τ, η_L_Σ = param

    d = size(x,1)
    n = size(x,2)
    xₜ   = x
    yₜ   = y
    yₜ₋₁ = y[:,1:end-1]

    @assert d == length(μ)
    @assert d == length(η_ϕ)
    @assert d == length(η_τ)
    @assert d == size(x,1)
    @assert size(yₜ,2) == size(yₜ₋₁,2)+1

    b⁻¹_τ = bijector(Exponential())           |> inverse
    b⁻¹_ϕ = bijector(Uniform{F}(-1, 1))       |> inverse

    L_Σ_chol, logabsJ_Σ = forward(b⁻¹_Σ, Flux.cpu(η_L_Σ))
    τ,        logabsJ_τ = with_logabsdet_jacobian(b⁻¹_τ, Flux.cpu(η_τ))
    ϕ,        logabsJ_ϕ = with_logabsdet_jacobian(b⁻¹_ϕ, η_ϕ)

    L⁻¹_Q_cpu   = inv(L_Σ_chol)./τ
    L⁻¹_Q_dense = if η_L_Σ isa CuArray
        Flux.gpu(L⁻¹_Q_cpu)
    else
        L⁻¹_Q_cpu
    end
    L⁻¹_Q = if L⁻¹_Q_dense isa CuArray
        L⁻¹_Q_dense
    else
        LowerTriangular(L⁻¹_Q_dense)
    end

    ℓp_Q = logpdf(LKJCholesky(d, 1), Cholesky(L_Σ_chol))
    ℓp_μ = sum(Base.Fix1(logpdf, Cauchy{F}( 0, 10)), μ)
    ℓp_ϕ = sum(Base.Fix1(logpdf, Uniform{F}(-1, 1)), ϕ)
    ℓp_τ = sum(Base.Fix1(logpdf, truncated(Cauchy{F}(0, 5), zero(F), nothing)), τ)

    μ_yₜ  = hcat(μ, μ .+ (ϕ.*(yₜ₋₁ .- μ)))
    ℓp_yₜ = sum(normlogpdf, L⁻¹_Q*(yₜ - μ_yₜ)) + n*sum(log, diag(L⁻¹_Q))

    L⁻¹_xₜ_diag = @. exp(-clamp(yₜ, -15, 15)/2)
    ℓp_xₜ       = sum(normlogpdf, L⁻¹_xₜ_diag.*xₜ) + sum(log, L⁻¹_xₜ_diag)

    likeadj*(ℓp_xₜ + ℓp_yₜ) + ℓp_τ + ℓp_ϕ + ℓp_μ + ℓp_Q +
        logabsJ_τ + logabsJ_ϕ + logabsJ_Σ
end

function LogDensityProblems.logdensity(model::Volatility, θ::AbstractVector)
    logdensity(model, model.recon_params(θ))
end

function LogDensityProblems.dimension(model::Volatility)
    @unpack x = model
    d = size(x,1)
    n = size(x,2)
    d*n + d*3 + ((d*(d - 1)) ÷ 2)
end


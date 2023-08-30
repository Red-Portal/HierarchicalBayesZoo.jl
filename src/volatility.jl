
struct Volatility{
    Mat       <: AbstractMatrix,
    Re,
    F         <: Real,
}
    x           ::Mat
    recon_params::Re
    likeadj     ::F

    #global_idxs     ::Vector{Int}
    #local_block_idxs::Vector{Vector{Int}}
    #block_idx       ::Int
end

struct VolatilityParam{
    F   <: Real,
    Mat <: AbstractMatrix{F},
    Vec <: AbstractVector{F},
}
    y      ::Mat
    μ      ::Vec
    η_ϕ    ::Vec
    η_τ    ::Vec
    η_L⁻¹_Σ::Vec
end

@functor VolatilityParam

function Volatility(blocksize::Integer = 128)
    data, header   = readdlm(datadir("datasets", "snp500.csv"), ','; header=true)
    snp500_closing = convert(Vector{Float32}, data[end-1000:end,5])
    snp500_logret  = log.(snp500_closing[2:end]) - log.(snp500_closing[1:end-1])
    snp500_logret_centered = snp500_logret .- mean(snp500_logret)

    x = vcat(snp500_logret', snp500_logret' + 0.1*randn(size(snp500_logret')))

    d = size(x, 1)
    n = size(x, 2)
    θ = VolatilityParam(
        similar(x, d, n),          # y
        similar(x, d),             # μ
        similar(x, d),             # η_ϕ
        similar(x, d),             # η_τ
        similar(x, (d^2 + d) ÷ 2), # η_τ
    )
    _, re = Optimisers.destructure(θ)

    # n_dims      = length(θ)
    # local_idxs  = first(1:, n_data-1)
    # global_idxs = last( 1:n_dims, n_dims - length(local_idxs))

    # local_blocks_nonoverlap = Iterators.partition(local_idxs, blocksize) |> collect
    # map(local_blocks_nonoverlap) do local_block
    #     local_block
    # end

    Volatility(x, re, 1f0) #, global_idxs, local_block_idxs, 1)
end

# function subsample_problem(
#     prob ::Volatility,
#     q    ::LocationScale,
#     batch::Integer,
# )
#     @unpack x = prob

#     x_sub    = vcat(x[:,1], x[:,batch])
#     prob_sub = @set prob.x = x_sub

#     prob_sub, AmortizedLocationScale(q,)
# end

# function subsample_problem(
#     prob ::Volatility,
#     q    ::StructuredLocationScale,
#     batch::Integer,
# )
#     AmortizedLocationScale(q, )
# end

function LogDensityProblems.capabilities(::Type{<: Volatility})
    LogDensityProblems.LogDensityOrder{0}()
end

function normalprod(x::AbstractArray, μ::AbstractArray, L⁻¹::AbstractMatrix)
    d = size(x, 1)
    n = size(x, 2)
    sum(normlogpdf, L⁻¹*(x .- μ)) + n*sum(log, diag(L⁻¹))
end

function diagnormalprod(x::AbstractArray, μ::AbstractArray, L⁻¹::AbstractMatrix)
    d = size(x, 1)
    n = size(x, 2)
    sum(normlogpdf, L⁻¹*(x .- μ)) + n*sum(log, L⁻¹)
end

function logdensity(model, param::VolatilityParam{F,M,V}) where {F,M,V}
    # Multivariate Stochastic Volatility 
    # μ     ~ Cauchy(0, 10)
    # ϕ     ~ Uniform(-1, 1)
    # Σ_Q⁻¹ ~ LKJ
    # τ     ~ Cauchy₊(0, 10)
    # L     = diag(τ)ᵀ Σ_Q⁻¹ diag(τ)
    # y₁    ~ N(μ, Q) 
    #
    # yₜ ~ N(μ + Φ*(yₜ₋₁ - μ), Q)
    # xₜ ~ N(0, exp(yₜ/2))
    @unpack x, likeadj = model
    @unpack y, μ, η_ϕ, η_τ, η_L⁻¹_Σ = param

    d = size(x,1)
    n = size(x,2)

    y₁   = y[:,1]
    yₜ   = y[:,2:end]
    yₜ₋₁ = y[:,1:end-1]

    @assert d == length(μ)
    @assert d == length(η_ϕ)
    @assert d == length(η_τ)
    @assert d == size(x,1)
    @assert size(yₜ,2) == size(yₜ₋₁,2)

    x₁    = x[:,1]
    xₜ    = x[:,2:end]

    b⁻¹_Σ = Bijectors.PDVecBijector()   |> inverse
    b⁻¹_ϕ = bijector(Uniform{F}(-1, 1)) |> inverse
    b⁻¹_τ = bijector(Exponential())     |> inverse

    L⁻¹_Σ, logabsJ_Σ = with_logabsdet_jacobian(b⁻¹_Σ, η_L⁻¹_Σ)
    ϕ,     logabsJ_ϕ = with_logabsdet_jacobian(b⁻¹_ϕ, η_ϕ)
    τ,     logabsJ_τ = with_logabsdet_jacobian(b⁻¹_τ, η_τ)

    L⁻¹_Q = L⁻¹_Σ ./ τ

    ℓp_μ = sum(Base.Fix1(logpdf, Cauchy{F}( 0, 10)),                  μ)
    ℓp_ϕ = sum(Base.Fix1(logpdf, Uniform{F}(-1, 1)),                  ϕ)
    ℓp_τ = sum(Base.Fix1(logpdf, truncated(Cauchy{F}(0, 5), 0, Inf)), τ)

    L⁻¹_y₁ = L⁻¹_Q .* (@. sqrt(1 - ϕ^2))
    ℓp_y₁  = sum(normlogpdf, L⁻¹_y₁*x) + sum(log, diag(L⁻¹_y₁))

    μ_yₜ  = μ .+ (ϕ.*(yₜ₋₁ .- μ))
    ℓp_yₜ = sum(normlogpdf, L⁻¹_Q*(yₜ - μ_yₜ)) + n*sum(log, diag(L⁻¹_Q))

    L⁻¹_x₁_diag = @. exp(-y₁/2)
    ℓp_x₁       = sum(normlogpdf, L⁻¹_x₁_diag.*x₁) + sum(log, L⁻¹_x₁_diag)

    L⁻¹_xₜ_diag = @. exp(-yₜ/2)
    ℓp_xₜ       = sum(normlogpdf, L⁻¹_xₜ_diag.*xₜ) + sum(log, L⁻¹_xₜ_diag)

    likeadj*(ℓp_xₜ + ℓp_yₜ) + ℓp_x₁ + ℓp_y₁ +
        ℓp_τ + ℓp_ϕ + ℓp_μ + logabsJ_τ + logabsJ_ϕ + logabsJ_Σ
end

function LogDensityProblems.logdensity(model, θ::AbstractVector)
    logdensity(model, model.recon_params(θ))
end

function LogDensityProblems.dimension(model::Volatility)
    @unpack x = model
    d = size(x,1)
    n = size(x,2)
    d*n + d*3 + ((d^2 + d) ÷ 2)
end

# function subsample_problem(model::Volatility, batch)
#     @unpack x, data_indices = model

#     n       = last(data_indices)
#     m       = length(batch)
#     likeadj = n/m
#     Volatility(x[:,batch], likeadj, batch, data_indices)
# end

function test()
    n_dims    = 10
    n_obs     = 100
    batchsize = 30

    data_indices  = 2:n_obs
    batch_indices = sample(data_indices, batchsize, replace=false)
    model = Volatility(
        randn(Float32, n_dims, n_obs), 1f0, batch_indices, data_indices,
    )

    y_total = randn(n_dims, n_obs)
    y_idxs  = Set(vcat(batch_indices, batch_indices .- 1)) |> collect

    y_batch = y_total[:, y_idxs]
    y₁      = y_total[:, 1]
    μ       = randn(n_dims)
    η_τ     = randn(n_dims)
    η_ϕ     = randn(n_dims)
    η_L⁻¹_Σ = randn((n_dims^2 + n_dims) ÷ 2)

    model = @set model.x = randn(Float32, n_dims, length(batch_indices))

    z = vcat(reshape(y_batch, :), y₁, μ, η_τ, η_ϕ, η_L⁻¹_Σ)
    Base.Fix1(LogDensityProblems.logdensity, model)(z)
    #@benchmark Zygote.gradient(Base.Fix1(LogDensityProblems.logdensity, $model), $z)
end


struct Volatility{
    Mat       <: AbstractMatrix,
    Re,
    F         <: Real,
}
    x           ::Mat
    recon_params::Re
    likeadj     ::F
    b⁻¹_Σ       ::CorrCholBijector

    #global_idxs     ::Vector{Int}
    #local_block_idxs::Vector{Vector{Int}}
    #block_idx       ::Int
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

function Volatility(; use_cuda = false)
    currencies = [
        "EUR",
        "JPY",
        "GBP",
        "AUD",
        "CAD",
        #"CHF",
        #"HKD",
        #"SGD",
        #"SEK",
        "KRW",
    ]

    df = mapreduce(vcat, currencies) do name
        df       = CSV.read(datadir("datasets", "currencies", "$(name)=X.csv"), DataFrame)
        df.Name .= name
        @chain df begin
            @select(:Date, :Close, :Name)
            @subset((:Date .>= Date("2013-01-01")) .&& (:Date .<= Date("2022-12-31")))
            @subset(:Close .!= "null")
            @transform(:Close = parse.(Float64, :Close))
        end
    end

    eur_date =  filter(row -> row.Name .== "EUR", df).Date

    for currency in currencies
        @assert eur_date == filter(row -> row.Name .== currency, df).Date 
    end

    closing            = reshape(Array(df.Close), (:,length(currencies))) |> transpose |> Array
    logreturn          = log.(closing[:,2:end]) - log.(closing[:,1:end-1])
    logreturn_centered = logreturn .- mean(logreturn, dims=2)
   
    x_cpu = Array{Float32}(logreturn_centered)
    #x_cpu = Array{Float32}(logreturn_centered)[:,end-50:end]
    x     = use_cuda ? Flux.gpu(x_cpu) : x_cpu

    d = size(x, 1)
    n = size(x, 2)
    θ = VolatilityParam(
        similar(x, d),               # μ
        similar(x, d),               # η_ϕ
        similar(x, d),               # η_τ
        similar(x, (d*(d - 1)) ÷ 2), # η_Σ
        similar(x, d, n),            # y
    )
    _, re = Optimisers.destructure(θ)

    Volatility(x, re, 1f0, CorrCholBijector(d))
end

function StructuredLocationScale(
    prob::Volatility; use_cuda=false
)
    x = prob.x
    d = size(x, 1)
    n = size(x, 2)

    d_local  = d
    d_global = d*3 + ((d*(d - 1)) ÷ 2)

    location = zeros(eltype(x), n*d_local + d_global)
    diagonal = vcat(
        fill(convert(eltype(x), sqrt(0.1)), d_global),
        fill(convert(eltype(x), 1.0), n*d_local)
    )

    C_idx = []

    # μ, η_ϕ, η_τ
    block_idx  = 0
    push!(C_idx, diagonal_block_indices(block_idx, 3*d))
    block_idx += 3*d

    # η_Σ
    block_idx += (d*(d - 1)) ÷ 2

    for _ = 1:n
        push!(C_idx, bordered_diagonal_block_indices(block_idx, d_global, d))
        block_idx += d
    end

    offdiag_row = vcat([row_idxs for (row_idxs, col_idxs) ∈ C_idx]...)
    offdiag_col = vcat([col_idxs for (row_idxs, col_idxs) ∈ C_idx]...)
    offdiag_val = zeros(eltype(x), length(offdiag_row))

    StructuredLocationScale(
        location,
        diagonal,
        offdiag_row,
        offdiag_col,
        offdiag_val;
        use_cuda
    )
end

function AdvancedVI.VIMeanFieldGaussian(prob::Volatility; use_cuda=false)
    x        = prob.x
    d        = size(x, 1)

    x = prob.x
    d = size(x, 1)
    n = size(x, 2)

    d_local  = d
    d_global = d*3 + ((d*(d - 1)) ÷ 2)

    location = zeros(eltype(x), n*d_local + d_global)
    diagonal = vcat(
        fill(convert(eltype(x), sqrt(0.1)), d_global),
        fill(convert(eltype(x), 1.0), n*d_local)
    )

    if use_cuda
        AdvancedVI.VIMeanFieldGaussian(
            location |> Flux.gpu, Diagonal(diagonal |> Flux.gpu))
    else
        AdvancedVI.VIMeanFieldGaussian(location, Diagonal(diagonal))
    end
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

function logdensity(model, param::VolatilityParam{F,M,V}) where {F,M,V}
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

    L⁻¹_Q_cpu   = inv(L_Σ_chol) ./ τ
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

    L⁻¹_xₜ_diag = @. exp(-yₜ/2)
    ℓp_xₜ       = sum(normlogpdf, L⁻¹_xₜ_diag.*xₜ) + sum(log, L⁻¹_xₜ_diag)

    likeadj*(ℓp_xₜ + ℓp_yₜ) + ℓp_τ + ℓp_ϕ + ℓp_μ + ℓp_Q +
        logabsJ_τ + logabsJ_ϕ + logabsJ_Σ
end

function LogDensityProblems.logdensity(model, θ::AbstractVector)
    logdensity(model, model.recon_params(θ))
end

function LogDensityProblems.dimension(model::Volatility)
    @unpack x = model
    d = size(x,1)
    n = size(x,2)
    d*n + d*3 + ((d*(d - 1)) ÷ 2)
end

# function subsample_problem(model::Volatility, batch)
#     @unpack x, data_indices = model

#     n       = last(data_indices)
#     m       = length(batch)
#     likeadj = n/m
#     Volatility(x[:,batch], likeadj, batch, data_indices)
# end


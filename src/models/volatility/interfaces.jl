
function Volatility()
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
   
    #x_cpu = Array{Float32}(logreturn_centered)
    x = Array{Float32}(logreturn_centered)[:,end-100:end]

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

function StructuredGaussian(prob::Volatility)
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

    # global variables (dense block)
    block_idx  = 0
    push!(C_idx, diagonal_block_indices(block_idx, d_global))
    block_idx += d_global

    # local variables (bordered block-diagonal)
    # t = 1, ... n
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
    )
end

function AdvancedVI.VIFullRankGaussian(
    prob::Volatility; use_cuda=false
)
    x = prob.x
    d = size(x, 1)
    n = size(x, 2)

    d_local  = d
    d_global = d*3 + ((d*(d - 1)) ÷ 2)

    location   = zeros(eltype(x), n*d_local + d_global)
    scale_diag = vcat(
        fill(convert(eltype(x), sqrt(0.1)), d_global),
        fill(convert(eltype(x), 1.0), n*d_local)
    )
    scale = Diagonal(scale_diag) |> Matrix

    if use_cuda
        AdvancedVI.VIFullRankGaussian(
            location |> Flux.gpu, LowerTriangular(scale |> Flux.gpu))
    else
        AdvancedVI.VIFullRankGaussian(location, LowerTriangular(scale))
    end
end

function AdvancedVI.VIMeanFieldGaussian(prob::Volatility; use_cuda=false)
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

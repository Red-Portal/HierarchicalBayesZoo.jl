
struct GermanHealthRobustPoisson
    data_portion::Float64
end

function problem(::Random.AbstractRNG, prob::GermanHealthRobustPoisson)
    data = RDatasets.dataset("COUNT", "rwm5yr")   

    y = data[:,"DocVis"]
    X = data[:, [
        "Age",
        "Educ",
        "HHNInc",
        "OutWork",
        "Female",
        "Married",
        "Kids",
        "Self",
        "EdLevel1",
        "EdLevel2",
        "EdLevel3",
        "EdLevel4",
    ]]

    n     = size(y,1)
    n_sub = ceil(Int, n*prob.data_portion)

    sub_idx  = sample(1:n, n_sub, replace=false)
    y        = Vector{Float32}(y[sub_idx])
    X        = Matrix{Float32}(X[sub_idx,:])

    X[:,[1,2,3]] .-= mean(X[:,[1,2,3]])
    X[:,[1,2,3]]  /= std( X[:,[1,2,3]])

    n = size(X,1)
    p = size(X,2)

    θ = RobustPoissonParam(
        [0f0], [0f0], [0f0], [0f0], zeros(Float32, p), zeros(Float32, n)
    )
    _, re = Optimisers.destructure(θ)
    RobustPoisson(X, y, re)
end

function StructuredGaussian(prob::RobustPoisson)
    @unpack X = prob

    p        = size(X,2)
    n        = size(X,1)
    d_local  = 1
    d_global = p+4

    location = zeros(Float32, n*d_local + d_global)
    diagonal = vcat(
        fill(sqrt(.1f0), d_global),
        fill(sqrt(.1f0), n*d_local)
    )

    C_idx     = []
    block_idx = 0

    # hyperparameters
    push!(C_idx, diagonal_block_indices(block_idx, d_global))
    block_idx += d_global

    # α ∈ ℝ^J
    for _ = 1:n
        push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:d_global, 1))
        block_idx += 1
    end

    offdiag_row = vcat([row_idxs for (row_idxs, col_idxs) ∈ C_idx]...)
    offdiag_col = vcat([col_idxs for (row_idxs, col_idxs) ∈ C_idx]...)
    offdiag_val = zeros(Float32, length(offdiag_row))

    StructuredLocationScale(
        location,
        diagonal,
        offdiag_row,
        offdiag_col,
        offdiag_val;
    )
end

# function AdvancedVI.VIFullRankGaussian(prob::ItemResponse)
#     @unpack J, K = prob
#     n = J

#     d_local  = 1
#     d_global = 2*K + 3

#     location   = zeros(Float32, n*d_local + d_global)
#     scale_diag = vcat(
#         fill(convert(Float32, sqrt(0.1)), d_global),
#         fill(convert(Float32, sqrt(0.1)), n*d_local)
#     )
#     scale = Diagonal(scale_diag) |> Matrix
#     AdvancedVI.VIFullRankGaussian(location, LowerTriangular(scale))
# end

function AdvancedVI.VIMeanFieldGaussian(prob::RobustPoisson)
    @unpack X = prob
    n = size(X,1)

    d_local  = 1
    d_global = 4 + size(X,2)

    location = zeros(Float32, n*d_local + d_global)
    diagonal = vcat(
        fill(convert(Float32, sqrt(.1)),  d_global),
        fill(convert(Float32, sqrt(.1)), n*d_local)
    )
    AdvancedVI.VIMeanFieldGaussian(location, Diagonal(diagonal))
end

function AdvancedVI.VIFullRankGaussian(prob::RobustPoisson)
    @unpack X = prob
    n = size(X,1)

    d_local  = 1
    d_global = 4 + size(X,2)

    location   = zeros(Float32, n*d_local + d_global)
    scale_diag = vcat(
        fill(convert(Float32, sqrt(.1)), d_global),
        fill(convert(Float32, sqrt(.1)), n*d_local)
    )
    scale = Diagonal(scale_diag) |> Matrix
    AdvancedVI.VIFullRankGaussian(location, LowerTriangular(scale))
end

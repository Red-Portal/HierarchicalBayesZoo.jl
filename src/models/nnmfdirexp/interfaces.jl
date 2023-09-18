
struct MovieLensNNMF
    data_portion::Float64
end

struct FashionNNMF
    data_portion::Float64
end

function problem(prob::MovieLensNNMF)
    # y[1] is the user
    # y[2] is the item
    # y[3] is the rating

    #I = 1682
    #U = 943
    y_entries = readdlm(datadir("datasets", "movielens", "u.data"), Int)[:,1:3]
    y         = sparse(y_entries[:,2], y_entries[:,1], y_entries[:,3])   

    I_sub = round(Int, size(y,1)*prob.data_portion)
    U_sub = round(Int, size(y,2)*prob.data_portion)
    y     = Matrix(y[1:I_sub, 1:U_sub])

    α  = .5f0
    λ₀ = 1f0
    K  = 3
    I  = size(y,1)
    U  = size(y,2)
    NNMFDirExp(α, λ₀, y, K, I, U)
end

function problem(prob::FashionNNMF)
    train_x, _ = MLDatasets.FashionMNIST(split=:train)[:]
    y  = reshape(train_x, (28*28, :))

    α  = .5f0
    λ₀ = 1f0
    K  = 3
    I  = size(y,1)
    U  = size(y,2)
    NNMFDirExp(α, λ₀, y, K, I, U)
end

function AdvancedVI.VIMeanFieldGaussian(prob::NNMFDirExp)
    d = LogDensityProblems.dimension(prob)
    m = zeros(Float32, d)
    σ = fill(sqrt(Float32(1.0)), d)
    AdvancedVI.VIMeanFieldGaussian(m, Diagonal(σ))
end

function AdvancedVI.VIFullRankGaussian(prob::NNMFDirExp)
    d = LogDensityProblems.dimension(prob)
    m = zeros(Float32, d)
    σ = fill(sqrt(Float32(1.0)), d)
    C = Diagonal(σ) |> Matrix
    AdvancedVI.VIFullRankGaussian(m, LowerTriangular(C))
end

function StructuredGaussian(prob::NNMFDirExp)
    @unpack K, I, U, likeadj = prob

    d_local  = K - 1
    d_global = K*I

    location = zeros(Float32, U*d_local + d_global)
    #diagonal = fill(sqrt(Float32(1.0)), U*d_local + d_global)
    diagonal = vcat(
        fill(convert(Float32, sqrt(.1)), d_global),
        fill(convert(Float32, 1.0), U*d_local)
    )

    C_idx = []

    # global variables (dense block)
    block_idx  = 0
    # for _ = 1:I
    #     push!(C_idx, diagonal_block_indices(block_idx, K))
    #     block_idx += K
    # end

    # local variables (bordered block-diagonal)
    for _ = 1:U
        push!(C_idx, bordered_diagonal_block_indices(block_idx, d_global, d_local))
        block_idx += d_local
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

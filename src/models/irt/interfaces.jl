
struct CritLangIRT
    data_portion::Float64
end

function problem(rng::Random.AbstractRNG, prob::CritLangIRT)
    data, _   = readdlm(datadir("datasets", "irt", "critlangacq", "data.csv"), ',', header=true)
    
    J           = size(data, 1) # n_students
    J_sub       = ceil(Int, J*prob.data_portion)
    student_idx = sample(rng, 1:J, J_sub, replace=false)
    data        = data[student_idx,:]

    students = convert(Array{Int},  data[:,2])
    corrects = convert(Array{Bool}, data[:,32:end-2])

    J = size(corrects, 1) # n_students
    K = size(corrects, 2) # n_questions
    N = J*K

    students  = repeat(1:J, outer=K)
    questions = repeat(1:K, inner=J)
    corrects  = reshape(corrects, :)

    @assert N == length(students)
    @assert N == length(corrects)
    @assert N == length(questions)

    θ = ItemResponseParam(
        [0f0], [0f0], [0f0], zeros(Float32, K), zeros(Float32, K), zeros(Float32, J)
    )
    _, re = Optimisers.destructure(θ)

    ItemResponse(J, K, N, students, questions, corrects, re)
end

function StructuredGaussian(prob::ItemResponse)
    @unpack J, K, question = prob

    n        = J
    d_local  = 1
    d_global = K*2 + 3

    location = zeros(Float32, n*d_local + d_global)
    diagonal = vcat(
        fill(sqrt(0.1f0), d_global),
        fill(sqrt(0.1f0), n*d_local)
    )

    C_idx = []

    # hyperparameters
    block_idx  = 0
    push!(C_idx, diagonal_block_indices(block_idx, 3+2*K))
    block_idx += 3+2*K

    # γ ∈ ℝ^K
    #push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:3, K))
    #block_idx += K

    #for _ = 1:K
    #    push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:3, 1))
    #    block_idx += 1
    #end

    # β ∈ ℝ^K
    #push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:3+K, K))
    #block_idx += K

    #for _ = 1:K
    #    push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:3, 1))
    #    block_idx += 1
    #end

    # α ∈ ℝ^J
    for _ = 1:J
        push!(C_idx, bordered_diagonal_block_indices(block_idx, 1:3+2*K, 1))
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

function AdvancedVI.VIFullRankGaussian(prob::ItemResponse)
    @unpack J, K = prob
    n = J

    d_local  = 1
    d_global = 2*K + 3

    location   = zeros(Float32, n*d_local + d_global)
    scale_diag = vcat(
        fill(convert(Float32, sqrt(0.1)), d_global),
        fill(convert(Float32, sqrt(0.1)), n*d_local)
    )
    scale = Diagonal(scale_diag) |> Matrix
    AdvancedVI.VIFullRankGaussian(location, LowerTriangular(scale))
end

function AdvancedVI.VIMeanFieldGaussian(prob::ItemResponse)
    @unpack J, K = prob
    n = J

    d_local  = 1
    d_global = 2*K + 3

    location = zeros(Float32, n*d_local + d_global)
    diagonal = vcat(
        fill(convert(Float32, sqrt(0.1)),  d_global),
        fill(convert(Float32, sqrt(0.1)), n*d_local)
    )
    AdvancedVI.VIMeanFieldGaussian(location, Diagonal(diagonal))
end

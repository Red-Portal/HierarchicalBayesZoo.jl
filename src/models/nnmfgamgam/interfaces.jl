
# struct BSSNNMF
#     data_portion::Float64
# end

function problem(prob::BSSNNMF)
    song  = load(datadir("datasets", "songs", "rumble.jld2"))
    y     = song["signal"]
    nfft  = 512
    spec  = tfd(y[:,1], Spectrogram(; nfft, noverlap=nfft÷2, window=hamming))
    y     = spec.power
    U_sub = round(Int, prob.data_portion*size(y,2))
    y_sub = y[:,1:U_sub]

    # Quantization to form Poisson noise
    y_sub_quant = round.(Int16, y_sub / nfft * 2^15) 

    α_θ = 1.0
    β_θ = 1.0
    α_β = 1.0
    β_β = 0.1
    K   = 3
    I   = size(y_sub_quant,1)
    U   = size(y_sub_quant,2)
    NNMFGamGam(α_θ, β_θ, α_β, β_β, y_sub_quant, K, I, U)
end

function AdvancedVI.VIMeanFieldGaussian(prob::NNMFGamGam)
    d = LogDensityProblems.dimension(prob)
    m = zeros(Float32, d)
    σ = fill(sqrt(Float32(1.0)), d)
    AdvancedVI.VIMeanFieldGaussian(m, Diagonal(σ))
end

function AdvancedVI.VIFullRankGaussian(prob::NNMFGamGam)
    d = LogDensityProblems.dimension(prob)
    m = zeros(Float32, d)
    σ = fill(sqrt(Float32(1.0)), d)
    C = Diagonal(σ) |> Matrix
    AdvancedVI.VIFullRankGaussian(m, LowerTriangular(C))
end

function StructuredGaussian(prob::NNMFGamGam)
    @unpack K, I, U, likeadj = prob

    d_local  = K
    d_global = K*I

    location = zeros(Float32, U*d_local + d_global)
    #diagonal = fill(sqrt(Float32(1.0)), U*d_local + d_global)
    diagonal = vcat(
        fill(convert(Float32, sqrt(1.0)), d_global),
        fill(convert(Float32, sqrt(0.1)), U*d_local)
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

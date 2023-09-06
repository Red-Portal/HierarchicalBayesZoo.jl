
struct StructuredLocationScale{
    Loc       <: AbstractVector,
    Idx       <: AbstractVector{<:Integer},
    ScaleVals <: AbstractVector
}
    location ::Loc

    scale_rows::Idx
    scale_cols::Idx
    # The first `length(location)` values are reserved to be
    # the diagonal of the matrix
    scale_vals::ScaleVals

    batch_idx::Idx
end

@functor StructuredLocationScale (location, scale_vals)

function StatsBase.entropy(q::StructuredLocationScale)
    @unpack location, scale_vals = q
    d = length(location)
    ℍ_base      = d*log(2*π*ℯ)/2
    logdetscale = sum(x -> log(abs(x)), scale_vals[1:d])
    ℍ_base + logdetscale
end

function StructuredLocationScale(
    location   ::AbstractVector{F},
    diagonal   ::AbstractVector{F},
    offdiag_row::AbstractVector{I},
    offdiag_col::AbstractVector{I},
    offdiag_val::AbstractVector{F};
    use_cuda = false
) where {F<:Real, I<:Integer}
    @assert length(offdiag_val) == length(offdiag_col)
    @assert length(offdiag_val) == length(offdiag_row)

    d = length(location)

    scale_rows = vcat(1:d,      offdiag_row)
    scale_cols = vcat(1:d,      offdiag_col)
    scale_vals = vcat(diagonal, offdiag_val)
    batch_idx  = collect(1:d)

    if use_cuda
        StructuredLocationScale(
            location   |> Flux.gpu,
            scale_rows |> Flux.gpu,
            scale_cols |> Flux.gpu,
            scale_vals |> Flux.gpu,
            batch_idx  |> Flux.gpu
        )
    else
        StructuredLocationScale(
            location,
            scale_rows,
            scale_cols,
            scale_vals,
            batch_idx
        )
    end
end

function diagonal_block_indices(block_start_idx::Int, block_dim::Int)
    meshgrid      = Iterators.product(1:block_dim, 1:block_dim) |> collect
    row_grid      = map(x -> x[1], meshgrid)
    col_grid      = map(x -> x[2], meshgrid)
    row_tril_grid = tril(row_grid, -1)
    col_tril_grid = tril(col_grid, -1)

    row_tril_idxs = row_tril_grid[row_tril_grid .!= 0]
    col_tril_idxs = col_tril_grid[col_tril_grid .!= 0]

    row_diagblock_idx = row_tril_idxs .+ block_start_idx
    col_diagblock_idx = col_tril_idxs .+ block_start_idx
    row_diagblock_idx, col_diagblock_idx
end

function bordered_diagonal_block_indices(
    block_start_idx::Int, border_size::Int, block_dim::Int
)
    row_diagblock_idx, col_diagblock_idx = diagonal_block_indices(
        block_start_idx, block_dim
    )

    row_border_idx = repeat((1:block_dim) .+ block_start_idx, inner=border_size)
    col_border_idx = repeat(1:border_size, outer=block_dim)

    row_idx = vcat(row_diagblock_idx, row_border_idx)
    col_idx = vcat(col_diagblock_idx, col_border_idx)
    row_idx, col_idx
end

function amortize(q::StructuredLocationScale, batch::AbstractVector{<:Integer})
    @set q.batch_idx = batch
end

_sparsity_preserving_mul(A::AbstractSparseMatrix, x::AbstractArray) = A*x

sparsity_preserving_mul(
    A::CUDA.CUSPARSE.CuSparseMatrixCSC,
    x::CUDA.CuMatrix,
     ::AbstractVector{<:Integer}
) = _sparsity_preserving_mul(A, x)

@adjoint function sparsity_preserving_mul(
    A   ::CUDA.CUSPARSE.CuSparseMatrixCSC,
    x   ::CUDA.CuMatrix,
    cols::AbstractVector{<:Integer}
)
    z = _sparsity_preserving_mul(A, x)
    z, Δ -> begin
        @unpack colPtr, rowVal, dims = A
        # non-zero entries of the Jacobian.
        jac_nz_vals = x[cols,:]
        Δ_nz_vals   = Δ[rowVal,:]
        ∂C_nz_vals  = sum(jac_nz_vals.*Δ_nz_vals, dims=2)[:,1]

        ∂C = CUDA.CUSPARSE.CuSparseMatrixCSC(
            colPtr, rowVal, ∂C_nz_vals, dims
        )
        (∂C, nothing, nothing)
    end
end

@adjoint function CUDA.CUSPARSE.sparse(rows, cols, vals)
    A = sparse(rows, cols, vals)
    A, Δ -> (nothing, nothing, nonzeros(Δ))
end

function Distributions.rand(
    rng      ::Random.AbstractRNG,
    q        ::StructuredLocationScale,
    n_samples::Integer
)
    @unpack location, scale_rows, scale_cols, scale_vals, batch_idx = q

    scale = sparse(scale_rows, scale_cols, scale_vals)
    d     = length(batch_idx)
    u     = randn(rng, eltype(location), d, n_samples)

    sparsity_preserving_mul(scale, u, scale_cols) .+ location
end


using Zygote

struct StructuredLocationScale{
    Loc       <: AbstractVector,
    Idx       <: AbstractVector{<:Integer},
    ScaleVals <: AbstractVector
}
    location ::Loc

    scale_rows::Idx
    scale_cols::Idx

    # The first 1:`length(location)` values are reserved to be
    # the diagonal of the matrix
    scale_vals::ScaleVals

    amortize_idx::Idx
end

@functor StructuredLocationScale

struct RestructStructuredLocScale{Q <: StructuredLocationScale}
    q::Q
end

function update(q_base::StructuredLocationScale, flat::AbstractVector)
    @unpack location, scale_rows, scale_cols, amortize_idx = q_base
    location′   = flat[1:length(location)]
    scale_vals′ = flat[length(location)+1:end]
    StructuredLocationScale(
        location′, scale_rows, scale_cols, scale_vals′, amortize_idx)
end

function (re::RestructStructuredLocScale)(flat::AbstractVector)
    update(re.q, flat)
end

@adjoint function (re::RestructStructuredLocScale)(flat::AbstractVector)
    q = update(re.q, flat)
    q, Δ -> begin
        ∂location   = Δ.location
        ∂scale_vals = Δ.scale_vals
        (nothing, vcat(∂location, ∂scale_vals),)
    end
end

function Optimisers.destructure(q::StructuredLocationScale)
    @unpack location, scale_vals = q 
    vcat(location, scale_vals), RestructStructuredLocScale(q)
end

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
    offdiag_val::AbstractVector{F}
) where {F<:Real, I<:Integer}
    @assert length(offdiag_val) == length(offdiag_col)
    @assert length(offdiag_val) == length(offdiag_row)

    d = length(location)

    scale_rows = vcat(1:d,      offdiag_row)
    scale_cols = vcat(1:d,      offdiag_col)
    scale_vals = vcat(diagonal, offdiag_val)
    amortize_idx  = collect(1:d)

    StructuredLocationScale(
        location,
        scale_rows,
        scale_cols,
        scale_vals,
        amortize_idx
    )
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

_sparsity_preserving_mul(A::AbstractSparseMatrix, x::AbstractArray) = A*x

sparsity_preserving_mul(
    A::AbstractSparseMatrix,
    x::AbstractMatrix,
     ::AbstractVector{<:Integer},
     ::AbstractVector{<:Integer}
) = _sparsity_preserving_mul(A, x)

@adjoint function sparsity_preserving_mul(
    A   ::AbstractSparseMatrix,
    x   ::AbstractMatrix,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer}
)
    z = _sparsity_preserving_mul(A, x)
    z, Δ -> begin
        @unpack colPtr, rowVal, dims = A
        # non-zero entries of the Jacobian.

        jac_nz_vals = x[cols,:]
        Δ_nz_vals   = Δ[rows,:]
        ∂C_nz_vals  = sum(jac_nz_vals.*Δ_nz_vals, dims=2)[:,1]

        #@assert rowVal == rows

        #∂C = sparse(rows, cols, ∂C_nz_vals)
        ∂C = CUDA.CUSPARSE.CuSparseMatrixCSC(
            colPtr, rowVal, ∂C_nz_vals, dims
        )
        (∂C, nothing, nothing, nothing)
    end
end

function diffable_sparse(rows, cols, vals, m, n)
    perm_idx    = sortperm(cols)
    rows_sorted = rows[perm_idx]
    cols_sorted = cols[perm_idx]
    vals_sorted = vals[perm_idx]
    sparse(rows_sorted, cols_sorted, vals_sorted, m, n)
end

@adjoint function sparse(rows, cols, vals, m, n)
    # rows, cols, vals are assumed to be sorted with the column index
    # A is assumed to be in the CSC format
    A = sparse(rows, cols, vals, m, n)
    A, Δ -> begin
        (nothing, nothing, nonzeros(Δ), nothing, nothing)
    end
end

function amortize(prob, q::StructuredLocationScale, batch)
    @set q.amortize_idx = convert(typeof(q.amortize_idx), amortize_indices(prob, batch))
end

function Distributions.rand(
    rng      ::Random.AbstractRNG,
    q        ::StructuredLocationScale,
    n_samples::Integer
)
    @unpack location, scale_rows, scale_cols, scale_vals, amortize_idx = q

    perm_idx          = sortperm(scale_cols)
    scale_rows_sorted = scale_rows[perm_idx]
    scale_cols_sorted = scale_cols[perm_idx]
    scale_vals_sorted = scale_vals[perm_idx]

    B     = length(amortize_idx)
    d     = length(location)
    scale = sparse(scale_rows_sorted, scale_cols_sorted, scale_vals_sorted, B, d)

    u = randn(rng, eltype(location), d, n_samples)

    sparsity_preserving_mul(scale, u, scale_rows_sorted, scale_cols_sorted) .+ location
end

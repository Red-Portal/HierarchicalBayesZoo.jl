
struct StructuredLocationScale{
    Loc     <: AbstractVector,
    IntVec  <: AbstractVector{<:Integer},
    RealVec <: AbstractVector{<:Real}
}
    location ::Loc

    colptr::IntVec
    colval::IntVec
    rowval::IntVec
    nzval ::RealVec

    amortize_idx::IntVec
end

@functor StructuredLocationScale

struct RestructStructuredLocScale{Q <: StructuredLocationScale}
    q::Q
end

function update(q_base::StructuredLocationScale, flat::AbstractVector)
    @unpack location, colval, colptr, rowval, nzval, amortize_idx = q_base
    location′ = flat[1:length(location)]
    nzval′    = flat[length(location)+1:end]
    StructuredLocationScale(location′, colptr, colval, rowval, nzval′, amortize_idx)
end

function (re::RestructStructuredLocScale)(flat::AbstractVector)
    update(re.q, flat)
end

@adjoint function (re::RestructStructuredLocScale)(flat::AbstractVector)
    q = update(re.q, flat)
    q, Δ -> begin
        ∂location = Δ.location
        ∂nzval    = Δ.nzval
        (nothing, vcat(∂location, ∂nzval),)
    end
end

function Optimisers.destructure(q::StructuredLocationScale)
    @unpack location, nzval = q 
    vcat(location, nzval), RestructStructuredLocScale(q)
end

function scalediag(q::StructuredLocationScale)
    @unpack colptr, rowval, nzval = q
    nzval[colptr[1:end-1]]
end

function StatsBase.entropy(q::StructuredLocationScale)
    @unpack location, nzval = q
    d           = length(location)
    ℍ_base      = d*log(2*π*ℯ)/2
    logdetscale = sum(x -> log(abs(x)), scalediag(q))
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

    nz_idx     = sortperm(scale_cols)
    scale_rows = scale_rows[nz_idx]
    scale_cols = scale_cols[nz_idx]
    scale_vals = scale_vals[nz_idx]

    scale_init = sparse(scale_rows, scale_cols, scale_vals)

    colval = scale_cols
    colptr = scale_init.colptr
    rowval = scale_init.rowval
    nzval  = scale_init.nzval

    @assert length(rowval) == length(scale_rows)

    amortize_idx = collect(1:d)

    StructuredLocationScale(
        location,
        colptr,
        colval,
        rowval,
        nzval,
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
    block_start_idx::Int,
    border_idxs    ::AbstractVector{<:Integer},
    block_dim      ::Int
)
    row_diagblock_idx, col_diagblock_idx = diagonal_block_indices(
        block_start_idx, block_dim
    )

    n_border       = length(border_idxs)
    row_border_idx = repeat((1:block_dim) .+ block_start_idx, inner=n_border)
    col_border_idx = repeat(border_idxs, outer=block_dim)

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

        ∂C = CUDA.CUSPARSE.CuSparseMatrixCSC(
            colPtr, rowVal, ∂C_nz_vals, dims
        )
        (∂C, nothing, nothing, nothing)
    end
end

# function _sparse_diffable(colptr, rowval, nzval, m, n)
#     CUDA.CUSPARSE.CuSparseMatrixCSC{eltype(nzval), eltype(colptr)}(
#         colptr, rowval, nzval, (m, n)
#     )
# end

# function sparse_diffable(colptr, rowval, nzval, m, n)
#     _sparse_diffable(colptr, rowval, nzval, m, n)
# end

@adjoint function CUDA.CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
    A = CUDA.CUSPARSE.CuSparseMatrixCSC{eltype(nzval), eltype(colptr)}(
        colptr, rowval, nzval, dims
    )
    A, Δ -> begin
        (nothing, nothing, nonzeros(Δ), nothing)
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
    @unpack location, colptr, colval, rowval, nzval, amortize_idx = q

    d     = length(location)
    scale = CUDA.CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, (d, d))

    u = randn(rng, eltype(location), d, n_samples)

    # Code for Randomized Quasi Monte Carlo
    #u′ = randn(rng, eltype(location), nextpow(2, d), n_samples)
    #u = u′[1:d,:]

    sparsity_preserving_mul(scale, u, rowval, colval) .+ location
end

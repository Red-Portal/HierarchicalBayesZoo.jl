
struct StructuredLocationScale{
    Vec           <: AbstractVector,
    VecBatch      <: AbstractMatrix,
    Mat           <: AbstractMatrix,
    MatBatch      <: AbstractArray,
    MatInt        <: AbstractMatrix{<:Integer},
    VecInt        <: AbstractVector{<:Integer}
}
    location_z::Vec
    location_y::VecBatch
    diagonal_z::Mat
    diagonal_y::MatBatch
    border    ::MatBatch
    diag_idx  ::MatInt
    triu_idx  ::MatInt
    batch_idx ::VecInt
end

@functor StructuredLocationScale (location_z, location_y, diagonal_z, diagonal_y, border)

function StatsBase.entropy(q::StructuredLocationScale)
    @unpack diagonal_z, diagonal_y, diag_idx = q
    d_z    = size(diagonal_z, 1)
    d_y    = size(diagonal_y, 1)
    n      = size(diagonal_y, 3)
    d      = d_z + d_y*n
    ℍ_base = d*log(2*π*ℯ)/2

    logdet_z = sum(x -> log(abs(x)), diag(diagonal_z))
    logdet_y = sum(x -> log(abs(x)), diagonal_y[diag_idx])
    ℍ_base + logdet_z + logdet_y
end

function IsoStructuredLocationScale(
    d_z     ::Int,
    d_y     ::Int,
    n       ::Int,
    isoscale::F;
    use_cuda = false
) where {F<:Real}
    m_z  = use_cuda ? CUDA.zeros(F, d_z)         : zeros(F, d_z)
    m_y  = use_cuda ? CUDA.zeros(F, d_y, n)      : zeros(F, d_y, n)
    B    = use_cuda ? CUDA.zeros(F, d_y, n, d_z) : zeros(F, d_y, n, d_z)
    D_z  = use_cuda ? CUDA.zeros(F, d_z, d_z)    : zeros(F, d_z, d_z)
    D_y  = use_cuda ? CUDA.zeros(F, d_y, d_y, n) : zeros(F, d_y, d_y, n)

    D_z[diagind(D_z)] .= isoscale

    diag_idx_cpu       = diagind(D_y[:,:,1])
    diag_idx_block_cpu = mapreduce(hcat, 1:n) do i
        (i-1)*d_y^2 .+ diag_idx_cpu
    end
    diag_idx_bock_i32    = convert(Array{Int32}, diag_idx_block_cpu)
    diag_idx_block       = use_cuda ? Flux.gpu(diag_idx_bock_i32) : diag_idx_bock_i32
    D_y[diag_idx_block] .= isoscale

    triu_entry         = triu(reshape(1:d_y*d_y, (d_y, d_y)), 1)
    triu_idx           = triu_entry[triu_entry .!= 0]
    block_triu_idx     = mapreduce(hcat, 1:n) do i
        (i-1)*d_y^2 .+ triu_idx
    end
    block_triu_idx_i32 = convert(Array{Int32}, block_triu_idx)
    block_triu_idx     = use_cuda ? Flux.gpu(block_triu_idx_i32) : block_triu_idx_i32

    batch_idx_cpu = convert(Array{Int32}, collect(1:n))
    batch_idx     = use_cuda ? Flux.gpu(batch_idx_cpu) : batch_idx_cpu

    StructuredLocationScale(
        m_z, m_y, D_z, D_y, B, diag_idx_block, block_triu_idx, batch_idx
    )
end

function amortize(
    q    ::StructuredLocationScale,
    batch::AbstractVector{<:Integer}
)
    @set q.batch_idx = batch
end

function tril_batch(A, triu_idx)
    A[triu_idx] .= zero(eltype(A))
    A
end

@adjoint function tril_batch(A, triu_idx)
    A[triu_idx] .= zero(eltype(A))
    A, Δ -> begin
        Δ[triu_idx] .= zero(eltype(A))
        (Δ, nothing)
    end
end

function Distributions.rand(
    rng      ::Random.AbstractRNG,
    q        ::StructuredLocationScale,
    n_samples::Integer
)
    @unpack location_z, location_y, diagonal_z, diagonal_y, border, triu_idx, batch_idx = q

    batchsize = length(batch_idx)

    n   = size(diagonal_y, 3)
    d_z = size(diagonal_z, 1)
    d_y = size(diagonal_y, 1)

    u_z       = randn(rng, eltype(location_z), d_z, n_samples)
    u_y_flat  = randn(rng, eltype(location_z), batchsize*d_y, n_samples)
    u_y_batch = reshape(u_y_flat, (d_y, n_samples, batchsize))

    m_batch   = reshape(location_y[:,batch_idx], :)
    B_batch   = reshape(border[:,batch_idx,:],     (d_y*batchsize, d_z))
    D_y_batch = reshape(diagonal_y[:,:,batch_idx], (d_y, d_y, batchsize))

    triu_idx_batch = reshape(triu_idx[:,1:batchsize],:)
    D_y_batch_tril = tril_batch(D_y_batch, triu_idx_batch)

    Du_batch_perm = NNlib.batched_mul(D_y_batch_tril, u_y_batch)
    Du_batch      = reshape(permutedims(Du_batch_perm, (1, 3, 2)), (d_y*batchsize, n_samples))

    y_batch = B_batch*u_z + Du_batch .+ m_batch
    z       = diagonal_z*u_z .+ location_z
    vcat(z, y_batch)
end

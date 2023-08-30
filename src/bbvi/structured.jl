
struct StructuredLocationScale{
    Loc       <: AbstractVector,
    ScaleDiag <: AbstractTriangular,
    SCaleBord <: AbstractMatrix,
    Ind       <: AbstractVector{<:Integer}
}
    m_global      ::Loc
    m_locals      ::Vector{Loc}
    D_global      ::ScaleDiag
    D_locals      ::Vector{ScaleDiag}
    B_locals      ::Vector{SCaleBord}
    amortize_index::Ind
end

@functor StructuredLocationScale (m_global, m_locals, D_global, D_locals, B_locals)

function StatsBase.entropy(q::StructuredLocationScale)
    @unpack m_locals, m_global, D_global, D_locals, amortize_index = q

    n        = length(m_locals)
    d_global = length(m_global)
    d_local  = length(m_locals) > 0 ? length(first(m_locals)) : 0
    d        = d_global + d_local*n
    ℍ_base   = d*log(2*π*ℯ)/2

    det_global = sum(log, diag(D_global))
    det_locals = sum(D_locals) do D_local
        sum(log, diag(D_local))
    end
    ℍ_base + det_global + det_locals
end

function IsoStructuredLocationScale(
    m₀_global::Union{<:AbstractVector{T}, Nothing},
    m₀_local ::AbstractVector{T},
    isoscale ::T,
    n_locals ::Integer
) where {T<:Real}
    d_global  = length(m₀_global)
    d_local   = length(m₀_local)

    D₀_global = similar(m₀_global, d_global, d_global)
    fill!(D₀_global, zero(eltype(D₀_global)))
    D₀_global[diagind(D₀_global)] .= one(eltype(D₀_global))
    D₀_global = D₀_global |> LowerTriangular

    m₀_locals = [copy(m₀_local) for _ ∈ 1:n_locals]
    D₀_locals = map(1:n_locals) do _
        D₀_local = similar(m₀_local, d_local,  d_local)
        fill!(D₀_local, zero(eltype(D₀_local)))
        D₀_local[diagind(D₀_local)] .= one(eltype(D₀_local))
        D₀_local |> LowerTriangular
    end

    B₀_locals = map(1:n_locals) do _
        B₀_local = similar(m₀_local, d_local, d_global)
        fill!(B₀_local, zero(eltype(B₀_local)))
        B₀_local 
    end

    StructuredLocationScale(
        m₀_global, m₀_locals,
        D₀_global, D₀_locals,
        B₀_locals, 1:n_locals
    )
end

function amortize(
    prob,
    q    ::StructuredLocationScale,
    batch::AbstractVector{<:Integer}
)
    @set q.amortize_index = batch
end

function Distributions.rand(
    rng      ::Random.AbstractRNG,
    q        ::StructuredLocationScale,
    n_samples::Integer
)
    @unpack m_locals, m_global, D_global, D_locals, B_locals, amortize_index = q

    d_global = length(m_global)
    d_local  = length(m_locals |> first)

    us_global = randn(rng, eltype(m_global), d_global, n_samples)
    zs_global = D_global*us_global .+ m_global

    zs_locals = map(amortize_index) do i
        us_local = randn(rng, eltype(m_global), d_local, n_samples)
        (B_locals[i]*us_global + D_locals[i]*us_local) .+ m_locals[i]
    end
    vcat(zs_locals..., zs_global)
end

module ConvexConjugates

export ConjugateCache, ConjugateCache3D
export compute_conjugate!, compute_conjugate_and_map!
export compute_conjugate, compute_conjugate_and_map
export compute_3d_conjugate!, compute_3d_conjugate_and_map!

# ==============================================================================
# 1. The Cache (Pre-allocations for Asymmetric Grids)
# ==============================================================================
struct ConjugateCache
    w::Matrix{Float64}
    I1::Matrix{Int}
    I2::Matrix{Int}
    hull_stack::Vector{Int}
    
    """
        ConjugateCache(N1::Int, N2::Int, M1::Int, M2::Int)
    
    Pre-allocates the memory required for the 2D Legendre transform.
    - `N1, N2`: Dimensions of the input grid (Source X).
    - `M1, M2`: Dimensions of the output grid (Target Y).
    """
    function ConjugateCache(N1::Int, N2::Int, M1::Int, M2::Int)
        new(zeros(Float64, N1, M2), 
            zeros(Int, N1, M2), 
            zeros(Int, M1, M2), 
            zeros(Int, max(N1, N2))) 
    end
end

# The Casual Outer Constructor
ConjugateCache(N::Int) = ConjugateCache(N, N, N, N)
ConjugateCache(N::Int, M::Int) = ConjugateCache(N, N, M, M)

# ==============================================================================
# 2. The Low-Level 1D Workhorses (Fenchel Sweeps)
# ==============================================================================

function fenchel_1d_map!(g_out::AbstractVector{Float64}, argmax_out::AbstractVector{<:Integer},
                         y_target::AbstractVector{Float64}, f_in::AbstractVector{Float64}, 
                         sign_f::Float64, x_source::AbstractVector{Float64}, stack::AbstractVector{Int})
    Nx = length(x_source)
    Ny = length(y_target)
    top = 0
    
    @inbounds for i in 1:Nx
        while top >= 2
            idx1 = stack[top-1]
            idx2 = stack[top]
            dx_src1 = x_source[idx2] - x_source[idx1]
            df1 = sign_f * f_in[idx2] - sign_f * f_in[idx1]
            dx_src2 = x_source[i] - x_source[idx2]
            df2 = sign_f * f_in[i] - sign_f * f_in[idx2]
            if df2 * dx_src1 >= df1 * dx_src2
                break
            end
            top -= 1
        end
        top += 1
        stack[top] = i
    end

    idx = 1
    @inbounds for j in 1:Ny
        while idx < top
            i1 = stack[idx]
            i2 = stack[idx+1]
            if (y_target[j] * x_source[i1] - sign_f * f_in[i1]) > (y_target[j] * x_source[i2] - sign_f * f_in[i2])
                break
            end
            idx += 1
        end
        best_i = stack[idx]
        g_out[j] = y_target[j] * x_source[best_i] - sign_f * f_in[best_i]
        argmax_out[j] = best_i 
    end
end

function fenchel_1d_value!(g_out::AbstractVector{Float64},
                           y_target::AbstractVector{Float64}, f_in::AbstractVector{Float64}, 
                           sign_f::Float64, x_source::AbstractVector{Float64}, stack::AbstractVector{Int})
    Nx = length(x_source)
    Ny = length(y_target)
    top = 0
    
    @inbounds for i in 1:Nx
        while top >= 2
            idx1 = stack[top-1]
            idx2 = stack[top]
            dx_src1 = x_source[idx2] - x_source[idx1]
            df1 = sign_f * f_in[idx2] - sign_f * f_in[idx1]
            dx_src2 = x_source[i] - x_source[idx2]
            df2 = sign_f * f_in[i] - sign_f * f_in[idx2]
            if df2 * dx_src1 >= df1 * dx_src2
                break
            end
            top -= 1
        end
        top += 1
        stack[top] = i
    end

    idx = 1
    @inbounds for j in 1:Ny
        while idx < top
            i1 = stack[idx]
            i2 = stack[idx+1]
            if (y_target[j] * x_source[i1] - sign_f * f_in[i1]) > (y_target[j] * x_source[i2] - sign_f * f_in[i2])
                break
            end
            idx += 1
        end
        g_out[j] = y_target[j] * x_source[stack[idx]] - sign_f * f_in[stack[idx]]
    end
end

# ==============================================================================
# 3. The HPC API (In-Place, Zero Allocation)
# ==============================================================================

"""
    compute_conjugate_and_map!(g, T1, T2, y1, y2, f, x1, x2, cache)

Computes the conjugate `g` (on target grids `y1, y2`) of the function `f` (on source grids `x1, x2`).
"""
function compute_conjugate_and_map!(g::Matrix{Float64}, T1::Matrix{Int}, T2::Matrix{Int}, 
                                    y1::Vector{Float64}, y2::Vector{Float64}, # Target
                                    f::Matrix{Float64}, 
                                    x1::Vector{Float64}, x2::Vector{Float64}, # Source
                                    cache::ConjugateCache)
    N1, N2 = size(f)
    M1, M2 = size(g)
    
    # Sweep 1: Maximize over x2 (maps x2 -> y2)
    for i in 1:N1
        @views fenchel_1d_map!(cache.w[i, :], cache.I1[i, :], y2, f[i, :], 1.0, x2, cache.hull_stack)
    end
    
    # Sweep 2: Maximize over x1 (maps x1 -> y1)
    for j in 1:M2
        @views fenchel_1d_map!(g[:, j], cache.I2[:, j], y1, cache.w[:, j], -1.0, x1, cache.hull_stack)
    end
    
    # Reconstruct the exact 2D map
    @inbounds for j in 1:M2, i in 1:M1
        T1[i, j] = cache.I2[i, j]
        T2[i, j] = cache.I1[T1[i, j], j]
    end
end

function compute_conjugate_and_map!(g::Matrix{Float64}, T1::Matrix{Int}, T2::Matrix{Int}, 
                                    y::Vector{Float64}, # Target Grid
                                    f::Matrix{Float64}, 
                                    x::Vector{Float64}, # Source Grid
                                    cache::ConjugateCache)
    
    return compute_conjugate_and_map!(g, T1, T2, y, y, f, x, x, cache)
end

"""
    compute_conjugate!(g, y1, y2, f, x1, x2, cache)

Strictly computes the conjugate `g` (on target grids `y1, y2`) of the function `f` (on source grids `x1, x2`).
"""
function compute_conjugate!(g::Matrix{Float64}, 
                            y1::Vector{Float64}, y2::Vector{Float64}, # Target
                            f::Matrix{Float64}, 
                            x1::Vector{Float64}, x2::Vector{Float64}, # Source
                            cache::ConjugateCache)
    N1, N2 = size(f)
    M1, M2 = size(g)
    
    # Sweep 1: Maximize over x2
    for i in 1:N1
        @views fenchel_1d_value!(cache.w[i, :], y2, f[i, :], 1.0, x2, cache.hull_stack)
    end
    
    # Sweep 2: Maximize over x1
    for j in 1:M2
        @views fenchel_1d_value!(g[:, j], y1, cache.w[:, j], -1.0, x1, cache.hull_stack)
    end
end

function compute_conjugate!(g::Matrix{Float64}, y::Vector{Float64}, 
                            f::Matrix{Float64}, x::Vector{Float64}, 
                            cache::ConjugateCache)
    
    return compute_conjugate!(g, y, y, f, x, x, cache)
end


# ==============================================================================
# 4. The Casual API (Allocating, User-Friendly)
# ==============================================================================

"""
    compute_conjugate_and_map(f, x1, x2, y1, y2)
"""
function compute_conjugate_and_map(f::Matrix{Float64}, 
                                   x1::Vector{Float64}, x2::Vector{Float64}, # Source
                                   y1::Vector{Float64}, y2::Vector{Float64}) # Target
    N1, N2 = size(f)
    M1 = length(y1)
    M2 = length(y2)
    
    if length(x1) != N1 || length(x2) != N2
        throw(ArgumentError("Source grid vectors x1, x2 must match the dimensions of matrix f."))
    end
    
    g = zeros(Float64, M1, M2)
    T1 = zeros(Int, M1, M2)
    T2 = zeros(Int, M1, M2)
    cache = ConjugateCache(N1, N2, M1, M2)
    
    compute_conjugate_and_map!(g, T1, T2, y1, y2, f, x1, x2, cache)
    
    return g, T1, T2
end

function compute_conjugate_and_map(f::Matrix{Float64}, 
                                   x::Vector{Float64}, # Source Grid
                                   y::Vector{Float64}) # Target Grid
    
    return compute_conjugate_and_map(f, x, x, y, y)
end

"""
    compute_conjugate(f, x1, x2, y1, y2)
"""
function compute_conjugate(f::Matrix{Float64}, 
                           x1::Vector{Float64}, x2::Vector{Float64}, # Source
                           y1::Vector{Float64}, y2::Vector{Float64}) # Target
    N1, N2 = size(f)
    M1 = length(y1)
    M2 = length(y2)
    
    if length(x1) != N1 || length(x2) != N2
        throw(ArgumentError("Source grid vectors x1, x2 must match the dimensions of matrix f."))
    end
    
    g = zeros(Float64, M1, M2)
    cache = ConjugateCache(N1, N2, M1, M2)
    
    compute_conjugate!(g, y1, y2, f, x1, x2, cache)
    
    return g
end

function compute_conjugate(f::Matrix{Float64}, 
                           x::Vector{Float64}, # Source Grid
                           y::Vector{Float64}) # Target Grid
    
    return compute_conjugate(f, x, x, y, y)
end

# ==============================================================================
# 5. Three Dimensions
# ==============================================================================
# The conjugate g(y) = max_x {⟨y,x⟩ − f(x)} factorizes into one 1D Fenchel sweep
# per axis, exactly as in 2D — the only new content is that recovering the
# *argmax point* now needs a three-level index composition instead of two.
#
# Sweep 3 (over x3):  w1[i1,i2,j3] = max_{i3} { y3[j3]x3[i3] − f[i1,i2,i3] }
# Sweep 2 (over x2):  w2[i1,j2,j3] = max_{i2} { y2[j2]x2[i2] + w1[i1,i2,j3] }
# Sweep 1 (over x1):  g[j1,j2,j3]  = max_{i1} { y1[j1]x1[i1] + w2[i1,j2,j3] }
#
# so `sign_f` runs 1, −1, −1: the first sweep subtracts f, and each later sweep
# adds back the (already negated) partial maximum.  Recording the winning index of
# each sweep as A3, A2, A1 and unwinding in the reverse order gives the exact
# maximizing source node — the 3D analogue of `T2[i,j] = I1[T1[i,j], j]`:
#
#     T1 = A1[j1,j2,j3];  T2 = A2[T1,j2,j3];  T3 = A3[T1,T2,j3]
#
# Each level is looked up at the indices the *later* sweeps already committed to,
# which is what makes the result an exact argmax of the full 3D problem and not a
# per-axis approximation.  It is verified against brute force in
# `test/test_conjugate_3d.jl`.
#
# Two 3D-specific implementation notes:
#
#   * **Int32 indices.**  Three N³ index arrays cost 408 MB at N=257 in Int64 and
#     204 MB in Int32.  A grid axis will not exceed 2³¹ nodes.
#   * **No `threadid()`.**  Per-thread scratch is handed to tasks explicitly
#     (`_tforeach` below), never indexed by `Threads.threadid()`.  Under the
#     `:dynamic` scheduling that `@threads` has defaulted to since Julia 1.8 a task
#     may resume on a different thread, so two tasks can collide on one buffer;
#     `RC3D/RC3D.jl` has that pattern and it is a latent race, not a style point.

"1D scratch for one sweep task: pack buffer, result buffer, argmax buffer, hull stack."
struct SweepBuffers
    fin::Vector{Float64}
    fout::Vector{Float64}
    aout::Vector{Int32}
    stack::Vector{Int}
end
SweepBuffers(nsrc::Int, ntgt::Int) =
    SweepBuffers(zeros(Float64, nsrc), zeros(Float64, ntgt), zeros(Int32, ntgt), zeros(Int, nsrc))

"""
    ConjugateCache3D(N, M = N; with_map = false, ntasks = Threads.nthreads())

Workspace for the 3D Legendre transform of a function on an `N×N×N` source grid
onto an `M×M×M` target grid.

    ConjugateCache3D(N1, N2, N3, M1, M2, M3; with_map = false, ntasks = Threads.nthreads())

`with_map = true` additionally allocates the three Int32 index arrays that
[`compute_3d_conjugate_and_map!`](@ref) needs; leave it `false` for the
value-only [`compute_3d_conjugate!`](@ref) and pay nothing for them.  At
`N = M = 257` the value cache is 272 MB and the index arrays add 204 MB.
"""
struct ConjugateCache3D
    N::Int
    M::Int
    w1::Array{Float64,3}      # (N, N, M) — after the x3 sweep
    w2::Array{Float64,3}      # (N, M, M) — after the x2 sweep
    A3::Array{Int32,3}        # (N, N, M) — argmax over x3
    A2::Array{Int32,3}        # (N, M, M) — argmax over x2
    A1::Array{Int32,3}        # (M, M, M) — argmax over x1
    bufs::Vector{SweepBuffers}
end

function ConjugateCache3D(N1::Int, N2::Int, N3::Int, M1::Int, M2::Int, M3::Int;
                          with_map::Bool = false, ntasks::Int = Threads.nthreads())
    ntasks ≥ 1 || throw(ArgumentError("ntasks must be ≥ 1, got $ntasks"))
    z3(dims...) = with_map ? zeros(Int32, dims...) : zeros(Int32, 0, 0, 0)
    nsrc = max(N1, N2, N3)          # the pack buffer / hull stack serve every sweep
    ntgt = max(M1, M2, M3)
    ConjugateCache3D(nsrc, ntgt,
                     zeros(Float64, N1, N2, M3), zeros(Float64, N1, M2, M3),
                     z3(N1, N2, M3), z3(N1, M2, M3), z3(M1, M2, M3),
                     [SweepBuffers(nsrc, ntgt) for _ in 1:ntasks])
end

ConjugateCache3D(N::Int, M::Int = N; kw...) = ConjugateCache3D(N, N, N, M, M, M; kw...)

"Run `f(range, buffers)` over a contiguous partition of `1:n`, one task per buffer set."
function _tforeach(f::F, n::Int, bufs::Vector{SweepBuffers}) where {F}
    nt = min(length(bufs), n)
    if nt ≤ 1
        f(1:n, bufs[1])
        return nothing
    end
    tasks = Vector{Task}(undef, nt)
    lo = 1
    for t in 1:nt
        hi = lo + cld(n - lo + 1, nt - t + 1) - 1
        let rng = lo:hi, b = bufs[t]
            tasks[t] = Threads.@spawn f(rng, b)
        end
        lo = hi + 1
    end
    foreach(wait, tasks)
    return nothing
end

"""
    compute_3d_conjugate_and_map!(g, T1, T2, T3, y1, y2, y3, f, x1, x2, x3, cache)
    compute_3d_conjugate_and_map!(g, T1, T2, T3, y, f, x, cache)

The conjugate `g[j] = max_i {⟨y[j], x[i]⟩ − f[i]}` on the target grid, **together
with the exact maximizing source index** in `T1, T2, T3` (indices into `x1, x2, x3`).

This is the 3D counterpart of [`compute_conjugate_and_map!`](@ref), and the exact
argmax is what the screening solver needs for three separate things: the
warm-started `:argmax` finishing window, the threshold-free combinatorial
bunching classifiers (Ω₀ = "the map lands on the ∅ node", and the isochoice set
`T⁻¹(y)` = the Laguerre cell of `y`), and the Laguerre cross-check on the profit.

`cache` must be built with `with_map = true`.

Tie-breaking is inherited from the 1D sweep and is worth knowing: only hull
vertices are candidates, and the backward scan advances while the next candidate
is *no worse*, so a tie resolves to the **last** hull vertex attaining the
maximum.  It is deterministic given `f` and the grids, but it is not symmetric
under a coordinate swap, so do not read a diagonal asymmetry of `T` as a defect
of the solver without checking whether it is only this rule (the 2D analysis of
tie-splitting is `reviews/phase2.9-e3-symmetry.md`).
"""
function compute_3d_conjugate_and_map!(g::Array{Float64,3},
                                       T1::Array{<:Integer,3}, T2::Array{<:Integer,3},
                                       T3::Array{<:Integer,3},
                                       y1::Vector{Float64}, y2::Vector{Float64},
                                       y3::Vector{Float64},            # Target
                                       f::Array{Float64,3},
                                       x1::Vector{Float64}, x2::Vector{Float64},
                                       x3::Vector{Float64},            # Source
                                       cache::ConjugateCache3D)
    N1, N2, N3 = size(f)
    M1, M2, M3 = size(g)
    _check3d(cache, N1, N2, N3, M1, M2, M3, x1, x2, x3, y1, y2, y3; with_map = true)
    w1, w2, A3, A2, A1 = cache.w1, cache.w2, cache.A3, cache.A2, cache.A1

    # --- Sweep 3: maximize over x3.  Strided in memory, so pack into the buffer.
    _tforeach(N2, cache.bufs) do rng, b
        @inbounds for i2 in rng, i1 in 1:N1
            for i3 in 1:N3; b.fin[i3] = f[i1, i2, i3]; end
            fenchel_1d_map!(b.fout, b.aout, y3, view(b.fin, 1:N3), 1.0, x3, b.stack)
            for j3 in 1:M3
                w1[i1, i2, j3] = b.fout[j3]
                A3[i1, i2, j3] = b.aout[j3]
            end
        end
    end

    # --- Sweep 2: maximize over x2.
    _tforeach(M3, cache.bufs) do rng, b
        @inbounds for j3 in rng, i1 in 1:N1
            for i2 in 1:N2; b.fin[i2] = w1[i1, i2, j3]; end
            fenchel_1d_map!(b.fout, b.aout, y2, view(b.fin, 1:N2), -1.0, x2, b.stack)
            for j2 in 1:M2
                w2[i1, j2, j3] = b.fout[j2]
                A2[i1, j2, j3] = b.aout[j2]
            end
        end
    end

    # --- Sweep 1: maximize over x1.  Contiguous in the leading index.
    _tforeach(M3, cache.bufs) do rng, b
        @inbounds for j3 in rng, j2 in 1:M2
            for i1 in 1:N1; b.fin[i1] = w2[i1, j2, j3]; end
            fenchel_1d_map!(b.fout, b.aout, y1, view(b.fin, 1:N1), -1.0, x1, b.stack)
            for j1 in 1:M1
                g[j1, j2, j3]  = b.fout[j1]
                A1[j1, j2, j3] = b.aout[j1]
            end
        end
    end

    # --- Unwind the composition.  Each level is read at the indices the later
    #     sweeps committed to, which is what makes this the exact 3D argmax.
    @inbounds for j3 in 1:M3, j2 in 1:M2, j1 in 1:M1
        t1 = A1[j1, j2, j3]
        t2 = A2[t1, j2, j3]
        T1[j1, j2, j3] = t1
        T2[j1, j2, j3] = t2
        T3[j1, j2, j3] = A3[t1, t2, j3]
    end
    return nothing
end

function compute_3d_conjugate_and_map!(g::Array{Float64,3},
                                       T1::Array{<:Integer,3}, T2::Array{<:Integer,3},
                                       T3::Array{<:Integer,3},
                                       y::Vector{Float64},     # Target grid
                                       f::Array{Float64,3},
                                       x::Vector{Float64},     # Source grid
                                       cache::ConjugateCache3D)
    return compute_3d_conjugate_and_map!(g, T1, T2, T3, y, y, y, f, x, x, x, cache)
end

"""
    compute_3d_conjugate!(g, y1, y2, y3, f, x1, x2, x3, cache)
    compute_3d_conjugate!(g, y, f, x, cache)

The conjugate only, without the argmax.  Same three sweeps as
[`compute_3d_conjugate_and_map!`](@ref) but skipping the index bookkeeping, and
the `cache` need not carry the index arrays.  This is the hot path of the ascent,
where `T_u` is read off as `∇u` rather than as the argmax.
"""
function compute_3d_conjugate!(g::Array{Float64,3},
                               y1::Vector{Float64}, y2::Vector{Float64},
                               y3::Vector{Float64},              # Target
                               f::Array{Float64,3},
                               x1::Vector{Float64}, x2::Vector{Float64},
                               x3::Vector{Float64},              # Source
                               cache::ConjugateCache3D)
    N1, N2, N3 = size(f)
    M1, M2, M3 = size(g)
    _check3d(cache, N1, N2, N3, M1, M2, M3, x1, x2, x3, y1, y2, y3; with_map = false)
    w1, w2 = cache.w1, cache.w2

    _tforeach(N2, cache.bufs) do rng, b
        @inbounds for i2 in rng, i1 in 1:N1
            for i3 in 1:N3; b.fin[i3] = f[i1, i2, i3]; end
            fenchel_1d_value!(b.fout, y3, view(b.fin, 1:N3), 1.0, x3, b.stack)
            for j3 in 1:M3; w1[i1, i2, j3] = b.fout[j3]; end
        end
    end

    _tforeach(M3, cache.bufs) do rng, b
        @inbounds for j3 in rng, i1 in 1:N1
            for i2 in 1:N2; b.fin[i2] = w1[i1, i2, j3]; end
            fenchel_1d_value!(b.fout, y2, view(b.fin, 1:N2), -1.0, x2, b.stack)
            for j2 in 1:M2; w2[i1, j2, j3] = b.fout[j2]; end
        end
    end

    _tforeach(M3, cache.bufs) do rng, b
        @inbounds for j3 in rng, j2 in 1:M2
            for i1 in 1:N1; b.fin[i1] = w2[i1, j2, j3]; end
            fenchel_1d_value!(b.fout, y1, view(b.fin, 1:N1), -1.0, x1, b.stack)
            for j1 in 1:M1; g[j1, j2, j3] = b.fout[j1]; end
        end
    end
    return nothing
end

function compute_3d_conjugate!(g::Array{Float64,3}, y::Vector{Float64},
                               f::Array{Float64,3}, x::Vector{Float64},
                               cache::ConjugateCache3D)
    return compute_3d_conjugate!(g, y, y, y, f, x, x, x, cache)
end

function _check3d(cache::ConjugateCache3D, N1, N2, N3, M1, M2, M3,
                  x1, x2, x3, y1, y2, y3; with_map::Bool)
    (length(x1), length(x2), length(x3)) == (N1, N2, N3) || throw(ArgumentError(
        "source grids $(length.((x1,x2,x3))) do not match size(f) = $((N1,N2,N3))."))
    (length(y1), length(y2), length(y3)) == (M1, M2, M3) || throw(ArgumentError(
        "target grids $(length.((y1,y2,y3))) do not match size(g) = $((M1,M2,M3))."))
    size(cache.w1) == (N1, N2, M3) && size(cache.w2) == (N1, M2, M3) || throw(ArgumentError(
        "cache was built for N=$(cache.N), M=$(cache.M); this call needs " *
        "source $((N1,N2,N3)) and target $((M1,M2,M3))."))
    if with_map && isempty(cache.A1)
        throw(ArgumentError("this cache carries no index arrays; rebuild it with " *
                            "ConjugateCache3D(N, M; with_map = true)."))
    end
    return nothing
end

end # module ConvexConjugates
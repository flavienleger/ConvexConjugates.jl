module ConvexConjugates

export ConjugateCache
export compute_conjugate!, compute_conjugate_and_map!
export compute_conjugate, compute_conjugate_and_map

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
    - `N1, N2`: Dimensions of the input grid (Source).
    - `M1, M2`: Dimensions of the output grid (Target).
    """
    function ConjugateCache(N1::Int, N2::Int, M1::Int, M2::Int)
        new(zeros(Float64, N1, M2), 
            zeros(Int, N1, M2), 
            zeros(Int, M1, M2), 
            zeros(Int, max(N1, N2))) 
    end
end

# ==============================================================================
# 2. The Low-Level 1D Workhorses (Fenchel Sweeps)
# ==============================================================================

function fenchel_1d_map!(g_out::AbstractVector{Float64}, argmax_out::AbstractVector{Int},
                         x_target::AbstractVector{Float64}, f_in::AbstractVector{Float64}, 
                         sign_f::Float64, y_source::AbstractVector{Float64}, stack::AbstractVector{Int})
    Ny = length(y_source)
    Nx = length(x_target)
    top = 0
    
    @inbounds for i in 1:Ny
        while top >= 2
            idx1 = stack[top-1]
            idx2 = stack[top]
            dy_src1 = y_source[idx2] - y_source[idx1]
            df1 = sign_f * f_in[idx2] - sign_f * f_in[idx1]
            dy_src2 = y_source[i] - y_source[idx2]
            df2 = sign_f * f_in[i] - sign_f * f_in[idx2]
            if df2 * dy_src1 >= df1 * dy_src2
                break
            end
            top -= 1
        end
        top += 1
        stack[top] = i
    end

    idx = 1
    @inbounds for j in 1:Nx
        while idx < top
            i1 = stack[idx]
            i2 = stack[idx+1]
            if (x_target[j] * y_source[i1] - sign_f * f_in[i1]) > (x_target[j] * y_source[i2] - sign_f * f_in[i2])
                break
            end
            idx += 1
        end
        best_i = stack[idx]
        g_out[j] = x_target[j] * y_source[best_i] - sign_f * f_in[best_i]
        argmax_out[j] = best_i 
    end
end

function fenchel_1d_value!(g_out::AbstractVector{Float64},
                           x_target::AbstractVector{Float64}, f_in::AbstractVector{Float64}, 
                           sign_f::Float64, y_source::AbstractVector{Float64}, stack::AbstractVector{Int})
    Ny = length(y_source)
    Nx = length(x_target)
    top = 0
    
    @inbounds for i in 1:Ny
        while top >= 2
            idx1 = stack[top-1]
            idx2 = stack[top]
            dy_src1 = y_source[idx2] - y_source[idx1]
            df1 = sign_f * f_in[idx2] - sign_f * f_in[idx1]
            dy_src2 = y_source[i] - y_source[idx2]
            df2 = sign_f * f_in[i] - sign_f * f_in[idx2]
            if df2 * dy_src1 >= df1 * dy_src2
                break
            end
            top -= 1
        end
        top += 1
        stack[top] = i
    end

    idx = 1
    @inbounds for j in 1:Nx
        while idx < top
            i1 = stack[idx]
            i2 = stack[idx+1]
            if (x_target[j] * y_source[i1] - sign_f * f_in[i1]) > (x_target[j] * y_source[i2] - sign_f * f_in[i2])
                break
            end
            idx += 1
        end
        g_out[j] = x_target[j] * y_source[stack[idx]] - sign_f * f_in[stack[idx]]
    end
end

# ==============================================================================
# 3. The HPC API (In-Place, Zero Allocation)
# ==============================================================================

function compute_conjugate_and_map!(g::Matrix{Float64}, T1::Matrix{Int}, T2::Matrix{Int}, 
                                    f::Matrix{Float64}, 
                                    y1::Vector{Float64}, y2::Vector{Float64}, 
                                    x1::Vector{Float64}, x2::Vector{Float64},
                                    cache::ConjugateCache)
    N1, N2 = size(f)
    M1, M2 = size(g)
    
    # Sweep 1: Maximize over y2 (maps y2 -> x2)
    for i in 1:N1
        @views fenchel_1d_map!(cache.w[i, :], cache.I1[i, :], x2, f[i, :], 1.0, y2, cache.hull_stack)
    end
    
    # Sweep 2: Maximize over y1 (maps y1 -> x1)
    for j in 1:M2
        @views fenchel_1d_map!(g[:, j], cache.I2[:, j], x1, cache.w[:, j], -1.0, y1, cache.hull_stack)
    end
    
    # Reconstruct the exact 2D map
    @inbounds for j in 1:M2, i in 1:M1
        T1[i, j] = cache.I2[i, j]
        T2[i, j] = cache.I1[T1[i, j], j]
    end
end

function compute_conjugate!(g::Matrix{Float64}, 
                            f::Matrix{Float64}, 
                            y1::Vector{Float64}, y2::Vector{Float64}, 
                            x1::Vector{Float64}, x2::Vector{Float64},
                            cache::ConjugateCache)
    N1, N2 = size(f)
    M1, M2 = size(g)
    
    # Sweep 1: Maximize over y2
    for i in 1:N1
        @views fenchel_1d_value!(cache.w[i, :], x2, f[i, :], 1.0, y2, cache.hull_stack)
    end
    
    # Sweep 2: Maximize over y1
    for j in 1:M2
        @views fenchel_1d_value!(g[:, j], x1, cache.w[:, j], -1.0, y1, cache.hull_stack)
    end
end

# ==============================================================================
# 4. The Casual API (Allocating, User-Friendly)
# ==============================================================================

function compute_conjugate_and_map(f::Matrix{Float64}, 
                                   y1::Vector{Float64}, y2::Vector{Float64}, 
                                   x1::Vector{Float64}, x2::Vector{Float64})
    N1, N2 = size(f)
    M1 = length(x1)
    M2 = length(x2)
    
    if length(y1) != N1 || length(y2) != N2
        throw(ArgumentError("Source grid vectors y1, y2 must match the dimensions of matrix f."))
    end
    
    g = zeros(Float64, M1, M2)
    T1 = zeros(Int, M1, M2)
    T2 = zeros(Int, M1, M2)
    cache = ConjugateCache(N1, N2, M1, M2)
    
    compute_conjugate_and_map!(g, T1, T2, f, y1, y2, x1, x2, cache)
    
    return g, T1, T2
end

function compute_conjugate(f::Matrix{Float64}, 
                           y1::Vector{Float64}, y2::Vector{Float64}, 
                           x1::Vector{Float64}, x2::Vector{Float64})
    N1, N2 = size(f)
    M1 = length(x1)
    M2 = length(x2)
    
    if length(y1) != N1 || length(y2) != N2
        throw(ArgumentError("Source grid vectors y1, y2 must match the dimensions of matrix f."))
    end
    
    g = zeros(Float64, M1, M2)
    cache = ConjugateCache(N1, N2, M1, M2)
    
    compute_conjugate!(g, f, y1, y2, x1, x2, cache)
    
    return g
end

end # module ConvexConjugates
# ==============================================================================
# ConvexConjugates.jl - 2D Asymmetric HPC Example
# ==============================================================================
# Run this from the terminal using: julia --project=@. examples/hpc_2d.jl
# ==============================================================================

using ConvexConjugates
using Printf

function run_hpc_example()
    println("Setting up 2D Asymmetric Fenchel Conjugate Example...")
    
    # 1. Initialization parameters for asymmetric grids
    # Primal grid dimensions (Input f)
    N1 = 300 
    N2 = 150
    
    # Dual grid dimensions (Output g)
    M1 = 200 
    M2 = 250
    
    # 2. Define the exact 1D coordinate grids
    # Primal grid (y1, y2)
    y1 = collect(range(-2.0, 2.0, length=N1))
    y2 = collect(range(-2.0, 2.0, length=N2))
    
    # Dual grid (x1, x2)
    x1 = collect(range(-2.0, 2.0, length=M1))
    x2 = collect(range(-2.0, 2.0, length=M2))
    
    # 3. Setup the Primal State
    # We use a standard parabola: f(y) = 0.5 * |y|^2
    # The continuous conjugate of 0.5*|y|^2 is exactly 0.5*|x|^2
    f = zeros(Float64, N1, N2)
    for j in 1:N2, i in 1:N1
        f[i, j] = 0.5 * (y1[i]^2 + y2[j]^2)
    end
    
    # ==========================================================================
    # THE HPC WAY: Pre-allocate EVERYTHING outside the hot loop
    # ==========================================================================
    g = zeros(Float64, M1, M2)
    T1 = zeros(Int, M1, M2)
    T2 = zeros(Int, M1, M2)
    
    # Allocate the heavy memory workspaces EXACTLY ONCE with the grid dimensions
    cache = ConjugateCache(N1, N2, M1, M2)
    
    println("Starting zero-allocation hot loop...")
    t0 = time()
    
    # Simulate a PDE or optimization gradient flow
    iters = 1000
    for step in 1:iters
        # We pass the pre-allocated arrays, grids, and the cache. 
        # This function call allocates exactly 0 bytes of memory.
        compute_conjugate_and_map!(g, T1, T2, f, y1, y2, x1, x2, cache)
        
        # ... (In a real solver, you would update `f` here based on `g` and `T1/T2`) ...
    end
    elapsed = time() - t0
    
    # ==========================================================================
    # Verification & Diagnostics
    # ==========================================================================
    @printf("Executed %d asymmetric 2D transforms in %.4f seconds.\n", iters, elapsed)
    
    # Verify the math: check a point near the center of the dual grid
    center_i = M1 ÷ 2
    center_j = M2 ÷ 2
    test_x1 = x1[center_i]
    test_x2 = x2[center_j]
    
    expected_g = 0.5 * (test_x1^2 + test_x2^2)
    actual_g = g[center_i, center_j]
    
    println("\n--- Math Check ---")
    @printf("Evaluating at X = (%.2f, %.2f)\n", test_x1, test_x2)
    @printf("Analytical g(x): %.6f\n", expected_g)
    @printf("Numerical g(x):  %.6f\n", actual_g)
    
    if abs(expected_g - actual_g) < 0.05
        println("✅ Transform matches analytical geometry on asymmetric grids!")
    else
        println("❌ Transform drifted from expected geometry.")
    end
end

# Execute the script
run_hpc_example()
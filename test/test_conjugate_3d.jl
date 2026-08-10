# Validation gate for the 3D Legendre transform (Phase 4.1).
#
# The 3D conjugate is three 1D sweeps and the map is a three-level index
# composition, so the thing that can silently go wrong is the *composition*: a
# per-axis argmax that is not the argmax of the joint problem still produces a
# plausible-looking map.  The only honest check is brute force,
#
#     g[j] = max_i {⟨y[j], x[i]⟩ − f[i]},   T[j] = argmax_i {…}
#
# evaluated over the full N³ × M³ product on grids small enough to afford it.
#
#   * `T` must agree **exactly** wherever the maximizer is unique, and the node
#     the sweeps choose must attain the brute-force maximum **bitwise**
#     (`suboptimality == 0`).  That is the load-bearing assertion: a per-axis
#     argmax that is not the joint argmax fails it, and unlike a value
#     comparison it has no round-off escape hatch.  Where the maximizer is not
#     unique, only attainment is required — the sweep's tie rule is documented
#     but is not something a test should freeze.
#   * `g` agrees only to machine epsilon, not bitwise, and that is expected
#     rather than a defect: brute force evaluates `y1x1 + y2x2 + y3x3 − f` in one
#     expression while the sweeps accumulate it across three sweeps, so the two
#     round differently *at the same node*.  (Measured: ≤9e−16.)  The value-only
#     path, which does the same arithmetic in the same order, must be bitwise
#     equal to the map path.
#
# Run:  julia --project=. -t 4 test/test_conjugate_3d.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using ConvexConjugates
using Printf, Random, Test

"Brute force over the full source grid: value, and the set of maximizers."
function brute(f, x1, x2, x3, y1, y2, y3)
    M1, M2, M3 = length(y1), length(y2), length(y3)
    N1, N2, N3 = size(f)
    g = zeros(M1, M2, M3)
    T = Array{NTuple{3,Int},3}(undef, M1, M2, M3)
    ties = falses(M1, M2, M3)
    for j3 in 1:M3, j2 in 1:M2, j1 in 1:M1
        best = -Inf; barg = (0, 0, 0); nbest = 0
        for i3 in 1:N3, i2 in 1:N2, i1 in 1:N1
            val = y1[j1]*x1[i1] + y2[j2]*x2[i2] + y3[j3]*x3[i3] - f[i1, i2, i3]
            if val > best
                best = val; barg = (i1, i2, i3); nbest = 1
            elseif val == best
                nbest += 1
            end
        end
        g[j1, j2, j3] = best; T[j1, j2, j3] = barg; ties[j1, j2, j3] = nbest > 1
    end
    return g, T, ties
end

"Value of the objective at the source node the sweep chose."
attained(f, x1, x2, x3, y1, y2, y3, j, t) =
    y1[j[1]]*x1[t[1]] + y2[j[2]]*x2[t[2]] + y3[j[3]]*x3[t[3]] - f[t[1], t[2], t[3]]

function check(name, f, x1, x2, x3, y1, y2, y3)
    N1, N2, N3 = size(f); M1, M2, M3 = length(y1), length(y2), length(y3)
    cache = ConjugateCache3D(N1, N2, N3, M1, M2, M3; with_map = true)
    g = zeros(M1, M2, M3)
    T1 = zeros(Int32, M1, M2, M3); T2 = similar(T1); T3 = similar(T1)
    compute_3d_conjugate_and_map!(g, T1, T2, T3, y1, y2, y3, f, x1, x2, x3, cache)

    # the value-only path must return bit-identical values
    gv = zeros(M1, M2, M3)
    compute_3d_conjugate!(gv, y1, y2, y3, f, x1, x2, x3,
                          ConjugateCache3D(N1, N2, N3, M1, M2, M3))

    gb, Tb, ties = brute(f, x1, x2, x3, y1, y2, y3)

    dval  = maximum(abs, g .- gb)
    dvv   = maximum(abs, g .- gv)
    nmis  = 0        # argmax differs where the maximizer is unique
    dsub  = 0.0      # how far the chosen node falls short of the true max
    for j3 in 1:M3, j2 in 1:M2, j1 in 1:M1
        t = (Int(T1[j1,j2,j3]), Int(T2[j1,j2,j3]), Int(T3[j1,j2,j3]))
        dsub = max(dsub, gb[j1,j2,j3] -
                   attained(f, x1, x2, x3, y1, y2, y3, (j1,j2,j3), t))
        if !ties[j1,j2,j3] && t != Tb[j1,j2,j3]
            nmis += 1
        end
    end
    ntie = count(ties)
    scale = max(1.0, maximum(abs, gb))
    @printf("%-30s |Δg| = %8.1e  |Δg_valueonly| = %8.1e  argmax misses = %4d/%-5d  suboptimality = %8.1e  (%d tied)\n",
            name, dval, dvv, nmis, M1*M2*M3, dsub, ntie)
    @test dsub == 0.0                      # the chosen node attains the max, bitwise
    @test nmis == 0                        # and it is *the* maximizer where unique
    @test dvv  == 0.0                      # value-only path is bitwise the map path
    @test dval ≤ 32 * eps(scale)           # sweeps vs one-shot: rounding only
    return nothing
end

Random.seed!(20260810)

@testset "3D conjugate + map vs brute force" begin
    # (1) The RC setting itself: source = product grid Y (vertex-centered),
    #     target = consumer grid X (cell-centered), f = the warm start |y|²/2.
    for (N, a) in ((9, 1.0), (17, 1.0), (13, 2.0))
        h  = (a + 1) / (N - 1)
        y  = [(k - 1) * h for k in 1:N]
        dx = 1 / N
        X  = [a + (i - 0.5) * dx for i in 1:N]
        f  = [0.5 * (y[i]^2 + y[j]^2 + y[k]^2) for i in 1:N, j in 1:N, k in 1:N]
        check("RC warm start N=$N a=$a", f, y, y, y, X, X, X)
    end

    # (2) Non-convex input.  The transform must still return the conjugate — i.e.
    #     it sees only the convex hull — and the map must point at a hull vertex.
    N = 11; M = 13
    x = collect(range(0.0, 2.0, length = N)); y = collect(range(1.0, 2.0, length = M))
    f = [sin(4x[i]) + cos(3x[j]) + x[k]^2 - 0.4x[i]*x[k] for i in 1:N, j in 1:N, k in 1:N]
    check("non-convex smooth", f, x, x, x, y, y, y)

    # (3) Pure noise — no structure at all for the hull scan to lean on.
    f = randn(N, N, N)
    check("random", f, x, x, x, y, y, y)

    # (4) Anisotropic grids and a rectangular target, to exercise the index
    #     bookkeeping where N1 ≠ N2 ≠ N3 and M ≠ N (a transposed A2/A3 lookup
    #     would go undetected on a cube).
    x1 = collect(range(0.0, 1.0, length = 7))
    x2 = collect(range(-0.5, 1.5, length = 9))
    x3 = collect(range(0.0, 3.0, length = 11))
    y1 = collect(range(0.5, 1.5, length = 5))
    y2 = collect(range(0.0, 1.0, length = 6))
    y3 = collect(range(-1.0, 1.0, length = 4))
    fa = [0.3 * (x1[i]^2 + x2[j]^2 + x3[k]^2) + 0.2randn() for i in 1:7, j in 1:9, k in 1:11]
    check("anisotropic 7×9×11 → 5×6×4", fa, x1, x2, x3, y1, y2, y3)

    # (5) Flat function: every node ties on the ⟨y,x⟩ term alone.  Nothing may
    #     crash, and every chosen node must attain the max.
    check("flat", zeros(N, N, N), x, x, x, y, y, y)

    # (6) Degenerate 2-node axis, the smallest hull the scan can be handed.
    xs = [0.0, 1.0]
    check("2-node axes", randn(2, 2, 2), xs, xs, xs, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
end

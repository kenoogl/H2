using Printf
using LinearAlgebra
using Random

include("heat3ds.jl")

"""
係数行列Aの対称性をテストする
ランダムベクトルx, yに対して (Ax, y) と (x, Ay) を比較
"""
function test_matrix_symmetry(wk, Δh, Δt, ZC, ΔZ, par; num_tests=5)
    println("=== Testing Matrix Symmetry ===")

    SZ = size(wk.θ)
    z_st = 2
    z_ed = SZ[3] - 1

    # 計算領域のサイズ
    n_inner = (SZ[1]-2) * (SZ[2]-2) * (z_ed - z_st + 1)
    println("Inner grid size: $n_inner points")

    is_symmetric = true

    for test_num in 1:num_tests
        # ランダムベクトル x, y を生成
        x = zeros(Float64, SZ[1], SZ[2], SZ[3])
        y = zeros(Float64, SZ[1], SZ[2], SZ[3])

        Random.seed!(test_num * 1234)
        for k in z_st:z_ed, j in 2:SZ[2]-1, i in 2:SZ[1]-1
            x[i,j,k] = randn()
            y[i,j,k] = randn()
        end

        # Ax を計算
        Ax = zeros(Float64, SZ[1], SZ[2], SZ[3])
        NonUniform.CalcAX!(Ax, x, Δh, Δt, wk.λ, wk.mask, wk.ρ, wk.cp, ZC, ΔZ, par)

        # Ay を計算
        Ay = zeros(Float64, SZ[1], SZ[2], SZ[3])
        NonUniform.CalcAX!(Ay, y, Δh, Δt, wk.λ, wk.mask, wk.ρ, wk.cp, ZC, ΔZ, par)

        # (Ax, y) を計算
        dot_Ax_y = 0.0
        for k in z_st:z_ed, j in 2:SZ[2]-1, i in 2:SZ[1]-1
            dot_Ax_y += Ax[i,j,k] * y[i,j,k]
        end

        # (x, Ay) を計算
        dot_x_Ay = 0.0
        for k in z_st:z_ed, j in 2:SZ[2]-1, i in 2:SZ[1]-1
            dot_x_Ay += x[i,j,k] * Ay[i,j,k]
        end

        # 相対誤差を計算
        abs_diff = abs(dot_Ax_y - dot_x_Ay)
        scale = max(abs(dot_Ax_y), abs(dot_x_Ay))
        rel_error = scale > 0 ? abs_diff / scale : abs_diff

        @printf("Test %d: (Ax,y) = %24.16e, (x,Ay) = %24.16e\n", test_num, dot_Ax_y, dot_x_Ay)
        @printf("         Difference = %24.16e, Relative error = %24.16e\n", abs_diff, rel_error)

        # 対称性の判定（相対誤差が1e-12より大きければ非対称）
        if rel_error > 1.0e-12
            println("         → NOT SYMMETRIC")
            is_symmetric = false
        else
            println("         → Symmetric (within tolerance)")
        end
    end

    println("\n=== Final Result ===")
    if is_symmetric
        println("The matrix appears to be SYMMETRIC")
    else
        println("The matrix is NOT SYMMETRIC")
    end
    println("====================")

    return is_symmetric
end


# テスト実行
function run_symmetry_test()
    println("Setting up problem...")

    # パラメータ設定
    NX = 20  # 小さいグリッドでテスト
    NY = 20
    NZ = 31  # Z方向は固定値（Zcase2!の要件）

    MX = NX + 2
    MY = NY + 2
    MZ = NZ + 2

    dx = 1.2e-3 / NX
    dy = 1.2e-3 / NY
    SZ = (MX, MY, MZ)
    Δh = (dx, dy, 1.0)
    ox = (0.0, 0.0, 0.0)

    Z = zeros(Float64, SZ[3])
    ΔZ= zeros(Float64, SZ[3]-1)
    ZC = zeros(Float64, SZ[3])
    ID = zeros(UInt8, SZ[1], SZ[2], SZ[3])

    wk = WorkBuffers(MX, MY, MZ)

    # 前処理
    preprocess!(wk.λ, wk.ρ, wk.cp, Z, ΔZ, ZC, ox, Δh, ID)

    # 境界条件
    bc_set = set_mode3_bc_parameters()

    wk.θ .= 300.0
    BoundaryConditions.apply_boundary_conditions!(wk.θ, wk.λ, wk.ρ, wk.cp, wk.mask, bc_set)

    Δt = 1000.0
    par = "sequential"

    # 対称性テスト
    test_matrix_symmetry(wk, Δh, Δt, ZC, ΔZ, par, num_tests=5)
end

# 実行
if abspath(PROGRAM_FILE) == @__FILE__
    run_symmetry_test()
end

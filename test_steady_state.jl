using Printf
using LinearAlgebra
using Test

include("heat3ds.jl")

"""
定常熱伝導の既知解テスト

1次元定常熱伝導問題:
- 下面 (z=0): T=300K (等温)
- 上面 (z=L): T=400K (等温)
- 側面: 断熱
- 熱源なし
- 一様物性値

解析解: T(z) = T_bottom + (T_top - T_bottom) * z/L
"""
function test_1d_steady_conduction()
  println("\n=== Test 1D Steady Heat Conduction ===")

  # パラメータ設定（小さいグリッド）
  NX = 10
  NY = 10
  NZ = 20

  MX = NX + 2
  MY = NY + 2
  MZ = NZ + 2

  dx = 1.0e-3 / NX
  dy = 1.0e-3 / NY
  dz = 1.0e-3 / NZ

  SZ = (MX, MY, MZ)
  Δh = (dx, dy, dz)
  ox = (0.0, 0.0, 0.0)

  # 一様格子（簡単のため）
  Z = zeros(Float64, SZ[3])
  ΔZ = zeros(Float64, SZ[3]-1)
  ZC = zeros(Float64, SZ[3])

  for k in 1:SZ[3]
    Z[k] = ox[3] + (k-2) * dz
  end

  for k in 1:SZ[3]-1
    ΔZ[k] = Z[k+1] - Z[k]
  end

  ZC[1] = Z[1]
  for k in 2:SZ[3]-1
    ZC[k] = 0.5 * (Z[k] + Z[k-1])
  end
  ZC[SZ[3]] = Z[SZ[3]]

  # ワークバッファ
  wk = WorkBuffers(MX, MY, MZ)

  # 一様物性値（Siliconを想定）
  λ_val = 8.88e-5  # 温度拡散率 [m^2/s]
  ρ_val = 2330.0   # 密度 [kg/m^3]
  cp_val = 720.0   # 比熱 [J/(kg·K)]

  wk.λ .= λ_val
  wk.ρ .= ρ_val
  wk.cp .= cp_val
  wk.hsrc .= 0.0  # 熱源なし

  # 境界条件: Z方向のみ温度指定、側面は断熱
  T_bottom = 300.0
  T_top = 400.0

  x_minus_bc = BoundaryConditions.heat_flux_bc(0.0)  # 断熱
  x_plus_bc = BoundaryConditions.heat_flux_bc(0.0)   # 断熱
  y_minus_bc = BoundaryConditions.heat_flux_bc(0.0)  # 断熱
  y_plus_bc = BoundaryConditions.heat_flux_bc(0.0)   # 断熱
  z_minus_bc = BoundaryConditions.isothermal_bc(T_bottom)  # 等温
  z_plus_bc = BoundaryConditions.isothermal_bc(T_top)      # 等温

  bc_set = BoundaryConditions.create_boundary_conditions(
    x_minus_bc, x_plus_bc,
    y_minus_bc, y_plus_bc,
    z_minus_bc, z_plus_bc
  )

  # 初期温度
  wk.θ .= (T_bottom + T_top) / 2.0

  # 境界条件適用
  BoundaryConditions.apply_boundary_conditions!(wk.θ, wk.λ, wk.ρ, wk.cp, wk.mask, bc_set)

  # RHS計算
  Δt = 1000.0
  par = "sequential"
  qsrf = zeros(Float64, SZ[1], SZ[2])

  NonUniform.calRHS!(wk, Δh, Δt, ΔZ, bc_set, qsrf, par)

  # ソルバー実行（定常解析: is_steady=true）
  # ※ まだ実装していないため、is_steadyパラメータはエラーになる
  tol = 1.0e-8
  F = stdout

  println("Solving with PBiCGSTAB (is_steady should be true)...")
  try
    # この呼び出しは失敗するはず（is_steadyパラメータがまだない）
    NonUniform.PBiCGSTAB!(wk, Δh, Δt, ZC, ΔZ, "gs", F, tol, par, is_steady=true)
  catch e
    println("Expected error (is_steady parameter not yet implemented): ", e)
    println("\nContinuing with current implementation...")
    NonUniform.PBiCGSTAB!(wk, Δh, Δt, ZC, ΔZ, "gs", F, tol, par)
  end

  # 解析解との比較
  println("\n--- Comparison with Analytical Solution ---")
  L = Z[SZ[3]] - Z[1]
  max_error = 0.0

  # 中央列 (i=NX/2, j=NY/2) で検証
  i_center = div(NX, 2) + 1
  j_center = div(NY, 2) + 1

  println("Z [m]          T_numerical [K]  T_analytical [K]  Error [K]")
  for k in 2:SZ[3]-1
    z = ZC[k]
    T_numerical = wk.θ[i_center, j_center, k]
    T_analytical = T_bottom + (T_top - T_bottom) * (z - Z[1]) / L
    error = abs(T_numerical - T_analytical)
    max_error = max(max_error, error)

    if (k-2) % 4 == 0  # 数点のみ表示
      @printf("%12.6e  %16.8f  %16.8f  %10.6f\n", z, T_numerical, T_analytical, error)
    end
  end

  println("\nMaximum error: $max_error [K]")

  # テスト判定
  tolerance = 1.0  # 1K以内の誤差を許容
  @test max_error < tolerance

  if max_error < tolerance
    println("✓ Test PASSED: Error is within tolerance ($tolerance K)")
  else
    println("✗ Test FAILED: Error exceeds tolerance ($tolerance K)")
  end

  println("=====================================\n")

  return max_error
end


"""
定常解析と非定常解析（大きなΔt）の比較
"""
function test_steady_vs_large_dt()
  println("\n=== Test Steady vs Large Δt ===")
  println("Testing if large Δt approximates steady-state solution...")

  # このテストは実装後に有効化
  println("(To be implemented after is_steady parameter is added)")
  println("================================\n")
end


# テスト実行
if abspath(PROGRAM_FILE) == @__FILE__
  println("Running Steady-State Tests...")
  println("=" ^ 60)

  test_1d_steady_conduction()
  test_steady_vs_large_dt()

  println("All tests completed.")
end

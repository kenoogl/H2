#!/usr/bin/env julia

"""
Heat3ds実行スクリプト

プロジェクトルートから実行:
  julia run.jl

パラメータ指定:
  julia run.jl 240 240 30 pbicgstab gs 1e-4 sequential true
"""

# srcディレクトリをロードパスに追加
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# heat3ds.jlをインクルード
include("src/heat3ds.jl")

# コマンドライン引数がある場合はパラメータとして使用
if length(ARGS) >= 7
  NX = parse(Int, ARGS[1])
  NY = parse(Int, ARGS[2])
  NZ = parse(Int, ARGS[3])
  solver = ARGS[4]
  smoother = ARGS[5]
  epsilon = parse(Float64, ARGS[6])
  par = ARGS[7]
  is_steady = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : false

  println("Running with custom parameters:")
  println("  Grid: $(NX)x$(NY)x$(NZ)")
  println("  Solver: $(solver), Smoother: $(smoother)")
  println("  Epsilon: $(epsilon), Parallel: $(par)")
  println("  Steady-state: $(is_steady)")

  q3d(NX, NY, NZ, solver, smoother, epsilon=epsilon, par=par, is_steady=is_steady)
else
  # デフォルト実行
  println("Running with default parameters:")
  println("  Grid: 240x240x30")
  println("  Solver: pbicgstab, Smoother: gs")
  println("  Epsilon: 1.0e-4, Parallel: sequential")
  println("  Steady-state: true")
  println()
  q3d(240, 240, 30, "pbicgstab", "gs", epsilon=1.0e-4, par="sequential", is_steady=true)
end

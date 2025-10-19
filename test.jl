#!/usr/bin/env julia

"""
Heat3dsテスト実行スクリプト

使用方法:
  julia test.jl [symmetry|steady]

  引数なし: 両方のテストを実行
  symmetry: 係数行列の対称性テスト
  steady: 定常熱伝導の既知解テスト
"""

# srcディレクトリをロードパスに追加
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

test_type = length(ARGS) > 0 ? ARGS[1] : "all"

if test_type == "symmetry" || test_type == "all"
  println("=== Running symmetry test ===")
  include("src/test_symmetry.jl")
  println()
end

if test_type == "steady" || test_type == "all"
  println("=== Running steady-state test ===")
  include("src/test_steady_state.jl")
  println()
end

if test_type != "symmetry" && test_type != "steady" && test_type != "all"
  println("Unknown test type: $(test_type)")
  println("Available tests: symmetry, steady, all")
  exit(1)
end

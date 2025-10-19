"""
単一スレッド数でのベンチマーク測定スクリプト

使用方法:
  # Sequential実行
  julia benchmark_single.jl sequential

  # 並列実行（スレッド数指定）
  julia -t 1 benchmark_single.jl thread
  julia -t 2 benchmark_single.jl thread
  julia -t 4 benchmark_single.jl thread
  julia -t 8 benchmark_single.jl thread

結果はbenchmark_results.csvに追記される
"""

# srcディレクトリをロードパスに追加
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# heat3ds.jlをインクルード
include("src/heat3ds.jl")

using Base.Threads

function run_benchmark(par_mode)
  println("=" ^ 80)
  println("ベンチマーク測定")
  println("=" ^ 80)
  println("パラメータ: NX=240, NY=240, NZ=30")
  println("ソルバー: PBiCGSTAB, スムーザー: Gauss-Seidel")
  println("収束判定: epsilon=1.0e-4, 定常解析: true")
  println("実行モード: $(par_mode)")
  println("スレッド数: $(nthreads())")
  println("=" ^ 80)
  println()

  # ガベージコレクション実行
  GC.gc()
  sleep(1)

  # ベンチマーク実行
  elapsed_time = @elapsed begin
    q3d(240, 240, 30, "pbicgstab", "gs", epsilon=1.0e-4, par=par_mode, is_steady=true)
  end

  println()
  println("=" ^ 80)
  println("測定結果")
  println("=" ^ 80)
  println("実行モード: $(par_mode)")
  println("スレッド数: $(nthreads())")
  println("実行時間: $(round(elapsed_time, digits=3)) 秒")
  println("=" ^ 80)
  println()

  # 結果をCSVファイルに追記
  mode_str = par_mode == "sequential" ? "Sequential" : "Parallel"
  file_exists = isfile("benchmark_results.csv")

  open("benchmark_results.csv", "a") do io
    if !file_exists
      println(io, "Mode,Threads,Time(s)")
    end
    println(io, "$(mode_str),$(nthreads()),$(elapsed_time)")
  end

  println("結果をbenchmark_results.csvに保存しました")

  return elapsed_time
end

# メイン実行
if length(ARGS) < 1
  println("使用方法:")
  println("  julia benchmark_single.jl sequential")
  println("  julia -t N benchmark_single.jl thread")
  exit(1)
end

par_mode = ARGS[1]

if par_mode == "thread" && nthreads() == 1
  @warn "par=\"thread\"が指定されましたが、スレッド数が1です。julia -t Nで起動してください。"
end

run_benchmark(par_mode)

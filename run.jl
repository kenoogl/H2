#!/usr/bin/env julia

"""
Heat3ds実行スクリプト

【基本実行】
  julia run.jl

【パラメータ指定】
  julia run.jl NX NY NZ solver smoother epsilon par [is_steady] [parareal]

  例: julia run.jl 240 240 30 pbicgstab gs 1e-4 sequential true false

【並列実行】
  julia -t 4 run.jl 240 240 30 pbicgstab gs 1e-4 thread true false

  注意: par="thread"の場合、julia -t N で起動してください

【Parareal並列実行】
  mpirun -np 4 julia run.jl 240 240 30 pbicgstab gs 1e-4 thread false true

  注意: parareal=trueの場合、MPIで起動してください

【引数】
  NX, NY, NZ : グリッドサイズ（内部セル数）
  solver     : pbicgstab | cg | sor
  smoother   : gs | "" (空文字列でスムーザーなし)
  epsilon    : 収束判定基準（例: 1e-4, 1e-6）
  par        : sequential | thread
  is_steady  : true (定常) | false (非定常) [省略可、デフォルト: false]
  parareal   : true (Parareal時間並列化) | false (逐次時間積分) [省略可、デフォルト: false]
"""

# srcディレクトリをロードパスに追加
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# heat3ds.jlをインクルード
include("src/heat3ds.jl")

# スレッド数の表示
using Base.Threads
println("Julia Threads: ", nthreads())

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
  parareal = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : false

  # par="thread"だがスレッド数が1の場合は警告
  if par == "thread" && nthreads() == 1
    @warn "par=\"thread\" specified but Julia has only 1 thread. Use 'julia -t N run.jl ...' for parallel execution."
  end
  
  # parareal=trueだがMPIが初期化されていない場合の警告
  if parareal
    try
      using MPI
      if !MPI.Initialized()
        @warn "Parareal mode specified but MPI not initialized. Use 'mpirun -np N julia run.jl ...' for MPI execution."
        @warn "Continuing with single-process parareal (limited functionality)."
      end
    catch
      @warn "MPI.jl not available. Parareal mode will use single-process fallback."
    end
  end

  println("Running with custom parameters:")
  println("  Grid: $(NX)x$(NY)x$(NZ)")
  println("  Solver: $(solver), Smoother: $(smoother)")
  println("  Epsilon: $(epsilon), Parallel: $(par)")
  println("  Steady-state: $(is_steady)")
  println("  Parareal: $(parareal)")
  println()

  q3d(NX, NY, NZ, solver, smoother, epsilon=epsilon, par=par, is_steady=is_steady, parareal=parareal)
else
  # デフォルト実行
  println("Running with default parameters:")
  println("  Grid: 240x240x30")
  println("  Solver: pbicgstab, Smoother: gs")
  println("  Epsilon: 1.0e-4, Parallel: sequential")
  println("  Steady-state: true")
  println("  Parareal: false")
  println()
  q3d(240, 240, 30, "pbicgstab", "gs", epsilon=1.0e-4, par="sequential", is_steady=true, parareal=false)
end

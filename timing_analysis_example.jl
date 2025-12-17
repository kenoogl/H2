# 詳細なタイミング分析レポートの使用例

# Add src to path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Include the Parareal module
include("src/parareal.jl")
using .Parareal

println("=== 詳細なタイミング分析レポートの使用例 ===")
println()

# 1. パフォーマンスアナライザーを作成
analyzer = Parareal.create_performance_analyzer(Float64)

# 2. 詳細タイミング分析を有効化
analyzer.enable_detailed_timing = true

# 3. サンプルデータを設定（実際の使用では、Parareal計算から自動的に収集される）
breakdown = analyzer.timing_breakdown

# 粗解法のタイミングデータ
breakdown.coarse_solver_breakdown["total_time"] = 2.5
breakdown.coarse_solver_breakdown["total_calls"] = 100.0
breakdown.coarse_solver_breakdown["average_time"] = 0.025
breakdown.coarse_solver_breakdown["setup_time"] = 0.1
breakdown.coarse_solver_breakdown["solver_time"] = 2.3
breakdown.coarse_solver_breakdown["cleanup_time"] = 0.1

# 精密解法のタイミングデータ
breakdown.fine_solver_breakdown["total_time"] = 8.3
breakdown.fine_solver_breakdown["total_calls"] = 50.0
breakdown.fine_solver_breakdown["average_time"] = 0.166
breakdown.fine_solver_breakdown["setup_time"] = 0.2
breakdown.fine_solver_breakdown["solver_time"] = 7.8
breakdown.fine_solver_breakdown["cleanup_time"] = 0.3

# MPI通信のタイミングデータ
breakdown.mpi_communication_breakdown["send_time"] = 0.5
breakdown.mpi_communication_breakdown["receive_time"] = 0.4
breakdown.mpi_communication_breakdown["synchronization_time"] = 0.3
breakdown.mpi_communication_breakdown["broadcast_time"] = 0.2
breakdown.mpi_communication_breakdown["allreduce_time"] = 0.1
breakdown.mpi_communication_breakdown["total_time"] = 1.5

# スレッド処理のタイミングデータ
breakdown.threading_breakdown["parallel_efficiency"] = 0.85
breakdown.threading_breakdown["load_balance_factor"] = 0.92
breakdown.threading_breakdown["thread_overhead"] = 0.15
breakdown.threading_breakdown["synchronization_overhead"] = 0.08

# オーバーヘッドデータ
breakdown.synchronization_overhead = 0.08
breakdown.load_imbalance_overhead = 0.12
breakdown.memory_overhead = 0.05

println("📊 方法1: 詳細タイミングレポートを生成")
println("=" ^ 50)

# 4. 詳細タイミングレポートを生成
detailed_report = Parareal.generate_detailed_timing_report(analyzer)
println(detailed_report)

println()
println("📊 方法2: コンソールに直接出力")
println("=" ^ 50)

# 5. コンソールに直接出力
Parareal.print_timing_breakdown(analyzer)

println()
println("📊 方法3: CSVファイルにエクスポート")
println("=" ^ 50)

# 6. CSVファイルにエクスポート
Parareal.export_performance_data_csv(analyzer, "detailed_timing_analysis.csv")

# CSVファイルの内容を表示
if isfile("detailed_timing_analysis.csv")
    println("CSVファイルの内容（最初の20行）:")
    println("-" ^ 40)
    open("detailed_timing_analysis.csv", "r") do file
        for i in 1:20
            line = readline(file)
            if !isempty(line)
                println(line)
            else
                break
            end
        end
    end
    
    # クリーンアップ
    rm("detailed_timing_analysis.csv")
end

println()
println("📊 方法4: 包括的パフォーマンスレポート（タイミング分析を含む）")
println("=" ^ 50)

# 7. 包括的レポート（タイミング分析を含む）
analyzer.enable_scaling_analysis = true
analyzer.enable_visualization = true

# スケーリングデータも追加
scaling = analyzer.scaling_analysis
scaling.strong_scaling_processes = [1, 2, 4, 8]
scaling.strong_scaling_times = [12.0, 6.5, 3.8, 2.2]
scaling.strong_scaling_speedups = [1.0, 1.85, 3.16, 5.45]
scaling.strong_scaling_efficiencies = [1.0, 0.92, 0.79, 0.68]

comprehensive_report = Parareal.generate_comprehensive_performance_report(analyzer)
# 最初の1000文字のみ表示
println(comprehensive_report[1:min(1000, length(comprehensive_report))] * "...")

println()
println("=" ^ 60)
println("✅ 詳細なタイミング分析レポートの使用方法:")
println()
println("1. create_performance_analyzer() でアナライザーを作成")
println("2. enable_detailed_timing = true で詳細分析を有効化")
println("3. Parareal計算実行中に自動的にタイミングデータが収集される")
println("4. generate_detailed_timing_report() でレポート生成")
println("5. print_timing_breakdown() でコンソール出力")
println("6. export_performance_data_csv() でCSVエクスポート")
println()
println("📍 実装場所: src/parareal.jl の約4555行目")
println("📍 関連構造体: TimingBreakdown, PerformanceAnalyzer")
println("📍 要件: Requirements 10.4 (MPI/スレッド成分の詳細タイミング分析)")
println("=" ^ 60)
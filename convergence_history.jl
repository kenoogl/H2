using Printf
using Plots
using Dates

"""
ConvergenceData
反復解法の収束履歴を保存する構造体

Fields:
- solver_name: 解法名 ("sor", "jacobi", "pbicgstab", "cg")
- smoother_name: スムーザー名 ("gs", "jacobi", "")
- iterations: 反復回数の配列
- residuals: 残差の配列
- start_time: 計算開始時刻
"""
mutable struct ConvergenceData
    solver_name::String
    smoother_name::String
    iterations::Vector{Int}
    residuals::Vector{Float64}
    start_time::Float64
    
    function ConvergenceData(solver::String, smoother::String="")
        new(solver, smoother, Int[], Float64[], time())
    end
end

"""
add_residual!(conv_data, iteration, residual)
収束履歴に残差を追加

Args:
- conv_data: ConvergenceData構造体
- iteration: 反復回数
- residual: 残差値
"""
function add_residual!(conv_data::ConvergenceData, iteration::Int, residual::Float64)
    push!(conv_data.iterations, iteration)
    push!(conv_data.residuals, residual)
end

"""
plot_convergence_curve(conv_data, filename)
収束曲線をプロット（対数表示）

Args:
- conv_data: ConvergenceData構造体
- filename: 出力ファイル名
- target_tol: 目標収束判定値（オプション、デフォルト1.0e-8）
- show_markers: マーカー表示の有無（オプション、デフォルトtrue）
"""
function plot_convergence_curve(conv_data::ConvergenceData, filename::String; target_tol::Float64=1.0e-8, show_markers::Bool=true)
    if isempty(conv_data.residuals)
        println("Warning: Convergence history data is empty")
        return
    end
    
    # タイトル作成
    title_str = "Convergence History: $(conv_data.solver_name)"
    if !isempty(conv_data.smoother_name)
        title_str *= " + $(conv_data.smoother_name)"
    end
    
    # マーカー設定
    marker_style = show_markers ? :circle : :none
    marker_size = show_markers ? 3 : 0
    
    # Y軸の範囲を取得して整数指数のティックを設定
    min_res = minimum(conv_data.residuals)
    max_res = maximum(conv_data.residuals)
    
    # 指数の範囲を決定（整数に丸める）
    min_exp = floor(Int, log10(min_res))
    max_exp = ceil(Int, log10(max_res))
    
    # 整数指数のティック値とラベルを生成
    tick_exponents = min_exp:max_exp
    tick_values = [10.0^exp for exp in tick_exponents]
    tick_labels = ["10^$exp" for exp in tick_exponents]
    
    p = plot(conv_data.iterations, conv_data.residuals,
        yscale=:log10,
        marker=marker_style,
        markersize=marker_size,
        linewidth=2,
        xlabel="Iteration",
        ylabel="Residual",
        title=title_str,
        label="Residual",
        size=(800, 600),
        grid=true,
        yticks=(tick_values, tick_labels)
    )
    
    # 目標収束判定線を追加
    if target_tol > 0.0
        hline!([target_tol], 
            linestyle=:dash, 
            linewidth=2, 
            color=:red, 
            label="Target tolerance ($(target_tol))"
        )
    end
    
    # 注記：整数指数ティックにより、annotationは不要
    
    savefig(p, filename)
    println("Convergence plot saved: $filename")
end

"""
export_convergence_csv(conv_data, filename)
収束履歴をCSVファイルに出力

Args:
- conv_data: ConvergenceData構造体  
- filename: 出力CSVファイル名
"""
function export_convergence_csv(conv_data::ConvergenceData, filename::String)
    if isempty(conv_data.residuals)
        println("Warning: Convergence history data is empty")
        return
    end
    
    open(filename, "w") do f
        # ヘッダー
        println(f, "# Convergence History Data")
        println(f, "# Solver: $(conv_data.solver_name)")
        if !isempty(conv_data.smoother_name)
            println(f, "# Smoother: $(conv_data.smoother_name)")
        end
        timestamp = Dates.format(Dates.unix2datetime(conv_data.start_time), "yyyy-mm-dd HH:MM:SS")
        println(f, "# Generated: $timestamp")
        println(f, "Iteration,Residual")
        
        # データ
        for i in 1:length(conv_data.iterations)
            res_str = @sprintf("%.14E", conv_data.residuals[i])
            println(f, "$(conv_data.iterations[i]),$res_str")
        end
    end
    
    println("Convergence history CSV saved: $filename")
end

"""
compare_convergence(conv_data_list, filename)
複数の解法の収束を比較プロット

Args:
- conv_data_list: ConvergenceData構造体の配列
- filename: 出力ファイル名
"""
function compare_convergence(conv_data_list::Vector{ConvergenceData}, filename::String; target_tol::Float64=1.0e-8, show_markers::Bool=true)
    if isempty(conv_data_list)
        println("Warning: Comparison data is empty")
        return
    end
    
    # 全データから指数範囲を決定
    all_residuals = Float64[]
    for conv_data in conv_data_list
        if !isempty(conv_data.residuals)
            append!(all_residuals, conv_data.residuals)
        end
    end
    
    # 整数指数のティックを設定
    if !isempty(all_residuals)
        min_exp = floor(Int, log10(minimum(all_residuals)))
        max_exp = ceil(Int, log10(maximum(all_residuals)))
        tick_exponents = min_exp:max_exp
        tick_values = [10.0^exp for exp in tick_exponents]
        tick_labels = ["10^$exp" for exp in tick_exponents]
    else
        tick_values = Float64[]
        tick_labels = String[]
    end
    
    p = plot(
        yscale=:log10,
        xlabel="Iteration",
        ylabel="Residual",
        title="Solver Convergence Comparison",
        size=(1000, 700),
        grid=true,
        yticks=(tick_values, tick_labels)
    )
    
    colors = [:blue, :red, :green, :orange, :purple]
    
    for (i, conv_data) in enumerate(conv_data_list)
        if !isempty(conv_data.residuals)
            label_str = conv_data.solver_name
            if !isempty(conv_data.smoother_name)
                label_str *= " + $(conv_data.smoother_name)"
            end
            
            color = colors[mod(i-1, length(colors)) + 1]
            marker_style = show_markers ? :circle : :none
            marker_size = show_markers ? 2 : 0
            
            plot!(p, conv_data.iterations, conv_data.residuals,
                marker=marker_style,
                markersize=marker_size,
                linewidth=2,
                color=color,
                label=label_str
            )
        end
    end
    
    # 目標収束判定線
    if target_tol > 0.0
        hline!([target_tol], 
            linestyle=:dash, 
            linewidth=2, 
            color=:black, 
            label="Target tolerance"
        )
    end
    
    savefig(p, filename)
    println("Comparison plot saved: $filename")
end

"""
get_convergence_info(conv_data)
収束情報を取得

Returns:
- 辞書形式の収束情報
"""
function get_convergence_info(conv_data::ConvergenceData)
    if isempty(conv_data.residuals)
        return Dict()
    end
    
    return Dict(
        "solver" => conv_data.solver_name,
        "smoother" => conv_data.smoother_name,
        "iterations" => length(conv_data.residuals),
        "initial_residual" => conv_data.residuals[1],
        "final_residual" => conv_data.residuals[end],
        "convergence_rate" => conv_data.residuals[end] / conv_data.residuals[1],
        "reduction_factor" => log10(conv_data.residuals[1] / conv_data.residuals[end])
    )
end
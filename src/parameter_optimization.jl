# Parameter management module for Parareal algorithm
module ParameterOptimization

using Printf
using Statistics
using LinearAlgebra

export PararealParameters, ValidationResult
export create_parareal_parameters, validate_parameters
export print_parameter_summary

"""
Pararealアルゴリズムのパラメータ設定
ユーザが直接指定する基本パラメータを格納
"""
struct PararealParameters{T <: AbstractFloat}
    # 時間ステップ設定
    coarse_dt::T              # 粗解法の時間ステップ
    fine_dt::T                # 精密解法の時間ステップ
    time_step_ratio::T        # 時間ステップ比率 (coarse_dt / fine_dt)
    
    # 時間分割設定
    n_time_windows::Int       # 時間窓数
    total_time::T             # 総シミュレーション時間
    
    # 収束設定
    max_iterations::Int       # 最大反復数
    convergence_tolerance::T  # 収束許容誤差
    
    # MPI設定
    n_mpi_processes::Int      # MPIプロセス数
    n_threads_per_process::Int # プロセス当たりのスレッド数
    
    function PararealParameters{T}(;
        coarse_dt::T,
        fine_dt::T,
        n_time_windows::Int = 4,
        total_time::T = T(1.0),
        max_iterations::Int = 10,
        convergence_tolerance::T = T(1e-6),
        n_mpi_processes::Int = 4,
        n_threads_per_process::Int = 1
    ) where {T <: AbstractFloat}
        
        # 基本的な妥当性チェック
        if coarse_dt <= 0
            error("粗解法の時間ステップは正の値でなければなりません: $coarse_dt")
        end
        if fine_dt <= 0
            error("精密解法の時間ステップは正の値でなければなりません: $fine_dt")
        end
        if coarse_dt <= fine_dt
            error("粗解法の時間ステップは精密解法より大きくなければなりません: coarse_dt=$coarse_dt, fine_dt=$fine_dt")
        end
        if n_time_windows < 2
            error("時間窓数は2以上でなければなりません: $n_time_windows")
        end
        if total_time <= 0
            error("総シミュレーション時間は正の値でなければなりません: $total_time")
        end
        if max_iterations < 1
            error("最大反復数は1以上でなければなりません: $max_iterations")
        end
        if convergence_tolerance <= 0
            error("収束許容誤差は正の値でなければなりません: $convergence_tolerance")
        end
        if n_mpi_processes < 1
            error("MPIプロセス数は1以上でなければなりません: $n_mpi_processes")
        end
        if n_threads_per_process < 1
            error("プロセス当たりのスレッド数は1以上でなければなりません: $n_threads_per_process")
        end
        
        # 時間ステップ比率を計算
        time_step_ratio = coarse_dt / fine_dt
        
        return new{T}(
            coarse_dt, fine_dt, time_step_ratio,
            n_time_windows, total_time,
            max_iterations, convergence_tolerance,
            n_mpi_processes, n_threads_per_process
        )
    end
end

"""
パラメータ検証結果
"""
struct ValidationResult{T <: AbstractFloat}
    is_valid::Bool                    # 全体的な妥当性
    warnings::Vector{String}          # 警告メッセージ
    recommendations::Vector{String}   # 推奨事項
    
    # 計算された特性
    time_steps_per_window::Int        # 時間窓あたりのステップ数
    total_coarse_steps::Int           # 総粗解法ステップ数
    total_fine_steps::Int             # 総精密解法ステップ数
    estimated_memory_gb::T            # 推定メモリ使用量
    
    function ValidationResult{T}(
        is_valid::Bool,
        warnings::Vector{String} = String[],
        recommendations::Vector{String} = String[],
        time_steps_per_window::Int = 0,
        total_coarse_steps::Int = 0,
        total_fine_steps::Int = 0,
        estimated_memory_gb::T = T(0.0)
    ) where {T <: AbstractFloat}
        
        return new{T}(
            is_valid, warnings, recommendations,
            time_steps_per_window, total_coarse_steps, total_fine_steps,
            estimated_memory_gb
        )
    end
end

"""
Pararealパラメータを作成
"""
function create_parareal_parameters(::Type{T} = Float64;
                                   coarse_dt::T,
                                   fine_dt::T,
                                   n_time_windows::Int = 4,
                                   total_time::T = T(1.0),
                                   max_iterations::Int = 10,
                                   convergence_tolerance::T = T(1e-6),
                                   n_mpi_processes::Int = 4,
                                   n_threads_per_process::Int = 1) where {T <: AbstractFloat}
    
    return PararealParameters{T}(
        coarse_dt = coarse_dt,
        fine_dt = fine_dt,
        n_time_windows = n_time_windows,
        total_time = total_time,
        max_iterations = max_iterations,
        convergence_tolerance = convergence_tolerance,
        n_mpi_processes = n_mpi_processes,
        n_threads_per_process = n_threads_per_process
    )
end

"""
パラメータの妥当性を検証
"""
function validate_parameters(params::PararealParameters{T};
                           grid_size::Union{NTuple{3,Int}, Nothing} = nothing,
                           thermal_diffusivity::Union{T, Nothing} = nothing) where {T <: AbstractFloat}
    
    warnings = String[]
    recommendations = String[]
    is_valid = true
    
    # 時間ステップ比率の妥当性チェック
    if params.time_step_ratio < 2.0
        push!(warnings, "時間ステップ比率が小さすぎます ($(params.time_step_ratio)). 並列化効果が期待できません.")
        push!(recommendations, "時間ステップ比率を5以上に設定することを推奨します.")
    elseif params.time_step_ratio > 1000.0
        push!(warnings, "時間ステップ比率が大きすぎます ($(params.time_step_ratio)). 収束性に問題が生じる可能性があります.")
        push!(recommendations, "時間ステップ比率を100以下に設定することを推奨します.")
    end
    
    # 時間窓数とMPIプロセス数の整合性チェック
    if params.n_time_windows < params.n_mpi_processes
        push!(warnings, "時間窓数 ($(params.n_time_windows)) がMPIプロセス数 ($(params.n_mpi_processes)) より少ないです.")
        push!(recommendations, "時間窓数をMPIプロセス数以上に設定してください.")
    end
    
    # 総時間と時間ステップの整合性チェック
    coarse_steps_per_window = Int(ceil(params.total_time / params.n_time_windows / params.coarse_dt))
    fine_steps_per_window = Int(ceil(params.total_time / params.n_time_windows / params.fine_dt))
    
    if coarse_steps_per_window < 1
        push!(warnings, "粗解法の時間ステップが大きすぎます. 時間窓あたり1ステップ未満になります.")
        is_valid = false
    end
    
    if fine_steps_per_window < 2
        push!(warnings, "精密解法の時間ステップが大きすぎます. 時間窓あたり2ステップ未満になります.")
        is_valid = false
    end
    
    # 安定性チェック（熱拡散率が与えられている場合）
    if thermal_diffusivity !== nothing && grid_size !== nothing
        min_spacing = minimum([1.0, 1.0, 1.0])  # デフォルト格子間隔
        stability_limit = min_spacing^2 / (6 * thermal_diffusivity)
        
        if params.fine_dt > stability_limit
            push!(warnings, "精密解法の時間ステップが安定性限界 ($stability_limit) を超えています.")
            push!(recommendations, "精密解法の時間ステップを $(stability_limit * 0.5) 以下に設定してください.")
        end
    end
    
    # メモリ使用量の推定
    estimated_memory = T(0.0)
    if grid_size !== nothing
        total_dofs = prod(grid_size)
        bytes_per_dof = sizeof(T) * 8  # 温度場 + 作業配列
        estimated_memory = (total_dofs * bytes_per_dof * params.n_time_windows) / (1024^3)
        
        if estimated_memory > 32.0
            push!(warnings, "推定メモリ使用量が大きいです ($(round(estimated_memory, digits=2)) GB).")
            push!(recommendations, "時間窓数を減らすか、問題サイズを小さくすることを検討してください.")
        end
    end
    
    # 計算される値
    total_coarse_steps = coarse_steps_per_window * params.n_time_windows
    total_fine_steps = fine_steps_per_window * params.n_time_windows
    
    return ValidationResult{T}(
        is_valid, warnings, recommendations,
        fine_steps_per_window, total_coarse_steps, total_fine_steps,
        estimated_memory
    )
end

"""
パラメータ設定の概要を表示
"""
function print_parameter_summary(params::PararealParameters{T}) where {T <: AbstractFloat}
    println("=== Parareal パラメータ設定 ===")
    println("時間ステップ設定:")
    println("  粗解法時間ステップ: $(params.coarse_dt)")
    println("  精密解法時間ステップ: $(params.fine_dt)")
    println("  時間ステップ比率: $(params.time_step_ratio)")
    println()
    println("時間分割設定:")
    println("  時間窓数: $(params.n_time_windows)")
    println("  総シミュレーション時間: $(params.total_time)")
    println()
    println("収束設定:")
    println("  最大反復数: $(params.max_iterations)")
    println("  収束許容誤差: $(params.convergence_tolerance)")
    println()
    println("並列化設定:")
    println("  MPIプロセス数: $(params.n_mpi_processes)")
    println("  プロセス当たりスレッド数: $(params.n_threads_per_process)")
    println("===============================")
end

end # module ParameterOptimization
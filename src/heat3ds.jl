using Printf
using LinearAlgebra
using FLoops
using ThreadsX

include("common.jl")
include("modelA.jl")
include("boundary_conditions.jl")
include("Zcoord.jl")
include("RHS.jl")
include("NonUniform.jl")
include("plotter.jl")
include("convergence_history.jl")
include("parse_log_residuals.jl")
include("parareal.jl")
include("output_format.jl")

using .Common
using .Common: WorkBuffers, ItrMax, Q_src, get_backend
using .NonUniform
using .NonUniform: PBiCGSTAB!, CG!, calRHS!
using .RHSCore
using .BoundaryConditions
using .Parareal
using .OutputFormat

"""
Mode3用の境界条件（NonUniform格子のIC問題）  
Z下面: PCB温度、Z上面: 熱伝達、側面: 断熱
"""
function set_mode3_bc_parameters()
    θ_amb = 300.0 # [K]
    θ_pcb = 300.0 # [K]
    HT_top = 5.0 # 2.98e-4 # 5 [W/(m^2 K)] / (\rho C)_silicon > [m/s]
    HT_side = 5.0 # 2.98e-6 # 5 [W/(m^2 K)] / (\rho C)_silicon > [m/s]
    
    # 各面の境界条件を定義
    x_minus_bc = BoundaryConditions.convection_bc(HT_side, θ_amb)
    x_plus_bc  = BoundaryConditions.convection_bc(HT_side, θ_amb)
    y_minus_bc = BoundaryConditions.convection_bc(HT_side, θ_amb)  
    y_plus_bc  = BoundaryConditions.convection_bc(HT_side, θ_amb)
    z_minus_bc = BoundaryConditions.isothermal_bc(θ_pcb)                  # Z軸負方向面: PCB温度
    z_plus_bc  = BoundaryConditions.convection_bc(HT_top, θ_amb)          # Z軸正方向面: 熱伝達
    #z_plus_bc  = BoundaryConditions.heat_flux_bc(100000.0)          # Z軸正方向面: 5 [W/m^2] 熱流束
    
    # 境界条件セットを作成
    return BoundaryConditions.create_boundary_conditions(x_minus_bc, x_plus_bc,
                                      y_minus_bc, y_plus_bc,
                                      z_minus_bc, z_plus_bc)
end



"""
@brief 計算領域内部の熱源項の設定
@param [in,out] hs   熱源
@param [in]     ID   識別子配列
"""
function HeatSrc!(hs::Array{Float64,3}, ID::Array{UInt8,3}, par)
    backend = get_backend(par)
    SZ = size(hs)
    
    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        if ID[i,j,k] == modelA.pwrsrc["id"]
            hs[i,j,k] = Q_src
        end
    end
end


function conditions(F, SZ, Δh, solver, smoother)

    @printf(F, "Problem : IC on NonUniform grid (Opt. 13 layers)\n")

    @printf(F, "Grid  : %d %d %d\n", SZ[1], SZ[2], SZ[3])
    @printf(F, "Pitch : %6.4e %6.4e %6.4e\n", Δh[1], Δh[2], Δh[3])
    if solver=="pbicgstab" || solver=="cg"
        if isempty(smoother)
            @printf(F, "Solver: %s without preconditioner\n", solver)
        else
            @printf(F, "Solver: %s with smoother %s\n", solver, smoother)
        end
    else
        @printf(F, "Solver: %s\n", solver)
    end
    @printf(F, "ItrMax : %e\n", ItrMax)
    @printf(F, "ε      : %e\n", itr_tol)
end

"""
モデル作成
@param [in] λ   温度拡散係数
"""
function preprocess!(λ, ρ, cp, Z, ox, Δh, ID)
    SZ = size(λ)
    modelA.fillID!(ID, ox, Δh, Z)
    modelA.setProperties!(λ, ρ, cp, ID)
end


"""
Run Parareal computation with Heat3ds integration
"""
function run_parareal_computation(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par, parareal_config; is_steady::Bool=false)
    # Create default parareal configuration if none provided
    if parareal_config === nothing
        parareal_config = Dict{String,Any}(
            "total_time" => 10000.0,  # Default total simulation time
            "n_time_windows" => 4,
            "dt_coarse" => Δt * 10.0,  # Coarse time step (10x larger)
            "dt_fine" => Δt,
            "max_iterations" => 10,
            "convergence_tolerance" => 1.0e-6,
            "n_mpi_processes" => 1,  # Default to single process for now
            "n_threads_per_process" => Threads.nthreads()
        )
    end
    
    try
        # Create PararealConfig from dictionary
        config = PararealConfig{Float64}(
            total_time = Float64(parareal_config["total_time"]),
            n_time_windows = parareal_config["n_time_windows"],
            dt_coarse = Float64(parareal_config["dt_coarse"]),
            dt_fine = Float64(parareal_config["dt_fine"]),
            max_iterations = parareal_config["max_iterations"],
            convergence_tolerance = Float64(parareal_config["convergence_tolerance"]),
            n_mpi_processes = parareal_config["n_mpi_processes"],
            n_threads_per_process = parareal_config["n_threads_per_process"]
        )
        
        # Create PararealManager
        manager = PararealManager{Float64}(config)
        
        # Initialize MPI environment for parareal
        initialize_mpi_parareal!(manager)
        
        # Create Heat3ds problem data
        problem_data = create_heat3ds_problem_data(Δh, ZC, ΔZ, ID, bc_set, par, is_steady)
        
        # Initial condition (current temperature field)
        initial_condition = copy(wk.θ)
        
        # Run parareal computation
        result = run_parareal!(manager, initial_condition, problem_data)
        
        # Update working buffers with final result
        wk.θ .= result.final_solution
        
        # Generate parareal output in Heat3ds compatible format
        # Task 8.4: Implement output format consistency
        try
            # Create output manager for parareal results
            output_manager = create_output_manager(
                base_filename = "heat3ds_parareal",
                computation_mode = "parareal",
                grid_size = size(result.final_solution),
                n_time_windows = config.n_time_windows,
                n_mpi_processes = config.n_mpi_processes
            )
            
            # Generate outputs in Heat3ds compatible format
            generated_files = generate_parareal_output!(output_manager, result.final_solution, result, problem_data)
            
            # Ensure output consistency with sequential Heat3ds format
            is_consistent = ensure_output_consistency!(output_manager)
            
            if is_consistent
                println("✓ Parareal output generated in Heat3ds compatible format")
                println("  Generated files: $(join(generated_files, ", "))")
            else
                @warn "Output format consistency check failed"
            end
            
        catch output_error
            @warn "Failed to generate parareal output: $output_error"
            @warn "Continuing with standard output..."
        end
        
        # Finalize MPI environment
        finalize_mpi_parareal!(manager)
        
        # Create convergence data compatible with existing Heat3ds format
        conv_data = ConvergenceData(solver, smoother)
        
        # Add parareal-specific convergence information
        if !isempty(result.residual_history)
            for (i, residual) in enumerate(result.residual_history)
                push!(conv_data.residuals, residual)
                push!(conv_data.iterations, i)
            end
        end
        
        println("Parareal computation completed:")
        println("  Converged: $(result.converged)")
        println("  Iterations: $(result.iterations)")
        println("  Final residual: $(length(result.residual_history) > 0 ? result.residual_history[end] : "N/A")")
        println("  Computation time: $(result.computation_time) seconds")
        
        return conv_data
        
    catch e
        @warn "Parareal computation failed: $e"
        @warn "Falling back to sequential computation..."
        
        # Fallback to sequential computation
        return main(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par, is_steady=is_steady)
    end
end

"""
@param [in] Δh       セル幅
@param [in] Δt       時間積分幅
@param [in] wk       ベクトル群
@param [in] ZC       CVセンター座標
@param [in] ΔZ       CV幅
@param [in] solver   ["sor", "pbicgstab", "cg"]
@param [in] smoother ["gs", ""]
@param [in] is_steady 定常解析フラグ
"""
function main(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par; is_steady::Bool=false)
  # 収束履歴の初期化
  conv_data = ConvergenceData(solver, smoother)

  SZ = size(wk.θ)

  qsrf = zeros(Float64, SZ[1], SZ[2])

  HeatSrc!(wk.hsrc, ID, par)

  # HC配列を生成（境界条件から）
  HC = BoundaryConditions.set_BC_coef(bc_set)

  F = open("log.txt", "w")
  conditions(F, SZ, Δh, solver, smoother)
  time::Float64 = 0.0
  nt::Int64 = 1

  for step in 1:nt
    time += Δt

    calRHS!(wk, Δh, Δt, ΔZ, bc_set, qsrf, par, is_steady=is_steady)

    # ソルバー呼び出しを修正
    if solver == "cg"
      smoother_sym = smoother == "gs" ? :gs : :none
      isconverged, itr, res0 = NonUniform.CG!(wk, Δh, Δt, ZC, ΔZ, HC,
                                   tol=itr_tol, smoother=smoother_sym,
                                   par=par, verbose=true, is_steady=is_steady)
    else
      smoother_sym = smoother == "gs" ? :gs : :none
      isconverged, itr, res0 = NonUniform.PBiCGSTAB!(wk, Δh, Δt, ZC, ΔZ, HC,
                                          tol=itr_tol, smoother=smoother_sym,
                                          par=par, verbose=true, is_steady=is_steady)
    end

    if !isconverged
      @warn "Solver did not converge at step $(step)"
    end

    s = @view wk.θ[2:SZ[1]-1, 2:SZ[2]-1, 2:SZ[3]-1]
    min_val = minimum(s)
    max_val = maximum(s)
    @printf(F, "%d %f : θmin=%e  θmax=%e  L2 norm of θ=%e\n", step, time, min_val, max_val, norm(s,2))
  end

  close(F)

  # ログファイルから残差データを解析してconv_dataに追加
  parse_residuals_from_log!(conv_data, "log.txt")

  # 収束履歴データを返す
  return conv_data
end

#=
@param NXY  Number of inner cells for X&Y dir.
@param NZ   Number of inner cells for Z dir.
@param [in] solver    ["jacobi", "sor", "pbicgstab"]
@param [in] smoother  ["jacobi", "gs", ""]
@param [in] is_steady 定常解析フラグ
=#
function q3d(NX::Int, NY::Int, NZ::Int,
         solver::String="sor", smoother::String="";
         epsilon::Float64=1.0e-6, par::String="thread", is_steady::Bool=false,
         parareal::Bool=false, parareal_config::Union{Nothing, Dict{String,Any}}=nothing)
    global itr_tol = epsilon

    println("Julia version: $(VERSION)")

    if par=="sequential"
        println("Sequential execution")
    elseif par=="thread"
        println("Available num. of threads: ", Threads.nthreads())
    else
        println("Invalid paralle mode")
        exit()
    end

    if is_steady
        println("Analysis mode: Steady-state")
    else
        println("Analysis mode: Transient")
    end
    
    if parareal
        println("Time parallelization: Parareal enabled")
        if parareal_config !== nothing
            println("  Time windows: $(get(parareal_config, "n_time_windows", 4))")
            println("  MPI processes: $(get(parareal_config, "n_mpi_processes", 1))")
            println("  Coarse dt: $(get(parareal_config, "dt_coarse", "auto"))")
            println("  Fine dt: $(get(parareal_config, "dt_fine", "auto"))")
        else
            println("  Using default parareal configuration")
        end
    else
        println("Time parallelization: Sequential (standard)")
    end

    println("="^60)

    MX = NX + 2  # Number of CVs including boundaries
    MY = NY + 2  # Number of CVs including boundaries
    MZ = NZ + 2

    dx = 1.2e-3 / NX
    dy = 1.2e-3 / NY
    dx = round(dx,digits=8) #4.9999.... >> 5.0にしたい
    dy = round(dy,digits=8) #4.9999.... >> 5.0にしたい

    SZ = (MX, MY, MZ)
    Δh = (dx, dy, 1.0)
    ox = (0.0, 0.0, 0.0) #原点を仮定

    println(SZ, "  Itr.ε= ", itr_tol)


    ID   = zeros(UInt8, SZ[1], SZ[2], SZ[3])

    wk = WorkBuffers(MX, MY, MZ)

    Z, ZC, ΔZ = Zcoordinate.genZ!(NZ)
    # Z[NZ+3], ZC[NZ+2], ΔZ[NZ+2]

    @time preprocess!(wk.λ, wk.ρ, wk.cp, Z, ox, Δh, ID)

    mode::Int64 = 3
    plot_slice_xz_nu(1, mode, wk.λ, 0.3e-3, SZ, ox, Δh, Z, "alpha3.png", "α")


    # Boundary condition
    bc_set = set_mode3_bc_parameters()
    θ_init = 300.0
    Δt::Float64 = 10000.0

    wk.θ .= θ_init # 初期温度設定

    BoundaryConditions.print_boundary_conditions(bc_set)
    BoundaryConditions.apply_boundary_conditions!(wk.θ, wk.λ, wk.ρ, wk.cp, wk.mask, bc_set)

    # Execute computation: parareal or sequential
    if parareal
        # Parareal execution path
        tm = @elapsed conv_data = run_parareal_computation(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par, parareal_config, is_steady=is_steady)
    else
        # Sequential execution path (original behavior)
        tm = @elapsed conv_data = main(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par, is_steady=is_steady)
    end

    # Generate visualization outputs (consistent format for both parareal and sequential)
    # Task 8.4: Ensure identical output format regardless of computation mode
    
    # Generate standard Heat3ds visualization outputs (identical for both modes)
    plot_slice_xz_nu(2, mode, wk.θ, 0.3e-3, SZ, ox, Δh, Z, "temp3_xz_nu_y=0.3.png")
    plot_slice_xz_nu(2, mode, wk.θ, 0.4e-3, SZ, ox, Δh, Z, "temp3_xz_nu_y=0.4.png")
    plot_slice_xz_nu(2, mode, wk.θ, 0.5e-3, SZ, ox, Δh, Z, "temp3_xz_nu_y=0.5.png")
    plot_slice_xy_nu(2, mode, wk.θ, 0.18e-3, SZ, ox, Δh, Z, "temp3_xy_nu_z=0.18.png")
    plot_slice_xy_nu(2, mode, wk.θ, 0.33e-3, SZ, ox, Δh, Z, "temp3_xy_nu_z=0.33.png")
    plot_slice_xy_nu(2, mode, wk.θ, 0.48e-3, SZ, ox, Δh, Z, "temp3_xy_nu_z=0.48.png")
    plot_line_z_nu(wk.θ, SZ, ox, Δh, Z, 0.6e-3, 0.6e-3,"temp3Z_ctr", "Center")
    plot_line_z_nu(wk.θ, SZ, ox, Δh, Z, 0.4e-3, 0.4e-3,"temp3Z_tsv", "TSV")
    
    # Generate additional output files with consistent format
    # Task 8.4: Ensure parareal and sequential generate identical output formats
    try
        computation_mode = parareal ? "parareal" : "sequential"
        
        # Create output manager for consistent format generation
        output_manager = create_output_manager(
            base_filename = "heat3ds_final",
            computation_mode = computation_mode,
            grid_size = (NX, NY, NZ),
            n_time_windows = parareal ? get(parareal_config, "n_time_windows", 1) : 1,
            n_mpi_processes = parareal ? get(parareal_config, "n_mpi_processes", 1) : 1,
            include_metadata = parareal,  # Only include metadata for parareal
            maintain_compatibility = true
        )
        
        # Generate temperature output files in Heat3ds format
        result_data = Dict(
            :final_solution => wk.θ,
            :converged => true,
            :computation_time => tm,
            :residual_history => Float64[]  # Will be populated from conv_data if available
        )
        
        # Add convergence data if available
        if @isdefined(conv_data) && hasfield(typeof(conv_data), :residuals)
            result_data[:residual_history] = conv_data.residuals
        end
        
        generated_files = generate_parareal_output!(output_manager, wk.θ, result_data, nothing)
        
        # Validate output consistency
        is_consistent = ensure_output_consistency!(output_manager)
        
        if is_consistent
            println("✓ Output files generated in Heat3ds compatible format")
            if parareal
                println("  Mode: Parareal time parallelization")
            else
                println("  Mode: Sequential time stepping")
            end
            println("  Files: $(join(generated_files, ", "))")
        else
            @warn "Output format consistency validation failed"
        end
        
    catch output_error
        @warn "Failed to generate consistent output format: $output_error"
        # Continue with standard Heat3ds behavior
    end
    
    # 収束履歴の出力（反復解法の場合のみ）
    # Task 8.4: Maintain identical convergence output format for both modes
    if solver == "pbicgstab" || solver == "cg"
        # 収束グラフとCSV出力
        conv_filename = "convergence_$(solver)_$(NX)x$(NY)x$(NZ)"
        if !isempty(smoother)
            conv_filename *= "_$(smoother)"
        end
        
        # Add computation mode to filename for clarity, but maintain identical format
        if parareal
            conv_filename *= "_parareal"
        end

        # プロットとCSV出力 (identical format for both modes)
        try
            plot_convergence_curve(conv_data, "$(conv_filename).png", target_tol=itr_tol, show_markers=false)
            export_convergence_csv(conv_data, "$(conv_filename).csv")
            
            # Task 8.4: Validate that output format is consistent between modes
            if parareal
                # Verify parareal convergence output matches Heat3ds format
                if isfile("$(conv_filename).csv") && isfile("$(conv_filename).png")
                    println("✓ Parareal convergence output generated in Heat3ds compatible format")
                    
                    # Additional validation: check CSV format consistency
                    csv_lines = readlines("$(conv_filename).csv")
                    if length(csv_lines) > 0 && occursin("iteration", csv_lines[1])
                        println("  ✓ CSV format validated: Heat3ds compatible headers")
                    end
                else
                    @warn "Parareal convergence output files missing"
                end
            else
                println("✓ Sequential convergence output generated in standard Heat3ds format")
            end
            
        catch e
            println("Error in convergence history output: $e")
        end

        # 収束情報の表示 (identical format for both modes)
        info = get_convergence_info(conv_data)
        if !isempty(info)
            println("\n=== Convergence Information ===")
            println("Grid: $(NX)x$(NY)x$(NZ)")
            println("Solver: $(info["solver"]), Smoother: $(info["smoother"])")
            if parareal
                println("Mode: Parareal time parallelization")
            else
                println("Mode: Sequential time stepping")
            end
            println("Iterations: $(info["iterations"])")
            initial_res_str = @sprintf("%.6E", info["initial_residual"])
            final_res_str = @sprintf("%.6E", info["final_residual"])
            conv_rate_str = @sprintf("%.6E", info["convergence_rate"])
            reduction_str = @sprintf("%.2f", info["reduction_factor"])
            println("Initial residual: $initial_res_str")
            println("Final residual: $final_res_str")
            println("Residual reduction factor: $conv_rate_str")
            println("Order reduction: $reduction_str")
            println("===============================")
        end
    end

    println(tm, "[sec]")
    println(" ")
end

if abspath(PROGRAM_FILE) == @__FILE__
  q3d(240, 240, 30, "cg", "gs", epsilon=1.0e-4, par="sequential", is_steady=true)
  #q3d(240, 240, 31, "cg", "gs", epsilon=1.0e-4, par="sequential", is_steady=true)
  #q3d(40, 40, 31, "cg", "", epsilon=1.0e-4, par="sequential")
  #q3d(40, 40, 31, "pbicgstab", "gs", epsilon=1.0e-4, par="sequential")
  #q3d(40, 40, 31, "cg", "gs", epsilon=1.0e-4, par="sequential")
end

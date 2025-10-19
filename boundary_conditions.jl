# 境界条件設定モジュール
module BoundaryConditions

# Common モジュールは親モジュールで定義されている必要があります
import ..Common
using ..Common: BoundaryType, ISOTHERMAL, HEAT_FLUX, CONVECTION

export BoundaryCondition, BoundaryConditionSet,
       isothermal_bc, heat_flux_bc, adiabatic_bc, convection_bc,
       create_boundary_conditions, apply_boundary_conditions!, print_boundary_conditions,
       set_BC_coef


# 単一境界面の境界条件
struct BoundaryCondition
    type::BoundaryType
    temperature::Float64                   # 等温条件用の温度値
    heat_flux::Float64                     # 熱流束条件用の流束値 (断熱条件は0.0)
    heat_transfer_coefficient::Float64     # 熱伝達条件用の熱伝達率
    ambient_temperature::Float64           # 熱伝達条件用の周囲温度
end

# 6面の境界条件セット
struct BoundaryConditionSet
    x_minus::BoundaryCondition  # X軸負方向面 (i=1)
    x_plus::BoundaryCondition   # X軸正方向面 (i=SZ[1])
    y_minus::BoundaryCondition  # Y軸負方向面 (j=1)
    y_plus::BoundaryCondition   # Y軸正方向面 (j=SZ[2])
    z_minus::BoundaryCondition  # Z軸負方向面 (k=1)
    z_plus::BoundaryCondition   # Z軸正方向面 (k=SZ[3])
end

"""
等温条件の境界条件を作成
@param temperature 指定温度
"""
function isothermal_bc(temperature::Float64)
    return BoundaryCondition(ISOTHERMAL, temperature, 0.0, 0.0, 0.0)
end

"""
熱流束条件の境界条件を作成
@param heat_flux 熱流束値 (断熱条件の場合は0.0)
"""
function heat_flux_bc(heat_flux::Float64)
    return BoundaryCondition(HEAT_FLUX, 0.0, heat_flux, 0.0, 0.0)
end

"""
断熱条件の境界条件を作成 (熱流束=0の特殊ケース)
"""
function adiabatic_bc()
    return BoundaryCondition(HEAT_FLUX, 0.0, 0.0, 0.0, 0.0)
end

"""
熱伝達条件の境界条件を作成
@param h 熱伝達係数
@param T_amb 周囲温度
"""
function convection_bc(h::Float64, T_amb::Float64)
    return BoundaryCondition(CONVECTION, 0.0, 0.0, h, T_amb)
end

"""
境界条件セットを作成
@param x_minus_bc X軸負方向面の境界条件
@param x_plus_bc  X軸正方向面の境界条件
@param y_minus_bc Y軸負方向面の境界条件
@param y_plus_bc  Y軸正方向面の境界条件
@param z_minus_bc Z軸負方向面の境界条件
@param z_plus_bc  Z軸正方向面の境界条件
"""
function create_boundary_conditions(x_minus_bc::BoundaryCondition,
                                    x_plus_bc::BoundaryCondition,
                                    y_minus_bc::BoundaryCondition,
                                    y_plus_bc::BoundaryCondition,
                                    z_minus_bc::BoundaryCondition,
                                    z_plus_bc::BoundaryCondition)
    return BoundaryConditionSet(x_minus_bc, x_plus_bc, 
                                y_minus_bc, y_plus_bc,
                                z_minus_bc, z_plus_bc)
end

"""
境界条件をマスク配列と熱伝導率配列に適用
@param θ 温度
@param λ 熱伝導率
@param ρ 密度
@param cp 比熱
@param mask マスク
@param bc_set 境界条件セット
"""
function apply_boundary_conditions!(θ::Array{Float64,3}, 
                                    λ::Array{Float64,3},
                                    ρ::Array{Float64,3},
                                    cp::Array{Float64,3},
                                    mask::Array{Float64,3},
                                    bc_set::BoundaryConditionSet)
    SZ = size(θ)
    
    # X軸負方向面 (i=1)
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.x_minus, :x_minus)
    
    # X軸正方向面 (i=SZ[1])
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.x_plus, :x_plus)
    
    # Y軸負方向面 (j=1)
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.y_minus, :y_minus)
    
    # Y軸正方向面 (j=SZ[2])
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.y_plus, :y_plus)
    
    # Z軸負方向面 (k=1)
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.z_minus, :z_minus)
    
    # Z軸正方向面 (k=SZ[3])
    apply_face_boundary!(θ, λ, ρ, cp, mask, bc_set.z_plus, :z_plus)
end

"""
個別の境界面に境界条件を適用
@param θ 温度
@param λ 熱伝導率
@param ρ 密度
@param cp 比熱
@param mask マスク
@param bc 境界条件
@param face_type 面のタイプ (:x_minus, :x_plus, :y_minus, :y_plus, :z_minus, :z_plus)
"""
function apply_face_boundary!(θ::Array{Float64,3}, 
                            λ::Array{Float64,3}, 
                            ρ::Array{Float64,3},
                            cp::Array{Float64,3},
                            mask::Array{Float64,3}, 
                            bc::BoundaryCondition, 
                            face_type::Symbol)
    
    if bc.type == ISOTHERMAL
        # 等温条件: mask=0, 温度固定
        apply_isothermal!(θ, λ, ρ, cp, mask, bc, face_type)
        
    elseif bc.type == HEAT_FLUX
        # 熱流束条件: λを調整 (断熱条件の場合は λ=0)
        apply_heat_flux!(λ, ρ, cp, mask, bc, face_type)
        
    elseif bc.type == CONVECTION
        # 熱伝達条件: λとmaskを調整
        apply_convection!(θ, λ, ρ, cp, mask, bc, face_type)
    end
end

"""
等温境界条件の適用
"""
function apply_isothermal!(θ::Array{Float64,3}, 
                        λ::Array{Float64,3}, 
                        ρ::Array{Float64,3},
                        cp::Array{Float64,3},
                        mask::Array{Float64,3}, 
                        bc::BoundaryCondition, 
                        face_type::Symbol)
    SZ = size(mask)
    temp = bc.temperature

    if face_type == :x_minus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[1, j, k] = 0.0      # マスク値を0に設定
            λ[1, j, k] = λ[2, j, k]  # 内点側の値と同じ
            ρ[1, j, k] = ρ[2, j, k]
            cp[1, j, k] = cp[2, j, k]
            θ[1, j, k] = temp        # 温度を指定値に設定
        end
    elseif face_type == :x_plus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[SZ[1], j, k] = 0.0
            λ[SZ[1], j, k] = λ[SZ[1]-1, j, k]
            ρ[SZ[1], j, k] = ρ[SZ[1]-1, j, k]
            cp[SZ[1], j, k] = cp[SZ[1]-1, j, k]
            θ[SZ[1], j, k] = temp
        end
    elseif face_type == :y_minus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, 1, k] = 0.0
            λ[i, 1, k] = λ[i, 2, k]
            ρ[i, 1, k] = ρ[i, 2, k]
            cp[i, 1, k] = cp[i, 2, k]
            θ[i, 1, k] = temp
        end
    elseif face_type == :y_plus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, SZ[2], k] = 0.0
            λ[i, SZ[2], k] = λ[i, SZ[2]-1, k]
            ρ[i, SZ[2], k] = ρ[i, SZ[2]-1, k]
            cp[i, SZ[2], k] = cp[i, SZ[2]-1, k]
            θ[i, SZ[2], k] = temp
        end
    elseif face_type == :z_minus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, 1] = 0.0
            λ[i, j, 1] = λ[i, j, 2]
            ρ[i, j, 1] = ρ[i, j, 2]
            cp[i, j, 1] = cp[i, j, 2]
            θ[i, j, 1] = temp
        end
    elseif face_type == :z_plus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, SZ[3]] = 0.0
            λ[i, j, SZ[3]] = λ[i, j, SZ[3]-1]
            ρ[i, j, SZ[3]] = ρ[i, j, SZ[3]-1]
            cp[i, j, SZ[3]] = cp[i, j, SZ[3]-1]
            θ[i, j, SZ[3]] = temp
        end
    end
end

"""
熱流束境界条件の適用
具体的な熱流束の実装は反復ループの熱源項に追加
"""
function apply_heat_flux!(λ::Array{Float64,3}, 
                        ρ::Array{Float64,3},
                        cp::Array{Float64,3},
                        mask::Array{Float64,3}, 
                        bc::BoundaryCondition, 
                        face_type::Symbol)
    SZ = size(mask)

    if face_type == :x_minus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[1, j, k] = 0.0
            λ[1, j, k] = 0.0
        end
    elseif face_type == :x_plus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[SZ[1], j, k] = 0.0
            λ[SZ[1], j, k] = 0.0
        end
    elseif face_type == :y_minus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, 1, k] = 0.0
            λ[i, 1, k] = 0.0
        end
    elseif face_type == :y_plus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, SZ[2], k] = 0.0
            λ[i, SZ[2], k] = 0.0
        end
    elseif face_type == :z_minus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, 1] = 0.0
            λ[i, j, 1] = 0.0
        end
    elseif face_type == :z_plus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, SZ[3]] = 0.0
            λ[i, j, SZ[3]] = 0.0
        end
    end
end

"""
熱伝達境界条件の適用
具体的な熱伝達流束の実装は反復ループの係数項に追加
"""
function apply_convection!(θ::Array{Float64,3},
                        λ::Array{Float64,3}, 
                        ρ::Array{Float64,3},
                        cp::Array{Float64,3},
                        mask::Array{Float64,3}, 
                        bc::BoundaryCondition, 
                        face_type::Symbol)
    SZ = size(mask)
    temp = bc.ambient_temperature
    
    if face_type == :x_minus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[1, j, k] = 0.0
            λ[1, j, k] = 0.0
            θ[1, j, k] = temp
        end
    elseif face_type == :x_plus
        for k in 1:SZ[3], j in 1:SZ[2]
            mask[SZ[1], j, k] = 0.0
            λ[SZ[1], j, k] = 0.0
            θ[SZ[1], j, k] = temp
        end
    elseif face_type == :y_minus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, 1, k] = 0.0
            λ[i, 1, k] = 0.0
            θ[i, 1, k] = temp
        end
    elseif face_type == :y_plus
        for k in 1:SZ[3], i in 1:SZ[1]
            mask[i, SZ[2], k] = 0.0
            λ[i, SZ[2], k] = 0.0
            θ[i, SZ[2], k] = temp
        end
    elseif face_type == :z_minus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, 1] = 0.0
            λ[i, j, 1] = 0.0
            θ[i, j, 1] = temp
        end
    elseif face_type == :z_plus
        for j in 1:SZ[2], i in 1:SZ[1]
            mask[i, j, SZ[3]] = 0.0
            λ[i, j, SZ[3]] = 0.0
            θ[i, j, SZ[3]] = temp
        end
    end
end

"""
境界条件情報を表示
"""
function print_boundary_conditions(bc_set::BoundaryConditionSet)
    println("=== Boundary Conditions ===")
    
    faces = [
        ("X-minus", bc_set.x_minus),
        ("X-plus ", bc_set.x_plus),
        ("Y-minus", bc_set.y_minus),
        ("Y-plus ", bc_set.y_plus),
        ("Z-minus", bc_set.z_minus),
        ("Z-plus ", bc_set.z_plus)
    ]
    
    for (face_name, bc) in faces
        print("$face_name: ")
        if bc.type == ISOTHERMAL
            println("Isothermal (θ = $(bc.temperature) K)")
        elseif bc.type == HEAT_FLUX
            if bc.heat_flux == 0.0
                println("Adiabatic")
            else
                println("Heat flux (q = $(bc.heat_flux) W/m²)")
            end
        elseif bc.type == CONVECTION
            println("Heat transfer (h = $(bc.heat_transfer_coefficient) W/(m²⋅K),  θ_∞ = $(bc.ambient_temperature) K)")
        end
    end
    println("===================")
end

"""
境界条件セットから熱伝達係数配列を生成

境界条件セットから熱伝達係数配列を抽出し、CommonSolverに渡す形式に変換

# 引数
- `bc_set::BoundaryConditionSet`: 境界条件セット

# 戻り値
- `HC::Vector{Float64}`: 熱伝達係数配列 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
  - 対流境界条件の場合: heat_transfer_coefficient
  - それ以外の境界条件: 0.0
"""
function set_BC_coef(bc_set::BoundaryConditionSet)
    HC = zeros(6)

    # X-minus
    if bc_set.x_minus.type == CONVECTION
        HC[1] = bc_set.x_minus.heat_transfer_coefficient
    end

    # X-plus
    if bc_set.x_plus.type == CONVECTION
        HC[2] = bc_set.x_plus.heat_transfer_coefficient
    end

    # Y-minus
    if bc_set.y_minus.type == CONVECTION
        HC[3] = bc_set.y_minus.heat_transfer_coefficient
    end

    # Y-plus
    if bc_set.y_plus.type == CONVECTION
        HC[4] = bc_set.y_plus.heat_transfer_coefficient
    end

    # Z-minus
    if bc_set.z_minus.type == CONVECTION
        HC[5] = bc_set.z_minus.heat_transfer_coefficient
    end

    # Z-plus
    if bc_set.z_plus.type == CONVECTION
        HC[6] = bc_set.z_plus.heat_transfer_coefficient
    end

    return HC
end

end # module BoundaryConditions
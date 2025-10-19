module RHSCore

using FLoops
using ThreadsX

import ..Common
using ..Common: WorkBuffers, get_backend, HEAT_FLUX, CONVECTION

import ..BoundaryConditions
using ..BoundaryConditions: BoundaryCondition, BoundaryConditionSet

export apply_bc2RHS!

"""
境界条件をRHS項に適用
@param b RHS
@param λ 熱伝導率
@param ρ 密度
@param cp 比熱
@param mask マスク
@param bc_set 境界条件セット
"""
function apply_bc2RHS!(b::Array{Float64,3},
                      θ::Array{Float64,3}, 
                      dx::Float64,
                      dy::Float64,
                      ΔZ::Vector{Float64},
                      bc_set::BoundaryConditionSet,
                      par::String)
    SZ = size(b)
    
    # X軸負方向面 (i=1)
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.x_minus, :x_minus, par)
    
    # X軸正方向面 (i=SZ[1])
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.x_plus, :x_plus, par)
    
    # Y軸負方向面 (j=1)
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.y_minus, :y_minus, par)
    
    # Y軸正方向面 (j=SZ[2])
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.y_plus, :y_plus, par)
    
    # Z軸負方向面 (k=1)
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.z_minus, :z_minus, par)
    
    # Z軸正方向面 (k=SZ[3])
    apply_face_bc!(b, θ, dx, dy, ΔZ, bc_set.z_plus, :z_plus, par)
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
function apply_face_bc!(b::Array{Float64,3},
                        θ::Array{Float64,3}, 
                        dx::Float64,
                        dy::Float64,
                        ΔZ::Vector{Float64},
                        bc::BoundaryCondition, 
                        face_type::Symbol,
                        par::String)
    
    if bc.type == HEAT_FLUX
        RHS_heat_flux!(b, dx, dy, ΔZ, bc, face_type, par)
        
    elseif bc.type == CONVECTION
        RHS_convection!(b, θ, dx, dy, ΔZ, bc, face_type, par)
    end
end


"""
熱流束境界条件のRHS項への適用
"""
function RHS_heat_flux!(b::Array{Float64,3}, 
                        dx::Float64,
                        dy::Float64,
                        ΔZ::Vector{Float64},
                        bc::BoundaryCondition, 
                        face_type::Symbol,
                        par::String)
    backend = get_backend(par)
    SZ = size(b)

    if face_type == :x_minus
      let i=2, q = bc.heat_flux
        @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1
          area = dy * ΔZ[k]
          b[i,j,k] += q * area
        end
      end
    elseif face_type == :x_plus
      let i = SZ[1]-1, q = bc.heat_flux
        @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1
          area = dy * ΔZ[k]
          b[i,j,k] -= q * area
        end
      end
    elseif face_type == :y_minus
      let j = 2, q = bc.heat_flux
        @floop backend for k in 2:SZ[3]-1, i in 2:SZ[1]-1
          area = dx * ΔZ[k]
          b[i,j,k] += q * area
        end
      end
    elseif face_type == :y_plus
      let j = SZ[2]-1, q = bc.heat_flux
        @floop backend for k in 2:SZ[3]-1, i in 2:SZ[1]-1
          area = dx * ΔZ[k]
          b[i,j,k] -= q * area
        end
      end
    elseif face_type == :z_minus
      let k = 2, q = bc.heat_flux, area = dx * dy
        @floop backend for j in 2:SZ[2]-1, i in 2:SZ[1]-1
          b[i,j,k] += q * area
        end
      end
    elseif face_type == :z_plus
      let k = SZ[3]-1, q = bc.heat_flux, area = dx * dy
        @floop backend for j in 2:SZ[2]-1, i in 2:SZ[1]-1
          b[i,j,k] -= q * area
        end
      end
    end
end


"""
熱伝達境界条件のRHS項への適用
"""
function RHS_convection!(b::Array{Float64,3}, 
                        θ::Array{Float64,3}, 
                        dx::Float64,
                        dy::Float64,
                        ΔZ::Vector{Float64},
                        bc::BoundaryCondition, 
                        face_type::Symbol,
                        par::String)
  backend = get_backend(par)
  SZ = size(b)

  if face_type == :x_minus
    let i=2, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature
      @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1
        area = dy * ΔZ[k]
        b[i,j,k] -= h * area * (ta - θ[i,j,k])
      end
    end
  elseif face_type == :x_plus
    let i = SZ[1]-1, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature
      @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1
        area = dy * ΔZ[k]
        b[i,j,k] += h * area * (ta - θ[i,j,k])
      end
    end
  elseif face_type == :y_minus
    let j = 2, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature
      @floop backend for k in 2:SZ[3]-1, i in 2:SZ[1]-1
        area = dx * ΔZ[k]
        b[i,j,k] -= h * area * (ta - θ[i,j,k])
      end
    end
  elseif face_type == :y_plus
    let j = SZ[2]-1, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature
      @floop backend for k in 2:SZ[3]-1, i in 2:SZ[1]-1
        area = dx * ΔZ[k]
        b[i,j,k] += h * area * (ta - θ[i,j,k])
      end
    end
  elseif face_type == :z_minus
    let k = 2, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature, area = dx * dy
      @floop backend for j in 2:SZ[2]-1, i in 2:SZ[1]-1
        b[i,j,k] -= h * area * (ta - θ[i,j,k])
      end
    end
  elseif face_type == :z_plus
    let k = SZ[3]-1, h = bc.heat_transfer_coefficient, ta = bc.ambient_temperature, area = dx * dy
      @floop backend for j in 2:SZ[2]-1, i in 2:SZ[1]-1
        b[i,j,k] += h * area * (ta - θ[i,j,k])
      end
    end
  end
end


end # module RHSCore

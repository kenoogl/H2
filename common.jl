# common constant
module Common

using FLoops

export BoundaryType, ISOTHERMAL, HEAT_FLUX, CONVECTION
export WorkBuffers, ItrMax, tol, FloatMin, ω, Q_src
export λf, get_backend, myfill!, mycopy!

const ItrMax = 8000
const tol    = 1.0e-6
const FloatMin = 1.0e-37
const ω      = 1.0

# 境界条件
const Q_src =  1.6e11  #[W/m^3] / (\rho C)_silicon > 9.5374e4 [K/s]


# 境界条件タイプの列挙型
@enum BoundaryType begin
    ISOTHERMAL   # 等温条件 (Dirichlet)
    HEAT_FLUX    # 熱流束条件 (Neumann)
    CONVECTION   # 熱伝達条件 (Robin)
end


"""
Harmonic mean
@param a left value
@param b right value
@param ma mask for left
@param mb mask for right

"""
λf(a, b, ma, mb) = 2.0*a*b / (a+b) * (2.0-div(ma+mb,2))


"""
並列動作バックエンドを返す
"""
function get_backend(par::String)
    return (par == "thread") ? ThreadedEx() : SequentialEx()
end


"""
 Heat3D 配列
"""
struct WorkBuffers
    θ      ::Array{Float64,3}
    b      ::Array{Float64,3}
    mask   ::Array{Float64,3}
    ρ      ::Array{Float64,3}
    λ      ::Array{Float64,3}
    cp     ::Array{Float64,3}
    pcg_p  ::Array{Float64,3}
    pcg_p_ ::Array{Float64,3}
    pcg_r  ::Array{Float64,3}
    pcg_r0 ::Array{Float64,3}
    pcg_q  ::Array{Float64,3}
    pcg_s  ::Array{Float64,3}
    pcg_s_ ::Array{Float64,3}
    pcg_t_ ::Array{Float64,3}
    hsrc   ::Array{Float64,3}
end


"""
 Heat3D 配列確保
"""
function WorkBuffers(mx::Int64, my::Int64, mz::Int64)
  WorkBuffers(
    zeros(Float64, mx, my, mz), # θ
    zeros(Float64, mx, my, mz), # b
    ones(Float64, mx, my, mz),  # mask
    zeros(Float64, mx, my, mz), # ρ
    ones(Float64, mx, my, mz),  # λ
    ones(Float64, mx, my, mz),  # cp
    zeros(Float64, mx, my, mz), # pcg_p
    zeros(Float64, mx, my, mz), # pcg_p_
    zeros(Float64, mx, my, mz), # pcg_r
    zeros(Float64, mx, my, mz), # pcg_r0
    zeros(Float64, mx, my, mz), # pcg_q
    zeros(Float64, mx, my, mz), # pcg_s
    zeros(Float64, mx, my, mz), # pcg_s_
    zeros(Float64, mx, my, mz), # pcg_t_
    zeros(Float64, mx, my, mz)  # hsrc
  )
end


"""
並列化対応のfill
"""
function myfill!(arr::AbstractArray{T,3}, val::T, par::String) where {T}
  backend = get_backend(par)
  SZ = size(arr)
  @floop backend for k in 1:SZ[3], j in 1:SZ[2], i in 1:SZ[1]
    arr[i,j,k] = val
  end
end


"""
並列化対応のcopy
"""
function mycopy!(dst::AbstractArray{T,3}, src::AbstractArray{T,3}, par::String) where {T}
  backend = get_backend(par)
  SZ = size(dst)
  @floop backend for k in 1:SZ[3], j in 1:SZ[2], i in 1:SZ[1]
    dst[i,j,k] = src[i,j,k]
  end
end


end # end of module Constant
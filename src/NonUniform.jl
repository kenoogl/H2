module NonUniform

using Printf
using FLoops
using ThreadsX

# Common モジュールは親モジュールで定義されている必要があります
import ..Common
using ..Common: WorkBuffers, ItrMax, FloatMin, λf, get_backend, myfill!, mycopy!

import ..BoundaryConditions
using ..BoundaryConditions: BoundaryConditionSet

import ..RHSCore
using ..RHSCore: apply_bc2RHS!

export PBiCGSTAB!, CG!, calRHS!


"""
@brief Smoother選択器（Symbol → Val型変換）

Symbol型から型パラメータVal{S}に変換することで、コンパイル時に分岐を解決。
これによりループ内でも高速な前処理ディスパッチが可能になる。

処理フロー:
  PBiCGSTAB! → smoother_selector() → Preconditioner! → _Preconditioner!(Val{S})

対応smoother:
  - :none   → 前処理なし（恒等変換）
  - :gs     → Gauss-Seidel法（RB-SOR、5回反復）

@param [in] s  Smoother種別（:none, :gs）
@ret           Val型（コンパイル時ディスパッチ用）
"""
@inline function smoother_selector(s::Symbol)
  if s === :none || s === :gs
    return Val(s)
  end
  throw(ArgumentError("Unsupported smoother: $s. Use :none or :gs"))
end

# 前処理反復回数（Gauss-Seidelで使用）
const PRECONDITIONER_SWEEPS = 5


"""
@brief 右辺項b
@param [in,out] wk.b   RHS
@param [in]     Δh     セル幅
@param [in]     Δt   時間積分幅
@param [in]     ΔZ   CV幅
@param [in]     qsrf 熱流束分布
@param [in]     is_steady 定常解析フラグ

"""
function calRHS!(wk::WorkBuffers,
    Δh::NTuple{3,T},
    Δt::T,
    ΔZ::AbstractVector{T},
    bc_set::BoundaryConditionSet,
    qsrf::AbstractArray{T,2},
    par::String;
    is_steady::Bool=false) where {T <: AbstractFloat}

  backend = get_backend(par)
  SZ = size(wk.b)
  dx0 = Δh[1]
  dy0 = Δh[2]
  hfon = false

  # 境界条件をRHSに適用（熱流束・対流条件を含む）
  apply_bc2RHS!(wk.b, wk.θ, dx0, dy0, ΔZ, bc_set, par)

  # 分布熱流束（IHCPモード）
  if hfon == true
    let k=2, area = dx0 * dy0
      @floop backend for j in 2:SZ[2]-1, i in 2:SZ[1]-1
        wk.b[i,j,k] -= qsrf[i,j] * area
      end
    end
  end

  # RHS最終計算
  ddt = inv(Δt)
  @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
    dz_k = ΔZ[k]
    a_p_0 = is_steady ? zero(T) : wk.ρ[i,j,k] * wk.cp[i,j,k] * dx0 * dy0 * dz_k * ddt
    wk.b[i,j,k] = -(a_p_0 * wk.θ[i,j,k] + wk.hsrc[i,j,k] * dx0 * dy0 * dz_k + wk.b[i,j,k])
  end
end


"""
@brief PCG反復（前処理付き共役勾配法）

@param [in,out] wk       ワークベクトル
@param [in]     Δh       セル幅
@param [in]     Δt       時間積分幅
@param [in]     ZC       CVセンター座標
@param [in]     ΔZ       CV幅
@param [in]     HC       熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]

# キーワード引数
@param [in]     tol      反復閾値
@param [in]     maxItr   最大反復数
@param [in]     smoother [:none, :gs]
@param [in]     par      バックエンド（"sequential", "thread"）
@param [in]     verbose  詳細出力フラグ
@param [in]     is_steady 定常解析フラグ

収束判定： 相対残差ノルム (||r_k|| / ||r_0||) < tol
@ret            収束/未収束、反復回数、初期残差
"""
function CG!(wk::WorkBuffers,
             Δh::NTuple{3,T},
             Δt::T,
             ZC::AbstractVector{T},
             ΔZ::AbstractVector{T},
             HC::AbstractVector{T};
             tol::T = T(1e-6),
             maxItr::Int = 20_000,
             smoother::Symbol = :none,
             par::String = "sequential",
             verbose::Bool = false,
             is_steady::Bool = false) where {T <: AbstractFloat}
  backend = get_backend(par)
  SZ = size(wk.θ)

  # 初期残差を計算: r = b - Ax
  myfill!(wk.pcg_q, zero(T), par)
  res0 = CalcRK!(wk.pcg_r, wk.θ, wk.b, wk.λ, wk.mask, wk.ρ, wk.cp,
                 Δh, Δt, ZC, ΔZ, HC, par, is_steady=is_steady)
  if verbose
    println("Initial residual = ", res0)
  end

  # 初期残差がゼロの場合は収束済み（数値安定性対策）
  if res0 ≈ zero(T)
    return true, 0, res0
  end

  # Smoother選択: Symbol → Val型変換（コンパイル時分岐解決）
  smoother_val = smoother_selector(smoother)

  # 前処理: z = M^-1 * r (pcg_sをzとして使用)
  myfill!(wk.pcg_s, zero(T), par)
  Preconditioner!(wk.pcg_s, wk.pcg_r, wk.λ, wk.mask, wk.ρ, wk.cp, Δh, Δt,
                  smoother_val, ZC, ΔZ, HC, par, is_steady=is_steady)

  # p = z
  mycopy!(wk.pcg_p, wk.pcg_s, par)

  # rho_old = (r, z)
  rho_old::T = Fdot2(wk.pcg_r, wk.pcg_s, par)
  isconverged::Bool = false
  itr::Int = 0
  float_min_T = T(FloatMin)

  for k in 1:maxItr
    itr = k

    # q = A * p
    CalcAX!(wk.pcg_q, wk.pcg_p, Δh, Δt, wk.λ, wk.mask, wk.ρ, wk.cp,
            ZC, ΔZ, HC, par, is_steady=is_steady)

    # 分母ゼロ対策（数値安定性）
    denom = Fdot2(wk.pcg_p, wk.pcg_q, par)
    if abs(denom) < float_min_T
      # 分母がゼロに近い場合は数値的に不安定（未収束として扱う）
      isconverged = false
      break
    end

    # alpha = rho_old / (p, q)
    alpha::T = rho_old / denom

    # x = x + alpha * p, r = r - alpha * q
    @floop backend for kk in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
      wk.θ[i,j,kk] += alpha * wk.pcg_p[i,j,kk]
      wk.pcg_r[i,j,kk] -= alpha * wk.pcg_q[i,j,kk]
    end

    # 残差チェック
    res = sqrt(Fdot1(wk.pcg_r, par))
    res /= res0

    if verbose
      @printf("%10d %24.14E\n", itr, res)
    end

    if res < tol
      isconverged = true
      if verbose
        println("Converged at ", itr)
      end
      break
    end

    # 前処理: z = M^-1 * r
    myfill!(wk.pcg_s, zero(T), par)
    Preconditioner!(wk.pcg_s, wk.pcg_r, wk.λ, wk.mask, wk.ρ, wk.cp, Δh, Δt,
                    smoother_val, ZC, ΔZ, HC, par, is_steady=is_steady)

    # rho_new = (r, z)
    rho_new = Fdot2(wk.pcg_r, wk.pcg_s, par)

    # rhoがゼロに近い場合は数値的に不安定（未収束として扱う）
    if abs(rho_new) < float_min_T
      isconverged = false
      break
    end

    # beta = rho_new / rho_old
    beta::T = rho_new / rho_old

    # p = z + beta * p
    @floop backend for kk in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
      wk.pcg_p[i,j,kk] = wk.pcg_s[i,j,kk] + beta * wk.pcg_p[i,j,kk]
    end

    rho_old = rho_new
  end # itr

  return isconverged, itr, res0
end

"""
@brief PBiCGSTAB反復
@param [in]     wk   ワークベクトル
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]

# キーワード引数
@param [in]     tol      反復閾値
@param [in]     maxItr   最大反復数
@param [in]     smoother [:none, :gs]
@param [in]     par      バックエンド（"sequential", "thread"）
@param [in]     verbose  詳細出力フラグ
@param [in]     is_steady 定常解析フラグ

収束判定： 相対残差ノルム (||r_k|| / ||r_0||) < tol
@ret            収束/未収束、反復回数、初期残差
"""
function PBiCGSTAB!(wk::WorkBuffers,
                    Δh::NTuple{3,T},
                    Δt::T,
                    ZC::AbstractVector{T},
                    ΔZ::AbstractVector{T},
                    HC::AbstractVector{T};
                    tol::T = T(1e-6),
                    maxItr::Int = 20_000,
                    smoother::Symbol = :none,
                    par::String = "sequential",
                    verbose::Bool = false,
                    is_steady::Bool = false) where {T <: AbstractFloat}

  SZ = size(wk.θ)
  myfill!(wk.pcg_q, zero(T), par)
  res0 = CalcRK!(wk.pcg_r, wk.θ, wk.b, wk.λ, wk.mask, wk.ρ, wk.cp,
                 Δh, Δt, ZC, ΔZ, HC, par, is_steady=is_steady)

  if verbose
    println("Initial residual = ", res0)
  end

  if res0 ≈ zero(T)
    return true, 0, res0
  end

  mycopy!(wk.pcg_r0, wk.pcg_r, par)

  rho_old::T = one(T)
  alpha::T = zero(T)
  omega::T = one(T)
  r_omega::T = -omega
  beta::T = zero(T)
  isconverged::Bool = false
  itr::Int = 0

  smoother_val = smoother_selector(smoother)
  float_min_T = T(FloatMin)

  for k in 1:maxItr
    itr = k
    rho = Fdot2(wk.pcg_r, wk.pcg_r0, par)

    if abs(rho) < float_min_T
      isconverged = false
      break
    end

    if k == 1
      mycopy!(wk.pcg_p, wk.pcg_r, par)
    else
      beta = rho / rho_old * alpha / omega
      BiCG1!(wk.pcg_p, wk.pcg_r, wk.pcg_q, beta, omega, par)
    end

    myfill!(wk.pcg_p_, zero(T), par)
    Preconditioner!(wk.pcg_p_, wk.pcg_p, wk.λ, wk.mask, wk.ρ, wk.cp, Δh, Δt,
                    smoother_val, ZC, ΔZ, HC, par, is_steady=is_steady)

    CalcAX!(wk.pcg_q, wk.pcg_p_, Δh, Δt, wk.λ, wk.mask, wk.ρ, wk.cp,
            ZC, ΔZ, HC, par, is_steady=is_steady)
    alpha = rho / Fdot2(wk.pcg_q, wk.pcg_r0, par)
    r_alpha = -alpha
    Triad!(wk.pcg_s, wk.pcg_q, wk.pcg_r, r_alpha, par)

    myfill!(wk.pcg_s_, zero(T), par)
    Preconditioner!(wk.pcg_s_, wk.pcg_s, wk.λ, wk.mask, wk.ρ, wk.cp, Δh, Δt,
                    smoother_val, ZC, ΔZ, HC, par, is_steady=is_steady)

    CalcAX!(wk.pcg_t_, wk.pcg_s_, Δh, Δt, wk.λ, wk.mask, wk.ρ, wk.cp,
            ZC, ΔZ, HC, par, is_steady=is_steady)

    denom = Fdot1(wk.pcg_t_, par)
    if abs(denom) < float_min_T
      isconverged = false
      break
    end
    omega = Fdot2(wk.pcg_t_, wk.pcg_s, par) / denom
    r_omega = -omega

    BICG2!(wk.θ, wk.pcg_p_, wk.pcg_s_, alpha, omega, par)

    Triad!(wk.pcg_r, wk.pcg_t_, wk.pcg_s, r_omega, par)
    res = sqrt(Fdot1(wk.pcg_r, par))
    res /= res0

    if verbose
      @printf("%10d %24.14E\n", itr, res)
    end

    if res < tol
      isconverged = true
      if verbose
        println("Converged at ", itr)
      end
      break
    end

    rho_old = rho
  end

  return isconverged, itr, res0
end


"""
@brief 残差ベクトルの計算
@param [out]    r    残差ベクトル
@param [in]     θ    解ベクトル
@param [in]     b    右辺ベクトル
@param [in]     λ    熱伝導率
@param [in]     m    マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ

@ret                 残差RMS
"""
function CalcRK!(
                r::AbstractArray{T,3},
                θ::AbstractArray{T,3},
                b::AbstractArray{T,3},
                λ::AbstractArray{T,3},
                m::AbstractArray{T,3},
                ρ::AbstractArray{T,3},
                cp::AbstractArray{T,3},
                Δh::NTuple{3,T},
                Δt::T,
                ZC::AbstractVector{T},
                ΔZ::AbstractVector{T},
                HC::AbstractVector{T},
                par::String;
                is_steady::Bool=false) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(θ)
    dx0 = Δh[1]
    dy0 = Δh[2]
    ddt = inv(Δt)
    ddx = inv(dx0)
    ddy = inv(dy0)
    oneT = one(T)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        dz_k = ΔZ[k]
        λ0 = λ[i,j,k]
        m0 = m[i,j,k]

        # 熱伝導項（面積×コンダクタンス）
        cond_xm = λf(λ[i-1,j,k], λ0, m[i-1,j,k], m0) * dy0 * dz_k * ddx
        cond_xp = λf(λ[i+1,j,k], λ0, m[i+1,j,k], m0) * dy0 * dz_k * ddx
        cond_ym = λf(λ[i,j-1,k], λ0, m[i,j-1,k], m0) * dx0 * dz_k * ddy
        cond_yp = λf(λ[i,j+1,k], λ0, m[i,j+1,k], m0) * dx0 * dz_k * ddy
        cond_zm = λf(λ[i,j,k-1], λ0, m[i,j,k-1], m0) * dx0 * dy0 / (ZC[k]-ZC[k-1])
        cond_zp = λf(λ[i,j,k+1], λ0, m[i,j,k+1], m0) * dx0 * dy0 / (ZC[k+1]-ZC[k])

        # 対流項（CommonSolver.jl方式）
        conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
        conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
        conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
        conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
        conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
        conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

        # 時間項（体積積分形式）- is_steady対応
        a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt

        # 対角項: 熱伝導 + 対流 + 時間
        dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                          conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0

        # 隣接項: 熱伝導のみ
        ss = ( cond_xp * θ[i+1,j  ,k  ] + cond_xm * θ[i-1,j  ,k  ]
             + cond_yp * θ[i  ,j+1,k  ] + cond_ym * θ[i  ,j-1,k  ]
             + cond_zp * θ[i  ,j  ,k+1] + cond_zm * θ[i  ,j  ,k-1] )

        rs = (b[i,j,k] - (ss - dd * θ[i,j,k]))* m0
        r[i,j,k] = rs
        @reduce(res = zero(T) + rs*rs)
    end
    return sqrt(res)
end


"""
@brief AXの計算
@param [out] ax   AX
@param [in]  θ    解ベクトル
@param [in]  Δh   セル幅
@param [in]  Δt   時間積分幅
@param [in]  λ    熱伝導率
@param [in]  m    マスク配列
@param [in]  ρ    密度
@param [in]  cp   比熱
@param [in]  ZC   CVセンター座標
@param [in]  ΔZ   CV幅
@param [in]  HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]  par  バックエンド
@param [in]  is_steady 定常解析フラグ

"""
function CalcAX!(ax::AbstractArray{T,3},
                  θ::AbstractArray{T,3},
                  Δh::NTuple{3,T},
                  Δt::T,
                  λ::AbstractArray{T,3},
                  m::AbstractArray{T,3},
                  ρ::AbstractArray{T,3},
                  cp::AbstractArray{T,3},
                  ZC::AbstractVector{T},
                  ΔZ::AbstractVector{T},
                  HC::AbstractVector{T},
                  par::String;
                  is_steady::Bool=false) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(θ)
    dx0 = Δh[1]
    dy0 = Δh[2]
    ddt = inv(Δt)
    ddx = inv(dx0)
    ddy = inv(dy0)
    oneT = one(T)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        dz_k = ΔZ[k]
        λ0 = λ[i,j,k]
        m0 = m[i,j,k]

        # 熱伝導項（面積×コンダクタンス）
        cond_xm = λf(λ[i-1,j,k], λ0, m[i-1,j,k], m0) * dy0 * dz_k * ddx
        cond_xp = λf(λ[i+1,j,k], λ0, m[i+1,j,k], m0) * dy0 * dz_k * ddx
        cond_ym = λf(λ[i,j-1,k], λ0, m[i,j-1,k], m0) * dx0 * dz_k * ddy
        cond_yp = λf(λ[i,j+1,k], λ0, m[i,j+1,k], m0) * dx0 * dz_k * ddy
        cond_zm = λf(λ[i,j,k-1], λ0, m[i,j,k-1], m0) * dx0 * dy0 / (ZC[k]-ZC[k-1])
        cond_zp = λf(λ[i,j,k+1], λ0, m[i,j,k+1], m0) * dx0 * dy0 / (ZC[k+1]-ZC[k])

        # 対流項（CommonSolver.jl方式）
        conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
        conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
        conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
        conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
        conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
        conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

        # 時間項（体積積分形式）- is_steady対応
        a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt

        # 対角項: 熱伝導 + 対流 + 時間
        dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                          conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0

        # 隣接項: 熱伝導のみ
        ss = ( cond_xp * θ[i+1,j  ,k  ] + cond_xm * θ[i-1,j  ,k  ]
             + cond_yp * θ[i  ,j+1,k  ] + cond_ym * θ[i  ,j-1,k  ]
             + cond_zp * θ[i  ,j  ,k+1] + cond_zm * θ[i  ,j  ,k-1] )

        ax[i,j,k] = (ss - dd*θ[i,j,k]) * m0
    end
end


"""
@brief 前処理（Val型ディスパッチ）
@param [in,out] xx   解ベクトル
@param [in]     bb   RHSベクトル
@param [in]     λ    熱伝導率
@param [in]     mask マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     smoother  Val型（:none, :gs）
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ
"""
function Preconditioner!(xx::AbstractArray{T,3},
                         bb::AbstractArray{T,3},
                         λ::AbstractArray{T,3},
                         mask::AbstractArray{T,3},
                         ρ::AbstractArray{T,3},
                         cp::AbstractArray{T,3},
                         Δh::NTuple{3,T},
                         Δt::T,
                         smoother::Val,
                         ZC::AbstractVector{T},
                         ΔZ::AbstractVector{T},
                         HC::AbstractVector{T},
                         par::String;
                         is_steady::Bool=false) where {T <: AbstractFloat}
    _Preconditioner!(xx, bb, λ, mask, ρ, cp, Δh, Δt, smoother, ZC, ΔZ, HC, par, is_steady)
end

# :none の場合（前処理なし）
@inline function _Preconditioner!(xx, bb, λ, mask, ρ, cp, Δh, Δt, ::Val{:none}, ZC, ΔZ, HC, par, is_steady)
    mycopy!(xx, bb, par)
end

# :gs の場合（Gauss-Seidel前処理）
function _Preconditioner!(xx::AbstractArray{T,3}, bb, λ, mask, ρ, cp, Δh, Δt, ::Val{:gs},
                          ZC, ΔZ, HC, par, is_steady) where {T <: AbstractFloat}
    for _ in 1:PRECONDITIONER_SWEEPS
        rbsor_simple!(xx, λ, bb, mask, ρ, cp, Δh, Δt, one(T), ZC, ΔZ, HC, par, is_steady=is_steady)
    end
end


"""
@brief ベクトルの内積
@param [in]     x    ベクトル
@ret            内積
"""
function Fdot1(x::AbstractArray{T,3}, par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(x)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        @reduce(sum = zero(T) + x[i,j,k] * x[i,j,k])
    end
    return sum
end


"""
@brief 2ベクトルの内積
@param [in]     x    ベクトル
@param [in]     y    ベクトル
@ret            内積
"""
function Fdot2(x::AbstractArray{T,3}, y::AbstractArray{T,3}, par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(x)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        @reduce(sum = zero(T) + x[i,j,k] * y[i,j,k])
    end
    return sum
end


"""
@brief BiCGstabの部分演算1
@param [in,out] p    ベクトル
@param [in]     r    ベクトル
@param [in]     q    ベクトル
@param [in]     beta 係数
@param [in]     omg  係数
"""
function BiCG1!(p::AbstractArray{T,3},
                r::AbstractArray{T,3},
                q::AbstractArray{T,3},
                beta::T,
                omg::T,
                par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(p)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        p[i,j,k] = r[i,j,k] + beta * (p[i,j,k] - omg * q[i,j,k])
    end
end


"""
@brief AXPYZ
@param [out]    z    ベクトル
@param [in]     y    ベクトル
@param [in]     x    ベクトル
@param [in]     a    係数
"""
function Triad!(z::AbstractArray{T,3},
                x::AbstractArray{T,3},
                y::AbstractArray{T,3},
                a::T,
                par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(z)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        z[i,j,k] = a * x[i,j,k] + y[i,j,k]
    end
end


"""
@brief BiCGstab 2
@param [in,out] z    ベクトル
@param [in]     y    ベクトル
@param [in]     x    ベクトル
@param [in]     a    係数
@param [in]     b    係数
"""
function BICG2!(z::AbstractArray{T,3},
                x::AbstractArray{T,3},
                y::AbstractArray{T,3},
                a::T,
                b::T,
                par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(z)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        z[i,j,k] += a * x[i,j,k] + b * y[i,j,k]
    end
end


"""
@brief SOR法の残差
@param [in,out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    右辺ベクトル
@param [in]     m    マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ
@ret                 残差RMS
"""
function resSOR(θ::AbstractArray{T,3},
                λ::AbstractArray{T,3},
                b::AbstractArray{T,3},
                m::AbstractArray{T,3},
                ρ::AbstractArray{T,3},
                cp::AbstractArray{T,3},
                Δh::NTuple{3,T},
                Δt::T,
                ω::T,
                ZC::AbstractVector{T},
                ΔZ::AbstractVector{T},
                HC::AbstractVector{T},
                par::String;
                is_steady::Bool=false) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(θ)
    dx0 = Δh[1]
    dy0 = Δh[2]
    ddt = inv(Δt)
    ddx = inv(dx0)
    ddy = inv(dy0)
    oneT = one(T)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        dz_k = ΔZ[k]
        pp = θ[i,j,k]
        λ0 = λ[i,j,k]
        m0 = m[i,j,k]

        # 熱伝導項（面積×コンダクタンス）
        cond_xm = λf(λ[i-1,j,k], λ0, m[i-1,j,k], m0) * dy0 * dz_k * ddx
        cond_xp = λf(λ[i+1,j,k], λ0, m[i+1,j,k], m0) * dy0 * dz_k * ddx
        cond_ym = λf(λ[i,j-1,k], λ0, m[i,j-1,k], m0) * dx0 * dz_k * ddy
        cond_yp = λf(λ[i,j+1,k], λ0, m[i,j+1,k], m0) * dx0 * dz_k * ddy
        cond_zm = λf(λ[i,j,k-1], λ0, m[i,j,k-1], m0) * dx0 * dy0 / (ZC[k]-ZC[k-1])
        cond_zp = λf(λ[i,j,k+1], λ0, m[i,j,k+1], m0) * dx0 * dy0 / (ZC[k+1]-ZC[k])

        # 対流項（CommonSolver.jl方式）
        conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
        conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
        conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
        conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
        conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
        conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

        # 時間項 - is_steady対応
        a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt

        # 対角項: 熱伝導 + 対流 + 時間
        dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                          conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0

        # 隣接項: 熱伝導のみ
        ss = ( cond_xp * θ[i+1,j  ,k  ] + cond_xm * θ[i-1,j  ,k  ]
             + cond_yp * θ[i  ,j+1,k  ] + cond_ym * θ[i  ,j-1,k  ]
             + cond_zp * θ[i  ,j  ,k+1] + cond_zm * θ[i  ,j  ,k-1] )

        dp = (((ss-b[i,j,k])/dd - pp)) * m0
        r = (dd + ω*(cond_xm+cond_ym+cond_zm))*dp / ω
        @reduce(res = zero(T) + r*r)
    end

    return sqrt(res)
end

"""
@brief SOR法
@param [in,out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    右辺ベクトル
@param [in]     m    マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     is_steady 定常解析フラグ
@ret                残差RMS
"""
function sor!(θ::AbstractArray{T,3},
              λ::AbstractArray{T,3},
              b::AbstractArray{T,3},
              m::AbstractArray{T,3},
              ρ::AbstractArray{T,3},
              cp::AbstractArray{T,3},
              Δh::NTuple{3,T},
              Δt::T,
              ω::T,
              ZC::AbstractVector{T},
              ΔZ::AbstractVector{T},
              HC::AbstractVector{T};
              is_steady::Bool=false) where {T <: AbstractFloat}
    SZ = size(θ)
    dx0 = Δh[1]
    dy0 = Δh[2]
    ddt = inv(Δt)
    ddx = inv(dx0)
    ddy = inv(dy0)
    oneT = one(T)

    res::T = zero(T)
    for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        dz_k = ΔZ[k]
        pp = θ[i,j,k]
        λ0 = λ[i,j,k]
        m0 = m[i,j,k]

        # 熱伝導項（面積×コンダクタンス）
        cond_xm = λf(λ[i-1,j,k], λ0, m[i-1,j,k], m0) * dy0 * dz_k * ddx
        cond_xp = λf(λ[i+1,j,k], λ0, m[i+1,j,k], m0) * dy0 * dz_k * ddx
        cond_ym = λf(λ[i,j-1,k], λ0, m[i,j-1,k], m0) * dx0 * dz_k * ddy
        cond_yp = λf(λ[i,j+1,k], λ0, m[i,j+1,k], m0) * dx0 * dz_k * ddy
        cond_zm = λf(λ[i,j,k-1], λ0, m[i,j,k-1], m0) * dx0 * dy0 / (ZC[k]-ZC[k-1])
        cond_zp = λf(λ[i,j,k+1], λ0, m[i,j,k+1], m0) * dx0 * dy0 / (ZC[k+1]-ZC[k])

        # 対流項（CommonSolver.jl方式）
        conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
        conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
        conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
        conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
        conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
        conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

        # 時間項 - is_steady対応
        a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt

        # 対角項: 熱伝導 + 対流 + 時間
        dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                          conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0

        # 隣接項: 熱伝導のみ
        ss = ( cond_xp * θ[i+1,j  ,k  ] + cond_xm * θ[i-1,j  ,k  ]
             + cond_yp * θ[i  ,j+1,k  ] + cond_ym * θ[i  ,j-1,k  ]
             + cond_zp * θ[i  ,j  ,k+1] + cond_zm * θ[i  ,j  ,k-1] )

        dp = (((ss-b[i,j,k])/dd - pp)) * m0
        pn = pp + ω * dp
        θ[i,j,k] = pn
        r = (dd + ω*(cond_xm+cond_ym+cond_zm))*dp / ω
        res += r*r
    end

    return sqrt(res)
end


"""
@brief RB-SOR法のカーネル
@param [in,out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    右辺ベクトル
@param [in]     m    マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     color R or B
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ
@ret                 残差2乗和
"""
function rbsor_core!(θ::AbstractArray{T,3},
                     λ::AbstractArray{T,3},
                     b::AbstractArray{T,3},
                     m::AbstractArray{T,3},
                     ρ::AbstractArray{T,3},
                     cp::AbstractArray{T,3},
                     Δh::NTuple{3,T},
                     Δt::T,
                     ω::T,
                     ZC::AbstractVector{T},
                     ΔZ::AbstractVector{T},
                     HC::AbstractVector{T},
                     color::Int,
                     par::String;
                     is_steady::Bool=false) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(θ)
    dx0 = Δh[1]
    dy0 = Δh[2]
    ddt = inv(Δt)
    ddx = inv(dx0)
    ddy = inv(dy0)
    oneT = one(T)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1
        @simd for i in 2+mod(k+j+color,2):2:SZ[1]-1
            dz_k = ΔZ[k]
            pp = θ[i,j,k]
            λ0 = λ[i,j,k]
            m0 = m[i,j,k]

            # 熱伝導項（面積×コンダクタンス）
            cond_xm = λf(λ[i-1,j,k], λ0, m[i-1,j,k], m0) * dy0 * dz_k * ddx
            cond_xp = λf(λ[i+1,j,k], λ0, m[i+1,j,k], m0) * dy0 * dz_k * ddx
            cond_ym = λf(λ[i,j-1,k], λ0, m[i,j-1,k], m0) * dx0 * dz_k * ddy
            cond_yp = λf(λ[i,j+1,k], λ0, m[i,j+1,k], m0) * dx0 * dz_k * ddy
            cond_zm = λf(λ[i,j,k-1], λ0, m[i,j,k-1], m0) * dx0 * dy0 / (ZC[k]-ZC[k-1])
            cond_zp = λf(λ[i,j,k+1], λ0, m[i,j,k+1], m0) * dx0 * dy0 / (ZC[k+1]-ZC[k])

            # 対流項（CommonSolver.jl方式）
            conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
            conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
            conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
            conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
            conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
            conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

            # 時間項 - is_steady対応
            a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt

            # 対角項: 熱伝導 + 対流 + 時間
            dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                              conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0

            # 隣接項: 熱伝導のみ
            ss = ( cond_xp * θ[i+1,j  ,k  ] + cond_xm * θ[i-1,j  ,k  ]
             + cond_yp * θ[i  ,j+1,k  ] + cond_ym * θ[i  ,j-1,k  ]
             + cond_zp * θ[i  ,j  ,k+1] + cond_zm * θ[i  ,j  ,k-1] )

            dp = (((ss-b[i,j,k])/dd - pp)) * m0
            θ[i,j,k] = pp + ω * dp
            r = (dd + ω*(cond_xm+cond_ym+cond_zm))*dp / ω
            @reduce(res = zero(T) + r*r)
        end
    end

    return res
end


"""
@brief RB-SOR法
@param [in,out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    右辺ベクトル
@param [in]     mask マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ
@ret                 残差RMS
"""
function rbsor!(θ::AbstractArray{T,3},
                λ::AbstractArray{T,3},
                b::AbstractArray{T,3},
                mask::AbstractArray{T,3},
                ρ::AbstractArray{T,3},
                cp::AbstractArray{T,3},
                Δh::NTuple{3,T},
                Δt::T,
                ω::T,
                ZC::AbstractVector{T},
                ΔZ::AbstractVector{T},
                HC::AbstractVector{T},
                par::String;
                is_steady::Bool=false) where {T <: AbstractFloat}
    res::T = zero(T)

    # 2色のマルチカラー(Red&Black)のセットアップ
    for c in 0:1
        res += rbsor_core!(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, c, par, is_steady=is_steady)
    end
    return sqrt(res)
end


"""
@brief RB-SOR法（簡易版、前処理用）
@param [in,out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    右辺ベクトル
@param [in]     mask マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     par  バックエンド
@param [in]     is_steady 定常解析フラグ
@ret                 残差RMS
"""
function rbsor_simple!(θ::AbstractArray{T,3},
                       λ::AbstractArray{T,3},
                       b::AbstractArray{T,3},
                       mask::AbstractArray{T,3},
                       ρ::AbstractArray{T,3},
                       cp::AbstractArray{T,3},
                       Δh::NTuple{3,T},
                       Δt::T,
                       ω::T,
                       ZC::AbstractVector{T},
                       ΔZ::AbstractVector{T},
                       HC::AbstractVector{T},
                       par::String;
                       is_steady::Bool=false) where {T <: AbstractFloat}
    return rbsor!(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, par, is_steady=is_steady)
end


"""
@brief SOR法による求解
@param [in/out] θ    解ベクトル
@param [in]     λ    熱伝導率
@param [in]     b    RHSベクトル
@param [in]     mask マスク配列
@param [in]     ρ    密度
@param [in]     cp   比熱
@param [in]     Δh   セル幅
@param [in]     Δt   時間積分幅
@param [in]     ω    加速係数
@param [in]     ZC   CVセンター座標
@param [in]     ΔZ   CV幅
@param [in]     HC   熱伝達係数 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
@param [in]     F    ファイルディスクリプタ
@param [in]     tol  収束判定基準
@param [in]     par  バックエンド
@param [in]     maxItr 最大反復数
@param [in]     is_steady 定常解析フラグ

"""
function solveSOR!(θ::AbstractArray{T,3},
                    λ::AbstractArray{T,3},
                    b::AbstractArray{T,3},
                    mask::AbstractArray{T,3},
                    ρ::AbstractArray{T,3},
                    cp::AbstractArray{T,3},
                    Δh::NTuple{3,T},
                    Δt::T,
                    ω::T,
                    ZC::AbstractVector{T},
                    ΔZ::AbstractVector{T},
                    HC::AbstractVector{T},
                    F,
                    tol::T,
                    par::String;
                    maxItr::Int=20000,
                    is_steady::Bool=false) where {T <: AbstractFloat}

    res0 = resSOR(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, par, is_steady=is_steady)
    if res0 == zero(T)
        res0 = one(T)
    end
    println("Initial residual = ", res0)

    n = 0
    for n in 1:maxItr
        res = rbsor!(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, par, is_steady=is_steady) / res0
        @printf(F, "%10d %24.14E\n", n, res)
        @printf(stdout, "%10d %24.14E\n", n, res)
        if res < tol
            println("Converged at ", n)
            return
        end
    end
end

end # end of module

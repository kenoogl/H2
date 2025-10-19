# NonUniform.jl → CommonSolver.jl準拠 移植計画書

**作成日:** 2025-10-19
**プロジェクト:** Heat3ds NonUniform格子熱伝導ソルバーの近代化
**目的:** NonUniform.jlをCommonSolver.jlのインターフェースに準拠させる

---

## 1. 基本方針

### 1.1 原則
- **CommonSolver.jlは参照のみ、変更しない**
- **NonUniform.jlをCommonSolver.jlのインターフェースに合わせる**
- **対流境界条件はCommonSolver.jl方式（HC配列）を採用**
- **Jacobi前処理は不要（Gauss-Seidelのみ）**
- **テストはheat3ds.jlの実行のみ**

### 1.2 参照実装
- **元実装:** `/Users/Daily/Development/H2/NonUniform.jl`
- **目標実装:** `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl`
- **RHS参考実装:** `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/DHCPSolver.jl`

---

## 2. 主要な変更内容

### 2.1 密度ρの扱い

**現状（NonUniform.jl）:**
```julia
ρ::Array{Float64,3}  # 3次元配列（空間変化する密度）
a_p_0 = ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k / Δt
```

**方針:** NonUniform.jl内で`ρ[i,j,k]`を保持。CommonSolver.jlはスカラーρを想定しているが、NonUniform.jlは独自に配列ρ対応を継続。

### 2.2 is_steadyパラメータ

**現状:** `is_steady::Bool=false`パラメータを持つ

**方針:** `is_steady`パラメータを**保持**し、全ての関数で一貫して使用。

**is_steady対応が必要な関数:**
- `calRHS!`
- `CalcRK!`
- `CalcAX!`
- `rbsor_core!` ← **追加**
- `resSOR` ← **追加**
- `sor!` ← **追加**
- `rbsor!`, `rbsor_simple!`, `solveSOR!` ← **is_steadyを伝播**
- `Preconditioner!` ← **is_steadyを伝播**
- `PBiCGSTAB!`
- `CG!`

### 2.3 対流境界条件の扱い

**重要な理解:**

対流境界条件の処理は**2箇所で異なる役割**を持つ：

1. **RHSCore.jl（既存）:** 境界面での熱伝達によるRHS項への寄与
   - `RHS_convection!()`で境界セルのRHS項に `h * area * (T_amb - T_boundary)` を追加
   - **この処理は保持する**

2. **NonUniform.jl（追加）:** HC配列による内部セルでの対流項の対角項への寄与
   - CalcAX!/CalcRK!等で対角係数に `HC[i] * area * (1 - mask)` を追加
   - **CommonSolver.jl方式を採用**

**方針:**
- **RHSCore.jl:** 対流境界条件処理を**保持**（変更なし）
- **NonUniform.jl:** HC配列による対流項を**追加**

**HC配列を追加する関数:**
- `CalcRK!`
- `CalcAX!`
- `rbsor_core!`
- `resSOR`
- `sor!`
- これらを呼び出す全ての関数（`rbsor!`, `rbsor_simple!`, `solveSOR!`, `Preconditioner!`, `PBiCGSTAB!`, `CG!`）

### 2.4 型パラメータ化

**変更前:**
```julia
function CalcAX!(ax::Array{Float64,3}, θ::Array{Float64,3}, ...)
```

**変更後:**
```julia
function CalcAX!(ax::AbstractArray{T,3}, θ::AbstractArray{T,3}, ...) where {T <: AbstractFloat}
```

全関数を型パラメータ化。

### 2.5 関数シグネチャの統一

**変更前:**
```julia
function PBiCGSTAB!(wk, Δh, Δt, ZC, ΔZ, smoother::String, F, tol, par; is_steady=false)
  # 返り値なし
end
```

**変更後:**
```julia
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

  return isconverged::Bool, itr::Int, res0::T
end
```

**主な変更:**
- `smoother`: `String` → `Symbol` (`:none`, `:gs`)
- `F`: ファイルディスクリプタを削除（verboseフラグで制御）
- `tol`, `maxItr`: 位置引数 → キーワード引数
- `HC::AbstractVector{T}`: 追加（6要素配列）
- `is_steady::Bool`: キーワード引数として保持
- 返り値: `(isconverged, itr, res0)`

---

## 3. 移植ステップ詳細

### ステップ1: Zcoord.jl の格子生成修正 ✅ (完了)

**実施済み内容:**
- 底面・表面セルを中点で計算（CommonSolver準拠）
- Z配列の外挿を修正

### ステップ2: Common.jl にユーティリティ関数を追加

**追加関数:**

```julia
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
```

**追加場所:** `common.jl` の `end # end of module Constant` の直前

**export文に追加:**
```julia
# common.jl の先頭付近のexport文に追加
export myfill!, mycopy!
```

### ステップ3: boundary_conditions.jl に set_BC_coef を追加

**追加関数:**
```julia
"""
境界条件セットからHC配列を生成
@param bc_set 境界条件セット
@ret HC配列 [h_xm, h_xp, h_ym, h_yp, h_zm, h_zp]
"""
function set_BC_coef(bc_set::BoundaryConditionSet)
    HC = zeros(Float64, 6)
    HC[1] = bc_set.x_minus.type == CONVECTION ? bc_set.x_minus.heat_transfer_coefficient : 0.0
    HC[2] = bc_set.x_plus.type == CONVECTION ? bc_set.x_plus.heat_transfer_coefficient : 0.0
    HC[3] = bc_set.y_minus.type == CONVECTION ? bc_set.y_minus.heat_transfer_coefficient : 0.0
    HC[4] = bc_set.y_plus.type == CONVECTION ? bc_set.y_plus.heat_transfer_coefficient : 0.0
    HC[5] = bc_set.z_minus.type == CONVECTION ? bc_set.z_minus.heat_transfer_coefficient : 0.0
    HC[6] = bc_set.z_plus.type == CONVECTION ? bc_set.z_plus.heat_transfer_coefficient : 0.0
    return HC
end
```

**追加場所:** `boundary_conditions.jl` の `end # module BoundaryConditions` の直前

**export文に追加:**
```julia
export set_BC_coef
```

### ステップ4: NonUniform.jl に smoother選択器を追加

**追加内容:**

```julia
# Smoother選択器（CommonSolver.jl準拠）
@inline function smoother_selector(s::Symbol)
  if s === :none || s === :gs
    return Val(s)
  end
  throw(ArgumentError("Unsupported smoother: $s. Use :none or :gs"))
end

# 前処理反復回数
const PRECONDITIONER_SWEEPS = 5
```

**追加場所:** `module NonUniform` の `export PBiCGSTAB!, CG!, calRHS!` の直後

### ステップ5: NonUniform.jl の型パラメータ化とHC配列対応

以下の順序で関数を修正：

#### 5.1 補助関数の型パラメータ化（HC不要、is_steady不要）

**対象関数:**
- `Fdot1`
- `Fdot2`
- `BiCG1!`
- `Triad!`
- `BICG2!`

**変更例（Fdot1）:**
```julia
# 変更前
function Fdot1(x::Array{Float64,3}, par::String)

# 変更後
function Fdot1(x::AbstractArray{T,3}, par::String) where {T <: AbstractFloat}
    backend = get_backend(par)
    SZ = size(x)

    @floop backend for k in 2:SZ[3]-1, j in 2:SZ[2]-1, i in 2:SZ[1]-1
        @reduce(sum = zero(T) + x[i,j,k] * x[i,j,k])
    end
    return sum
end
```

#### 5.2 CalcRK! の修正

**関数シグネチャ:**
```julia
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
    HC::AbstractVector{T},  # 追加
    par::String;
    is_steady::Bool=false) where {T <: AbstractFloat}
```

**ループ内の実装:**
```julia
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
```

#### 5.3 CalcAX! の修正

**CalcRK!と同様に修正。最終行のみ異なる:**
```julia
ax[i,j,k] = (ss - dd*θ[i,j,k]) * m0
```

#### 5.4 resSOR の修正

**関数シグネチャ:**
```julia
function resSOR(θ::AbstractArray{T,3},
                λ::AbstractArray{T,3},
                b::AbstractArray{T,3},
                m::AbstractArray{T,3},
                ρ::AbstractArray{T,3},  # 追加
                cp::AbstractArray{T,3}, # 追加
                Δh::NTuple{3,T},
                Δt::T,
                ω::T,
                ZC::AbstractVector{T},
                ΔZ::AbstractVector{T},
                HC::AbstractVector{T},  # 追加
                par::String;
                is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
```

**時間項の計算（CalcRK!と同様）:**
```julia
# 時間項 - is_steady対応
a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt
```

**その他の実装はCalcRK!と同じパターン。**

#### 5.5 rbsor_core! の修正

**関数シグネチャ:**
```julia
function rbsor_core!(θ::AbstractArray{T,3},
                     λ::AbstractArray{T,3},
                     b::AbstractArray{T,3},
                     m::AbstractArray{T,3},
                     ρ::AbstractArray{T,3},  # 追加
                     cp::AbstractArray{T,3}, # 追加
                     Δh::NTuple{3,T},
                     Δt::T,
                     ω::T,
                     ZC::AbstractVector{T},
                     ΔZ::AbstractVector{T},
                     HC::AbstractVector{T},  # 追加
                     color::Int,
                     par::String;
                     is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
```

**時間項の計算（CalcRK!と同様）:**
```julia
# 時間項 - is_steady対応
a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt
```

**その他の実装はCalcRK!と同じパターン。**

#### 5.6 sor! の修正

**関数シグネチャ:**
```julia
function sor!(θ::AbstractArray{T,3},
              λ::AbstractArray{T,3},
              b::AbstractArray{T,3},
              m::AbstractArray{T,3},
              ρ::AbstractArray{T,3},  # 追加
              cp::AbstractArray{T,3}, # 追加
              Δh::NTuple{3,T},
              Δt::T,
              ω::T,
              ZC::AbstractVector{T},
              ΔZ::AbstractVector{T},
              HC::AbstractVector{T};  # 追加
              is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
```

**時間項の計算（CalcRK!と同様）:**
```julia
# 時間項 - is_steady対応
a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt
```

#### 5.7 rbsor!, rbsor_simple!, solveSOR! の修正

これらは`rbsor_core!`や`resSOR`を呼び出すため、`ρ`, `cp`, `HC`, `is_steady`引数を追加して伝播。

**rbsor!:**
```julia
function rbsor!(θ::AbstractArray{T,3},
                λ::AbstractArray{T,3},
                b::AbstractArray{T,3},
                mask::AbstractArray{T,3},
                ρ::AbstractArray{T,3},  # 追加
                cp::AbstractArray{T,3}, # 追加
                Δh::NTuple{3,T},
                Δt::T,
                ω::T,
                ZC::AbstractVector{T},
                ΔZ::AbstractVector{T},
                HC::AbstractVector{T},  # 追加
                par::String;
                is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
    res = zero(T)
    for c in 0:1
        res += rbsor_core!(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, c, par, is_steady=is_steady)
    end
    return sqrt(res)
end
```

**rbsor_simple!:**
```julia
function rbsor_simple!(θ::AbstractArray{T,3},
                       λ::AbstractArray{T,3},
                       b::AbstractArray{T,3},
                       mask::AbstractArray{T,3},
                       ρ::AbstractArray{T,3},  # 追加
                       cp::AbstractArray{T,3}, # 追加
                       Δh::NTuple{3,T},
                       Δt::T,
                       ω::T,
                       ZC::AbstractVector{T},
                       ΔZ::AbstractVector{T},
                       HC::AbstractVector{T},  # 追加
                       par::String;
                       is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
    return rbsor!(θ, λ, b, mask, ρ, cp, Δh, Δt, ω, ZC, ΔZ, HC, par, is_steady=is_steady)
end
```

**solveSOR!:**
```julia
function solveSOR!(θ::AbstractArray{T,3},
                    λ::AbstractArray{T,3},
                    b::AbstractArray{T,3},
                    mask::AbstractArray{T,3},
                    ρ::AbstractArray{T,3},  # 追加
                    cp::AbstractArray{T,3}, # 追加
                    Δh::NTuple{3,T},
                    Δt::T,
                    ω::T,
                    ZC::AbstractVector{T},
                    ΔZ::AbstractVector{T},
                    HC::AbstractVector{T},  # 追加
                    F,
                    tol::T,
                    par::String;
                    maxItr::Int=20000,
                    is_steady::Bool=false) where {T <: AbstractFloat}  # 追加

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
```

#### 5.8 Preconditioner! の修正

**変更後:**
```julia
function Preconditioner!(xx::AbstractArray{T,3},
                         bb::AbstractArray{T,3},
                         λ::AbstractArray{T,3},
                         mask::AbstractArray{T,3},
                         ρ::AbstractArray{T,3},  # 追加
                         cp::AbstractArray{T,3}, # 追加
                         Δh::NTuple{3,T},
                         Δt::T,
                         smoother::Val,
                         ZC::AbstractVector{T},
                         ΔZ::AbstractVector{T},
                         HC::AbstractVector{T},  # 追加
                         par::String;
                         is_steady::Bool=false) where {T <: AbstractFloat}  # 追加
    _Preconditioner!(xx, bb, λ, mask, ρ, cp, Δh, Δt, smoother, ZC, ΔZ, HC, par, is_steady)
end

# :none の場合
@inline function _Preconditioner!(xx, bb, λ, mask, ρ, cp, Δh, Δt, ::Val{:none}, ZC, ΔZ, HC, par, is_steady)
    mycopy!(xx, bb, par)
end

# :gs の場合
function _Preconditioner!(xx::AbstractArray{T,3}, bb, λ, mask, ρ, cp, Δh, Δt, ::Val{:gs},
                          ZC, ΔZ, HC, par, is_steady) where {T <: AbstractFloat}
    for _ in 1:PRECONDITIONER_SWEEPS
        rbsor_simple!(xx, λ, bb, mask, ρ, cp, Δh, Δt, one(T), ZC, ΔZ, HC, par, is_steady=is_steady)
    end
end
```

#### 5.9 PBiCGSTAB! の修正

**完全な新実装:**

```julia
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
```

#### 5.10 CG! の修正

**PBiCGSTAB!と同様の方針で修正。**

主な違い:
- `pcg_r0`不要
- `rho = Fdot2(wk.pcg_r, wk.pcg_s, par)`（zとrの内積）

#### 5.11 calRHS! の修正

**変更後:**
```julia
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
    # RHSCore.apply_bc2RHS!はそのまま使用
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
```

### ステップ6: heat3ds.jl の修正

**main関数の修正:**

```julia
function main(Δh, Δt, wk, ZC, ΔZ, ID, solver, smoother, bc_set, par; is_steady::Bool=false)
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
    parse_residuals_from_log!(conv_data, "log.txt")
    return conv_data
end
```

---

## 4. HC配列・is_steady対応が必要な関数一覧

### 4.1 HC配列 + is_steady対応が必要な関数

以下の関数すべてに`HC::AbstractVector{T}`引数と`is_steady::Bool`引数を追加:

1. **CalcRK!** - 残差計算
2. **CalcAX!** - 行列ベクトル積
3. **resSOR** - SOR残差計算（ρ, cp引数も追加）
4. **rbsor_core!** - RB-SORカーネル（ρ, cp引数も追加）
5. **sor!** - SOR法（ρ, cp引数も追加）

### 4.2 HC配列 + is_steady伝播が必要な関数

6. **rbsor!** - RB-SOR (ρ, cp, HC, is_steadyを追加してrbsor_core!に渡す)
7. **rbsor_simple!** - 前処理用RB-SOR (ρ, cp, HC, is_steadyを追加してrbsor!に渡す)
8. **solveSOR!** - SORソルバー (ρ, cp, HC, is_steadyを追加)
9. **Preconditioner!** - 前処理 (ρ, cp, HC, is_steadyを追加)
10. **PBiCGSTAB!** - ソルバー本体
11. **CG!** - ソルバー本体

### 4.3 is_steady対応のみ必要な関数

12. **calRHS!** - RHS計算

---

## 5. 変更サマリー表

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| **ファイル名** | RHS.jl | RHSCore.jl（既存） |
| **RHSCore対流処理** | 保持 | 保持（変更なし） |
| **型** | Float64固定 | ジェネリック型T |
| **smoother** | String ("gs") | Symbol (:gs, :none) |
| **HC配列** | なし | HC::AbstractVector{T} (6要素) |
| **対流項** | RHSCoreのみ | RHSCore + CalcAX!/CalcRK!/rbsor_core!/resSOR/sor! |
| **is_steady** | 一部のみ | 全ての関数で一貫適用 |
| **ρ, cp** | CalcRK!/CalcAX!のみ | resSOR/rbsor_core!/sor!にも追加 |
| **返り値** | なし | (isconverged, itr, res0) |
| **Z格子** | 修正前 | 底面・表面セル=中点（修正済） |

---

## 6. 実装済み項目

- ✅ **Zcoord.jl の格子生成修正**（底面・表面セルを中点計算、Z配列外挿修正）

---

## 7. テスト計画

### 7.1 コンパイルテスト
```bash
julia -e 'include("heat3ds.jl")'
```

### 7.2 実行テスト
```bash
julia heat3ds.jl
```

### 7.3 確認項目
- [ ] コンパイルエラーがないこと
- [ ] ソルバーが収束すること
- [ ] 温度分布が妥当であること（最小・最大値が物理的に妥当）
- [ ] 収束履歴が出力されること
- [ ] PNGファイルが正しく生成されること
- [ ] 定常解析（is_steady=true）が正しく動作すること

---

## 8. 次のセッションでの作業開始手順

### 8.1 このドキュメントを読む
```bash
cat /Users/Daily/Development/H2/MIGRATION_PLAN.md
```

### 8.2 現在の状態を確認
```bash
cd /Users/Daily/Development/H2
git status  # (gitを使っている場合)
ls -l *.jl
```

### 8.3 作業再開の指示
「MIGRATION_PLAN.mdのステップ2から実装を開始してください」と Claude に指示する。

---

## 9. トラブルシューティング

### 9.1 型エラーが発生した場合
- `Float64`が残っていないか確認
- `Array` → `AbstractArray`、`Vector` → `AbstractVector`、`Tuple` → `NTuple`の変更漏れがないか

### 9.2 メソッドエラーが発生した場合
- 関数呼び出し時のHC配列が抜けていないか
- ρ, cp引数が抜けていないか（resSOR, rbsor_core!, sor!等）
- smoother引数がSymbolになっているか（String → Symbol変換漏れ）
- is_steady引数が伝播されているか

### 9.3 収束しない場合
- HC配列の値が正しいか確認（set_BC_coefの出力をチェック）
- is_steadyフラグが正しく渡されているか
- 時間項a_p_0の計算がis_steadyに応じて正しく切り替わっているか

### 9.4 対流境界条件が効いていない場合
- RHSCore.apply_bc2RHS!が呼ばれているか確認
- HC配列がゼロでないか確認
- CalcAX!/CalcRK!等で対流項が計算されているか確認

---

## 10. 参考資料

### 10.1 CommonSolver実装例
- `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl`
- 特に`PBiCGSTAB!`, `CalcRK!`, `CalcAX!`, `rbsor_core!`, `resSOR`の実装を参照

### 10.2 DHCPSolver実装例
- `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/DHCPSolver.jl`
- `calRHS!`の実装を参照

### 10.3 境界条件システム
- `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/core/BoundaryConditions.jl`
- `set_BC_coef`の実装を参照（ただし今回は独自実装）

---

**最終更新:** 2025-10-19
**ステータス:** ステップ1完了（Zcoord.jl修正済）、ステップ2以降は未実装
**次のステップ:** ステップ2（Common.jlにユーティリティ関数追加）から開始

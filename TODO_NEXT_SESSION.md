# 次のセッション作業ガイド

**作成日**: 2025-10-19
**プロジェクト**: Heat3ds NonUniform格子熱伝導ソルバーの近代化
**進捗**: ステップ5.8まで完了（全体の約70%完了）

---

## 現在の状況

### 完了したステップ ✅

**ステップ2**: Common.jlにユーティリティ関数追加
- `myfill!`: 並列化対応のfill関数
- `mycopy!`: 並列化対応のcopy関数
- export文に追加済み

**ステップ3**: boundary_conditions.jlにset_BC_coef関数追加
- `set_BC_coef`: 境界条件セットからHC配列（6要素）を生成
- export文に追加済み

**ステップ4**: NonUniform.jlにsmoother選択器追加
- `smoother_selector`: Symbol→Val型変換関数
- `PRECONDITIONER_SWEEPS = 5`: 定数追加
- using文に`myfill!`, `mycopy!`追加

**ステップ5.1-5.8**: NonUniform.jlの型パラメータ化とHC配列対応（8/11完了）
- ✅ 5.1: 補助関数の型パラメータ化（Fdot1, Fdot2, BiCG1!, Triad!, BICG2!）
- ✅ 5.2: CalcRK!の修正（型パラメータ化、HC配列、is_steady対応）
- ✅ 5.3: CalcAX!の修正（型パラメータ化、HC配列、is_steady対応）
- ✅ 5.4-5.6: resSOR, rbsor_core!, sor!の修正
- ✅ 5.7: rbsor!, rbsor_simple!, solveSOR!の修正（引数伝播）
- ✅ 5.8: Preconditioner!の修正（Val型ディスパッチ）

### 残りのステップ ⏸

**ステップ5.9**: PBiCGSTAB!の修正（完全な新実装）
- 参照実装: `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl`
- 現在の実装: `/Users/Daily/Development/H2/NonUniform.jl` 195行目付近
- 主な変更:
  - 型パラメータ化（Float64 → AbstractFloat）
  - 関数シグネチャの変更（String smoother → Symbol smoother）
  - HC配列の追加
  - is_steady対応
  - 返り値の追加: `(isconverged::Bool, itr::Int, res0::T)`
  - verbose, maxItrをキーワード引数化

**ステップ5.10**: CG!の修正（完全な新実装）
- 参照実装: `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl`
- PBiCGSTAB!と同様のパターンで修正
- 主な違い: `pcg_r0`不要、`rho = Fdot2(wk.pcg_r, wk.pcg_s, par)`

**ステップ5.11**: calRHS!の修正
- 型パラメータ化（Tuple → NTuple、Vector → AbstractVector）
- is_steady対応（既に対応済みの可能性あり、確認が必要）

**ステップ6**: heat3ds.jlの修正
- main関数の修正
- HC配列の生成: `HC = BoundaryConditions.set_BC_coef(bc_set)`
- ソルバー呼び出しの修正:
  - smoother: String → Symbol (`:gs`, `:none`)
  - HC配列を引数に追加
  - is_steadyを伝播
  - 返り値の受け取り: `isconverged, itr, res0 = NonUniform.PBiCGSTAB!(...)`

---

## 次のセッションでの開始手順

### 1. コンテキストの確認

```bash
cd /Users/Daily/Development/H2
git status
git log --oneline -5
```

### 2. MIGRATION_PLAN.mdの確認

```bash
cat /Users/Daily/Development/H2/MIGRATION_PLAN.md
```

特にステップ5.9の実装例（540-645行目付近）を参照。

### 3. 参照実装の確認

CommonSolver.jlのPBiCGSTAB!を参照:

```bash
# PBiCGSTAB!の実装を確認
cat /Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl | grep -A 100 "^function PBiCGSTAB!"
```

### 4. 作業開始

Claudeに以下のように指示:

```
MIGRATION_PLAN.mdに従って、ステップ5.9（PBiCGSTAB!の修正）から作業を開始してください。
参照実装は /Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl です。
```

---

## 重要な実装パターン

### 型パラメータ化の基本パターン

```julia
# 変更前
function func!(x::Array{Float64,3}, y::Float64)

# 変更後
function func!(x::AbstractArray{T,3}, y::T) where {T <: AbstractFloat}
```

### HC配列の使用パターン（CalcRK!/CalcAX!などで既に実装済み）

```julia
# 対流項（CommonSolver.jl方式）
conv_xm = HC[1] * dy0 * dz_k * (oneT - m[i-1,j,k])
conv_xp = HC[2] * dy0 * dz_k * (oneT - m[i+1,j,k])
conv_ym = HC[3] * dx0 * dz_k * (oneT - m[i,j-1,k])
conv_yp = HC[4] * dx0 * dz_k * (oneT - m[i,j+1,k])
conv_zm = HC[5] * dx0 * dy0 * (oneT - m[i,j,k-1])
conv_zp = HC[6] * dx0 * dy0 * (oneT - m[i,j,k+1])

# 対角項に追加
dd = (oneT-m0) + (cond_xp + cond_xm + cond_yp + cond_ym + cond_zp + cond_zm +
                  conv_xp + conv_xm + conv_yp + conv_ym + conv_zp + conv_zm + a_p_0)*m0
```

### is_steady対応パターン（CalcRK!/CalcAX!などで既に実装済み）

```julia
# 時間項 - is_steady対応
a_p_0 = is_steady ? zero(T) : ρ[i,j,k] * cp[i,j,k] * dx0 * dy0 * dz_k * ddt
```

### PBiCGSTAB!の新しい関数シグネチャ（MIGRATION_PLAN.md 542-554行目）

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
    # 実装...
    return isconverged::Bool, itr::Int, res0::T
end
```

---

## テスト計画

### コンパイルテスト

```bash
julia -e 'include("heat3ds.jl")'
```

### 実行テスト

```bash
julia heat3ds.jl
```

### 確認項目

- [ ] コンパイルエラーがないこと
- [ ] ソルバーが収束すること
- [ ] 温度分布が妥当であること（最小・最大値が物理的に妥当）
- [ ] 収束履歴が出力されること
- [ ] PNGファイルが正しく生成されること
- [ ] 定常解析（is_steady=true）が正しく動作すること

---

## トラブルシューティング

### 型エラーが発生した場合

- `Float64`が残っていないか確認
- `Array` → `AbstractArray`、`Vector` → `AbstractVector`、`Tuple` → `NTuple`の変更漏れがないか

### メソッドエラーが発生した場合

- 関数呼び出し時のHC配列が抜けていないか
- ρ, cp引数が抜けていないか
- smoother引数がSymbolになっているか（String → Symbol変換漏れ）
- is_steady引数が伝播されているか

### 収束しない場合

- HC配列の値が正しいか確認（set_BC_coefの出力をチェック）
- is_steadyフラグが正しく渡されているか
- 時間項a_p_0の計算がis_steadyに応じて正しく切り替わっているか

---

## 参考ファイル

- **移植計画書**: `/Users/Daily/Development/H2/MIGRATION_PLAN.md`
- **参照実装**: `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/CommonSolver.jl`
- **RHS参考実装**: `/Users/Daily/Development/IHCP/TrialClaudeMCPCodex/julia/src/solvers/DHCPSolver.jl`
- **プロジェクト説明**: `/Users/Daily/Development/H2/CLAUDE.md`

---

## 推定残り時間

- ステップ5.9（PBiCGSTAB!）: 30-45分
- ステップ5.10（CG!）: 20-30分
- ステップ5.11（calRHS!）: 10-15分
- ステップ6（heat3ds.jl）: 15-20分
- テスト・デバッグ: 30-60分

**合計推定時間**: 2-3時間

---

**最終更新**: 2025-10-19
**次回作業開始**: ステップ5.9（PBiCGSTAB!の修正）から

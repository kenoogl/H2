# Build functions for geometric model
# Ver. 1.0  2025-07-23
module modelA
export fillID!, setLambda!, model_test

using Printf
using LinearAlgebra
using Plots

const pg_dpth = 0.005e-3
const s_dpth = 0.1e-3
const d_ufill = 0.05e-3
const r_bump = 0.03e-3
const r_tsv = 0.02e-3
const h_tsv = 0.1e-3

# Unit [m]
const zm12 = 0.6e-3
const zm11 = 0.55e-3
const zm10 = 0.5e-3
const zm9  = zm10 - pg_dpth
const zm8 = 0.4e-3
const zm7 = 0.35e-3
const zm6 = zm7 - pg_dpth
const zm5 = 0.25e-3
const zm4 = 0.2e-3
const zm3 = zm4 - pg_dpth
const zm2 = 0.1e-3
const zm1 = 0.05e-3
const zm0 = 0.0



# Unit [mm]
#  +----- top            0.6  = zm12
#  Heat sink (depth=0.05)
#  +-----                0.55 = zm11
#  underfill (depth=0.05)
#  +-----                0.5 = zm10
#      +--               0.5 - pg_dpth = zm9
#  Silcon 3 (depth=0.1)           
#  +-----                0.4 = zm8
#  underfill (depth=0.05)
#  +-----                0.35 = zm7
#      +--               0.35 - pg_dpth = zm6
#  Silcon 2 (depth=0.1)             
#  +-----                0.25 = zm5
#  underfill (depth=0.05)
#  +-----                0.2 = zm4
#      +--               0.2 - pg_dpth = zm3
#  Silcon 1 (depth=0.1)             
#  +-----                0.1 = zm2  
#  underfill (depth=0.05)
#  +------               0.05 = zm1 
#  Subtrate (depth=0.05)
#  +------ bottom        0.0 = zm0 


# boundary box of each element [mm]
substrate = Dict(
    "x0" => 0.0, "y0" => 0.0, "z0" => zm0,
    "Lx" => 1.2e-3, "Ly" => 1.2e-3, "Lz" => 0.05e-3, "mat_id" => 4
)
silicon_1 = Dict(
    "x0" => 0.1e-3, "y0" => 0.1e-3, "z0" => zm2,
    "Lx" => 1.0e-3, "Ly" => 1.0e-3, "Lz" => 0.1e-3, "mat_id" => 2
)
silicon_2 = Dict(
    "x0" => 0.1e-3, "y0" => 0.1e-3, "z0" => zm5,
    "Lx" => 1.0e-3, "Ly" => 1.0e-3, "Lz" => 0.1e-3, "mat_id" => 2
)
silicon_3 = Dict(
    "x0" => 0.1e-3, "y0" => 0.1e-3, "z0" => zm8,
    "Lx" => 1.0e-3, "Ly" => 1.0e-3, "Lz" => 0.1e-3, "mat_id" => 2
)
heatsink = Dict(
    "x0" => 0.0, "y0" => 0.0, "z0" => zm11,
    "Lx" => 1.2e-3, "Ly" => 1.2e-3, "Lz" => 0.05e-3, "mat_id" => 5
)

# geomに登録
geom = Dict{String, Any}[]
push!(geom, silicon_1)
push!(geom, silicon_2)
push!(geom, silicon_3)
push!(geom, substrate)
push!(geom, heatsink)

# 物性値
# λ 熱伝導率 [W/mK]
# ρ 密度 [kg/m^3]
# C 比熱 [J/KgK]
# α 温度拡散率 [m^2/s]

# TSV : Yellow
cupper = Dict(
    "id" => 1, "λ" => 386.0, "ρ" => 8960.0, "C" => 383.0, "α" => 1.12e-4
)
# Silicon : Green
silicon = Dict(
    "id" => 2, "λ" => 149.0, "ρ" => 2330.0, "C" => 720.0, "α" => 8.88e-5
)
# bump ハンダ : Purple
solder = Dict(
    "id" => 3, "λ" => 50.0, "ρ" => 8500.0, "C" => 197.0, "α" => 2.99e-5
)
# PCB subtrate : Orange
FR4 = Dict(
    "id" => 4, "λ" => 0.4, "ρ" => 1850.0, "C" => 1000.0, "α" => 2.16e-7
)
# Heatsink : Blue
A1060 = Dict(
    "id" => 5, "λ" => 222.0, "ρ" => 2700.0, "C" => 921.0, "α" => 8.93e-5
)
# Underfill : Grey
Resin = Dict(
    "id" => 6, "λ" => 1.5, "ρ" => 2590.0, "C" => 1050.0, "α" => 5.52e-7
)
# Power grid, Silicon : Red
pwrsrc = Dict(
    "id" => 7, "λ" => 149.0, "ρ" => 2330.0, "C" => 720.0, "α" => 8.88e-5
)

mat = Dict{String, Any}[]
push!(mat, cupper)
push!(mat, silicon)
push!(mat, solder)
push!(mat, FR4)
push!(mat, A1060)
push!(mat, Resin)
push!(mat, pwrsrc)

# =======================================

function searchMat(m::Int64)
    for i in 1:length(mat)
        if mat[i]["id"] == m
            return i
        end
    end
    # if exit for-loop
    println("search material error")
    exit(0)
end



function chk_idx(i, nx)
    if i<1
        i = 1
    end
    if i>nx
        i = nx
    end
    return i
end

function find_Ri(x::Float64, r, x0, dx, nx)
    is = floor( Int64, (x-r-x0)/dx+1.5 )
    ie = floor( Int64, (x+r-x0)/dx+1.5 )
    is = chk_idx(is, nx)
    ie = chk_idx(ie, nx)
    return is, ie
end

function find_Rj(y::Float64, r, y0, dy, ny)
    js = floor( Int64, (y-r-y0)/dy+1.5 )
    je = floor( Int64, (y+r-y0)/dy+1.5 )
    js = chk_idx(js, ny)
    je = chk_idx(je, ny)
    return js, je
end

function find_i(x::Float64, x0, dx, nx)
    i = floor( Int64, (x-x0)/dx+1.5 )
    return chk_idx(i, nx)
end

function find_j(y::Float64, y0, dy, ny)
    j = floor( Int64, (y-y0)/dy+1.5 )
    return chk_idx(j, ny)
end

function find_k(Z::Vector{Float64}, zc, nz)

    # 浮動小数点誤差を考慮した範囲チェック
    eps_tol = 1e-12
    if zc < Z[1] - eps_tol || zc > Z[nz] + eps_tol
        println("out of scope in Z : find_z()")
        println(zc, " ", round(Z[nz],digits=8))
        exit()
    end

    for k in 1:nz-1
        if Z[k] ≤ zc < Z[k+1]
            return k
        end
    end

    # zcがZ[nz]とほぼ等しい場合（浮動小数点誤差を考慮）
    if abs(zc - Z[nz]) < eps_tol
        return nz - 1
    end
end


# ジオメトリのbboxを計算
function find_index(b, L, ox, Δh, SZ, Z::Vector{Float64})
    st = zeros(Int64, 3)
    ed = zeros(Int64, 3)

    st[1] = chk_idx( find_i(b[1], ox[1], Δh[1], SZ[1]), SZ[1])
    st[2] = chk_idx( find_i(b[2], ox[2], Δh[2], SZ[2]), SZ[2])
    st[3] = find_k(Z, b[3], SZ[3])
    ed[1] = chk_idx( find_i(b[1]+L[1], ox[1], Δh[1], SZ[1]), SZ[1])
    ed[2] = chk_idx( find_i(b[2]+L[2], ox[2], Δh[2], SZ[2]), SZ[2])
    ed[3] = find_k(Z, b[3]+L[3], SZ[3])
    return st, ed
end


"""
    is_included_rect(a1, a2, b1, b2) -> Bool

セルA(a1,a2) が直方体のジオメトリ領域B(b1,b2) に 50%以上含まれている場合 true を返す。
a1,a2,b1,b2 は対角をなす2点の座標 (x,y,z) タプル。
"""
function is_included_rect(a1, a2, b1, b2)
    # Aの体積
    volA = abs((a2[1] - a1[1]) * (a2[2] - a1[2]) * (a2[3] - a1[3]))

    # 重なり体積
    axlo, aylo, azlo = min.(a1, a2)
    axhi, ayhi, azhi = max.(a1, a2)
    bxlo, bylo, bzlo = min.(b1, b2)
    bxhi, byhi, bzhi = max.(b1, b2)

    ox = max(0.0, min(axhi, bxhi) - max(axlo, bxlo))
    oy = max(0.0, min(ayhi, byhi) - max(aylo, bylo))
    oz = max(0.0, min(azhi, bzhi) - max(azlo, bzlo))

    overlap_vol = ox * oy * oz

    return overlap_vol >= 0.5 * volA
end

"""
    is_half_or_more_included_cuboid_in_cylinder(a1, a2, cyl_center, cyl_radius, cyl_zmin, cyl_zmax; samples=50)

直方体(a1,a2)の体積のうち、円柱（Z軸方向）に含まれる割合が50%以上ならtrue。
円柱は中心xy座標 `cyl_center`、半径 `cyl_radius`、
高さ範囲 [cyl_zmin, cyl_zmax] で指定。
"""
function is_included_cyl(a1, a2, cyl_ctr, cyl_r, cyl_zmin, cyl_zmax; samples=50)
    # 直方体の境界
    xlo, ylo, zlo = min.(a1, a2)
    xhi, yhi, zhi = max.(a1, a2)

    volA = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)

    inside_count = 0
    total_count = 0

    for i in 1:samples, j in 1:samples, k in 1:samples
        # 小セルの中心座標
        x = xlo + (i - 0.5) * (xhi - xlo) / samples
        y = ylo + (j - 0.5) * (yhi - ylo) / samples
        z = zlo + (k - 0.5) * (zhi - zlo) / samples

        # 円柱内判定
        dx = x - cyl_ctr[1]
        dy = y - cyl_ctr[2]
        r2 = dx^2 + dy^2
        if r2 <= cyl_r^2 && cyl_zmin <= z <= cyl_zmax
            inside_count += 1
        end
        total_count += 1
    end

    overlap_vol = volA * inside_count / total_count
    return overlap_vol >= 0.5 * volA
end

"""
    is_half_or_more_included_cuboid_in_sphere(a1, a2, center, radius; samples=50) -> Bool

直方体 A (対角点 a1, a2) の体積のうち、球 (center, radius) に含まれる割合が50%以上なら true を返す。
評価は一様グリッドサンプリングで行い、samples^3 点で近似する（samplesを増やすと精度↑, 計算量↑）。

引数:
- a1, a2 :: (x,y,z) 直方体の対角点（順不同）
- center :: (cx,cy,cz) 球の中心
- radius :: 半径
- samples :: 各軸の分割数（既定 50）

注意:
- 直方体は軸平行（AABB）を仮定
- 早期判定: 直方体の8頂点がすべて球内なら即 true（完全包含）
"""
function is_included_sph(a1, a2, center, radius; samples::Int=50)
    # 直方体の境界（昇順にそろえる）
    xlo, ylo, zlo = min.(a1, a2)
    xhi, yhi, zhi = max.(a1, a2)

    # 体積
    volA = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)
    volA ≤ 0 && return false  # 退避：変な入力

    cx, cy, cz = center
    r2 = radius^2

    # 早期 true 判定：8頂点がすべて球内なら完全包含
    corners = ((xlo,ylo,zlo),(xlo,ylo,zhi),(xlo,yhi,zlo),(xlo,yhi,zhi),
               (xhi,ylo,zlo),(xhi,ylo,zhi),(xhi,yhi,zlo),(xhi,yhi,zhi))
    all_inside = all(((x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ r2) for (x,y,z) in corners)
    if all_inside
        return true
    end

    # サンプリング（セル中心）
    inside_count = 0
    total_count  = samples^3

    dx = (xhi - xlo) / samples
    dy = (yhi - ylo) / samples
    dz = (zhi - zlo) / samples

    x = xlo + dx/2
    for i in 1:samples
        y = ylo + dy/2
        for j in 1:samples
            z = zlo + dz/2
            for k in 1:samples
                # 点が球内か
                if (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ r2
                    inside_count += 1
                end
                z += dz
            end
            y += dy
        end
        x += dx
    end

    overlap_vol_est = volA * (inside_count / total_count)
    return overlap_vol_est ≥ 0.5 * volA
end

# ジオメトリのbboxをフィル
function FillPlate!(ID::Array{UInt8,3}, ox, Δh, Z::Vector{Float64})
    SZ = size(ID)
    b = zeros(Float64, 3)
    L = zeros(Float64, 3)
    c1= zeros(Float64, 3)
    c2= zeros(Float64, 3)
    d1= zeros(Float64, 3)
    d2= zeros(Float64, 3)

    for m in 1:5
        b[1] = geom[m]["x0"]
        b[2] = geom[m]["y0"]
        b[3] = geom[m]["z0"]
        L[1] = geom[m]["Lx"]
        L[2] = geom[m]["Ly"]
        L[3] = geom[m]["Lz"]
        st, ed = find_index(b, L, ox, Δh, SZ, Z)
        #@printf(stdout, "SILICON : [%d %d %d] - [%d %d %d]\n",st[1],st[2],st[3], ed[1],ed[2], ed[3])
        d1 = b
        d2 .= b + L
        for k in st[3]-1:ed[3], j in st[2]-1:ed[2], i in st[1]-1:ed[1]
            c1[1] = ox[1] + Δh[1]*(i-1)
            c1[2] = ox[2] + Δh[2]*(j-1)
            c1[3] = Z[k]
            c2[1] = c1[1] + Δh[1]
            c2[2] = c1[2] + Δh[2]
            c2[3] = Z[k+1]
            #@printf(stdout, "cell : [%f %f %f] - [%f %f %f]\n",c1[1],c1[2],c1[3], c2[1],c2[2], c2[3])
            if is_included_rect(c1, c2, d1, d2) && (0 == ID[i,j,k])
                l = searchMat( geom[m]["mat_id"] )
                ID[i,j,k] = mat[l]["id"]
                #println("$i $j $k")
            end
        end
    end
end


# 厚さ5µの領域
function FillPowerGrid!(ID::Array{UInt8,3}, ox, Δh, Z::Vector{Float64}, lx=0.2e-3, ly=0.2e-3)
    SZ = size(ID)
    c1= zeros(Float64, 3)
    c2= zeros(Float64, 3)
    d1= zeros(Float64, 3)
    d2= zeros(Float64, 3)
    s = s_dpth-pg_dpth
    pg_count = 0
    for z in [zm2+s, zm5+s, zm8+s], y in [0.3e-3, 0.7e-3], x in [0.3e-3, 0.7e-3]
        b = [x, y, z]
        L = [lx, ly, pg_dpth]
        st, ed = find_index(b, L, ox, Δh, SZ, Z)
        d1 = b
        d2 .= b + L
        #@printf(stdout, "target : [%f %f %f] - [%f %f %f]\n",d1[1],d1[2],d1[3], d2[1],d2[2],d2[3])
        for k in st[3]-1:ed[3], j in st[2]-1:ed[2], i in st[1]-1:ed[1]
            c1[1] = ox[1] + Δh[1]*(i-1)
            c1[2] = ox[2] + Δh[2]*(j-1)
            c1[3] = Z[k]
            c2[1] = c1[1] + Δh[1]
            c2[2] = c1[2] + Δh[2]
            c2[3] = Z[k+1]
            if is_included_rect(c1, c2, d1, d2) && (0 == ID[i,j,k])
                ID[i,j,k] = pwrsrc["id"]
                pg_count += 1
                #println("$i $j $k")
            end
        end
    end
end 


function FillResin!(ID::Array{UInt8,3})
    SZ = size(ID)
    for k in 1:SZ[3], j in 1:SZ[2], i in 1:SZ[1]
        if 0 == ID[i,j,k]
            ID[i,j,k] = Resin["id"]
        end
    end
end


function FillTSV!(ID::Array{UInt8,3}, ox, Δh, Z::Vector{Float64})
    SZ = size(ID)
    c1= zeros(Float64, 3)
    c2= zeros(Float64, 3)
    cyl_ctr= zeros(Float64, 2)
    cyl_r = r_tsv

    for z in [zm2, zm5, zm8], y in [0.3e-3, 0.5e-3, 0.7e-3, 0.9e-3], x in [0.3e-3, 0.5e-3, 0.7e-3, 0.9e-3]
        cyl_ctr[1] = x
        cyl_ctr[2] = y
        cyl_zmin = z
        cyl_zmax = z + h_tsv
        is, ie = find_Ri(x, cyl_r, ox[1], Δh[1], SZ[1])
        js, je = find_Rj(y, cyl_r, ox[2], Δh[2], SZ[2])
        ks = find_k(Z, cyl_zmin, SZ[3])
        ke = find_k(Z, cyl_zmax, SZ[3])
        #@printf(stdout, "TSV : [%d - %d]\n",ks, ke)

        for k in ks-1:ke, j in js-1:je, i in is-1:ie
            c1[1] = ox[1] + Δh[1]*(i-1)
            c1[2] = ox[2] + Δh[2]*(j-1)
            c1[3] = Z[k]
            c2[1] = c1[1] + Δh[1]
            c2[2] = c1[2] + Δh[2]
            c2[3] = Z[k+1]
            if (0 == ID[i,j,k]) && is_included_cyl(c1, c2, cyl_ctr, cyl_r, cyl_zmin, cyl_zmax)
                ID[i,j,k] = cupper["id"]
            end
        end
    end
end


function FillSolder!(ID::Array{UInt8,3}, ox, Δh, Z::Vector{Float64})
    SZ = size(ID)
    r = r_bump # ball radius
    c1= zeros(Float64, 3)
    c2= zeros(Float64, 3)
    ctr= zeros(Float64, 3)
    dp = d_ufill*0.5

    for z in [zm1+dp, zm4+dp, zm7+dp, zm10+dp], y in [0.3e-3, 0.5e-3, 0.7e-3, 0.9e-3], x in [0.3e-3, 0.5e-3, 0.7e-3, 0.9e-3]
        ctr[1] = x
        ctr[2] = y
        ctr[3] = z
        is, ie = find_Ri(x, r, ox[1], Δh[1], SZ[1])
        js, je = find_Rj(y, r, ox[2], Δh[2], SZ[2])
        ks = find_k(Z, z-dp, SZ[3])
        ke = find_k(Z, z+dp, SZ[3])
        #@printf(stdout, "BUMP: [%d %d %d - %d %d %d]\n", is, js, ks, ie, je, ke)

        for k in ks-1:ke, j in js-1:je, i in is-1:ie
            c1[1] = ox[1] + Δh[1]*(i-1)
            c1[2] = ox[2] + Δh[2]*(j-1)
            c1[3] = Z[k]
            c2[1] = c1[1] + Δh[1]
            c2[2] = c1[2] + Δh[2]
            c2[3] = Z[k+1]
            if (0 == ID[i,j,k]) && is_included_sph(c1, c2, ctr, r)
                ID[i,j,k] = solder["id"]
            end
        end
    end
end




function setProperties!(λ::Array{Float64,3},
                        ρ::Array{Float64,3},
                        cp::Array{Float64,3},
                        ID::Array{UInt8,3})
    SZ = size(λ)
    for k in 1:SZ[3], j in 1:SZ[2], i in 1:SZ[1]
        t = ID[i,j,k]
        if t == pwrsrc["id"]
            λ[i,j,k] = pwrsrc["λ"]
            ρ[i,j,k] = pwrsrc["ρ"]
           cp[i,j,k] = pwrsrc["C"]

        elseif t == cupper["id"]
            λ[i,j,k] = cupper["λ"]
            ρ[i,j,k] = cupper["ρ"]
           cp[i,j,k] = cupper["C"]

        elseif t == silicon["id"]
            λ[i,j,k] = silicon["λ"]
            ρ[i,j,k] = silicon["ρ"]
           cp[i,j,k] = silicon["C"]

        elseif t == FR4["id"]
            λ[i,j,k] = FR4["λ"]
            ρ[i,j,k] = FR4["ρ"]
           cp[i,j,k] = FR4["C"]

        elseif t == A1060["id"]
            λ[i,j,k] = A1060["λ"]
            ρ[i,j,k] = A1060["ρ"]
           cp[i,j,k] = A1060["C"]

        elseif t == solder["id"]
            λ[i,j,k] = solder["λ"]
            ρ[i,j,k] = solder["ρ"]
           cp[i,j,k] = solder["C"]

        else
            λ[i,j,k] = Resin["λ"]
            ρ[i,j,k] = Resin["ρ"]
           cp[i,j,k] = Resin["C"]
        end
    end
end


function fillID!(ID::Array{UInt8,3}, ox, Δh, Z::Vector{Float64})
    println("FillPowerGrid")
    FillPowerGrid!(ID, ox, Δh, Z)
    println("FillTSV")
    FillTSV!(ID, ox, Δh, Z)
    println("FillPlate")
    FillPlate!(ID, ox, Δh, Z)
    println("FillSolder")
    FillSolder!(ID, ox, Δh, Z)
    println("FillResin")
    FillResin!(ID)
end

# @param NXY  Number of inner cells for X&Y dir.
# @param NZ   Number of inner cells for Z dir.
function model_test(NXY::Int64, NZ::Int64=13)
    MX = MY = NXY + 2  # Number of CVs including boundary cells
    NZ = 13

    MZ = NZ + 2

    dx::Float64 = 1.2e-3 / NXY
    dy::Float64 = 1.2e-3 / NXY
    dz::Float64 = zm12 / NZ
    Δh = (dx, dy, dz) 
    ox = (0.0, 0.0, 0.0)
    SZ = (MX, MY, MZ)
    println(SZ)
    println(Δh)
    
    ID = zeros(UInt8, SZ[1], SZ[2], SZ[3])
    genZ!(Z, SZ, ox, Δh[3])

    @time fillID!(ID, ox, Δh, Z)


    #plot_slice(ID, SZ, "id.png")
end

end # end of moduleA

if abspath(PROGRAM_FILE) == @__FILE__
    using .modelA
    model_test(240,120)
end


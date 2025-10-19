using Printf

"""
parse_residuals_from_log(filename)
log.txtファイルから残差データを解析してConvergenceDataに追加

Args:
- filename: ログファイル名
- conv_data: ConvergenceData構造体

Returns:
- 解析成功時はtrue、失敗時はfalse
"""
function parse_residuals_from_log!(conv_data::ConvergenceData, filename::String="log.txt")
    if !isfile(filename)
        println("Warning: Log file $filename not found")
        return false
    end
    
    iteration_count = 0
    
    open(filename, "r") do f
        for line in eachline(f)
            # 残差行のパターンをマッチ：例 "         1     1.18960680068334E-01"
            if occursin(r"^\s+(\d+)\s+([0-9]+\.[0-9]+E[+-]\d+)", line)
                m = match(r"^\s+(\d+)\s+([0-9]+\.[0-9]+E[+-]\d+)", line)
                if m !== nothing
                    iter = parse(Int, m.captures[1])
                    residual = parse(Float64, m.captures[2])
                    add_residual!(conv_data, iter, residual)
                    iteration_count += 1
                end
            end
        end
    end
    
    if iteration_count > 0
        println("Loaded $iteration_count residual data points from log")
        return true
    else
        println("Warning: No residual data found in log file")
        return false
    end
end
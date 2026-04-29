import os
import re
import glob
import sqlite3
import pandas as pd

def parse_size_str(s):
    s = s.strip().upper()
    if s == "SKIP" or s == "0B": return 0
    match = re.match(r"([\d\.]+)\s*([KMGT]?B)", s)
    if not match: return 0
    val, unit = float(match.group(1)), match.group(2)
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return int(val * units.get(unit, 1))

def parse_results_txt(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    runs, current_run = [], {}
    run_idx = 0
    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            m = re.search(r"Backend:\s+(\w+)\s+\|", line)
            if m:
                if current_run and "Backend" in current_run: 
                    current_run["RunIdx"] = run_idx
                    runs.append(current_run)
                    run_idx += 1
                current_run = {"Backend": m.group(1)}
            m = re.search(r"MPI Grid:\s+\(([\d,\s]+)\)", line)
            if m and current_run is not None:
                # Extract the total number of MPI ranks from the grid
                grid_parts = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
                num_ranks = 1
                for p in grid_parts: num_ranks *= p
                current_run["MPI_Ranks"] = num_ranks
            m = re.search(r"Local Vol:\s+\(([\d,\s]+)\)", line)
            if m and current_run is not None:
                current_run["Local_Vol"] = m.group(1)
            m = re.search(r"Msg Size:\s+(.+)$", line)
            if m and current_run is not None:
                msg_parts = m.group(1).split(",")
                sizes = [parse_size_str(p.split(":")[1]) for p in msg_parts if ":" in p]
                total_bytes = sum(sizes)
                current_run["Total_Msg_MB"] = total_bytes / (1024*1024)
                current_run["Max_Msg_MB"] = max(sizes) / (1024*1024) if sizes else 0
            m = re.search(r"Iters\(Time\):(\d+)\s+\|\s+Math_Load\(AI\):(\d+)", line)
            if m and current_run is not None:
                current_run["Iters"], current_run["Math_Load"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"T_compute:\s+([\d\.]+)\s+ms\s+\(Isolated\)\s+\|\s+([\d\.]+)\s+ms\s+\(Overlap\)", line)
            if m and current_run is not None:
                current_run["T_comp_iso"], current_run["T_comp_ovl"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"T_comm:\s+([\d\.]+)\s+ms\s+\(Isolated\)\s+\|\s+([\d\.]+)\s+ms\s+\(Overlap\)", line)
            if m and current_run is not None:
                current_run["T_comm_iso"], current_run["T_comm_ovl"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"T_total:\s+([\d\.]+)\s+ms", line)
            if m and current_run is not None:
                current_run["T_total_ovl"] = float(m.group(1)) # Renamed to T_total_ovl for consistency
            m = re.search(r"Overlap Speedup:\s+([\d\.]+)x", line)
            if m and current_run is not None: current_run["Speedup"] = float(m.group(1))

    if current_run and "Backend" in current_run:
        current_run["RunIdx"] = run_idx
        runs.append(current_run)
    
    for run in runs:
        if all(k in run for k in ["T_comp_iso", "T_comm_iso", "T_total_ovl"]):
            run["T_serial"] = run["T_comp_iso"] + run["T_comm_iso"]
            run["Serial_Speedup"] = run["T_serial"] / run["T_total_ovl"]
            run["T_ideal"] = max(run["T_comp_iso"], run["T_comm_iso"])
            
            run["Contention_Factor"] = run["T_ideal"] / run["T_total_ovl"]
            
            denom = min(run["T_comp_iso"], run["T_comm_iso"])
            if denom > 0:
                eff = (run["T_serial"] - run["T_total_ovl"]) / denom
                run["Overlap_Efficiency"] = max(0.0, min(1.0, eff))
            else:
                run["Overlap_Efficiency"] = 0.0
            
            run["Comm_Comp_Ratio"] = run["T_comm_iso"] / run["T_comp_iso"] if run["T_comp_iso"] > 0 else 0
            
    return pd.DataFrame(runs)

def extract_metrics_from_sqlite(db_path):
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%DRAM%' OR metricName LIKE '%SM%'")
        metric_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        if not metric_map:
            conn.close()
            return None

        query_nvtx = """
        SELECT text, start, end 
        FROM NVTX_EVENTS 
        WHERE text LIKE 'Profiling_Overlapped_%'
        """
        cursor.execute(query_nvtx)
        nvtx_ranges = cursor.fetchall()

        results = []
        for text, start, end in nvtx_ranges:
            match = re.search(r'_Run(\d+)', text)
            if not match: continue
            run_idx = int(match.group(1))

            run_metrics = {"RunIdx": run_idx}
            for mid, mname in metric_map.items():
                query_val = f"SELECT AVG(value) FROM GPU_METRICS WHERE metricId = ? AND timestamp >= ? AND timestamp <= ?"
                cursor.execute(query_val, (mid, start, end))
                val = cursor.fetchone()[0]
                
                clean_name = mname.replace("Throughput", "BW").replace("Utilization", "Util").replace(" ", "_")
                run_metrics[f"Avg_{clean_name}"] = val if val is not None else 0.0

            results.append(run_metrics)

        conn.close()
        return results
    except Exception as e:
        print(f"Error processing {db_path}: {e}")
        return None

def parse_nsys_directory(results_dir):
    sqlite_files = sorted(glob.glob(os.path.join(results_dir, "**/*.sqlite"), recursive=True))
    if not sqlite_files:
        return pd.DataFrame()
        
    all_rank_data = []
    for f in sqlite_files:
        metrics = extract_metrics_from_sqlite(f)
        if metrics:
            all_rank_data.extend(metrics)
            
    if not all_rank_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rank_data)
    
    agg_funcs = {}
    for col in df.columns:
        if col != "RunIdx":
            agg_funcs[col] = ['mean', 'min', 'max']
            
    agg_df = df.groupby("RunIdx").agg(agg_funcs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    
    # Clean up column names (mean of average is just the average across ranks)
    rename_map = {}
    for col in agg_df.columns:
        if col.endswith('_mean'):
            rename_map[col] = col.replace('_mean', '')
    agg_df = agg_df.rename(columns=rename_map)
    
    return agg_df

def load_run_data(run_dirs):
    """
    Loads data from a list of results directories (or a single directory string).
    
    Autodetection:
    - If a directory name contains '_timing_', it parses Timing Data (results.txt).
    - If a directory name contains '_metrics_' (or legacy _profile_ / _nsys_), 
      it parses GPU Metrics (.sqlite files).
      
    If provided an equal number of timing and metrics directories (e.g. [dir_timing, dir_metrics]),
    it assumes they map 1:1 and merges them perfectly by RunIdx.
    """
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
        
    timing_dirs = []
    metrics_dirs = []
    
    for d in run_dirs:
        name = os.path.basename(os.path.normpath(d))
        if "_metrics_" in name or "_profile_" in name or "_nsys_" in name:
            metrics_dirs.append(d)
        else:
            timing_dirs.append(d)
            
    all_data = []
    
    if len(timing_dirs) == len(metrics_dirs) and len(timing_dirs) > 0:
        for t_dir, m_dir in zip(timing_dirs, metrics_dirs):
            df_t = parse_results_txt(os.path.join(t_dir, "results.txt"))
            df_m = parse_nsys_directory(m_dir)
            if not df_t.empty:
                df_t["Has_Timing_Data"] = True
                if not df_m.empty:
                    df = pd.merge(df_t, df_m, on="RunIdx", how="left")
                    df["Has_GPU_Metrics"] = True
                else:
                    df = df_t
                    df["Has_GPU_Metrics"] = False
                all_data.append(df)
    else:
        for t_dir in timing_dirs:
            df = parse_results_txt(os.path.join(t_dir, "results.txt"))
            if not df.empty:
                df["Has_Timing_Data"] = True
                df["Has_GPU_Metrics"] = False
                all_data.append(df)
        for m_dir in metrics_dirs:
            df = parse_nsys_directory(m_dir)
            if not df.empty:
                df["Has_GPU_Metrics"] = True
                df["Has_Timing_Data"] = False
                all_data.append(df)
                
    if not all_data:
        print("Warning: No valid data found in provided directories.")
        return pd.DataFrame()
        
    df_final = pd.concat(all_data, ignore_index=True)
    if "RunIdx" in df_final.columns:
        df_final = df_final.drop(columns=["RunIdx"])
    return df_final

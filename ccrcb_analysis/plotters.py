import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _check_cols(df, *cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot produce this plot. Missing required columns: {missing}.\n"
                         f"This usually means you loaded a 'Timing' run without hardware metrics. "
                         f"Please ensure you pass a '_metrics_' directory to load_run_data() if you want hardware plots.\n"
                         f"Available columns: {df.columns.tolist()}")

def plot_heatmap(df, metric, title="Heatmap", xlabel="Message Size (MB)", ylabel="Math Load", cmap="viridis", ax=None, **kwargs):
    """
    Plots a heatmap for a specific metric across Math_Load and Max_Msg_MB.
    """
    _check_cols(df, "Math_Load", "Max_Msg_MB", metric)
    
    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot the data
    pivot = df.pivot_table(index="Math_Load", columns="Max_Msg_MB", values=metric)
    
    # Plot heatmap
    cax = ax.imshow(pivot.values, cmap=cmap, aspect='auto', origin='lower')
    
    # Formatting
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel(xlabel, **kwargs.get('xlabel_kwargs', {}))
    ax.set_ylabel(ylabel, **kwargs.get('ylabel_kwargs', {}))
    ax.set_title(title, **kwargs.get('title_kwargs', {}))
    
    # Annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="w" if val < pivot.values.max()/2 else "k")
                
    plt.colorbar(cax, ax=ax, label=metric)
    return ax

def plot_contention_scatter(df, x_metric="Avg_DRAM_Read_Bandwidth_[BW_%]", y_metric="Contention_Factor", hue="Backend", ax=None, **kwargs):
    """
    Plots a scatter comparing Hardware Utilization vs Contention Factor.
    """
    _check_cols(df, x_metric, y_metric)
    
    if ax is None:
        figsize = kwargs.pop('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
    backends = df[hue].unique() if hue in df.columns else ["All"]
    colors = plt.cm.get_cmap('tab10', len(backends))
    
    for i, backend in enumerate(backends):
        sub_df = df[df[hue] == backend] if hue in df.columns else df
        ax.scatter(sub_df[x_metric], sub_df[y_metric], label=backend, color=colors(i), 
                   s=kwargs.get('s', 50), alpha=kwargs.get('alpha', 0.7))
                   
    ax.set_xlabel(kwargs.get('xlabel', x_metric))
    ax.set_ylabel(kwargs.get('ylabel', y_metric))
    ax.set_title(kwargs.get('title', f"{y_metric} vs {x_metric}"))
    if len(backends) > 1:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return ax

def plot_timing_histogram(df, title="Timing Breakdown", ax=None, **kwargs):
    """
    Plots a grouped bar chart (histogram) comparing T_comp_iso, T_comm_iso, T_ideal, and T_total_ovl across Backends.
    Expects df to be filtered to a single configuration (e.g. specific Math_Load and Max_Msg_MB).
    """
    _check_cols(df, "Backend", "T_comp_iso", "T_comm_iso", "T_ideal", "T_total_ovl")
    
    if ax is None:
        figsize = kwargs.pop('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
    backends = sorted(df['Backend'].unique())
    x = np.arange(len(backends))
    width = 0.2
    
    agg_df = df.groupby('Backend')[['T_comp_iso', 'T_comm_iso', 'T_ideal', 'T_total_ovl']].mean().reindex(backends)
    
    ax.bar(x - 1.5*width, agg_df['T_comp_iso'], width, label='Compute (Iso)', color='skyblue')
    ax.bar(x - 0.5*width, agg_df['T_comm_iso'], width, label='Comm (Iso)', color='lightcoral')
    ax.bar(x + 0.5*width, agg_df['T_ideal'], width, label='Ideal (Max)', color='lightgreen')
    ax.bar(x + 1.5*width, agg_df['T_total_ovl'], width, label='Total (Overlap)', color='gold')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    return ax

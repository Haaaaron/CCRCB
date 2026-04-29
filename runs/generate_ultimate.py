import os
import sys

def generate_ultimate_sweep(num_ranks=2):
    # Backends to test
    backends = ["MPI", "NCCL", "NVSHMEM"]
    
    # 1. Target Message Sizes (One direction, 8 bytes per site)
    # We define the "Face" (L1*L2*L3) to hit these targets exactly
    # Target MB: 0.125, 0.5, 2, 8, 32, 128, 512, 2048, 8192
    
    targets = [
        {"name": "128KB", "vol": (16, 32, 32)},
        {"name": "512KB", "vol": (32, 32, 64)},
        {"name": "2MB",   "vol": (64, 64, 64)},
        {"name": "8MB",   "vol": (128, 64, 128)},
        {"name": "32MB",  "vol": (128, 128, 256)},
        {"name": "128MB", "vol": (256, 256, 256)},
        {"name": "512MB", "vol": (512, 256, 512)},
        {"name": "2GB",   "vol": (512, 512, 1024)},
        {"name": "8GB",   "vol": (1024, 1024, 1024)},
    ]

    # 2. Arithmetic Intensity Scaling
    math_loads = [1, 4, 16, 64, 256, 1024]

    filename = "runs-ultimate.txt"
    with open(filename, "w") as f:
        f.write("# ====================================================================================\n")
        f.write("# ULTIMATE 2D PARAMETER SWEEP: KB TO GB MESSAGE SCALING\n")
        f.write(f"# partitioning: Optimized for {num_ranks} ranks (Global D0 = {num_ranks})\n")
        f.write("# Columns: Volume (Global) | Iters | Math_per_Load | Warmup | Repeats | Comm_Mask | Backend\n")
        f.write("# ====================================================================================\n\n")

        for t in targets:
            f.write(f"# --- Target Msg Size: {t['name']} ---\n")
            l1, l2, l3 = t['vol']
            global_vol = f"({num_ranks},{l1},{l2},{l3})"
            
            for mpl in math_loads:
                for b in backends:
                    # Adaptive Iterations: Keep benchmarks from running forever
                    # Goal: ~2-5 seconds per run
                    iters = 1000
                    if l1*l2*l3 > 10**6: iters = 200
                    if l1*l2*l3 > 10**7: iters = 50
                    if l1*l2*l3 > 10**8: iters = 10
                    if l1*l2*l3 > 5*10**8: iters = 5
                    
                    # Force repeats low for the absolute giants
                    repeats = 10
                    if l1*l2*l3 > 10**8: repeats = 5

                    f.write(f"{global_vol:<22} {iters:<6} {mpl:<6} 50      {repeats:<8} (1,1,1,1,1,1,1,1)    {b}\n")
            f.write("\n")
    
    print(f"Generated {filename} optimized for {num_ranks} ranks.")

if __name__ == "__main__":
    # Default to 2 ranks as per thea nodes, but allow override
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    generate_ultimate_sweep(nr)

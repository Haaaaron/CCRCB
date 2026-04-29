import os

def generate_massive_2d_sweep():
    # Backends to test
    backends = ["MPI", "NCCL", "NVSHMEM"]
    
    # 2D Parameter Grid:
    # 1. Message Size Scaling (Total Msg MB per rank)
    # We will vary one dimension (d1) while keeping d0=2 (global) for communication stress
    # Local message size (one direction) = d1*d2*d3 * 8 bytes
    # To get 1MB: 125,000 sites. 
    # To get 256MB: 33,554,432 sites.
    
    # Grid configurations for d1, d2, d3 to achieve specific target face sizes
    # Target MB:   [0.125, 0.5, 2, 8, 32, 128, 256, 512]
    # Local Faces: (64^3, 128x64^2, 128^2x64, 128^3, 256x128^2, 256^2x128, 256^3, 512x256^2)
    # Global Vol = (2, L1, L2, L3)
    volumes = [
        "(2,64,64,64)",      # ~0.125MB face / 0.25MB msg
        "(2,128,64,64)",     # ~0.25MB face  / 0.5MB msg
        "(2,128,128,64)",    # ~0.5MB face   / 1MB msg
        "(2,128,128,128)",   # ~1MB face     / 2MB msg
        "(2,256,128,128)",   # ~2MB face     / 4MB msg
        "(2,256,256,128)",   # ~4MB face     / 8MB msg
        "(2,256,256,256)",   # ~8MB face     / 16MB msg
        "(2,512,256,256)",   # ~16MB face    / 32MB msg
        "(2,512,512,256)",   # ~32MB face    / 64MB msg
        "(2,512,512,512)",   # ~64MB face    / 128MB msg
        "(2,1024,512,512)",  # ~128MB face   / 256MB msg
        "(2,1024,1024,512)", # ~256MB face   / 512MB msg
    ]

    # 2. Arithmetic Intensity Scaling
    # mpl = Math Per Load
    math_loads = [1, 4, 16, 64, 256, 1024]

    with open("runs-massive.txt", "w") as f:
        f.write("# ====================================================================================\n")
        f.write("# MASSIVE 2D PARAMETER SWEEP: MESSAGE SIZE vs ARITHMETIC INTENSITY\n")
        f.write("# Rules: Global X=2 -> Local X=1 (Pure Overlap Stress)\n")
        f.write("# Target: Scale T_comm / T_comp from 0.01 to 100\n")
        f.write("# ====================================================================================\n\n")

        for vol in volumes:
            # Calculate total local volume to ensure we don't crash (8 bytes per site)
            # Vol looks like (2, d1, d2, d3) -> Local (1, d1, d2, d3)
            # Memory = d1*d2*d3 * 8
            f.write(f"# --- VOLUME: {vol} ---\n")
            for mpl in math_loads:
                for b in backends:
                    # Iters scaled down for massive volumes to keep runtimes reasonable
                    iters = 500
                    if "512" in vol: iters = 100
                    if "1024" in vol: iters = 20
                    
                    f.write(f"{vol:<19} {iters:<6} {mpl:<6} 50      10       (1,1,1,1,1,1,1,1)    {b}\n")
            f.write("\n")

if __name__ == "__main__":
    generate_massive_2d_sweep()

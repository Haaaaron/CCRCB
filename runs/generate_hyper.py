import os

def generate_hyper_scaling():
    # Backends to test
    backends = ["MPI", "NCCL", "NVSHMEM"]
    
    # helper for power of 2 ranges
    # 16, 32, 64, 128, 256 (8 values)
    p2 = [16, 32, 64, 128, 256]

    with open("runs-hyper.txt", "w") as f:
        f.write("# ====================================================================================\n")
        f.write("# HYPER-SCALING 4D SYSTEMATIC SUITE (V3)\n")
        f.write("# Rules: All 4D, Max global volume 256^4, Power-of-2 scaling\n")
        f.write("# Columns: Volume (Global) | Iters | Math_per_Load | Warmup | Repeats | Comm_Mask | Backend\n")
        f.write("# ====================================================================================\n\n")

        # 1. Symmetric Scaling (Global: L^4)
        # Tests aggregate throughput as everything grows
        f.write("# --- 1. SYMMETRIC SCALING (L^4) ---\n")
        for l in p2:
            vol_str = f"({l},{l},{l},{l})"
            for b in backends:
                f.write(f"{vol_str:<17} 500    1      50      10       (1,1,1,1,1,1,1,1)    {b}\n")
        f.write("\n")

        # 2. Message Size Sweep (Fix X=2, Vary Y=Z=T)
        # X=2 Global -> X=1 Local (Minimal compute, pure communication stress)
        # Face sizes: 16^3 (32KB) to 256^3 (128MB)
        f.write("# --- 2. MESSAGE SIZE SWEEP (Global X=2, vary YZT) ---\n")
        for l in p2:
            vol_str = f"(2,{l},{l},{l})"
            for b in backends:
                f.write(f"{vol_str:<17} 500    1      50      10       (1,1,1,1,1,1,1,1)    {b}\n")
        f.write("\n")

        # 3. Bulk Compute Scaling (Fix Y=Z=T=128, vary X)
        # Fix message size at 128^3 * 8 bytes = 16MB
        # Scale X from 2 to 256 to see overlap efficiency
        f.write("# --- 3. BULK COMPUTE SCALING (Fixed 16MB Messages, grow X) ---\n")
        for x in [2, 4, 8, 16, 32, 64, 128, 256]:
            vol_str = f"({x},128,128,128)"
            for b in backends:
                f.write(f"{vol_str:<17} 500    1      50      10       (1,1,1,1,1,1,1,1)    {b}\n")
        f.write("\n")

        # 4. Arithmetic Intensity Sweep (Mid-Large 4D Volume)
        f.write("# --- 4. ARITHMETIC INTENSITY SWEEP (Vol=128^4) ---\n")
        vol_str = "(128,128,128,128)"
        for mpl in [1, 4, 16, 64, 256]:
            for b in backends:
                f.write(f"{vol_str:<17} 500    {mpl:<6} 50      10       (1,1,1,1,1,1,1,1)    {b}\n")
        f.write("\n")

        # 5. Communication Mask Sweep (Vol=128^4)
        f.write("# --- 5. COMMUNICATION MASK SWEEP (Vol=128^4) ---\n")
        masks = [
            "(1,0,0,0,0,0,0,0)", # X-Forward
            "(1,1,0,0,0,0,0,0)", # X-Both
            "(1,1,1,1,0,0,0,0)", # XY
            "(1,1,1,1,1,1,0,0)", # XYZ
            "(1,1,1,1,1,1,1,1)", # XYZT
        ]
        for mask in masks:
            for b in backends:
                f.write(f"{vol_str:<17} 500    1      50      10       {mask:<20} {b}\n")
        f.write("\n")

if __name__ == "__main__":
    generate_hyper_scaling()

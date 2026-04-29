import os

def generate_uniform_sweep():
    # Backends to test
    backends = ["MPI", "NCCL", "NVSHMEM"]
    
    # We want local volume to be (L, L, L, L)
    # Face size = L^3
    # Msg size (one direction) = L^3 * 8 bytes
    
    # Target L values and their approximate face sizes (MB) and local volumes (MB)
    # L=16:  Face=32KB,   Vol=0.5MB
    # L=32:  Face=256KB,  Vol=8MB
    # L=64:  Face=2MB,    Vol=128MB
    # L=100: Face=8MB,    Vol=800MB
    # L=128: Face=16MB,   Vol=2GB
    # L=160: Face=32MB,   Vol=5GB
    # L=200: Face=64MB,   Vol=12GB
    # L=256: Face=128MB,  Vol=32GB
    # L=320: Face=256MB,  Vol=80GB (Might be too big for some GPUs)
    
    l_values = [16, 32, 64, 100, 128, 160, 200, 256]
    
    # Math loads (Arithmetic Intensity)
    math_loads = [1, 4, 16, 64, 256, 1024]

    # For 2 ranks, mpi_dims=(2,1,1,1).
    # To have local (L,L,L,L), global must be (2L, L, L, L).
    # NOTE: Even on 2 ranks, if we enable all 8 directions in the mask,
    # the code will try to pack/unpack all 8, but only Dim 0 will actually 
    # call MPI/NCCL/NVSHMEM because others have neighbor=PROC_NULL.
    # To get UNIFORM communication in all directions, one needs 16 ranks (2,2,2,2).
    # On 2 ranks, only 2 directions communicate, but they are still uniform with each other.

    output_file = "runs/runs-uniform.txt"
    with open(output_file, "w") as f:
        f.write("# ====================================================================================\n")
        f.write("# UNIFORM 4D SWEEP: CUBIC LOCAL VOLUMES (L,L,L,L)\n")
        f.write("# This ensures all enabled directions have identical message sizes.\n")
        f.write("# To be run on 2 ranks. Global grid = (2L, L, L, L) -> Local grid = (L, L, L, L)\n")
        f.write("# ====================================================================================\n\n")

        for L in l_values:
            vol = f"({2*L},{L},{L},{L})"
            face_mb = (L**3 * 8) / (1024*1024)
            total_vol_mb = (L**4 * 8) / (1024*1024)
            f.write(f"# --- L={L}: Face={face_mb:.2f}MB, LocalVol={total_vol_mb:.1f}MB ---\n")
            for mpl in math_loads:
                for b in backends:
                    # Scale iterations to keep runtimes reasonable
                    iters = 200
                    if L >= 128: iters = 100
                    if L >= 200: iters = 50
                    if L >= 256: iters = 20
                    
                    # (1,1,1,1,1,1,1,1) means all directions enabled
                    f.write(f"{vol:<19} {iters:<6} {mpl:<6} 50      10       (1,1,1,1,1,1,1,1)    {b}\n")
            f.write("\n")
    
    print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_uniform_sweep()

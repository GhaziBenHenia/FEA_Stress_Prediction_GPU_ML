import os
import re
import gc
import numpy as np
import pyvista as pv
import torch

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "DataSet1000/data"   
OUTPUT_DIR = "DataSet1000/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# match the per-sample vtu files
randE_vtu_re = re.compile(r"rand_E(\d+)_p0_000000\.vtu$")
vm_vtu_re = re.compile(r"vm(\d+)_p0_000000\.vtu$")

# -------------------------
# Helpers
# -------------------------
def find_sample_files(data_dir):
    files = os.listdir(data_dir)
    randE = {}
    vm = {}
    for f in files:
        if f.endswith(".vtu"):
            m1 = randE_vtu_re.match(f)
            m2 = vm_vtu_re.match(f)
            if m1:
                sid = int(m1.group(1))
                randE[sid] = os.path.join(data_dir, f)
            elif m2:
                sid = int(m2.group(1))
                vm[sid] = os.path.join(data_dir, f)
    sample_ids = sorted(set(randE.keys()) & set(vm.keys()))
    return randE, vm, sample_ids

def extract_undirected_edges_from_mesh(mesh):
    """
    Extract unique undirected edges from a pyvista mesh.

    Returns: numpy array shape (2, n_edges) dtype=int64
    """
    cells = mesh.cells
    offsets = mesh.offset
    edges_set = set()

    # iterate cells using offsets
    n_cells = len(offsets)
    for i in range(n_cells):
        start = offsets[i]
        end = offsets[i+1] if (i+1) < n_cells else len(cells)
        pts = cells[start+1:end]  

        # keep (min,max) ordering to deduplicate
        L = len(pts)
        for a in range(L):
            for b in range(a+1, L):
                u = int(pts[a]); v = int(pts[b])
                if u == v:
                    continue
                if u < v:
                    edges_set.add((u, v))
                else:
                    edges_set.add((v, u))

    if len(edges_set) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    edges = np.array(list(edges_set), dtype=np.int64).T  # shape (2, n_edges)
    return edges

# -------------------------
# HELPER: loader example for training
# -------------------------
def build_data_from_sample(sample_npz_path, mesh_meta_path):
    meta = np.load(mesh_meta_path)
    coords = meta["coords"].astype(np.float32)      # (N,3)
    edge_index = meta["edge_index"].astype(np.int64)  # (2, n_edges)

    arr = np.load(sample_npz_path)
    E = arr["E"].astype(np.float32)  # (N,) float32 after cast
    y = arr["y"].astype(np.float32)

    # node features: coords + E (reshape E to (N,1))
    x = np.concatenate([coords, E.reshape(-1, 1)], axis=1).astype(np.float32)
    x_t = torch.from_numpy(x)
    edge_t = torch.from_numpy(edge_index)
    y_t = torch.from_numpy(y.reshape(-1, 1))

    return Data(x=x_t, edge_index=edge_t, y=y_t)

def main():
    # -------------------------
    # MAIN: discover files
    # -------------------------
    randE_files, vm_files, sample_ids = find_sample_files(DATA_DIR)
    print(f"Found {len(sample_ids)} matching samples.")

    if len(sample_ids) == 0:
        raise SystemExit("No matching samples found. Check DATA_DIR and filenames.")

    # -------------------------
    # Step 1: load reference mesh once (first sample)
    # -------------------------
    first_sid = sample_ids[0]
    print(f"Loading reference mesh for sample {first_sid} ...")

    # Prefer vtu file for mesh (we use the randE or vm vtu — both should be identical topo)
    ref_vtu_path = randE_files.get(first_sid, vm_files[first_sid])
    ref_mesh = pv.read(ref_vtu_path)

    coords = np.array(ref_mesh.points, dtype=np.float32)  # (N,3)
    print("Nodes:", coords.shape[0])

    # extract edges (unique, undirected)
    edges_np = extract_undirected_edges_from_mesh(ref_mesh)  # shape (2, n_edges) int64
    print("Edges:", edges_np.shape[1])

    # Save mesh metadata once (coords + edge_index)
    mesh_meta_path = os.path.join(OUTPUT_DIR, "mesh_metadata.npz")
    np.savez_compressed(mesh_meta_path,
                        coords=coords.astype(np.float32),
                        edge_index=edges_np.astype(np.int64))
    print("Saved mesh metadata to:", mesh_meta_path)

    # Free mesh memory
    del ref_mesh
    gc.collect()

    # -------------------------
    # Step 2: process each sample (save only per-node arrays)
    # -------------------------
    print("Processing each sample and saving E & y compressed (float32)...")
    saved = 0
    for sid in sample_ids:
        print(f" Sample {sid} ...", end="")
        e_path = randE_files[sid]
        vm_path = vm_files[sid]

        # reading the vtu loads mesh into memory, but we free it quickly
        e_mesh = pv.read(e_path)
        E_vals = np.array(e_mesh.point_data["E"], dtype=np.float32)  # (N,)
        del e_mesh

        vm_mesh = pv.read(vm_path)
        vm_vals = np.array(vm_mesh.point_data["sigma_vm"], dtype=np.float32)  # (N,)
        del vm_mesh

        # quick sanity checks
        if E_vals.shape[0] != coords.shape[0] or vm_vals.shape[0] != coords.shape[0]:
            print("  -> node count mismatch, skipping")
            continue

        # Save as float32 to preserve all values,
        E_save = E_vals.astype(np.float32)
        y_save = vm_vals.astype(np.float32)

        out_path = os.path.join(OUTPUT_DIR, f"sample_{sid:04d}.npz")
        np.savez_compressed(out_path, E=E_save, y=y_save)
        saved += 1
        print(" saved")

        # free and collect
        del E_vals, y_save, E_save, vm_vals
        gc.collect()

    print(f"Done. Saved {saved} sample files in {OUTPUT_DIR}")
    print("Mesh metadata stored in mesh_metadata.npz")

if __name__ == "__main__":
    main()

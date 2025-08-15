import numpy as np
import os
import pyvista as pv
import re

DATA_DIR = "DataSet1000/processed_data"
RAW_DIR = "DataSet1000/data"  

sample_re = re.compile(r"sample_(\d{4})\.npz$")

file_names = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]

print(f"Validating {len(file_names)} samples in {DATA_DIR}...\n")

for file in file_names:
    path = os.path.join(DATA_DIR, file)
    m = sample_re.match(file)
    if not m:
        print(f"Skipping file with unexpected name: {file}")
        continue
    sid = int(m.group(1))

    # Find corresponding raw VM VTU file
    raw_vm_file = f"vm{sid}_p0_000000.vtu"
    raw_vm_path = os.path.join(RAW_DIR, raw_vm_file)
    if not os.path.exists(raw_vm_path):
        print(f"Sample {file}: Raw VM file not found: {raw_vm_file}")
        continue

    try:
        arr = np.load(path)
        y_np = arr["y"].astype(np.float32)
    except Exception as e:
        print(f"Sample: {file}")
        print(f" - ERROR loading npz file: {e}")
        continue

    try:
        vm_mesh = pv.read(raw_vm_path)
        vm_vals = np.array(vm_mesh.point_data["sigma_vm"], dtype=np.float32)
        del vm_mesh
    except Exception as e:
        print(f"Sample: {file}")
        print(f" - ERROR loading VTU file: {e}")
        continue

    # Compare arrays
    if y_np.shape != vm_vals.shape:
        print(f"Sample: {file}")
        print(f" - Shape mismatch: npz {y_np.shape}, vtu {vm_vals.shape}")
        continue

    # Allclose for float comparison
    if np.allclose(y_np, vm_vals, atol=1e-3, rtol=1e-3):
        print(f"Sample: {file} - OK (matches raw VM data)")
    else:
        max_diff = np.max(np.abs(y_np - vm_vals))
        print(f"Sample: {file} - MISMATCH! Max abs diff: {max_diff}")

print("\nValidation complete.")

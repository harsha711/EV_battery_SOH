import scipy.io as sio
import h5py

mat_path = "path/to/RW9.mat"

# Try scipy first (works for MATLAB v5/v7)
try:
    data = sio.loadmat(mat_path)
    print("Loaded with scipy.io")
    for key in data:
        if not key.startswith('__'):
            print(f"  Key: {key}, Type: {type(data[key])}, Shape: {getattr(data[key], 'shape', 'N/A')}")
except NotImplementedError:
    # v7.3 .mat files are HDF5
    print("Loaded with h5py (v7.3 format)")
    with h5py.File(mat_path, 'r') as f:
        def print_structure(name, obj):
            print(f"  {name}: {type(obj)}")
        f.visititems(print_structure)

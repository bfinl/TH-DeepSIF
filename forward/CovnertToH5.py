import h5py
import numpy as np
from scipy.io import loadmat
import os
from multiprocessing import Pool
import traceback

# Configuration
base_path = "hfo_source/train_998_snr15_extenE7dwr002_10012024/"
total_files = 310127
rows, cols = 500, 994
hdf5_path = "hfo_source/train_998_snr15_extenE7dwr002.h5"

# -------- Load function (parallelizable) --------
def load_single_file(i):
    filepath = os.path.join(base_path, f"{i}.mat")
    try:
        with h5py.File(filepath, 'r') as f:
            data = f['nmm'][()]
    except OSError:
        try:
            data = loadmat(filepath)['nmm']
        except Exception:
            print(f"Failed to load {filepath}")
            traceback.print_exc()
            return i, None  # Mark failure

    # Final check
    data = np.array(data, dtype=np.float32)
    if data.shape != (rows, cols):
        print(f"Skipping {i}: shape mismatch {data.shape}")
        return i, None

    return i, data

# -------- Main execution --------
if __name__ == '__main__':
    with h5py.File(hdf5_path, 'w') as h5f:
        dset = h5f.create_dataset('nmm', shape=(total_files, rows, cols), dtype=np.float32)

        with Pool(processes=8) as pool:  # Adjust number of workers as needed
            for i, data in pool.imap(load_single_file, range(1, total_files + 1)):
                if data is not None:
                    dset[i - 1] = data
                else:
                    print("is None: ",i)
import numpy as np
import sys
import os

def load_data(fname, tail=0):
  """Load sequentially appended arrays from a file and take the average and variance."""
  with open(fname, "rb") as f:
    fsize = os.fstat(f.fileno()).st_size
    data = np.load(f)
    arr_size = f.tell()
    fsize -= fsize % arr_size  # cut off any half-saved data

    total_elements = fsize // arr_size

    if tail:
        n = min(tail, total_elements)
    else:
        n = total_elements

    f.seek(fsize - n * arr_size)

    all_data = []
    for _ in range(n):
        all_data.append(np.load(f))

    all_data = np.array(all_data)
    return np.mean(all_data, axis=0), np.var(all_data, axis=0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_density.py <path_to/density_matrix.npy>")
        sys.exit(1)

    fname = sys.argv[1]
    mean, var = load_data(fname, tail=1000) # averages over the last 1000 steps
    print("Shape:", mean.shape)
    print("Mean density matrix:\n", mean)

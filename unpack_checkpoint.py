"""
Concatenates the sharded, compressed .npz files of one time_<t>/ checkpoint into
the whole-domain .npy files forced_ns_droplets.cu reads. Leaves the .npz alone.

    python3 unpack_checkpoint.py data_cosine/forced_True/N_256_Re_398.1/time_89.0
"""
import numpy as np
import pathlib,sys

time_dir = pathlib.Path(sys.argv[1])

def shards(d, stem):
    return sorted(d.glob(f"{stem}_*.npz"), key = lambda p: int(p.stem.split("_")[-1]))

parts = [np.load(f) for f in shards(time_dir, "Fields_k")]
for key in ("uk", "vk", "wk"):
    np.save(time_dir/f"{key}.npy", np.concatenate([p[key] for p in parts], axis = 1))

for sub_dir in time_dir.iterdir():
    if not sub_dir.is_dir(): continue

    n_files = shards(sub_dir, "n")
    if n_files:
        np.save(sub_dir/"n.npy", np.concatenate([np.load(f)["n"] for f in n_files], axis = 0))

    state_files = shards(sub_dir, "state")
    if state_files:
        parts = [np.load(f) for f in state_files]
        for key in ("pos", "vel", "mass", "prtclid"): #! the only state the GPU code reads
            np.save(sub_dir/f"{key}.npy", np.concatenate([p[key] for p in parts], axis = 0))

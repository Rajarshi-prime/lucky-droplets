# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment notes

- This is a sandbox environment. Do NOT run Python code/scripts here (no `python`, `python3`, etc. via Bash or other tools). Instead, ask the user to run them separately.
- Do NOT LOOK at git history.
- Before answering, reread your response and replace any word a physics graduate student outside computer science might not know.
- Before editing, find the relevant code and identify the smallest change needed. Implement only that change. After editing, review the diff and remove anything that is not necessary.
- Make and agent called 'distiller' and store in /claude. 'distiller' reviews the diff at the end of every prompt response, and suggests edits that simplifies the changes according to the python style given below. Implement those changes.

# Python style

- Write the simplest correct implementation appropriate for this existing codebase.
- Match the style of existing files. Read them before writing code.
- Ask for a file name, read the file and then write code in the same style as that file.
- Make the smallest diff that solves the task. Do not touch unrelated code.
- Do not refactor unrelated code, if you notice something that could be improved, suggest that in bullet points rather than modifying.
- Prefer modifying existing functions over creating new ones.
- Prefer few functions. Do not split code into helpers used only once.
- A helper that is one expression behind a name is redundant. A helper that is a step of the algorithm is not.
- No classes unless state genuinely requires one.
- Minimum module docstrings. Comments only where logic is non-obvious, one line, plain words.
- No `if __name__ == "__main__":` unless I ask.
- No type hints, logging, argument parsing, or try/except unless asked.
- Do not add features, options, configuration, validation, or error handling I did not request.
- Do not add dependencies unless asked.
- Vectorize with NumPy where possible. No Python loops over arrays.
- Preserve existing formatting and line-break style.
- Do not reformat code that you did not need to change.
- Do not introduce abstractions or design patterns unless the existing code already uses them.
- Keep the implementation concise. Prefer obvious code over generalized code.

# Words to avoid in code and responses

- wire, mid-migration, live, simply, just, engine, landed,


## What this is

A pseudo-spectral direct numerical simulation (DNS) of forced homogeneous isotropic turbulence, one-way coupled to cloud-droplet-like inertial particles ("lucky droplets"). Velocity fields are evolved in Fourier space; particles/droplets are represented either as a continuum number-density field advected under the "slow manifold" approximation (small Stokes number) or as explicit Lagrangian point particles (finite Stokes number, the "big" particles). Parallelism is MPI with a 1D slab decomposition along x; particle-to-grid/grid-to-particle interpolation is numba-JIT-accelerated.

Intended runtime is an MPI cluster; this sandbox only has a small local `data_cosine/` sample dataset.

## Commands

There is no build step, linter, test suite, or dependency manifest in this repo. The runtime dependencies, taken from the imports, are `numpy`, `scipy`, `h5py`, `mpi4py`, `numba`.

Simulations are MPI programs invoked as (per-script arguments are positional, always `<kmax*eta> <gravity 0|1>` as the last two argv entries):

```
mpirun -np <num_process> python3 <forced-dns-*.py> <kmax_eta> <gravity_flag>
```

Each script reads only `sys.argv[-1]` (gravity on/off) and `sys.argv[-2]` (the target `kmax*eta`, variable `m`, which sets the viscosity); everything else is edited in the parameter block near the top of the file. `<num_process>` must divide `N` (grid resolution, hardcoded per-script, typically 256), since `Np = N//num_process` slices per rank.

Do not actually run these commands yourself in this sandbox — ask the user to run them.

## Architecture

### The five `forced-dns-*.py` entry points

These are independent, mostly-duplicated top-level scripts (not a shared library with variant flags) — each hardcodes its own grid size, timestep, save path, and physics. Startup is now shared (`initial_conditions.py`); checkpointing still is not, so a change to `save()` needs porting across the relevant variants by hand.

- `forced-dns-sm.py` — baseline: forced NS + slow-manifold number-density field `n`, no explicit particles. This is the file to imitate for style.
- `forced-dns-sm-clip-new.py` — same as `sm.py` but replaces the naive "clip negative `n` to zero and rescale" step with a spline/Newton root-find (`clip_zero(n, target = 1)`) that clips to a target mean exactly.
- `forced-dns-log-sm.py` — evolves `log(n)` instead of `n` directly (avoids needing clipping for positivity), different `dt`/`dt_save`/`M0`. Its RHS carries `vgradlogn`/`divv` terms instead of the `divvn` term the others use, and it has `clip_error` rather than `clip_zero`.
- `forced-dns-sm-big.py` — adds explicit Lagrangian "big" particles (`MPI_particles` from `particles.py`) for several Stokes numbers and initial conditions at once, alongside the slow-manifold field for the small particles. The only variant that places particles from scratch by the Q criterion.
- `forced-dns-sm-big-rndm.py` — big-particle variant with stochastic forcing on the particles (`stoch_updt` in `particles.py`). It has no density field at all (no `n` argument anywhere in its RHS/RK4/save), and selects the oldest rather than the newest checkpoint folder on restart.

Common structure inside each script: a parameter block and grid/FFT setup at module level → `forcing()` → `clip_zero()`/`clip_error()` → `full_RHS()` (spectral RHS for velocity + scalar/log-density) → `RK4()` time integrator → `save()` (periodic checkpointing) → `evolve_and_save()` (the main time-stepping loop) → an `InitialConditions(...)` construction and the two startup calls. The scripts are straight-line: setup, loading, and the final `evolve_and_save(t, u, n)` call all run at module level, so importing one runs the simulation. `load_hdf5()` survives in each script but nothing calls it.

### Read path vs write path differ

Every variant **reads** restart data from `loadPath`, an absolute cluster path defined next to `savePath` in the parameter block, and **writes** to a relative `savePath` under the working directory. The two do not have to agree — `sm-clip-new.py` writes to `./data_cosine_clip_new/...` and `sm-big-rndm.py` to `./data_cosine/forced_<isforcing>/stochastic/highhighn/N_<N>_Re_<re>`.

### `initial_conditions.py` — the shared startup, used by all five scripts

`InitialConditions` owns both startup paths: find and load a saved state, or build a fresh one. Two switches in each script's parameter block choose between them — `forcestart` (fresh random velocity field vs. saved fields) and `start_big_particle` (place particles from scratch vs. load their saved state). Each script builds the grid, the spectral operators and the MPI fft helpers itself and passes them to the constructor; the class allocates its own scratch arrays, so the script's work buffers stay out of the interface.

The variants are expressed as constructor arguments rather than as separate code paths:

- `mode` — which time folder to restart from. Both particle variants use `"all_stokes"`: per Stokes number, the newest folder holding that Stokes number's particle directory, then the *minimum* across Stokes numbers, then one `dt_save` earlier. The minimum guarantees a time at which every particle set has state; the `dt_save` step avoids a final folder that was still being written. The density-only variants use `"second_last"`, which just sorts by modification time. `"first"` (oldest folder) is still supported but no longer used by any script.
- `clip` — the variant's `clip_zero` or `clip_error`, applied once to the density after it is loaded or built. For a 4-D `n` it is applied one Stokes row at a time, because the mean inside it is a global sum across ranks.
- `ntransform` / `ndefault` — log-sm passes `np.log` and `0.0`, since it evolves `log(n)` while the files on disk hold `n`.
- `load_dealias` — sm, clip and log multiply by `dealias` when going to physical space after loading; big does not. Passed as `False` there.
- `fresh_particles` — how particles are placed when starting them from scratch: `"qcriterion"` (big) or `"velocity"` (rndm, which keeps the positions `MPI_particles` gave them and only interpolates the fluid velocity onto them). Only consulted when `start_big_particle` is set.
- `init_name` — `None` for rndm, whose particle folders are `<wg>_stb_<stb>_sts_<sts>` with no `_init_` suffix. `particle_dir()` is the single place that name is built.
- `n = None` — rndm has no density field at all, and every density branch checks for this.

A `time_<t>` folder name carries `tdec = max(0, int(np.ceil(-np.log10(dt_save))))` decimals, so `0.5` and `0.1` give one and `0.05` gives two. That line appears three times — in `InitialConditions.__init__` (as `self.tdec`, used by `find_path` to match) and in the parameter block of `big` and `rndm` (used by their `save()` to write). Those two are the only scripts that create `time_` folders; the other three always write to `savePath/"last"`. Keep the three copies in step, and note that changing `dt_save` renames the folders, so a run cannot restart from data saved under a different `dt_save`.

There is one particle loader, `load_particles()`: each rank takes the saved particles falling inside its own x-range, so the number of saved files need not match the rank count. Reading `state_{rank}.npz` alone is the special case where the saved run used as many slabs as this run has ranks — rndm's old rank-wise reader was exactly that case and has been removed.

Fields are sharded across `load_num_slabs` `Fields_k_{slab}.npz` files, and `load_num_slabs` is whatever the *writing* run used — it is independent of the current run's `num_process`. `load_fields()` reads the velocity and the density in a single pass, caching each slab's density file so a slab is opened once rather than once per x-slice.

### `restart.py` — standalone, superseded, not used by any script

An earlier refactor of the same restart logic, now covered by `InitialConditions`. **No script imports it**, so changing it alone changes nothing that runs. Its contents are uncommitted (the file is empty at HEAD), which is why it was left in place rather than removed.

### `particles.py` — Lagrangian particle engine

`MPI_particles` holds per-particle state (position, velocity, mass) and does cosine-kernel interpolation between the Eulerian grid and particle positions (`uinterp_cosine`, `interp_cosine`, `exterp_cosine_scalar/vector`, `interp_exterp_cosine_scalar/vector`), particle exchange across MPI ranks when particles cross slab boundaries (`particle_exchange`), and the particle RHS (`pRHS`, `pRHS_inertial`, `stoch_updt`). The core interpolation/deposition kernels (`_calc_usend_numba`, `_calc_uadd_numba_scalar`, `_calc_uadd_numba_vector`) are free functions decorated with `@njit(parallel=True)` — keep them numba-compatible (no Python objects, no unsupported numpy features) when editing.

### Analysis/plotting scripts

`plot.py`, `plot-correlation.py`, and the dated `plot-YYYYMMDD.py` files are standalone, cell-based (`#%%`) analysis scripts that read simulation output from `data_cosine/...` and are not imported by anything. They are disposable/exploratory — new analysis is normally added as a new dated file rather than by editing an old one in place.

## Data layout

`data_cosine/forced_<True|False>/N_<N>_Re_<Re>/time_<t>/`:
- `Fields_k_<slab>.npz` — Fourier-space velocity field shards, keyed `uk`, `vk`, `wk`.
- `Energy_spectrum.npz`, `Flux_spectrum.npz` — diagnostics saved alongside the fields.
- Per-Stokes-number subdirectories holding `n_<rank>.npz` (number density) and/or `state_<rank>.npz` (big-particle state: `pos`, `vel`, `mass`, `prtclid`, `umat`).

The subdirectory naming is inconsistent across variants and across vintages of saved data — both `<wg>_sts_<sts>_stb_<stb>_init_<init>` and `<wg>_stb_<stb>_sts_<sts>_init_<init>` orderings appear in the same sample `time_` folder. The density field uses the first ordering (built inline in `load_fields`) and the particle state the second (`particle_dir`), so code that searches for these directories must match the ordering the writing script used.

`data_cosine/` in this repo is a small local sample (one `time_89.0` folder at `N_256_Re_398.1`); it is gitignored and is not the real dataset, which lives on the cluster.

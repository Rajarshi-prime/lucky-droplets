# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment notes

- This is a sandbox environment. Do NOT run Python code/scripts here (no `python`, `python3`, etc. via Bash or other tools). Instead, ask the user to run them separately.
- Do NOT LOOK at git history.
- Before answering, reread your response and replace any word a physics graduate student outside computer science might not know.
- Keep responses brief, concise and clear. State the cause and the fix. Cut background, restatement and anything I did not ask for.
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
USE PROPER ENGLISH WITH FULL SENTENCES.

- wire, mid-migration, live, simply, just, engine, landed, folded, cadence, stack


## What this is

Pseudo-spectral DNS of forced homogeneous isotropic turbulence, one-way coupled to cloud-droplet-like inertial particles ("lucky droplets"). Velocity evolves in Fourier space. Droplets are either a continuum number-density field `n` under the slow-manifold approximation (small Stokes) or explicit Lagrangian points (finite Stokes, the "big" particles). MPI with a 1-D slab decomposition along x; grid-particle interpolation is numba-JIT.

Runtime is an MPI cluster. This sandbox holds only a small `data_cosine/` sample (one `time_89.0` folder at `N_256_Re_398.1`), gitignored, not the real dataset.

## Commands

No build step, linter, tests, or dependency manifest. Imports need `numpy`, `scipy`, `h5py`, `mpi4py`, `numba`.

```
mpirun -np <num_process> python3 <forced-dns-*.py> <kmax_eta> <gravity_flag>
```

Only `sys.argv[-2]` (target `kmax*eta`, sets viscosity) and `sys.argv[-1]` (gravity 0|1) are read; the rest is the parameter block at the top of each file. `num_process` must divide `N`, since a rank owns `Np = N//num_process` slices. Ask the user to run these.

## The five `forced-dns-*.py` entry points

Independent, mostly duplicated top-level scripts, not a library with flags. Startup is shared (`initial_conditions.py`); checkpointing is not, so a `save()` change needs porting by hand.

| script | what differs |
| --- | --- |
| `forced-dns-sm.py` | baseline, forced NS + slow-manifold `n`, no particles. **Imitate this for style.** |
| `forced-dns-sm-clip-new.py` | `clip_zero(n, target = 1)` by spline/Newton root-find instead of clip-and-rescale |
| `forced-dns-log-sm.py` | evolves `log(n)`; RHS carries `vgradlogn`/`divv` not `divvn`; has `clip_error` |
| `forced-dns-sm-big.py` | Lagrangian particles for several Stokes numbers at once; only variant placing them by the Q criterion; only variant saving fields and particles on separate intervals |
| `forced-dns-sm-big-rndm.py` | stochastic particle forcing (`stoch_updt`); no density field; one `dt_save` for everything |

Order inside each: parameter block → `forcing()` → `clip_*()` → `full_RHS()` → `RK4()` → `save()` → `evolve_and_save()`. All of it runs at module level, so importing one starts the simulation. `load_hdf5()` survives in each but nothing calls it. Reads come from `loadPath` (absolute, cluster), writes go to a relative `savePath`; the two need not agree.

**Where the time goes in `big`.** The density loop runs 20 times per RHS call and four times per step at about eleven MPI transforms each — roughly 880 per step against 96 for the velocity — and `pRHS` calls `send`, so 80 `Alltoall` + `Alltoallv` rounds per step. The 20 fields cannot be collapsed into one: `v` is built from `tps` and is shared, but the fields differ through the droplet feedback `fc`. The transform of `fc` is not bookkeeping — `fc` is laid down by the cosine kernel and reaches the grid Nyquist, so the forward transform is what projects it onto the retained modes.

**Known and unfixed in `big`.** The blow-up guard tests `uk.max()`, which orders complex numbers by real part first, so a blow-up in the imaginary part passes. `clip_zero` divides by `newmean` unguarded, and its two branches differ in kind: `list = True` mutates and returns the same array, the plain form mutates and returns a different one. The end-of-step clip calls the `list = True` form on a single Stokes row, so the loop runs over the rank's `Np` x-slices and rescales each group of planes separately — the same answer as the plain form only at `num_process = N`, where `Np` is 1. `h` is bounded by `0.1*stmin`, a dimensionless Stokes number rather than the relaxation time `st*tf`; left that way on purpose. `dtmin` sets the first step and never floors `hnew`. The save trigger holds only while `h < dt_save_particles`, which `dtmax` does not enforce. Clipping is applied to the RK4 stage values, dropping the scheme below fourth order in `n` — a known compromise. `save` hands the module-level time array to `full_RHS` where its own `tt` is in hand, harmless as `pRHS` never reads it. Dead: `load_trunc`, `load_hdf5`, `cond_ky`/`cond_kz`, `param`, `ek_arr0`, and the `rs` → `nprtcls0` → `nmin_thresh` chain. `review-20261006.md` carries these with line numbers, along with the open speed-ups.

## `initial_conditions.py`

`InitialConditions` owns both startup paths: load a saved state, or build a fresh one, chosen by `forcestart` and `start_big_particle`. Variants are constructor arguments, not separate code paths:

- `mode` — `"all_stokes"` (both particle variants): of the folders holding a `Fields_k_*.npz`, the newest per Stokes number, then the *minimum* across them, then one `dt_save` earlier, so every set has state and a partially-written folder is skipped. `"second_last"` for density-only variants; `"first"` unused.
- `clip` / `ntransform` / `ndefault` — the variant's clip function; log-sm passes `np.log` and `0.0` since disk holds `n`. A 4-D `n` is clipped one Stokes row at a time, as the mean inside is a global sum. `big` adds a `list = True` argument to its `clip_zero`, which does that loop over the first axis itself; `apply_clip` calls the plain form row by row.
- `load_dealias` — `False` for big, `True` elsewhere.
- `fresh_particles` — `"qcriterion"` (big) or `"velocity"` (rndm).
- `init_name = None` for rndm, whose folders omit `_init_`. `particle_dir()` is the only place that name is built.
- `n = None` for rndm; every density branch checks it.

**Important — a restart must come from the last checkpoint where both the fields and the particles were written.** `big` saves fields (`Fields_k_*`, both spectra, `n_*`) every `dt_save_fields`, while `save()` writes particle state (`state_*`) on every call, so most `time_` folders hold particles alone and cannot be restarted from. `find_path` keeps only the folders holding a `Fields_k_*.npz` before it looks for particle directories, then steps back one `dt_save_fields`. Keep `dt_save_fields` an integer multiple of `dt_save_particles`: restarts are safe either way, but folder names carry only `tdec` decimals, so incommensurate intervals let a particle-only save round into a fields folder's name.

**Folders left by a run on different intervals are a trap.** `tdec` is 1 for `dt_save = 0.5` and for `min(1.0, 0.1)` alike, so nothing is renamed, and `new_dir.mkdir(exist_ok = True)` means a particle-only save at `96.5` drops fresh `state_*` into an old `time_96.5` still holding the previous run's `Fields_k_*` and `n_*`. `find_path` then sees a complete folder that mixes two trajectories. Clear `savePath` of folders off the new `dt_save_fields` grid before changing the intervals — and note `savePath` is relative, so it is the same tree as `loadPath` whenever the job is launched from the repository root.

`tdec` sets the decimals in `time_<t>` names and lives only in `big` and `rndm`, the two that create those folders; `big` takes it from `min(dt_save_fields, dt_save_particles)`, `rndm` from its single `dt_save`. `InitialConditions` keeps no copy — `find_path` parses the time out of the folder name and compares numbers with `np.isclose`, so what is written and what is searched for cannot drift apart. What `big` passes in the positional `dt_save` slot is `dt_save_fields`, since `find_path` steps back one fields interval. Every `plot-*.py` hard-codes `time_{t:.1f}`, so an interval below 0.1 renames every folder and the analysis scripts stop finding them.

`load_particles()` gives each rank the saved particles inside its own x-range, so the saved file count need not match the rank count. `load_num_slabs` is whatever the *writing* run used.

Two open items here. `find_path` raises a bare `IndexError` from its trailing `[0]` when a Stokes number has no saved particle directory: `tlast[jj]` stays at the 0 it was initialised with, `np.min` takes it, and no folder carries that time. And `place_particles` expands `np.where(mask[i-1])` and recomputes `mask[i-1].sum()` inside the `jj` loop, so four masks expand sixteen times at startup.

`restart.py` is superseded, imported by nothing, empty at HEAD.

## `particles.py`

`MPI_particles` holds position/velocity/mass, interpolates both ways with a cosine kernel (`interp_cosine`, `exterp_cosine_*`, `interp_exterp_cosine_*`), exchanges particles across slab boundaries (`particle_exchange`), and carries the particle RHS (`pRHS`, `pRHS_inertial`, `stoch_updt`). The kernels `_calc_usend_numba`, `_calc_uadd_numba_scalar/vector` are `@njit(parallel=True)` free functions — keep them numba-compatible.

Weights are `(1 + cos(pi*d/(2*dx)))/4` per direction, non-negative and summing to 1, so interpolation is a convex combination: a non-negative field cannot come back negative or overshoot the grid range.

`send` takes the stage coordinate as its first argument and the arrays that must follow it as the second. Only the first is wrapped with `%= L`, so `self.coord` and the RK4 accumulator can hold positions outside `[0,L)` between calls. Nothing reads them there — interpolation uses the wrapped first argument, routing goes through `sendbuf[:,0] % L`, and `save` wraps `stb.coord` by passing it in the first slot. Both arguments must carry the same number of rows, since one mask indexes all of them.

Two things to know: `update_intrinsic` does no MPI, whatever its docstring says; and `rb` omits `rho_p`, so it sits a factor `rho_p**(1/3)` above the radius convention of `factor` and of `rs` in the scripts. Only `stoch_updt` reads `rb`, so `rndm` alone is affected, and its collision area is 100 times too large at `rho_p = 1000`.

## Data layout

`data_cosine/forced_<True|False>/N_<N>_Re_<Re>/time_<t>/`:
- `Fields_k_<slab>.npz` — velocity shards, keys `uk`, `vk`, `wk`.
- `Energy_spectrum.npz`, `Flux_spectrum.npz`.
- Per-Stokes subdirectories with `n_<slab>.npz` (key `n`, shape `(Np,N,N)`, that slab's x-range) and/or `state_<rank>.npz` (`pos`, `vel`, `mass`, `prtclid`, `umat`, then `om2`, `s2`, `R`, `n` — the velocity-gradient invariants and the density read off at the droplet with the same cosine kernel). Folders from earlier runs carry the first five keys only, so one tree can hold both shapes.

**Two subdirectory orderings coexist in the same `time_` folder.** Density is `<wg>_sts_<sts>_stb_<stb>_init_<init>` (inline in `load_fields`); particle state is `<wg>_stb_<stb>_sts_<sts>_init_<init>` (`particle_dir`). Match the ordering the writing script used.

## Analysis scripts

`plot.py`, `plot-correlation.py` and the dated `plot-YYYYMMDD.py` files are standalone `#%%` cell scripts reading `data_cosine/...`, imported by nothing. New analysis normally goes in a new dated file.

`plot-20261005-mod.py` is the current one: it interpolates flow quantities onto the big particles and conditions droplet growth on them.

- `load_u` was filling one x-plane only, since `Np` is 1 when `num_process = 256`; `rank_data = range(0,N)` fixes it, and the same fix went into every `plot-*.py` that has a `load_u`. Any `urms`/`re_lmbd` recorded before that came from a near-empty field.
- `calc_invariants(u)` builds `A_ij = du_i/dx_j` spectrally and returns `om2`, `s2`, `R`. With `A = S + W`, `aa = tr(A A)` and `at = tr(A A^T)`, so `at - aa = omega^2` and `0.5*(aa + at) = S_ij S_ij` and neither `S` nor `W` is formed — that halves peak memory to about 2.6 GB.
- `interp_quantities` interpolates `om2, s2, R, n` at the droplets and stacks `Q, om2, s2, R, dissip, n, urel`. `Q = om2/4 - s2/2` and `dissip = 2*nu*s2` are linear in the interpolated fields, so they need no separate interpolation.
- `load_series` walks time on the outside, so the flow is computed once per snapshot and shared by all 20 `(stb, init)` pairs, and accumulates running histograms instead of holding per-particle arrays. `all` and `top10` come out of one pass.
- `MPI_particles` runs with `comm = MPI.COMM_WORLD` at size 1 (`plot-correlation.py` set that precedent). These scripts need a plain `python` process: `interpmat` is sized `Nprtcl//comm.size` while the loaders read the whole domain on every rank.
- json keys: `t`, `mean`, `std`, `n`, `nmean`, then `qmean_<q>`, `dmass_<q>`, `vals_<q>` per entry of `qnames`.

Known and unfixed there: `mask_max_growers` uses `PchipInterpolator`, which extrapolates when `cdf[0] > frac` and can silently make `top10` identical to `all`; the `dmass` fallback in `load_instant` is per-rank, so a partly-missing next folder pairs the wrong droplets; `Q_bins`/`R_bins` are wider than the fluctuations they resolve; `urel_bins` spans about 40x the real slip; `dissip` duplicates `s2` exactly.

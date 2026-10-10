# Single-GPU CUDA port of `forced-dns-sm-big.py`

Written 2026-10-10 to be executed in a fresh session. Self-contained: everything below was
established by reading the code, and nothing depends on the conversation that produced it.

---

## 1. What is being built, and why

A memory-efficient **fp64 single-GPU CUDA C** solver reproducing `forced-dns-sm-big.py` and its
dependence on `particles.py`. The bottleneck in the MPI code is the number-density fields: one
`full_RHS` call issues 24 velocity transforms plus 11 per Stokes number, so an RK4 step costs
880 density transforms against 96 for the velocity, and roughly 1216 collective calls.

The goal is **memory**, not arithmetic throughput. Double precision throughout. Do not propose
single precision; it has been considered and declined.

### Why CUDA C and not CuPy

1. The 4x memory lever — the compact dealiased spectrum, `build_line_offsets` (808) and
   `compact_point` (240) in `forced_ns_ck54_lowmem.cu` — is already written there, and its mask
   matches this problem's exactly. `forced_ns_ck54_lowmem.cu:191` sets
   `KTH2 = (ceil(sqrt(2)N/3) - 0.5)^2`; `forced-dns-sm-big.py:159` sets
   `dealias = kint <= 2**0.5*N/3`. Both reduce to `round(|k|) <= 120` at N = 256.
2. `cufftSetAutoAllocation(...,0)` with one work area shared between plans (`FFT1`, line 743)
   gives explicit control that CuPy's plan cache does not.
3. CuPy allocates temporaries implicitly — the wrong property when peak usage decides whether a
   run starts.

### Why classical RK4 and not CK54

RK4 holds four registers of the dominant array where CK54 holds two, so it halves the reachable
field count — 24 against 49 at N = 384. The twenty fields fit either way, so **RK4 costs nothing
the card could otherwise do.** Keep CK54 behind `-DUSE_CK54` as the one check that isolates the
velocity path (see C1).

---

## 2. Verification — the user's, by checkpoint comparison, after the code is complete

The user runs the current MPI code for one step at a chosen `dt_save`, loads the same checkpoint
into the GPU code, steps once, and compares.

**This happens after delivery, not during the port.** No agent waits on a command. The agents
write the whole of §5 and hand it over; the user compiles and compares later.

**That is the whole method.** No numpy oracle, no `--selftest`, no generated check scripts, no
assertion scaffold. This has been decided twice. If an agent proposes test machinery, that is a
finding, not a contribution.

Two consequences:

- **`unpack_checkpoint.py` is load-bearing.** `forced-dns-sm-big.py:539` writes
  `np.savez_compressed` — a ZIP of deflate-compressed `.npy` members. `forced_ns_ck54_lowmem.cu`
  reads raw `.npy` only; line 1174 tests the suffix. Without the converter the GPU code cannot
  read the files the comparison depends on. Do not link a zip library; a short Python converter
  run once per restart is correct.
- **Localisation is free.** On a mismatch, compare field by field in this order: `uk`, then each
  `n[jj]`, then the particle state. All three are in the checkpoint, so that separates the
  velocity path from the density path from the droplet path with no extra code.

---

## 3. The field count is a runtime input

The ensemble is the outer product of `stb_s` and `init_names`, **init varying slowest**:
`f = i_init*n_stb + i_stb`, `NFIELDS = n_stb*n_init`. This matches the two flat arrays at
`forced-dns-sm-big.py:41-43`, where `stb_s = [...4 values...]*5` and
`init_name = ['random']*4 + ['Qpos']*4 + ...`.

It must run from **1 field upward with no recompile**. Arrays sized from `NFIELDS` at startup,
every kernel taking it as an argument. A `#define NFIELDS 20` or any fixed-length array is a
failure. `Nprtcl` per field follows from that field's `stb` by `forced-dns-sm-big.py:48`.

### Memory, fp64, RK4 (four compact registers)

| N | per field | fixed (velocity + working) | 1 field | 20 fields | **max fields in 23 GiB** |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.221 GiB | 1.42 GiB | 1.6 GiB | 5.8 GiB | **97** |
| 384 | 0.744 GiB | 4.78 GiB | 5.5 GiB | 19.7 GiB | **24** |
| 512 | 1.762 GiB | 11.31 GiB | 13.1 GiB | 46.6 GiB | **6** |
| 640 | 3.439 GiB | 22.07 GiB | 25.5 GiB | — | **0** |

Target card: NVIDIA RTX PRO 4000, 24 GB. N = 512 is reachable for up to six fields; the full
twenty fit at N = 384. The compact fraction is `(4/3)pi(sqrt(2)/3)^3 = 0.4388`, so every
dealiased spectral array is 56% zeros and only arrays passing through an FFT need full size.

The code computes its footprint before allocating and refuses with the count that would fit,
rather than failing inside `cudaMalloc`.

---

## 4. The droplet units — corrected recently, easy to regress

`coord[:,-1]` is a **physical mass**, not a mass normalised by `M0`. `self.M0` is assigned at
`particles.py:71` and read nowhere. The chain:

- `n_cs_factor = M0/rhop` is the volume of one small droplet, so `c_s = n*n_cs_factor` and `n` is
  a number density with mean 1. (Check: mean `n = 1` gives volume fraction 7.2e-7 against the
  1e-6 documented at `forced-dns-sm-big.py:54`.)
- `growthfactor*m^(2/3)` reduces to `pi*rb**2`, so `exterpmat[:,0]` is the **number of small
  droplets swept per unit time**.
- `rhs[:,-1] = exterpmat[:,0]*rhop*n_cs_factor` is the mass growth rate.
- `fc` deposits `exterpmat/(dx*dy*dz)`, so `dn/dt = -fc` exactly, matching
  `forced-dns-sm-big.py:426`.
- `rb = (m*(0.75/pi)/rhop)**(1./3.)` — no `M0`, and `rhop` divides.

**A `*M0` or `/M0` appearing in any of these is a regression to a convention that was corrected.**

---

## 5. Files

Five new. Nothing existing is modified.

| file | owner |
| --- | --- |
| `droplet_kernels.cuh` | kernel-coder |
| `forced_ns_droplets.cu` | cuda-coder |
| `unpack_checkpoint.py` | cuda-coder |
| `tests/cpu_emu.h` | cuda-coder |
| `HANDOVER-GPU-PORT.md` | cuda-coder, from gpu-manager's notes |

`HANDOVER-GPU-PORT.md` is written last and is what the user reads when there is time to test: the
exact build commands for both builds, the exact run commands, the checkpoint comparison of §2 in
order, and a list of everything the agents could not check here.

Read-only references: `forced-dns-sm-big.py`, `particles.py`, `initial_conditions.py`,
`forced_ns_ck54_lowmem.cu`. **`initial_conditions.py` is not ported** — the user generates
initial conditions with it as it stands, and the new code only reads what that leaves on disk.

---

## 6. Tasks

| | what | budget |
| --- | --- | --- |
| G0 | Write `tests/cpu_emu.h` so the `-DCPU_EMU` build has something to compile against | +25 |
| G1 | `unpack_checkpoint.py` | +25 |
| G2 | **The contract**: `droplet_kernels.cuh` signatures, struct layouts, the memory map as a function of `NFIELDS`, the startup refusal | +70 |
| G3 | Scalar `k_scatter_s` / `k_gather_s` from 284 and 309 | +35 |
| G4 | Density state, compact spectral, 4 registers; `vk` from `u - tau_s Du/Dt` | +45 |
| G5 | The advection, inside the stage after `inverse_u()` at 855 | +60 |
| G6 | Positivity clip, global mean, on `block_add` (261) | +45 |
| G7 | Cosine weights replacing `cubic_w` (513) | +12 |
| G8 | Droplet state: mass, velocity, Stokes drag, advance | +70 |
| G9 | The deposition kernel, `atomicAdd`, 64 points per droplet | +55 |
| G10 | Classical RK4 beside the existing `rk_step`, switched at compile time | +50 |
| G11 | Per-Stokes output and restart reading | +80 |

**G0 is no longer a gate.** `tests/cpu_emu.h` is not in this repository, so the no-GPU build at
`forced_ns_ck54_lowmem.cu:71` has nothing to include. `cuda-coder` writes it from the uses in
`forced_ns_ck54_lowmem.cu` — `emu_launch`, the thread indices, `atomicAdd`, the memory and
stream calls — and the port continues whether or not it is right, since nothing here can compile
it. The handover records it as written but never compiled.

**G2 is the pivot.** Before it, serial; after it, kernel-coder and cuda-coder work in parallel
on separate files. Two writers in one file is how this goes wrong.

### Notes the coders need and will not find in the source

- **G7.** The swap is a drop-in. `k_part_interp` uses `ns = 4` and anchors at `-1` (line 540);
  the cosine kernel is also 4 points anchored at `-1` (`cosorder = np.arange(-1,3)`), weights
  `(1 + cos(pi*d/(2*dx)))/4`. Replace the weight body at 538-540; the triple product at 550-566
  is weight-agnostic.
- **G9, two traps.** The z-stride of a physical buffer is `ZP = 2*Nf`, **not** `N`, because it is
  an in-place R2C array — copy the indexing from `k_part_interp` line 556. And several droplets
  can land in one cell, which is why `particles.py:32,48` use a plain `range` for the droplet
  loop with `prange` only inside; on the GPU that needs `atomicAdd`.
- **What disappears from `particles.py`.** Most of it. `particle_exchange` (221, already dead),
  `send` (263) and every call opening `pRHS`/`pRHS_inertial`/`stoch_updt`, every
  `Neighbor_alltoall(v)` branch, `cart_comm`, `nneighbors`, `startdom`/`enddom`, the
  `outcond`/`cond` split, `uinterp`, `uinterp_cosine`, `interp_exterp_cosine_*`,
  `exterp_cosine_vector`, `Mmat`, `interporder`, and the module-level `RK4` at 955 that nothing
  calls. What survives collapses to its interior branch — the one that runs when the stencil is
  local, which on one GPU is always.

---

## 7. Control surface

- **C1.** Built `-DUSE_CK54` with zero density fields, reproduces `forced_ns_ck54_lowmem.cu` to
  round-off. `rk_stage` (858) and the CK54 coefficients (833) stay present and reachable.
- **C2.** The `-DCPU_EMU` build keeps working; every kernel runs as a serial loop under
  `emu_launch`. Nothing depending on `blockDim`, shared memory or warp behaviour.
- **C3.** fp64 default; `-DVOIGT_N` and `-DNS_DOUBLE` stay compile flags.
- **C4.** Physical constants stay a hand-edited block at the top (`sts`, `M0`, `rhop`, `nu0`,
  save intervals). No configuration file. Only the ensemble is on the command line.
- **C5.** The field count is a runtime input, working from 1 upward.
- **C6.** Reads and writes the existing layout so every `plot-*.py` keeps working — `time_<t>` to
  `tdec` decimals, and the two subdirectory spellings, which are **not** interchangeable:
  density `<wg>_sts_<sts>_stb_<stb>_init_<init>`, particles
  `<wg>_stb_<stb>_sts_<sts>_init_<init>`. The `plot-*.py` scripts hard-code `time_{t:.1f}`.
- **C7.** The four reference files are unchanged by the port, measured against the commit made
  before it began — **not HEAD**, since three of them carry uncommitted fixes (see §9).
- **C8.** Physics matches `forced-dns-sm-big.py`: two-pass phase-shift dealiasing, classical
  RK4, clip-and-rescale, droplet coupling, cosine weights. The units chain in §4 holds.

---

## 8. Agents

Already written in `.claude/agents/`. Invoke `gpu-manager` to drive; it spawns the rest.

| agent | model | effort | owns |
| --- | --- | --- | --- |
| gpu-manager | sonnet | medium | order, reviews, write lock. Writes no code |
| cuda-coder | opus | high | `forced_ns_droplets.cu`, the converter |
| kernel-coder | opus | xhigh | `droplet_kernels.cuh` — the race-prone half |
| memory-reviewer | opus | high | every allocation, lifetime, peak against §3 |
| kernel-reviewer | opus | max | races, atomics, the `ZP` stride, index arithmetic |
| equivalence-reviewer | opus | high | the numbers against the Python, units tracing |
| control-reviewer | sonnet | medium | C1–C8, nothing else |

`distiller` also exists, for the Python side only — it does not know this repository's CUDA
style and should not be pointed at the `.cu`.

### Frontmatter, verified against the docs and the official plugin agents on disk

`effort:` is the correct key, taking `low | medium | high | xhigh | max`. `model:` takes the bare
aliases `opus | sonnet | haiku | fable | inherit`. The subagent-spawning tool is `Agent`; `Task`
is the deprecated spelling. `color:` takes
`red | blue | green | yellow | purple | orange | pink | cyan`. Other keys exist and are unused
here: `disallowedTools`, `skills`, `mcpServers`, `permissionMode`, `maxTurns`, `isolation`,
`background`, `omitClaudeMd`, `memory`, `hooks`.

**One untested detail.** `gpu-manager` restricts its spawn list with
`Agent(cuda-coder, kernel-coder, ...)`, following the official agents, which use the same form
with a plugin prefix (`Agent(claude-security:explore)`). The bare-name form for project-local
agents could not be tested here. **If the manager cannot spawn anything on the first run, change
that line to plain `Agent`** — the restriction is defence in depth, not load-bearing, since the
manager's own instructions already name which agent reviews which task.

The two highest settings sit where a mistake is invisible rather than loud: `kernel-coder` writes
the deposition kernel, the one piece with a genuine race, and `kernel-reviewer` exists only to
find errors that produce plausible numbers.

Nothing compiles or runs in the sandbox — no `nvcc`, no `clang++`, no numpy. Review is the only
verification before the code reaches the cluster, which is why the reviewers are narrow and
adversarial. The reviewers stand in for the gates: a task is finished when its reviewer has no
remaining finding, and the run carries on to the next task without stopping.

---

## 9. The C7 baseline, and what is in it

**The baseline is commit `e076637` ("plan for agent codes."), reachable as `HEAD` when the port
begins.** C7 is checked with `git diff e076637 -- <file>`; anything non-empty there is the
port's doing. The working tree was clean at the time of writing apart from
`.claude/settings.local.json`, which is machine-local and deliberately untracked.

That commit carries fixes made shortly before this plan. They are **not** the port's work and
must not be reverted:

- `particles.py` — mass convention: `n_cs_factor` moved out of `growthfactor` into the mass
  equation; `decelerationfactor` derived from `growthfactor`; `rb` corrected at lines 218 and
  937; the `/M0` dropped at 949; three comments.
- `initial_conditions.py` — `n.ndim > 1` to `n.ndim > 3` at lines 114, 121 and 172, plus line
  112 moved from `n.shape[0] != self.Np` to the same spelling. `n` is 4-D for the particle
  variants and 3-D for the density-only ones; `> 1` was true for both.
- `forced-dns-sm-big.py` — one comment at line 426.

These are verified by hand and by review but **not yet verified on the cluster**, so they are a
known open risk sitting underneath this port rather than settled ground. Outstanding there: `rb0s` at
`forced-dns-sm-big-rndm.py:179` should drop by `(M0*rhop)**(1/3) = 4.16`; `big` should be
unchanged to round-off; and the 20 density fields, which differ only through `fc`, should
separate after the fix where they were near-identical before. `rndm` results produced before the
fix used a collision area roughly 10x to 15x too large.

---

## 10. Related, not part of this work

`plan-mpi-batching.tex` is a separate plan for the cluster code: batching the Stokes axis through
the existing MPI transforms, taking collective calls per step from ~1216 to ~152. Independent of
this port and can proceed in parallel. Its agents are **not** written.

---

## 11. Launch prompt for a fresh session

```
Read PLAN-GPU-PORT.md in full, then drive it using the gpu-manager agent.

These decisions are closed. Do not re-open them or propose alternatives:
  - CUDA C, extending forced_ns_ck54_lowmem.cu. Not CuPy.
  - fp64 throughout. Single precision was considered and declined.
  - Classical RK4. CK54 stays only behind -DUSE_CK54, as the velocity-path check.
  - Verification is mine, by checkpoint comparison against the MPI code.
    No oracle, no --selftest, no generated check scripts, no assertion scaffold.
  - initial_conditions.py is not ported. The new code only reads what it writes.

Do not stop to ask me to run anything. I am not available to test during this
run. Go from G0 to G11 without pausing, and hand me the complete code at the
end. tests/cpu_emu.h is missing: write it from how forced_ns_ck54_lowmem.cu
uses it and carry on.

The C7 baseline is already committed as e076637, reachable as HEAD. Check the
port against it with: git diff e076637 -- <file>

Nothing compiles or runs in this sandbox — no nvcc, no clang++, no numpy. The
reviewers are the only check available, so a task is finished when its reviewer
has no finding left, and you move straight to the next one.

Finish by writing HANDOVER-GPU-PORT.md: every command I will need, in order,
and everything you could not check here. Then report.
```

`gpu-manager` runs straight through. Nothing in the sequence waits on the user; whatever cannot
be settled by review is recorded in `HANDOVER-GPU-PORT.md` and tested on the cluster later.

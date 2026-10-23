# %% [markdown]
# # Forced Navier-Stokes + odd viscosity + Lagrangian particles — LOW-MEMORY solver, classical RK4
# 
# Colab runner for **`forced_ns_rk4_lowmem.cu`**: classical RK4 in three compact registers
# (uⁿ, running sum, stage; the stage register also receives the stage's right-hand side once the
# stage field has gone to physical space). **~72 GiB at 1024³ fp64** including the cuFFT work
# area. One evaluation of the nonlinear term per stage on a grid offset by $\theta_i+\varphi_n$
# (offsets chosen so that $\sum_i b_i e^{2\pi i\theta_i}=0$ for $b = (1,2,2,1)/6$, plus a random
# offset per step): 32 FFTs per step, against 40 for the CK54 shift solver and 80 for the original.
# `INTEGRATOR = "ck54"` runs the CK54 shift solver unchanged; `DEALIAS = "exact"` the two-grid scheme.
# `INIT_FILE` starts a new run from a field of **any** resolution, e.g. a 768³ spin-up padded to
# 1024³ (see "Starting from a field of another resolution" below the run cells).
# 
# **Measured** (CPU emulator, N = 64 forced turbulence, $k_{\max}\eta = 1.07$, one large-eddy time,
# error against a tiny-dt reference; CFL = u_rms dt/dx):
# 
# | | velocity | gradients | shell at $k_{\max}$ | time per step | largest stable CFL |
# |---|---|---|---|---|---|
# | CK54 exact (original lowmem) | 6.5e-5 | 3.2e-4 | 1.2e-3 | 143 ms | 0.38 |
# | CK54 + shifts | 5.7e-4 | 2.8e-3 | 1.7e-2 | 74 ms | 0.34 |
# | **RK4 + shifts (this file)** | **4.5e-4** | **2.0e-3** | **1.1e-2** | **60 ms** | **0.32** |
# | N = 64 vs N = 96 (resolution error) | 2.0e-2 | 9.5e-2 | | | |
# 
# (errors at CFL 0.17). Long-time statistics agree with the exact scheme within the statistical
# noise (last shell up to 2% high); particle accuracy is set by the cubic interpolation and is
# unchanged. Keep CFL below ~0.3 (the dt advisor below uses the RK4 stability polynomial).
# Cell 9b repeats the error measurement on the GPU from your own saved state.
# 
# Base solver: **`forced_ns_ck54_lowmem.cu`** — flow + particles only, laid out so that
# **N = 1024 in fp64 fits on one 80 GB GPU** (~53 GiB + the cuFFT work area).
# 
# Same physics, integrator, forcing, dealiasing, particles, output files and restart
# files as `forced_ns_ck54_part.cu`; it stores u and q as only the modes the dealias
# sphere keeps, and forms the nonlinear term as five products (Basdevant 1983)
# in one reusable buffer. There is no passive-scalar field in this build, so
# `SCALAR` must stay `False`. Validation: `tests/run_checks.sh`, and on the GPU
# `tests/compare_runs.py` against the old solver at N = 384.
# 
# $$D_t u = -\nabla P + \nu\nabla^2u + \nu_{\rm odd}(\hat e_a\times\nabla^2u) + f,
# \qquad \nabla\cdot u = 0$$
# $$D_t\theta = \kappa\nabla^2\theta - G\,u_g + f_\theta
# \qquad\qquad \frac{dX_p}{dt} = u(X_p,t)$$
# 
# ## How to run
# 
# 1. **Runtime > Change runtime type > GPU** (A100 or better for N=768).
# 2. Run cell 1 and upload `forced_ns_rk4_lowmem.cu` when prompted.
# 3. Edit **INPUT**, then **Runtime > Run all**.
# 
# Cells 1-9 are the run. Cell 10 (`Analysis helpers`) loads every output file and
# **must be run before any plotting cell** — it defines `t, E, Z, vf, pf, ...`
# that all the plots use. If a plot cell says `NameError`, you skipped cell 10.
# 
# ## The two switches
# 
# | `SCALAR` | `PARTICLES` | what you get |
# |---|---|---|
# | `True`  | `False` | Eulerian tracer field $\theta$ only (the original behaviour) |
# | `False` | `True`  | Lagrangian particles only |
# | `True`  | `True`  | both, sharing the same transforms |
# | `False` | `False` | plain forced Navier-Stokes |
# 
# Turning `SCALAR` off frees three component arrays (`d_AW` drops from 7 slots to
# 6), so `False/True` at N=768 fp32 needs **20.0 GiB** instead of 25.4.
# 
# **Particles** are advected by the same real-space velocity the momentum RHS
# already builds in pass 0 of each RK stage, so they need *no extra transform* —
# one 64-point cubic-Lagrange interpolation per particle per stage. They use the
# identical CK54 2N recursion as the fields, so they are fourth order in time too.
# Positions are unwrapped fp64, which makes dispersion exact and trajectories
# continuous across the box edges.
# 
# **Built-in check:** an incompressible flow keeps fluid tracers uniformly
# distributed, so $\langle|u_p|^2\rangle$ (particles) must equal $2E$ (grid). The
# solver prints the gap every save.

# %%
#@title 1. Upload / locate the source
import os, glob, subprocess, math, time, sys
import numpy as np

SRC = "./forced_ns_rk4_lowmem.cu"
if not os.path.exists(SRC):
    hits = glob.glob("./**/forced_ns_rk4*lowmem*.cu", recursive=True)
    if hits:
        SRC = hits[0]
    else:
        from google.colab import files
        up = files.upload()                      # pick forced_ns_rk4_lowmem.cu
        name = list(up)[0]
        if name != os.path.basename(SRC):
            os.rename("./" + name, SRC)
print("source:", SRC, os.path.getsize(SRC), "bytes")

# %%
# =========================  INPUT  =========================================
# Edit this cell, then Runtime > Run all.
####################################################################################

N             = 400         # COMPILE-TIME: changing this recompiles (~1 min)
PRECISION     = "fp64"       # "fp32" or "fp64"   (also compile-time)
INTEGRATOR    = "rk4"        # "rk4" (default: classical RK4, 3 registers, ~72 GiB at 1024^3)
                             # "ck54" (2 registers, ~62 GiB) or "rk2" (validation)
DEALIAS       = "shift"      # "shift": one grid per stage (fast, default)
                             # "exact": two grids per stage (= the lowmem solver)
                             # "none" : one plain grid, aliased -- diagnostics only
SHIFT_SEED    = 20260930     # seed of the per-step random grid offset

# --- flow -------------------------------------------------------------------
NU            = None         # None -> nu0*(3m/(sqrt(2)N))^(2(lp-1/3))
DT            = 2e-3         # see the dt advisor cell below
TFINAL        = 40.0
SAVE_INTERVAL = 0.4
DUMP_EVERY    = 25           # field dumps are 25.8 GB EACH at N=1024 fp64. Check the
                             # disk estimate printed below before lowering this.
EPSILON       = 0.75         # energy injection rate
FORCED_SHELL  = 2
EINIT         = 1.0

# --- odd viscosity ----------------------------------------------------------
NU_ODD        = 0.0          # 0 disables. Try ~5*nu first.
ODD_AXIS      = 2            # 0=x, 1=y, 2=z

# --- EULERIAN passive tracer field  theta(x,t) ------------------------------
SCALAR        = False        # must stay False: no tracer FIELD in the low-memory build
SC            = 1.0          # Schmidt number, kappa = nu/Sc
GRAD          = 1.0          # mean gradient G  (0 -> no gradient source)
GRAD_DIR      = 0            # 0=x, 1=y, 2=z
EPS_THETA     = 0.0          # tracer variance injection rate (0 disables)
SHELL_THETA   = None         # None -> same shell as the velocity
THETA_INIT    = None         # None -> zero if GRAD!=0 else random.  Forced to
                             # random automatically when EPS_THETA>0, because
                             # shell forcing cannot amplify theta == 0.

# --- LAGRANGIAN tracer particles --------------------------------------------
PARTICLES     = True         # <<< master switch for the PARTICLES (independent)
N_PART        = int(N**3/2)       # 1e5-1e6 is cheap; cost is ~0 next to the FFTs
PART_START    = 20.0         # seed at this time -- skips the spin-up transient
                             # so dispersion / autocorrelation mean something
PART_INIT     = "random"     # "random" (uniform) or "lattice" (regular cube)
PART_INTERP   = "cubic"      # "cubic" (4th order) or "linear" (2nd, validation)
PART_DUMP_EVERY = 1          # particle dumps are small: every save is fine
PART_SEED     = 20240824

# --- RESTART from a previous run --------------------------------------------
RESTART       = None         # None = fresh random IC.
                             # Set to a previous output directory, e.g.
                             #   "./Forced_NS_N768"
                             # to continue from the newest save in it.
                             # TFINAL is then the ABSOLUTE end time, not extra
                             # time: restarting at t=40 with TFINAL=40 does
                             # nothing, you want TFINAL=80.
RESTART_COUNTER = None       # None = newest save; or an int to pick one

# --- NEW run from a field of ANOTHER resolution (e.g. spin up at 768^3) -------
INIT_FILE     = None         # None = random IC (or RESTART). Or a spectral field
                             # (3,n,n,n/2+1) of ANY n -- e.g. a dump of a 768^3 run:
                             #   "./drive/MyDrive/N768/Velocity_Data_100.npy"
                             # zero-padded (n < N) or truncated (n > N) to N.
                             # A field ON THE GRID: convert it with make_ic.py.
                             # New run: t and counters start at 0, EINIT is ignored.
NU_OF_N       = None         # with NU = None: take the default nu of THIS N instead
                             # of N's. For a 768^3 spin-up that will be padded to
                             # 1024^3 set NU_OF_N = 1024, so nu does not change at the
                             # switch (that costs ~2 large-eddy times, see below).

# --- misc -------------------------------------------------------------------
OUTDIR        = None         # None -> ./Forced_NS_N<N>
FAST_MATH     = True
REAL_IC       = False        # True reproduces the old notebook's real-valued IC

####################################################################################
NU_EFF = NU if NU is not None else 1.0*(3*1/(math.sqrt(2)*(NU_OF_N or N)))**(2*(1-1/3))
KAPPA  = NU_EFF / SC
OUT    = OUTDIR or ("./Forced_NS_N%d" % N)

def is_smooth(n):
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1

Nf      = N//2 + 1
bytes_c = 16 if PRECISION == "fp64" else 8
if DEALIAS == "shift" and INTEGRATOR == "rk2":
    raise ValueError("DEALIAS = \"shift\" needs INTEGRATOR = \"rk4\" or \"ck54\"")
if SCALAR:
    raise ValueError("forced_ns_rk4_lowmem.cu has no scalar field: set SCALAR = False "
                     "(or use run_ck54_solver_colab_part.ipynb)")
# kept modes per component: the dealias sphere, exactly as the solver counts them
_k = np.fft.fftfreq(N, 1.0/N); _k[N//2] = N//2
_KC = int(1.4142135623730951*N/3.0) + 1; _KTH2 = (_KC - 0.5)**2
_rem = _KTH2 - (_k[:, None]**2 + _k[None, :]**2)
M_KEPT  = int(np.minimum(np.where(_rem > 0, np.floor(np.sqrt(np.maximum(_rem, 0))) + 1, 0), Nf).sum())
NREG    = 3 if INTEGRATOR == "rk4" else 2             # compact vector registers
mem     = (4*N*N*Nf + 3*NREG*M_KEPT) * bytes_c / 2**30  # 4 full slots + compact registers
pmem    = ((18 if NREG == 3 else 15) * N_PART * 8 / 2**30) if PARTICLES else 0.0
kmax    = math.sqrt(2)*N/3
eta     = (NU_EFF**3/EPSILON)**0.25
etaB    = eta/math.sqrt(SC)
dx      = 2*math.pi/N

nsave  = int(TFINAL/DT)//max(1, int(SAVE_INTERVAL/DT)) + 1
fdumps = (nsave + DUMP_EVERY - 1)//DUMP_EVERY if DUMP_EVERY > 0 else 0
pdumps = (nsave + PART_DUMP_EVERY - 1)//PART_DUMP_EVERY if PARTICLES else 0
disk   = fdumps * (3 + (1 if SCALAR else 0)) * N*N*Nf * bytes_c / 1e9
disk  += pdumps * 6 * N_PART * 8 / 1e9

print("N = %d  %s  %s  dealias %s   nu = %.6e%s" %
      (N, PRECISION, INTEGRATOR, DEALIAS, NU_EFF, "  (user-set)" if NU is not None else
       ("  (default of N=%d)" % NU_OF_N if NU_OF_N else "")))
print("tracer field  : %-5s        particles: %s" %
      ("ON" if SCALAR else "OFF", ("ON, %d" % N_PART) if PARTICLES else "OFF"))
print("field memory  ~ %.1f GiB   (4 full slots + %d compact, %.1f%% kept; old layout %.1f GiB)%s" %
      (mem, 3*NREG, 100.0*M_KEPT/(N*N*Nf), 12*N*N*Nf*bytes_c/2**30,
       ("  + %.3f GiB particles" % pmem) if PARTICLES else ""))
print("kmax*eta      = %.3f      (velocity; want >~ 1)" % (kmax*eta))
if NU is None and NU_OF_N and NU_OF_N > N:
    print("                (the N=%d viscosity: under-resolved on purpose for a spin-up that"
          " will be padded to %d^3 with INIT_FILE)" % (NU_OF_N, NU_OF_N))
if SCALAR:
    print("kmax*eta_B    = %.3f      (tracer, Sc = %g)" % (kmax*etaB, SC))
    if kmax*etaB < 1.0:
        print("   *** tracer under-resolved: raise N or lower SC ***")
if NU_ODD > 0:
    k_odd = EPSILON**0.25 * NU_ODD**-0.75
    k_c   = EPSILON**0.25 * NU_EFF**-0.5 * NU_ODD**-0.25
    print("nu_odd/nu     = %.1f   k_odd = %.1f   k_c = %.1f   k_max = %.0f" %
          (NU_ODD/NU_EFF, k_odd, k_c, kmax))
    if k_c > kmax:
        print("   *** condensation peak k_c unresolved: raise N ***")
    if k_odd < 2:
        print("   *** odd effects reach the forcing scale: lower NU_ODD ***")
print("steps         = %d   diagnostic saves = %d" % (int(TFINAL/DT)+1, nsave))
print("DISK          ~ %.1f GB of dumps (%d field, %d particle)" %
      (disk, fdumps, pdumps))
if disk > 50:
    print("   *** will not fit a stock Colab disk (~100 GB): raise DUMP_EVERY ***")
if PARTICLES and PART_START <= 0:
    print("   note: PART_START=0 seeds inside the spin-up transient; dispersion")
    print("         and autocorrelation will not be interpretable.")
if not is_smooth(N):
    print("\n   *** N = %d is not 2,3,5,7-smooth -- cuFFT will be slow ***" % N)
print("\noutput directory:", OUT)
if RESTART:
    print("RESTART       : from %s%s" %
          (RESTART, "" if RESTART_COUNTER is None else (" save #%d" % RESTART_COUNTER)))
    print("                TFINAL = %g is the ABSOLUTE end time." % TFINAL)
    print("                EINIT is ignored; the loaded field sets the energy.")

# ---- INIT_FILE: check it now, not after the build
if INIT_FILE and RESTART:
    raise ValueError("set RESTART (continue a run) or INIT_FILE (new run from a field), not both")
if INIT_FILE:
    import re
    if not os.path.isfile(INIT_FILE):
        raise FileNotFoundError("INIT_FILE = %r does not exist (Drive mounted? cell 19 shows how)"
                                % INIT_FILE)
    _a = np.load(INIT_FILE, mmap_mode="r")             # reads the header only
    if _a.ndim != 4 or _a.shape[0] != 3 or _a.shape[2] != _a.shape[1] or \
       _a.shape[3] != _a.shape[1]//2 + 1 or _a.dtype.kind != "c":
        raise ValueError("INIT_FILE has shape %s, dtype %s: need a spectral field (3,n,n,n/2+1) "
                         "complex. A field on the grid: python3 make_ic.py OUT.npy %s"
                         % (_a.shape, _a.dtype, INIT_FILE))
    n_in = _a.shape[1]; del _a
    print("INIT_FILE     : %s" % INIT_FILE)
    print("                n = %d -> N = %d: %s" % (n_in, N, "zero-padded" if n_in < N else
          ("TRUNCATED (energy beyond the new sphere is lost)" if n_in > N else "same size")))
    _m = re.search(r"Velocity_Data_(\d+)\.npy$", INIT_FILE)
    _ri = os.path.join(os.path.dirname(INIT_FILE), "Restart_Info_%s.txt" % _m.group(1)) if _m else ""
    nu_src = None
    if _ri and os.path.exists(_ri):
        for _l in open(_ri):
            _p = _l.split()
            if len(_p) == 2 and _p[0] == "nu":
                nu_src = float(_p[1])
    tau_eta = (NU_EFF/EPSILON)**0.5
    if nu_src is not None:
        print("                written with nu = %.6e   this run nu = %.6e" % (nu_src, NU_EFF))
    if nu_src is not None and abs(nu_src/NU_EFF - 1) > 0.01:
        print("   *** nu changes by x%.2f at the switch. The cascade has to re-form: measured"
              % (NU_EFF/nu_src))
        print("       (48^3 -> 64^3, nu x1/1.47) ~4 time units = 2 E/eps before the statistics")
        print("       are native. With the same nu (NU_OF_N in the spin-up) it is ~5 tau_eta. ***")
    elif n_in < N:
        print("                transient: discard the first ~10 tau_eta = %.3f time units" % (10*tau_eta))
        print("                (measured for n/N = 3/4: native within the noise after ~5 tau_eta;")
        print("                 for n/N = 1/2 it took ~30 tau_eta)")
    if PARTICLES and PART_START > 20*tau_eta:
        print("                the field is developed already: PART_START = %.2f (~20 tau_eta) is"
              " enough" % (20*tau_eta))
    print("                t starts at 0, so TFINAL is the duration of this run")



# %%
#@title 3. dt advisor — largest linearly stable timestep for THESE parameters
# Scans z = dt(-nu k^2 + i(k U + nu_odd k^2)) over all k with the stability
# polynomial of the selected scheme.
#
# DEALIAS = "shift" with CK54: the measured largest stable dt was 5-10% below the
# exact scheme's, so keep a 10% margin. With RK4 the shifts cost no stability.
#
# The particles do NOT enter this limit: they are slaved to the field and use
# the same scheme, so they are stable whenever it is.
#
# This scans with a chosen u. The advective eigenvalue is really set by max|u|,
# which in forced turbulence is 3-4x u_rms, so several u are shown. Judge your
# DT against the max|u| column, not the rms one.
from fractions import Fraction as F

CK_A = [F(0), F(-567301805773,1357537059087), F(-2404267990393,2016746695238),
        F(-3550918686646,2091501179385), F(-1275806237668,842570457699)]
CK_B = [F(1432997174477,9575080441755), F(5161836677717,13612068292357),
        F(1720146321549,2090206949498), F(3134564353537,4481467310338),
        F(2277821191437,14882151754819)]
R2_A = [F(0), F(-5857864376269050,10000000000000000)]
R2_B = [F(7071067811865475,10000000000000000)]*2

def stab_poly(A, B):
    q, u = [F(0)], [F(1)]
    def padd(a, b):
        n = max(len(a), len(b))
        a = a + [F(0)]*(n-len(a))
        b = b + [F(0)]*(n-len(b))
        return [x+y for x, y in zip(a, b)]
    for i in range(len(A)):
        q = padd([A[i]*x for x in q], [F(0)] + u)
        u = padd(u, [B[i]*x for x in q])
    return [float(x) for x in u]

POLY = {"ck54": stab_poly(CK_A, CK_B), "rk2": stab_poly(R2_A, R2_B),
        "rk4":  [1, 1, .5, 1/6, 1/24]}

def dt_max(scheme, nuodd, U):
    c = POLY[scheme]
    k = np.linspace(1, kmax, 4000)
    lo, hi = 1e-10, 1.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        z = mid*(-NU_EFF*k**2 + 1j*(k*U + nuodd*k**2))
        R = sum(cj*z**j for j, cj in enumerate(c))
        lo, hi = (mid, hi) if np.all(np.abs(R) <= 1+1e-12) else (lo, mid)
    return lo

US = (1.0, 1.5, 3.0, 4.5)
print("%6s %7s | %s" % ("scheme", "stages",
                        "  ".join("u=%-4g dt_max" % u for u in US)))
for sch, st_ in (("rk2", 2), ("rk4", 4), ("ck54", 5)):
    row = "  ".join("%10.3e" % dt_max(sch, NU_ODD, u) for u in US)
    print("%6s %7d |  %s%s" % (sch, st_, row,
          "  <-- selected" if sch == INTEGRATOR else ""))

d_rms, d_max = dt_max(INTEGRATOR, NU_ODD, 1.5), dt_max(INTEGRATOR, NU_ODD, 4.5)
print("\nyour DT = %.3e" % DT)
print("   %5.0f%% of the limit at u = 1.5 (rms)" % (DT/d_rms*100))
print("   %5.0f%% of the limit at u = 4.5 (a realistic max|u|)  <-- judge on this"
      % (DT/d_max*100))
if DT > d_max:
    print("   *** UNSTABLE at realistic max|u| ***")
elif DT > d_max/2:
    print("   *** thin margin. If E(k) turns up at high k, halve DT ***")
else:
    print("   margin looks reasonable")

# %%
#@title 4. GPU check + fit verdict
q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader"], capture_output=True, text=True)
if q.returncode != 0 or not q.stdout.strip():
    raise RuntimeError("no GPU found. Runtime > Change runtime type > GPU.")
print(q.stdout.strip())
name, memtot, cc = [x.strip() for x in q.stdout.strip().split(",")]
GPU_MEM = float(memtot.split()[0]) / 1024
ARCH    = "sm_" + cc.replace(".", "")
need    = mem + pmem
print("\narch = %s   device memory = %.1f GiB   solver ~ %.1f GiB" %
      (ARCH, GPU_MEM, need))
print("  NOTE: the cuFFT plan workspace is ON TOP of this and is not small at")
print("        N >= 768. The solver prints the real number at startup as")
print("        'cuFFT work area (batch 1): X GiB' -- check it against the headroom here.")
head = GPU_MEM - need
if head < 0.20*GPU_MEM:
    print("\n  *** only %.1f GiB headroom -- cuFFT plan creation will likely fail"
          % head)
    print("      once the workspace is added. Reduce N or use fp32. Cell 7's")
    print("      --plan-only prints the exact total including the work area. ***")
else:
    print("  fits, headroom %.1f GiB before the cuFFT workspace" % head)

# %%
#@title 5. BUILD  (re-run only when N, PRECISION, FAST_MATH or REAL_IC change)
CUDA = "/usr/local/cuda-13.1"
LIB  = CUDA + "/targets/x86_64-linux/lib"
BIN  = os.path.abspath("./forced_ns")

flags = ["-O3", "-std=c++17", "-arch=" + ARCH, "-DVOIGT_N=%d" % N]
if PRECISION == "fp64":
    flags.append("-DNS_DOUBLE")
if REAL_IC:
    flags.append("-DNS_IC_REAL_PHASE")
if FAST_MATH:
    flags.append("-use_fast_math")

cmd = ([CUDA + "/bin/nvcc"] + flags + [SRC, "-o", BIN,
       "-L" + LIB, "-Xlinker", "-rpath=" + LIB, "-lcufft", "-lcurand"])
print(" ".join(cmd), "\n")
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-4000:])
print("\nreturn code %d   (%.0f s)" % (r.returncode, time.time()-t0))
assert r.returncode == 0, "compilation failed"

# %%
#@title 6. Argument list + streaming runner
def solver_args(extra=(), ic=True):
    a = ["./forced_ns",
         "--integrator",    INTEGRATOR,
         "--dealias",       DEALIAS,
         "--shift-seed",    str(SHIFT_SEED),
         "--nu",            repr(NU_EFF),
         "--dt",            repr(DT),
         "--tfinal",        repr(TFINAL),
         "--save-interval", repr(SAVE_INTERVAL),
         "--eps",           repr(EPSILON),
         "--shell",         str(FORCED_SHELL),
         "--einit",         repr(EINIT),
         "--dump-every",    str(DUMP_EVERY),
         "--out",           OUT]
    if NU_ODD > 0:
        a += ["--nu-odd", repr(NU_ODD), "--odd-axis", str(ODD_AXIS)]

    # --- Eulerian tracer field ----------------------------------------------
    if SCALAR:
        a += ["--sc", repr(SC), "--grad", repr(GRAD), "--grad-dir", str(GRAD_DIR)]
        if EPS_THETA > 0:
            a += ["--eps-theta", repr(EPS_THETA)]
        if SHELL_THETA is not None:
            a += ["--shell-theta", str(SHELL_THETA)]
        if THETA_INIT:
            a += ["--theta-init", THETA_INIT]
    else:
        a += ["--no-scalar"]

    # --- restart --------------------------------------------------------------
    if ic and INIT_FILE:
        a += ["--init", INIT_FILE]
    if ic and RESTART:
        a += ["--restart", RESTART]
        if RESTART_COUNTER is not None:
            a += ["--restart-counter", str(RESTART_COUNTER)]

    # --- Lagrangian particles ------------------------------------------------
    if PARTICLES and N_PART > 0:
        a += ["--particles",       str(N_PART),
              "--part-start",      repr(float(PART_START)),
              "--part-init",       PART_INIT,
              "--part-interp",     PART_INTERP,
              "--part-dump-every", str(PART_DUMP_EVERY),
              "--part-seed",       str(PART_SEED)]
    return a + list(extra)

def run(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    print("\nreturn code", p.returncode)
    return p.returncode

args = solver_args()
print(" ".join(args))

# %%
#@title 7. Memory plan (exact) + self-tests
# Allocates every buffer and builds all six plans -- the fastest way to find out
# whether your N actually fits, before committing to a long run.
# --plan-only: exact memory including the real cuFFT work area, allocates nothing
if run(solver_args(["--plan-only"])) == 0:
    run(solver_args(["--verify"]))   # FFT + compact round trips + frozen-field alias cancellation

# %%
#@title 8. Throughput check (no I/O). Multiply by the step count before committing.
# Particles are seeded before the timing loop, so this number includes them.
# Runs from the random field (ic=False): the cost per step does not depend on the
# field, and this way a RESTART / INIT_FILE of 10-25 GB is not read three times.
print("=== %s, --dealias %s" % (INTEGRATOR, DEALIAS))
run(solver_args(["--bench", "20"], ic=False))
print("\n=== CK54 + shifts (forced_ns_ck54_lowmem_shift.cu), for comparison")
run(solver_args(["--bench", "20", "--integrator", "ck54", "--dealias", "shift"], ic=False))
print("\n=== CK54, two grids per stage (the original lowmem solver), for comparison")
run(solver_args(["--bench", "20", "--integrator", "ck54", "--dealias", "exact"], ic=False))

# %%
#@title 9. RUN
os.makedirs(OUT, exist_ok=True)
run(args)

# %% [markdown]
# ---
# ## 9b. Optional: how much extra error on YOUR flow?
# 
# Restarts from one of your saved fields three times, for `ACC_T` time units each, with no particles:
# 
# * reference: `--dealias exact` at `DT/4`
# * `--dealias exact` at `DT`: the CK54 time-stepping error
# * `--dealias shift` at `DT`: the time-stepping error **plus** the aliasing left over
# 
# It then prints the errors relative to the reference (energy norm, gradient norm, and per shell).
# **Cost:** about (4 × 2 + 2 + 1) × ACC_T/DT single-grid step equivalents; at 512³ on an A100 with
# `DT = 2e-3` and `ACC_T = 0.25`, roughly 8 min. **Disk:** 3 field dumps (the start dumps are deleted),
# each 3·N²·(N/2+1)·16 bytes in fp64 (3.2 GB at 512³, 25.8 GB at 1024³).
# 

# %%
#@title 9b. Accuracy check: shift vs exact from your own saved state
ACC_FROM    = RESTART or OUT      # a directory holding Velocity_Data_<c>.npy + Restart_Info_<c>.txt
ACC_COUNTER = None                # None = newest dump there
ACC_T       = 0.25                # time units per run (~a quarter large-eddy time is plenty)
ACC_DIR     = "./accuracy_check"

import shutil, re
def acc_args(dealias, dt, out):
    a = ["./forced_ns", "--integrator", INTEGRATOR, "--dealias", dealias,
         "--shift-seed", str(SHIFT_SEED), "--nu", repr(NU_EFF), "--eps", repr(EPSILON),
         "--shell", str(FORCED_SHELL), "--dt", repr(dt), "--save-interval", repr(t_si),
         "--dump-every", "1", "--tfinal", repr(t_end), "--restart", ACC_FROM, "--out", out,
         "--no-scalar"]
    if NU_ODD > 0:
        a += ["--nu-odd", repr(NU_ODD), "--odd-axis", str(ODD_AXIS)]
    if ACC_COUNTER is not None:
        a += ["--restart-counter", str(ACC_COUNTER)]
    return a

# ---- find the start field, with a clear message if there is none
if not ACC_FROM or not os.path.isdir(ACC_FROM):
    raise FileNotFoundError(
        "ACC_FROM = %r does not exist. Run cell 9 (RUN) first in this session, or set "
        "ACC_FROM to a folder holding Velocity_Data_<c>.npy (e.g. an earlier run copied "
        "to Drive). Note: a Colab runtime reset wipes .." % ACC_FROM)
dumps = sorted(int(m.group(1)) for m in
               (re.search(r"Velocity_Data_(\d+)\.npy$", f)
                for f in glob.glob(os.path.join(ACC_FROM, "Velocity_Data_*.npy"))) if m)
if not dumps:
    have = sorted(os.listdir(ACC_FROM))
    raise FileNotFoundError(
        "no Velocity_Data_<c>.npy in %s (it holds %d files%s). Field dumps are written "
        "every DUMP_EVERY-th save (DUMP_EVERY = %s, SAVE_INTERVAL = %s: one dump every "
        "%g time units, the first at the start of the run). Run cell 9 until a dump from "
        "the developed turbulence exists, or point ACC_FROM at a folder that has one."
        % (ACC_FROM, len(have), (": " + ", ".join(have[:6]) + (" ..." if len(have) > 6 else ""))
           if have else "", DUMP_EVERY, SAVE_INTERVAL, DUMP_EVERY * SAVE_INTERVAL))
c0 = dumps[-1] if ACC_COUNTER is None else ACC_COUNTER
if c0 not in dumps:
    raise FileNotFoundError("ACC_COUNTER = %d: no Velocity_Data_%d.npy in %s; available: %s"
                            % (c0, c0, ACC_FROM, dumps))
info = os.path.join(ACC_FROM, "Restart_Info_%d.txt" % c0)
if os.path.exists(info):
    t0 = [float(l.split()[1]) for l in open(info) if l.startswith("time ")][0]
else:
    t0 = c0 * SAVE_INTERVAL
    print("note: no Restart_Info_%d.txt -- assuming t0 = counter x SAVE_INTERVAL = %g" % (c0, t0))
if t0 < 2.0:
    print("WARNING: the start field is at t = %g, i.e. (close to) the smooth initial condition,\n"
          "         which has almost no aliasing: the errors below will look far too small.\n"
          "         Use a dump from the developed turbulence (e.g. t >= 10)." % t0)
if ACC_COUNTER is None and c0 != max(dumps):
    ACC_COUNTER = c0
nst  = max(1, int(round(ACC_T / DT)))
t_si = nst * DT * (1 + 1e-9)          # save exactly after nst steps (DT) / 4 nst steps (DT/4)
t_end = t0 + nst * DT + 0.5 * DT
print("from %s save %d (t = %g): %d steps of %g, reference %d steps of %g" %
      (ACC_FROM, c0, t0, nst, DT, 4 * nst, DT / 4))
shutil.rmtree(ACC_DIR, ignore_errors=True); os.makedirs(ACC_DIR)
runs = [("ref",   "exact", DT / 4), ("exact", "exact", DT), ("shift", "shift", DT)]
for name, dl, dt in runs:
    print("\n=== %s: --dealias %s --dt %g" % (name, dl, dt))
    assert run(acc_args(dl, dt, os.path.join(ACC_DIR, name))) == 0
    for f in glob.glob(os.path.join(ACC_DIR, name, "Velocity_Data_%d.npy" % (c0 + 1))):
        os.remove(f)                                  # the start field: same as the restart file

def last_dump(d):
    return max(glob.glob(os.path.join(d, "Velocity_Data_*.npy")),
               key=lambda f: int(re.search(r"_(\d+)\.npy$", f).group(1)))

R = np.load(last_dump(os.path.join(ACC_DIR, "ref")), mmap_mode="r")
kk = np.fft.fftfreq(N, 1.0 / N); kk[N // 2] = N // 2
k2 = kk[:, None, None]**2 + kk[None, :, None]**2 + np.arange(Nf)[None, None, :]**2
wz = np.where((np.arange(Nf) == 0) | (np.arange(Nf) == N // 2), 1.0, 2.0)[None, None, :]
kb = np.minimum(np.floor(np.sqrt(k2) + 0.5).astype(int), N // 2)
kmx = int(np.sqrt(2) * N / 3)
print("\nerrors after %g time units, relative to the exact scheme at DT/4:" % (nst * DT))
print("%-8s %10s %10s   %s" % ("run", "velocity", "gradients",
      "  ".join("k=%d" % k for k in (kmx // 4, kmx // 2, 3 * kmx // 4, kmx))))
for name in ("exact", "shift"):
    B = np.load(last_dump(os.path.join(ACC_DIR, name)), mmap_mode="r")
    dU = np.zeros(k2.shape); UU = np.zeros(k2.shape)
    for c in range(3):                                # one component at a time: RAM
        dU += np.abs(np.asarray(B[c]) - np.asarray(R[c]))**2
        UU += np.abs(np.asarray(R[c]))**2
    sh = lambda a: np.bincount(kb.ravel(), weights=(wz * a).ravel(), minlength=N // 2 + 1)
    es = np.sqrt(sh(dU) / np.maximum(sh(UU), 1e-300))
    print("%-8s %10.2e %10.2e   %s" % (name, np.sqrt(np.sum(wz * dU) / np.sum(wz * UU)),
          np.sqrt(np.sum(wz * k2 * dU) / np.sum(wz * k2 * UU)),
          "  ".join("%.1e" % es[k] for k in (kmx // 4, kmx // 2, 3 * kmx // 4, kmx))))
print("\n'exact' is the %s time-stepping error at DT; 'shift' adds the aliasing left over."
      % INTEGRATOR.upper())
print("For scale: at N = 64 and CFL 0.17, RK4 gave 1.6e-4 (exact) / 4.5e-4 (shift) after one")
print("large-eddy time, and the grid's own resolution error (N=64 vs 96) was 2.0e-2.")


# %% [markdown]
# ---
# ## Restarting from a previous run
# 
# Set `RESTART = "./Forced_NS_N768"` in INPUT and re-run. It picks up the
# newest save in that directory.
# 
# **The spectral velocity is the complete state.** The 2N register `q` is
# *overwritten* at stage 1 of every step (because $A_1=0$), and the forcing
# $f=(\epsilon/E_{\rm shell})u$ is recomputed from $u$ every stage, so nothing
# except $u$ (and $\theta$, and the particle positions) carries across a step
# boundary. Restarting from a dump is equivalent to never having stopped, up to
# the precision the dump was rounded to.
# 
# The only things the `.npy` files could not carry were the simulation time and
# the particle dispersion origin, so the solver now also writes:
# 
# * `Restart_Info_<c>.txt` — one per field dump: time, N, precision, and the
#   physical parameters, so a restart can check you have not changed `N` and warn
#   you if $\nu$, $\epsilon$ or $\nu_{\rm odd}$ differ.
# * `Particle_Origin.npy` — written once at seeding, holds $(X_0, V_0)$. With it
#   present, MSD and $R_L$ **continue the same curve** across a restart instead of
#   resetting their clock.
# 
# Notes:
# 
# * `TFINAL` is the **absolute** end time. Restarting at $t=40$ with `TFINAL=40`
#   does nothing — you want `TFINAL=80`.
# * `EINIT` is ignored on restart; the loaded field sets the energy.
# * Output counters continue from the restart point, and the resumed leg writes
#   `Energy_Enstrophy_Time_Series_c<n>.npy`, so **nothing is overwritten**. The
#   helpers cell stitches all legs back into one series automatically.
# * An fp32 dump can restart an fp64 build and vice versa — the loader converts.
#   `N` must match, and is checked. To change `N`, use `INIT_FILE` (next section).
# * If the restart fails for any reason the solver **exits** rather than quietly
#   starting from a random field.

# %% [markdown]
# ---
# ## Starting from a field of another resolution (`INIT_FILE`)
# 
# Spin up cheaply at a lower resolution, then pad to the full one:
# 
# 1. **Spin-up at 768³** (fits a 40 GB A100: ~30 GiB). Set `N = 768`, **`NU_OF_N = 1024`**
#    (the 1024³ viscosity, so $\nu$ does not jump later), `PARTICLES = False`, and run
#    ~10 time units (5 large-eddy times) from the random field. A 768³ step costs about
#    0.4× a 1024³ step, and the 4/3 larger dx allows a 4/3 larger `DT`, so the spin-up
#    costs ~0.3× what it would at 1024³.
# 2. **Copy one dump to Drive** (cell 19): `Velocity_Data_<c>.npy` (10.9 GB fp64) and its
#    `Restart_Info_<c>.txt`. Browser upload is impractical at this size; mount Drive.
# 3. **Production at 1024³:** `N = 1024`, `NU_OF_N = None`,
#    `INIT_FILE = "./drive/MyDrive/.../Velocity_Data_<c>.npy"`.
#    The solver keeps every mode of the file at the same wavevector, times $(N/n)^3$, so
#    $u(\mathbf{x})$ is the same function, and prints where the file's energy went
#    (kept / mean flow / Nyquist / outside the sphere / divergent part).
# 
# `INIT_FILE = None` (and `RESTART = None`) starts from the random field as before.
# A field from another code that is stored **on the grid**, `u[c, x, y, z]` or
# `u[x, y, z, c]` or three files: `python3 make_ic.py OUT.npy U.npy` (`--axes zyx` if
# the array is stored z, y, x), then `INIT_FILE = "OUT.npy"`.
# 
# **How long until the padded run is a native run?** Measured on the CPU emulator,
# 6 independent start fields each, against the long-time statistics of a native 64³ run
# ($k_{\max}\eta = 1.07$; 48 → 64 is the same ratio as 768 → 1024):
# 
# | start field | new shells $k_{\max}(n) < k < k_{\max}(N)$ | excess at the old cutoff | dissipation, $E$ | native within the noise after |
# |---|---|---|---|---|
# | **48³, same $\nu$** ($k_{\max}\eta$ 0.81 → 1.07) | empty → 1.04 of native at $1\,\tau_\eta$, settled by $3\,\tau_\eta$ | +57% → +2% at $2\,\tau_\eta$ | within 1% throughout | **~5 $\tau_\eta$** |
# | 32³, same $\nu$ (0.54 → 1.07) | overshoot +25% at $2\,\tau_\eta$, −10% at 8–17 $\tau_\eta$ | +64% | dissipation −7% at 6–12 $\tau_\eta$ | ~30 $\tau_\eta$ |
# | 48³, its own default $\nu$ (×1.47) | 0.75 of native after $8\,\tau_\eta$ | −58% (deficit) | dissipation 0.68 → 1.05 → 1.0 | ~4 time units ≈ 2 $E/\epsilon$ |
# | random field (for comparison) | | | dissipation 0.40 at t = 1, 1.21 at t = 4 | ~6–8 time units |
# 
# So: **768³ with `NU_OF_N = 1024`, then discard ~10 $\tau_\eta$** — at the default 1024³
# $\nu$ ($\tau_\eta$ = 0.019) that is 0.2 time units, ~200 steps. Seed particles after that
# (`PART_START` ≈ 20 $\tau_\eta$). Exactness of the padding itself (field vs an independent numpy
# implementation 1e-16; $u$ at the old grid points unchanged to 4e-16) is checked by
# `tests/run_checks.sh` section 18.
# 

# %% [markdown]
# ---
# # Analysis
# 
# **Run the next cell first.** It loads every output file and defines the names
# (`t, E, Z, vf, pf, ...`) that all the plotting cells use.

# %%
#@title 10. Analysis helpers  —  RUN THIS BEFORE ANY PLOT CELL
import os, glob
import numpy as np
import matplotlib.pyplot as plt

def _sorted(pat):
    """files named <stem>_<counter>.npy, sorted numerically by counter"""
    fs = glob.glob(os.path.join(OUT, pat))
    out = []
    for p in fs:
        tail = p.rsplit("_", 1)[1].split(".")[0]
        if tail.isdigit():
            out.append((int(tail), p))
    return [p for _, p in sorted(out)]

def _series(stem):
    """Stitch a time series across restarts.

    A fresh run writes <stem>.npy; a leg resumed with --restart writes
    <stem>_c<n>.npy so it cannot clobber the first leg. Concatenate them in
    time order and drop the duplicated sample at each restart point (the
    resumed leg's first sample is the save it restarted FROM)."""
    segs = []
    for p in sorted(glob.glob(os.path.join(OUT, stem + "*.npy"))):
        a = np.load(p)
        if a.ndim == 2 and a.shape[1] > 0:
            segs.append(a)
    if not segs:
        return None, 0
    A = np.concatenate(segs, axis=1)
    A = A[:, np.argsort(A[0], kind="stable")]
    keep = np.concatenate([[True], np.diff(A[0]) > 1e-12])
    return A[:, keep], len(segs)

vf  = _sorted("Velocity_Data_*.npy")             # spectral velocity snapshots
sf  = _sorted("Scalar_Data_*.npy")               # spectral tracer snapshots
pf  = _sorted("Particle_Data_*.npy")             # particle snapshots
sv  = _sorted("Velocity_Statistics_Data_*.npy")  # E(k)
stt = _sorted("Scalar_Statistics_Data_*.npy")    # E_theta(k)

ts, nseg = _series("Energy_Enstrophy_Time_Series")
if ts is None:
    raise FileNotFoundError(
        "no time series in %s -- did cell 9 (RUN) finish? %s"
        % (OUT, sorted(os.listdir(OUT))[:8] if os.path.isdir(OUT) else "no such dir"))
if ts.shape[0] == 6:
    t, E, Z, Tvar, chi, flux = ts
else:
    t, E, Z = ts
    Tvar = chi = flux = None
has_theta = bool(SCALAR) and Tvar is not None and np.any(Tvar != 0)

ps, pseg = _series("Particle_Statistics_Time_Series")
has_part = bool(PARTICLES) and ps is not None
if has_part:
    tp, msd, up2, lag = ps

def trapz(y, x):
    """np.trapz was renamed in NumPy 2; do it by hand so both work."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum(np.diff(x) * 0.5 * (y[1:] + y[:-1])))

def midplane_x(spec_slab_fn, N, Nf, chunk=32):
    """One physical plane at x-index N//2 out of a spectral (N,N,Nf) field.

    A full irfftn of a 768^3 snapshot allocates 3.6 GB per component and will
    OOM a standard Colab VM. We only want one plane, and
    exp(2*pi*i*kx*(N/2)/N) = (-1)^kx, so the x-transform is just a signed sum
    that can be streamed in chunks from an mmap'd file."""
    sgn = (-1.0)**np.arange(N)
    s = np.zeros((N, Nf), dtype=np.complex128)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        s += (spec_slab_fn(i, j) * sgn[i:j, None, None]).sum(axis=0)
    return np.fft.irfft2(s/N, s=(N, N))          # 1/N * 1/N^2 = 1/N^3   OK

print("output dir :", OUT)
print("time series: %d samples, t = %.2f .. %.2f%s" %
      (len(t), t[0], t[-1], ("   [%d legs stitched]" % nseg) if nseg > 1 else ""))
print("field dumps: %d velocity, %d scalar" % (len(vf), len(sf)))
print("spectra    : %d velocity, %d scalar" % (len(sv), len(stt)))
print("particles  : %s" % (("%d dumps, %d samples%s" %
      (len(pf), len(tp), ("   [%d legs]" % pseg) if pseg > 1 else ""))
      if has_part else "none"))
print("tracer     : %s" % ("present" if has_theta else "none"))
_ri = sorted(glob.glob(os.path.join(OUT, "Restart_Info_*.txt")))
print("restartable: %d checkpoints%s" %
      (len(_ri), ("   newest: " + os.path.basename(_ri[-1])) if _ri else ""))

# %%
#@title 11. Time series
n = 3 if has_theta else 2
fig, ax = plt.subplots(1, n, figsize=(5*n, 4))
ax[0].plot(t, E, lw=1); ax[0].set_xlabel("t"); ax[0].set_ylabel("E")
ax[1].plot(t, Z, lw=1); ax[1].set_xlabel("t"); ax[1].set_ylabel("enstrophy")
ax[1].set_yscale("log")
if has_part:
    ax[0].axvline(PART_START, color="k", ls=":", lw=1)
    ax[0].text(PART_START, ax[0].get_ylim()[1], " particles seeded",
               va="top", fontsize=8)
if has_theta:
    ax[2].plot(t, Tvar, lw=1)
    ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$\langle\theta^2\rangle/2$")
plt.tight_layout(); plt.show()

Elate = E[2*len(E)//3:]
print("steady state (last third):  E = %.4f   u_rms = %.4f" %
      (Elate.mean(), np.sqrt(2*Elate.mean()/3)))
print("max|u| is typically 3-4x u_rms -- compare against the dt advisor.")

# %%
#@title 12. Energy budget — the odd-viscosity validation
# Odd viscosity is non-dissipative, so  dE/dt = eps - 2 nu Omega  must hold
# WHETHER OR NOT nu_odd is on. If it does not, that is a bug.
#
# The sharp form of this test is the INTEGRATED one. Differentiating an E(t)
# sampled every SAVE_INTERVAL (~2 samples per eddy turnover) costs you 10-30%
# by itself, so the derivative plot cannot tell a correct code from a broken
# one. Integrating the right-hand side and comparing to E(t) has no such issue.
dEdt = np.gradient(E, t)
rhs  = EPSILON - 2*NU_EFF*Z
cum  = np.concatenate([[0.0], np.cumsum(np.diff(t)*0.5*(rhs[1:] + rhs[:-1]))])
E_pred = E[0] + cum

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, dEdt, lw=1, label=r"$dE/dt$ (finite-differenced)")
ax[0].plot(t, rhs,  lw=1, ls="--", label=r"$\epsilon - 2\nu\Omega$")
ax[0].set_xlabel("t"); ax[0].set_ylabel("energy budget"); ax[0].legend(fontsize=8)
ax[0].set_title("derivative form (noisy by construction)", fontsize=9)
ax[1].plot(t, E, lw=1.5, label=r"$E(t)$ measured")
ax[1].plot(t, E_pred, lw=1, ls="--", label=r"$E(0)+\int(\epsilon-2\nu\Omega)dt$")
ax[1].set_xlabel("t"); ax[1].set_ylabel("E"); ax[1].legend(fontsize=8)
ax[1].set_title("integrated form  <-- this is the real test", fontsize=9)
plt.tight_layout(); plt.show()

m = slice(len(t)//3, None)
res = np.abs(dEdt[m]-rhs[m]).mean()
print("derivative form : mean|residual| = %.4e  (%.1f%% -- tens of percent here"
      " is EXPECTED, it is the differencing)" %
      (res, res/np.abs(rhs[m]).mean()*100))
drift = np.abs(E - E_pred).max() / max(E.max()-E.min(), 1e-30)
print("integrated form : max drift = %.2f%% of the range of E" % (drift*100))
print("   under ~1% in fp64 is a pass. fp32 with -use_fast_math sits higher;")
print("   what matters is that it does NOT change when NU_ODD is switched on.")

# %%
#@title 13. Tracer variance budget   (skipped when SCALAR = False)
# steady state:  chi  ==  2*eps_theta - 2 G <u_g theta>
if not has_theta:
    print("SCALAR = False -- no Eulerian tracer field in this run. "
          "See the particle cells below instead.")
else:
    src = 2*EPS_THETA - 2.0*GRAD*flux
    plt.figure(figsize=(6, 4))
    plt.plot(t, chi, lw=1, label=r"$\chi=2\kappa\langle|\nabla\theta|^2\rangle$")
    plt.plot(t, src, lw=1, ls="--",
             label=r"$2\epsilon_\theta-2G\langle u_g\theta\rangle$")
    plt.xlabel("t"); plt.ylabel("tracer variance budget"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    m = slice(2*len(t)//3, None)
    a, b = chi[m].mean(), src[m].mean()
    print("last third:  chi = %.5f   source = %.5f   mismatch = %.2f%%" %
          (a, b, abs(a-b)/max(abs(b), 1e-30)*100))

# %% [markdown]
# ---
# ## Lagrangian particle diagnostics
# 
# Three things, in this order.
# 
# 1. **$\langle|u_p|^2\rangle$ vs $2E$** — a *consistency* check, not physics.
#    Fluid tracers in an incompressible flow stay uniformly distributed, so
#    sampling $u$ on the particles and averaging on the grid must agree. A
#    systematic gap means the interpolation is losing energy: re-run with
#    `PART_INTERP = "linear"` and confirm the gap gets much worse, which proves
#    that is what you are looking at.
# 
# 2. **Mean square displacement** — Taylor (1921): ballistic
#    $\langle|\Delta X|^2\rangle = \langle|u|^2\rangle\tau^2$ for $\tau\ll T_L$,
#    diffusive $=2\langle|u|^2\rangle T_L\tau$ for $\tau\gg T_L$. Both asymptotes
#    are drawn with **no fitted constants** — measured $\langle|u|^2\rangle$ and
#    measured $T_L$.
# 
# 3. **Lagrangian velocity autocorrelation** $R_L(\tau)$, whose integral is $T_L$.
#    Only meaningful if the particles were seeded in a statistically steady flow —
#    that is what `PART_START` is for.

# %%
#@title 14. Particle statistics
if not has_part:
    print("PARTICLES = False (or no particle data) -- nothing to plot here.")
else:
    tau = tp - tp[0]                       # time since seeding
    RL  = lag / lag[0]                     # normalised autocorrelation

    zc   = np.where(RL <= 0)[0]            # integrate R_L to its first zero
    icut = zc[0] if len(zc) else len(RL)
    T_L  = trapz(RL[:icut], tau[:icut]) if icut > 1 else float("nan")

    twoE = np.interp(tp, t, 2*E)           # grid 2E at the particle sample times
    u2   = up2.mean()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    ax[0].plot(tp, up2,  lw=1.5, label=r"$\langle|u_p|^2\rangle$  (particles)")
    ax[0].plot(tp, twoE, lw=1, ls="--", label=r"$2E$  (grid)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("kinetic energy x2")
    ax[0].legend(fontsize=8)
    ax[0].set_title("interpolation consistency", fontsize=9)

    good = tau > 0
    ax[1].loglog(tau[good], msd[good], lw=1.5, label=r"$\langle|\Delta X|^2\rangle$")
    ax[1].loglog(tau[good], u2*tau[good]**2, "k--", lw=1,
                 label=r"ballistic  $\langle u^2\rangle\tau^2$")
    if np.isfinite(T_L):
        ax[1].loglog(tau[good], 2*u2*T_L*tau[good], "r-.", lw=1,
                     label=r"diffusive  $2\langle u^2\rangle T_L\tau$")
        ax[1].axvline(T_L, color="g", ls=":", lw=1)
    ax[1].set_xlabel(r"$\tau = t - t_{\rm seed}$"); ax[1].set_ylabel("MSD")
    ax[1].legend(fontsize=8); ax[1].set_title("Taylor dispersion", fontsize=9)

    ax[2].plot(tau, RL, lw=1.5)
    ax[2].axhline(0, color="k", lw=0.5)
    if np.isfinite(T_L):
        ax[2].axvline(T_L, color="g", ls=":", lw=1, label="$T_L$ = %.3f" % T_L)
        ax[2].legend(fontsize=8)
    ax[2].set_xlabel(r"$\tau$"); ax[2].set_ylabel(r"$R_L(\tau)$")
    ax[2].set_title("Lagrangian autocorrelation", fontsize=9)
    plt.tight_layout(); plt.show()

    gap = (up2 - twoE) / twoE * 100
    m = slice(len(tp)//3, None)
    print("<|u_p|^2> vs 2E:  mean gap = %+.3f %%   max |gap| = %.3f %%" %
          (gap[m].mean(), np.abs(gap).max()))
    print("   sub-percent = the interpolation is fine. A few percent with")
    print("   PART_INTERP='cubic' would be suspicious; re-run with 'linear' and")
    print("   it should blow up to ~5-15%, which confirms the diagnostic works.")
    if np.isfinite(T_L):
        print("\nT_L = %.4f    eddy turnover L/u_rms ~ %.3f" %
              (T_L, (2*np.pi/FORCED_SHELL)/np.sqrt(u2/3)))
        print("eddy diffusivity  D = <u^2> T_L / 3 = %.4f" % (u2*T_L/3))

# %%
#@title 15. Particles in the flow — slab overlay, 3D scatter, velocity pdf
if not (has_part and pf):
    print("no particle dumps -- set PARTICLES = True and re-run.")
else:
    P = np.load(pf[-1])                    # (2, 3, np)
    L = 2*np.pi
    X = P[0] % L                           # wrap for plotting only
    V = P[1]
    speed = np.sqrt((V**2).sum(axis=0))
    print("%s   %d particles   speed: mean %.3f  max %.3f" %
          (os.path.basename(pf[-1]), X.shape[1], speed.mean(), speed.max()))

    fig = plt.figure(figsize=(15, 5))

    # (a) thin slab of particles over the u_x plane
    axa = fig.add_subplot(1, 3, 1)
    if vf:
        U = np.load(vf[-1], mmap_mode="r")  # (3,N,N,Nf), never fully loaded
        ux = midplane_x(lambda i, j: np.asarray(U[0, i:j]), N, Nf)
        axa.imshow(ux.T, origin="lower", cmap="viridis",
                   extent=[0, L, 0, L], aspect="equal")
        del U, ux
    xc, half = L/2, L/(2*N)*8              # 8-cell-thick slab about x = pi
    sel = np.abs(((X[0]-xc+L/2) % L) - L/2) < half
    axa.plot(X[1][sel], X[2][sel], ".", ms=1.2, color="w", alpha=0.8)
    axa.set_xlabel("y"); axa.set_ylabel("z")
    axa.set_title("$u_x$ plane at $x=\\pi$ + %d particles in an 8-cell slab"
                  % sel.sum(), fontsize=9)

    # (b) 3D scatter coloured by speed
    axb = fig.add_subplot(1, 3, 2, projection="3d")
    k = min(6000, X.shape[1])
    idx = np.random.default_rng(0).choice(X.shape[1], k, replace=False)
    sc = axb.scatter(X[0][idx], X[1][idx], X[2][idx], c=speed[idx],
                     s=1.5, cmap="magma")
    axb.set_xlabel("x"); axb.set_ylabel("y"); axb.set_zlabel("z")
    axb.set_title("%d particles, coloured by $|u_p|$" % k, fontsize=9)
    fig.colorbar(sc, ax=axb, shrink=0.6)

    # (c) velocity pdf vs Gaussian
    axc = fig.add_subplot(1, 3, 3)
    vp = V.ravel()
    sd = vp.std()
    bins = np.linspace(-5*sd, 5*sd, 120)
    axc.hist(vp, bins=bins, density=True, histtype="step", lw=1.5,
             label="particle velocity components")
    axc.plot(bins, np.exp(-bins**2/(2*sd**2))/(sd*np.sqrt(2*np.pi)), "k--",
             lw=1, label="Gaussian, same variance")
    axc.set_yscale("log"); axc.set_xlabel(r"$u_i$"); axc.set_ylabel("pdf")
    axc.legend(fontsize=8)
    axc.set_title("single-point velocity pdf (should be ~Gaussian)", fontsize=9)

    plt.tight_layout(); plt.show()

# %%
#@title 16. Particle trajectories
# Uses the UNWRAPPED positions, so paths stay continuous across the box edges.
if not (has_part and len(pf) >= 2):
    print("need at least 2 particle dumps "
          "(lower PART_DUMP_EVERY or raise TFINAL).")
else:
    ntr  = 40
    traj = np.stack([np.load(f)[0][:, :ntr] for f in pf])    # (nt, 3, ntr)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    for p in range(ntr):
        ax1.plot(traj[:, 0, p], traj[:, 1, p], lw=0.8)
    ax1.plot(traj[0, 0, :], traj[0, 1, :], "k.", ms=4, label="start")
    ax1.set_xlabel("x (unwrapped)"); ax1.set_ylabel("y (unwrapped)")
    ax1.set_title("%d trajectories, projected" % ntr, fontsize=9)
    ax1.legend(fontsize=8); ax1.set_aspect("equal", "datalim")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for p in range(min(12, ntr)):
        ax2.plot(traj[:, 0, p], traj[:, 1, p], traj[:, 2, p], lw=0.9)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.set_title("12 trajectories in 3D", fontsize=9)
    plt.tight_layout(); plt.show()

    d = traj - traj[0]
    print("MSD from this %d-particle sample at t_end: %.4f "
          "(full ensemble is in cell 14)" % (ntr, (d[-1]**2).sum(axis=0).mean()))

# %%
#@title 17. Spectra
Ek = np.load(sv[-1])
k  = np.arange(len(Ek))

fig, ax = plt.subplots(figsize=(6.5, 4.8))   # own figure: never inherit axes
ax.loglog(k[1:], Ek[1:], lw=1, label="E(k)")
if has_theta and stt:
    ax.loglog(k[1:], np.load(stt[-1])[1:], lw=1, label=r"$E_\theta(k)$")
ax.loglog(k[1:], 0.3*k[1:]**(-5/3.), "k--", lw=1, label=r"$k^{-5/3}$")
ax.axvline(FORCED_SHELL, color="r", ls=":", lw=1, label="forcing")
ax.axvline(kmax, color="0.5", ls="--", lw=1, label=r"$k_{max}$ (dealias)")
if NU_ODD > 0:
    ax.axvline(EPSILON**0.25*NU_ODD**-0.75, color="m", ls="-.", lw=1,
               label=r"$k_{odd}$")
    ax.axvline(EPSILON**0.25*NU_EFF**-0.5*NU_ODD**-0.25, color="g", ls="-.",
               lw=1, label=r"$k_c$")
ax.set_xlabel("k"); ax.set_ylabel("spectrum"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

tail = Ek[int(0.6*kmax):int(0.95*kmax)]
if len(tail) > 3 and tail[-1] > tail[0]:
    print("*** E(k) turns UP approaching k_max. That is a timestep/dealiasing")
    print("    symptom, not physics. Halve DT and compare. ***")

# %%
#@title 18. Snapshot slices  (uses the LAST dump, not the initial condition)
# The original notebook loaded Velocity_Data_0.npy, which is written at t = 0
# BEFORE the first step -- i.e. the initial condition, not the result.
if not vf:
    print("no field dumps found (DUMP_EVERY may have skipped them all).")
else:
    cidx = int(vf[-1].rsplit("_", 1)[1].split(".")[0])
    print("%s   (save #%d, t = %.2f)" %
          (os.path.basename(vf[-1]), cidx, cidx*SAVE_INTERVAL))
    U  = np.load(vf[-1], mmap_mode="r")
    ux = midplane_x(lambda i, j: np.asarray(U[0, i:j]), N, Nf)

    ncol = 2 if (has_theta and sf) else 1
    fig, axs = plt.subplots(1, ncol, figsize=(5.5*ncol, 4.4), squeeze=False)
    im = axs[0][0].imshow(ux.T, origin="lower", cmap="viridis",
                          extent=[0, 2*np.pi, 0, 2*np.pi])
    axs[0][0].set_title("$u_x$ at $x=\\pi$, t = %.1f" % (cidx*SAVE_INTERVAL))
    axs[0][0].set_xlabel("y"); axs[0][0].set_ylabel("z")
    plt.colorbar(im, ax=axs[0][0])
    del U, ux

    if ncol == 2:
        p = os.path.join(OUT, "Scalar_Data_%d.npy" % cidx)
        if os.path.exists(p):
            Tf = np.load(p, mmap_mode="r")
            th = midplane_x(lambda i, j: np.asarray(Tf[i:j]), N, Nf)
            im = axs[0][1].imshow(th.T, origin="lower", cmap="RdBu_r",
                                  extent=[0, 2*np.pi, 0, 2*np.pi])
            axs[0][1].set_title(r"$\theta'$ at $x=\pi$")
            axs[0][1].set_xlabel("y"); axs[0][1].set_ylabel("z")
            plt.colorbar(im, ax=axs[0][1])
            del Tf, th
    plt.tight_layout(); plt.show()

# %%
#@title 19. Copy results to Drive (optional)
# from google.colab import drive
# drive.mount("./drive")
# !cp -r {OUT} ./drive/MyDrive/



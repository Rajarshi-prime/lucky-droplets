"""
Builds the initial condition with initial_conditions.py and runs forced_ns_droplets on it.
One process only: the CUDA code owns the whole domain, so never under mpirun.
"""
import numpy as np
import pathlib, subprocess, os, glob, sys
from numpy.fft import fftfreq, rfft, irfft, rfftn, irfftn
from mpi4py import MPI
from initial_conditions import InitialConditions
from particles import MPI_particles

comm = MPI.COMM_WORLD
rank, num_process = comm.Get_rank(), comm.Get_size()

## ––––––––––––- What this run does ––––––––––––––
forcestart = False #! True : fresh velocity field, False : load the saved fields
start_big_particle = False #! True : place the big particles afresh, False : load their saved state
isforcing = True
build = True #! compile forced_ns_droplets.cu before running it
m = 2.0 #! Desired kmax*eta. forced-dns-sm-big.py reads this from sys.argv[-2]
gravity = False #! likewise, from sys.argv[-1] there
wg = "with_g" if gravity else "wo_g"

## ––––––––––––- The CUDA toolchain ––––––––––––––
cuda_home = os.environ.get("CUDA_HOME") or max(glob.glob("/usr/local/cuda*"), default = "/usr/local/cuda")
nvcc = os.path.join(cuda_home, "bin", "nvcc")
if not os.path.exists(nvcc): nvcc = "nvcc"
cuda_lib = os.path.join(cuda_home, "targets", "x86_64-linux", "lib")
binary = os.path.abspath("./forced_ns_droplets")

arch = None #! None asks the card; set "sm_89" and so on to override
q = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap","--format=csv,noheader"],
                   capture_output = True, text = True)
if q.returncode != 0 or not q.stdout.strip():
    if arch is None: raise SystemExit("no GPU found, and arch is not set by hand")
else:
    name, memtot, cc = [s.strip() for s in q.stdout.strip().split(",")]
    if arch is None: arch = "sm_" + cc.replace(".", "")
    print(f"{name}, {float(memtot.split()[0])/1024:.1f} GiB, arch {arch}")

## ––––––––––––- Time steps ––––––––––––––
N = 256
dt =  0.2*0.256/N #! Such that increasing resolution will decrease the dt
dtmax = 5*0.256/N
dtmin = 0.0005*0.256/N
T = 100
one_step = False #! True stops after one step, which is the checkpoint comparison
dt_save_fields = 1.0
dt_save_particles = 0.1 #! dt_save_fields should be a multiple of this
tdec = max(0, int(np.ceil(-np.log10(min(dt_save_fields,dt_save_particles))))) #! decimals in the time_ folder names
sts = 0.001
stb_s = [0.017,0.017/4, 0.017*4,0.017*6.35]*5 #! For 5 different initial conditions.
uniquestbs = len(np.unique(stb_s))
init_name = ['random']*uniquestbs + ['Qpos']*uniquestbs + ['Qneg']*uniquestbs + ['Qpos_high']*uniquestbs + ['Qneg_high']*uniquestbs
if uniquestbs !=4 : raise SystemExit("Check the initial big Stokes numbers")
stb_s = np.array(stb_s)
Nprtcl = np.round(8192*(0.017/stb_s)**1.5).astype(np.int32)
d = 3
M0 = 7.2e-4 #2.6\mu m particles with volfrac 1e-6.
rhop = 1000

## ––––––––––––- Grid and viscosity ––––––––––––––
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np, (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint = False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing = 'ij')

Kx = Ky = fftfreq(N, 1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])
kx, ky, kz = np.meshgrid(Kx, Ky[sx], Kz, indexing = 'ij')

kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

lp = 1 # Hyperviscosity power
nu0 = 0.5 #! Viscosity for N = 1
kmax = N*2**0.5//3
eta = m/kmax
nu = nu0*(eta)**(2*(lp - 1/3))
einit = 1*(TWO_PI)**3 # Initial energy
nshells = 1 # Number of consecutive shells to be forced
tf = eta**2/nu #* the expected kolmogorov time scale
tps = sts*tf # The particle timescale
eta_dim = 0.6*1e-3 #! Kolmogorov length scale in meter in clouds
tau_eta_dim = 0.03 #! Kolmogorov time scale in seconds in clouds.
g = 9.81*(tau_eta_dim**2/eta_dim)*(eta/tf**2) if gravity else 0 # Gravity in the simulation units
f0 = 0.5 * TWO_PI**3/ nshells #! Total power input at each shells

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
dealias = kint<=2**0.5*N/3 #! Spherical dealiasing
phase_k = np.exp(1j*(kx*dx/2. + ky*dy/2. + kz*dz/2.)) *dealias
conjphase_k = np.conjugate(phase_k)*dealias
invlap = dealias/np.where(lap == 0, np.inf, lap)
normalize = np.where((kz == 0) + (kz == N//2), 1/(N**6/TWO_PI**3), 2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.

re = 1/nu if nu != 0 else "inf"
loadPath = pathlib.Path(f"./data_cosine/forced_{isforcing}/N_{N}_Re_{re:.1f}") #! Where the restart data is read from
savePath = pathlib.Path(f"./gpu_ic/forced_{isforcing}/N_{N}_Re_{re:.1f}") #! Where the initial condition is handed over
outPath = pathlib.Path("./gpu_out")

## ––––––––––––- zeros arrays for fields ––––––––––––––
u = np.zeros((3, Np, N, N), dtype = np.float64)
n = np.zeros((len(stb_s), Np, N, N), dtype = np.float64)
uk = np.zeros((3, N, Np, Nf), dtype = np.complex128)
rhsu, rhsv, rhsw = u[0].copy(), u[0].copy(), u[0].copy()

## ––––––FFT + iFFT + derivative functions––––––-
def rfft_mpi(u, fu):
    fu[:] = rfftn(u)
    return fu

def irfft_mpi(fu, u):
    u[:] = irfftn(fu, u.shape, axes = (0,1,2)) #! axes named: numpy 2 deprecates giving s without them
    return u

def diff_x(u, u_x):
    u_x[:] = irfft(1j*kx_diff*rfft(u, axis = 0), N, axis = 0)
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis = 1), N, axis = 1)
    return u_y

def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis = 2), N, axis = 2)
    return u_z

def e3d_to_e1d(x):
    return np.histogram(k.ravel(),bins = shells,weights = x.ravel())[0]

def clip_zero(x,nislist = False):
    """Clips negative values to zero and rescales the to conserve the mean
    """
    if not nislist:
        oldmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
        x[:] = np.clip(x,0,None)
        newmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
        if oldmean * newmean <= 0:
            raise ValueError("means must be nonzero and share a sign")
        x[:] *= oldmean / newmean
        return x #! rescaled in place, so a copy here would duplicate the largest array in the program

    for jj in range(x.shape[0]):
        oldmean = comm.allreduce(np.sum(x[jj]),op = MPI.SUM)/N**3
        x[jj] = np.clip(x[jj],0,None)
        newmean = comm.allreduce(np.sum(x[jj]),op = MPI.SUM)/N**3
        if oldmean * newmean <= 0:
            raise ValueError("means must be nonzero and share a sign")
        x[jj] *= oldmean / newmean
    return x

## ––––––––––––- Building the initial condition ––––––––––––––
#! start_big_particle makes find_path return loadPath/"last", whatever time_ folders exist
if not forcestart:
    needed = loadPath/"last" if start_big_particle else loadPath
    if not needed.is_dir():
        raise SystemExit(f"{needed} is not there. Set forcestart = True, or check that m = {m} "
                         f"is the value this data was written at (it sets Re = {re:.1f} in the path).")

stbs = []
for i,stb in enumerate(stb_s):
    stbs.append(MPI_particles(comm, L, N, Nprtcl[i],sts, stb,g,nu, tf,rhop,M0 ,d,X,Y,Z, x,y,z))
    stbs[i].to_interp(2*d+1) # u, v_s and c_s
    stbs[i].to_exterp(1)

ic = InitialConditions(comm, N, L, d, loadPath, u, uk, n,
                       kx, ky, kz, k, kint, dealias, invlap, normalize, einit,
                       rfft_mpi, irfft_mpi, e3d_to_e1d,
                       wg, tps, tf, dt_save_fields,
                       mode = "all_stokes", clip = clip_zero, load_dealias = False,
                       start_big_particle = start_big_particle,
                       phase_k = phase_k, conjphase_k = conjphase_k,
                       diff_x = diff_x, diff_y = diff_y, diff_z = diff_z,
                       fresh_particles = "qcriterion", stb_s = stb_s, init_name = init_name,
                       uniquestbs = uniquestbs, Nprtcl = Nprtcl)

paths, tinit = ic.initialize_fields(forcestart)
ic.initialize_particles(stbs, paths)
n = clip_zero(n,nislist = True)
tend = tinit if one_step else T #! the loop runs while tt <= tend and tt starts at tinit

divmax = np.max(np.abs(diff_x(u[0],rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw)))
e0 = 0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + u[2]**2)
print(f"tinit {tinit}, max divergence {divmax}, physical energy {e0}")
#! the solver prints these same three lines before it allocates; they should agree digit for digit
print(f"N {N}  m {m:g}  kmax {kmax:g}  eta {eta:g}  nu {nu:g}  Re {1/nu:g}  tf {tf:g}")
print(f"sts {sts:g}  M0 {M0:g}  rhop {float(rhop):g}  eps {f0/TWO_PI**3*isforcing:g}  gravity {g:g}  lp {lp}")
print(f"dtmin {dtmin:g}  dtmax {dtmax:g}  dt_save_fields {dt_save_fields:g}  dt_save_particles {dt_save_particles:g}")

## ––––––––––––- Handing it to the CUDA code ––––––––––––––
#! raw .npy, whole domain. The two subdirectory spellings are not interchangeable
rundir = savePath/f"time_{tinit:.{tdec}f}"
rundir.mkdir(parents = True, exist_ok = True)
for key, comp in (("uk", 0), ("vk", 1), ("wk", 2)):
    np.save(rundir/f"{key}.npy", uk[comp])

for jj in range(len(stb_s)):
    dens_dir = rundir/f"{wg}_sts_{sts:.3f}_stb_{stb_s[jj]:.3f}_init_{init_name[jj]}"
    dens_dir.mkdir(parents = True, exist_ok = True)
    np.save(dens_dir/"n.npy", n[jj])

    prtcl_dir = rundir/f"{wg}_stb_{stb_s[jj]:.3f}_sts_{sts:.3f}_init_{init_name[jj]}"
    prtcl_dir.mkdir(parents = True, exist_ok = True)
    coord = stbs[jj].coord
    np.save(prtcl_dir/"pos.npy", coord[:,:d])
    np.save(prtcl_dir/"vel.npy", coord[:,d:2*d])
    np.save(prtcl_dir/"mass.npy", coord[:,-1])
    np.save(prtcl_dir/"prtclid.npy", stbs[jj].prtclid.astype(np.float64))

#! the .cu holds the same value as each -DP_ default; repr() round-trips a double where str() does not
if build:
    cmd = [nvcc,"-O3","-std=c++17",f"-arch={arch}",f"-DNS_N={N}","-DNS_DOUBLE", #! no -use_fast_math: it reassociates the arithmetic being compared
           f"-DP_LP={lp}", f"-DP_NU0={nu0!r}", f"-DP_STS={sts!r}", f"-DP_M0={M0!r}",
           f"-DP_RHOP={float(rhop)!r}", f"-DP_EPS={f0/TWO_PI**3*isforcing!r}",
           f"-DP_NSHELLS={nshells}", f"-DP_ETA_DIM={eta_dim!r}", f"-DP_TAU_ETA_DIM={tau_eta_dim!r}",
           f"-DP_DTMIN={dtmin!r}", f"-DP_DTMAX={dtmax!r}",
           f"-DP_DT_SAVE_FIELDS={float(dt_save_fields)!r}", f"-DP_DT_SAVE_PARTICLES={float(dt_save_particles)!r}",
           "forced_ns_droplets.cu","-o",binary,
           "-L"+cuda_lib, "-Xlinker", "-rpath="+cuda_lib, "-lcufft"] #! -rpath so the binary finds libcufft at run time
    print(" ".join(cmd), "\n")
    r = subprocess.run(cmd, capture_output = True, text = True)
    print(r.stdout[-4000:]); print(r.stderr[-4000:])
    if r.returncode != 0: raise SystemExit("build failed")

args = [binary, "--restart", str(rundir), "--out", str(outPath),
        "--stb", ",".join(repr(float(s)) for s in stb_s[:uniquestbs]), #! distinct values: the solver takes the outer product
        "--init", ",".join(init_name[::uniquestbs]),
        "--m", repr(float(m)), "--gravity", str(int(gravity)), "--tend", repr(float(tend))]
print(" ".join(args), "\n")
p = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, bufsize = 1)
for line in p.stdout:
    sys.stdout.write(line); sys.stdout.flush()
p.wait()
if p.returncode != 0: raise SystemExit(p.returncode)

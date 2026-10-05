"""
Finding and loading a saved state, or building a fresh one, for the forced-dns-* scripts.

Loading script builds the grid, the spectral operators and the MPI fft helpers itself and
hands them to InitialConditions, which owns the two startup paths: restart from saved
data, or start from scratch. The variants differ in where they look for the restart
folder (mode), how they store the number density (ntransform, ndefault, clip) and how
they place their big particles when starting fresh (fresh_particles).
"""
import numpy as np
import os
from mpi4py import MPI
from scipy.interpolate import PchipInterpolator


class InitialConditions:
    def __init__(self, comm, N, L, d, loadPath, u, uk, n,kx, ky, kz, k, kint, dealias, invlap, normalize, einit,rfft_mpi, irfft_mpi, e3d_to_e1d,wg, tps, tf, dt_save,mode = "second_last", clip = None, ntransform = None, ndefault = 1.0,load_dealias = True, start_big_particle = False,phase_k = None, conjphase_k = None, diff_x = None, diff_y = None, diff_z = None,fresh_particles = "qcriterion", stb_s = None, init_name = None, uniquestbs = None, Nprtcl = None):
        self.comm, self.N, self.L, self.d = comm, N, L, d
        self.rank, self.num_process = comm.Get_rank(), comm.Get_size()
        self.Np = N//self.num_process
        self.sx = slice(self.rank*self.Np, (self.rank + 1)*self.Np)
        self.X = np.linspace(0, L, N, endpoint = False)
        self.dx = self.X[1] - self.X[0]
        self.TWO_PI = 2*np.pi

        self.loadPath = loadPath
        self.u, self.uk, self.n = u, uk, n
        self.kx, self.ky, self.kz, self.k, self.kint = kx, ky, kz, k, kint
        self.dealias, self.invlap, self.normalize, self.einit = dealias, invlap, normalize, einit
        self.phase_k, self.conjphase_k = phase_k, conjphase_k
        self.rfft_mpi, self.irfft_mpi, self.e3d_to_e1d = rfft_mpi, irfft_mpi, e3d_to_e1d
        self.diff_x, self.diff_y, self.diff_z = diff_x, diff_y, diff_z

        self.wg, self.dt_save = wg, dt_save
        self.sts = f"{tps/tf:.3f}" #! the small-particle Stokes number, as it appears in the folder names
        self.tdec = max(0, int(np.ceil(-np.log10(dt_save)))) #! decimals in the time_ folder names, as save() writes them
        self.mode, self.clip = mode, clip
        self.ntransform, self.ndefault = ntransform, ndefault
        self.load_dealias, self.start_big_particle = load_dealias, start_big_particle
        self.fresh_particles = fresh_particles
        self.stb_s, self.init_name = stb_s, init_name
        self.uniquestbs, self.Nprtcl = uniquestbs, Nprtcl

    # ---------------- locating the restart folder ----------------

    def find_path(self):
        """Finds the restart folder and the time it was written at.

        modes:
          "second_last" : the second newest time_* folder (sm, clip, log)
          "all_stokes"  : the newest time_* folder holding particle states for EVERY
                          Stokes number, then one dt_save before it (big, rndm)
          "first"       : the oldest time_* folder (unused)
        Falls back to loadPath/"last" with tinit = 0 if there is no time_* folder.
        """
        paths = sorted([x for x in self.loadPath.iterdir() if "time_" in str(x)], key = os.path.getmtime)
        if len(paths) == 0 or self.start_big_particle:
            return self.loadPath/"last", 0.
        if self.mode == "second_last":
            paths = paths[-2] if len(paths) > 1 else paths[-1]
        elif self.mode == "first":
            paths = paths[0]
        elif self.mode == "all_stokes":
            tlast = [0]*len(self.stb_s)
            for jj in range(len(self.stb_s)):
                for path in paths[::-1]:
                    if self.particle_dir(jj) in os.listdir(path):
                        tlast[jj] = max(tlast[jj], float(str(path).split("time_")[-1]))
            tlast = np.min(tlast) - self.dt_save #! Loading the second last
            paths = [path for path in paths if f"time_{tlast:.{self.tdec}f}" in str(path)][0]
        else:
            raise SystemExit(f"Unknown restart mode {self.mode}")
        return paths, float(str(paths).split("time_")[-1])

    def particle_dir(self, jj):
        """The subdirectory holding one Stokes number's saved big particles."""
        name = f"{self.wg}_stb_{self.stb_s[jj]:.3f}_sts_{self.sts}"
        return name if self.init_name is None else f"{name}_init_{self.init_name[jj]}" #! rndm saves without the init name

    # ---------------- the two startup paths ----------------

    def load_fields(self, paths, loadn):
        """Reads the velocity field, and the number density if it was saved, one slab at a time."""
        uk, n = self.uk, self.n
        #! the subdirectory holding the n_{slab}.npz files, or one per Stokes number
        if n is None or not loadn: subdirs = None
        elif self.stb_s is None or self.start_big_particle: subdirs = [f"{self.wg}_sts_{self.sts}"]
        else: subdirs = [f"{self.wg}_sts_{self.sts}_stb_{stb:.3f}_init_{init}" for stb, init in zip(self.stb_s, self.init_name)]
        load_num_slabs = len([x for x in paths.iterdir() if "Fields" in str(x) and ".npz" in str(x)])
        data_per_rank = self.N//load_num_slabs
        slab_old = np.inf
        nFields = {}
        for lidx, j in enumerate(range(self.rank*self.Np, (self.rank + 1)*self.Np)): #! This rank owns these slices
            slab = j//data_per_rank
            idx = j%data_per_rank
            if slab_old != slab:
                Field = np.load(paths/f"Fields_k_{slab}.npz")
                if subdirs is not None:
                    for sub in subdirs: nFields[sub] = np.load(paths/sub/f"n_{slab}.npz")['n']
            slab_old = slab
            uk[0,:,lidx] = Field['uk'][:,idx]
            uk[1,:,lidx] = Field['vk'][:,idx]
            uk[2,:,lidx] = Field['wk'][:,idx]

            if n is None: continue
            if subdirs is None:
                if n.ndim >1: n[:,lidx] = self.ndefault
                else: n[lidx] = self.ndefault
            elif len(subdirs) == 1 and n.ndim >1: #! one saved field shared by every Stokes number
                row = nFields[subdirs[0]][idx]
                n[:,lidx] = self.ntransform(row) if self.ntransform is not None else row
            else:
                for ii, sub in enumerate(subdirs):
                    row = nFields[sub][idx]
                    if self.ntransform is not None: row = self.ntransform(row)
                    if n.ndim >1: n[ii,lidx] = row
                    else: n[lidx] = row
        return uk, n

    def fresh_fields(self):
        """Builds a random divergence-free velocity field with the initial energy einit."""
        u, uk, kint = self.u, self.uk, self.kint
        kx, ky, kz = self.kx, self.ky, self.kz
        irfft_mpi, rfft_mpi = self.irfft_mpi, self.rfft_mpi

        kinit = 31 # Wavenumber of maximum non-zero initial pressure mode.
        thu = np.random.uniform(0, self.TWO_PI,  self.k.shape)
        thv = np.random.uniform(0, self.TWO_PI,  self.k.shape)
        thw = np.random.uniform(0, self.TWO_PI,  self.k.shape)

        eprofile = kint**2*np.exp(-kint**2/2)/self.normalize
        amp = (eprofile/np.where(kint == 0, np.inf, kint**2))**0.5

        uk[0] = amp*np.exp(1j*thu)*(kint**2<kinit**2)*(kint>0)*self.dealias
        uk[1] = amp*np.exp(1j*thv)*(kint**2<kinit**2)*(kint>0)*self.dealias
        uk[2] = amp*np.exp(1j*thw)*(kint**2<kinit**2)*(kint>0)*self.dealias

        u[0] = irfft_mpi(uk[0], u[0])
        u[1] = irfft_mpi(uk[1], u[1])
        u[2] = irfft_mpi(uk[2], u[2])

        uk[0] = rfft_mpi(u[0],uk[0])
        uk[1] = rfft_mpi(u[1],uk[1])
        uk[2] = rfft_mpi(u[2],uk[2])

        trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
        uk[0] = uk[0] + self.invlap*kx*trm
        uk[1] = uk[1] + self.invlap*ky*trm
        uk[2] = uk[2] + self.invlap*kz*trm

        ek = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*self.normalize #! This is the 3D ek array
        ek_arr0 = self.comm.allreduce(self.e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array
        e0 = np.sum(ek_arr0)
        uk[0] = uk[0] *(self.einit/e0)**0.5
        uk[1] = uk[1] *(self.einit/e0)**0.5
        uk[2] = uk[2] *(self.einit/e0)**0.5

        u[0] = irfft_mpi(uk[0], u[0])
        u[1] = irfft_mpi(uk[1], u[1])
        u[2] = irfft_mpi(uk[2], u[2])

        if self.n is not None: self.n[:] = self.ndefault

    def apply_clip(self):
        """Puts the loaded or freshly built density on the same footing the evolution keeps it."""
        if self.clip is None or self.n is None: return
        if self.n.ndim >1:
            for ii in range(self.n.shape[0]): self.n[ii] = self.clip(self.n[ii]) #! the mean is a global sum, so one Stokes number at a time
        else:
            self.n[:] = self.clip(self.n)

    def initialize_fields(self, forcestart):
        """Either builds the fields from scratch or loads them. Returns the restart folder and time."""
        u, uk = self.u, self.uk
        if forcestart:
            self.fresh_fields()
            self.apply_clip()
            if self.rank == 0: print("Velocity and number density fields are set!")
            return None, 0.

        if self.rank == 0: print("Found existing simulation! Using last saved data.")
        paths, tinit = self.find_path()
        loadn = len([True for i in paths.iterdir() if "sts_" in str(i)]) > 0
        if self.rank == 0: print(f"tinit is {tinit} \nLoading data from {paths}")

        self.load_fields(paths, loadn)
        dealias = self.dealias if self.load_dealias else 1.
        u[0] = self.irfft_mpi(uk[0]*dealias, u[0])
        u[1] = self.irfft_mpi(uk[1]*dealias, u[1])
        u[2] = self.irfft_mpi(uk[2]*dealias, u[2])
        self.apply_clip()

        self.comm.Barrier()
        if self.rank == 0: print("Data loaded successfully")
        return paths, tinit

    # ---------------- the big particles ----------------

    def place_particles(self, stbs):
        """Places the big particles on the Q-criterion masks: random, Qpos, Qneg and their tails."""
        comm, u, uk, n, d = self.comm, self.u, self.uk, self.n, self.d
        kx, ky, kz, dealias = self.kx, self.ky, self.kz, self.dealias
        irfft_mpi, rfft_mpi = self.irfft_mpi, self.rfft_mpi
        diff_x, diff_y, diff_z = self.diff_x, self.diff_y, self.diff_z
        X, Nprtcl, uniquestbs = self.X, self.Nprtcl, self.uniquestbs

        vtemp = np.zeros_like(u) #! scratch, only needed while the particles are being placed
        rhs = np.zeros_like(u)
        pk = np.zeros_like(uk[0])
        ntemp = np.zeros_like(u[0])

        if self.rank == 0: print("Starting big particles from scratch")
        # ------------------ calculating Q ------------------ #
        u[0] = irfft_mpi(uk[0]*self.phase_k*dealias, u[0])
        u[1] = irfft_mpi(uk[1]*self.phase_k*dealias, u[1])
        u[2] = irfft_mpi(uk[2]*self.phase_k*dealias, u[2])

        rhs[0] = u[0]*diff_x(u[0],vtemp[0]) + u[1]*diff_y(u[0],vtemp[1]) + u[2]*diff_z(u[0],vtemp[2])
        rhs[1] = u[0]*diff_x(u[1],vtemp[0]) + u[1]*diff_y(u[1],vtemp[1]) + u[2]*diff_z(u[1],vtemp[2])
        rhs[2] = u[0]*diff_x(u[2],vtemp[0]) + u[1]*diff_y(u[2],vtemp[1]) + u[2]*diff_z(u[2],vtemp[2])

        Qk = -0.5* (1j*kx*rfft_mpi(rhs[0], pk) + 1j*ky*rfft_mpi(rhs[1], pk) +1j*kz* rfft_mpi(rhs[2], pk) )*self.conjphase_k*dealias*0.5
        #? The first 0.5 is coming from the formula. The last one due to phase shift.

        u[0] = irfft_mpi(uk[0]*dealias, u[0])
        u[1] = irfft_mpi(uk[1]*dealias, u[1])
        u[2] = irfft_mpi(uk[2]*dealias, u[2])

        rhs[0] = u[0]*diff_x(u[0],vtemp[0]) + u[1]*diff_y(u[0],vtemp[1]) + u[2]*diff_z(u[0],vtemp[2])
        rhs[1] = u[0]*diff_x(u[1],vtemp[0]) + u[1]*diff_y(u[1],vtemp[1]) + u[2]*diff_z(u[1],vtemp[2])
        rhs[2] = u[0]*diff_x(u[2],vtemp[0]) + u[1]*diff_y(u[2],vtemp[1]) + u[2]*diff_z(u[2],vtemp[2])

        Qk += -0.5* (1j*kx*rfft_mpi(rhs[0], pk) + 1j*ky*rfft_mpi(rhs[1], pk) +1j*kz* rfft_mpi(rhs[2], pk) )*dealias*0.5

        Q = irfft_mpi(Qk, ntemp)
        # ---------------------------------------------------- #

        # ------------------ creating Q masks ------------------ #
        Qmax = comm.allreduce(np.max(Q),op = MPI.MAX)
        Qmin = comm.allreduce(np.min(Q),op = MPI.MIN)
        Qpos = Q>0
        Qneg = Q<0

        def calc_maxmin10_percentile(Q, Qmin, Qmax, frac=0.1, nbins=20000):
            edges = np.linspace(Qmin, Qmax, nbins + 1)

            local_hist, _ = np.histogram(Q, bins=edges)
            hist = comm.allreduce(local_hist, op=MPI.SUM)
            total = comm.allreduce(np.sum(Q>= Qmin),op = MPI.SUM)

            cdf = np.cumsum(hist) / total
            mask = np.concatenate(([True], np.diff(cdf) > 0)) #* useful code to only take monotonically increasing function.
            fit = PchipInterpolator(cdf[mask],edges[1:][mask])

            return fit(frac), fit(1-frac)

        Qlow10,Qhigh10  = calc_maxmin10_percentile(Q,Qmin,Qmax)

        countup10 = comm.allreduce(np.sum(Q>=Qhigh10),op = MPI.SUM)
        if (countup10 <  Nprtcl).any() :   raise SystemExit(f"Need more percentile!")

        Qpos10 = Q>=Qhigh10
        Qneg10 = Q< Qlow10

        mask = (Qpos,Qneg,Qpos10,Qneg10)
        # ------------------------------------------------------ #

        for jj in range(len(self.stb_s[:uniquestbs])):
            for i in range(1,5):
                stb = stbs[jj + i*uniquestbs]
                ind1, ind2,ind3 = np.where(mask[i-1]) #! -1  to ensure the first index is 0.
                masktot = comm.allreduce(mask[i-1].sum(),op = MPI.SUM)

                nprtcl_to_choose = int((mask[i-1].sum()/masktot) * (Nprtcl[jj + i*uniquestbs])) #! The number of particles to choose in the given rank.
                tot_nprtcl_chosen = comm.allreduce(nprtcl_to_choose,op = MPI.SUM)
                remainder =  Nprtcl[jj + i*uniquestbs] - tot_nprtcl_chosen
                maskarray = comm.allgather(mask[i-1].sum())

                ranksorted = np.arange(self.num_process)[np.argsort(maskarray)[::-1]]
                if self.rank in ranksorted[:remainder]: nprtcl_to_choose += 1

                offset = comm.scan(nprtcl_to_choose, op=MPI.SUM) -  nprtcl_to_choose

                stb.coord = np.zeros((int(nprtcl_to_choose),stb.coord.shape[-1]))
                if nprtcl_to_choose >0:
                    stb.coord[:,0] = np.random.choice(X[ind1.ravel()],size = nprtcl_to_choose,replace = False)
                    stb.coord[:,1] = np.random.choice(X[ind2.ravel()],size = nprtcl_to_choose,replace = False)
                    stb.coord[:,2] = np.random.choice(X[ind3.ravel()],size = nprtcl_to_choose,replace = False)
                    stb.prtclid = offset +  np.arange(nprtcl_to_choose).reshape(-1,1)

        for  jj in range(len(self.stb_s)):
            stb = stbs[jj]
            stb.interpmat = np.zeros((stb.coord.shape[0], stb.interpmat.shape[1]))
            stb.interpmat = stb.interp_cosine(stb.coord,np.concatenate((u,u,n[jj,None,...]), axis = 0))
            stb.exterpmat = np.zeros((stb.prtclid.shape[0], 1))
            stb.coord[:,d:2*d] = stb.interpmat[:,:d]
            stb.coord[:,-1] = self.stb_s[jj]**1.5*stb.factor
            stb.update_intrinsic()

    def attach_velocity(self, stbs):
        """Gives already-placed particles the fluid velocity at their positions (rndm)."""
        u, d = self.u, self.d
        for stb in stbs:
            stb.interpmat = stb.interp_cosine(stb.coord,np.concatenate((u,u[0][None,:]),axis = 0))
            stb.coord[:,d:2*d] = stb.interpmat[:,:d]

    def load_particles(self, paths, stbs):
        """Each rank takes the saved particles that fall inside its own x-range.

        When the saved run used as many slabs as this one has ranks, the x-range is
        exactly one slab and this reads state_{rank}.npz alone.
        """
        d, X = self.d, self.X
        rstart = X[self.sx][0]
        rend = X[self.sx][-1] + self.dx
        cond = lambda x: (x[:,0]>=rstart)*(x[:,0]<rend)
        #* the rank contains particles in [rstart,rend)
        load_num_slabs = len([x for x in paths.iterdir() if "Fields" in str(x) and ".npz" in str(x)])
        data_rank_start = int(np.floor(rstart/(self.L)*load_num_slabs))
        data_rank_end = int(np.ceil(rend/(self.L)*load_num_slabs))
        for jj in range(len(self.stb_s)):
            stb = stbs[jj]
            if self.Nprtcl[jj] > 0:
                dir_b = paths / self.particle_dir(jj)
                parts = []
                pids  = []
                for r in range(data_rank_start, data_rank_end):
                    p = np.load(dir_b / f"state_{r}.npz")
                    mask = cond(p["pos"])
                    if mask.shape[0] > 0:
                        coord = np.concatenate([p["pos"][mask], p["vel"][mask], p["mass"][mask, None]], axis=1)
                        parts.append(coord)
                        pids.append(p["prtclid"][mask])

                if len(parts)>0:
                    stb.coord   = np.concatenate(parts, axis=0)
                    stb.prtclid = np.concatenate(pids,  axis=0)
                else:
                    stb.coord = np.zeros((0,2*d + 1))
                    stb.prtclid = np.zeros((0,1))
                stb.interpmat = np.zeros((stb.coord.shape[0], stb.interpmat.shape[1]))
                stb.exterpmat = np.zeros((stb.prtclid.shape[0], 1))
                stb.update_intrinsic()

    def initialize_particles(self, stbs, paths):
        """Either places the big particles from scratch or loads their saved state."""
        if self.stb_s is None: return
        if self.start_big_particle:
            if self.fresh_particles == "qcriterion": self.place_particles(stbs)
            else: self.attach_velocity(stbs)
        else:
            if paths is None: raise SystemExit("Please provide particle data path or start afresh")
            self.load_particles(paths, stbs)
        self.comm.Barrier()
        if self.rank == 0: print("Particles are set!")

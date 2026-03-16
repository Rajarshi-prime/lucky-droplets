"""
Evolves forced dns with small stokes particles in the slow manifold approximation
"""
import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, rfft, irfft
from mpi4py import MPI
from time import time
import pathlib,os,sys,h5py
from particles import MPI_particles
curr_path = pathlib.Path(__file__).parent
forcestart = True
gravity = False
start_big_particle = False
wg = "with_g" if gravity else "wo_g"
## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
isforcing = True
# viscosity_integrator = "implicit"
viscosity_integrator = "explicit"
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0.
## ---------------------------------------

## ------------- Time steps --------------
N = 256
dt =  0.256/N #! Such that increasing resolution will decrease the dt
T = 40
dt_save = 0.5
st = round(dt_save/dt) #!Savestep : Confusing
sts = 0.001
stb0 = 0.05
Nprtcl = 0
d = 3
M0 = 4e-6
rhop = 1000


"""
Typical velocities in cloud : 10 m/s = u_rms
Kolmogorov timescale : 1 sec = tau_eta
g = 9.81 m/s^2 = 9.81 *(0.1*u_rms)/(tau_eta) ~ 2.5
"""
## ---------------------------------------

## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
## -----------------------------------------------




## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

if rank ==0 : print(kx_diff.shape, ky_diff.shape, kz_diff.shape)

## -------------------------------------------------

## ----------- Parameters ----------
lp = 1 # Hyperviscosity power
# nu0 = 8.192 #! Viscosity for N = 1
# nu0 = 4.714 #! Viscosity from Pope's 256 run 
nu0 = 0.8 #! Viscosity for N = 1
m = 1.5 #! Desired kmax*eta
nu = nu0*(3*m/N)**(2*(lp - 1/3)) #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity
# nu = nu0/(N**(4/3))  #? old school new one should give the same value at m = 2.0

einit = 1*(TWO_PI)**3 # Initial energy
# einit = 0.5*(TWO_P*I)**3 # Initial energy for pope's viscosity

nshells = 1 # Number of consecutive shells to be forced
shell_no = np.arange(1,1+nshells) # the shells to be forced 

tf = (9*(m/N)**(2/3))/nu0 #* the expected kolvmogorov time scale 
tps = sts*tf # The particle timescale
g = 9.81*0.1/tf if gravity else 0 # Gravity in the simulation units
#----  Kolmogorov length scale - \eta \epsilon etc...---------

f0 = (nu0)**3 * TWO_PI**3/ nshells #! Total power input at each shells


if rank ==0 : print(f" Power input  density : {nshells*f0/TWO_PI**3} \n Viscosity : {nu}, Re : {1/nu},dt : {dt}, desired t_eta {tf}")

param = dict()
param["nu"] = nu
param["hyperviscous"] = lp
param["Initial energy"] = einit
param["Gridsize"] = N
param["Processes"] = num_process
param["Final_time"] = T
param["time_step"] = dt
param["interval of saving indices"] = st

## ---------------------------------

# savePath = pathlib.Path(f"/home/rajarshi.chattopadhyay/python/3D-DNS/data/samriddhi-tests-euler-spherical-dealias-final/N_{N}")
re = 1/nu if nu !=0 else "inf"
savePath = pathlib.Path(f"./data_cosine/forced_{isforcing}/N_{N}_Re_{re:.1f}")

if rank == 0:
    print(savePath)
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

## ------------Useful Operators-------------------

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
# kh = (kx**2 + ky**2)**0.5
dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)
invlap = dealias/np.where(lap == 0, np.inf,  lap)

# Hyperviscous operator
vis = nu*(k)**(2*lp) ## This is in Fourier Space

normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.

cond_ky = np.abs(np.round(Ky))<=N//3
cond_kz = np.abs(np.round(Kz))<=N//3
## -------------------------------------------------

stb = MPI_particles(comm, L, N, Nprtcl,sts, stb0,g,nu, tf,rhop,M0 ,d,X,Y,Z, x,y,z)
stb.to_interp(2*d+1) # u, v_s and c_s
stb.to_exterp(1)
if rank == 0: print(f"{stb.coord[:,-1]}, {stb.st},{(M0*3.5e-8/stb.factor)**(2/3.)}")

## -------------zeros arrays -----------------------
u  = np.zeros((3, Np, N, N), dtype= np.float64)
v  = np.zeros((3, Np, N, N), dtype= np.float64)
vn  = np.zeros((3, Np, N, N), dtype= np.float64)
omg= np.zeros((3, Np, N, N), dtype= np.float64)
n = u[0].copy()
nnew = n.copy()


uk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
vnk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
pk = uk[0].copy()
ek = np.zeros_like(pk, dtype = np.float64)
Pik = np.zeros_like(pk, dtype = np.float64)
ek_arr = np.zeros(Nf)
Pik_arr = np.zeros(Nf)
factor = np.zeros(Nf)
factor3d = np.zeros_like(pk,dtype= np.float64)
uknew = np.zeros_like(uk)




fk = np.zeros_like(uk)

rhsuk = np.zeros_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()

rhsu = np.zeros_like(u[0])
rhsv = rhsu.copy()
rhsw = rhsu.copy()




ku = np.zeros((3, N, Np, Nf), dtype = np.complex128)
kn = n.copy()
fc = n.copy()
fck = pk.copy()

arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf), dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)


## -----------------------------------------------------


## ------FFT + iFFT + derivative functions------- 
def rfft_mpi(u, fu):
    arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

def irfft_mpi(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis = 0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  Nf))
    u[:] = irfft2(arr_temp_fr, (N, N), axes = (1, 2))
    return u    


def diff_x(u,  u_x):
    arr_mpi_r[:] = np.moveaxis(np.reshape(u, (Np,  num_process,  Np,  N)),[0,1], [1,0])
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.moveaxis(arr_mpi_r,  [0,1], [1,0]), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis= 2), N, axis= 2)
    return u_z

def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0] 

    
    


def forcing(uk,fk):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    #? Only if you are forcing 1 or two shells 
    # for shell in shell_no:
    #     ek_arr[shell] = comm.allreduce(np.sum(ek*(kint>= shell-0.5)*(kint< shell +0.5)),op = MPI.SUM)
    
    ek_arr[:] = np.where(np.abs(ek_arr)< 1e-10,np.inf, ek_arr)
    """Change forcing starts here"""
    # Const Power Input
    factor[:] = 0.
    factor[shell_no] = f0/(2*ek_arr[shell_no])
    factor3d[:] = factor[kint]
    
    
    # # Constant shell energy
    # factor[:] = np.tanh(np.where(np.abs(ek_arr0) < 1e-10, 0, (ek_arr0/ek_arr)**0.5 - 1)) #! The factors for each shell is calculated
    # factor3d[:] = factor[kint]
    
    fk[0] = factor3d*uk[0]*dealias
    fk[1] = factor3d*uk[1]*dealias
    fk[2] = factor3d*uk[2]*dealias

    """Change forcing ends here here"""
    
    pk[:] = invlap  * (kx*fk[0] + ky*fk[1] + kz*fk[2])*dealias
    
    fk[0] = fk[0] + kx*pk
    fk[1] = fk[1] + ky*pk
    fk[2] = fk[2] + kz*pk
    
    
    return fk*isforcing*dealias
    
     
def clip_zero(x):
    """Clips negative values to zero and rescales the to conserve the mean
    """
    oldmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
    x[:] = np.clip(x,0,None)
    newmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
    
    return x*oldmean/newmean

def full_RHS(t,uk, n,sump, coord, stb,ku =ku,kn = kn,fc = fc,visc = 1,forc = 1):
    ## The RHS terms of u, v and w excluding the forcing and the hypervisocsity term 
    fk[:] = forcing(uk,fk)*forc
        
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    omg[0] = irfft_mpi(1j*(ky*uk[2] - kz*uk[1]),omg[0])
    omg[1] = irfft_mpi(1j*(kz*uk[0] - kx*uk[2]),omg[1])
    omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0]),omg[2])
    
    rhsu[:] = (omg[2]*u[1] - omg[1]*u[2])
    rhsv[:] = (omg[0]*u[2] - omg[2]*u[0])
    rhsw[:] = (omg[1]*u[0] - omg[0]*u[1]) 
    
    
    rhsuk[:]  = (rfft_mpi(rhsu, rhsuk) + fk[0])*dealias 
    rhsvk[:]  = (rfft_mpi(rhsv, rhsvk) + fk[1])*dealias 
    rhswk[:]  = (rfft_mpi(rhsw, rhswk) + fk[2])*dealias 
    
    ## The pressure term
    pk[:] = 1j*invlap  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

    ## The RHS term with the pressure   
    ku[0] = rhsuk - 1j*kx*pk - nu*((-lap)**lp)*uk[0]*isexplicit * visc
    ku[1] = rhsvk - 1j*ky*pk - nu*((-lap)**lp)*uk[1]*isexplicit * visc
    ku[2] = rhswk - 1j*kz*pk - nu*((-lap)**lp)*uk[2]*isexplicit * visc
    
    #the rhs for the number density
    v[0] = irfft_mpi((uk[0] - tps*(-1j*kx*pk - nu*((-lap)**lp)*uk[0]*isexplicit * visc))*dealias, v[0])
    v[1] = irfft_mpi((uk[1] - tps*(-1j*ky*pk - nu*((-lap)**lp)*uk[1]*isexplicit * visc))*dealias, v[1])
    v[2] = irfft_mpi((uk[2] - tps*(-1j*kz*pk - nu*((-lap)**lp)*uk[2]*isexplicit * visc))*dealias, v[2]) -tps*g
    

    sump, kp, fc = stb.pRHS(t, coord, u,v,n,fc,sump)
    fck[:] = rfft_mpi(fc, fck)*dealias
    vn[:] = n[None,...]*v
    vnk[0] = rfft_mpi(vn[0],vnk[0])*dealias
    vnk[1] = rfft_mpi(vn[1],vnk[1])*dealias
    vnk[2] = rfft_mpi(vn[2],vnk[2])*dealias
    kn[:] = irfft_mpi((-1j* (kx*vnk[0] + ky*vnk[1] + kz*vnk[2]) - fck)*dealias, kn) #! fck is the growth rate. so the - sign
        
    return ku,kn,kp,sump

    
def RK4(t,h,stb, uk,n,uknew = uknew, nnew = nnew):
    """Template on how to evolve the particle + flow system"""
    sump = 1.0*stb.coord
    uknew[:] = 1.0*uk
    nnew[:] = 1.0*n
    
    ku[:],kn[:],kp,sump = full_RHS(t,uk, clip_zero(n),sump,stb.coord,stb)
    sump += h/6.0*kp
    uknew += h/6.0*ku
    nnew += h/6.0*kn
    

    ku[:],kn[:],kp,sump = full_RHS(t + h/2,uk + ku*h/2, clip_zero(n + kn*h/2),sump, stb.coord + h/2 *kp, stb)
    sump += h/3.0*kp
    uknew += h/3.0*ku
    nnew += h/3.0*kn
    
    ku[:],kn[:],kp,sump = full_RHS(t + h/2,uk + ku*h/2, clip_zero(n + kn*h/2),sump, stb.coord + h/2 *kp, stb)
    sump += h/3.0*kp
    uknew += h/3.0*ku
    nnew += h/3.0*kn
    
    ku[:],kn[:],kp,sump = full_RHS(t + h,uk + ku*h, clip_zero(n + kn*h),sump, stb.coord + h *kp, stb)
    sump += h/6.0*kp
    uknew += h/6.0*ku
    nnew += h/6.0*kn
    
    return sump*1.0, uknew,nnew


## -----------------------------------------------------------


## ---------------- Saving data + energy + Showing total energy ---------------------
def load_trunc(x):
    x1 = np.zeros((*x.shape[:-2],N,Nf),dtype = np.complex128)
    x1[...,cond_ky,:x.shape[-1]] = x.copy()
    return irfftn(x1,(N,N), axes = (-2,-1))
    
def load_npz(paths,uk,n,tps = tps, tf = tf,loadn=  True): #! Rewrite
    load_num_slabs = len([x for x in (paths).iterdir() if "Fields" in str(x) and ".npz" in str(x)])
    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        
        # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
        
        """Loading the truncated data"""
        if slab_old != slab:  
            Field = np.load(paths/f"Fields_k_{slab}.npz")
            if loadn== True:
                if start_big_particle: 
                    nField = np.load(paths/f"{wg}_sts_{tps/tf:.3f}/n_{slab}.npz")
                else: 
                    nField = np.load(paths/f"{wg}_sts_{tps/tf:.3f}_stb_{stb0:.3f}/n_{slab}.npz")
            
        slab_old = slab
        uk[0,:,lidx] = Field['uk'][:,idx]
        uk[1,:,lidx] = Field['vk'][:,idx]
        uk[2,:,lidx] = Field['wk'][:,idx]
        if loadn== True: n[lidx] = nField['n'][idx]
        else: n[lidx] = 1.0

        """Loading the OG data"""
        # if slab_old != slab:  Field = np.load(paths/f"Fields_{slab}.npz")
        # slab_old = slab
        # u[0,lidx] = Field['u'][idx]
        # u[1,lidx] = Field['v'][idx]
        # u[2,lidx] = Field['w'][idx]  
    return uk, n
    
    
def load_hdf5(paths, u, n,tps =tps, tf = tf):
    with h5py.File(paths/'Fields.hdf5','r+', driver = 'mpio', comm = comm) as f:
        u[0] = f['u'][sx,...][:]
        u[1] = f['v'][sx,...][:]
        u[2] = f['w'][sx,...][:]
        n[:] = f[f'/st_{tps/tf:.3f}/n'][sx,...][:]
    
    return u,n

def save(i,uk,n,stb,tf = tf, tps = tps): 
    
    # div = diff_x(u[0], rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw)
    # if rank == 0: print(f"Rank {rank} has divergence {np.sum(np.abs(div))}")
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array

    ku[:],_,_,_ = full_RHS(t,uk, n,stb.coord, stb.coord, stb,visc = 0,forc = 0)
    Pik[:] = np.real(np.conjugate(uk[0])*ku[0]+np.conjugate(uk[1])*ku[1]+ np.conjugate(uk[2])*ku[2])*dealias*normalize
    Pik_arr[:] = comm.allreduce(e3d_to_e1d(Pik),op = MPI.SUM)
    Pik_arr[:] = np.cumsum(Pik_arr[::-1])[::-1]
    
    ek_arr[:] = 0.0
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    # ----------- ----------------------------
    #                 Saving the data (field)
    # ----------- ----------------------------
    if Nprtcl > 0: new_dir = savePath/f"time_{t[i]:.1f}/{wg}_sts_{tps/tf:.3f}_stb_{stb0:.3f}"
    else: new_dir = savePath/f"last/{wg}_sts_{tps/tf:.3f}"
    try: new_dir.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass
    new_dir = new_dir.parent #* coming back to the old directory
    comm.Barrier()

    np.savez_compressed(f"{new_dir}/Fields_k_{rank}",uk = uk[0],vk = uk[1],wk = uk[2])
    np.savez_compressed(f"{new_dir}/Energy_spectrum",ek = ek_arr)
    np.savez_compressed(f"{new_dir}/Flux_spectrum",Pik = Pik_arr)
    if Nprtcl > 0: np.savez_compressed(f"{new_dir}/{wg}_sts_{tps/tf:.3f}_stb_{stb0:.3f}/n_{rank}",n = n)
    else: np.savez_compressed(f"{new_dir}/{wg}_sts_{tps/tf:.3f}/n_{rank}",n = n)
    
    if Nprtcl> 0:
        new_dir = savePath/f"time_{t[i]:.1f}/{wg}_stb_{stb0:.3f}_sts_{tps/tf:.3f}"
        try: new_dir.mkdir(parents=True,  exist_ok=True)
        except FileExistsError: pass
        comm.Barrier()
        stb.coord,[stb.interpmat,stb.exterpmat,stb.prtclid] = stb.send(stb.coord,[stb.interpmat,stb.exterpmat,stb.prtclid])
        stb.st = (stb.coord[:,-1]/stb.factor)**(2/3.)
        stb.interpmat = stb.interp_cosine(stb.coord,np.concatenate((u,u,n[None,...]), axis = 0))
        np.savez_compressed(new_dir/f"state_{rank}.npz",pos= stb.coord[:,:d],vel = stb.coord[:,d:2*d], mass = stb.coord[:,-1],prtclid = stb.prtclid, umat = stb.interpmat[:,:d])
    
    comm.Barrier()
    
    # ----------- ----------------------------
    #          Calculating and printing
    # ----------- ----------------------------
    eng1 = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2)*dx*dy*dz), op = MPI.SUM)
    eng2 = np.sum(ek_arr)
    nmin = comm.allreduce(np.min(n), op = MPI.MIN)
    nmax = comm.allreduce(np.max(n), op = MPI.MAX)
    nmean = comm.allreduce(np.sum(n), op = MPI.SUM)/N**3
    divmax = comm.allreduce(np.max(np.abs(diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
    #! Needs to be changed 
    # # dissp = -nu*comm.allreduce(np.sum((kc**(2*lp)*(np.abs(uk[0])**2 + np.abs(uk[1])**2) +sin_to_cos( ks**(2*lp)*(np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)))), op = MPI.SUM)
    if rank == 0:
        print( "#----------------------------","\n",f"Energy at time {t[i]} is : {eng1}, {eng2}","\n","#----------------------------")
        print(f"Maximum divergence {divmax}")
        print(f"n mean, max ,min : {nmean, nmax, nmin}")
        # print( "#----------------------------","\n",f"Total dissipation at time {t[i]} is : {dissp}","\n","#----------------------------")

    return "Done!"    
    
    
    
    
def save_hdf5(i,uk,n,tf = tf, tps = tps):
    
    # div = diff_x(u[0], rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw)
    # if rank == 0: print(f"Rank {rank} has divergence {np.sum(np.abs(div))}")
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    
    k1u[:],_ = RHS(uk,n, k1u,k1n, visc = 0,forc = 0.0)
    Pik[:] = np.real(np.conjugate(uk[0])*k1u[0]+np.conjugate(uk[1])*k1u[1]+ np.conjugate(uk[2])*k1u[2])*dealias*normalize
    Pik_arr[:] = comm.allreduce(e3d_to_e1d(Pik),op = MPI.SUM)
    Pik_arr[:] = np.cumsum(Pik_arr[::-1])[::-1]
    
    ek_arr[:] = 0.0
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    # ----------- ----------------------------
    #                 Saving the data (field)
    # ----------- ----------------------------
    new_dir = savePath/f"time_{t[i]:.1f}"
    try: new_dir.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass
    comm.Barrier()

    with h5py.File(new_dir/'Fields.hdf5','w', driver = 'mpio', comm = comm) as f:
        f.create_dataset('u', (N,N,N),dtype = np.float64)
        f.create_dataset('v', (N,N,N),dtype = np.float64)
        f.create_dataset('w', (N,N,N),dtype = np.float64)
        f.create_dataset('Energy_Spectrum', data = ek_arr,dtype = np.float64)
        f.create_dataset('Flux_Spectrum', data = Pik_arr,dtype = np.float64)
        f['u'][sx,...] = u[0]
        f['v'][sx,...] = u[1]
        f['w'][sx,...] = u[2]
        f.attrs['nu'] = nu
        f.attrs['Power input'] = f0 /TWO_PI**3
        f.attrs['eta'] = m/(N//3)
        f.attrs['t_eta'] = tf
        f.attrs['forcing'] = f"Isotropic with const power input in shells {shell_no}"
        f.attrs['N'] = N
    
        ndat= f.create_dataset(f'/st_{tp/tf:.3f}/n', (N,N,N),dtype = np.float64)
        ndat[sx,...] = n
        ndat.attrs['t_f'] = tf
        ndat.attrs['t_p'] = tp
        

    comm.Barrier()
    
    # ----------- ----------------------------
    #          Calculating and printing
    # ----------- ----------------------------
    eng1 = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2)*dx*dy*dz), op = MPI.SUM)
    eng2 = np.sum(ek_arr)
    nmin = comm.allreduce(np.min(n), op = MPI.MIN)
    nmax = comm.allreduce(np.max(n), op = MPI.MAX)
    nmean = comm.allreduce(np.sum(n), op = MPI.SUM)/N**3
    divmax = comm.allreduce(np.max(np.abs(diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
    #! Needs to be changed 
    # # dissp = -nu*comm.allreduce(np.sum((kc**(2*lp)*(np.abs(uk[0])**2 + np.abs(uk[1])**2) +sin_to_cos( ks**(2*lp)*(np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)))), op = MPI.SUM)
    if rank == 0:
        print( "#----------------------------","\n",f"Energy at time {t[i]} is : {eng1}, {eng2}","\n","#----------------------------")
        print(f"Maximum divergence {divmax}")
        print(f"n mean, max ,min : {nmean, nmax, nmin}")
        # print( "#----------------------------","\n",f"Total dissipation at time {t[i]} is : {dissp}","\n","#----------------------------")
    comm.Barrier()
    return "Done!"        
## -------------------------------------------------    
    
## ------------- Evolving the system ----------------- 
def evolve_and_save(t,  u,n): 

    comm.Barrier()
    h = t[1] - t[0]
    
    if viscosity_integrator == "implicit": hypervisc= dealias*(1. + h*vis)**(-1)
    else: hypervisc = 1.
    
    
    t3  = time()
    calc_time = 0
    uk[0] = rfft_mpi(u[0], uk[0])*dealias
    uk[1] = rfft_mpi(u[1], uk[1])*dealias
    uk[2] = rfft_mpi(u[2], uk[2])*dealias
    
    for i in range(t.size-1):
        calc_time += time() - t3
        if rank == 0:  print(f"step {i} in time {time() - t3}", end= '\r')
        ## ------------- saving the data -------------------- ##
        if i % st ==0 : save(i,uk,n,stb)
        ## -------------------------------------------------- ##
        t3 = time()
        comm.Barrier()
        
        
        
        stb.coord, uknew[:],nnew[:] = RK4(t,h,stb, uk,n)
        stb.update_intrinsic()
        n[:] = clip_zero(nnew)
        uknew[:] = (uknew)*hypervisc
        
        # ------------------- ensuring n in non-negative ------------------- #

        # pk[:] = rfft_mpi(n,pk)*dealias
        # n[:] = irfft_mpi(pk,n)
        # ------------------------------------------------------------------- #
        
        
       
        
        """ Enforcing the reality condition """
        u[0] = irfft_mpi(uknew[0], u[0])
        u[1] = irfft_mpi(uknew[1], u[1])
        u[2] = irfft_mpi(uknew[2], u[2])
        
        uk[0] = rfft_mpi(u[0],uk[0])
        uk[1] = rfft_mpi(u[1],uk[1])
        uk[2] = rfft_mpi(u[2],uk[2])
        
        
        """Enforcing div free conditon"""
        pk[:] = invlap  * (kx*uk[0] + ky*uk[1] + kz*uk[2])
        uk[0] = uk[0] + kx*pk
        uk[1] = uk[1] + ky*pk
        uk[2] = uk[2] + kz*pk
  
        #! Althought RHS should obey the above two conditions, the rfft adds dependent degrees of freedom for kz = 0 that is evolved separately. Additionally, in some extreme cases, numerical errors can build up. We add these lines to avoid them.
         
        
        
 
        ## -------------------------------------------------------
        if uk.max() > 100*N**3 : 
            print("Threshold exceeded at time", t[i+1], "Code about to be terminated")
            comm.Abort()
        
        
        comm.Barrier()
        
    ## ---------- Saving the final data ------------
    save(i+1, uk,n,stb)
    if rank ==0: print(f"average calculation time per step {calc_time/(t.size-1)}")
    ## ---------------------------------------------

    

## --------------- Initializing ---------------------


"""Structure 
If there exists a folder with the parameter names and has time folders in it. 
Load the parameters from parameters.txt
If the parameters match the current code parameters enter the last time folder.
Finally load the data.
If not start from scratch."""



#! Modify the loading process!

if not forcestart:
    ## ------------------------- Beginning from existing data -------------------------
    if rank ==0 : print("Found existing simulation! Using last saved data.")
    """Loading the data from the last time  """    
    
    paths = sorted([x for x in pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_{isforcing}/N_{N}_Re_{re:.1f}").iterdir() if "time_" in str(x)], key=os.path.getmtime)
    if len(paths) >0: 
        paths = paths[-2]
        tinit = float(str(paths).split("time_")[-1])
    else: 
        paths = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_{isforcing}/N_{N}_Re_{re:.1f}/last")
        tinit = 0.
    issts = [True for i in paths.iterdir() if "sts_" in str(i)]
    loadn = True if len(issts) > 0 else False
    # ------------------- specifying manually ------------------- #
    # tinit = 20.0
    # paths = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_{isforcing}/N_{N}_Re_{re:.1f}/time_{tinit:.1f}")
    # ------------------------------------------------------------ #
    

    if rank ==0 : print(f"Loading data from {paths}")
    
    uk[:],n[:] = load_npz(paths,uk, n,loadn = loadn)
    
    # u,n = load_hdf5(paths,u,n)
    u[0] = irfft_mpi(uk[0]*dealias, u[0])
    u[1] = irfft_mpi(uk[1]*dealias, u[1])
    u[2] = irfft_mpi(uk[2]*dealias, u[2])
    if not start_big_particle: #! Change this to different ranks.
        prtcl_data = np.load(paths/f"{wg}_stb_{stb0:.3f}_sts_{tps/tf:.3f}/state_{rank}")
        stb.coord = np.zeros((prtcl_data["pos"].shape, 2*d + 1))
        stb.coord[:,:d] = prtcl_data["pos"]
        stb.coord[:,d:2*d] = prtcl_data["vel"]
        stb.coord[:,-1] = prtcl_data["mass"]
        stb.prtclid = prtcl_data["prtclid"]
        stb.interpmat = np.zeros((stb.prtclid.shape[0], 2*d+1))
        stb.exterpmat = np.zeros((stb.prtclid.shape[0], 1))
    

    del paths
    comm.Barrier()
    if rank ==0: print("Data loaded successfully")
    
    

if forcestart:
    ## ---------------------- Beginning from start ----------------------------------

    kinit = 31 # Wavenumber of maximum non-zero initial pressure mode.    
    thu = np.random.uniform(0, TWO_PI,  k.shape)
    thv = np.random.uniform(0, TWO_PI,  k.shape)
    thw = np.random.uniform(0, TWO_PI,  k.shape)

    # eprofile = 1/np.where(kint ==0, np.inf,kint**(2.0))/normalize
    eprofile = kint**2*np.exp(-kint**2/2)/normalize
    
    
    amp = (eprofile/np.where(kint == 0, np.inf, kint**2))**0.5
    
    uk[0] = amp*np.exp(1j*thu)*(kint**2<kinit**2)*(kint>0)*dealias
    uk[1] = amp*np.exp(1j*thv)*(kint**2<kinit**2)*(kint>0)*dealias
    uk[2] = amp*np.exp(1j*thw)*(kint**2<kinit**2)*(kint>0)*dealias
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    uk[0] = rfft_mpi(u[0],uk[0])
    uk[1] = rfft_mpi(u[1],uk[1])
    uk[2] = rfft_mpi(u[2],uk[2])
    
    trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
    uk[0] = uk[0] + invlap*kx*trm
    uk[1] = uk[1] + invlap*ky*trm
    uk[2] = uk[2] + invlap*kz*trm
    
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
    # if rank ==0: print(ek_arr0, np.sum(ek_arr0))
    e0 = np.sum(ek_arr0)
    uk[0] = uk[0] *(einit/e0)**0.5
    uk[1] = uk[1] *(einit/e0)**0.5
    uk[2] = uk[2] *(einit/e0)**0.5
    
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    tinit = 0.
    n = np.ones_like(n)
    


ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
if rank ==0: print(ek_arr0, np.sum(ek_arr0))
ek_arr0[0:shell_no[0]] = 0.
ek_arr0[shell_no[-1] + 1:] = 0.


divmax = comm.allreduce(np.max(np.abs( diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
if rank ==0 : print(f" max divergence {divmax}")

#----------------- The initial energy ------------------
e0 = comm.allreduce(0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + (u[2]**2)),op = MPI.SUM)
nmean = comm.allreduce(n.sum(),op =MPI.SUM)/N**3
nmax = comm.allreduce(n.max(),op =MPI.MAX)
nmin = comm.allreduce(n.min(),op =MPI.MIN)
if rank ==0 : print(f"Initial Physical space energy: {e0}, mean max and min of n  {nmean}, {nmax} , {nmin}")
#-------------------------------------------------------
# raise SystemExit

# --------------------------------------------------

## ----- executing the code -------------------------
t = np.arange(tinit,T+ 0.5*dt, dt)
# t = np.arange(tinit,10*dt, dt)
# print(len(t))
t1 = time()
evolve_and_save(t,u,n)
t2 = time() - t1 
# --------------------------------------------------
if rank ==0: print(t2)
## --------- saving the calculation time -----------
if rank ==0: 
    with open(savePath/f"calcTime.txt","a") as f:
        f.write(str({f"time taken to run from {tinit} to {T} is": t2}))
## --------------------------------------------------


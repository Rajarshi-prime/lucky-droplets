"""
Postprocessing the 10% of the luckiest lucky droplets.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib,os,h5py,json
from scipy.fft import fftfreq,rfftn, irfftn
from scipy.interpolate import CubicSpline, splrep,splev,splder
from scipy.optimize import newton,brentq
from scipy.interpolate import PchipInterpolator
from mpi4py import MPI
from particles import MPI_particles
# %%
N =256
num_process = 256
Np = N//num_process
dt_save = 0.5
gravity = True
wg = "with_g" if gravity else "wo_g"
datapath = lambda t,sts,stb,name: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/time_{t:.1f}/{wg}_sts_{sts:.3f}_stb_{stb:.3f}_init_{name}")
sts = 0.001
stb_s = [0.017,0.017/4, 0.017*4,0.017*6.35]
names = ['random','Qpos','Qneg','Qpos_high','Qneg_high']
stb_s.sort()
stb_s= np.array(stb_s)
Nprtcl = np.round(8192*(0.017/stb_s)**1.5).astype(np.int32) #! 10240 0.017 St particles.
M0 = 7.2e-4
rhop = 1000
PI = np.pi
Twith_PI = 2*PI
m = 2
kmax = (N*2**0.5)//3
eta = m/kmax
rs = eta*(9*sts/2/rhop)**0.5 
nprtcls0 = 3*M0*Twith_PI**3/(rhop * 4 * PI *rs**3)
rs,Twith_PI/N,nprtcls0
nu0 = 0.59
lp = 1
nu = nu0*(eta)**(2*(lp - 1/3)) 
tf =eta**2/nu
#%%
tf
#%%
stb_s,Nprtcl
#%%
cols = ["#178aac","#8417ac","#ac3917","#40ac17"]
#%%
PI = np.pi
Twith_PI = 2*PI
Nf = N//2 + 1
L = Twith_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*Twith_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky,  Kz,  indexing = 'ij')
k = (kx**2 + ky**2 + kz**2)**0.5
shells = np.arange(-0.5,Nf, 1.)
normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/Twith_PI**3),2/(N**6/Twith_PI**3))

#%%
# names = ["","highn/","highhighn/","synthetic/"]
M0 = 7.2e-4
prtcl_path = lambda t,stb,name = "random": pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/time_{t:.1f}/{wg}_stb_{stb:.3f}_sts_0.001_init_{name}/")

paths = sorted([prtcl_path(0,stb_s[0]).parent.parent/f"{i}" for i in os.listdir(prtcl_path(0,stb_s[0]).parent.parent) if "time_" in i], key = lambda x: float(str(x).split("time_")[-1]))
#%%
tlast = [0]*len(stb_s)
for jj in range(len(stb_s)):
    for path in paths[::-1]:

        if len([i for i in os.listdir(path) if f"{wg}_stb_{stb_s[jj]:.3f}_sts_{sts}" in i]) != 0:

            tlast[jj] = max(tlast[jj], float(str(path).split("time_")[-1]))
tlast = np.min(tlast) #! the last


times = np.arange(0,tlast + 0.5*dt_save, dt_save)
# times = np.arange(times[0], times[-1] + 0.5, 1) #! Reducing data loading.

#%%
def load_u(paths):
    if type(paths) == str: paths = pathlib.Path(paths)
    uk = np.zeros((3,N,N,Nf), dtype = np.complex128)
    load_num_slabs = len([x for x in (paths).iterdir() if "Fields" in str(x) and ".npz" in str(x)])
    data_per_rank = N//load_num_slabs
    rank = 0
    rank_data = range(0,N) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        
        # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
        
        """Loading the truncated data"""
        if slab_old != slab:  
            Field = np.load(paths/f"Fields_k_{slab}.npz")


        slab_old = slab
        uk[0,:,lidx] = Field['uk'][:,idx]
        uk[1,:,lidx] = Field['vk'][:,idx]
        uk[2,:,lidx] = Field['wk'][:,idx]
    return irfftn(uk, s = (N,N,N), axes = (-3,-2,-1))
#%%
u = load_u("/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/last")
#%%
0.5*(u**2).sum()*dx*dy*dz
#%%
urms = (2/3.*np.mean(u**2)/2)**0.5
epsilon =  (nu0)**3 
lmbd = (15*nu/epsilon)**0.5*urms
re_lmbd = urms*lmbd/nu
re_lmbd,urms, epsilon, nu

#%%
def load_n(path):
    n = np.zeros((N,N,N))
    slabs = len([f for f in os.listdir(path) if "n_" in f])
    for i in range(slabs):
        n[i*(N//slabs):(i+1)*(N//slabs)] = np.load(path/f"n_{i}.npz")['n']
    return n

def calc_invariants(u):
    A = irfftn(1j*np.einsum('j...,i...->ij...',np.array([kx,ky,kz]),rfftn(u,axes = (-3,-2,-1))),s = (N,N,N),axes = (-3,-2,-1))
    aa = np.einsum('ij...,ij...->...',A,A)
    at = np.einsum('ij...,ji...->...',A,A)
    R = -np.einsum('ij...,jk...,ki...->...',A,A,A,optimize = True)/3.
    return aa - at, 0.5*(aa + at), R #! A is symmetric plus antisymmetric, so these two contractions give omega^2 and S_ij S_ij

# %%
def load_instant(stb, init,time):
    path = prtcl_path(time,stb,init)
    num_process = len([i for i in os.listdir(path) if "state_" in i])
    print(f"St = {stb:.3f}, Time = {time:.1f}, name= {init},num_process= {num_process}",end = "\r")
    mass = np.zeros(0)
    urel = np.zeros(0)
    pos = np.zeros((0,3))
    mass_next = np.zeros(0)
    id = np.zeros(0)
    id_next = np.zeros(0)
    for rank in range(num_process):
        data = np.load(prtcl_path(time,stb,init)/f"state_{rank}.npz")
        try:
            data_next = np.load(prtcl_path(time + dt_save,stb,init)/f"state_{rank}.npz")
        except FileNotFoundError:
            data_next = np.load(prtcl_path(time,stb,init)/f"state_{rank}.npz")
        id = np.concatenate((id,data['prtclid'].ravel()))
        mass = np.concatenate((mass,data['mass'].ravel()))
        urel = np.concatenate((urel, np.linalg.norm(data['umat'] - data['vel'], axis = 1).ravel()))
        pos = np.concatenate((pos,data['pos']))

        id_next= np.concatenate((id_next,data_next['prtclid'].ravel()))
        mass_next = np.concatenate((mass_next,data_next['mass'].ravel()))
        
    sortedidx = np.argsort(id)
    mass = mass[sortedidx]
    urel = urel[sortedidx]
    pos = pos[sortedidx]

    sortedidx_next = np.argsort(id_next)
    mass_next = mass_next[sortedidx_next]
    dmass = mass_next - mass
    
    return mass,urel, dmass, pos
#%%
all_init = [[load_instant(stb,name,0.0)[0].sum() for stb in stb_s] for name in names]
all_init
#%%

def mask_max_growers(stb,init,t = times,frac = 0.9):
    initmass = load_instant(stb,init,t[0])[0]
    print(f"Standard deviation for initial mass is : {initmass.std()}")
    finalmass = load_instant(stb,init,t[-1])[0]
    edges = np.linspace(finalmass.min(),finalmass.max(),1001)
    hist = np.histogram(finalmass, bins=edges)[0]/len(finalmass)

    cdf = np.cumsum(hist)
    mask = np.concatenate(([True],np.diff(cdf)>0))


    fit = PchipInterpolator(cdf[mask],edges[1:][mask])

    masshigh10 = fit(frac)
    return finalmass>=masshigh10
#%%
mask_max_growers(0.004,"Qneg").sum()
#%%
urel_bins = np.linspace(0,10,1001)
om2_bins = np.logspace(-2,3,101)/tf**2
s2_bins = np.logspace(-2,3,101)/tf**2
dissip_bins = 2*nu*s2_bins
n_bins = np.logspace(-2,2,101)
Q_bins = np.linspace(-50,50,101)/tf**2
R_bins = np.linspace(-50,50,101)/tf**3
qbins = [Q_bins,om2_bins,s2_bins,R_bins,dissip_bins,n_bins,urel_bins]
qnames = ["q","om2","s2","R","dissip","n","urel"] #! same order as qbins

comm = MPI.COMM_WORLD
def interp_quantities(prtcl,pos,fields,urel):
    om2,s2,R,n = prtcl.interp_cosine(pos,fields).T
    return np.stack((om2/4 - s2/2, om2, s2, R, 2*nu*s2, n, urel),axis = -1) #! Q and dissip are linear in om2 and s2, so they need no separate interpolation

def load_series(times,stb_s,names,qbins = qbins):
    mass_mean = np.zeros((len(stb_s),len(names),2,len(times))) #! all and top10
    mass_std = np.zeros((len(stb_s),len(names),2,len(times)))
    qmean = np.zeros((len(stb_s),len(names),2,len(times),len(qbins)))
    nmean = np.zeros((len(stb_s),len(names),len(times))) #! the field mean, so it carries no all/top10 axis
    cnt = [np.zeros((len(stb_s),len(names),2,len(b)-1)) for b in qbins]
    tot = [np.zeros((len(stb_s),len(names),2,len(b)-1)) for b in qbins]
    fields = np.zeros((4,N,N,N))
    prtcls = [MPI_particles(comm, L, N, Nprtcl[kk],sts, stb,0, nu, tf,rhop,M0,3,X,Y,Z, x,y,z) for kk,stb in enumerate(stb_s)]
    for prtcl in prtcls: prtcl.to_interp(4)
    masks = [[mask_max_growers(stb,init,t = times) for init in names] for stb in stb_s]
    for i,t in enumerate(times):
        fields[:3] = calc_invariants(load_u(prtcl_path(t,stb_s[0],names[0]).parent)) #! the flow is shared by every Stokes number
        for kk,stb in enumerate(stb_s):
            for nami,init in enumerate(names):
                fields[3] = load_n(datapath(t,sts,stb,init))
                nmean[kk,nami,i] = np.mean(fields[3])
                mass,urel,dmass,pos = load_instant(stb,init,t)
                q = interp_quantities(prtcls[kk],pos,fields,urel)
                for w,mask in enumerate((slice(None), masks[kk][nami])):
                    mass_mean[kk,nami,w,i] = mass[mask].mean()
                    mass_std[kk,nami,w,i] = mass[mask].std()
                    qmean[kk,nami,w,i] = q[mask].mean(axis = 0)
                    if i < len(times)-1:
                        for v in range(len(qbins)):
                            cnt[v][kk,nami,w] += np.histogram(q[mask,v],bins = qbins[v])[0]
                            tot[v][kk,nami,w] += np.histogram(q[mask,v],bins = qbins[v],weights = dmass[mask])[0]
    qvals = [0.5*(b[1:] + b[:-1]) for b in qbins]
    prof = np.empty((len(stb_s),len(names),2,len(qbins)),dtype = object)
    vals = np.empty(prof.shape,dtype = object)
    for kk,nami,w,v in np.ndindex(prof.shape):
        cond = cnt[v][kk,nami,w]>0 #! drop the bins with no droplets in them
        prof[kk,nami,w,v] = tot[v][kk,nami,w][cond]/cnt[v][kk,nami,w][cond]
        vals[kk,nami,w,v] = qvals[v][cond]
    return mass_mean, mass_std, qmean, nmean, prof, vals
#%%
db = {}
mass_mean,mass_std,qmean,nmean,prof,vals = load_series(times,stb_s,names)
for kk,stb in enumerate(stb_s):
    for nami,init in enumerate(names):
        for w,which in enumerate(["all","top10"]):
            key = f"{wg}/{stb:.3f}/{init}/{which}"
            db[key] = {"t":times.tolist(),"mean":mass_mean[kk,nami,w].tolist(),"std":mass_std[kk,nami,w].tolist(),"n":int(Nprtcl[kk]),"nmean":nmean[kk,nami].tolist()}
            db[key].update({f"qmean_{qn}":qmean[kk,nami,w,:,iq].tolist() for iq,qn in enumerate(qnames)})
            db[key].update({f"dmass_{qn}":p.tolist() for qn,p in zip(qnames,prof[kk,nami,w])})
            db[key].update({f"vals_{qn}":p.tolist() for qn,p in zip(qnames,vals[kk,nami,w])})
with open(f"mass_data_{wg}.json", "w") as f:
    json.dump(db, f)

raise SystemExit     
# %%

"""
Postprocessing the 10% of the luckiest lucky droplets.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib,os,h5py,json,sys
from scipy.fft import fftfreq,rfftn, irfftn
from scipy.interpolate import CubicSpline, splrep,splev,splder
from scipy.optimize import newton,brentq
from scipy.interpolate import PchipInterpolator
# %%
N =256
num_process = 256
Np = N//num_process
dt_save = 0.1
gravity = False  if int(sys.argv[-1]) ==0 else True
wg = "with_g" if gravity else "wo_g"
print(f'Gravity is {gravity}')
datapath = lambda t,sts,stb,name: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_469.8/time_{t:.1f}/{wg}_sts_{sts:.3f}_stb_{stb:.3f}_init_{name}")
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
nu0 = 0.50
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
prtcl_path = lambda t,stb,name = "random": pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_469.8/time_{t:.1f}/{wg}_stb_{stb:.3f}_sts_0.001_init_{name}/")

paths = sorted([prtcl_path(0,stb_s[0]).parent.parent/f"{i}" for i in os.listdir(prtcl_path(0,stb_s[0]).parent.parent) if "time_" in i], key = lambda x: float(str(x).split("time_")[-1]))
#%%
tlast = [0]*len(stb_s)
for jj in range(len(stb_s)):
    for path in paths[::-1]:

        if len([i for i in os.listdir(path) if f"{wg}_stb_{stb_s[jj]:.3f}_sts_{sts}" in i]) != 0:

            tlast[jj] = max(tlast[jj], float(str(path).split("time_")[-1]))
tlast = np.min(tlast)


times = np.arange(0,tlast + 0.5*dt_save, dt_save)
# times = np.arange(times[0], times[-1] + 0.5, 1) #! Reducing data loading.
# times = times[:100]
times
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
u = load_u("/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_469.8/last")
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
    aa = np.einsum('ij...,ji...->...',A,A)
    at = np.einsum('ij...,ij...->...',A,A)
    R = -np.einsum('ij...,jk...,ki...->...',A,A,A,optimize = True)/3.
    return at - aa, 0.5*(aa + at), R #! aa is tr(A^2) and at is tr(A A^T), so at - aa is omega^2

# %%
def load_instant(stb, init, time):
    path = prtcl_path(time, stb, init)
    num_process = len([i for i in os.listdir(path) if "state_" in i])
    print(f"St = {stb:.3f}, Time = {time:.1f}, name= {init},num_process= {num_process}", end="\r")
    mass = np.zeros(0)
    urel = np.zeros(0)
    pos = np.zeros((0, 3))
    om2 = np.zeros(0)
    s2 = np.zeros(0)
    R = np.zeros(0)
    n = np.zeros(0)
    mass_next = np.zeros(0)
    id = np.zeros(0)
    id_next = np.zeros(0)
    for rank in range(num_process):
        data = np.load(prtcl_path(time,stb,init)/f"state_{rank}.npz")
        try: 
            data_next = np.load(prtcl_path(time + dt_save,stb,init)/f"state_{rank}.npz")
        except FileNotFoundError:
            data_next = np.load(prtcl_path(time, stb, init) / f"state_{rank}.npz")
        id = np.concatenate((id, data['prtclid'].ravel()))
        mass = np.concatenate((mass, data['mass'].ravel()))
        urel = np.concatenate((urel, np.linalg.norm(data['umat'] - data['vel'], axis=1).ravel()))
        pos = np.concatenate((pos, data['pos']))
        om2 = np.concatenate((om2, data['om2'].ravel()))
        s2 = np.concatenate((s2, data['s2'].ravel()))
        R = np.concatenate((R, data['R'].ravel()))
        n = np.concatenate((n, data['n'].ravel()))

        id_next = np.concatenate((id_next, data_next['prtclid'].ravel()))
        mass_next = np.concatenate((mass_next, data_next['mass'].ravel()))
    
    sortedidx = np.argsort(id)
    id = id[sortedidx]

    mass = mass[sortedidx]
    urel = urel[sortedidx]
    pos = pos[sortedidx]
    om2 = om2[sortedidx]
    s2 = s2[sortedidx]
    R = R[sortedidx]
    n = n[sortedidx]
    isitnan = sum([np.isnan(arr).any() for arr in [mass,urel,pos,om2,s2,R,n]])
    if isitnan > 0:
        raise SystemExit(f"NaN values for St = {stb:.3f}, Time = {time:.1f}, name= {init},num_process= {num_process}")
    sortedidx_next = np.argsort(id_next)
    mass_next = mass_next[sortedidx_next]
    dmass = mass_next - mass

    return mass, urel, dmass, pos, om2, s2, R, n
#%%
all_init = [[load_instant(stb,name,11.9)[0].sum() for stb in stb_s] for name in names]
all_init
#%%
times
#%%

def mask_max_growers(stb,init,t = times,frac = 0.9):
    initmass = load_instant(stb,init,t[0])[0]
    # print(f"Standard deviation for initial mass is : {initmass.std()}")
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
def load_mass_series(stb,init,times,load_luckiest = False,urel_bins = urel_bins):
    Ntimes = len(times)
    prtcl_mass = np.zeros(Ntimes)
    prtcl_mass_std = np.zeros(Ntimes)
    urel_vals = 0.5*(urel_bins[1:] + urel_bins[:-1])
    urel_pdf = np.zeros(len(urel_bins)-1)
    temp_pdf = np.zeros(len(urel_bins)-1)
    if load_luckiest == True:
        mask = mask_max_growers(stb,init,t = times, frac = 0.9)
    for i in range(Ntimes):
        
        dat,urel,dmass_temp = load_instant(stb,init,times[i])[:3]
        Nprtcl = len(dat)
        if load_luckiest == True: 
            dat = dat[mask]
            dmass_temp = dmass_temp[mask]
            urel = urel[mask]
            
        temp_pdf += np.histogram(urel.ravel(),bins = urel_bins)[0]
        urel_pdf += np.histogram(urel.ravel(),bins = urel_bins,weights = dmass_temp.ravel())[0]
        prtcl_mass[i] = dat.mean()
        prtcl_mass_std[i] = dat.std()
    cond = temp_pdf>0
    dmass_urel = urel_pdf[cond]/temp_pdf[cond]
    return prtcl_mass,prtcl_mass_std,Nprtcl,dmass_urel,urel_vals[cond]
        
#%%
load_mass_series(0.004,"Qneg",times)
#%%
db = {}
for stb in stb_s:
    for init in names: 
        for load_luckiest in [True, False]:
            which = "top10" if load_luckiest else "all"
            key = f"{wg}/{stb:.3f}/{init}/{which}"
            m, s,nprtcl,dmass_urel,urel_vals = load_mass_series(stb, init, times, load_luckiest=load_luckiest)
            db[key] = {"t":times.tolist(),"mean":m.tolist(),"std":s.tolist(),"n":nprtcl,"grwth_pdf":dmass_urel.tolist(), "urel_vals":urel_vals.tolist()}

with open(f"mass_data_{wg}_re_{1/nu:.1f}.json", "w") as f:
    json.dump(db, f)
print(f"Saved in mass_data_{wg}_re_{1/nu:.1f}.json")
raise SystemExit     
# %%

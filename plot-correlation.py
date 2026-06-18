#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib,h5py
from mpi4py import MPI
from scipy.fft import fftfreq,rfftn, irfftn
from scipy.special import gamma,factorial,comb
from scipy.stats import kstat
# %%
N = 256
Nbin = 16
num_process_load = 128
Np = N//num_process_load
g = 0.
nu = 1/434.1
datapath = lambda st,t: f"/mnt/pfs/rajarshi.chattopadhyay/codes/prtcls-in-turb/data_cosine/forced_True/N_256_Re_434.1/time_{t:.1f}/wo_g_st_{st}"
#%%


PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky,  Kz,  indexing = 'ij')
k = (kx**2 + ky**2 + kz**2)**0.5
n = np.zeros((N,N,N))
kint = np.round(k).astype(np.int16)
dealias = kint < N/3
shells = np.arange(-0.5,3**0.5*Nf, 1.)
normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
Xbins = Ybins = Zbins = np.linspace(0, L, Nbin+1, endpoint= True)
xbins, ybins, zbins = np.meshgrid(Xbins, Ybins, Zbins, indexing='ij')

def clip_zero(x):
    oldmean = x.mean()
    x = np.clip(x,0,None)
    newmean = x.mean()
    return x*oldmean/newmean
def e3d_to_e1d(x):  return np.histogram(kint.ravel(),bins = shells,weights=x.ravel())[0]  #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 

#%%


#%%
n_field = np.zeros((N,N,N))
dataload=  "/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_1003.2/last/wo_g_sts_0.001/"
for i in range(num_process_load):
    data = dataload + f"/n_{i}.npz"
    n_field[i*Np:(i+1)*Np] = np.load(data)['n']
#%%
prtcl_path = lambda st,t: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/prtcls-in-turb/data_cosine/forced_True/N_256_Re_434.1/time_{t:.1f}/wo_g_st_{st:.3f}")
Nprtcl = 4*128**3
# times = np.arange(6,28.2,0.5)
times = np.array([40.0])
Ntimes = len(times)
prtcl_mass = np.zeros((Ntimes,Nprtcl))
prtcl_id = np.zeros((Ntimes,Nprtcl))
prtcl_pos = np.zeros((Ntimes,Nprtcl,3))
prtcl_vel = np.zeros((Ntimes,Nprtcl,3))
prtcl_umat = np.zeros((Ntimes,Nprtcl,3))
#%%
def calc_fields(t, num_process):
    uk = np.zeros((3,N,N,Nf),dtype = np.complex128)
    for i in range(num_process):
        print(f"Loading Rank {i} for time {t}",end = "\r")
        sx = slice(i*N//num_process,(i+1)*N//num_process)
        data = np.load(f"/mnt/pfs/rajarshi.chattopadhyay/codes/prtcls-in-turb/data_cosine/forced_True/N_256_Re_434.1/time_{t:.1f}/Fields_k_{i}.npz")
        uk[0,:,sx,:] = data['uk']
        uk[1,:,sx,:] = data['vk']
        uk[2,:,sx,:] = data['wk']
    kvec = np.array([kx,ky,kz])
    Ak = 1j*np.einsum('j...,i...->ij...',kvec,uk)
    # S = 0.5*irfftn(Ak + np.moveaxis(Ak,[0,1],[1,0]),axes = (-3,-2,-1))
    A = irfftn(Ak,axes = (-3,-2,-1))
    dissip = 2*nu*np.einsum('ij...,ij...->...',0.5*(A + np.moveaxis(A,[0,1],[1,0])),0.5*(A + np.moveaxis(A,[0,1],[1,0])))
    Q = -0.5*np.einsum('ij...,ji...-> ...', A, A)
    modA = np.einsum('ij...,ij...-> ...', A, A)
    
    
    return dissip, Q, modA

dissip, Q, modA = calc_fields(times[0],num_process_load) 
# %%
st = 0.001
for i,t in enumerate(times):
    print(t, end = '\r')
    prtcl_count = 0
    for rank in range(num_process_load):
        print(f"Loading Rank {rank} for time {t}",end = "\r")
        data = np.load(prtcl_path(st,t)/f"state_{rank}.npz")
        prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
        prtcl_pos[i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['pos']
        prtcl_vel[i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['vel']
        prtcl_umat[i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['umat']
        prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
        prtcl_count += data['prtclid'].shape[0]
    argsind = prtcl_id[i].argsort()
    prtcl_pos[i] = prtcl_pos[i,argsind]
    prtcl_mass[i] = prtcl_mass[i,argsind]
    prtcl_id[i] = prtcl_id[i,argsind]


#%%
# prtcl_pos = np.random.uniform(0,L,size = prtcl_pos.shape)
#%%
sep = np.linalg.norm(prtcl_pos[:,] - prtcl_pos[0][None,...],axis = -1)
umag = np.linalg.norm(prtcl_umat[:,] - prtcl_umat[0][None,...],axis = -1)
vmag = np.linalg.norm(prtcl_vel[:,] - prtcl_vel[0][None,...],axis = -1)
#%%
#%%
from particles import MPI_particles
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()

stb = MPI_particles(comm, L, N, Nprtcl,st, st,g,nu, 0.4,1000,1e-6,3,X,Y,Z, x,y,z)
stb.to_exterp(1)
#%%
def calc_n_exterp(pos,n,stb):
    stb.coord[:,:stb.d] = pos
    stb.exterpmat[:] = 1.0
    n[:] = stb.exterp_cosine_scalar(stb.coord,n)
    return n

#%%

#%%
# n = calc_n_exterp(prtcl_pos[0],n,stb)*TWO_PI**3/Nprtcl
n = np.histogramdd(prtcl_pos[0], bins = [Xbins, Ybins, Zbins])[0]
#%%

#%%

p1  = plt.imshow(n_field[100],cmap = "Greys",extent = (0,X[-1],0,Y[-1]),origin = 'lower')
plt.colorbar(p1)
#%%
norm1 = mpl.colors.LogNorm(vmin = 1e-2, vmax = 1e2)
# p1  = plt.imshow(np.exp(-st**2*Q[100]),cmap = "Greys",extent = (0,X[-1],0,Y[-1]),origin = 'lower')
p1  = plt.imshow(2*nu*modA[100],cmap = "Greys_r",extent = (0,X[-1],0,Y[-1]),origin = 'lower',norm = norm1)
plt.colorbar(p1,extend = 'max')
#%%
norm2 = mpl.colors.CenteredNorm()
p2  = plt.imshow(2*nu*Q[100],cmap = "RdBu",extent = (0,X[-1],0,Y[-1]),origin = 'lower',norm = norm2)
plt.colorbar(p2)
#%%
p3  = plt.imshow(dissip[100],cmap = "Greys_r",extent = (0,X[-1],0,Y[-1]),origin = 'lower',norm = norm1)
plt.colorbar(p3,extend = 'max')
#%%
n.mean()
#%%
#%%
# Pn,nbins = np.histogram(n_field.ravel(), bins = np.arange(n_field.max()+2)-0.5, density = True)[:2]
n_field = n_field
Pn,nbins = np.histogram(n_field.ravel(), bins =1000, density = True)[:2]
nvals = 0.5*(nbins[1:] + nbins[:-1])
#%%
moments = []
error_moments = []
for i in range(1,6):
    moments.append(((nvals**i*Pn).sum()*(nbins[1] -nbins[0]))**(1.0/(1.0*i)))
    error_moments.append(((
        (
        (nvals**(2*i)*Pn).sum()*(nbins[1] -nbins[0])-moments[-1]**(2*i)
        )/N**3
        )**0.5)**(1.0/(1.0*i)))
# moments = []
# for i in range(1,100):
#     moments.append(((n**i).mean())**(1.0/(1.0*i)))
moments,error_moments
#%%

alpha = -1/(1- moments[2])
moment_theo = lambda m: (1/alpha)*(gamma(alpha + m)/gamma(alpha))**(1/m)
# plt.errorbar(range(1,len(moments)+1),moments,error_moments,ls = '--')
plt.plot(range(1,len(moments)+1),moment_theo(np.arange(1,len(moments)+1)))
#%%
# print(alpha)
# ptheo = nvals**(alpha -1)*np.exp(-nvals*alpha)/(gamma(alpha)*((1/alpha)**alpha))
# ptheo = np.exp(-(nvals-n_field.mean())**2/(2*n_field.std()**2))/(2*np.pi*n_field.std()**2)**0.5
lmbd = 380.5
# nstd = (1.0/lmbd)**0.5
nstd = n_field.std()
ptheo = np.exp(-(nvals-1)**2/(2*nstd**2))/(2*np.pi*nstd**2)**0.5

# ptheo = n_field.mean()**(nvals)*np.exp(-n_field.mean())/factorial(nvals)
# ptheoext = comb(Nprtcl,nvals)*(1/Nbin**3)**nvals*(1 -1/Nbin**3)**(Nprtcl-nvals)
plt.plot(nvals, Pn,'.')
plt.plot(nvals, ptheo,'--')
# plt.plot(nvals, ptheoext,'-.')
# plt.yscale('log')
# %%
np.sum(ptheo)*(nbins[1] - nbins[0])
# %%
1./n_field.std()**2
# %%

# %%

import numpy as np

data = np.random.normal(size=10000)
for n in range(1, 5):
    print(f"κ_{n} = {kstat(data, n):.4f}")
# %%

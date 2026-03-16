#%%
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.fft import fftfreq,rfftn, irfftn
# %%
N =256
num_process = 256
Np = N//num_process
datapath = "/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_273.5/last/wo_g_sts_0.001"
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
shells = np.arange(-0.5,Nf, 1.)
normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))

def e3d_to_e1d(x):  return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0]  #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
#%%

# %%
n = np.zeros((N,N,N))
for i in range(num_process):
    data = datapath + f"/n_{i}.npz"
    n[i*Np:(i+1)*Np] = np.load(data)['n']
#%%
nk = rfftn(n,axes = (-3,-2,-1))
nspectra = e3d_to_e1d(np.abs(nk)**2*normalize)
nspectra.sum(), (n**2).sum()*dx*dy*dz

#%%
k =np.arange(nspectra.size)
plt.loglog(k[1:],(k**(-1.5)*nspectra)[1:])

#%%
plt.plot(n[0,0],'.-')
# %%
p1 = plt.imshow(n[10,],origin = 'lower', cmap = 'Greys',vmin = 0)
plt.colorbar(p1)

#%%
prtcl_path = lambda t: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_186.4/time_{t:.1f}/wo_g_stb_0.050_sts_0.001/")
Nprtcl = 256*40
times = np.arange(0,40.2,0.5)
Ntimes = len(times)
prtcl_mass = np.zeros((Ntimes,Nprtcl))
prtcl_id = np.zeros((Ntimes,Nprtcl))
# %%
for i,t in enumerate(times):
    prtcl_count = 0
    for rank in range(num_process):
        data = np.load(prtcl_path(t)/f"state_{rank}.npz")
        prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
        
        prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
        prtcl_count += data['prtclid'].shape[0]
# %%
tot_mass = prtcl_mass.sum(axis = 1)

#%%
# tot_mass[-1]
plt.plot(times, tot_mass/tot_mass[0]-1)
plt.ylabel('m/m(0)')
plt.xlabel('t')
# %%

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
from scipy.fft import fftfreq,rfftn, irfftn
# %%
N =256
num_process = 256
Np = N//num_process
datapath = lambda t,sts,stb: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_271.4/time_{t:.1f}/wo_g_sts_{sts:.3f}_stb_{stb:.3f}")
sts = 0.001
stb_s = [0.017,0.017/4, 0.017*4,0.017/9,0.017*6.35]
stb_s.sort()
stb_s= np.array(stb_s)
Nprtcl = np.round(8192*(0.017/stb_s)**1.5).astype(np.int32) #! 10240 0.017 St particles.
#%%
stb_s,Nprtcl

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
# n = np.zeros((N,N,N))
# t = 30.0
# for i in range(num_process):
#     data = datapath(t,sts,stb_s[0])/ f"n_{i}.npz"
#     n[i*Np:(i+1)*Np] = np.load(data)['n']
# #%%
# #%%
# nk = rfftn(n,axes = (-3,-2,-1))
# nspectra = e3d_to_e1d(np.abs(nk)**2*normalize)
# nspectra.sum(), (n**2).sum()*dx*dy*dz

# #%%
# k1d =np.arange(nspectra.size)
# plt.loglog(k1d[1:],(nspectra)[1:])

# #%%
# plt.plot(n[0,0],'.-')
# # %%
# p1 = plt.imshow(n[10,],origin = 'lower', cmap = 'Greys',vmin = 0)
# plt.colorbar(p1)

#%%
prtcl_path = lambda t,stb: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_271.4/time_{t:.1f}/wo_g_stb_{stb:.3f}_sts_0.001/")
times = np.arange(0,25.1,0.5)
Ntimes = len(times)
prtcl_mass = np.zeros((Ntimes,Nprtcl[0]))
prtcl_id = np.zeros((Ntimes,Nprtcl[0]))
# %%
for i,t in enumerate(times):
    prtcl_count = 0
    print(f"Time = {t:.1f}", end = "\r")
    for rank in range(num_process):
        data = np.load(prtcl_path(t,stb_s[0])/f"state_{rank}.npz")
        prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
        
        prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
        prtcl_count += data['prtclid'].shape[0]
# %%
tot_mass = prtcl_mass.sum(axis = 1)

#%%
# tot_mass[-1]
plt.plot(times, tot_mass,'.-')
plt.ylabel('m')
plt.xlabel('t')
plt.yscale('log')
# %%
times = np.arange(0,25.1,0.5)
# Ntimes = len(times)
tot_mass = []
for kk,stb in enumerate(stb_s):
    if kk< 2: 
        # continue
        times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
    else: 
        times = np.arange(0,40.1,0.5)
    Ntimes = len(times)
    prtcl_mass = np.zeros((Ntimes,Nprtcl[kk]))
    prtcl_id = np.zeros((Ntimes,Nprtcl[kk]))
    for i,t in enumerate(times):
        print(f"St = {stb:.3f}, Time = {t:.1f}",end = "\r")
        prtcl_count = 0
        for rank in range(num_process):
            data = np.load(prtcl_path(t,stb)/f"state_{rank}.npz")
            prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
            
            prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
            prtcl_count += data['prtclid'].shape[0]
    tot_mass.append( prtcl_mass.sum(axis = 1))
    
    # plt.plot(times, tot_mass[:,kk],'.-',label = f'stb = {stb:.3f}')


# plt.ylabel('Vol frac')
# plt.xlabel('t')
# plt.yscale('log') 
# plt.legend()
# %%
mpl.rcParams['text.usetex'] =  True
plt.figure(figsize = (4,3),dpi = 600)
M0 = M0 = 7.2e-6 #2.6\mu m particles with volfrac 1e-8.
cols = ["#390099","#9e0059","#ff0054","#ff5400","#ffbd00"]
cols.reverse()
for kk,stb in enumerate(stb_s):
    if kk<1 : continue
    if kk< 2:
        times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
    else: 
        times = np.arange(0,40.1,0.5)
    plt.plot(times, tot_mass[kk]/tot_mass[kk][0],'.-',color = cols[kk],label = fr'{stb/sts:.1f}')
 

plt.ylabel(r'$M_b/M_b(t = 0)$')
plt.xlabel(r'$t$')
plt.ylim(1,1.01)
plt.xlim(0,1.0)
# plt.yscale('log',base = 8) 
plt.legend(ncols = 4, handlelength = 1,frameon = False,loc = 'upper center', bbox_to_anchor=(0.5,1.15))
# plt.text(-1.0,1.023,r"$St_b(0)/St_s$",ha = 'center')
plt.tight_layout()
# %%
tot_mass[0]
# %%
mdot = np.gradient(tot_mass,times,axis = 0)
# %%
for kk,stb in enumerate(stb_s):
    if kk>0:
        plt.plot(times, mdot[:,kk]/tot_mass[:,kk],'.-',label = f'stb = {stb:.3f}')


plt.ylabel('Mass')
plt.xlabel('t')
plt.yscale('log') 
plt.legend()
# %%
tot_mass[0]/TWO_PI**3
# %% [Checking mass pdf]

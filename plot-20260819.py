#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib,os,h5py
from scipy.fft import fftfreq,rfftn, irfftn
from scipy.interpolate import CubicSpline, splrep,splev,splder
from scipy.optimize import newton,brentq
# %%
N =256
num_process = 256
Np = N//num_process
datapath = lambda t,sts,stb: pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/stochastic/highn/N_256_Re_398.1/time_{t:.1f}/wo_g_sts_{sts:.3f}_stb_{stb:.3f}")
sts = 0.001
stb_s = [0.017,0.017/4, 0.017*4,0.017*6.35]
stb_s.sort()
stb_s= np.array(stb_s)
Nprtcl = np.round(8192*(0.017/stb_s)**1.5).astype(np.int32) #! 10240 0.017 St particles.
M0 = 7.2e-4
rhop = 1000
PI = np.pi
TWO_PI = 2*PI
m = 2
kmax = (N*2**0.5)//3
eta = m/kmax
rs = eta*(9*sts/2/rhop)**0.5 
nprtcls0 = 3*M0*TWO_PI**3/(rhop * 4 * PI *rs**3)
rs,TWO_PI/N,nprtcls0

#%%
stb_s,Nprtcl
cols = ["#178aac","#8417ac","#ac3917","#40ac17"]
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
nmin = TWO_PI**3/nprtcls0/(dx*dy*dz)
nmin
#%%
def find_s(n,target = 1):
    '''Creting the h(s) function by choosing different s'''
    s = np.linspace(-n.max(),n.max(),20)
    hs = s*0.0
    for i in range(s.size):
        hs[i] = np.mean(np.clip(s[i] + n[...]  ,0,None)) - target
    spl = CubicSpline(s, hs)
    dspl = spl.derivative()
    s0 = 0.0
    sstar = newton(spl, s0, fprime=dspl)
    return sstar
    
# %%
nold_clip = np.zeros((N,N,N))
nnew_clip = np.zeros((N,N,N))
nthresh_clip = np.zeros((N,N,N))
nlog_clip = np.zeros((N,N,N))
nno_clip = np.zeros((N,N,N))

ns = (nold_clip,nnew_clip,nthresh_clip,nlog_clip,nno_clip)
nspectra = [] 
paths = ("/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/last/wo_g_sts_0.001/",
"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine_clip_new/forced_True/N_256_Re_398.1/last/wo_g_sts_0.001/",
"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine_thresh_clip/forced_True/N_256_Re_398.1/last/wo_g_sts_0.001/",
"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/last/wo_g_sts_0.001_log/",
"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine_nocap/forced_True/N_256_Re_273.5/last/wo_g_sts_0.001/"    
)
names = ["old_clip", "new_clip", "thresh_clip", "log_clip", "no_clip"]
load_num_slabs = 256
data_per_rank = N//load_num_slabs
rank_data = range(0,N) # The rank contains these slices 
slab_old = np.inf

for i in range(5):
    n = ns[i]
    for lidx,j in enumerate(rank_data):
        print(lidx,end = '\r')
        slab = j//data_per_rank
        idx = j%data_per_rank
            
            # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
            
        """Loading the truncated data"""
        
        if slab_old != slab:
            data = paths[i] + f"n_{slab}.npz"
        slab_old = slab
          
        n[lidx] = np.load(data)['n'][idx]  

    nkact = rfftn(n,axes = (-3,-2,-1))
    nspectra.append(e3d_to_e1d(np.abs(nkact)**2*normalize))
    slab_old = slab

#%%
ns[-1].mean(),ns[-1].std(), ns[-1].max(), ns[-1].min()
#%%
def clip_zero(n,target = 1):
    # """Clips negative values to zero and rescales the to conserve the mean
    # """
    # oldmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
    # x[:] = np.clip(x,0,None)
    # newmean = comm.allreduce(np.sum(x),op = MPI.SUM)/N**3
    '''Creting the h(s) function by choosing different s and then finding sstar. Returning the clipped sstar'''
    if np.abs((np.clip(n,0,None) - n).sum()/N**3) < 1e-8:
        return n
    nmax = np.max(np.abs(n))
    s = np.linspace(-nmax,nmax,20)
    hs = s*0.0
    for i in range(s.size):
        ss = np.clip(s[i ] + n  ,0,None).sum()/N**3 - target
        hs[i] = ss


    tck = splrep(s, hs, k=5)
    spl = lambda x: splev(x, tck)
    s0 = 0
    a, b = sorted([s0, s[0]])
    sstar = brentq(spl, a, b, xtol=1e-10)
    print(sstar)
    return np.clip(sstar + n, 0, None)
#%%
nsclipped = clip_zero(ns[-1])
print(np.abs(nsclipped - ns[-1]).max())
# nnew= np.clip(sstar + n[None,...]  ,0,None)
# nmine = np.clip(n,0,None)
# nmine *= np.mean(n)/np.mean(nmine)
# %%
nsclipped.mean(),nsclipped.std(), nsclipped.max(), nsclipped.min() 
#%%
nkact = rfftn(n,axes = (-3,-2,-1))
nspectra= e3d_to_e1d(np.abs(nkact)**2*normalize)
# nknew = rfftn(nnew,axes = (-3,-2,-1))
# nspectra_new= e3d_to_e1d(np.abs(nknew)**2*normalize)
# nkmine = rfftn(nmine,axes = (-3,-2,-1))
# nspectra_mine= e3d_to_e1d(np.abs(nkmine)**2*normalize)
#%%
# theshold_spectra =  nspectra
# new_clip_spectra =  nspectra
# og_clip_spectra = nspectra
#%%
k1d =np.arange(nspectra[0].size)
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
for label, spectra,col in zip(names, nspectra,colors):
    plt.loglog(k1d[1:],(spectra)[1:],c = col,label = label)
plt.legend()
# plt.loglog(k1d[1:],(nspectra_new)[1:],c = 'b')
# plt.loglog(k1d[1:],(nspectra_mine)[1:],c = 'r')
#%%
k1d =np.arange(nspectra.size)
plt.loglog(k1d[1:],(theshold_spectra)[1:],c = 'b', label = '1 prtcl thresh')
plt.loglog(k1d[1:],(og_clip_spectra)[1:],c = 'r',label = 'OG_clipping')
plt.xlabel('k')
plt.ylabel(r'$|\hat{n}(k)|^2$')
plt.legend(ncols = 3,handlelength = 1, loc = 'lower left')
#%%
s, hs = create_h(n)
plt.plot(s,hs, '.')
print(s,hs)
#%%
p1 = plt.imshow(n[30],cmap = 'Greys', origin = "lower",vmin = 0)
plt.colorbar(p1)
#%%
npdf, nbins = np.histogram(n.ravel(), bins = np.logspace(-10,10,201))
plt.plot(nbins[1:],npdf)
plt.xscale('log')
plt.yscale('log')
#%%
# n = np.zeros((N,N,N))
# # t = 30.0
# num_process_load = 128
# for i in range(num_process):
#     data = f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/last/wo_g_sts_0.001_log/n_{i}.npz"
#     print(np.load(data)['n'].shape)
#     n[i*Np:(i+1)*Np] = np.load(data)['n']

# #%%
# nspectra_loaded = np.load(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/N_256_Re_398.1/last/n_spectrum.npz")['nk']
#%%

nk = rfftn(n,axes = (-3,-2,-1))
nspectra = e3d_to_e1d(np.abs(nk)**2*normalize)
nspectra.sum(), (n**2).sum()*dx*dy*dz
# with h5py.File(f"./comparing_spectra.hdf5","a") as f:
#     f.create_dataset('nspectra_1003.2',data = nspectra, dtype = np.float64)

# #%%
k1d =np.arange(nspectra.size)
plt.loglog(k1d[1:],(nspectra)[1:])
# plt.loglog(k1d[1:],(2*nspectra_loaded/(N**6/TWO_PI**3))[1:])
#%%
with h5py.File(f"./comparing_spectra.hdf5","r") as f:
    nspectra_398 = f['nspectra_log_398.1'][:]
    nspectra_271 = f['nspectra_271.4'][:]
    nspectra_1003 = f['nspectra_1003.2'][:]
k1d =np.arange(nspectra_398.size)
plt.loglog(k1d[1:],(nspectra_398)[1:],'.-',label = 'log_398.1')
plt.loglog(k1d[1:],(nspectra_271)[1:],'.-',label = '271.4')
plt.loglog(k1d[1:],(nspectra_1003)[1:],'.-',label = '1003.2')
plt.legend()
# #%%
# plt.plot(n[0,0],'.-')
# # %%
# p1 = plt.imshow(n[10,],origin = 'lower', cmap = 'Greys',vmin = 0)
# plt.colorbar(p1)

#%%
names = ["","highn/","highhighn/","synthetic/"]
M0s = 7.2*np.array([1e-4,1e-3,1e-2,1e-1])
prtcl_path = lambda t,stb,name = "": pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/lucky-droplets/data_cosine/forced_True/stochastic/{name}N_256_Re_398.1/time_{t:.1f}/wo_g_stb_{stb:.3f}_sts_0.001/")
times = [float(i.split("_")[-1]) for i in os.listdir(prtcl_path(0,stb_s[0]).parent.parent) if "time_" in i]
times.sort()
times = np.array(times)
times = np.arange(times[0], times[-1] + 0.5, 1) #! Reducing data loading.
Ntimes = len(times)
prtcl_mass = np.zeros((Ntimes,Nprtcl[0]))
prtcl_id = np.zeros((Ntimes,Nprtcl[0]))
#%%
#%%

# # %%
# for i,t in enumerate(times):
#     prtcl_count = 0
#     print(f"Time = {t:.1f}", end = "\r")
#     for rank in range(num_process):
#         data = np.load(prtcl_path(t,stb_s[0])/f"state_{rank}.npz")
#         prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
        
#         prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
#         prtcl_count += data['prtclid'].shape[0]
        
#     sortedidx = np.argsort(prtcl_id[i])
#     prtcl_id[i] = prtcl_id[i,sortedidx]
#     prtcl_mass[i] = prtcl_mass[i,sortedidx]
    

# #%%
# # %%
# tot_mass = prtcl_mass.sum(axis = 1)
# std_mass = np.std(prtcl_mass/prtcl_mass[0],axis = 1)
# #%%
# #%%

# # tot_mass[-1]
# plt.plot(times, tot_mass/tot_mass[0],'.-')
# plt.ylabel('m')
# plt.xlabel('t')
# plt.yscale('log',base = 8)
# # %%
# prtcl_count = np.random.randint(0,Nprtcl[0],100)
# # plt.plot(times, prtcl_mass[:,prtcl_count]/prtcl_mass[0,prtcl_count],'-',alpha = 0.2,color = cols[0])
# plt.fill_between(times,-std_mass+tot_mass/(tot_mass[0]) ,std_mass+tot_mass/(tot_mass[0]),color = cols[0],alpha = 0.3,lw = 0)

# plt.plot(times, tot_mass/(tot_mass[0]),'-',color = cols[0])
# plt.ylabel('m/m(0)')
# plt.xlabel('t')
# plt.xlim(times[0],times[-1])
# plt.ylim(1,None)
# # plt.yscale('log',base = 8)
# # plt.xscale('log',base = 2)
# %%

# %%
def load_and_plot(nami,ax,xs,ys,dets,down_lim,up_lim):
    
    tot_mass = []
    std_mass = []
    sample_traj = []
    for kk,stb in enumerate(stb_s):
        # continue
        # if kk< 2: 
        #     # continue
        #     times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
        # else: 
        #     times = np.arange(0,40.1,0.5)
        Ntimes = len(times)
        prtcl_mass = np.zeros((Ntimes,Nprtcl[kk]))
        prtcl_id = np.zeros((Ntimes,Nprtcl[kk]))
        for i,t in enumerate(times):
            prtcl_count = 0
            num_process = len([i for i in os.listdir(prtcl_path(t,stb,names[nami])) if "state_" in i])
            print(f"St = {stb:.3f}, Time = {t:.1f}, name= {names[nami]},num_process= {num_process}",end = "\r")
            # continue
            for rank in range(num_process):
                data = np.load(prtcl_path(t,stb,names[nami])/f"state_{rank}.npz")
                prtcl_id[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
                
                prtcl_mass[i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
                prtcl_count += data['prtclid'].shape[0]
                
            sortedidx = np.argsort(prtcl_id[i])
            prtcl_id[i] = prtcl_id[i,sortedidx]
            prtcl_mass[i] = prtcl_mass[i,sortedidx]
            
        prtcl_count = np.random.randint(0,Nprtcl[kk],100)
        sample_traj.append(prtcl_mass[:,prtcl_count]/prtcl_mass[0,prtcl_count])
        tot_mass.append( prtcl_mass.sum(axis = 1))
        std_mass.append( np.std(prtcl_mass/prtcl_mass[0],axis = 1))
        
        
        
        ax.plot(times, tot_mass[kk]/tot_mass[kk][0],'-',color = cols[kk],label = fr'${stb/sts:.1f}$')
        ax.fill_between(times,-std_mass[kk]+tot_mass[kk]/(tot_mass[kk][0]) ,std_mass[kk]+tot_mass[kk]/(tot_mass[kk][0]),color = cols[kk],alpha = 0.3,lw = 0)
        
        
        
        xs.append(times)
        ys.append(tot_mass[kk])
        down_lim.append(-std_mass[kk]+tot_mass[kk])
        up_lim.append(std_mass[kk]+tot_mass[kk])
        dets.append(fr'${stb/sts:.1f}, M0 = 7.2 \times 10^{{{np.log10(M0s[nami]/7.2):.0f}}}$')
    
    
    ax.set_xlabel(r'$t$')
    # ax.set_ylim(1,1.01)
    # ax.set_xlim(0,1.0)
    ax.set_title(fr"$7.2 \times 10^{{{np.log10(M0s[nami]/7.2):.0f}}}$")
    ax.set_yscale('log',base = 8) 
    # plt.text(-1.0,1.023,r"$St_b(0)/St_s$",ha = 'center')
# %%

#%%
mpl.rcParams['text.usetex'] =  True
fig, axs = plt.subplots(1,4, figsize = (7,1.5),dpi = 300,sharex = True)
for nami in range(len(names)):
    # continue
    xs, ys, dets,up_lim,down_lim = [], [], [],[], []
    load_and_plot(nami,axs[nami],xs,ys,dets,down_lim,up_lim)
axs[-1].legend(ncols = 4, handlelength= 1, frameon = False, loc= "upper center",bbox_to_anchor = (-1.3,1.5))
axs[0].set_ylabel(r'$M_b/M_b(t = 0)$')
fig.suptitle(fr"$St_s = {sts:.3f}$",y = -0.2)
# fig.tight_layout()

#%% 
np.savez_compressed("")
#%%

# Ntimes = len(times)
tot_mass = []
std_mass = []
sample_traj = []
for kk,stb in enumerate(stb_s):
    # if kk< 2: 
    #     # continue
    #     times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
    # else: 
    #     times = np.arange(0,40.1,0.5)
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
            
        sortedidx = np.argsort(prtcl_id[i])
        prtcl_id[i] = prtcl_id[i,sortedidx]
        prtcl_mass[i] = prtcl_mass[i,sortedidx]
        
    prtcl_count = np.random.randint(0,Nprtcl[kk],100)
    sample_traj.append(prtcl_mass[:,prtcl_count]/prtcl_mass[0,prtcl_count])
    tot_mass.append( prtcl_mass.sum(axis = 1))
    std_mass.append( np.std(prtcl_mass/prtcl_mass[0],axis = 1))
    
    # plt.plot(times, tot_mass[:,kk],'.-',label = f'stb = {stb:.3f}')


# plt.ylabel('Vol frac')
# plt.xlabel('t')
# plt.yscale('log') 
# plt.legend()
# %%
mpl.rcParams['text.usetex'] =  True
plt.figure(figsize = (4,3),dpi = 600)
M0 = M0 = 7.2e-6 #2.6\mu m particles with volfrac 1e-8.
# cols = ["#390099","#9e0059","#ff0054","#ff5400","#ffbd00"]
# cols.reverse()
for kk,stb in enumerate(stb_s):
    # if kk<1 : continue
    # if kk< 2:
    #     times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
    # else: 
    #     times = np.arange(0,40.1,0.5)
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
mdot = np.gradient(np.array(tot_mass),times,axis = 1)
# %%
xscale = 'none'
plt.figure(figsize = (6,4.5), dpi = 200)
for kk,stb in enumerate(stb_s):
    # plt.plot(times, sample_traj[kk],'--',alpha = 0.2,color = cols[kk])
    plt.fill_between(times,-std_mass[kk]+tot_mass[kk]/(tot_mass[kk][0]) ,std_mass[kk]+tot_mass[kk]/(tot_mass[kk][0]),color = cols[kk],alpha = 0.3,lw = 0) 

    plt.plot(times, tot_mass[kk]/(tot_mass[kk][0]),'-',color = cols[kk],label = fr'${stb:.3f}$')

plt.ylabel(r'$M_b/M_b(0)$')
plt.xlabel(r'$t$')
plt.ylim(1,None)
plt.yscale('log',base = 8) 
if xscale == 'log':
    plt.xscale('log',base = 2) 
    plt.xlim(times[1],times[-1])
else:
    plt.xlim(times[0],times[-1])
plt.title(fr"$St_s = {sts:.3f}$")

# plt.legend(ncols = 4, handlelength = 1,frameon = False,loc = 'upper center', bbox_to_anchor=(0.5,1.15),title = r"$St_b(0)$")
plt.legend( handlelength = 1,frameon = False,title = r"$St_b(0)$")
# plt.text(0.16,2.125,r'$St_b(0)$')
plt.tight_layout()
# %%
xscale = 'none'
plt.figure(figsize = (6,4.5), dpi = 200)
for kk,stb in enumerate(stb_s):
    # plt.plot(times, sample_traj[kk],'--',alpha = 0.2,color = cols[kk])


    plt.plot(times, mdot[kk],'-',color = cols[kk],label = fr'${stb:.3f}$')

plt.ylabel(r'Growth Rate')
plt.xlabel(r'$t$')
# plt.ylim(1,None)
# plt.yscale('log',base = 8) 
if xscale == 'log':
    plt.xscale('log',base = 2) 
    plt.xlim(times[1],times[-1])
else:
    plt.xlim(times[0],times[-1])
plt.title(fr"$St_s = {sts:.3f}$")

# plt.legend(ncols = 4, handlelength = 1,frameon = False,loc = 'upper center', bbox_to_anchor=(0.5,1.15),title = r"$St_b(0)$")
plt.legend( handlelength = 1,frameon = False,title = r"$St_b(0)$")
# plt.text(0.16,2.125,r'$St_b(0)$')
plt.tight_layout()
# %%
np.array(tot_mass).shape
# %%
# ------------------- thresholds ------------------- #
st_ths = [0.017*(i**2) for i in range(2,6)]
m_ths = [(tot_mass[1][0]/Nprtcl[1])*(i**3) for i in range(2,6)]
# -------------------------------------------------- #
# %%
t_th = np.ones((len(Nprtcl),len(m_ths)))*np.inf

for kk,stb in enumerate(stb_s):
    # if kk< 2: 
    #     # continue
    #     times = np.array(list(np.arange(0,1.95,0.1)) +  list(np.arange(2,40.1,0.5)))
    # else: 
    #     times = np.arange(0,40.1,0.5)
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
            
        sortedidx = np.argsort(prtcl_id[i])
        prtcl_id[i] = prtcl_id[i,sortedidx]
        prtcl_mass[i] = prtcl_mass[i,sortedidx]
        if i==0: print(stb, prtcl_mass[i])
        for mm,mth in enumerate(m_ths):
            if np.sum(prtcl_mass[i]>= mth) >= 10:
                t_th[kk,mm] = min(t_th[kk,mm],t)
        
    prtcl_count = np.random.randint(0,Nprtcl[kk],100)
    sample_traj.append(prtcl_mass[:,prtcl_count]/prtcl_mass[0,prtcl_count])
    tot_mass.append( prtcl_mass.sum(axis = 1))
    std_mass.append( np.std(prtcl_mass/prtcl_mass[0],axis = 1))
# %%
t_th,m_ths
# %%

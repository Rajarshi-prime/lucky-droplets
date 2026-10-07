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
# %%
N =256
num_process = 256
Np = N//num_process
dt_save = 0.5
gravity = False
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
Ntimes = len(times)
prtcl_mass = np.zeros((Ntimes,Nprtcl[0]))
prtcl_id = np.zeros((Ntimes,Nprtcl[0]))

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


# %%
def load_instant(stb, init,time):
    prtcl_count = 0
    path = prtcl_path(time,stb,init)
    num_process = len([i for i in os.listdir(path) if "state_" in i])
    print(f"St = {stb:.3f}, Time = {time:.1f}, name= {init},num_process= {num_process}",end = "\r")
    mass = np.zeros(0)
    urel = np.zeros(0)
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
        prtcl_count += data['prtclid'].shape[0]
        
        id_next= np.concatenate((id_next,data_next['prtclid'].ravel()))
        mass_next = np.concatenate((mass_next,data_next['mass'].ravel()))
        
    sortedidx = np.argsort(id)
    id = id[sortedidx]
    mass = mass[sortedidx]
    urel = urel[sortedidx]
    
    sortedidx_next = np.argsort(id_next)
    id_next = id_next[sortedidx_next]
    mass_next = mass_next[sortedidx_next]
    dmass = mass_next - mass
    
    return mass,urel, dmass
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
dmass_bins = np.linspace(0,1e-3,1001)
def load_mass_series(stb,init,times,load_luckiest = False,dmass_bins = dmass_bins):
    Ntimes = len(times)
    prtcl_mass = np.zeros(Ntimes)
    prtcl_mass_std = np.zeros(Ntimes)
    dmass_vals = 0.5*(dmass_bins[1:] + dmass_bins[:-1])
    dmass_pdf = np.zeros(len(dmass_bins)-1)
    temp_pdf = np.zeros(len(dmass_bins)-1)
    if load_luckiest == True:
        mask = mask_max_growers(stb,init,t = times, frac = 0.9)
    for i in range(Ntimes):
        
        dat,urel,dmass_temp = load_instant(stb,init,times[i])
        Nprtcl = len(dat)
        if load_luckiest == True: 
            dat = dat[mask]
            dmass_temp = dmass_temp[mask]
            urel = urel[mask]
            
        temp_pdf += np.histogram(dmass_temp.ravel(),bins = dmass_bins)[0]
        dmass_pdf += np.histogram(dmass_temp.ravel(),bins = dmass_bins,weights = urel.ravel())[0]
        prtcl_mass[i] = dat.mean()
        prtcl_mass_std[i] = dat.std()
    cond = temp_pdf>0
    dmass_urel = dmass_pdf[cond]/temp_pdf[cond]
    return prtcl_mass,prtcl_mass_std,Nprtcl,dmass_urel,dmass_vals[cond]
        
#%%
load_mass_series(0.004,"Qneg",times)
#%%
db = {}
for stb in stb_s:
    for init in names: 
        for load_luckiest in [True, False]:
            which = "top10" if load_luckiest else "all"
            key = f"{wg}/{stb:.3f}/{init}/{which}"
            m, s,nprtcl,dmass_urel,dmass_vals = load_mass_series(stb, init, times, load_luckiest=load_luckiest)
            db[key] = {"t":times.tolist(),"mean":m.tolist(),"std":s.tolist(),"n":nprtcl,"grwth_pdf":dmass_urel.tolist(), "dmass_vals":dmass_vals.tolist()}
with open(f"mass_data_{wg}.json", "w") as f:
    json.dump(db, f)

raise SystemExit     
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
        times = np.arange(0,5.6,0.5)
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
        # dets.append(fr'${stb/sts:.1f}, M0 = 7.2 \times 10^{{{np.log10(M0[nami]/7.2):.0f}}}$')
    
    
    ax.set_xlabel(r'$t$')
    # ax.set_ylim(1,1.01)
    # ax.set_xlim(0,1.0)
    ax.set_title(fr"{names[nami]}")
    ax.set_yscale('log',base = 8) 
    # plt.text(-1.0,1.023,r"$St_b(0)/St_s$",ha = 'center')
# %%

#%%
mpl.rcParams['text.usetex'] =  True
fig, axs = plt.subplots(1,len(names), figsize = (9,1.5),dpi = 300,sharex = True)
xs, ys, dets,up_lim,down_lim = [], [], [],[], []
for nami in range(len(names)):
    # continue
    load_and_plot(nami,axs[nami],xs,ys,dets,down_lim,up_lim)
axs[-1].legend(ncols = 4, handlelength= 1, frameon = False, loc= "upper center",bbox_to_anchor = (-1.3,1.5))
axs[0].set_ylabel(r'$M_b/M_b(t = 0)$')
for ax in axs:
    ax.set_ylim(0,2)
fig.suptitle(fr"$St_s = {sts:.3f}$",y = -0.2)
# fig.tight_layout()

#%%
cols_st = ["#390099","#9e0059","#ff0054","#ff5400","#ffbd00"]
def load_and_plot_st(kk, stb, ax, xs, ys, dets, down_lim, up_lim):
    times = np.arange(0, 15.1, 0.5)
    Ntimes = len(times)

    for nami, name in enumerate(names):
        prtcl_mass = np.zeros((Ntimes, Nprtcl[kk]))
        prtcl_id   = np.zeros((Ntimes, Nprtcl[kk]))

        for i, t in enumerate(times):
            prtcl_count = 0
            num_process = len([f for f in os.listdir(prtcl_path(t, stb, name)) if "state_" in f])
            print(f"St = {stb:.3f}, t = {t:.1f}, name = {name}, nproc = {num_process}", end="\r")

            for rank in range(num_process):
                data = np.load(prtcl_path(t, stb, name) / f"state_{rank}.npz")
                sl = slice(prtcl_count, prtcl_count + data['prtclid'].shape[0])
                prtcl_id[i, sl]   = data['prtclid'].ravel()
                prtcl_mass[i, sl] = data['mass']
                prtcl_count += data['prtclid'].shape[0]

            idx = np.argsort(prtcl_id[i])
            prtcl_id[i]   = prtcl_id[i, idx]
            prtcl_mass[i] = prtcl_mass[i, idx]

        tot_mass = prtcl_mass.sum(axis=1)
        std_mass = np.std(prtcl_mass / prtcl_mass[0], axis=1)

        ax.plot(times, tot_mass / tot_mass[0], '-', color=cols_st[nami], label=name)
        # ax.fill_between(times,tot_mass / tot_mass[0] - std_mass,tot_mass / tot_mass[0] + std_mass,color=cols_st[nami], alpha=0.3, lw=0)

        xs.append(times)
        ys.append(tot_mass)
        down_lim.append(tot_mass - std_mass)
        up_lim.append(tot_mass + std_mass)

    ax.set_xlabel(r'$t$')
    ax.set_title(fr'$St_b/St_s = {stb/sts:.1f}$')
    

#%%
mpl.rcParams['text.usetex'] = True
fig, axs = plt.subplots(1, len(stb_s), figsize=(9, 1.5), dpi=300, sharex=True)

xs, ys, dets, up_lim, down_lim = [], [], [], [], []
for kk, stb in enumerate(stb_s):
    load_and_plot_st(kk, stb, axs[kk], xs, ys, dets, down_lim, up_lim)

axs[-1].legend(ncols=len(names), handlelength=1, frameon=False, loc="upper center", bbox_to_anchor=(-1.3, 1.5))
axs[0].set_ylabel(r'$M_b/M_b(t = 0)$')
# for ax in axs:
    # ax.set_yscale('log', base=8)
    # ax.set_ylim(0, 2)
fig.suptitle(fr"$St_s = {sts:.3f}$", y=-0.2)

#%%
mpl.rcParams['text.usetex'] = True
fig, axs = plt.subplots(1, len(stb_s), figsize=(13, 2), dpi=300, sharex=True)

for kk, stb in enumerate(stb_s):
    ax = axs[kk]
    for nami, name in enumerate(names):
        times = xs[kk*len(names)+ nami]
        tot_mass = ys[kk*len(names)+nami]
        name = names[nami]
        ax.plot(times/tf, tot_mass / tot_mass[0], '-', color=cols_st[nami], label=name)
        ax.set_xlabel(r'$t/\tau_\eta$')
        ax.set_title(fr'$St_b/St_s = {stb/sts:.1f}$')
axs[-1].legend(ncols=len(names), handlelength=1, frameon=False, loc="upper center", bbox_to_anchor=(-1.3, 1.5))
axs[0].set_ylabel(r'$M_b/M_b(t = 0)$')
# for ax in axs:
    # ax.set_yscale('log', base=8)
    # ax.set_ylim(0, 2)
fig.suptitle(fr"$St_s = {sts:.3f}$", y=-0.2)
fig.tight_layout()

#%%
len(xs)

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
    times = np.arange(0,7.2,0.5)
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

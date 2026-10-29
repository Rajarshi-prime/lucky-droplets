#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys,pickle
#%%
t = np.arange(0,10)
#%%
d_s = 1e-5 #! All in meters 
rhop = 1e3
ms = 1/6.*np.pi*d_s**3.*rhop
g = 9.81 #! In m.s^{-2}
nu = 1e-5
cs = 7.2e-7 #! Volume fraction of small particles (equiv to M0 = \rhop cs = 7.2e-4)
tbfactor = rhop/nu*(1/18.)*(6/(np.pi*rhop))**(2./3.)
ts = rhop/nu*(1/18.)*(6/(np.pi*rhop))**(2./3.)*(ms**(2./3.))
factor = (3.*(np.pi*rhop)**0.5/(4.))**(2./3.)*cs #! (\frac{3\sqrt{\pi \rho}}{4})^{\frac23} c_s
dbs = [1e-5*2, 1e-5*(2.52)**0.5,1e-5*(6.35)**0.5,1e-5*(10.08)**0.5 ]
# stb_s = [0.017*4,0.017*,0.017*6.35,0.017*10.08]
db0 = dbs[int(sys.argv[-1])]
mb0 = 1/6.*np.pi*db0**3.*rhop
tb0 = mb0**(2./3.)*tbfactor
terminal_vb = -g*tb0
#%%
decel = False
def RHS(t, x,factor = factor,tbfactor =tbfactor,ts = ts, ms = ms,g = g,deceleration = decel):
    #? x[0] = mass, x[1] = vel
    y = 0.0*x
    tb = x[0]**(2./3.)*tbfactor
    y[0] = factor*x[0]**(2./3.)*(1 + (ms/x[0])**(1./3.))**2*np.abs(x[1] + ts*g)
    y[1] = deceleration*factor*x[0]**(-1./3.)*(1 + (ms/x[0])**(1./3.))**2*np.abs(x[1] + ts*g)*(-ts*g - x[1]) -g - x[1]/tb
    return y
# %%
x0 = np.array([mb0,0.0])
# %%
sol     = solve_ivp(lambda t, x: RHS(t, x, deceleration=False), [0, 1800], x0)
sol_new = solve_ivp(lambda t, x: RHS(t, x, deceleration=True),  [0, 1800], x0)
# %%
with open(f"size_dbs_{d_s*1e6}_um_{db0:.7f}.pkl", "wb") as f:
    pickle.dump((sol,sol_new,terminal_vb),f) #! sol_new is with deceleration
#%%
with open(f"size_dbs_{d_s*1e6}_um_{db0:.7f}.pkl", "rb") as f:
    sol, sol_new, terminal_vb = pickle.load(f) #! sol_new is with deceleration
#%%
print(f"File saved in size_dbs_{d_s*1e6}_um_{db0:.7f}.pkl")
# #%%
# plt.plot(sol.t,sol.y[0]/x0[0],'-.')
# plt.plot(sol_new.t,sol_new.y[0]/x0[0],'--')
# plt.xlabel("time (sec)")
# plt.ylabel("m(t)/m(0)")
# plt.yscale('log',base =8)
# # plt.xscale('log',base = 10)
# plt.ylim(1,1+1/125)
# plt.xlim(0,10)
# # %%
# plt.plot(sol.t,sol.y[1]/terminal_vb,'-.')
# plt.plot(sol_new.t,sol_new.y[1]/terminal_vb,'--')
# plt.xlabel("time (sec)")
# plt.ylabel(r"$v/v_{T}(0)$")
# plt.xscale("log")

# #%%
# terminal_vb
# # %%

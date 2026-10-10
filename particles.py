import numpy as np
from mpi4py import MPI
from numba import njit,prange
import math,atexit
# from line_profiler import LineProfiler
# profiler = LineProfiler()
# def save_profile(profiler,rank):
#     with open(f"profiling/rank_{rank}.txt","w") as f: profiler.print_stats(stream=f)

@njit(parallel= True,fastmath = {'nnan', 'ninf', 'nsz', 'arcp', 'contract'}) #! Does not approximate cos
def _calc_usend_numba(ufield, xidx, yidx,zidx,ypos, zpos,usend, Nprtcl,N , dy , dz , cosorder , Y , Z,comp ):

        for p in prange(Nprtcl):   
            for i in prange(cosorder):
                indi = i - cosorder//2+1 
                tempyidx = (yidx[p] + indi)%N
                dyshift = (1 + np.cos((ypos[p] - Y[tempyidx])*(np.pi/(2*dy))))/4.0

                for j in prange(cosorder):
                    indj = j - cosorder//2+1 
                    tempzidx = (zidx[p] + indj)%N
                    dzshift = (1 + np.cos((zpos[p] - Z[tempzidx])*(np.pi/(2*dz))))/4.0
                    
                    for c in prange(comp):
                        usend[p,c] += ufield[c,xidx[p],tempyidx,tempzidx]*dyshift*dzshift 
        return usend


@njit(parallel= True,fastmath = {'nnan', 'ninf', 'nsz', 'arcp', 'contract'}) #! Does not approximate cos
def _calc_uadd_numba_scalar(cfield, xidx, yidx,zidx,ypos, zpos,udat, Nprtcl,N,dx, dy,  dz, cosorder, Y, Z):
    
    for p in range(Nprtcl):   
        for i in prange(cosorder):
            indi = i - cosorder//2+1 
            tempyidx = (yidx[p] + indi)%N
            dyshift = (1 + np.cos((ypos[p] - Y[tempyidx])*(np.pi/(2*dy))))/4.0

            for j in prange(cosorder):
                indj = j - cosorder//2 + 1
                tempzidx = (zidx[p] + indj)%N
                dzshift = (1 + np.cos((zpos[p] - Z[tempzidx])*(np.pi/(2*dz))))/4.0
                cfield[ xidx[p],tempyidx,tempzidx] += udat[p]*dyshift*dzshift/(dx*dy*dz)
    return cfield*1.0

@njit(parallel= True,fastmath = {'nnan', 'ninf', 'nsz', 'arcp', 'contract'}) #! Does not approximate cos
def _calc_uadd_numba_vector(cfield, xidx, yidx,zidx,ypos, zpos,udat, Nprtcl,N,dx, dy,  dz, cosorder, Y, Z,comp):
    
    for p in range(Nprtcl):   
        for i in prange(cosorder):
            indi = i - cosorder//2+1 
            tempyidx = (yidx[p] + indi)%N
            dyshift = (1 + np.cos((ypos[p] - Y[tempyidx])*(np.pi/(2*dy))))/4.0

            for j in prange(cosorder):
                indj = j - cosorder//2+1 
                tempzidx = (zidx[p] + indj)%N
                dzshift = (1 + np.cos((zpos[p] - Z[tempzidx])*(np.pi/(2*dz))))/4.0
                for c in prange(comp):
                    cfield[c, xidx[p],tempyidx,tempzidx] += udat[p,c]*dyshift*dzshift/(dx*dy*dz)
    return cfield*1.0
    
class MPI_particles:
    
    def __init__(self,comm, L, N, Nprtcl,st_s, st,g,nu, tau_eta,rho_p,M0 ,d,X,Y,Z, x,y,z):
        
        self.comm, self.L, self.N, self.Nprtcl, self.d,self.g, self.tau_eta = comm, L, N, Nprtcl,d,g,tau_eta
        self.X,self.Y,self.Z, self.x,self.y,self.z = X,Y,Z, x,y,z
        self.dx = X[1] - X[0]
        self.dy = Y[1] - Y[0]
        self.dz = Z[1] - Z[0]
        self.M0 = M0
        self.rhop = rho_p
        self.rank = comm.Get_rank()
        self.num_process =  comm.Get_size()
        self.Np = self.N//self.num_process
        self.sx = slice(self.rank*self.Np,(self.rank+1)*self.Np)
        self.Nprtcl_proc = self.Nprtcl//self.num_process
        self.startdom = self.X[(self.rank*self.Np)%N]
        self.glob_startdom = self.comm.allgather(self.startdom)
        self.enddom = self.startdom + self.Np*self.dx
        self.glob_enddom = self.comm.allgather(self.enddom)
        self.interporder = 2 #! Power of polynomial +1 
        self.nums = np.arange(self.interporder)
        self.cosorder = 4
        # self.factor = (nu*tau_eta/ (2*rho_p))**(1.5) *(36*np.pi*rho_p)/M0 #! Factor connecting mass and stokes number.
        # self.growthfactor = 9 *np.pi*nu/(2*rho_p)
        # self.decelerationfactor = (rho_p/(8*nu))**0.5
        self.factor = ((4*np.pi*rho_p)*(tau_eta)**(1.5)*(nu)**(1.5) *27 )/(3 * rho_p**(1.5)*2**(1.5))#! Factor connecting mass and stokes number.
        
        self.n_cs_factor = (M0/self.rhop) #* Factor relating volume fraction (c_s) and to the n whose mean is 1. c_s =  n *n_cs_factor.
        self.growthfactor = (3*(np.pi**0.5)/(4*rho_p))**(2./3.) #! Factor for decay of n. Multiply with rho_p and n_cs_factor in the Mdot equation.
        self.decelerationfactor = self.growthfactor*self.rhop*self.n_cs_factor
        
        self.nneighbors = math.ceil(self.cosorder/(2*self.Np)) 
        neighbors = [(self.rank - (self.nneighbors -i))%self.num_process for i in range(self.nneighbors)] + [(self.rank + (i + 1))%self.num_process for i in range(self.nneighbors)]
        indexes = list(np.cumsum([2*self.nneighbors]*self.num_process))
        allneighbours = self.comm.allgather(neighbors)
        allneighbours = list(np.ravel(allneighbours))
        
        self.cart_comm = comm.Create_graph(indexes, allneighbours, reorder = False)
        # print(f"Rank {self.rank} : neighbors {self.cart_comm.neighbors}\n")
        # raise SystemExit


        # ------------------- initializing particles ------------------- #
    
        self.coord = np.random.uniform(0,self.L,(self.Nprtcl_proc,2*self.d +1 )) #! Contains both position and velocity and the physical mass initialized randomly between zero and two pi
        self.prtclid = np.arange(self.rank*self.Nprtcl_proc,(self.rank
         + 1)*self.Nprtcl_proc).reshape((-1,1)) #! Unique particle ID
        self.coord[:,-1] = st**1.5*self.factor
        self.st_s = st_s
        self.st = (self.coord[:,-1]/self.factor)**(2/3.)
        # --------------------------------------------------------------- #
        
        # ------------------- Mmat matrix for interp ------------------- #
        
        self.Mmat = np.zeros((self.interporder,self.interporder))
        for i in range(self.interporder):
            for j in range(self.interporder):
                for s in range(j,self.interporder):
                    self.Mmat[j,i] += (-1)**(s-j)*math.factorial(self.interporder)*(self.interporder - s-1)**(self.interporder-i-1)/(math.factorial(s-j)*math.factorial(self.interporder +j -s))
                
            self.Mmat[:,i] = self.Mmat[:,i]/(math.factorial(self.interporder-i-1)*math.factorial(i))
        self.Mmat = np.array(self.Mmat)
        self.cosorder = np.arange(-1, 3)

        # --------------------------------------------------------------- #
        # atexit.register(save_profile,profiler,self.rank)
    
    def to_interp(self,interpdim):
        """ The input should be the field variable and the number of components:
        eg: u = 3, c = 1 etc.
        """

        self.interpmat = np.zeros((self.coord.shape[0],interpdim))
        self.interpdim = interpdim
        
        # ------------------- initializing numba array -------------------- #
        self.calc_usend(np.zeros((interpdim,2,2,2)),np.zeros((2,5)))
        self.comm.Barrier()
        if self.rank ==0 : print(f"usend activated")
        # ----------------------------------------------------------------- #
            
            
    def to_exterp(self,exterpdim):
        """ The input should be the field variable and the number of components:
        eg: u = 3, c = 1 etc.
        """
        self.exterpmat = np.ones((self.coord.shape[0],exterpdim))
        self.exterpdim = exterpdim
        
        # ------------------- initializing numba array -------------------- #
        if exterpdim >1:
            self.calc_uadd_vector(np.zeros((exterpdim,2,2,2)),np.zeros((2,5 + exterpdim)))
        else: 
            self.calc_uadd_scalar(np.zeros((2,2,2)),np.zeros((2,5 + exterpdim)))
        self.comm.Barrier()
        if self.rank ==0 : print(f" uadd activated")
        # ----------------------------------------------------------------- #    
        
        
    #@profiler
    def calc_usend(self,ufield, coordrecv):
        Nprtcl = coordrecv.shape[0]
        usend = np.zeros((Nprtcl,self.interpdim))
        if Nprtcl > 0: 
            xidx = np.round(coordrecv[:,0]).astype(np.int32)
            yidx = np.round(coordrecv[:,1]).astype(np.int32)
            zidx = np.round(coordrecv[:,2]).astype(np.int32)
            ypos = coordrecv[:,-2]
            zpos = coordrecv[:,-1]
            Npass = ufield.shape[-1]
            
            return _calc_usend_numba(ufield, xidx, yidx,zidx,ypos, zpos,usend, Nprtcl,Npass, self.dy , self.dz, self.cosorder.size , self.Y , self.Z,self.interpdim)

        else: return usend
        
    #@profiler
    def calc_uadd_scalar(self,cfield,coordrecv):
        Nprtcl = coordrecv.shape[0]
        if Nprtcl > 0: 
            xidx = np.round(coordrecv[:,0]).astype(np.int32)
            yidx = np.round(coordrecv[:,1]).astype(np.int32)
            zidx = np.round(coordrecv[:,2]).astype(np.int32)
            ypos = coordrecv[:,-2]
            zpos = coordrecv[:,-1]
            udat = coordrecv[:,self.d]
            Npass = cfield.shape[-1]
            
            return _calc_uadd_numba_scalar(cfield, xidx, yidx,zidx,ypos, zpos,udat, Nprtcl,Npass,self.dx, self.dy,  self.dz, self.cosorder.size, self.Y, self.Z)   
        else:
            return cfield*1.0
    #@profiler
    def calc_uadd_vector(self,cfield,coordrecv):
        Nprtcl = coordrecv.shape[0]
        if Nprtcl > 0: 
            xidx = np.round(coordrecv[:,0]).astype(np.int32)
            yidx = np.round(coordrecv[:,1]).astype(np.int32)
            zidx = np.round(coordrecv[:,2]).astype(np.int32)
            ypos = coordrecv[:,-2]
            zpos = coordrecv[:,-1]
            udat = coordrecv[:,self.d:self.d + self.exterpdim]
            Npass = cfield.shape[-1]
            
            return _calc_uadd_numba_vector(cfield, xidx, yidx,zidx,ypos, zpos,udat, Nprtcl,Npass,self.dx, self.dy,  self.dz, self.cosorder.size, self.Y, self.Z,self.exterpdim)   
            

        else: return cfield*1.0

    

    
    def update_intrinsic(self):
        """Calculates the intrinsic variables"""
        
        self.st = (self.coord[:,-1]/self.factor)**(2/3.)
        self.rhs = 0.0*self.coord
        self.rb = (self.coord[:,-1]*(0.75/np.pi)/self.rhop)**(1./3.)
        
    #@profiler
    def particle_exchange(self,coord):
        """Change it to all to all v at a later time"""
        outcond = (coord[:,0] < self.startdom) + (coord[:,0] >= self.enddom) #! The index before which the particles are to be sent
        cond = ~outcond
        sendbuf = coord[outcond].copy()
        
        """Sort the left and the right sends according to their x and using a for loop count the send """
            
        
        cdim = sendbuf.shape[-1]
        
        
        sendbuf = sendbuf[(sendbuf[:,0]%self.L).argsort()]
        sendcounts = np.zeros(self.num_process, dtype = np.int32)
        
        for i in range(self.num_process):
            sendcounts[i] = np.sum(((sendbuf[:,0]%self.L)>=self.glob_startdom[i])*(((sendbuf[:,0]%self.L)< self.glob_enddom[i])))*cdim
        
        sendbuf = sendbuf.ravel()

        sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
        recvcounts = np.empty(self.num_process,dtype = np.int32)
        self.comm.Alltoall(sendcounts,recvcounts)

        rdispls = [0] + list(np.cumsum(recvcounts)[:-1]) 
        recvbuf = np.zeros(sum(recvcounts),dtype = np.float64)
        self.comm.Alltoallv([sendbuf, sendcounts,sdispls, MPI.DOUBLE],[recvbuf,recvcounts,rdispls,MPI.DOUBLE])
        recvbuf = recvbuf.reshape((-1,cdim))


        coord = np.concatenate((coord[cond],recvbuf),axis = 0)
        coord[:,:self.d] %= self.L
        
        # ------------------- cleaning up ------------------- #
        self.comm.Barrier()
        del outcond,cond, sendbuf,recvbuf
        # gc.collect()
        # --------------------------------------------------- #
        
        return coord
    
    #@profiler
    def send(self,x,args):
        "Input: x, args = [y,z,....] where y,z are the additional arguments to be sent to the next process"
        outcond = (x[:,0] < self.startdom) + (x[:,0] >= self.enddom) #! The index before which the particles are to be sent
        cond = ~outcond
        sendbuf = x[outcond].copy()
        
        """Sort the left and the right sends according to their x and using a for loop count the send """
        
        

        for i in range(len(args)): 
            sendbuf = np.concatenate((sendbuf,args[i][outcond]),axis = 1)
            
            
        
        cdim = sendbuf.shape[-1]
        
        
        sendbuf = sendbuf[(sendbuf[:,0]%self.L).argsort()]
        sendcounts = (np.round(np.searchsorted(sendbuf[:,0]%self.L,self.glob_enddom,side = 'left')-np.searchsorted(sendbuf[:,0]%self.L,self.glob_startdom,side = 'left'))*cdim).astype(np.int32)

        sendbuf = sendbuf.ravel()

        sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
        recvcounts = np.empty(self.num_process,dtype = np.int32)
        self.comm.Alltoall(sendcounts,recvcounts)

        rdispls = [0] + list(np.cumsum(recvcounts)[:-1]) 
        recvbuf = np.zeros(sum(recvcounts),dtype = np.float64)
        self.comm.Alltoallv([sendbuf, sendcounts,sdispls, MPI.DOUBLE],[recvbuf,recvcounts,rdispls,MPI.DOUBLE])
        recvbuf = recvbuf.reshape((-1,cdim))

        x = np.concatenate((x[cond],recvbuf[:,:x.shape[-1]]),axis = 0)
        x[:,:self.d] %= self.L
        count = x.shape[-1]
        for i in range(len(args)): 
            args[i] = np.concatenate((args[i][cond],recvbuf[:,count:count + args[i].shape[-1]]),axis = 0)
            count += args[i].shape[-1]
            
        # ------------------- cleaning up ------------------- #
        # self.comm.Barrier()
        del outcond,cond, sendbuf,recvbuf
        # gc.collect()
        # --------------------------------------------------- #
        return x,args
    
    
    def uinterp(self,coord,u): #! old: modify.
        """Interpolates the u field at the location of the particle.
        If the particle is at the edge of the domain, it gets data from the neighboring domains. 
        For multiplying with neighboring ones, this uses the M matrix.
        """
        pos = coord[:,:self.d]
        idx = np.round(pos//self.dx).astype(np.int32)
        idx[:,0] -= int(round(self.X[self.sx][0]//self.dx))
        delx = (pos%self.dx)/self.dx    
        poly = delx[...,None]**self.nums
        self.interpmat *=  0.0
        for i in range(self.interporder):
            slx = (idx[:,0]- self.interporder//2+ 1 + i)   
            leftcond = slx < 0
            rightcond = slx >= self.Np
            cond = ~(leftcond + rightcond)
            # ------------------- send the indices ------------------- #

            idxsend = np.concatenate((
                np.concatenate((slx[leftcond,None],idx[leftcond,1:],delx[leftcond,1:]),axis = 1),
                np.concatenate((slx[rightcond,None],idx[rightcond,1:],delx[rightcond,1:]),axis = 1)
            ),axis = 0).ravel()
            # --------------------------------------------------------- #



            
            sendcounts = np.array([leftcond.sum()*(2*self.d - 1), rightcond.sum()*(2*self.d -1)],dtype = int) 
            sdispls = [0,sendcounts[0]]
            recvcounts = np.empty(2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0,recvcounts[0]]
            idxrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([idxsend,sendcounts,sdispls, MPI.DOUBLE],[idxrecv,recvcounts,rdispls,MPI.DOUBLE])
            
            
            # Recieve the xidx,yidx list
            idxrecv = idxrecv.reshape((-1,(2*self.d - 1)))
            idxrecv[:,0] = np.round(idxrecv[:,0])%self.Np
            idxrecv[:,:self.d] = np.round(idxrecv[:,:self.d])
            yidxshift = (idxrecv[:,1,None].astype(np.int32) + np.arange(-self.interporder//2+1, self.interporder//2 +1))%self.N
            zidxshift = (idxrecv[:,2,None].astype(np.int32) + np.arange(-self.interporder//2+1, self.interporder//2 +1))%self.N
            polyy = idxrecv[:,-2, None]**self.nums
            polyz = idxrecv[:,-1, None]**self.nums
            # Calculate the reduced u matrix
            usend = np.moveaxis(np.einsum('...jk,jq,kr,...q,...r->...',u[...,idxrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]], self.Mmat, self.Mmat,polyy,polyz), [0,1],[1,0]) # (Nprtcl,d) matrix. The last d for the number of components index
            # send the reduced u
            sendcounts = np.round([recvcounts[0]*self.d/(2*self.d - 1), recvcounts[1]*self.d/(2*self.d - 1)]).astype(np.int32)
            sdispls = [0,sendcounts[0]]
            recvcounts = np.empty(2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0,recvcounts[0]]
            urecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([usend, sendcounts,sdispls, MPI.DOUBLE],[urecv,recvcounts,rdispls,MPI.DOUBLE])
            
            # Receive u form the matrix
            urecv = urecv.reshape((-1,self.d))
            split = recvcounts[0]//self.d

            # Add to the value

            self.interpmat[leftcond] += urecv[:split]*np.einsum('p,...p->...',self.Mmat[i],poly[leftcond,0])[:,None]

            self.interpmat[rightcond] += urecv[split:]*np.einsum('p,...p->...',self.Mmat[i],poly[rightcond,0])[:,None]

            
            for j in range(self.interporder): 
                sly = (idx[cond,1]-self.interporder//2+1 + j)%self.N #! For saving computations 
                for k in range(self.interporder):
                    slz = (idx[cond,2]-self.interporder//2+1 + k)%self.N
                    temparr = np.einsum('p,q,r,...p,...q,...r->...',self.Mmat[i],self.Mmat[j],self.Mmat[k],poly[cond,0,:],poly[cond,1,:],poly[cond,2,:])
                    self.interpmat[cond]  += np.moveaxis(u[...,slx[cond],sly,slz],[0,1],[1,0])*temparr[:,None]
        
        return self.interpmat


    def uinterp_cosine(self,coord,u):
        """Interpolates the u field at the location of the particle.
        If the particle is at the edge of the domain, it gets data from the neighboring domains. 
        For multiplying with neighboring ones, this uses the M matrix.
        """
        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np

        self.interpmat *=  0.0
        order = 4
                
        sly = (idx[:,1,None] + self.cosorder)%self.N
        slz = (idx[:,2,None] + self.cosorder)%self.N

        dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
        
        
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 
            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
            
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])
            
            
            # Recieve the xidx,yidx list
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])

            # Calculate the reduced u matrix
            yidxshift = (np.round(coordrecv[:,1,None]).astype(np.int32) + self.cosorder)%self.N
            zidxshift = (np.round(coordrecv[:,2,None]).astype(np.int32) + self.cosorder)%self.N
            dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            usend = np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]],dyshift,dzshift), [0,1],[1,0]).ravel() # (Nprtcl,d) matrix, d is the components of u.
            # send the reduced u

            sendcounts = np.round(recvcounts*self.d/cdim).astype(np.int32)
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.zeros(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            urecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([usend, sendcounts,sdispls, MPI.DOUBLE],[urecv,recvcounts,rdispls,MPI.DOUBLE])
            # Receive u form the matrix
            urecv = urecv.reshape((-1,self.d))

            # Add to the value


            self.interpmat[outcond.nonzero()[0][sortarg]] += urecv*dxshift[outcond.nonzero()[0][sortarg],None]

                
            
            self.interpmat[cond]  += np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]],dyshift_core[cond],dzshift_core[cond]), [0,1],[1,0])*dxshift[cond,None]
                

        return self.interpmat
    
    
    #@profiler
    def interp_cosine(self,coord,u):
        """Interpolates the u field at the location of the particle.
        If the particle is at the edge of the domain, it gets data from the neighboring domains. 
        For multiplying with neighboring ones, this uses the M matrix.
        """
        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np

        self.interpmat *=  0.0
        order = 4
        
        # sly = (idx[:,1,None] + self.cosorder)%self.N
        # slz = (idx[:,2,None] + self.cosorder)%self.N

        # dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        # dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
        
        
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 
            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
            
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])
            del coordsend
            
            # Recieve the xidx,yidx list
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])
            # Calculate the reduced u matrix
            # yidxshift = (np.round(coordrecv[:,1,None]).astype(np.int32) + self.cosorder)%self.N
            # zidxshift = (np.round(coordrecv[:,2,None]).astype(np.int32) + self.cosorder)%self.N
            # dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            # dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            # usend = np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,:,None],zidxshift[...,None,:]],dyshift,dzshift), [0,1],[1,0]).ravel() # (Nprtcl,d) matrix, d is the components of u.
            # # send the reduced u
            usend = self.calc_usend(u,coordrecv).ravel()
            del coordrecv
            sendcounts = np.round(recvcounts*self.interpdim/cdim).astype(np.int32)
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.zeros(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            urecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([usend, sendcounts,sdispls, MPI.DOUBLE],[urecv,recvcounts,rdispls,MPI.DOUBLE])
            # Receive u form the matrix
            urecv = urecv.reshape((-1,self.interpdim))
            del usend
            
            # Add to the value


            self.interpmat[outcond.nonzero()[0][sortarg]] += urecv*dxshift[outcond.nonzero()[0][sortarg],None]
            del urecv,sortarg
                
            
            # self.interpmat[cond]  += np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,slx[cond,None,None], sly[cond,:.None],slz[cond,None,:]],dyshift_core[cond],dzshift_core[cond]), [0,1],[1,0])*dxshift[cond,None]
            coordrecv = np.concatenate((slx[cond,None],idx[cond,1:],pos[cond,1:]),axis = 1)
            self.interpmat[cond] += self.calc_usend(u,coordrecv)*dxshift[cond,None]    
            # ------------------- cleaning up ------------------- #
            del cond,outcond,dxshift,slx,coordrecv
            # gc.collect()
            # --------------------------------------------------- #
        
        # ------------------- cleaning up ------------------- #
        # self.comm.Barrier()
        del pos,idx
        # gc.collect()
        # --------------------------------------------------- #
        return self.interpmat
    
    #@profiler
    def exterp_cosine_scalar(self,coord,c):
        """Adds exterpmat to the existing c"""
        
        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np
        order = 4
        c *= 0.0
        
        # sly = (idx[:,1,None] + self.cosorder)%self.N
        # slz = (idx[:,2,None] + self.cosorder)%self.N

        # dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        # dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
        
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 

            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],self.exterpmat[outcond]*dxshift[outcond,None],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
            
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])
            del coordsend,sortarg
            
            # ------------------- Recieve the u shape ------------------ #
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])

            # ---------------------------------------------------------- #

            
            # ------------------- Calculate the reduced u matrix ------------------ #
            # yidxshift = (coordrecv[:,1,None].astype(np.int32) + self.cosorder)%self.N
            # zidxshift = (coordrecv[:,2,None].astype(np.int32) + self.cosorder)%self.N
            # dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            # dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            # np.add.at(c,(...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]),coordrecv[:,self.d][:,None,None]*dyshift[...,None]*dzshift[...,None,:]/(self.dx *self.dy *self.dz))
            c[:] = self.calc_uadd_scalar(c,coordrecv)
            del coordrecv
            # ---------------------------------------------------------------------- #
            
            # np.add.at(c,(...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]),(self.exterpmat[cond,0]*dxshift[cond])[:,None,None]*dyshift_core[cond,:,None]*dzshift_core[cond,None,:]/(self.dx *self.dy *self.dz))
            coordrecv = np.concatenate((slx[cond,None],idx[cond,1:],self.exterpmat[cond]*dxshift[cond,None],pos[cond,1:]),axis = 1)
            c[:] = self.calc_uadd_scalar(c,coordrecv)
        
            
            # ------------------- cleaning up ------------------- #
            
            del slx, dxshift,outcond, cond
            # gc.collect()
            # --------------------------------------------------- #
            
        
        # ------------------- cleaning up ------------------- #
        # self.comm.Barrier()
        del pos, idx
        # gc.collect()
        # --------------------------------------------------- #
        
        return c
        
    
    def exterp_cosine_vector(self,coord,c):
        """Adds exterpmat to the existing c"""

        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np
        order = 4
        c *= 0.0
        # sly = (idx[:,1,None] + self.cosorder)%self.N
        # slz = (idx[:,2,None] + self.cosorder)%self.N

        # dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        # dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
  
        # ccomp = np.prod(self.exterpdim.shape[1:]) #! Determines the component of the exterpdim.
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 
            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],self.exterpmat[outcond]*dxshift[outcond,None],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
                
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])
            del coordsend, outcond, sortarg
            # ------------------- Recieve the u shape ------------------ #
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])
            # ---------------------------------------------------------- #

            
            # ------------------- Calculate the reduced u matrix and add to the c matrix ------------------ #
            # yidxshift = (coordrecv[:,1,None].astype(np.int32) + self.cosorder)%self.N
            # zidxshift = (coordrecv[:,2,None].astype(np.int32) + self.cosorder)%self.N
            # dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            # dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            # np.add.at(c,(...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]),np.moveaxis(coordrecv[:,self.d:self.d+ccomp],[0,1],[1,0])[...,None,None]*dyshift[...,None]*dzshift[...,None,:])
            
            c[:] = self.calc_uadd_vector(c,coordrecv)
            del coordrecv
            # ----------------------------------------------------------------------------------------------- #     
            
            # np.add.at(c,(...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]),np.moveaxis(self.exterpmat[cond]*dxshift[cond,None],[0,1],[1,0])[...,None,None]*dyshift_core[None, cond,:,None]*dzshift_core[None,cond,None,:])
            coordrecv = np.concatenate((slx[cond,None],idx[cond,1:],self.exterpmat[cond]*dxshift[cond,None],pos[cond,1:]),axis = 1)
            c[:] = self.calc_uadd_vector(c,coordrecv)
        
            # ------------------- cleaning up ------------------- #
            del slx, dxshift, cond
            # gc.collect()
            # --------------------------------------------------- #
            
        
        # ------------------- cleaning up ------------------- #
        self.comm.Barrier()
        del pos, idx
        # gc.collect()
        # --------------------------------------------------- #
          
        return c

    def interp_exterp_cosine_scalar(self,coord,u, c):
        
        "Interpolates u to interpmat, exterpolates exterpdim to c using cosine weights. C and exterpdims are scalar"
        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np
        self.interpmat *=  0.0
        c *=0.0
        order = 4
        
        sly = (idx[:,1,None] + self.cosorder)%self.N
        slz = (idx[:,2,None] + self.cosorder)%self.N

        dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
        
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 
            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],self.exterpmat[outcond]*dxshift[outcond,None],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
            
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1]) 
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])

            
            # ------------------- Recieve the u shape ------------------ #
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])

            # ---------------------------------------------------------- #

            
            # ------------------- Calculate the reduced u matrix ------------------ #
            yidxshift = (coordrecv[:,1,None].astype(np.int32) + self.cosorder)%self.N
            zidxshift = (coordrecv[:,2,None].astype(np.int32) + self.cosorder)%self.N
            dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            np.add.at(c,(...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]),coordrecv[:,self.d][:,None,None]*dyshift[...,None]*dzshift[...,None,:]/(self.dx *self.dy *self.dz))
            usend = np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]],dyshift,dzshift), [0,1],[1,0]).ravel() # (Nprtcl,d) matrix, d is the components of u.
            # ---------------------------------------------------------------------- #

            # ------------------- send the reduced u ------------------ #
            sendcounts = np.round(recvcounts*self.d/cdim).astype(np.int32)
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.zeros(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            urecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([usend, sendcounts,sdispls, MPI.DOUBLE],[urecv,recvcounts,rdispls,MPI.DOUBLE])

            
            # ----------------------------------------------------------- #

            
            # Receive u form the matrix
            urecv = urecv.reshape((-1,self.d))



            # Add to the value

            self.interpmat[outcond.nonzero()[0][sortarg]] += urecv*dxshift[outcond.nonzero()[0][sortarg],None]
            
                        

                
            self.interpmat[cond]  += np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]],dyshift_core[cond],dzshift_core[cond]), [0,1],[1,0])*dxshift[cond,None]
            
            np.add.at(c,(...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]),(self.exterpmat[cond,0]*dxshift[cond])[:,None,None]*dyshift_core[cond,:,None]*dzshift_core[cond,None,:]/(self.dx *self.dy *self.dz))
        
        return self.interpmat,c
    
    
    def interp_exterp_cosine_vector(self,coord,u, c):
        "Interpolates u to interpmat, exterpolates exterpdim to c using cosine weights.C and exterpdims are vectors"
        pos = coord[:,:self.d]
        idx = (pos//self.dx).astype(np.int32)
        idx[:,0] %= self.Np
        self.interpmat *=  0.0
        c *= 0.0
        order = 4
        
        sly = (idx[:,1,None] + self.cosorder)%self.N
        slz = (idx[:,2,None] + self.cosorder)%self.N

        dyshift_core = (1 + np.cos((pos[:,-2,None] - self.Y[sly])*(np.pi/(2*self.dy))))/4.0
        dzshift_core = (1 + np.cos((pos[:,-1,None] - self.Z[slz])*(np.pi/(2*self.dz))))/4.0
  
        ccomp = np.prod(self.exterpdim.shape[1:]) #! Determines the component of the exterpdim.
        for i in range(order):
            slx = (idx[:,0]-order//2+1 + i)
            dxshift =  (1 + np.cos((pos[:,0] - self.X[slx%self.N])*(np.pi/(2*self.dx))))/4.0
            outcond = (slx < 0) + (slx >= self.Np)
            cond = ~outcond
            #? Send the xidx,yidx,zidx list 
            coordsend =  np.concatenate((slx[outcond,None],idx[outcond,1:],self.exterpmat[outcond]*dxshift[outcond,None],pos[outcond,1:]),axis = 1)
            sortarg = coordsend[:,0].argsort()
            coordsend = coordsend[sortarg]
            sendcounts = np.zeros(self.nneighbors*2, dtype = np.int32)
            cdim = coordsend.shape[-1]
            
            for ii in range(self.nneighbors):
                sendcounts[ii] = np.sum((coordsend[:,0] < -(self.nneighbors - ii -1)*self.Np)*(coordsend[:,0] >=-(self.nneighbors - ii)*self.Np))*cdim
                sendcounts[self.nneighbors + ii] = np.sum((coordsend[:,0] >= (1 + ii)*self.Np)*(coordsend[:,0] < (2 + ii)*self.Np))*cdim
                
            coordsend = coordsend.ravel()
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.empty(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            coordrecv = np.zeros(sum(recvcounts),dtype = np.float64)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            self.cart_comm.Neighbor_alltoallv([coordsend, sendcounts,sdispls, MPI.DOUBLE],[coordrecv,recvcounts,rdispls,MPI.DOUBLE])
            
            # ------------------- Recieve the u shape ------------------ #
            coordrecv = coordrecv.reshape((-1,cdim))
            coordrecv[:,0] %= self.Np
            coordrecv[:,:self.d] = np.round(coordrecv[:,:self.d])
            # ---------------------------------------------------------- #

            
            # ------------------- Calculate the reduced u matrix and add to the c matrix ------------------ #
            yidxshift = (coordrecv[:,1,None].astype(np.int32) + self.cosorder)%self.N
            zidxshift = (coordrecv[:,2,None].astype(np.int32) + self.cosorder)%self.N
            dyshift = (1 + np.cos((coordrecv[:,-2,None] - self.Y[yidxshift])*(np.pi/(2*self.dy))))/4.0
            dzshift = (1 + np.cos((coordrecv[:,-1,None] - self.Z[zidxshift])*(np.pi/(2*self.dz))))/4.0
            np.add.at(c,(...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]),np.moveaxis(coordrecv[:,self.d:self.d+ccomp],[0,1],[1,0])[...,None,None]*dyshift[...,None]*dzshift[...,None,:])
            usend = np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,coordrecv[:,0,None,None].astype(np.int32), yidxshift[...,None],zidxshift[...,None,:]],dyshift,dzshift), [0,1],[1,0]).ravel() # (Nprtcl,d) matrix, d is the components of u.
            # ----------------------------------------------------------------------------------------------- #

            # ------------------- send the reduced u ------------------ #
            sendcounts = np.round(recvcounts*self.d/cdim).astype(np.int32)
            sdispls = [0] + list(np.cumsum(sendcounts)[:-1])
            recvcounts = np.zeros(self.nneighbors*2,dtype = 'i')
            self.cart_comm.Neighbor_alltoall(sendcounts,recvcounts)
            rdispls = [0] + list(np.cumsum(recvcounts)[:-1])
            urecv = np.zeros(sum(recvcounts),dtype = np.float64)
            self.cart_comm.Neighbor_alltoallv([usend, sendcounts,sdispls, MPI.DOUBLE],[urecv,recvcounts,rdispls,MPI.DOUBLE])

            
            # ----------------------------------------------------------- #

            
            # Receive u form the matrix
            urecv = urecv.reshape((-1,self.d))


            # Add to the value
            self.interpmat[outcond.nonzero()[0][sortarg]] += urecv*dxshift[outcond.nonzero()[0][sortarg],None]
                
            self.interpmat[cond]  += np.moveaxis(np.einsum('...jqr,jq,jr->...j',u[...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]],dyshift_core[cond],dzshift_core[cond]), [0,1],[1,0])*dxshift[cond,None]
            
            
            np.add.at(c,(...,slx[cond,None,None], sly[cond,:,None],slz[cond,None,:]),np.moveaxis(self.exterpmat[cond]*dxshift[cond,None],[0,1],[1,0])[...,None,None]*dyshift_core[None, cond,:,None]*dzshift_core[None,cond,None,:]/(self.dx *self.dy *self.dz))
            
        return self.interpmat,c


    
    def pRHS(self,t,coord,u,us,c,fc,sump):
        """
        Evolves the step from t to t + h using Euler. Returns the sump, dot{coord}, and field c. Also redistributes according to coord.
        """

        coord,[self.coord, self.interpmat,self.exterpmat,self.prtclid,sump] = self.send(coord,[self.coord, self.interpmat,self.exterpmat,self.prtclid,sump])
        self.st = (coord[:,-1]/self.factor)**(2./3.)
        self.rhs = 0.0*coord
        
        self.interpmat = self.interp_cosine(coord,np.concatenate((u,us,c[None,...]), axis = 0)) #! Interpmat has u,us and c => 2*d+1 components.
        
        self.exterpmat[:,0] = self.growthfactor*(1 + (self.st_s/self.st)**0.5)**2 *coord[:,-1]**(2./3.)* self.interpmat[:,-1]*np.linalg.norm(coord[:,self.d:self.d*2] - self.interpmat[:,self.d: 2*self.d],axis = 1) #! exterpmat is the number of small droplets swept per unit time.
        fc[:] = self.exterp_cosine_scalar(coord, fc)

        
        self.rhs[:,:self.d] = coord[:,self.d:2*self.d]
        self.rhs[:,self.d:2*self.d] =  (self.interpmat[:,:self.d]- coord[:,self.d:2*self.d])/(self.st[:,None] *self.tau_eta)  + ((self.decelerationfactor*(1 + (self.st_s/self.st)**0.5)**2 /coord[:,-1]**(1./3.))* self.interpmat[:,-1]*np.linalg.norm(coord[:,self.d:self.d*2] - self.interpmat[:,self.d: 2*self.d],axis = 1) )[:,None]*(self.interpmat[:,self.d: 2*self.d] - coord[:,self.d:self.d*2]) 
        self.rhs[:,2*self.d - 1] -= self.g #! Adding gravity in the appropriate units.
        self.rhs[:,-1] = self.exterpmat[:,0]*self.rhop*self.n_cs_factor
        
        return sump, self.rhs, fc
        
        
    def pRHS_inertial(self,t,coord,u,sump):
        """
        Evolves the step from t to t + h using Euler. Returns the sump, dot{coord}. 
        """

        coord,[self.coord, self.interpmat,self.prtclid,sump] = self.send(coord,[self.coord, self.interpmat,self.prtclid,sump])
        self.st = (coord[:,-1]/self.factor)**(2/3.)
        self.rhs = 0.0*coord

        self.interpmat = self.interp_cosine(coord,np.concatenate((u,u[0][None,:]),axis = 0)) #! Interpmat has u => d+1 components.
        # self.interpmat = self.interp_cosine(coord,np.concatenate(u,u[0][None,...],axis = 0)) #! Interpmat has u => d components.
        

        
        self.rhs[:,:self.d] = coord[:,self.d:2*self.d]
        self.rhs[:,self.d:2*self.d] =  (self.interpmat[:,:self.d]- coord[:,self.d:2*self.d])/(self.st[:,None] *self.tau_eta)  
        
        self.rhs[:,2*self.d - 1] -= self.g #! Adding gravity in the appropriate units.
        return sump, self.rhs
    
    
    def stoch_updt(self,dt,t,coord,us,modA, nprtcl,rs):
        """
        Evolves the step from t to t + h using stochastic Euler. Returns the modified coord which is the stochastically evolved coord. Uses the original evolved state
        """

        coord,[self.coord, self.interpmat,self.prtclid] = self.send(coord,[self.coord, self.interpmat,self.prtclid])
        self.st = (coord[:,-1]/self.factor)**(2/3.)
        self.rb = (coord[:,-1]*(0.75/np.pi)/self.rhop)**(1./3.)
        self.rhs = 0.0*coord
        
        self.interpmat = self.interp_cosine(coord,np.concatenate((us,modA[None,...]), axis = 0)) #! Interpmat has u => d components.

        volume = np.pi*(self.rb + rs)**2*np.linalg.norm(coord[:,self.d:2*self.d]  - self.interpmat[:,:self.d],axis = -1)*dt
        plambda = nprtcl/(2*np.pi)**3*volume*self.interpmat[:,-1]
        ncoll = np.random.poisson(plambda)
        nprtcl -= ncoll.sum() 
        # print(volume.mean(),self.dx**3)
        ms = self.rhop*np.pi*(rs)**3*(4.0/3.0)
        
        self.rhs[:,-1] = ncoll*ms #! This is delta m
        self.rhs[:,self.d:2*self.d] =  (self.rhs[:,-1]/(coord[:,-1] + self.rhs[:,-1]))[:,None]*(self.interpmat[:,:self.d] - coord[:,self.d:2*self.d]  )

        return  coord + self.rhs

    
def RK4(t,h,prtcl, u,us,c):
    """Template on how to evolve the particle + flow system"""
    sum = 1.0*prtcl.coord
    sum_u = 1.0*u
    sum_us = 1.0*us
    sum_c = 1.0*c
    
    sum, kp, fc = prtcl.pRHS(t, prtcl.coord, u,us,c,fc,sum)
    sum += h/6.0*kp
    ku,kus,kc = RHS(t, u,us, c,fc)
    sum_u += h/6.0*ku
    sum_us += h/6.0*kus
    sum_c += h/6.0*kc
    
    sum, kp, fc  = prtcl.pRHS(t+ h/2,prtcl.coord + h/2* kp,u + ku*h/2, us + kus*h/2,c + kc*h/2,fc,sum)
    sum += h/3.0*kp
    ku,kus,kc = RHS(t, u + ku*h/2,us + kus *h/2, c + kc*h/2, fc)
    sum_u += h/3.0*ku
    sum_us += h/3.0*kus
    sum_c+= h/3.0*kc
    
    sum, kp, fc = prtcl.pRHS(t + h/2, prtcl.coord + h/2*kp, u+ ku * h/2, us+ kus * h/2,c+ kc * h/2,fc,sum)
    sum += h/3.0*kp
    ku = RHS(t, u + ku*h/2, us + kus*h/2,c + kc*h/2,fc)
    sum_u = h/3.0*ku
    sum_us = h/3.0*kus
    sum_c = h/3.0*kc
    
    sum, kp, fc = prtcl.pRHS(t + h, prtcl.coord + h*kp, u + ku*h, us + kus*h, c + kc*h, fc,sum)
    sum += h/6.0*kp
    ku = RHS(t, u + ku*h,us + kus*h,c + kc*h, fc)
    sum_u = h/6.0*ku
    sum_us = h/6.0*kus
    sum_c = h/6.0*kc
    
    return sum*1.0, sum_u*1.0,sum_us*1.0,sum_c*1.0

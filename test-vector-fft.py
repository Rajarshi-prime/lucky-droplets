import numpy as np 
from scipy.fft import fft ,  rfft2
from mpi4py import MPI

## ––––––––––––––-MPI things––––––––––––––
N = 512
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
Np = N//num_process
Nf = N//2 + 1
## ––––––––––––––––––––––––––––––––––––––-

arr_temp_fr = np.zeros((Np, N, Nf), dtype = np.complex128)
arr_mpi = np.zeros((num_process, Np, Np, Nf), dtype = np.complex128)

arr_temp_fr3 = np.zeros((3, Np, N, Nf), dtype = np.complex128)
arr_mpi3 = np.zeros((num_process, 3, Np, Np, Nf), dtype = np.complex128)
recv3 = np.zeros_like(arr_mpi3)


## ––––––FFT one component at a time, as in the solver––––––- 
def rfft_mpi(u, fu):
    arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

## ––––––FFT of all three components in one Alltoall––––––- 
def rfft_mpi_vec(u, fu):
    arr_temp_fr3[:] = rfft2(u, axes = (2, 3))
    arr_mpi3[:] = np.moveaxis(np.reshape(arr_temp_fr3, (3, Np, num_process, Np, Nf)), 2, 0)
    comm.Alltoall([arr_mpi3, MPI.DOUBLE_COMPLEX], [recv3, MPI.DOUBLE_COMPLEX])
    fu.reshape(3, num_process, Np, Np, Nf)[:] = np.moveaxis(recv3, 1, 0) #! the component axis arrives inside the rank axis
    fu[:] = fft(fu, axis = 1)
    return fu


## ––––––Timing the two versions––––––-
nrep = 50
u = np.random.randn(3, Np, N, N)
fu_s = np.zeros((3, N, Np, Nf), dtype = np.complex128)
fu_v = np.zeros_like(fu_s)

rfft_mpi(u[0], fu_s[0])
rfft_mpi(u[1], fu_s[1])
rfft_mpi(u[2], fu_s[2])
rfft_mpi_vec(u, fu_v)

comm.Barrier()
t0 = MPI.Wtime()
for i in range(nrep):
    rfft_mpi(u[0], fu_s[0])
    rfft_mpi(u[1], fu_s[1])
    rfft_mpi(u[2], fu_s[2])
time_scalar = comm.allreduce(MPI.Wtime() - t0, op = MPI.MAX)/nrep

comm.Barrier()
t0 = MPI.Wtime()
for i in range(nrep):
    rfft_mpi_vec(u, fu_v)
time_vector = comm.allreduce(MPI.Wtime() - t0, op = MPI.MAX)/nrep

diff = comm.allreduce(np.max(np.abs(fu_s - fu_v)), op = MPI.MAX)

if rank == 0:
    print(f"scalar time per call : {time_scalar}")
    print(f"vector time per call : {time_vector}")
    print(f"speedup (scalar/vector) : {time_scalar/time_vector}")
    print(f"max abs difference : {diff}")

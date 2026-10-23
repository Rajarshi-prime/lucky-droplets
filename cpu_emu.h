// ===========================================================================
//  cpu_emu.h -- CPU emulation layer for forced_ns_ck54_lowmem.cu and
//  forced_ns_droplets.cu under -DCPU_EMU. Not part of any GPU build.
//
//  Not compiled in this sandbox (no clang++, no fftw3 available here). Written
//  purely from how forced_ns_ck54_lowmem.cu uses these names: the LAUNCH
//  macro (emu_launch + __global__/__device__ kernel bodies run as a plain
//  grid-stride loop), the CUDA runtime/cuFFT/cuRAND calls grepped out of that
//  file, and block_add's own CPU_EMU branch (which bypasses __shared__ and
//  atomicAdd entirely, so those only need to compile, not race).
//
//  Sits beside the .cu files on purpose. Both include it as "cpu_emu.h", a
//  quoted include, which searches the including file's own directory first --
//  so no -I flag is needed. (forced_ns_ck54_lowmem.cu's header still documents
//  -Itests from when this lived there; that is now a harmless no-op.)
//
//  BUILD:
//    clang++ -O2 -std=c++17 -DCPU_EMU -DNS_DOUBLE -DNS_N=32 \
//            -x c++ forced_ns_droplets.cu -lfftw3 -o forced_ns_cpu
// ===========================================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <fftw3.h>

// ------------------------------------------------------------- vector types
struct double2 { double x, y; };
struct float2  { float  x, y; };

// ------------------------------------------------------------- kernel attrs
// Single translation unit, single thread at a time: these are no-ops.
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static            // only block_add's CUDA branch reads this,
                                      // and that branch is #ifndef CPU_EMU'd out

// ------------------------------------------------------------- thread index
// A "kernel" is an ordinary function run once per (blockIdx, threadIdx) pair
// by emu_launch, exactly as the CUDA back-end would launch it; GRID_STRIDE
// inside the kernel body does the rest. One flat grid dimension only -- the
// file never reads .y or .z.
struct EmuDim { long x; };
static EmuDim blockIdx{0}, threadIdx{0}, blockDim{0}, gridDim{0};

template <class F>
static inline void emu_launch(long grid, long block, F&& f)
{
    gridDim.x = grid; blockDim.x = block;
    for (long b = 0; b < grid; ++b) {
        blockIdx.x = b;
        for (long t = 0; t < block; ++t) {
            threadIdx.x = t;
            f();
        }
    }
}

// atomicAdd: serial emulator, no race possible. Returns the old value, as CUDA does.
static inline double atomicAdd(double* addr, double val) { double old = *addr; *addr += val; return old; }
static inline float  atomicAdd(float*  addr, float  val) { float  old = *addr; *addr += val; return old; }
static inline void __syncthreads() {}   // single-threaded: nothing to synchronise

// rsqrt: a CUDA device intrinsic with no host equivalent. k_add_force,
// k_finalize and k_velocity_rhs all call it on a double; without this the
// -DCPU_EMU build fails at the first of them.
static inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }

// ------------------------------------------------------------- CUDA runtime
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
typedef int cudaStream_t;
typedef int cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost };

static inline const char* cudaGetErrorString(cudaError_t) { return "cpu_emu: no CUDA error"; }
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = 0; return cudaSuccess; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

// device "memory" is plain host memory; cudaMemcpy* is memmove regardless of kind
static inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : 1; }
static inline cudaError_t cudaMallocHost(void** p, size_t n) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : 1; }
static inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memmove(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = 0) { std::memmove(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = 0) { std::memset(d, v, n); return cudaSuccess; }

struct cudaDeviceProp { char name[256] = "cpu_emu"; size_t totalGlobalMem = (size_t)1 << 34; };
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp(); return cudaSuccess; }
static inline cudaError_t cudaMemGetInfo(size_t* free_, size_t* total_) { *free_ = *total_ = (size_t)1 << 34; return cudaSuccess; }
static inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = 0; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0) { return cudaSuccess; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.f; return cudaSuccess; }

// ------------------------------------------------------------- cuFFT (fftw3)
// Batch-1 in-place R2C/C2R, same shapes as FFT1::plan() in the main file.
// Handles index a static plan table since cuFFT work-area plumbing has no
// FFTW equivalent; cufftSetAutoAllocation/SetWorkArea are accepted and ignored.
typedef int cufftResult;
static const cufftResult CUFFT_SUCCESS = 0;
typedef int cufftType;
static const cufftType CUFFT_R2C = 0x2a, CUFFT_C2R = 0x2c, CUFFT_D2Z = 0x6a, CUFFT_Z2D = 0x6c;
typedef double  cufftDoubleReal;
typedef double2 cufftDoubleComplex;
typedef float   cufftReal;
typedef float2  cufftComplex;
typedef int cufftHandle;

struct EmuPlan { fftw_plan p = nullptr; int rank = 0; int n[3] = {0,0,0}; cufftType type = 0; };
static std::unordered_map<cufftHandle, EmuPlan> g_emu_plans;
static cufftHandle g_emu_next_handle = 1;

static inline cufftResult cufftCreate(cufftHandle* h) { *h = g_emu_next_handle++; g_emu_plans[*h] = EmuPlan(); return CUFFT_SUCCESS; }
static inline cufftResult cufftDestroy(cufftHandle h) { auto it = g_emu_plans.find(h); if (it != g_emu_plans.end()) { if (it->second.p) fftw_destroy_plan(it->second.p); g_emu_plans.erase(it); } return CUFFT_SUCCESS; }
static inline cufftResult cufftSetAutoAllocation(cufftHandle, int) { return CUFFT_SUCCESS; }
static inline cufftResult cufftSetWorkArea(cufftHandle, void*) { return CUFFT_SUCCESS; }
static inline cufftResult cufftSetStream(cufftHandle, cudaStream_t) { return CUFFT_SUCCESS; }

// Only the exact shapes FFT1::plan() asks for (batch 1, in-place, real_t=double
// under -DNS_DOUBLE, which is the only CPU_EMU build in the header comment)
// need to work; this is not a general wrapper of cufftMakePlanMany.
static inline cufftResult cufftMakePlanMany(cufftHandle h, int rank, int* n,
    int* /*inembed*/, int /*istride*/, int /*idist*/,
    int* /*onembed*/, int /*ostride*/, int /*odist*/,
    cufftType type, int /*batch*/, size_t* workSize)
{
    EmuPlan& e = g_emu_plans[h];
    e.rank = rank; e.type = type;
    for (int i = 0; i < rank; ++i) e.n[i] = n[i];
    e.p = nullptr;      // built lazily on first Exec call, once the buffer pointer is known
    *workSize = 0;
    return CUFFT_SUCCESS;
}

static inline cufftResult cufftExecD2Z(cufftHandle h, cufftDoubleReal* in, cufftDoubleComplex* out)
{
    EmuPlan& e = g_emu_plans[h];
    if (!e.p) e.p = fftw_plan_dft_r2c_3d(e.n[0], e.n[1], e.n[2], in, (fftw_complex*)out, FFTW_ESTIMATE);
    fftw_execute_dft_r2c(e.p, in, (fftw_complex*)out);
    return CUFFT_SUCCESS;
}
static inline cufftResult cufftExecZ2D(cufftHandle h, cufftDoubleComplex* in, cufftDoubleReal* out)
{
    EmuPlan& e = g_emu_plans[h];
    if (!e.p) e.p = fftw_plan_dft_c2r_3d(e.n[0], e.n[1], e.n[2], (fftw_complex*)in, out, FFTW_ESTIMATE);
    fftw_execute_dft_c2r(e.p, (fftw_complex*)in, out);
    return CUFFT_SUCCESS;
}
// fp32 path is declared for completeness (NS_DOUBLE is the only path ever
// built here) but does not call fftwf_*; it is not reachable from -DCPU_EMU.
static inline cufftResult cufftExecR2C(cufftHandle, cufftReal*, cufftComplex*) { std::abort(); }
static inline cufftResult cufftExecC2R(cufftHandle, cufftComplex*, cufftReal*) { std::abort(); }

// ------------------------------------------------------------- cuRAND
// Not a cryptographic or statistical match to Philox -- only used for
// k_part_init's optional uniform-random seeding, which no checkpoint
// comparison depends on (positions come from the restart file).
struct curandStatePhilox4_32_10_t { unsigned long long s; };
static inline void curand_init(unsigned long long seed, unsigned long long subseq, unsigned long long, curandStatePhilox4_32_10_t* st)
{
    st->s = seed ^ (subseq * 0x9E3779B97F4A7C15ULL + 0x2545F4914F6CDD1DULL);
}
static inline double curand_uniform_double(curandStatePhilox4_32_10_t* st)
{
    st->s ^= st->s << 13; st->s ^= st->s >> 7; st->s ^= st->s << 17;
    return (double)(st->s >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
}

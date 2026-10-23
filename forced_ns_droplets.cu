// ===========================================================================
//  forced_ns_droplets.cu -- single-GPU fp64 port of forced-dns-sm-big.py
//  (forced NS, slow-manifold number density n, "big" inertial droplets),
//  built per PLAN-GPU-PORT.md. New file; forced_ns_ck54_lowmem.cu is not
//  modified, only extended in style: the compact-spectral storage, the
//  two-pass phase-shift dealiasing, the FFT1 work-area trick and the npy
//  I/O helpers below are the same technique as that file, rewritten here
//  because this is physically a different equation (no odd viscosity, a
//  different forcing normalisation, and NFIELDS extra transported scalars)
//  -- see PLAN-GPU-PORT.md section 1 for why CUDA C and classical RK4.
//
//  PHYSICS SOURCE: forced-dns-sm-big.py (velocity: forcing()/full_RHS's u
//  part; density: full_RHS's per-Stokes loop) and particles.py (pRHS,
//  collapsed to its interior/no-MPI branch -- PLAN-GPU-PORT.md section 6
//  notes). Struct layout, memory map and the droplet kernels themselves are
//  droplet_kernels.cuh's contract (G2); this file is the host driver.
//
//  NFIELDS is a runtime argument (plan section 3): the command line gives the
//  distinct stb values and the distinct init names, and the ensemble is their
//  outer product with init varying slowest -- NFIELDS = n_stb*n_init,
//  f = i_init*n_stb + i_stb, matching forced-dns-sm-big.py:41-43. Before any
//  large cudaMalloc, startup_refusal() computes the footprint and refuses with
//  the largest NFIELDS that would fit rather than failing inside cudaMalloc.
//
//  UNITS (plan section 4): coord[:,6] is a physical mass, never scaled by
//  M0; only n_cs_factor = M0/rhop carries M0. A *M0 or /M0 anywhere else in
//  the droplet path is a regression.
//
//  BUILD   run_gpu.py issues this itself, with the thirteen -DP_* overrides and
//          the CUDA paths filled in. By hand, with $CUDA the toolkit root:
//            $CUDA/bin/nvcc -O3 -std=c++17 -arch=sm_89 -DNS_N=256 -DNS_DOUBLE \
//              forced_ns_droplets.cu -o forced_ns_droplets \
//              -L$CUDA/targets/x86_64-linux/lib \
//              -Xlinker -rpath=$CUDA/targets/x86_64-linux/lib -lcufft
//          The -L and -rpath are not optional on a machine where libcufft is
//          not on the default library path: without -rpath the link succeeds
//          and the binary then dies at startup unable to find libcufft.so.
//          No -use_fast_math: it would reassociate the arithmetic this code is
//          checked against.
//  CPU     clang++ -O2 -std=c++17 -DCPU_EMU -DNS_DOUBLE -DNS_N=32 \
//              -x c++ forced_ns_droplets.cu -lfftw3 -o forced_ns_droplets_cpu
//
//  RUN     ./forced_ns_droplets --restart <time_dir> --stb 0.017,0.00425,0.068,0.10795 \
//              --init random,Qpos,Qneg,Qpos_high,Qneg_high --tend <t> --out <dir>
//          The two lists are the DISTINCT values; the ensemble is their outer
//          product, init slowest, so the line above is the full 20 fields.
//          The step size is adaptive and is not on the command line; --tend set
//          to the restart folder's own time takes exactly one step.
//          <time_dir> must already have been through unpack_checkpoint.py (it
//          reads uk/vk/wk.npy, n.npy, pos/vel/mass/prtclid.npy, not the MPI
//          code's .npz -- plan section 2, "unpack_checkpoint.py is
//          load-bearing"). Output goes to <dir>/time_<t>/ with the same two
//          subdirectory spellings, raw .npy.
// ===========================================================================

#ifdef CPU_EMU
#include "cpu_emu.h"
#else
#include <cufft.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <dirent.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)
#define CUFFT_CHECK(call) do { cufftResult e_ = (call); if (e_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT %s:%d: error %d\n", __FILE__, __LINE__, (int)e_); \
    exit(1); } } while (0)

#ifdef CPU_EMU
#define LAUNCH(k, g, b, ...) emu_launch((long)(g), (long)(b), [&] { k(__VA_ARGS__); })
#else
#define LAUNCH(k, g, b, ...) k<<<(g), (b), 0, g_stream>>>(__VA_ARGS__)
#endif

// ------------------------------------------------------------------ precision
#ifdef NS_DOUBLE
typedef double  real_t;
typedef double2 cplx_t;
static const cufftType FWD_TYPE = CUFFT_D2Z;
static const cufftType BWD_TYPE = CUFFT_Z2D;
static const char* NPY_DTYPE = "<c16";
#else
#error "fp64 throughout -- single precision was considered and declined (plan section 1)"
#endif

// ------------------------------------------------------------------ physical parameters (C4)
// Every one of these keeps the value forced-dns-sm-big.py's own parameter block
// carries, so a build with no -D flags behaves exactly as the hand-edited block
// did. run_gpu.py passes the whole set on the nvcc line, which is what lets that
// one Python file be the only place the parameters are set: a value changed
// there reaches the solver instead of silently disagreeing with it.
#ifndef P_LP
#define P_LP 1                      // hyperviscous power lp (forced-dns-sm-big.py:99)
#endif
#ifndef P_NU0
#define P_NU0 0.5                   // viscosity at N = 1 (:101)
#endif
#ifndef P_STS
#define P_STS 0.001                 // small-droplet Stokes number (:40)
#endif
#ifndef P_M0
#define P_M0 7.2e-4                 // small-droplet mass (:54)
#endif
#ifndef P_RHOP
#define P_RHOP 1000.0               // droplet density (:55)
#endif
#ifndef P_EPS
#define P_EPS 0.5                   // f0/(2 pi)^3 = 0.5/nshells (:131). Zero disables forcing
#endif
#ifndef P_NSHELLS
#define P_NSHELLS 1                 // number of forced shells (:129)
#endif
#ifndef P_ETA_DIM
#define P_ETA_DIM 0.6e-3            // Kolmogorov length in cloud, metres (:116)
#endif
#ifndef P_TAU_ETA_DIM
#define P_TAU_ETA_DIM 0.03          // Kolmogorov time in cloud, seconds (:117)
#endif
// These two expand where they are used, below, where N is already defined.
#ifndef P_DTMIN
#define P_DTMIN (0.0005 * 0.256 / N)      // first step of every run (:35)
#endif
#ifndef P_DTMAX
#define P_DTMAX (5.0 * 0.256 / N)         // step ceiling (:34)
#endif
#ifndef P_DT_SAVE_FIELDS
#define P_DT_SAVE_FIELDS 1.0        // (:37)
#endif
#ifndef P_DT_SAVE_PARTICLES
#define P_DT_SAVE_PARTICLES 0.1     // (:38)
#endif

// ------------------------------------------------------------------ grid
#ifndef NS_N
#define NS_N 256
#endif
static constexpr int    N   = NS_N;
static constexpr int    Nf  = N / 2 + 1;
static constexpr long   NS_ = (long)N * N * Nf;
static constexpr long   ZP  = 2L * Nf;
static constexpr long   NRP = (long)N * N * ZP;
static constexpr long   NLINE = (long)N * N;
static constexpr long   NR_ = (long)N * N * N;             // physical real pts (logical, unpadded)

static constexpr double PI_D = 3.14159265358979323846;
static constexpr double L_D  = 2.0 * PI_D;
static constexpr double DX   = L_D / N;
static constexpr int    LP   = P_LP;          // forced-dns-sm-big.py's hyperviscous power lp
// LP sets the viscosity scaling nu = nu0*eta^(2(lp - 1/3)), but k_velocity_rhs
// applies -nu*k^2*u, which IS lp = 1. Any other value would scale nu for a
// dissipation operator this file does not implement, so it is refused here
// rather than run as wrong physics.
static_assert(LP == 1, "only lp = 1 is implemented: k_velocity_rhs applies -nu*k^2*u");
// build_shell(1, ...) forces shell 1 and nothing else. nshells also divides
// P_EPS, so a larger value would weaken the forcing while still driving one
// shell -- wrong, and silently so. Refused here instead.
static_assert(P_NSHELLS == 1, "only nshells = 1 is implemented: build_shell forces shell 1 alone");

static constexpr int    KCEIL = (int)(1.4142135623730951 * N / 3.0) + 1;   // forced-dns-sm-big.py:159
static constexpr double KTH2  = ((double)KCEIL - 0.5) * ((double)KCEIL - 0.5);

static constexpr int  TPB  = 256;
static constexpr int  POFF = N;
static constexpr int  PLEN = 3 * N + 1;
static constexpr double N3d = (double)N * N * N;
static constexpr double N6d = N3d * N3d;

static inline int nblocks(long n) {
    long b = (n + TPB - 1) / TPB;
    return (int)std::min<long>(std::max<long>(b, 1), 16384);
}
static cudaStream_t g_stream;

// ------------------------------------------------------------------ device math (forced_ns_ck54_lowmem.cu's own)
__device__ __forceinline__ cplx_t C_(real_t re, real_t im) { cplx_t c; c.x = re; c.y = im; return c; }
__device__ __forceinline__ int    wnum(int i) { return (i <= N / 2) ? i : i - N; }

struct SpecPt { int kx, ky, kz; double k2; };
__device__ __forceinline__ SpecPt spec_point(long t)
{
    SpecPt s;
    int  iz = (int)(t % Nf);
    long r  = t / Nf;
    int  iy = (int)(r % N);
    int  ix = (int)(r / N);
    s.kx = wnum(ix); s.ky = wnum(iy); s.kz = iz;
    s.k2 = (double)s.kx * s.kx + (double)s.ky * s.ky + (double)s.kz * s.kz;
    return s;
}

struct CPt { int kx, ky, kz; double k2; long c; };
__device__ __forceinline__ bool compact_point(long t, const long* __restrict__ loff, CPt& s)
{
    int  iz   = (int)(t % Nf);
    long line = t / Nf;
    long o0   = loff[line];
    if (iz >= loff[line + 1] - o0) return false;
    int iy = (int)(line % N);
    int ix = (int)(line / N);
    s.kx = wnum(ix); s.ky = wnum(iy); s.kz = iz;
    s.k2 = (double)s.kx * s.kx + (double)s.ky * s.ky + (double)s.kz * s.kz;
    s.c  = o0 + iz;
    return true;
}

#define GRID_STRIDE(t, n) \
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < (n); t += (long)gridDim.x * blockDim.x)

__device__ __forceinline__ void block_add(double v, double* out)
{
#ifdef CPU_EMU
    *out += v;
#else
    __shared__ double sh[TPB];
    __syncthreads();
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
#endif
}

// block_add's shape for a minimum and for a maximum. CUDA has no atomicMin or
// atomicMax on a double, so the block result goes out through the unsigned
// long long form on the bit pattern: for POSITIVE IEEE-754 doubles the bit
// pattern read as an unsigned integer is monotone with the value, so this
// orders them correctly. Both callers only ever reduce positive quantities (a
// droplet mass, a squared modulus), which is the whole of why it is legal --
// it would be wrong for a negative or mixed-sign input. The leading
// __syncthreads(), like block_add's, makes these illegal to call from inside
// divergent control flow: the per-thread partial is built across the whole
// grid-stride loop first and these are then called ONCE, outside it.
__device__ __forceinline__ void block_min(double v, double* out)
{
#ifdef CPU_EMU
    if (v < *out) *out = v;
#else
    __shared__ double sh[TPB];
    __syncthreads();
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s && sh[threadIdx.x + s] < sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMin((unsigned long long*)out, (unsigned long long)__double_as_longlong(sh[0]));
#endif
}

__device__ __forceinline__ void block_max(double v, double* out)
{
#ifdef CPU_EMU
    if (v > *out) *out = v;
#else
    __shared__ double sh[TPB];
    __syncthreads();
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax((unsigned long long*)out, (unsigned long long)__double_as_longlong(sh[0]));
#endif
}

// ------------------------------------------------------------------ velocity-path kernels
// Duplicated verbatim from forced_ns_ck54_lowmem.cu:284-507 -- same technique,
// and (see below) the same physics as forced-dns-sm-big.py's full_RHS u-part.
//
// Python's nonlinear term is the rotational form omega x u (rhsu = omg[2]*u[1]
// - omg[1]*u[2], etc). k_product/k_accum_prod instead build the convective
// form d_j(u_i u_j) through the five Basdevant products. The two differ by the
// pure gradient grad(|u|^2/2) (u.grad(u) = grad(|u|^2/2) + omega x u).
//
// THAT EQUIVALENCE HOLDS ONLY FOR THE PROJECTED ku, NOT FOR vk.
// k_velocity_rhs projects the accumulator onto the solenoidal subspace, which
// removes the gradient, so ku is the same whichever form built it. But
// vk = uk - tps*(ku - rhsuk) subtracts the UNPROJECTED accumulator, where
// nothing removes it, so vk needs the true rotational rhsuk. That is what
// k_rot_corr_prod/k_accum_rot_corr (droplet_kernels.cuh) add back, once per
// phase-shift pass, before k_make_v runs. Do not drop them on the grounds that
// the two forms agree -- they agree after projection only.
//
// The forcing match (k_shell_energy + k_add_force against forcing()) is
// exact for nshells = 1, shell_no = {1}: ek_arr (forced-dns-sm-big.py:276-283)
// carries an extra 0.5*(2 pi)^3 that g_eps = 0.5 absorbs, i.e.
//     ek_arr = 0.5*(2 pi)^3 * (*Ek),   evaluated: 124.0251067 * (*Ek).
// Checked as a number, not only as algebra: the Python's per-mode weight at
// kz != 0 is 0.5*2*(2 pi)^3/N^6 and k_shell_energy's is 2/N^6, whose ratio is
// 0.5*(2 pi)^3. Anything comparing the two sides must DIVIDE by that to go
// from an ek_arr threshold to an *Ek one -- see k_add_force's ek_floor, which
// was written multiplying and was wrong by its square.
__global__ void k_scatter(const cplx_t* __restrict__ U, long M,
                          const long* __restrict__ loff,
                          cplx_t* __restrict__ dst, real_t scale, int shifted,
                          const cplx_t* __restrict__ phase)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) {
            dst[t] = C_(0, 0); dst[t + NS_] = C_(0, 0); dst[t + 2 * NS_] = C_(0, 0);
            continue;
        }
        double aRe = scale, aIm = 0.0;
        if (shifted) {
            cplx_t p = phase[s.kx + s.ky + s.kz + POFF];
            aRe = (double)p.x * scale;  aIm = (double)p.y * scale;
        }
        for (int c = 0; c < 3; ++c) {
            cplx_t v = U[s.c + c * M];
            dst[t + c * NS_] = C_((real_t)(aRe * v.x - aIm * v.y),
                                  (real_t)(aRe * v.y + aIm * v.x));
        }
    }
}

__global__ void k_gather(const cplx_t* __restrict__ src, cplx_t* __restrict__ U,
                         long M, const long* __restrict__ loff)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        U[s.c]         = src[t];
        U[s.c + M]     = src[t + NS_];
        U[s.c + 2 * M] = src[t + 2 * NS_];
    }
}

__device__ __forceinline__ void prod_ab(int w, int& a, int& b)
{
    a = (w == 1 || w == 4) ? 1 : 0;
    b = (w == 0) ? 0 : (w == 1 || w == 2) ? 1 : 2;
}

// The five Basdevant products, physical space: 0: ux ux - uz uz, 1: uy uy -
// uz uz, 2: ux uy, 3: ux uz, 4: uy uz. u holds the three padded real
// components at stride NRP.
__global__ void k_product(const real_t* __restrict__ u, real_t* __restrict__ P, int w)
{
    int a, b;
    prod_ab(w, a, b);
    GRID_STRIDE(t, NR_) {
        int  z = (int)(t % N);
        long r = t / N;
        int  y = (int)(r % N);
        int  x = (int)(r / N);
        long i = ((long)x * N + y) * ZP + z;
        real_t va = u[i + a * NRP], vb = u[i + b * NRP];
        real_t v  = va * vb;
        if (w < 2) { real_t vz = u[i + 2 * NRP]; v -= vz * vz; }
        P[i] = v;
    }
}

__global__ void k_accum_prod(cplx_t* __restrict__ q, long M,
                             const cplx_t* __restrict__ P,
                             const long* __restrict__ loff, int w, int shifted,
                             const cplx_t* __restrict__ phase)
{
    int c1, c2;
    prod_ab(w, c1, c2);
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        double fRe = 0.5, fIm = 0.0;
        if (shifted) {
            cplx_t p = phase[s.kx + s.ky + s.kz + POFF];
            fRe = 0.5 * (double)p.x;  fIm = -0.5 * (double)p.y;
        }
        cplx_t a = P[t];
        double vr = fRe * a.x - fIm * a.y;
        double vi = fRe * a.y + fIm * a.x;
        double k1 = (c2 == 0) ? s.kx : (c2 == 1) ? s.ky : s.kz;
        cplx_t o = q[s.c + c1 * M];
        q[s.c + c1 * M] = C_((real_t)(o.x + k1 * vi), (real_t)(o.y - k1 * vr));
        if (c1 != c2) {
            double k2d = (c1 == 0) ? s.kx : (c1 == 1) ? s.ky : s.kz;
            o = q[s.c + c2 * M];
            q[s.c + c2 * M] = C_((real_t)(o.x + k2d * vi), (real_t)(o.y - k2d * vr));
        }
    }
}

__device__ __forceinline__ long shell_c(long t, const long* __restrict__ loff)
{
    return loff[t / Nf] + (t % Nf);
}

__global__ void k_shell_energy(const cplx_t* __restrict__ V, long M,
                               const long* __restrict__ idx, int ns,
                               const long* __restrict__ loff,
                               double* __restrict__ out)
{
    double acc = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ns; j += gridDim.x * blockDim.x) {
        long t = idx[j];
        long c = shell_c(t, loff);
        SpecPt s = spec_point(t);
        double wz = (s.kz == 0 || s.kz == N / 2) ? 1.0 : 2.0;
        cplx_t u0 = V[c], u1 = V[c + M], u2 = V[c + 2 * M];
        double m2 = (double)u0.x * u0.x + (double)u0.y * u0.y
                  + (double)u1.x * u1.x + (double)u1.y * u1.y
                  + (double)u2.x * u2.x + (double)u2.y * u2.y;
        acc += (wz / N6d) * m2;
    }
    block_add(acc, out);
}

__global__ void k_add_force(cplx_t* __restrict__ q, long M,
                            const cplx_t* __restrict__ V,
                            const long* __restrict__ idx, int ns,
                            const long* __restrict__ loff,
                            const double* __restrict__ Ek, double eps)
{
    // forced-dns-sm-big.py:283 sends a shell whose energy is below 1e-10 to inf
    // before dividing, so factor comes out 0 and that shell is not
    // forced. Without the guard this divides by zero and the whole field is
    // NaN from the first stage. It was left out while the -DUSE_CK54 build had
    // to match forced_ns_ck54_lowmem.cu bit for bit; that build is gone, so the
    // Python's behaviour is what this should follow.
    //
    // ek_arr is the LARGER of the two: the Python's normalize carries
    // (2 pi)^3/N^6 against k_shell_energy's 1/N^6, and the Python's leading 0.5
    // against this kernel's none, so ek_arr = 0.5*(2 pi)^3 * (*Ek) -- the same
    // factor g_eps = 0.5 absorbs. The Python's 1e-10 on ek_arr is therefore
    // 1e-10/(0.5*(2 pi)^3) on *Ek, so the threshold DIVIDES. Multiplying puts
    // it (0.5*(2 pi)^3)^2 = 15382 times too high, which silently stops forcing
    // a shell the Python would still drive.
    const double ek_floor = 1e-10 / (0.5 * L_D * L_D * L_D);
    double g = (*Ek > ek_floor) ? eps / (*Ek) : 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ns;
         j += gridDim.x * blockDim.x) {
        long t = idx[j];
        long c = shell_c(t, loff);
        SpecPt s = spec_point(t);
        double ik = (s.k2 > 0.0) ? rsqrt(s.k2) : 0.0;
        double h0 = s.kx * ik, h1 = s.ky * ik, h2 = s.kz * ik;

        cplx_t u0 = V[c], u1 = V[c + M], u2 = V[c + 2 * M];
        double f0r = g * u0.x, f0i = g * u0.y;
        double f1r = g * u1.x, f1i = g * u1.y;
        double f2r = g * u2.x, f2i = g * u2.y;
        double dre = h0 * f0r + h1 * f1r + h2 * f2r;
        double dim = h0 * f0i + h1 * f1i + h2 * f2i;

        q[c].x         += (real_t)(f0r - dre * h0);  q[c].y         += (real_t)(f0i - dim * h0);
        q[c + M].x     += (real_t)(f1r - dre * h1);  q[c + M].y     += (real_t)(f1i - dim * h1);
        q[c + 2 * M].x += (real_t)(f2r - dre * h2);  q[c + 2 * M].y += (real_t)(f2i - dim * h2);
    }
}

// End-of-step divergence-free projection, forced-dns-sm-big.py:667-670:
//   pk     = invlap*(kx*u0 + ky*u1 + kz*u2)
//   u_j   += k_j*pk
// elementwise over the COMPACT spectrum, in place. invlap is
// dealias/where(lap == 0, inf, lap) (line 162) with lap = -(kx^2+ky^2+kz^2)
// (line 153), so it is -1/k^2 at every retained mode and EXACTLY 0 at k = 0,
// which leaves the mean mode alone. Every mode this kernel reaches is inside
// the dealias band by construction, so the dealias factor is 1 and does not
// appear. There is NO imaginary unit anywhere here: unlike the nonlinear
// term's i*k gradient, this is a real multiplicative correction, applied to
// the real and imaginary parts with the same real coefficient.
__global__ void k_project(cplx_t* __restrict__ U, long M, const long* __restrict__ loff)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        const long c = s.c;
        cplx_t u0 = U[c], u1 = U[c + M], u2 = U[c + 2 * M];
        double il = (s.k2 > 0.0) ? (-1.0 / s.k2) : 0.0;
        double pr = il * (s.kx * (double)u0.x + s.ky * (double)u1.x + s.kz * (double)u2.x);
        double pi = il * (s.kx * (double)u0.y + s.ky * (double)u1.y + s.kz * (double)u2.y);
        U[c]         = C_((real_t)((double)u0.x + s.kx * pr), (real_t)((double)u0.y + s.kx * pi));
        U[c + M]     = C_((real_t)((double)u1.x + s.ky * pr), (real_t)((double)u1.y + s.ky * pi));
        U[c + 2 * M] = C_((real_t)((double)u2.x + s.kz * pr), (real_t)((double)u2.y + s.kz * pi));
    }
}

// ------------------------------------------------------------------ adaptive step and blow-up guard
// Smallest droplet mass in one field. forced-dns-sm-big.py:633-638 reduces
// min(st) with st = (mass/factor)^(2/3) (particles.py:216) and combines
// 0.5*st*tf into hnew; x^(2/3) is monotone increasing on positive x, so the
// minimum may be taken over the MASS here and the power applied once on the
// host per field instead of once per droplet. The MPI allreduce is dropped:
// one GPU is one domain. coord is Nprtcl*7 with the mass at offset 6 of each
// row, so this walks every 7th double starting at index 6. out must be SEEDED
// WITH DBL_MAX, never memset to zero -- zero would win every minimum. A field
// with Nprtcl = 0 runs the loop zero times and leaves that seed in place,
// which is the "contributes nothing" sentinel the host then takes the minimum with.
__global__ void k_mass_min(const double* __restrict__ coord, long Nprtcl, double* __restrict__ out)
{
    double v = DBL_MAX;
    GRID_STRIDE(t, Nprtcl) {
        double m = coord[t * 7 + 6];
        if (m < v) v = m;
    }
    block_min(v, out);
}

// forced-dns-sm-big.py:678's blow-up guard, corrected. The Python tests
// uk.max(), which on a complex array orders by real part first: it misses a
// blow-up confined to the imaginary part, misses a divergence to large
// NEGATIVE real parts, and -- since `nan > x` is False -- disarms itself
// entirely once the field goes non-finite. This reduces the squared MODULUS
// instead (compared against (100*N^3)^2 on the host, so no sqrt per mode) and
// carries the non-finite count as a SEPARATE signal, because `>` and fmax skip
// a NaN exactly the way the Python does. `!(q <= DBL_MAX)` is true for both
// NaN and +inf and false for every finite q, and needs no isfinite(), which
// <cmath> is entitled to hide behind std::. A finite pair whose squared
// modulus overflows is counted non-finite too; its modulus is then ~1e154 and
// past the threshold anyway.
__global__ void k_uk_guard(const cplx_t* __restrict__ U, long M, const long* __restrict__ loff,
                           double* __restrict__ maxmod2, double* __restrict__ nonfinite)
{
    double m2 = 0.0, bad = 0.0;
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        for (int c = 0; c < 3; ++c) {
            cplx_t v = U[s.c + c * M];
            double q = (double)v.x * v.x + (double)v.y * v.y;
            if (q > m2) m2 = q;
            if (!(q <= DBL_MAX)) bad += 1.0;
        }
    }
    block_max(m2, maxmod2);
    block_add(bad, nonfinite);
}

// ------------------------------------------------------------------ FFT work area
struct FFT1 {
    cufftHandle fwd = 0, bwd = 0;
    void*  work = nullptr;
    size_t ws   = 0;
    void plan() {
        int n[3]      = { N, N, N };
        int emb_r[3]  = { N, N, (int)ZP };
        int emb_c[3]  = { N, N, Nf };
        static_assert(NRP < 2147483647L, "N too large for the 32-bit cuFFT plan API");
        CUFFT_CHECK(cufftCreate(&fwd));
        CUFFT_CHECK(cufftCreate(&bwd));
        CUFFT_CHECK(cufftSetAutoAllocation(fwd, 0));
        CUFFT_CHECK(cufftSetAutoAllocation(bwd, 0));
        size_t w1 = 0, w2 = 0;
        CUFFT_CHECK(cufftMakePlanMany(fwd, 3, n, emb_r, 1, (int)NRP,
                                      emb_c, 1, (int)NS_, FWD_TYPE, 1, &w1));
        CUFFT_CHECK(cufftMakePlanMany(bwd, 3, n, emb_c, 1, (int)NS_,
                                      emb_r, 1, (int)NRP, BWD_TYPE, 1, &w2));
        ws = std::max(w1, w2);
    }
    void attach() {
        if (ws) CUDA_CHECK(cudaMalloc(&work, ws));
        CUFFT_CHECK(cufftSetWorkArea(fwd, work));
        CUFFT_CHECK(cufftSetWorkArea(bwd, work));
        CUFFT_CHECK(cufftSetStream(fwd, g_stream));
        CUFFT_CHECK(cufftSetStream(bwd, g_stream));
    }
    void forward(cplx_t* b) { CUFFT_CHECK(cufftExecD2Z(fwd, (cufftDoubleReal*)b, (cufftDoubleComplex*)b)); }
    void inverse(cplx_t* b) { CUFFT_CHECK(cufftExecZ2D(bwd, (cufftDoubleComplex*)b, (cufftDoubleReal*)b)); }
    void destroy() {
        if (fwd) cufftDestroy(fwd);
        if (bwd) cufftDestroy(bwd);
        if (work) cudaFree(work);
    }
};

static std::vector<long> build_line_offsets()
{
    std::vector<long> off(NLINE + 1, 0);
    long acc = 0;
    for (long line = 0; line < NLINE; ++line) {
        int ix = (int)(line / N), iy = (int)(line % N);
        int kx = (ix <= N / 2) ? ix : ix - N;
        int ky = (iy <= N / 2) ? iy : iy - N;
        off[line] = acc;
        int L = 0;
        while (L < Nf) {
            double k2 = (double)kx * kx + (double)ky * ky + (double)L * L;
            if (!(k2 < KTH2)) break;
            ++L;
        }
        acc += L;
    }
    off[NLINE] = acc;
    return off;
}

// forced-dns-sm-big.py: nshells = 1, shell_no = np.arange(1, 2) = {1}.
static void build_shell(int shell, long** dev, int* count, const std::vector<long>& off)
{
    std::vector<long> idx;
    int R = shell + 2;
    for (int kx = -R; kx <= R; ++kx)
        for (int ky = -R; ky <= R; ++ky)
            for (int kz = 0; kz <= R; ++kz) {
                double k2 = (double)kx * kx + (double)ky * ky + (double)kz * kz;
                int kint = (int)llround(sqrt(k2));
                if (kint > N / 2) kint = N / 2;
                if (kint != shell) continue;
                int ix = (kx < 0) ? kx + N : kx;
                int iy = (ky < 0) ? ky + N : ky;
                long line = (long)ix * N + iy;
                if (kz >= off[line + 1] - off[line]) {
                    fprintf(stderr, "forced shell %d reaches outside the dealias band\n", shell);
                    exit(1);
                }
                idx.push_back(line * Nf + kz);
            }
    *count = (int)idx.size();
    CUDA_CHECK(cudaMalloc(dev, std::max<size_t>(1, idx.size()) * sizeof(long)));
    CUDA_CHECK(cudaMemcpy(*dev, idx.data(), idx.size() * sizeof(long),
                          cudaMemcpyHostToDevice));
}

static void build_phase_table(cplx_t** d_phase)
{
    std::vector<cplx_t> h(PLEN);
    for (int m = 0; m < PLEN; ++m) {
        double th = (double)(m - POFF) * (0.5 * DX);
        h[m].x = (real_t)cos(th);
        h[m].y = (real_t)sin(th);
    }
    CUDA_CHECK(cudaMalloc(d_phase, PLEN * sizeof(cplx_t)));
    CUDA_CHECK(cudaMemcpy(*d_phase, h.data(), PLEN * sizeof(cplx_t),
                          cudaMemcpyHostToDevice));
}

#include "droplet_kernels.cuh"

// ------------------------------------------------------------------ globals
static long    g_M = 0;
static cplx_t *d_U = nullptr, *d_q = nullptr, *d_P = nullptr;   // velocity state(3*g_M), nonlinear acc(3*g_M), scratch(NS_)
static cplx_t *d_u = nullptr;                                     // physical flow velocity, 3 slots (aliases cplx_t NS_ each)
static real_t *d_v = nullptr, *d_vtemp = nullptr;                 // physical drift velocity, unshifted/shifted, 3*NRP each
                                                                  // (also the scatter target for vk: 3*NRP reals = 3*NS_ complex)
static real_t *d_vn = nullptr;                                    // physical vn product scratch, 3*NRP (reused per field)
static real_t *d_fc = nullptr;                                    // physical deposition field, 1*NRP (reused per field)
static long   *d_loff = nullptr;
static cplx_t *d_phase = nullptr;
static long   *d_shell = nullptr;
static int     g_nshell = 0;
static double *d_Ek = nullptr;
static double *d_sum = nullptr;    // clip_zero's two global sums (old mean, new mean)
// Step-control scalars: [0] the per-field minimum droplet mass, [1] the
// maximum squared modulus of uk, [2] the non-finite mode count. A buffer of
// its own rather than two more slots on d_sum, whose two are both busy for the
// whole of clip_zero_phys.
static double *d_red = nullptr;
static FFT1    g_fft;

// classical-RK4 velocity registers (plan's "four compact registers"). There is
// no separate vk register: k_make_v writes the drift velocity over d_q, which
// is dead from that point until the next stage zeroes it.
static cplx_t *d_Uy = nullptr, *d_Uacc = nullptr, *d_ku = nullptr;
static cplx_t *d_divvnk = nullptr, *d_fck = nullptr;               // shared, per-field-reused compact scratch, g_M each

// ------------------------------------------------------------------ host-side physical scalars (plan section 4, forced-dns-sm-big.py's parameter block)
static double g_nu = 0.0;          // nu
static double g_tf = 0.0;          // tf, the Kolmogorov time (= tau_eta)
static double g_tps = 0.0;         // tps = sts * tf, the SMALL-droplet particle timescale
static double g_sts = P_STS;       // sts, small-Stokes parameter
static double g_rhop = P_RHOP;
static double g_M0 = P_M0;
static double g_gravity = 0.0;     // g, in simulation units (0 unless gravity flag set)
static double g_eps = P_EPS;       // f0/(2 pi)^3, nshells = 1
static double g_factor = 0.0;      // particles.py's self.factor (mass<->Stokes)
static double g_growthfactor = 0.0, g_decelerationfactor = 0.0, g_n_cs_factor = 0.0;
// Step-size bounds and save intervals, hand-edited here exactly as in
// forced-dns-sm-big.py:34-39 (C4). The step size is adaptive and is never a
// command-line argument; only the end time is. DTMAX must stay BELOW
// 2*DT_SAVE_PARTICLES -- see save_due for why.
static constexpr double DTMAX             = P_DTMAX;
static constexpr double DTMIN             = P_DTMIN;
static constexpr double DT_SAVE_FIELDS    = P_DT_SAVE_FIELDS;
static constexpr double DT_SAVE_PARTICLES = P_DT_SAVE_PARTICLES;
// save_due's window half-width is sin(pi*h/(2*Ts)), which peaks at h = Ts and
// turns NEGATIVE for h > 2*Ts -- at which point the test can never fire and the
// run saves nothing at all, silently. Only DTMAX bounds h, so the relation is
// checked here rather than left to a reader of the comment.
static_assert(DTMAX < 2.0 * DT_SAVE_PARTICLES,
              "DTMAX must stay below 2*DT_SAVE_PARTICLES or the save trigger never fires");
static_assert(DTMIN > 0.0 && DTMIN <= DTMAX, "need 0 < DTMIN <= DTMAX");

// prtclid is carried on the host only: one GPU is one domain, so nothing ever
// reorders a field's droplets and array position is preserved through the
// step. It exists so the checkpoint comparison can match droplet to droplet
// against the MPI run, which DOES reorder (particles.py's send).
struct FieldSpec { double stb; std::string init_name; long Nprtcl; std::vector<double> prtclid; };
static std::vector<FieldSpec> g_fields;
static std::vector<DropletField> g_df;     // host-side field table; every kernel takes raw device pointers out of it

// ------------------------------------------------------------------ G2: startup refusal
// bytes = fixed + sum over fields of per_field(f), checked against cudaMemGetInfo
// BEFORE anything on either list has been cudaMalloc'd (PLAN-GPU-PORT.md
// section 3 / droplet_kernels.cuh's G2 comment) -- its caller in main() sits
// between g_fft.plan() and g_fft.attach(), so cudaMemGetInfo's free total has
// not already been reduced by the buffers this counts. per_field counts the
// PHYSICAL carried density n_y (NRP reals), the two compact increment registers
// nk_acc/nk_k and three carried coord registers the
// .cuh contract specifies; the trial registers are shared scratch and are
// counted once, in fixed, at the size of the largest field. Nprtcl varies by
// field (it is the row count of that field's own pos.npy, see particle_rows),
// so both the total and the "largest count that would fit" are summed exactly,
// never estimated from an average.
//
// Already allocated when this runs, and so deliberately NOT counted (they are
// already out of free_b, and together are a few MB at most): d_loff, d_phase,
// d_shell. Counted but not yet allocated: the cuFFT work area, which plan()
// has sized into g_fft.ws but attach() has not yet taken.
static void startup_refusal(const std::vector<FieldSpec>& fields, size_t fft_ws)
{
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    // The trial registers are one buffer each, reused by every field in the
    // serial per-field loop, so they are sized for the LARGEST field and enter
    // the fixed term once rather than being multiplied by the field count.
    long coord_max = 0;
    for (const auto& f : fields) coord_max = std::max(coord_max, f.Nprtcl);
    size_t trial = fields.empty() ? 0
                 : (size_t)g_M * sizeof(cplx_t)                        // d_nk_trial
                 + (size_t)coord_max * 7 * sizeof(double)              // d_coord_trial
                 + (size_t)coord_max * sizeof(double);                 // d_exterp0

    size_t fixed = fft_ws                                             // cuFFT work area (shared, FFT1::attach)
                 + (size_t)NS_ * sizeof(cplx_t)                        // d_P
                 + 3 * (size_t)NS_ * sizeof(cplx_t)                    // d_u
                 + 2 * 3 * (size_t)NRP * sizeof(real_t)                // d_v, d_vtemp
                 + 3 * (size_t)NRP * sizeof(real_t)                    // d_vn
                 + (size_t)NRP * sizeof(real_t)                        // d_fc
                 + 5 * 3 * (size_t)g_M * sizeof(cplx_t)                // d_q, d_U, d_Uy, d_Uacc, d_ku
                 + 2 * (size_t)g_M * sizeof(cplx_t)                    // d_divvnk, d_fck
                 + 6 * sizeof(double)                                  // d_Ek, d_sum, d_red
                 + trial;

    auto field_bytes = [](const FieldSpec& f) {
        return (size_t)NRP * sizeof(real_t)                            // n_y, the PHYSICAL carried density
             + 2 * (size_t)g_M * sizeof(cplx_t)                        // nk_acc/nk_k, the compact increments
             + 3 * (size_t)f.Nprtcl * 7 * sizeof(double);              // coord_y/acc/k
    };
    size_t perfield_bytes = 0;
    for (const auto& f : fields) perfield_bytes += field_bytes(f);
    size_t need = fixed + perfield_bytes;

    if (need > free_b) {
        // The count that would fit is an EXACT prefix sum over the field list in
        // ensemble order, not (free - fixed)/average: Nprtcl varies by a factor
        // of 128 across the documented stb values and stb cycles fastest, so a
        // short prefix can be far more expensive per field than the average and
        // an averaged estimate under-refuses. The shared trial term stays at the
        // whole list's largest field, so maxfit errs on the low side if the
        // prefix that fits excludes that field -- the safe direction.
        size_t avail = (free_b > fixed) ? free_b - fixed : 0;
        size_t run = 0;
        long   maxfit = 0;
        for (const auto& f : fields) {
            size_t c = field_bytes(f);
            if (run + c > avail) break;
            run += c;
            ++maxfit;
        }
        fprintf(stderr,
            "refusing to start: %zu fields need %.3f GiB (%.3f fixed + %.3f per-field), "
            "only %.3f GiB free. The largest field count that would fit here is %ld.\n",
            fields.size(), need / 1073741824.0, fixed / 1073741824.0,
            perfield_bytes / 1073741824.0, free_b / 1073741824.0, maxfit);
        exit(1);
    }
}

// ------------------------------------------------------------------ .npy I/O
// Simplified against forced_ns_ck54_lowmem.cu's streamed version (984-1260):
// whole-array host buffers, no chunking. Correct for the sizes this file
// reads/writes (one checkpoint's uk/n/particle arrays, not a tracer-particle
// dump), traded for code size under this session's time budget -- flagged in
// HANDOVER-GPU-PORT.md for revisiting if N grows enough to make the host
// buffer itself a problem.
static void write_npy(const std::string& path, const char* dtype,
                      const std::vector<long>& shape, const void* data, size_t nbytes)
{
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shp += ", ";
    }
    shp += ")";
    std::string hdr = "{'descr': '" + std::string(dtype) +
                      "', 'fortran_order': False, 'shape': " + shp + ", }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad   = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += "\n";
    unsigned short hlen = (unsigned short)hdr.size();
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror(path.c_str()); return; }
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    if (nbytes) fwrite(data, 1, nbytes, f);
    fclose(f);
}

struct NpyInfo { FILE* f = nullptr; std::string descr; std::vector<long> shape; bool fortran = false; };

static bool npy_open(const std::string& path, NpyInfo& o)
{
    o.f = fopen(path.c_str(), "rb");
    if (!o.f) return false;
    unsigned char m[8];
    if (fread(m, 1, 8, o.f) != 8 || memcmp(m, "\x93NUMPY", 6) != 0) return false;
    unsigned hlen = 0;
    if (m[6] == 1) { unsigned short h = 0; if (fread(&h, 2, 1, o.f) != 1) return false; hlen = h; }
    else           { unsigned h = 0;       if (fread(&h, 4, 1, o.f) != 1) return false; hlen = h; }
    std::string hdr(hlen, ' ');
    if (fread(&hdr[0], 1, hlen, o.f) != hlen) return false;
    size_t p = hdr.find("'descr'");
    if (p == std::string::npos) return false;
    size_t q1 = hdr.find('\'', hdr.find(':', p)), q2 = hdr.find('\'', q1 + 1);
    o.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
    p = hdr.find("'fortran_order'");
    o.fortran = (p != std::string::npos && hdr.find("True", p) < hdr.find(',', p));
    p = hdr.find("'shape'");
    if (p == std::string::npos) return false;
    size_t b1 = hdr.find('(', p), b2 = hdr.find(')', b1);
    const char* c = hdr.c_str() + b1 + 1; const char* e = hdr.c_str() + b2;
    while (c < e) { char* end = nullptr; long v = strtol(c, &end, 10);
        if (end == c) { ++c; continue; } o.shape.push_back(v); c = end; }
    return true;
}

// Loads a whole-domain complex128 .npy into a device buffer of ncplx points.
static bool load_complex_npy(const std::string& path, cplx_t* dst, long ncplx)
{
    NpyInfo ni;
    if (!npy_open(path, ni)) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    long n = 1; for (long d : ni.shape) n *= d;
    if (ni.descr != NPY_DTYPE || ni.fortran || n != ncplx) {
        fprintf(stderr, "%s: dtype '%s' n=%ld, expected <c16 n=%ld\n", path.c_str(), ni.descr.c_str(), n, ncplx);
        fclose(ni.f); return false;
    }
    std::vector<double> host((size_t)n * 2);
    bool ok = fread(host.data(), sizeof(double), (size_t)n * 2, ni.f) == (size_t)n * 2;
    fclose(ni.f);
    if (!ok) { fprintf(stderr, "%s: truncated\n", path.c_str()); return false; }
    std::vector<cplx_t> hc((size_t)n);
    for (long i = 0; i < n; ++i) { hc[i].x = host[2*i]; hc[i].y = host[2*i+1]; }
    CUDA_CHECK(cudaMemcpy(dst, hc.data(), (size_t)n * sizeof(cplx_t), cudaMemcpyHostToDevice));
    return true;
}

// Loads a whole-domain float64 .npy into a device buffer of nval doubles.
static void dump_complex_npy(const std::string& path, const cplx_t* src, long ncplx)
{
    std::vector<cplx_t> hc((size_t)ncplx);
    CUDA_CHECK(cudaMemcpy(hc.data(), src, (size_t)ncplx * sizeof(cplx_t), cudaMemcpyDeviceToHost));
    std::vector<double> host((size_t)ncplx * 2);
    for (long i = 0; i < ncplx; ++i) { host[2*i] = hc[i].x; host[2*i+1] = hc[i].y; }
    write_npy(path, NPY_DTYPE, { N, N, Nf }, host.data(), host.size() * sizeof(double));
}

// ------------------------------------------------------------------ directory naming (C6)
// Two DIFFERENT subdirectory spellings coexist under one time_<t>/ folder --
// not interchangeable (CLAUDE.md, "Two subdirectory orderings coexist"):
// density is sts_..._stb_... (forced-dns-sm-big.py:546), particles are
// stb_..._sts_... (particles.py's particle_dir). unpack_checkpoint.py writes
// its .npy files into whichever of these already exists, so both must be
// searched for here.
static std::string fmt3(double v) { char b[32]; snprintf(b, sizeof b, "%.3f", v); return b; }

static std::string density_dir(const std::string& time_dir, const char* wg, double sts,
                                double stb, const std::string& init)
{
    return time_dir + "/" + wg + "_sts_" + fmt3(sts) + "_stb_" + fmt3(stb) + "_init_" + init;
}
static std::string particle_dir(const std::string& time_dir, const char* wg, double stb,
                                 double sts, const std::string& init)
{
    return time_dir + "/" + wg + "_stb_" + fmt3(stb) + "_sts_" + fmt3(sts) + "_init_" + init;
}

// ------------------------------------------------------------------ G2 allocation rule
// One cudaMalloc per register kind, sized for every field up front, offsets
// into it -- NOT one cudaMalloc per field (droplet_kernels.cuh's G2 comment,
// memory-reviewer finding transcribed there).
// Only the CARRIED registers are per-field: n_y/nk_acc/nk_k and
// coord_y/coord_acc/coord_k hold state from one stage (or step) to the next,
// with every other field's stage body running in between, so they cannot be
// shared -- coord_k in particular is written by k_droplet_rhs at one stage and
// read by k_rk4_trial_r at the next. nk_trial,
// coord_trial and exterp0 are not state at all -- each is written at the top of
// a field's stage body, read through that body, and recomputed from the carried
// registers next stage -- and fields run strictly serially inside a stage, so
// one buffer each serves the whole ensemble, sized for the largest field. That
// is the same reuse d_P, d_vn and d_fc already get, and at twenty fields it is
// the largest single saving in this file.
static cplx_t  *d_nk_base    = nullptr;   // NFIELDS * 2 * g_M
static real_t  *d_n_base     = nullptr;   // NFIELDS * NRP, the physical carried densities
static double  *d_coord_base = nullptr;   // (sum Nprtcl) * 3 * 7
static cplx_t  *d_nk_trial   = nullptr;   // g_M, shared by every field
static double  *d_coord_trial = nullptr;  // (max Nprtcl) * 7, shared
static double  *d_exterp0    = nullptr;   // max Nprtcl, shared

static void allocate_fields(const std::vector<FieldSpec>& fields)
{
    long NFIELDS = (long)fields.size();
    g_df.resize(NFIELDS);

    // NFIELDS = 0 is a supported run (velocity only, no density or droplets),
    // and a zero-size cudaMalloc is an error on some drivers: skip it and
    // leave the base pointer null. Nothing dereferences it, since every use of
    // these buffers is inside a loop over g_df, which is then empty.
    long coord_total = 0, coord_max = 0;
    for (auto& f : fields) { coord_total += f.Nprtcl; coord_max = std::max(coord_max, f.Nprtcl); }
    if (NFIELDS) {
        CUDA_CHECK(cudaMalloc(&d_nk_base,  (size_t)NFIELDS * 2 * g_M * sizeof(cplx_t)));
        CUDA_CHECK(cudaMalloc(&d_n_base,   (size_t)NFIELDS * NRP * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_nk_trial, (size_t)g_M * sizeof(cplx_t)));
    }
    if (coord_total) {
        CUDA_CHECK(cudaMalloc(&d_coord_base,  (size_t)coord_total * 3 * 7 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_coord_trial, (size_t)coord_max * 7 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_exterp0,     (size_t)coord_max * sizeof(double)));
    }

    long coord_off = 0;
    for (long f = 0; f < NFIELDS; ++f) {
        DropletField df{};
        df.stb      = fields[f].stb;
        df.init_id  = (int)f;
        df.Nprtcl   = fields[f].Nprtcl;
        cplx_t* nkb = d_nk_base + f * 2 * g_M;
        df.nk_acc = nkb; df.nk_k = nkb + g_M;
        df.n_y = d_n_base + (size_t)f * NRP;
        double* cb = d_coord_base + coord_off * 3 * 7;
        long np7 = df.Nprtcl * 7;
        df.coord_y = cb; df.coord_acc = cb + np7; df.coord_k = cb + 2*np7;
        // every field's trial pointers are the SAME buffer, by the argument above
        df.nk_trial = d_nk_trial; df.coord_trial = d_coord_trial; df.exterp0 = d_exterp0;
        g_df[f] = df;
        coord_off += df.Nprtcl;
    }
}

// defined with the RK4 stage code below; load_n_into_ny needs it here
static void clip_zero_phys(real_t* n);

// Converts a packed (N,N,N) float64 host array (unpack_checkpoint.py's n.npy
// layout) into the field's own padded (N,N,ZP) buffer n_y and clips it there.
// The carried state IS this physical field, so there is no transform here: the
// upload goes straight into n_y, with the two pad reals per line left at the
// zero the host buffer was built with. One-time cost at startup, done on the
// host -- not a kernel, since it runs once per field, not per step.
static void load_n_into_ny(const std::string& npy_path, real_t* n_y)
{
    NpyInfo ni;
    if (!npy_open(npy_path, ni)) { fprintf(stderr, "cannot open %s\n", npy_path.c_str()); exit(1); }
    long n = 1; for (long d : ni.shape) n *= d;
    // ni.fortran is checked here for the same reason the other three readers
    // check it: the repack below walks C order, so a Fortran-ordered file would
    // be read transposed and nothing downstream would notice.
    if (ni.descr != "<f8" || ni.fortran || n != NR_) {
        fprintf(stderr, "%s: bad n.npy (dtype %s, fortran %d, n=%ld, want C-order <f8 n=%ld)\n",
        npy_path.c_str(), ni.descr.c_str(), (int)ni.fortran, n, NR_); exit(1); }
    std::vector<double> packed((size_t)NR_);
    if (fread(packed.data(), sizeof(double), (size_t)NR_, ni.f) != (size_t)NR_) {
        fprintf(stderr, "%s: truncated\n", npy_path.c_str()); exit(1);
    }
    fclose(ni.f);
    std::vector<double> padded((size_t)NRP, 0.0);
    for (long x = 0; x < N; ++x)
        for (long y = 0; y < N; ++y)
            memcpy(&padded[(x*N+y)*ZP], &packed[(x*N+y)*N], N * sizeof(double));
    CUDA_CHECK(cudaMemcpy(n_y, padded.data(), (size_t)NRP * sizeof(real_t), cudaMemcpyHostToDevice));
    // initial_conditions.py:169's apply_clip, which runs clip_zero on each Stokes
    // row of the loaded n before the first step. Without this the GPU would start
    // a restart from the raw field on disk and the MPI code from the clipped one.
    clip_zero_phys(n_y);
}

// Nprtcl comes from the checkpoint, NOT from forced-dns-sm-big.py:48's
// formula. What that run actually saved is num_process*(Nprtcl//num_process)
// droplets (particles.py:77), which equals the formula's value only when the
// rank count divides it; otherwise the file holds fewer rows than the formula
// predicts and a sizing taken from the formula either fails the shape check in
// load_particles_into_coordy or reads past the end of the file. So the row
// count is read out of pos.npy's header here -- before allocate_fields, while
// the field list is being built, since every device array is sized from it.
static long particle_rows(const std::string& pdir)
{
    NpyInfo ni;
    std::string p = pdir + "/pos.npy";
    if (!npy_open(p, ni)) {
        fprintf(stderr, "cannot open %s -- run unpack_checkpoint.py on the restart folder first\n", p.c_str());
        exit(1);
    }
    if (ni.descr != "<f8" || ni.fortran || ni.shape.size() != 2 || ni.shape[1] != 3) {
        fprintf(stderr, "%s: expected a C-order (Nprtcl, 3) <f8 array\n", p.c_str());
        exit(1);
    }
    fclose(ni.f);
    return ni.shape[0];
}

// Interleaves pos.npy (Nprtcl,3) / vel.npy (Nprtcl,3) / mass.npy (Nprtcl,)
// into the Nprtcl*7 coord layout droplet_kernels.cuh documents (note 4 there),
// and reads prtclid.npy into the host-side tag array. prtclid is float64 on
// disk, not int64: particles.py's send() rebuilds it by concatenating with an
// MPI.DOUBLE receive buffer, so what np.savez_compressed writes is <f8.
static void load_particles_into_coordy(const std::string& pdir, long Nprtcl, double* coord_y,
                                       std::vector<double>& prtclid)
{
    std::vector<double> pos((size_t)Nprtcl * 3), vel((size_t)Nprtcl * 3), mass((size_t)Nprtcl);
    auto load_one = [&](const char* name, double* dst, long nval) {
        NpyInfo ni;
        std::string p = pdir + "/" + name;
        if (!npy_open(p, ni)) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
        long n = 1; for (long d : ni.shape) n *= d;
        if (ni.descr != "<f8" || n != nval) { fprintf(stderr, "%s: bad shape/dtype\n", p.c_str()); exit(1); }
        if (fread(dst, sizeof(double), (size_t)nval, ni.f) != (size_t)nval) {
            fprintf(stderr, "%s: truncated\n", p.c_str()); exit(1);
        }
        fclose(ni.f);
    };
    prtclid.resize((size_t)Nprtcl);
    load_one("pos.npy", pos.data(), Nprtcl * 3);
    load_one("vel.npy", vel.data(), Nprtcl * 3);
    load_one("mass.npy", mass.data(), Nprtcl);
    load_one("prtclid.npy", prtclid.data(), Nprtcl);
    std::vector<double> coord((size_t)Nprtcl * 7);
    for (long p = 0; p < Nprtcl; ++p) {
        for (int c = 0; c < 3; ++c) { coord[p*7+c] = pos[p*3+c]; coord[p*7+3+c] = vel[p*3+c]; }
        coord[p*7+6] = mass[p];
    }
    CUDA_CHECK(cudaMemcpy(coord_y, coord.data(), (size_t)Nprtcl * 7 * sizeof(double), cudaMemcpyHostToDevice));
}

__global__ void k_add_scalar(real_t* __restrict__ a, long n, real_t c)
{
    GRID_STRIDE(t, n) { a[t] += c; }
}

// clip_zero (forced-dns-sm-big.py:309) on one padded (N,N,ZP) physical buffer:
// clip the negatives, then rescale by old mean / new mean. The ratio needs both
// global sums back on the host first, which is why this is two kernels and a
// synchronising copy -- the MPI code's two comm.allreduce calls.
static void clip_zero_phys(real_t* n)
{
    double h_old = 0.0, h_new = 0.0;
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, 2 * sizeof(double), g_stream));
    LAUNCH(k_clip_accum, nblocks(NR_), TPB, n, NR_, d_sum, d_sum + 1);
    CUDA_CHECK(cudaMemcpy(&h_old, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_new, d_sum + 1, sizeof(double), cudaMemcpyDeviceToHost));
    if (h_old * h_new <= 0.0) { fprintf(stderr, "clip_zero: means must be nonzero and share a sign\n"); exit(1); }
    LAUNCH(k_clip_rescale, nblocks(NR_), TPB, n, NR_, (real_t)(h_old / h_new));
}

// ------------------------------------------------------------------ classical RK4, one step
// Mirrors forced-dns-sm-big.py's RK4 (432-478) exactly: stage i's trial state
// is y + c_i*k_{i-1} (c = {0, h/2, h/2, h}), each stage's derivative is
// weighted by w = {h/6, h/3, h/3, h/6} into a running accumulator, and
// y_new = y + accumulator. Velocity, every field's density and every field's
// droplets share the same c/w and advance through the same four stages
// together (full_RHS's per-Stokes loop, run once per stage here).
static void rk4_step(double h)
{
    const double cT[4] = { 0.0, h/2, h/2, h };
    const double wA[4] = { h/6, h/3, h/3, h/6 };
    const real_t norm  = (real_t)(1.0 / N3d);
    const int nb  = nblocks(NS_);
    const int nbr = nblocks(NR_);
    const int nbq = nblocks(3 * g_M);
    const int nbs = std::max(1, std::min(64, (g_nshell + TPB - 1) / TPB));
    const double cellvol = DX * DX * DX;

    CUDA_CHECK(cudaMemsetAsync(d_Uacc, 0, 3 * g_M * sizeof(cplx_t), g_stream));
    for (auto& df : g_df) CUDA_CHECK(cudaMemsetAsync(df.nk_acc, 0, g_M * sizeof(cplx_t), g_stream));
    for (auto& df : g_df) CUDA_CHECK(cudaMemsetAsync(df.coord_acc, 0, df.Nprtcl * 7 * sizeof(double), g_stream));

    for (int st = 0; st < 4; ++st) {
        // ---- velocity trial state
        LAUNCH(k_rk4_trial_c, nbq, TPB, d_U, d_Uy, d_ku, 3 * g_M, (real_t)cT[st]);

        // ---- nonlinear accumulation, two phase-shift passes (rk_stage's own loop)
        CUDA_CHECK(cudaMemsetAsync(d_q, 0, 3 * g_M * sizeof(cplx_t), g_stream));
        CUDA_CHECK(cudaMemsetAsync(d_Ek, 0, sizeof(double), g_stream));
        LAUNCH(k_shell_energy, nbs, TPB, d_U, g_M, d_shell, g_nshell, d_loff, d_Ek);
        // forced-dns-sm-big.py's full_RHS does the SHIFTED pass first and the
        // unshifted one second, and hands the leftover unshifted u/v to pRHS,
        // so pass 0 here must be the shifted one: shifted = 1 - pass. The
        // forcing is added in the unshifted pass (`rhsuk += ... + fk[0]*dealias`).
        for (int pass = 0; pass < 2; ++pass) {
            const int shifted = 1 - pass;
            LAUNCH(k_scatter, nb, TPB, d_U, g_M, d_loff, d_u, norm, shifted, d_phase);
            for (int c = 0; c < 3; ++c) g_fft.inverse(d_u + c * NS_);
            for (int w = 0; w < 5; ++w) {
                LAUNCH(k_product, nbr, TPB, (const real_t*)d_u, (real_t*)d_P, w);
                g_fft.forward(d_P);
                LAUNCH(k_accum_prod, nb, TPB, d_q, g_M, (const cplx_t*)d_P, d_loff, w, shifted, d_phase);
            }
            // the gradient the Basdevant products carry in place of the
            // rotational form's -- needed because vk subtracts the UNPROJECTED
            // accumulator (see this file's velocity-path comment, and
            // droplet_kernels.cuh's k_rot_corr_prod)
            LAUNCH(k_rot_corr_prod, nblocks(NR_), TPB, (const real_t*)d_u, (real_t*)d_P);
            g_fft.forward(d_P);
            LAUNCH(k_accum_rot_corr, nblocks(NS_), TPB, d_q, g_M, (const cplx_t*)d_P, d_loff, shifted, d_phase);
            if (!shifted) LAUNCH(k_add_force, nbs, TPB, d_q, g_M, d_U, d_shell, g_nshell, d_loff, d_Ek, g_eps);
        }
        // d_u now holds physical UNSHIFTED u (pass 1, the last) -- the "u"
        // k_droplet_rhs interpolates, matching what full_RHS passes to pRHS.

        LAUNCH(k_velocity_rhs, nb, TPB, d_q, d_U, d_ku, g_M, d_loff, (real_t)g_nu);
        // vk is written over the accumulator it reads: k_velocity_rhs has already
        // taken what it needed from d_q, nothing else reads d_q before the next
        // stage memsets it, and k_make_v reads and writes the same element t with
        // no cross-index dependency -- so rhsuk and vk may be one buffer (that is
        // why neither parameter carries __restrict__).
        LAUNCH(k_make_v, nbq, TPB, d_U, d_ku, d_q, d_q, 3 * g_M, (real_t)g_tps);

        // ---- drift velocity v/vtemp, physical, shared across every field this stage
        // d_q now holds vk: three components at stride g_M, so this is the VECTOR
        // k_scatter (same call shape as d_U's above), not k_scatter_s. It scatters
        // straight into the physical buffer and the transform runs in place there:
        // 3*NRP reals is exactly 3*NS_ complex (NRP = 2*NS_) in the same in-place
        // R2C layout, so no separate full-spectral staging buffer is needed.
        LAUNCH(k_scatter, nb, TPB, d_q, g_M, d_loff, (cplx_t*)d_vtemp, norm, 1, d_phase);
        for (int c = 0; c < 3; ++c) g_fft.inverse((cplx_t*)d_vtemp + c * NS_);
        LAUNCH(k_add_scalar, nblocks(NRP), TPB, d_vtemp + 2 * NRP, NRP, (real_t)(-g_tps * g_gravity));

        LAUNCH(k_scatter, nb, TPB, d_q, g_M, d_loff, (cplx_t*)d_v, norm, 0, d_phase);
        for (int c = 0; c < 3; ++c) g_fft.inverse((cplx_t*)d_v + c * NS_);
        LAUNCH(k_add_scalar, nblocks(NRP), TPB, d_v + 2 * NRP, NRP, (real_t)(-g_tps * g_gravity));

        LAUNCH(k_rk4_accum_c, nbq, TPB, d_Uacc, d_ku, 3 * g_M, (real_t)wA[st]);

        // ---- every field's density + droplets, this stage
        for (auto& df : g_df) {
            const int nbp = nblocks(df.Nprtcl);
            LAUNCH(k_rk4_trial_r, nblocks(df.Nprtcl * 7), TPB, df.coord_trial, df.coord_y, df.coord_k,
                   df.Nprtcl * 7, cT[st]);

            // physical trial n, clipped (forced-dns-sm-big.py's clip_zero, nislist
            // row). The carried state n_y is PHYSICAL, so the trial is built in
            // physical space too, exactly as forced-dns-sm-big.py:426 builds it:
            // the compact stage derivative nk_k is scattered and inverse-
            // transformed into d_P, which k_rk4_trial_p turns into n_y + c*kn in
            // place. nk_k is memset at startup and finite after every stage, so
            // d_P is finite here even at st = 0, where cT[0] = 0 would otherwise
            // make 0*NaN = NaN out of uninitialised scratch.
            LAUNCH(k_scatter_s, nb, TPB, df.nk_k, d_loff, d_P, norm, 0, d_phase);
            g_fft.inverse(d_P);
            LAUNCH(k_rk4_trial_p, nbr, TPB, (real_t*)d_P, df.n_y, NR_, (real_t)cT[st]);
            clip_zero_phys((real_t*)d_P);

            // droplets first, while d_P still holds the PHYSICAL clipped n -- the
            // same array full_RHS passes to pRHS as n[ii]. Nothing between here and
            // the forward transform below wants the spectrum, so the physical field
            // is read in place instead of being copied to a buffer of its own.
            LAUNCH(k_droplet_rhs, nbp, TPB, df.coord_trial, (const real_t*)d_u, d_v, (const real_t*)d_P,
                   df.Nprtcl, g_factor, g_sts, g_tf, g_growthfactor, g_decelerationfactor,
                   g_n_cs_factor, g_rhop, g_gravity, df.coord_k, df.exterp0);
            CUDA_CHECK(cudaMemsetAsync(d_fc, 0, NRP * sizeof(real_t), g_stream));
            LAUNCH(k_deposit, nbp, TPB, df.coord_trial, df.exterp0, df.Nprtcl, d_fc, cellvol);

            // forward-transform the clipped n; df.nk_trial now holds its spectrum (nk_work)
            g_fft.forward(d_P);
            LAUNCH(k_gather_s, nb, TPB, d_P, df.nk_trial, d_loff);

            // divergence of v*n, two phase-shift passes (full_RHS's divvnk, 405-423)
            CUDA_CHECK(cudaMemsetAsync(d_divvnk, 0, g_M * sizeof(cplx_t), g_stream));
            LAUNCH(k_scatter_s, nb, TPB, df.nk_trial, d_loff, d_P, norm, 1, d_phase);
            g_fft.inverse(d_P);
            for (int c = 0; c < 3; ++c) {
                LAUNCH(k_mul_phys, nblocks(NRP), TPB, (const real_t*)d_P, d_vtemp + c * NRP, d_vn + c * NRP, NRP);
                g_fft.forward((cplx_t*)(d_vn + c * NRP));
            }
            LAUNCH(k_div_accum_c, nb, TPB, d_divvnk, g_M, (const cplx_t*)d_vn, (const cplx_t*)(d_vn + NRP),
                   (const cplx_t*)(d_vn + 2 * NRP), d_loff, 1, d_phase);
            // The unshifted pass needs irfft(nk_trial), the BAND-LIMITED clipped
            // n -- forced-dns-sm-big.py's `vn[:] = irfft_mpi(nk, ntemp)[None,...]*v`
            // with nk = rfft(n)*dealias. It is NOT the clipped physical n the
            // droplets read above: clipping breaks the band limit, so that field
            // (which k_droplet_rhs wants, and which full_RHS passes to pRHS as
            // n[ii]) and irfft(nk_trial) differ by the content the dealias mask
            // removed. One extra inverse transform per field per stage, into d_P
            // again -- which is also why the droplet calls had to come first.
            LAUNCH(k_scatter_s, nb, TPB, df.nk_trial, d_loff, d_P, norm, 0, d_phase);
            g_fft.inverse(d_P);
            for (int c = 0; c < 3; ++c) {
                LAUNCH(k_mul_phys, nblocks(NRP), TPB, (const real_t*)d_P, d_v + c * NRP, d_vn + c * NRP, NRP);
                g_fft.forward((cplx_t*)(d_vn + c * NRP));
            }
            LAUNCH(k_div_accum_c, nb, TPB, d_divvnk, g_M, (const cplx_t*)d_vn, (const cplx_t*)(d_vn + NRP),
                   (const cplx_t*)(d_vn + 2 * NRP), d_loff, 0, d_phase);

            g_fft.forward((cplx_t*)d_fc);
            LAUNCH(k_gather_s, nb, TPB, (const cplx_t*)d_fc, d_fck, d_loff);

            CUDA_CHECK(cudaMemsetAsync(df.nk_k, 0, g_M * sizeof(cplx_t), g_stream));
            LAUNCH(k_rk4_accum_c, nblocks(g_M), TPB, df.nk_k, d_divvnk, g_M, (real_t)(-1.0));
            LAUNCH(k_rk4_accum_c, nblocks(g_M), TPB, df.nk_k, d_fck, g_M, (real_t)(-1.0));

            LAUNCH(k_rk4_accum_c, nblocks(g_M), TPB, df.nk_acc, df.nk_k, g_M, (real_t)wA[st]);
            LAUNCH(k_rk4_accum_r, nblocks(df.Nprtcl * 7), TPB, df.coord_acc, df.coord_k, df.Nprtcl * 7, wA[st]);
        }
    }

    LAUNCH(k_rk4_accum_c, nbq, TPB, d_Uy, d_Uacc, 3 * g_M, (real_t)1.0);

    // forced-dns-sm-big.py:657-663, the reality condition, then 667-670, the
    // divergence-free projection, in that order, on the accepted state. The
    // first is the round trip u = irfft(uknew); uk = rfft(u): rfft adds
    // dependent degrees of freedom at kz = 0 which the step evolves
    // separately, and the round trip is what makes them mutually consistent
    // again. It must be a no-op to round-off on an already-valid state, and it
    // is because this file's inverse carries the 1/N^3 and its forward carries
    // nothing: k_scatter applies norm = 1/N^3 so scatter+inverse IS numpy's
    // irfft, and cuFFT's D2Z alone IS numpy's rfft -- the same pairing every
    // other transform in this file uses. d_u is free here: the stage loop has
    // finished with the physical velocity it held, and the per-field loop
    // below works in d_P.
    LAUNCH(k_scatter, nb, TPB, d_Uy, g_M, d_loff, d_u, norm, 0, d_phase);
    for (int c = 0; c < 3; ++c) g_fft.inverse(d_u + c * NS_);
    for (int c = 0; c < 3; ++c) g_fft.forward(d_u + c * NS_);
    LAUNCH(k_gather, nb, TPB, (const cplx_t*)d_u, d_Uy, g_M, d_loff);
    LAUNCH(k_project, nb, TPB, d_Uy, g_M, d_loff);

    // d_U is NOT refreshed from d_Uy here. It is the stage trial register, and
    // k_rk4_trial_c rewrites every one of its 3*g_M elements at stage 0 of the
    // next step before anything reads it, so a copy would be dead. save_state
    // reads d_Uy directly for the same reason. That is one 3*g_M copy saved per
    // step -- 0.165 GiB at N = 256, 1.32 GiB at N = 512 -- and no arithmetic
    // changes, so results are bit-identical.
    for (auto& df : g_df) {
        // forced-dns-sm-big.py's `nnew += h/6*kn` chain, carried physically: the
        // accumulated compact increment is scattered and inverse-transformed
        // into d_P, then added onto n_y point by point. k_rk4_trial_p with
        // coeff = 1 IS that add -- it computes first[i] = second[i] + 1*first[i]
        // -- so with n_y first and d_P second it is n_y += d_P, in place and ZP-
        // aware, with no buffer of its own.
        LAUNCH(k_scatter_s, nb, TPB, df.nk_acc, d_loff, d_P, norm, 0, d_phase);
        g_fft.inverse(d_P);
        LAUNCH(k_rk4_trial_p, nbr, TPB, df.n_y, (const real_t*)d_P, NR_, (real_t)1.0);
        // forced-dns-sm-big.py:639, `n[jj] = clip_zero(nnew[jj])`: the ACCEPTED
        // state is clipped and rescaled once the step is over, so the carried
        // density -- and what gets written out -- obeys the same convention the
        // MPI code's does. This is a deliberate extra pass, NOT a duplicate of
        // the stage-trial clip above (that one clips a throwaway trial state;
        // this one clips the state the next step starts from). n_y is already
        // physical and padded, so the clip is a direct call on it.
        clip_zero_phys(df.n_y);

        LAUNCH(k_rk4_accum_r, nblocks(df.Nprtcl * 7), TPB, df.coord_y, df.coord_acc, df.Nprtcl * 7, 1.0);
        LAUNCH(k_wrap_pos, nblocks(df.Nprtcl), TPB, df.coord_y, df.Nprtcl);
    }
}

// Strips the ZP padding and writes a packed (N,N,N) float64 .npy -- the same
// layout unpack_checkpoint.py produces, so a restart can read this file back.
static void dump_n_npy(const std::string& path, real_t* n_y)
{
    // n_y is the carried state itself and is already physical, so this is a copy
    // down and a de-pad -- no transform, and so no 1/N^3 normalisation to apply.
    std::vector<double> padded((size_t)NRP);
    CUDA_CHECK(cudaMemcpy(padded.data(), n_y, (size_t)NRP * sizeof(real_t), cudaMemcpyDeviceToHost));
    std::vector<double> packed((size_t)NR_);
    for (long x = 0; x < N; ++x)
        for (long y = 0; y < N; ++y)
            memcpy(&packed[(x*N+y)*N], &padded[(x*N+y)*ZP], N * sizeof(double));
    write_npy(path, "<f8", { N, N, N }, packed.data(), packed.size() * sizeof(double));
}

// prtclid rides along unchanged: nothing on one GPU reorders a field's
// droplets, so row p out is the droplet row p came in as. Writing it is what
// makes a droplet-by-droplet comparison against the MPI run possible at all,
// since that run's send() does reorder.
static void dump_particles_npy(const std::string& pdir, double* coord_y, long Nprtcl,
                               const std::vector<double>& prtclid)
{
    std::vector<double> coord((size_t)Nprtcl * 7);
    CUDA_CHECK(cudaMemcpy(coord.data(), coord_y, (size_t)Nprtcl * 7 * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> pos((size_t)Nprtcl * 3), vel((size_t)Nprtcl * 3), mass((size_t)Nprtcl);
    for (long p = 0; p < Nprtcl; ++p) {
        for (int c = 0; c < 3; ++c) { pos[p*3+c] = coord[p*7+c]; vel[p*3+c] = coord[p*7+3+c]; }
        mass[p] = coord[p*7+6];
    }
    write_npy(pdir + "/pos.npy",  "<f8", { Nprtcl, 3 }, pos.data(),  pos.size()  * sizeof(double));
    write_npy(pdir + "/vel.npy",  "<f8", { Nprtcl, 3 }, vel.data(),  vel.size()  * sizeof(double));
    write_npy(pdir + "/mass.npy", "<f8", { Nprtcl },    mass.data(), mass.size() * sizeof(double));
    write_npy(pdir + "/prtclid.npy", "<f8", { Nprtcl, 1 }, prtclid.data(), prtclid.size() * sizeof(double));
}

// ------------------------------------------------------------------ save (G11, C6)
// forced-dns-sm-big.py's save(), reduced to what this port writes. Output goes
// to <out>/time_<t>/ to tdec decimals, with the SAME two subdirectory
// spellings the restart read builds -- density <wg>_sts_..._stb_..., particles
// <wg>_stb_..._sts_..., not interchangeable. A savefields save writes the
// velocity, every field's n and every field's particles; a particles-only save
// writes the particles alone, which is why a fields folder is the only kind a
// restart can come from. The files inside are raw .npy, not the MPI code's
// compressed .npz, and Energy_spectrum/Flux_spectrum are not written at all --
// both are known, deliberate gaps recorded in HANDOVER-GPU-PORT.md.
static void save_state(const std::string& out_dir, const char* wg, double tt, bool savefields)
{
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    int tdec = std::max(0, (int)std::ceil(-std::log10(std::min(DT_SAVE_FIELDS, DT_SAVE_PARTICLES))));
    char tbuf[64];
    snprintf(tbuf, sizeof tbuf, "time_%.*f", tdec, tt);
    std::string tdir = out_dir + "/" + tbuf;
    mkdir(out_dir.c_str(), 0755);
    mkdir(tdir.c_str(), 0755);

    if (savefields) {
        // d_Uy, the carried state, NOT d_U: d_U is the stage trial register and
        // only ever equals the accepted state between steps. Reading d_Uy here
        // is what lets rk4_step skip a 3*g_M copy per step. scale 1.0, since the
        // restart read gathers uk.npy in without scaling and the two must agree.
        LAUNCH(k_scatter, nblocks(NS_), TPB, d_Uy, g_M, d_loff, d_u, (real_t)1.0, 0, d_phase);
        dump_complex_npy(tdir + "/uk.npy", d_u + 0 * NS_, NS_);
        dump_complex_npy(tdir + "/vk.npy", d_u + 1 * NS_, NS_);
        dump_complex_npy(tdir + "/wk.npy", d_u + 2 * NS_, NS_);
    }
    for (size_t i = 0; i < g_fields.size(); ++i) {
        std::string pdir = particle_dir(tdir, wg, g_fields[i].stb, g_sts, g_fields[i].init_name);
        mkdir(pdir.c_str(), 0755);
        dump_particles_npy(pdir, g_df[i].coord_y, g_fields[i].Nprtcl, g_fields[i].prtclid);
        if (!savefields) continue;
        std::string ddir = density_dir(tdir, wg, g_sts, g_fields[i].stb, g_fields[i].init_name);
        mkdir(ddir.c_str(), 0755);
        dump_n_npy(ddir + "/n.npy", g_df[i].n_y);
    }
    fprintf(stderr, "saved %s%s\n", tdir.c_str(), savefields ? "" : " (particles only)");
}

// forced-dns-sm-big.py:622-624's save trigger, a PROXIMITY test and not a
// counter: |sin(pi*tt/Ts)| is small exactly when tt is near a multiple of Ts,
// and the half-width of that window, sin(pi*h/(2*Ts)), scales with the CURRENT
// step size, so it must be recomputed every step rather than fixed.
// KNOWN CONSTRAINT between the hand-edited constants, not something to fix in
// code: if h ever exceeded 2*Ts the right-hand side would exceed 1, which
// |sin| can never reach, and saves would stop for good. Only DTMAX holds h
// below that, so DTMAX must stay under 2*DT_SAVE_PARTICLES, the smaller of the
// two intervals.
// The two sine arguments are grouped as the Python groups them -- tt/Ts*PI,
// not PI*tt/Ts -- because near a multiple of Ts the whole test is decided by
// how the argument rounds, and regrouping would move the result by an ulp of
// an argument of order 2800 and could flip a save on or off.
static bool save_due(double tt, double h, double Ts)
{
    return std::fabs(std::sin(tt / Ts * PI_D)) - std::sin(0.5 * h / Ts * PI_D) < 1e-12;
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv)
{
    std::string restart_dir, out_dir = "./gpu_out";
    std::string stb_csv, init_csv;
    double m_keta = 2.0;
    int    gravity_flag = 0;
    double tend = 0.0;
    bool   have_tend = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        bool has = (i + 1 < argc);
        if      (!strcmp(a, "--restart") && has) restart_dir = argv[++i];
        else if (!strcmp(a, "--out")     && has) out_dir     = argv[++i];
        else if (!strcmp(a, "--stb")     && has) stb_csv     = argv[++i];
        else if (!strcmp(a, "--init")    && has) init_csv    = argv[++i];
        else if (!strcmp(a, "--tend")    && has) { tend = atof(argv[++i]); have_tend = true; }
        else if (!strcmp(a, "--m")       && has) m_keta      = atof(argv[++i]);
        else if (!strcmp(a, "--gravity") && has) gravity_flag= atoi(argv[++i]);
        else { fprintf(stderr, "unrecognized argument: %s\n", a); return 1; }
    }
    if (restart_dir.empty()) { fprintf(stderr, "--restart <time_dir> is required (unpack_checkpoint.py'd)\n"); return 1; }
    if (!have_tend) { fprintf(stderr, "--tend <t> is required: the time to stop at. For a one-step "
                                      "comparison give the restart folder's own time\n"); return 1; }

    // ---- parameters, forced-dns-sm-big.py's block verbatim (section: Parameters)
    const char* wg = gravity_flag ? "with_g" : "wo_g";
    double kmax = std::floor(N * 1.4142135623730951 / 3.0);
    double eta  = m_keta / kmax;
    const double nu0 = P_NU0, lp_exp = 2.0 * ((double)LP - 1.0 / 3.0);
    g_nu = nu0 * std::pow(eta, lp_exp);
    g_tf = eta * eta / g_nu;
    g_sts = P_STS;
    g_tps = g_sts * g_tf;
    const double tau_eta_dim = P_TAU_ETA_DIM, eta_dim = P_ETA_DIM;
    g_gravity = gravity_flag ? 9.81 * (tau_eta_dim * tau_eta_dim / eta_dim) * (eta / (g_tf * g_tf)) : 0.0;
    g_eps = P_EPS;   // f0/(2 pi)^3, nshells = 1 -- see the comment above k_scatter
    g_M0 = P_M0; g_rhop = P_RHOP;
    g_n_cs_factor = g_M0 / g_rhop;
    g_growthfactor = std::pow(3.0 * std::sqrt(PI_D) / (4.0 * g_rhop), 2.0 / 3.0);
    g_decelerationfactor = g_growthfactor * g_rhop * g_n_cs_factor;
    g_factor = (4.0 * PI_D * g_rhop) * std::pow(g_tf, 1.5) * std::pow(g_nu, 1.5) * 27.0
             / (3.0 * std::pow(g_rhop, 1.5) * std::pow(2.0, 1.5));

    // Every parameter this run is actually using, printed before anything is
    // allocated. The P_* defaults above and run_gpu.py's own block are two
    // copies of the same numbers, and nothing makes a direct nvcc build use the
    // Python's values -- so the binary says what it got, and that line can be
    // read against what run_gpu.py printed. forced-dns-sm-big.py:136 prints the
    // same handful for the same reason.
    fprintf(stderr,
        "N %d  m %g  kmax %g  eta %g  nu %g  Re %g  tf %g\n"
        "sts %g  M0 %g  rhop %g  eps %g  gravity %g  lp %d\n"
        "dtmin %g  dtmax %g  dt_save_fields %g  dt_save_particles %g\n",
        N, m_keta, kmax, eta, g_nu, 1.0 / g_nu, g_tf,
        g_sts, g_M0, g_rhop, g_eps, g_gravity, LP,
        DTMIN, DTMAX, DT_SAVE_FIELDS, DT_SAVE_PARTICLES);

    // ---- ensemble (plan section 3: NFIELDS is a runtime input)
    std::vector<std::string> stb_tok, init_tok;
    { std::string t; for (char c : stb_csv)  { if (c == ',') { stb_tok.push_back(t); t.clear(); } else t += c; } if (!t.empty() || !stb_tok.empty()) stb_tok.push_back(t); }
    { std::string t; for (char c : init_csv) { if (c == ',') { init_tok.push_back(t); t.clear(); } else t += c; } if (!t.empty() || !init_tok.empty()) init_tok.push_back(t); }
    // The ensemble is the OUTER PRODUCT of the two lists, init varying slowest:
    // f = i_init*n_stb + i_stb, NFIELDS = n_stb*n_init (plan section 3). Give
    // the 4 distinct Stokes numbers and the 5 distinct init names and this
    // builds the same 20 flat entries forced-dns-sm-big.py:41-43 does, in the
    // same order. Either list empty gives NFIELDS = 0, which is a supported
    // run: the velocity alone, with no density field and no droplets.
    // Nprtcl per field is read out of that field's own pos.npy (particle_rows),
    // not computed from the Stokes number: see particle_rows' comment.
    for (size_t ii = 0; ii < init_tok.size(); ++ii)
        for (size_t is = 0; is < stb_tok.size(); ++is) {
            double stb = atof(stb_tok[is].c_str());
            long Nprtcl = particle_rows(particle_dir(restart_dir, wg, stb, g_sts, init_tok[ii]));
            g_fields.push_back({ stb, init_tok[ii], Nprtcl });
        }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&g_stream));

    std::vector<long> loff = build_line_offsets();
    g_M = loff[NLINE];
    CUDA_CHECK(cudaMalloc(&d_loff, (NLINE + 1) * sizeof(long)));
    CUDA_CHECK(cudaMemcpy(d_loff, loff.data(), (NLINE + 1) * sizeof(long), cudaMemcpyHostToDevice));
    build_phase_table(&d_phase);
    build_shell(1, &d_shell, &g_nshell, loff);     // forced-dns-sm-big.py: shell_no = {1}

    // plan() only sizes the work area; attach() is what allocates it. The
    // refusal therefore runs between the two, so that nothing it counts has
    // already been taken out of cudaMemGetInfo's free total.
    g_fft.plan();
    startup_refusal(g_fields, g_fft.ws);
    g_fft.attach();

    CUDA_CHECK(cudaMalloc(&d_P, NS_ * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_u, 3 * NS_ * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_v, 3 * NRP * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_vtemp, 3 * NRP * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_vn, 3 * NRP * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_fc, NRP * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_Ek, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_red, 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q, 3 * g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_U, 3 * g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_Uy, 3 * g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_Uacc, 3 * g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_ku, 3 * g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_divvnk, g_M * sizeof(cplx_t)));
    CUDA_CHECK(cudaMalloc(&d_fck, g_M * sizeof(cplx_t)));

    allocate_fields(g_fields);
    // Stage 0's trial has coeff = 0, and k_rk4_trial_c/k_rk4_trial_r multiply
    // the "k" register through rather than special-casing a zero coefficient,
    // so those registers must start at a defined zero -- fresh cudaMalloc
    // memory is uninitialised and 0 * NaN is NaN, not 0.
    CUDA_CHECK(cudaMemsetAsync(d_ku, 0, 3 * g_M * sizeof(cplx_t), g_stream));
    for (auto& df : g_df) {
        CUDA_CHECK(cudaMemsetAsync(df.nk_k, 0, g_M * sizeof(cplx_t), g_stream));
        CUDA_CHECK(cudaMemsetAsync(df.coord_k, 0, df.Nprtcl * 7 * sizeof(double), g_stream));
    }

    // ---- restart (G11): unpack_checkpoint.py's whole-domain .npy layout
    if (!load_complex_npy(restart_dir + "/uk.npy", d_u + 0 * NS_, NS_) ||
        !load_complex_npy(restart_dir + "/vk.npy", d_u + 1 * NS_, NS_) ||
        !load_complex_npy(restart_dir + "/wk.npy", d_u + 2 * NS_, NS_)) {
        fprintf(stderr, "restart: run unpack_checkpoint.py on %s first\n", restart_dir.c_str());
        return 1;
    }
    LAUNCH(k_gather, nblocks(NS_), TPB, d_u, d_Uy, g_M, d_loff);
    // d_Uy alone is the loaded state. d_U needs no seeding: every reader of it
    // is inside a stage, and k_rk4_trial_c writes it in full first.

    for (size_t i = 0; i < g_fields.size(); ++i) {
        std::string ddir = density_dir(restart_dir, wg, g_sts, g_fields[i].stb, g_fields[i].init_name);
        std::string pdir = particle_dir(restart_dir, wg, g_fields[i].stb, g_sts, g_fields[i].init_name);
        load_n_into_ny(ddir + "/n.npy", g_df[i].n_y);
        load_particles_into_coordy(pdir, g_fields[i].Nprtcl, g_df[i].coord_y, g_fields[i].prtclid);
    }

    // ---- the time loop, forced-dns-sm-big.py:615-685 with the MPI
    // collectives dropped. tt starts at the restart folder's own time, so a
    // one-step comparison is a run with --tend set to that same time: the
    // loop condition holds once, one step is taken, and tt then advances past
    // tend. h starts at DTMIN, which is line 615 exactly -- EVERY run,
    // restart or not, takes its first step at dtmin and only then adapts.
    double tt = 0.0;
    size_t tpos = restart_dir.rfind("time_");
    if (tpos != std::string::npos) tt = atof(restart_dir.c_str() + tpos + 5);
    double h = DTMIN;
    const double uk_limit = 100.0 * N3d;      // forced-dns-sm-big.py:678's threshold
    const double seed_max = DBL_MAX;

    double calc_time = 0.0;              // summed step time, excluding saves
    long   nstep = 0;
    while (tt <= tend) {
        bool savefields = save_due(tt, h, DT_SAVE_FIELDS);
        if (savefields || save_due(tt, h, DT_SAVE_PARTICLES)) save_state(out_dir, wg, tt, savefields);

        // Started after the save, as forced-dns-sm-big.py:626 resets its own
        // clock there, so this is the step and not the I/O. It is a true step
        // time rather than launch overhead because the blocking cudaMemcpy in
        // the guard below drains the stream before the clock is read again --
        // every kernel of this step has finished by then.
        auto t_step = std::chrono::steady_clock::now();

        rk4_step(h);

        // hnew, forced-dns-sm-big.py:633-638: the smallest accepted droplet
        // mass in each field, combined as 0.5*st*tf with
        // st = (mass/factor)^(2/3). There is deliberately NO floor at DTMIN --
        // the Python has none, dtmin appearing nowhere in its step logic -- and
        // a field with no droplets leaves the DBL_MAX seed in place, so it
        // cannot drag hnew down.
        double hnew = DTMAX;
        for (size_t i = 0; i < g_df.size(); ++i) {
            DropletField& df = g_df[i];
            CUDA_CHECK(cudaMemcpy(d_red, &seed_max, sizeof(double), cudaMemcpyHostToDevice));
            LAUNCH(k_mass_min, nblocks(df.Nprtcl), TPB, df.coord_y, df.Nprtcl, d_red);
            double massmin = 0.0;
            CUDA_CHECK(cudaMemcpy(&massmin, d_red, sizeof(double), cudaMemcpyDeviceToHost));
            // A field with no droplets leaves the DBL_MAX seed, which is what
            // makes it contribute nothing. Anything else non-positive or
            // non-finite is a droplet that has gone wrong, and both failure
            // modes here are silent: mass 0 gives h = 0 and a loop that never
            // advances, while a NaN mass is skipped by block_min's unsigned
            // bit-pattern compare and leaves the seed, so the step jumps to
            // DTMAX as though the field were empty. Stop instead.
            if (df.Nprtcl > 0 && !(massmin > 0.0 && massmin < DBL_MAX)) {
                fprintf(stderr, "stokes %g init %s: smallest droplet mass is %g at time %g, "
                                "which is not a usable step size\n",
                        df.stb, g_fields[i].init_name.c_str(), massmin, tt);
                return 1;
            }
            hnew = std::min(hnew, 0.5 * std::pow(massmin / g_factor, 2.0 / 3.0) * g_tf);
        }
        h = hnew;

        // The reality condition and the projection ran at the end of rk4_step
        // (forced-dns-sm-big.py:657-670), so d_Uy already holds the cleaned
        // accepted state this guard must test -- and a diverged state is never
        // written, since the save check sits at the top of the next iteration.
        CUDA_CHECK(cudaMemsetAsync(d_red + 1, 0, 2 * sizeof(double), g_stream));
        LAUNCH(k_uk_guard, nblocks(NS_), TPB, d_Uy, g_M, d_loff, d_red + 1, d_red + 2);
        double maxmod2 = 0.0, nonfinite = 0.0;
        CUDA_CHECK(cudaMemcpy(&maxmod2, d_red + 1, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&nonfinite, d_red + 2, sizeof(double), cudaMemcpyDeviceToHost));
        if (maxmod2 > uk_limit * uk_limit || nonfinite > 0.0) {
            fprintf(stderr, "Threshold exceeded at time %g: max|uk| = %g against %g, with %g "
                            "non-finite mode components. Nothing written for this step.\n",
                    tt, std::sqrt(maxmod2), uk_limit, nonfinite);
            return 1;
        }

        double dt_wall = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t_step).count();
        calc_time += dt_wall;
        ++nstep;
        // forced-dns-sm-big.py:620 and :644 print the same two things each step.
        // stderr, so piping stdout to a file still shows progress. "from" because
        // tt is still the time this step STARTED at -- it advances below, and by
        // the next h rather than the one used, which is the Python's own
        // ordering. Newline not \r: a long run's timings are worth keeping.
        fprintf(stderr, "step %ld  from t %.6f  took %.3f s  next h %.3e\n",
                nstep, tt, dt_wall, h);

        // tt advances by the NEW h, not by the h integrated with:
        // forced-dns-sm-big.py assigns h = hnew at line 643 and only then does
        // tt += h at line 685. Reproduced exactly, so that the time_<t> folder
        // names line up with the MPI run's.
        tt += h;
    }

    if (nstep) fprintf(stderr, "%ld steps, %.3f s total, %.3f s per step\n",
                       nstep, calc_time, calc_time / (double)nstep);
    save_state(out_dir, wg, tt, true);
    return 0;
}

// ===========================================================================
//  droplet_kernels.cuh -- the GPU-port CONTRACT (plan section 6, task G2) and
//  the kernels built on it (G3 scalar scatter/gather, G6 clip, G7 cosine
//  weights, G8 droplet RHS, G9 deposition, G10 classical-RK4 combinators).
//
//  Included from forced_ns_droplets.cu, AFTER that file has already defined
//  -- by duplicating the matching section of forced_ns_ck54_lowmem.cu, which
//  this header does not re-declare --:
//    real_t, cplx_t, C_(), N, Nf, NS_, NR_, ZP, NRP, POFF, NLINE, PI_D, L_D, DX, TPB,
//    GRID_STRIDE(t,n), SpecPt/spec_point(), CPt/compact_point(), block_add(),
//    nblocks(). Same mask (KTH2), same compact layout, same (N,N,Nf) full
//    layout and ZP = 2*Nf physical stride as the velocity path, so a droplet
//    field's compact array is scatter/gather-compatible with the velocity
//    kernels' d_loff table -- one table, shared, built once.
//
//  ---------------------------------------------------------------------------
//  1. STRUCT LAYOUT. One DropletField per ensemble member (plan section 3:
//     NFIELDS = n_stb*n_init, f = i_init*n_stb + i_stb, matching
//     forced-dns-sm-big.py:41-43). Allocated from an array sized at startup
//     from NFIELDS, never a fixed-length array or #define.
//
//  2. MEMORY MAP AS A FUNCTION OF NFIELDS. Only per-field STATE persists
//     across the field loop within an RK4 stage and across stages; the
//     working buffers a field's RHS evaluation touches (fc, the physical n,
//     the vn triple product, ...) are shared scratch, sized once, reused
//     serially for field 0, then field 1, ... -- exactly the MPI code's
//     `for ii in range(len(stb_s))` loop in full_RHS (forced-dns-sm-big.py,
//     around line 390), which also reuses one fc/nk/vn per iteration.
//
//       per field:  1 * NRP * sizeof(real_t)        (n_y, the carried density,
//                                                     PHYSICAL, padded (N,N,ZP))
//                 + 2 * g_M * sizeof(cplx_t)        (nk_acc, nk_k)
//                 + 3 * Nprtcl[f] * 7 * sizeof(double)  (coord_y, coord_acc, coord_k)
//
//     The density register is therefore NRP*8 + 2*g_M*16 bytes per field in the
//     fp64 build, replacing the old 3*g_M*16. The CARRIED density state is the
//     PHYSICAL field n_y, NRP reals in the same padded (N,N,ZP) in-place-R2C
//     layout k_clip_accum/k_clip_rescale walk -- no longer a compact g_M
//     spectrum. nk_acc and nk_k stay COMPACT g_M complex: they hold RK4
//     increments assembled from d_divvnk and d_fck, which are g_M-sized by
//     construction, so there is no out-of-band content for them to lose. The
//     shared nk_trial is unchanged.
//
//     ONLY those six are per-field: each carries state from one stage to the
//     next, with every other field's stage body running in between. The trial
//     registers do not -- nk_trial, coord_trial and exterp0 are each written at
//     the top of a field's stage body, read through that body, and recomputed
//     from the carried registers next stage -- so they are ONE buffer apiece,
//     shared by the whole ensemble and sized for the largest field, counted in
//     the fixed term below and not multiplied by the field count.
//
//       fixed (shared scratch, independent of NFIELDS):
//                   1 * NRP  * sizeof(real_t)         (fc, physical)
//                   3 * NRP  * sizeof(real_t)          (vn, the v*n triple product: a SEPARATE
//                                                        allocation of its own (d_vn), NOT an alias
//                                                        of the velocity path's physical buffers)
//                   1 * NS_  * sizeof(cplx_t)          (nk physical-pass scratch, scatter/gather;
//                                                        the clipped physical n the droplets read
//                                                        is this same buffer, read before the
//                                                        forward transform overwrites it)
//                   1 * g_M  * sizeof(cplx_t)          (nk_trial, shared)
//                   max_f Nprtcl[f] * 7 * sizeof(double)   (coord_trial, shared)
//                   max_f Nprtcl[f] * sizeof(double)        (exterp0, shared)
//
//     g_M is the SAME kept-mode count as the velocity path (same KTH2, same N),
//     so it does not vary with NFIELDS either. Plan section 3's budget table
//     calls these "RK4 (four compact registers)"; the fourth is the trial
//     register, which is shared rather than per-field, so the table's per-field
//     figures are superseded by HANDOVER-GPU-PORT.md's memory section.
//
//     ALLOCATION RULE (memory-reviewer finding): the pointers above are
//     offsets into ONE cudaMalloc per register kind, sized for every field
//     up front -- cudaMalloc(&d_nk_base, NFIELDS*2*g_M*sizeof(cplx_t)),
//     cudaMalloc(&d_n_base, NFIELDS*NRP*sizeof(real_t)) and
//     cudaMalloc(&d_coord_base, (sum of Nprtcl[f])*3*7*sizeof(double)) with a
//     prefix-sum offset table (one long per field, built the same way as
//     build_line_offsets()), THEN DropletField::coord_y etc. are set to
//     d_coord_base + offset[f]. NFIELDS separate small cudaMalloc calls is a
//     defect, not a style choice: it fragments the allocator and defeats the
//     single-work-area intent this whole port is for (plan section 1, item 2).
//
//  3. STARTUP REFUSAL. There is no single per_field figure to multiply by:
//     Nprtcl[f] varies by a factor of 128 across the documented stb values, so
//     forced_ns_droplets.cu sums the field list exactly,
//         bytes = fixed + sum over f of [ NRP*sizeof(real_t) + 2*g_M*sizeof(cplx_t)
//                                       + 3*Nprtcl[f]*7*sizeof(double) ]
//     and checks it against cudaMemGetInfo's free total BEFORE any droplet-field
//     cudaMalloc. If it does not fit, it reports the largest field count that
//     would by walking a PREFIX SUM over the list in ensemble order and stopping
//     at the first field that does not fit -- not by solving an averaged
//     per_field*x <= free - fixed, which would under-refuse on a prefix whose
//     fields are more expensive than the average. The shared trial buffers sit
//     in `fixed`, sized to the largest field in the whole list, which makes the
//     reported count conservative when that field is outside the prefix. See
//     startup_refusal() there.
//
//  4. coord LAYOUT. Nprtcl*7 doubles per field, row-major per particle:
//     [0:3) position, [3:6) velocity, [6] physical mass -- bit-for-bit the
//     same column order as particles.py's self.coord (2*d+1 with d=3), so
//     unpack_checkpoint.py's pos.npy/vel.npy/mass.npy need only be
//     interleaved back into this layout on load (G11), not reshaped further.
//
//  5. UNITS. coord[:,6] (mass) is PHYSICAL, never divided or multiplied by
//     M0 -- plan section 4. n_cs_factor = M0/rhop is the only place M0
//     appears in the droplet path.
// ===========================================================================

// ------------------------------------------------------------------ G2: struct
struct DropletField {
    double stb;          // this field's big-droplet Stokes number (I/O + physics)
    int    init_id;       // index into the host's init-name table; I/O path only
    long   Nprtcl;         // particle count for this field (forced-dns-sm-big.py:48)

    // density: the CARRIED state is PHYSICAL, NRP reals in the padded (N,N,ZP)
    // in-place-R2C layout; the RK4 increments stay compact spectral. nk_trial is
    // the SAME buffer in every field (note 2): shared scratch, not this field's.
    real_t *n_y;
    cplx_t *nk_acc, *nk_k, *nk_trial;

    // droplets -- Nprtcl*7 doubles each, see note 4 above; coord_trial, like
    // nk_trial, is shared scratch pointing at one ensemble-wide buffer
    double *coord_y, *coord_acc, *coord_k, *coord_trial;

    // the swept-droplet rate, Nprtcl doubles: k_droplet_rhs writes it,
    // k_deposit reads it in the same field's stage body (plan G9) -- kept out
    // of coord_k since it is not one of the 7 integrated quantities, only a
    // deposition input, and shared across fields for the same reason as above
    double *exterp0;
};

// ------------------------------------------------------------------ G7: cosine weights
// Replacement for cubic_w (forced_ns_ck54_lowmem.cu:513): same 4-point stencil
// anchored at offset -1 (particles.py's cosorder = arange(-1,3)), weights
// (1 + cos(pi*d/(2*dx)))/4 (particles.py:17-19 etc). s is the fractional part
// of pos/dx in [0,1); cos is even so the (1-s)/(2-s) taps need no sign care.
__device__ __forceinline__ void cosine_w(double s, double* w)
{
    w[0] = (1.0 + cos(PI_D * (s + 1.0) * 0.5)) * 0.25;
    w[1] = (1.0 + cos(PI_D *  s        * 0.5)) * 0.25;
    w[2] = (1.0 + cos(PI_D * (1.0 - s) * 0.5)) * 0.25;
    w[3] = (1.0 + cos(PI_D * (2.0 - s) * 0.5)) * 0.25;
}

// ------------------------------------------------------------------ G3: scalar scatter/gather
// k_scatter / k_gather (forced_ns_ck54_lowmem.cu:284,309) for ONE compact
// component instead of three -- the number-density field has no vector index.
__global__ void k_scatter_s(const cplx_t* __restrict__ Uc, const long* __restrict__ loff,
                            cplx_t* __restrict__ dst, real_t scale, int shifted,
                            const cplx_t* __restrict__ phase)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) { dst[t] = C_(0, 0); continue; }
        double aRe = scale, aIm = 0.0;
        if (shifted) {
            cplx_t p = phase[s.kx + s.ky + s.kz + POFF];
            aRe = (double)p.x * scale; aIm = (double)p.y * scale;
        }
        cplx_t v = Uc[s.c];
        dst[t] = C_((real_t)(aRe * v.x - aIm * v.y), (real_t)(aRe * v.y + aIm * v.x));
    }
}

__global__ void k_gather_s(const cplx_t* __restrict__ src, cplx_t* __restrict__ Uc,
                           const long* __restrict__ loff)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        Uc[s.c] = src[t];
    }
}

// ------------------------------------------------------------------ G4: rotational-form correction
// The rotational accumulator F[u x omega] and the convective/Basdevant one
// k_product/k_accum_prod build differ by the pure gradient grad((ux^2 + uy^2
// - uz^2)/2): u x omega = -(u.grad)u + grad(|u|^2/2), and the five Basdevant
// products already carry grad(uz^2) in place of grad(|u|^2/2). The Leray
// projection inside k_velocity_rhs removes that gradient, so ku is the same
// whichever form built it -- but vk = uk - tps*(ku - rhsuk) uses the
// UNPROJECTED accumulator, where nothing removes it, so vk needs the true
// rhsuk. These two kernels add the missing gradient to the accumulator, which
// then equals forced-dns-sm-big.py's rhsuk/rhsvk/rhswk. Do not delete them on
// the grounds that the two forms agree after projection: vk is not projected.
//
// Physical half-square g = (ux^2 + uy^2 - uz^2)/2, padded (N,N,ZP) layout and
// NRP component stride -- k_product's indexing exactly, one pass's u.
__global__ void k_rot_corr_prod(const real_t* __restrict__ u, real_t* __restrict__ P)
{
    GRID_STRIDE(t, NR_) {
        int  z = (int)(t % N);
        long r = t / N;
        int  y = (int)(r % N);
        int  x = (int)(r / N);
        long i = ((long)x * N + y) * ZP + z;
        double ux = (double)u[i], uy = (double)u[i + NRP], uz = (double)u[i + 2 * NRP];
        P[i] = (real_t)(0.5 * (ux * ux + uy * uy - uz * uz));
    }
}

// q_c += i*k_c*f*G for the three components, G = F[g] in the full (N,N,Nf)
// layout, f = 0.5 unshifted and conj(phase)/2 shifted -- the same two-pass
// average k_accum_prod applies to every other product, so call this once per
// pass with that pass's `shifted` flag, alongside the five k_accum_prod calls.
__global__ void k_accum_rot_corr(cplx_t* __restrict__ q, long M,
                                 const cplx_t* __restrict__ G,
                                 const long* __restrict__ loff, int shifted,
                                 const cplx_t* __restrict__ phase)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        double fRe = 0.5, fIm = 0.0;
        if (shifted) {
            cplx_t p = phase[s.kx + s.ky + s.kz + POFF];
            fRe = 0.5 * (double)p.x;  fIm = -0.5 * (double)p.y;
        }
        cplx_t a = G[t];
        double vr = fRe * a.x - fIm * a.y;
        double vi = fRe * a.y + fIm * a.x;
        double karr[3] = { (double)s.kx, (double)s.ky, (double)s.kz };
        for (int c = 0; c < 3; ++c) {
            cplx_t o = q[s.c + c * M];
            // i*k*(vr + i vi) = k*(-vi + i vr)
            q[s.c + c * M] = C_((real_t)(o.x - karr[c] * vi), (real_t)(o.y + karr[c] * vr));
        }
    }
}

// elementwise, compact layout: vk = uk - tps*(ku - rhsuk) (forced-dns-sm-big.py,
// full_RHS, the three "vk[c] = ..." lines right after the pressure-corrected
// ku is formed). Component-agnostic: the three components are contiguous at
// stride M_compact, so ONE call takes the full three-component count
// n = 3*g_M, not one call (or one count) per component. rhsuk must be the
// UNPROJECTED rotational-form accumulator, i.e. the convective accumulator
// after k_rot_corr_prod/k_accum_rot_corr have corrected it.
// rhsuk and vk carry no __restrict__ on purpose: the caller passes one buffer
// for both (the nonlinear accumulator is dead once this has read it), and each
// thread reads and writes only its own index t.
__global__ void k_make_v(const cplx_t* __restrict__ uk, const cplx_t* __restrict__ ku,
                         const cplx_t* rhsuk, cplx_t* vk,
                         long n, real_t tps)
{
    GRID_STRIDE(t, n) {
        cplx_t a = uk[t], b = ku[t], c = rhsuk[t];
        vk[t] = C_((real_t)(a.x - tps * (b.x - c.x)), (real_t)(a.y - tps * (b.y - c.y)));
    }
}

// ------------------------------------------------------------------ G6: positivity clip
// clip_zero (forced-dns-sm-big.py:309): clip negative n to zero, rescale to
// restore the pre-clip mean. Two kernels because the rescale ratio needs both
// sums back on the host first (the MPI code's two comm.allreduce calls).
//
// n sits in the padded (N,N,ZP) in-place-FFT layout (ZP = 2*Nf), the same
// layout every other physical buffer uses (k_product's own index translation,
// forced_ns_ck54_lowmem.cu:331-346) -- NOT a packed (N,N,N) array. npts is
// the LOGICAL point count NR_ = N*N*N; the two pad reals per line
// (forced_ns_ck54_lowmem.cu's own ZP = N + 2 at even N) are untouched and
// never enter the sum. Indexing the buffer as flat over npts instead (as if
// it were packed) would pull uninitialised pad memory into old_sum/new_sum --
// found and fixed in this session's review.
__global__ void k_clip_accum(real_t* __restrict__ n, long npts,
                             double* __restrict__ old_sum, double* __restrict__ new_sum)
{
    double so = 0.0, sn = 0.0;
    GRID_STRIDE(t, npts) {
        int  z = (int)(t % N);
        long r = t / N;
        int  y = (int)(r % N);
        int  x = (int)(r / N);
        long i = ((long)x * N + y) * ZP + z;
        double v = (double)n[i];
        so += v;
        v = v > 0.0 ? v : 0.0;
        n[i] = (real_t)v;
        sn += v;
    }
    block_add(so, old_sum);
    block_add(sn, new_sum);
}

__global__ void k_clip_rescale(real_t* __restrict__ n, long npts, real_t ratio)
{
    GRID_STRIDE(t, npts) {
        int  z = (int)(t % N);
        long r = t / N;
        int  y = (int)(r % N);
        int  x = (int)(r / N);
        long i = ((long)x * N + y) * ZP + z;
        n[i] = (real_t)((double)n[i] * (double)ratio);
    }
}

// ------------------------------------------------------------------ G8: droplet RHS
// Collapses particles.py's pRHS to its interior branch (plan's "what
// survives" note): one GPU is one domain, so every particle is always local
// and particle_exchange/send/Neighbor_alltoall never run. u is the flow
// velocity, v is k_make_v's output (the slow-manifold drift velocity), both
// physical, stride NRP per component, ZP z-stride (k_part_interp's layout).
// n is the field's own physical number density. factor/sts/tau_eta/rhop/
// growthfactor/decelerationfactor/n_cs_factor are the scalars built once on
// the host from particles.py's constructor (same formulas, plan section 4).
__global__ void k_droplet_rhs(const double* __restrict__ coord,     // trial state, Nprtcl*7
                              const real_t* __restrict__ u,          // 3*NRP
                              const real_t* __restrict__ v,          // 3*NRP
                              const real_t* __restrict__ n,          // NRP
                              long Nprtcl, double factor, double sts, double tau_eta,
                              double growthfactor, double decelerationfactor,
                              double n_cs_factor, double rhop, double gravity,
                              double* __restrict__ k_out,             // Nprtcl*7
                              double* __restrict__ exterp0)            // Nprtcl
{
    const double inv_dx = (double)N / L_D;
    GRID_STRIDE(p, Nprtcl) {
        double px = coord[p * 7 + 0], py = coord[p * 7 + 1], pz = coord[p * 7 + 2];
        double vx = coord[p * 7 + 3], vy = coord[p * 7 + 4], vz = coord[p * 7 + 5];
        double mass = coord[p * 7 + 6];

        double gx = px * inv_dx, gy = py * inv_dx, gz = pz * inv_dx;
        double fx = floor(gx), fy = floor(gy), fz = floor(gz);
        double wx[4], wy[4], wz[4];
        cosine_w(gx - fx, wx); cosine_w(gy - fy, wy); cosine_w(gz - fz, wz);
        int ix0 = (int)fx - 1, iy0 = (int)fy - 1, iz0 = (int)fz - 1;

        double u0 = 0.0, u1 = 0.0, u2 = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, nint = 0.0;
        for (int a = 0; a < 4; ++a) {
            int xi = ((ix0 + a) % N + N) % N;
            for (int b = 0; b < 4; ++b) {
                int yi = ((iy0 + b) % N + N) % N;
                double wab = wx[a] * wy[b];
                long base = ((long)xi * N + yi) * ZP;      // ZP stride: in-place R2C buffer, not N
                for (int c = 0; c < 4; ++c) {
                    int zi = ((iz0 + c) % N + N) % N;
                    double w = wab * wz[c];
                    long i = base + zi;
                    u0 += w * (double)u[i];            u1 += w * (double)u[i + NRP];       u2 += w * (double)u[i + 2 * NRP];
                    v0 += w * (double)v[i];            v1 += w * (double)v[i + NRP];       v2 += w * (double)v[i + 2 * NRP];
                    nint += w * (double)n[i];
                }
            }
        }

        double st  = pow(mass / factor, 2.0 / 3.0);
        double stfac = 1.0 + sqrt(sts / st); stfac *= stfac;
        double relx = vx - v0, rely = vy - v1, relz = vz - v2;
        double relspeed = sqrt(relx * relx + rely * rely + relz * relz);

        double ex0 = growthfactor * stfac * pow(mass, 2.0 / 3.0) * nint * relspeed;
        double decel = decelerationfactor * stfac * nint * relspeed / pow(mass, 1.0 / 3.0);

        k_out[p * 7 + 0] = vx;
        k_out[p * 7 + 1] = vy;
        k_out[p * 7 + 2] = vz;
        k_out[p * 7 + 3] = (u0 - vx) / (st * tau_eta) + decel * (v0 - vx);
        k_out[p * 7 + 4] = (u1 - vy) / (st * tau_eta) + decel * (v1 - vy);
        k_out[p * 7 + 5] = (u2 - vz) / (st * tau_eta) + decel * (v2 - vz) - gravity;
        k_out[p * 7 + 6] = ex0 * rhop * n_cs_factor;
        exterp0[p] = ex0;
    }
}

// ------------------------------------------------------------------ G9: deposition
// particles.py's exterp_cosine_scalar: dn/dt gets -fc, fc = exterpmat/(dx*dy*dz)
// (plan section 4), the same 64 taps (4x4x4) as k_droplet_rhs's interpolation,
// recomputed here rather than cached (one cos() pass is cheaper than storing
// 64 weights per particle). Several droplets can land in one cell -- the
// reason particles.py keeps a plain loop over droplets with prange only
// inside (particles.py:32,48) -- so this needs atomicAdd, unlike k_droplet_rhs
// which only ever reads the grid.
__global__ void k_deposit(const double* __restrict__ coord, const double* __restrict__ exterp0,
                          long Nprtcl, real_t* __restrict__ fc, double cellvol)
{
    const double inv_dx = (double)N / L_D;
    GRID_STRIDE(p, Nprtcl) {
        double px = coord[p * 7 + 0], py = coord[p * 7 + 1], pz = coord[p * 7 + 2];
        double gx = px * inv_dx, gy = py * inv_dx, gz = pz * inv_dx;
        double fx = floor(gx), fy = floor(gy), fz = floor(gz);
        double wx[4], wy[4], wz[4];
        cosine_w(gx - fx, wx); cosine_w(gy - fy, wy); cosine_w(gz - fz, wz);
        int ix0 = (int)fx - 1, iy0 = (int)fy - 1, iz0 = (int)fz - 1;
        double amt = exterp0[p] / cellvol;
        for (int a = 0; a < 4; ++a) {
            int xi = ((ix0 + a) % N + N) % N;
            for (int b = 0; b < 4; ++b) {
                int yi = ((iy0 + b) % N + N) % N;
                double wab = wx[a] * wy[b];
                long base = ((long)xi * N + yi) * ZP;      // ZP stride, same trap as above
                for (int c = 0; c < 4; ++c) {
                    int zi = ((iz0 + c) % N + N) % N;
                    atomicAdd(&fc[base + zi], (real_t)(amt * wab * wz[c]));
                }
            }
        }
    }
}

// ------------------------------------------------------------------ G10: classical RK4
// Generic accumulate/trial pair, compact-spectral and real-droplet-state
// flavours, shared by the velocity path and by every field's nk/coord:
//   trial = y + coeff*k            (the RK4 stage input)
//   acc  += coeff*k                (the running y_new = y + h/6(k1+2k2+2k3+k4))
__global__ void k_rk4_trial_c(cplx_t* __restrict__ trial, const cplx_t* __restrict__ y,
                              const cplx_t* __restrict__ k, long n, real_t coeff)
{
    GRID_STRIDE(t, n) {
        trial[t] = C_((real_t)(y[t].x + coeff * k[t].x), (real_t)(y[t].y + coeff * k[t].y));
    }
}

__global__ void k_rk4_accum_c(cplx_t* __restrict__ acc, const cplx_t* __restrict__ k,
                              long n, real_t coeff)
{
    GRID_STRIDE(t, n) { acc[t].x += (real_t)(coeff * k[t].x); acc[t].y += (real_t)(coeff * k[t].y); }
}

__global__ void k_rk4_trial_r(double* __restrict__ trial, const double* __restrict__ y,
                              const double* __restrict__ k, long n, double coeff)
{
    GRID_STRIDE(t, n) { trial[t] = y[t] + coeff * k[t]; }
}

__global__ void k_rk4_accum_r(double* __restrict__ acc, const double* __restrict__ k,
                              long n, double coeff)
{
    GRID_STRIDE(t, n) { acc[t] += coeff * k[t]; }
}

// The same trial construction for the density, which is carried PHYSICALLY:
// p = n_y + coeff*p, where p holds irfft(nk_k) on entry (the stage derivative,
// scattered from the compact nk_k and inverse-transformed, so already in the
// padded layout) and the stage's trial n on exit. ZP-AWARE, like k_clip_accum
// and k_clip_rescale and for the same reason: npts is the LOGICAL point count
// NR_ = N*N*N and every thread maps it through the (N,N,ZP) stride, so the two
// pad reals per line are neither read nor written. A flat walk over NRP would
// pull uninitialised pad memory into the trial state. Safe in place, which is
// the only way it is called: as with k_make_v's rhsuk/vk aliasing, each thread
// reads and writes its own index and no other, so p carries no __restrict__.
__global__ void k_rk4_trial_p(real_t* p, const real_t* __restrict__ n_y,
                              long npts, real_t coeff)
{
    GRID_STRIDE(t, npts) {
        int  z = (int)(t % N);
        long r = t / N;
        int  y = (int)(r % N);
        int  x = (int)(r / N);
        long i = ((long)x * N + y) * ZP + z;
        p[i] = (real_t)((double)n_y[i] + (double)coeff * (double)p[i]);
    }
}

// wraps a droplet field's positions into [0, L) after the stage completes --
// particles.py's send() wraps only the first (coordinate) argument; nothing
// else here reads an unwrapped position, so this is called once per field
// per STEP -- outside the stage loop, after the final k_rk4_accum_r has formed
// the accepted state, exactly where send() would have run.
__global__ void k_wrap_pos(double* __restrict__ coord, long Nprtcl)
{
    GRID_STRIDE(p, Nprtcl) {
        for (int c = 0; c < 3; ++c) {
            double x = fmod(coord[p * 7 + c], L_D);
            if (x < 0.0) x += L_D;
            coord[p * 7 + c] = x;
        }
    }
}

// ------------------------------------------------------------------ G5: density advection pieces
// Physical elementwise product, ZP-strided: c = a*b, used for vn = n_phys*vtemp
// (forced-dns-sm-big.py full_RHS, the `vn[:] = irfft(...)*vtemp` lines). One
// call per vector component -- a is the (broadcast) scalar n, b one component
// of v or vtemp.
__global__ void k_mul_phys(const real_t* __restrict__ a, const real_t* __restrict__ b,
                           real_t* __restrict__ c, long npts)
{
    GRID_STRIDE(t, npts) { c[t] = a[t] * b[t]; }
}

// q += scale * i*(kx*Vx + ky*Vy + kz*Vz), compact, two-pass phase-shift (the
// same averaging k_accum_prod uses): shifted=0 uses f=0.5, shifted=1 uses
// f=conj(phase)/2 -- forced-dns-sm-big.py's `divvnk += 1j*kx*rfft(vn[0],..)*
// conjphase_k*dealias*0.5` (pass 0, phase-shifted input) and the unshifted
// `*dealias*0.5` pass (pass 1). Vx/Vy/Vz are the FULL (N,N,Nf) transform of
// that pass's vn components (d_P-style scratch, reused per component serially
// -- same technique as k_product/k_accum_prod).
__global__ void k_div_accum_c(cplx_t* __restrict__ q, long M,
                              const cplx_t* __restrict__ Vx, const cplx_t* __restrict__ Vy,
                              const cplx_t* __restrict__ Vz, const long* __restrict__ loff,
                              int shifted, const cplx_t* __restrict__ phase)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        double fRe = 0.5, fIm = 0.0;
        if (shifted) {
            cplx_t p = phase[s.kx + s.ky + s.kz + POFF];
            fRe = 0.5 * (double)p.x; fIm = -0.5 * (double)p.y;
        }
        cplx_t vx = Vx[t], vy = Vy[t], vz = Vz[t];
        // i*k*(fRe+i fIm)*v, summed over the three components
        double accR = 0.0, accI = 0.0;
        double karr[3] = { (double)s.kx, (double)s.ky, (double)s.kz };
        cplx_t varr[3] = { vx, vy, vz };
        for (int d = 0; d < 3; ++d) {
            double vr = fRe * varr[d].x - fIm * varr[d].y;
            double vi = fRe * varr[d].y + fIm * varr[d].x;
            // i*k*(vr + i vi) = k*(-vi + i*vr)
            accR += karr[d] * (-vi);
            accI += karr[d] * vr;
        }
        cplx_t o = q[s.c];
        q[s.c] = C_((real_t)(o.x + accR), (real_t)(o.y + accI));
    }
}

// ------------------------------------------------------------------ G10: classical-RK4 velocity RHS
// The dt-free, Bc-free core of k_finalize (forced_ns_ck54_lowmem.cu:398): the
// pressure projection and the (linear, explicit) hyperviscosity term, with no
// 2N recursion -- ku = rhs - grad(p) - nu*(-lap)^lp*u, exactly
// forced-dns-sm-big.py:388-390's `ku`. q holds the same nonlinear accumulator
// k_accum_prod/k_add_force built (rhsuk+fk already summed there); nuodd is
// always 0 here (sm-big has no odd viscosity term).
__global__ void k_velocity_rhs(const cplx_t* __restrict__ q, const cplx_t* __restrict__ U,
                               cplx_t* __restrict__ ku, long M, const long* __restrict__ loff,
                               real_t nu)
{
    GRID_STRIDE(t, NS_) {
        CPt s;
        if (!compact_point(t, loff, s)) continue;
        const long c = s.c;
        cplx_t r0 = q[c], r1 = q[c + M], r2 = q[c + 2 * M];
        cplx_t u0 = U[c], u1 = U[c + M], u2 = U[c + 2 * M];
        double ik = (s.k2 > 0.0) ? rsqrt(s.k2) : 0.0;
        double h0 = s.kx * ik, h1 = s.ky * ik, h2 = s.kz * ik;
        double dre = h0 * r0.x + h1 * r1.x + h2 * r2.x;
        double dim = h0 * r0.y + h1 * r1.y + h2 * r2.y;
        double nk2 = (double)nu * s.k2;
        ku[c]         = C_((real_t)(r0.x - dre * h0 - nk2 * u0.x), (real_t)(r0.y - dim * h0 - nk2 * u0.y));
        ku[c + M]     = C_((real_t)(r1.x - dre * h1 - nk2 * u1.x), (real_t)(r1.y - dim * h1 - nk2 * u1.y));
        ku[c + 2 * M] = C_((real_t)(r2.x - dre * h2 - nk2 * u2.x), (real_t)(r2.y - dim * h2 - nk2 * u2.y));
    }
}

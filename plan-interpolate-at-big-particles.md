# Scope load_series to a single (stb, init) pair

## Context

`load_series` currently loops `stb_s` and `names` internally and returns arrays whose leading axes
are `(kk, nami, ...)`. The cell that writes the json then loops over those same two indices to
read the arrays back out. The array shape mirrors the caller's loops — bookkeeping carried in two
places at once, and it makes a single pair impossible to run on its own: you cannot look at
`0.004 / Qneg` without computing all twenty.

Scoping the function to one pair removes both problems. The `which` axis of length 2 stays inside,
so `all` and `top10` are still accumulated in a single pass over the data.

**Known cost, accepted.** `calc_invariants(load_u(...))` depends only on `t`, so the all-pair
version computed it once per snapshot and shared it across twenty pairs. Per-pair, it moves inside
each call and runs twenty times per snapshot. Rough estimate, not measured: the full sweep goes
from ~15 h to ~70 h, while a single pair drops to ~3.5 h from an all-or-nothing 15 h.

## The change, in `plot-20261005-mod.py`

### `load_series` (currently line 212)

Signature becomes `load_series(stb,init,times,qbins = qbins)`. Every accumulator loses its
`(len(stb_s), len(names))` prefix:

```python
def load_series(stb,init,times,qbins = qbins):
    mass_mean = np.zeros((2,len(times))) #! all and top10
    mass_std = np.zeros((2,len(times)))
    qmean = np.zeros((2,len(times),len(qbins)))
    cnt = [np.zeros((2,len(b)-1)) for b in qbins]
    tot = [np.zeros((2,len(b)-1)) for b in qbins]
    fields = np.zeros((4,N,N,N))
    mask10 = mask_max_growers(stb,init,t = times)
    prtcl = MPI_particles(comm, L, N, len(mask10),sts, stb,0, nu, tf,rhop,M0,3,X,Y,Z, x,y,z)
    prtcl.to_interp(4)
    for i,t in enumerate(times):
        fields[:3] = calc_invariants(load_u(prtcl_path(t,stb,init).parent))
        fields[3] = load_n(datapath(t,sts,stb,init))
        mass,urel,dmass,pos = load_instant(stb,init,t)
        q = interp_quantities(prtcl,pos,fields,urel)
        for w,mask in enumerate((slice(None), mask10)):
            mass_mean[w,i] = mass[mask].mean()
            mass_std[w,i] = mass[mask].std()
            qmean[w,i] = q[mask].mean(axis = 0)
            for v in range(len(qbins)):
                cnt[v][w] += np.histogram(q[mask,v],bins = qbins[v])[0]
                tot[v][w] += np.histogram(q[mask,v],bins = qbins[v],weights = dmass[mask])[0]
    qvals = [0.5*(b[1:] + b[:-1]) for b in qbins]
    prof = np.empty((2,len(qbins)),dtype = object)
    vals = np.empty(prof.shape,dtype = object)
    for w,v in np.ndindex(prof.shape):
        cond = cnt[v][w]>0 #! drop the bins with no droplets in them
        prof[w,v] = tot[v][w][cond]/cnt[v][w][cond]
        vals[w,v] = qvals[v][cond]
    return mass_mean, mass_std, qmean, prof, vals
```

Three things fall out of the narrower scope beyond the shape change:

- The `prtcls` list becomes one `prtcl`.
- `mask_max_growers` has to run first anyway, and `len(mask10)` is the droplet count for this
  pair — so the `MPI_particles` constructor no longer needs `Nprtcl[kk]`, and it uses the count
  actually on disk rather than the nominal one from line 29.
- `load_u` takes `prtcl_path(t,stb,init).parent` instead of `prtcl_path(t,stb_s[0],names[0]).parent`.
  Same directory either way, since the parent is the `time_` folder, but it no longer reaches past
  its own arguments.

`interp_quantities` is untouched.

### The json cell (currently line 245)

The call moves inside the loops, and `nami` is no longer needed:

```python
db = {}
for kk,stb in enumerate(stb_s):
    for init in names:
        mass_mean,mass_std,qmean,prof,vals = load_series(stb,init,times)
        for w,which in enumerate(["all","top10"]):
            key = f"{wg}/{stb:.3f}/{init}/{which}"
            db[key] = {"t":times.tolist(),"mean":mass_mean[w].tolist(),"std":mass_std[w].tolist(),"n":int(Nprtcl[kk]),"qmean":qmean[w].tolist()}
            db[key].update({f"dmass_{qn}":p.tolist() for qn,p in zip(qnames,prof[w])})
            db[key].update({f"vals_{qn}":p.tolist() for qn,p in zip(qnames,vals[w])})
```

`kk` survives only to index `Nprtcl` for the `"n"` field. The json keys and their contents are
unchanged from the current version.

## Verification

Python cannot be run in this sandbox, so these are for the cluster.

1. **Single pair now runs alone** — `load_series(0.004,"Qneg",times[:3])` should return without
   touching any other Stokes number. This is the capability the change is for; check it first.
2. **Same numbers as before.** On a two- or three-snapshot slice, the new `mass_mean[w]` must
   match the old `mass_mean[kk,nami,w]` for the same pair, to round-off. Likewise `qmean`, and the
   `prof`/`vals` pair for at least one quantity.
3. **Particle count.** Confirm `len(mask10)` equals `Nprtcl[kk]` for each pair. If it does not,
   the saved data has a different count from the nominal formula and the `"n"` field in the json
   is the one to trust less.
4. **Bin occupancy.** `cnt[v][0].sum()` should equal `len(times)*len(mask10)`. A shortfall means
   droplets fall outside the hand-set edges for quantity `v` and that range needs widening.
5. **Timing.** Print the wall time around `calc_invariants(load_u(...))` and around
   `load_n` + `load_instant` on a few snapshots. That settles whether the 20x recompute actually
   costs what I estimated, and whether it is worth revisiting.
6. Run the `distiller` agent on the diff. Tell it to edit only through small `Edit` calls and to
   check the file line count afterwards — an earlier pass truncated this file by 313 lines.

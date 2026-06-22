# M3 — unguided sampling evaluation

Unguided ODE sampling from the trained KSE prior, NFE sweep + statistics.

NFE counting verified exact (Euler N=NFE; RK4 4N). Generated samples reproduce
KSE spatial structure (two-point correlation tracks the real negative-correlation
trough at the cell scale) and a centered marginal.

Known gap (logged risk, not a bug): generated std ~0.91 vs real ~1.29 — the pilot
prior (24 trajectories, 0.34M params, 4000 steps) slightly under-shoots energy.
Acceptable for M3/M4/M4b mechanism validation; revisit with a larger/longer prior
before final paper numbers.

| NFE | 2pt_err | marg_dist | gen_std |
|----:|--------:|----------:|--------:|
| 10  | 0.193   | 0.278     | 0.759   |
| 20  | 0.186   | 0.237     | 0.828   |
| 50  | 0.182   | 0.211     | 0.872   |
| 100 | 0.180   | 0.202     | 0.888   |

import numpy as np
import time

#generates complex gaussian noise
def complex_standard_normal(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

#for numerical stability
def ensure_psd_cholesky(G, eps=1e-12, max_tries=6):
    jitter = 0.0
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(G + jitter * np.eye(3, dtype=np.complex128))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    jitter = max(jitter, 1e-6)
    L = np.linalg.cholesky(G + jitter * np.eye(3, dtype=np.complex128))
    return L, jitter

#K: number of time frames, mu_counts: mean photon counts, rng: random generators
def one_measurement_g2_g3(g12, g23, g31, K, mu_counts, rng):
    # coherence matrix
    g13 = np.conjugate(g31)
    G = np.array([
        [1.0 + 0.0j,         g12,               g13],
        [np.conjugate(g12),  1.0 + 0.0j,        g23],
        [np.conjugate(g13),  np.conjugate(g23), 1.0 + 0.0j]
    ], dtype=np.complex128)

    L, jitter = ensure_psd_cholesky(G)

    z = complex_standard_normal((3, K), rng)  #matrix of measurements
    E = L @ z   #E is correlated complex field
    I = np.abs(E)**2  #instensity

    N = rng.poisson(lam=mu_counts * I).astype(np.float64)     #shot noise
    N1, N2, N3 = N[0], N[1], N[2]                            
    m1, m2, m3 = N1.mean(), N2.mean(), N3.mean()

    g2_12 = (N1*N2).mean() / (m1*m2)
    g2_23 = (N2*N3).mean() / (m2*m3)         #correlation results
    g2_31 = (N3*N1).mean() / (m3*m1)
    g3_123 = (N1*N2*N3).mean() / (m1*m2*m3)

    return g2_12, g2_23, g2_31, g3_123, jitter

if __name__ == "__main__":
   # t0 = time.perf_counter()

    inp = "gamma_earth_rotation.npz"
    d = np.load(inp)

    t = d["t"]
    gamma12 = d["gamma12"]
    gamma23 = d["gamma23"]
    gamma31 = d["gamma31"]

    # settings
    step = 1
    K = 200000
    mu_counts = 2000.0

    #average over multiple independent measurements 
    Nreal = 30        
    seed0 = 12345    #needed to get the same results every run

    idx = np.arange(0, len(t), step, dtype=int)
    t_use = t[idx]

    g2_12_mean = np.empty(len(idx))
    g2_23_mean = np.empty(len(idx))
    g2_31_mean = np.empty(len(idx))
    g3_123_mean = np.empty(len(idx))
    jitters_max = np.empty(len(idx))

    # loop timepoints
    for n, k in enumerate(idx):
        s12 = s23 = s31 = s3 = 0.0
        jmax = 0.0

        # loop independent measurements
        for r in range(Nreal):
            rng = np.random.default_rng(seed0 + 1000000*n + r)  # deterministic but independent per (t,r)
            g2_12, g2_23, g2_31, g3_123, jitter = one_measurement_g2_g3(
                gamma12[k], gamma23[k], gamma31[k],
                K=K, mu_counts=mu_counts, rng=rng
            )
            s12 += g2_12
            s23 += g2_23
            s31 += g2_31
            s3  += g3_123
            jmax = max(jmax, jitter)

        g2_12_mean[n] = s12 / Nreal
        g2_23_mean[n] = s23 / Nreal
        g2_31_mean[n] = s31 / Nreal
        g3_123_mean[n] = s3  / Nreal
        jitters_max[n] = jmax

    # derived, after averaging
    ReT = 0.5 * (g3_123_mean - 1.0 - (g2_12_mean-1.0) - (g2_23_mean-1.0) - (g2_31_mean-1.0))
    a12 = np.sqrt(np.maximum(g2_12_mean - 1.0, 0.0))
    a23 = np.sqrt(np.maximum(g2_23_mean - 1.0, 0.0))
    a31 = np.sqrt(np.maximum(g2_31_mean - 1.0, 0.0))
    absT = a12 * a23 * a31
    cosphi = np.full_like(absT, np.nan, dtype=np.float64)
    m = absT > 0
    cosphi[m] = np.clip(ReT[m] / absT[m], -1.0, 1.0)

    out = "mc_hbt_ideal_avg_outputs.npz"
    np.savez(
        out,
        t=t_use, step=step, K=K, mu_counts=mu_counts, Nreal=Nreal,
        g2_12=g2_12_mean, g2_23=g2_23_mean, g2_31=g2_31_mean,
        g3_123=g3_123_mean,
        ReT=ReT, absT=absT, cosphi=cosphi,
        jitters=jitters_max
    )

    t1 = time.perf_counter()
    print(f"Loaded: {inp}")
    print(f"Saved:  {out}")
    print(f"Timepoints: {len(t_use)} (step={step})")
    print(f"K={K}, mu_counts={mu_counts}, Nreal={Nreal}")
    #print(f"Elapsed seconds: {t1 - t0:.3f}")
    print(f"jitter applied: {np.max(jitters_max):.3e}")
    print(f"cosphi finite fraction: {np.isfinite(cosphi).mean():.3f}")

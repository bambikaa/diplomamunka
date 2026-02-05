import numpy as np
import time

def complex_standard_normal(shape, rng):
    """CN(0,1): (N(0,1)+iN(0,1))/sqrt(2)"""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

def ensure_psd_cholesky(G, eps=1e-12, max_tries=6):
    """
    Try Cholesky; if numerical issues occur, add small diagonal jitter.
    """
    jitter = 0.0
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(G + jitter * np.eye(3, dtype=np.complex128))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    # last resort: bigger jitter
    jitter = max(jitter, 1e-6)
    L = np.linalg.cholesky(G + jitter * np.eye(3, dtype=np.complex128))
    return L, jitter

def mc_stats_for_timepoint(g12, g23, g31, K, mu_counts, rng):
    """
    One timepoint MC:
      - Build 3x3 coherence matrix Gamma
      - Generate K samples of complex field
      - Convert to intensities and Poisson photon counts
      - Return g2s, g3, ReT_hat, |T|_hat, cosphi_hat
    """
    # Note: gamma13 = conj(g31) because g31 corresponds to baseline 3-1
    g13 = np.conjugate(g31)

    G = np.array([
        [1.0 + 0.0j,      g12,              g13],
        [np.conjugate(g12), 1.0 + 0.0j,      g23],
        [np.conjugate(g13), np.conjugate(g23), 1.0 + 0.0j]
    ], dtype=np.complex128)

    L, jitter = ensure_psd_cholesky(G)

    # Generate complex fields: shape (3, K)
    z = complex_standard_normal((3, K), rng)
    E = L @ z
    I = np.abs(E)**2  # intensities, shape (3,K)

    # Photon counts (Poisson)
    # mu_counts is mean scaling so that <N> roughly mu_counts (since <I>~1)
    N = rng.poisson(lam=mu_counts * I).astype(np.float64)

    N1, N2, N3 = N[0], N[1], N[2]
    m1, m2, m3 = N1.mean(), N2.mean(), N3.mean()

    # Avoid division by zero if mu_counts too small
    if (m1 <= 0) or (m2 <= 0) or (m3 <= 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, jitter

    g2_12 = (N1 * N2).mean() / (m1 * m2)
    g2_23 = (N2 * N3).mean() / (m2 * m3)
    g2_31 = (N3 * N1).mean() / (m3 * m1)

    g3_123 = (N1 * N2 * N3).mean() / (m1 * m2 * m3)

    # Extract Re(T) from g2, g3 relation for chaotic light
    ReT = 0.5 * (g3_123 - 1.0 - (g2_12 - 1.0) - (g2_23 - 1.0) - (g2_31 - 1.0))

    # Estimate |gamma_ij| from g2-1 (clip for safety)
    a12 = np.sqrt(max(g2_12 - 1.0, 0.0))
    a23 = np.sqrt(max(g2_23 - 1.0, 0.0))
    a31 = np.sqrt(max(g2_31 - 1.0, 0.0))
    absT = a12 * a23 * a31

    # cos(phi_cl) = Re(T)/|T|
    cosphi = np.nan
    if absT > 0:
        cosphi = np.clip(ReT / absT, -1.0, 1.0)

    return g2_12, g2_23, g2_31, g3_123, ReT, absT, cosphi, jitter

def apply_timebin_scaling(g2_minus1, ReT, M):
    """
    Long time bin compared to coherence time:
      - (g2 - 1) scales ~ 1/M
      - Re(T) term (connected 3rd-order) scales ~ 1/M^2
    We return scaled versions (g2, ReT).
    """
    g2_scaled = 1.0 + (g2_minus1 / M)
    ReT_scaled = ReT / (M**2)
    return g2_scaled, ReT_scaled

if __name__ == "__main__":
    t0 = time.perf_counter()

    inp = "gamma_earth_rotation.npz"
    d = np.load(inp)

    t = d["t"]
    gamma12 = d["gamma12"]
    gamma23 = d["gamma23"]
    gamma31 = d["gamma31"]

    # -------------------------
    # User-tunable MC settings
    # -------------------------
    # Subsample times to make it faster at first (e.g., every 5 minutes)
    step = 5  # keep every 5th sample; if your gamma file is 1-min spacing, step=5 -> 5-min spacing

    # Number of bins/samples per timepoint
    K = 20000  # start: 2e4; increase for smoother g3 / ReT

    # Mean photon counts per detector per bin (controls shot noise)
    mu_counts = 200.0  # try 50..500; too small -> noisy / division issues

    # Coherence averaging factor M = dt / tau_c  (realistic is huge)
    # For your numbers: M ~ 1.7e4  (see earlier calc)
    # We'll do "physics scaling" instead of explicit M loop:
    M = 1

    seed = 12345
    rng = np.random.default_rng(seed)

    # -------------------------
    # Allocate outputs
    # -------------------------
    idx = np.arange(0, len(t), step, dtype=int)
    t_use = t[idx]

    g2_12 = np.empty(len(idx))
    g2_23 = np.empty(len(idx))
    g2_31 = np.empty(len(idx))
    g3_123 = np.empty(len(idx))
    ReT = np.empty(len(idx))
    absT = np.empty(len(idx))
    cosphi = np.empty(len(idx))
    jitters = np.empty(len(idx))

    # -------------------------
    # MC loop over timepoints
    # -------------------------
    for n, k in enumerate(idx):
        out = mc_stats_for_timepoint(
            gamma12[k], gamma23[k], gamma31[k],
            K=K, mu_counts=mu_counts, rng=rng
        )
        g2_12[n], g2_23[n], g2_31[n], g3_123[n], ReT[n], absT[n], cosphi[n], jitters[n] = out

        # Apply realistic time-bin suppression (optional but recommended for realism)
        # If you want "idealized" (no suppression), comment this block out.
        g2_12[n], ReT_12 = apply_timebin_scaling(g2_12[n] - 1.0, ReT[n], M)  # ReT_12 not used
        g2_23[n], ReT_23 = apply_timebin_scaling(g2_23[n] - 1.0, ReT[n], M)
        g2_31[n], ReT_31 = apply_timebin_scaling(g2_31[n] - 1.0, ReT[n], M)

        # ReT scales as 1/M^2 (use one scaling; the above returns same ReT_scaled each time)
        ReT[n] = ReT[n] / (M**2)

        # Recompute absT and cosphi consistently from scaled g2s
        a12 = np.sqrt(max(g2_12[n] - 1.0, 0.0))
        a23 = np.sqrt(max(g2_23[n] - 1.0, 0.0))
        a31 = np.sqrt(max(g2_31[n] - 1.0, 0.0))
        absT[n] = a12 * a23 * a31
        cosphi[n] = np.nan if absT[n] == 0 else np.clip(ReT[n] / absT[n], -1.0, 1.0)

    # -------------------------
    # Save outputs
    # -------------------------
    out_file = "mc_hbt_outputs.npz"
    np.savez(
        out_file,
        t=t_use,
        step=step,
        K=K,
        mu_counts=mu_counts,
        M=M,
        g2_12=g2_12,
        g2_23=g2_23,
        g2_31=g2_31,
        g3_123=g3_123,
        ReT=ReT,
        absT=absT,
        cosphi=cosphi,
        jitters=jitters,
        seed=seed
    )

    t1 = time.perf_counter()
    print(f"Loaded: {inp}")
    print(f"Saved:  {out_file}")
    print(f"Timepoints used: {len(t_use)} (step={step})")
    print(f"K bins per timepoint: {K}, mu_counts={mu_counts}, M(note)={M}")
    print(f"Elapsed seconds: {t1 - t0:.3f}")
    print(f"Max Cholesky jitter applied: {np.max(jitters):.3e}")

import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# Helpers
# =========================
def seconds_to_hours(t):
    return (t - t[0]) / 3600.0

def wrap_angle_rad(phi):
    return float(np.arctan2(np.sin(phi), np.cos(phi)))

def wrap_deg(phi_deg):
    return ((phi_deg + 180.0) % 360.0) - 180.0

def angdist_deg(a, b):
    return abs(wrap_deg(a - b))

#creates a limb-darkened disk image with a gaussian spot
#N: image size (NxN), fov: field of view [rad], theta: star diameter [rad]
#u_ld: limb-darkening coefficient, spot_r: spot radial distance from center [rad]
##spot_phi: spot azimuthal angle [rad], sigma_s: spot size [rad], A: spot amplitude

#fov needed because otherwise fft creates false spectrum

def build_image_ld_spot(N, fov, theta, u_ld, spot_r, spot_phi, sigma_s, A):
    x = (np.arange(N) - (N - 1) / 2) * (fov / N)
    y = (np.arange(N) - (N - 1) / 2) * (fov / N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Rxy = np.sqrt(X**2 + Y**2)

    dx = fov / N
    dy = fov / N
    R_disk = theta / 2.0
    inside = Rxy <= R_disk

    # limb-darkened disk
    disk = np.zeros_like(Rxy, dtype=np.float64)
    rho = (Rxy[inside] / R_disk)
    mu = np.sqrt(np.maximum(1.0 - rho**2, 0.0))
    disk[inside] = 1.0 - u_ld * (1.0 - mu)

    # gaussian spot
    x_s = spot_r * np.cos(spot_phi)
    y_s = spot_r * np.sin(spot_phi)
    spot = A * np.exp(-((X - x_s) ** 2 + (Y - y_s) ** 2) / (2.0 * sigma_s ** 2))
    spot = spot * inside

    I = disk + spot
    I[I < 0] = 0.0

    # flux normalize
    norm = I.sum() * dx * dy
   # if norm <= 0:
   #     raise ValueError("Image normalization failed.")
    I = I/norm
    return I, dx, dy

#calculates visibility from intensity image

def fft_visibility(I, dx, dy):
    N = I.shape[0]
    u_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    v_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dy))
    V = np.fft.fftshift(np.fft.fft2(I)) * dx * dy
    V /= V[N // 2, N // 2]   #normalize V(0,0)=1
    return u_grid, v_grid, V


#needed beacause uv points are not on grid
#F: visibility array, U,V: uv grids, u0,v0: desired uv point
def bilinear_sample_complex(U, V, F, u0, v0):
    if (u0 < U[0]) or (u0 > U[-1]) or (v0 < V[0]) or (v0 > V[-1]):   #limit check
        return 0.0 + 0.0j

    #find grid square
    iu = np.searchsorted(U, u0) - 1
    iv = np.searchsorted(V, v0) - 1
    iu = np.clip(iu, 0, len(U) - 2)
    iv = np.clip(iv, 0, len(V) - 2)

    u1, u2 = U[iu], U[iu + 1]
    v1, v2 = V[iv], V[iv + 1]
    #ratios for interpolation
    tu = 0.0 if u2 == u1 else (u0 - u1) / (u2 - u1)
    tv = 0.0 if v2 == v1 else (v0 - v1) / (v2 - v1)

    #visibility values at corners
    f11 = F[iv, iu]
    f21 = F[iv, iu + 1]
    f12 = F[iv + 1, iu]
    f22 = F[iv + 1, iu + 1]

    #interpolated visibility values
    f1 = (1 - tu) * f11 + tu * f21
    f2 = (1 - tu) * f12 + tu * f22
    return (1 - tv) * f1 + tv * f2

#correlation model
def model_g2_g3(A, spot_phi, fixed, uv):
    """
    Forward model: given (A, phi) -> predicted g2_12,g2_23,g2_31,g3_123 time series
    """
    N, fov, theta, u_ld, spot_r, sigma_s = fixed
    t, u12, v12, u23, v23, u31, v31 = uv

    I, dx, dy = build_image_ld_spot(N, fov, theta, u_ld, spot_r, spot_phi, sigma_s, A)
    U, V, Vis = fft_visibility(I, dx, dy)

    g12 = np.array([bilinear_sample_complex(U, V, Vis, u12[k], v12[k]) for k in range(len(t))])
    g23 = np.array([bilinear_sample_complex(U, V, Vis, u23[k], v23[k]) for k in range(len(t))])
    g31 = np.array([bilinear_sample_complex(U, V, Vis, u31[k], v31[k]) for k in range(len(t))])

    a12 = np.abs(g12) ** 2
    a23 = np.abs(g23) ** 2
    a31 = np.abs(g31) ** 2

    g2_12 = 1.0 + a12
    g2_23 = 1.0 + a23
    g2_31 = 1.0 + a31

    g3 = 1.0 + a12 + a23 + a31 + 2.0 * np.real(g12 * g23 * g31)
    return g2_12, g2_23, g2_31, g3

# =========================
# Chi2 grid search
# =========================
def compute_weights(d12, d23, d31, d3):
    eps = 1e-30
    s2 = np.sqrt(np.mean((d12-1.0)**2 + (d23-1.0)**2 + (d31-1.0)**2) / 3.0) + eps
    s3 = np.sqrt(np.mean((d3-1.0)**2)) + eps
    w2 = 1.0 / (s2**2)
    w3 = 1.0 / (s3**2)
    return w2, w3, s2, s3

def chi2_for_params(A, phi, fixed, uv_fit, d12, d23, d31, d3, w2, w3):
    m12, m23, m31, m3 = model_g2_g3(A, phi, fixed, uv_fit)
    r2 = ((d12 - m12)**2 + (d23 - m23)**2 + (d31 - m31)**2).mean()
    r3 = ((d3  - m3 )**2).mean()
    return w2 * r2 + w3 * r3

def coarse_grid_search(A_grid, phi_grid, fixed, uv_fit, d12, d23, d31, d3, w2, w3):
    chi = np.empty((len(A_grid), len(phi_grid)), dtype=np.float64)
    for i, A in enumerate(A_grid):
        for j, phi in enumerate(phi_grid):
            chi[i, j] = chi2_for_params(A, phi, fixed, uv_fit, d12, d23, d31, d3, w2, w3)
    i0, j0 = np.unravel_index(np.argmin(chi), chi.shape)
    return chi, i0, j0

# =========================
# Valley finding on chi2 grid
# =========================
def local_minima_2d(chi, tol=0.0):
    """
    Local minima vs 8-neighborhood.
    periodic boundary in phi; non-periodic in A.
    """
    nA, nP = chi.shape
    mins = []
    for i in range(nA):
        for j in range(nP):
            c = chi[i, j]
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = i + di
                    if ii < 0 or ii >= nA:
                        continue
                    jj = (j + dj) % nP
                    if c > chi[ii, jj] + tol:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                mins.append((i, j))
    return mins

def dedupe_minima(min_list, A_grid, phi_grid_deg, chi, dA=0.02, dphi_deg=10.0):
    """
    Cluster nearby minima and keep best per cluster.
    """
    mins_sorted = sorted(min_list, key=lambda ij: chi[ij[0], ij[1]])
    kept = []
    used = np.zeros(len(mins_sorted), dtype=bool)

    for idx, (i, j) in enumerate(mins_sorted):
        if used[idx]:
            continue
        A0 = A_grid[i]
        p0 = phi_grid_deg[j]
        # mark cluster
        for k in range(idx, len(mins_sorted)):
            if used[k]:
                continue
            ii, jj = mins_sorted[k]
            if abs(A_grid[ii] - A0) <= dA and angdist_deg(phi_grid_deg[jj], p0) <= dphi_deg:
                used[k] = True
        kept.append((i, j))
    return kept

def refine_around_valley(A0, phi0, fixed, uv_fit, d12, d23, d31, d3, w2, w3,
                         A_halfwidth=0.06, A_steps=61, phi_halfwidth_deg=12, phi_step_deg=1):
    """
    Local refinement grid around (A0,phi0).
    """
    A_grid = np.clip(A0 + np.linspace(-A_halfwidth, A_halfwidth, A_steps), 0.0, None)
    phi_grid = np.array([wrap_angle_rad(phi0 + np.deg2rad(k))
                         for k in np.arange(-phi_halfwidth_deg, phi_halfwidth_deg + 1, phi_step_deg)],
                        dtype=np.float64)

    chi = np.empty((len(A_grid), len(phi_grid)), dtype=np.float64)
    for i, A in enumerate(A_grid):
        for j, phi in enumerate(phi_grid):
            chi[i, j] = chi2_for_params(A, phi, fixed, uv_fit, d12, d23, d31, d3, w2, w3)

    i0, j0 = np.unravel_index(np.argmin(chi), chi.shape)
    return A_grid, phi_grid, chi, float(A_grid[i0]), float(phi_grid[j0]), float(chi[i0, j0])

# =========================
# Main script
# =========================
if __name__ == "__main__":
    t_start = time.perf_counter()

    # ---------- Load averaged MC data ----------
    inp = "mc_hbt_ideal_avg_outputs.npz"
    d = np.load(inp)

    t = d["t"]
    th = seconds_to_hours(t)

    g2_12 = d["g2_12"]
    g2_23 = d["g2_23"]
    g2_31 = d["g2_31"]
    g3_123 = d["g3_123"]

    ReT = d["ReT"]
    absT = d["absT"]
    cosphi = d["cosphi"]

    K = int(d["K"])
    mu_counts = float(d["mu_counts"])
    step = int(d["step"])
    Nreal = int(d.get("Nreal", -1))

    print(f"Loaded: {inp}")
    print(f"Points: {len(t)} (step={step}), K={K}, mu_counts={mu_counts}, Nreal={Nreal}")
    finite = np.isfinite(cosphi).mean()
    print(f"cosphi finite fraction: {finite:.3f}")

    # ---------- Load UV tracks ----------
    g = np.load("gamma_earth_rotation.npz")
    u12, v12 = g["u12"][:len(t)], g["v12"][:len(t)]
    u23, v23 = g["u23"][:len(t)], g["v23"][:len(t)]
    u31, v31 = g["u31"][:len(t)], g["v31"][:len(t)]
    full_uv = (t, u12, v12, u23, v23, u31, v31)

    # ---------- Load fixed source params ----------
    src = np.load("source_image.npz", allow_pickle=True)
    N = int(src["N"])
    fov = float(src["fov"])
    theta = float(src["theta"])
    u_ld = float(src["u_ld"])

    x_s_true, y_s_true, sigma_s_true, A_true = src["spot_params"]
    spot_r_true = float(np.sqrt(x_s_true**2 + y_s_true**2))
    phi_true = float(np.arctan2(y_s_true, x_s_true))
    sigma_s = float(sigma_s_true)
    fixed = (N, fov, theta, u_ld, spot_r_true, sigma_s)

    print("\nTruth (reference):")
    print(f"  A_true   = {float(A_true):.4f}")
    print(f"  phi_true = {np.rad2deg(float(phi_true)):.2f} deg")

    # ---------- Thinning for speed ----------
    thin = 1  # set 2 or 3 if slow
    sl = slice(None, None, thin)
    t_fit = t[sl]
    uv_fit = (t_fit, u12[sl], v12[sl], u23[sl], v23[sl], u31[sl], v31[sl])

    d12 = g2_12[sl]
    d23 = g2_23[sl]
    d31 = g2_31[sl]
    d3 = g3_123[sl]

    w2, w3, s2, s3 = compute_weights(d12, d23, d31, d3)
    print("\nWeight scales:")
    print(f"  s2(RMS g2-1) = {s2:.3e}, s3(RMS g3-1) = {s3:.3e}")

    # ---------- COARSE reconstruction ----------
    A_grid1 = np.linspace(0.0, 0.4, 41)                 # 0.01
    phi_grid1 = np.deg2rad(np.arange(-180, 181, 5))     # 5 deg

    print("\nRunning coarse grid search...")
    chi_coarse, i_best, j_best = coarse_grid_search(
        A_grid1, phi_grid1, fixed, uv_fit, d12, d23, d31, d3, w2, w3
    )
    A_best0 = float(A_grid1[i_best])
    phi_best0 = float(phi_grid1[j_best])
    chi_best0 = float(chi_coarse[i_best, j_best])

    print("Coarse best:")
    print(f"  A0   = {A_best0:.4f}")
    print(f"  phi0 = {np.rad2deg(phi_best0):.2f} deg")
    print(f"  chi2 = {chi_best0:.6g}")

    # ---------- Find ALL valleys on coarse grid ----------
    chi_min = float(np.min(chi_coarse))
    chi_max = float(np.max(chi_coarse))
    tol = 0.001 * (chi_max - chi_min)  # plateau tolerance

    mins = local_minima_2d(chi_coarse, tol=tol)
    phi_grid1_deg = np.rad2deg(phi_grid1)

    valleys = dedupe_minima(mins, A_grid1, phi_grid1_deg, chi_coarse, dA=0.02, dphi_deg=10.0)
    valleys = sorted(valleys, key=lambda ij: chi_coarse[ij[0], ij[1]])

    print(f"\nLocal-min pixels: {len(mins)}  -> clustered valleys: {len(valleys)}")
    topM = min(10, len(valleys))

    # ---------- Refine each valley locally ----------
    refined = []
    print("\nRefining top valleys...")
    for r in range(topM):
        i, j = valleys[r]
        A0 = float(A_grid1[i])
        p0 = float(phi_grid1[j])

        A_loc, phi_loc, chi_loc, Ab, pb, cb = refine_around_valley(
            A0, p0, fixed, uv_fit, d12, d23, d31, d3, w2, w3,
            A_halfwidth=0.06, A_steps=61, phi_halfwidth_deg=12, phi_step_deg=1
        )
        refined.append((cb, Ab, pb, A0, p0))
        print(f"  valley {r+1:2d}: coarse(A={A0:.3f},phi={np.rad2deg(p0):7.2f}) "
              f"-> refined(A={Ab:.4f},phi={np.rad2deg(pb):7.2f}), chi2={cb:.6g}")

    refined = sorted(refined, key=lambda x: x[0])
    chi_best = refined[0][0]
    A_best = refined[0][1]
    phi_best = refined[0][2]

    # ---------- Compare truth chi2 on coarse grid ----------
    i_truth = int(np.argmin(np.abs(A_grid1 - float(A_true))))
    j_truth = int(np.argmin(np.abs(wrap_deg(phi_grid1_deg - np.rad2deg(float(phi_true))))))
    chi_truth = float(chi_coarse[i_truth, j_truth])

    print("\nFinal (best refined valley):")
    print(f"  A_best   = {A_best:.4f}")
    print(f"  phi_best = {np.rad2deg(phi_best):.2f} deg")
    print(f"  chi2 best  = {chi_best:.6g}")
    print(f"  chi2 truth = {chi_truth:.6g}")
    print(f"  ratio truth/best = {chi_truth/chi_best:.6f}")

    # ---------- Save everything ----------
    np.savez(
        "recon_and_valleys_outputs.npz",
        A_grid1=A_grid1, phi_grid1=phi_grid1, chi_coarse=chi_coarse,
        refined=np.array(refined, dtype=np.float64),
        A_best=A_best, phi_best=phi_best, chi_best=chi_best,
        A_true=float(A_true), phi_true=float(phi_true),
        thin=thin, w2=w2, w3=w3, s2=s2, s3=s3
    )
    print("Saved: recon_and_valleys_outputs.npz")

    # =========================
    # PLOTS
    # =========================

    # --- Plot g2-1 ---
    #plt.figure(figsize=(9, 5))
    #plt.plot(th, g2_12 - 1.0, label=r"$g^{(2)}_{12}-1$")
    #plt.plot(th, g2_23 - 1.0, label=r"$g^{(2)}_{23}-1$")
    #plt.plot(th, g2_31 - 1.0, label=r"$g^{(2)}_{31}-1$")
    #plt.xlabel("Time [hours]")
    #plt.ylabel(r"$g^{(2)}-1$")
    #plt.title("Averaged: Second-order HBT contrast vs time")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # --- Plot g3-1 ---
    #plt.figure(figsize=(9, 5))
    #plt.plot(th, g3_123 - 1.0, label=r"$g^{(3)}_{123}-1$")
    #plt.xlabel("Time [hours]")
    #plt.ylabel(r"$g^{(3)}-1$")
    #plt.title("Averaged: Third-order correlation excess vs time")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # --- Plot cos(phi_cl) from data ---
    #plt.figure(figsize=(9, 5))
    #plt.plot(th, cosphi, marker="o", markersize=2, linestyle="none", label=r"$\cos(\Phi_{\rm cl})$")
    #plt.xlabel("Time [hours]")
    #plt.ylabel(r"$\cos(\Phi_{\rm cl})$")
    #plt.ylim(-1.05, 1.05)
    #plt.title("Averaged: closure information proxy from data")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # --- Plot chi2 heatmap and mark valleys ---
    plt.figure(figsize=(10, 5))
    extent = [phi_grid1_deg[0], phi_grid1_deg[-1], A_grid1[0], A_grid1[-1]]
    plt.imshow(chi_coarse, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="scaled chi2")

    # mark refined valleys
    for r in range(topM):
        cb, Ab, pb, A0, p0 = refined[r]
        plt.scatter([wrap_deg(np.rad2deg(pb))], [Ab], marker="x", s=90)

    # mark best and truth
    plt.scatter([wrap_deg(np.rad2deg(phi_best))], [A_best], marker="*", s=180, label="best (refined)")
    plt.scatter([wrap_deg(np.rad2deg(float(phi_true)))], [float(A_true)],
                facecolors="none", edgecolors="white", s=150, linewidths=2, label="truth")

    plt.xlabel("phi [deg]")
    plt.ylabel("A")
    plt.title("Chi2(A,phi) coarse grid with refined valley minima")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Overlay: data vs model for best + truth ---
    m12_b, m23_b, m31_b, m3_b = model_g2_g3(A_best, phi_best, fixed, full_uv)
    m12_t, m23_t, m31_t, m3_t = model_g2_g3(float(A_true), float(phi_true), fixed, full_uv)

    plt.figure(figsize=(10, 5))
    plt.plot(th, g3_123, label="data g3")
    plt.plot(th, m3_b, "--", label="model g3 (best)")
    plt.plot(th, m3_t, ":", label="model g3 (truth)")
    plt.xlabel("Time [hours]")
    plt.ylabel("g3")
    plt.title("Data vs model: g3")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(th, g2_12, label="data g2_12")
    plt.plot(th, m12_b, "--", label="model g2_12 (best)")
    plt.plot(th, m12_t, ":", label="model g2_12 (truth)")
    plt.xlabel("Time [hours]")
    plt.ylabel("g2_12")
    plt.title("Data vs model: g2_12")
    plt.legend()
    plt.tight_layout()
    plt.show()

    t_end = time.perf_counter()
    print(f"\nElapsed seconds: {t_end - t_start:.3f}")

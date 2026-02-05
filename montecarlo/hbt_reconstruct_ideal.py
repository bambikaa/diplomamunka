import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------
# Image + FFT + sampling
# -------------------------
def build_image_ld_spot(N, fov, theta, u_ld, spot_r, spot_phi, sigma_s, A):
    x = (np.arange(N) - (N - 1) / 2) * (fov / N)
    y = (np.arange(N) - (N - 1) / 2) * (fov / N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Rxy = np.sqrt(X**2 + Y**2)

    dx = fov / N
    dy = fov / N

    R_disk = theta / 2.0
    inside = Rxy <= R_disk

    disk = np.zeros_like(Rxy, dtype=np.float64)
    rho = (Rxy[inside] / R_disk)
    mu = np.sqrt(1.0 - rho**2)
    disk[inside] = 1.0 - u_ld * (1.0 - mu)

    x_s = spot_r * np.cos(spot_phi)
    y_s = spot_r * np.sin(spot_phi)
    spot = A * np.exp(-((X - x_s) ** 2 + (Y - y_s) ** 2) / (2.0 * sigma_s ** 2))
    spot *= inside

    I = disk + spot
    I[I < 0] = 0.0

    norm = I.sum() * dx * dy
    if norm <= 0:
        raise ValueError("Image normalization failed.")
    I /= norm
    return I, dx, dy

def fft_visibility(I, dx, dy):
    N = I.shape[0]
    u_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    v_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dy))
    V = np.fft.fftshift(np.fft.fft2(I)) * dx * dy
    V /= V[N // 2, N // 2]
    return u_grid, v_grid, V

def bilinear_sample_complex(U, V, F, u0, v0):
    if (u0 < U[0]) or (u0 > U[-1]) or (v0 < V[0]) or (v0 > V[-1]):
        return 0.0 + 0.0j
    iu = np.searchsorted(U, u0) - 1
    iv = np.searchsorted(V, v0) - 1
    iu = np.clip(iu, 0, len(U) - 2)
    iv = np.clip(iv, 0, len(V) - 2)

    u1, u2 = U[iu], U[iu + 1]
    v1, v2 = V[iv], V[iv + 1]
    tu = 0.0 if u2 == u1 else (u0 - u1) / (u2 - u1)
    tv = 0.0 if v2 == v1 else (v0 - v1) / (v2 - v1)

    f11 = F[iv, iu]
    f21 = F[iv, iu + 1]
    f12 = F[iv + 1, iu]
    f22 = F[iv + 1, iu + 1]

    f1 = (1 - tu) * f11 + tu * f21
    f2 = (1 - tu) * f12 + tu * f22
    return (1 - tv) * f1 + tv * f2

def wrap_angle(phi):
    return float(np.arctan2(np.sin(phi), np.cos(phi)))

def model_g2_g3(A, spot_phi, fixed, uv):
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

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    t0 = time.perf_counter()

    # Averaged MC data
    data = np.load("mc_hbt_ideal_avg_outputs.npz")
    t = data["t"]
    d_g2_12 = data["g2_12"]
    d_g2_23 = data["g2_23"]
    d_g2_31 = data["g2_31"]
    d_g3 = data["g3_123"]
    Nreal = int(data["Nreal"])
    K = int(data["K"])
    mu_counts = float(data["mu_counts"])

    # UV tracks (align by length)
    g = np.load("gamma_earth_rotation.npz")
    u12, v12 = g["u12"][:len(t)], g["v12"][:len(t)]
    u23, v23 = g["u23"][:len(t)], g["v23"][:len(t)]
    u31, v31 = g["u31"][:len(t)], g["v31"][:len(t)]
    uv = (t, u12, v12, u23, v23, u31, v31)

    # Fixed params from source
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

    print("Averaged dataset:")
    print(f"  Nreal={Nreal}, K={K}, mu_counts={mu_counts}")
    print("Truth (for reference):")
    print(f"  A_true   = {A_true:.4f}")
    print(f"  phi_true = {np.rad2deg(phi_true):.2f} deg")

    # Optional thinning for speed in the grid search
    thin = 1
    sl = slice(None, None, thin)
    t_fit = t[sl]
    uv_fit = (t_fit, u12[sl], v12[sl], u23[sl], v23[sl], u31[sl], v31[sl])
    d12 = d_g2_12[sl]; d23 = d_g2_23[sl]; d31 = d_g2_31[sl]; d3 = d_g3[sl]

    # Empirical scaling of residuals
    eps = 1e-30
    s2 = np.sqrt(np.mean((d12-1.0)**2 + (d23-1.0)**2 + (d31-1.0)**2) / 3.0) + eps
    s3 = np.sqrt(np.mean((d3-1.0)**2)) + eps
    w2 = 1.0 / (s2**2)
    w3 = 1.0 / (s3**2)

    # -------------------------
    # Coarse grid
    # -------------------------
    A_grid1 = np.linspace(0.0, 0.4, 41)             # 0.01
    phi_grid1 = np.deg2rad(np.arange(-180, 181, 5)) # 5 deg

    chi = np.empty((len(A_grid1), len(phi_grid1)), dtype=np.float64)

    for i, A in enumerate(A_grid1):
        for j, phi in enumerate(phi_grid1):
            m12, m23, m31, m3 = model_g2_g3(A, phi, fixed, uv_fit)
            r2 = ((d12 - m12)**2 + (d23 - m23)**2 + (d31 - m31)**2).mean()
            r3 = ((d3  - m3 )**2).mean()
            chi[i, j] = w2 * r2 + w3 * r3

    i0, j0 = np.unravel_index(np.argmin(chi), chi.shape)
    A0 = float(A_grid1[i0])
    phi0 = float(phi_grid1[j0])

    # -------------------------
    # Fine grid around minimum
    # -------------------------
    A_grid2 = np.clip(A0 + np.linspace(-0.06, 0.06, 61), 0.0, 0.4)  # ~0.002 step
    phi_grid2 = np.array([wrap_angle(phi0 + np.deg2rad(k)) for k in range(-10, 11)], dtype=np.float64)  # 1 deg

    chi2 = np.empty((len(A_grid2), len(phi_grid2)), dtype=np.float64)
    for i, A in enumerate(A_grid2):
        for j, phi in enumerate(phi_grid2):
            m12, m23, m31, m3 = model_g2_g3(A, phi, fixed, uv_fit)
            r2 = ((d12 - m12)**2 + (d23 - m23)**2 + (d31 - m31)**2).mean()
            r3 = ((d3  - m3 )**2).mean()
            chi2[i, j] = w2 * r2 + w3 * r3

    i1, j1 = np.unravel_index(np.argmin(chi2), chi2.shape)
    A_best = float(A_grid2[i1])
    phi_best = wrap_angle(float(phi_grid2[j1]))

    print("\nReconstruction from averaged g2+g3:")
    print(f"  A_best   = {A_best:.4f}")
    print(f"  phi_best = {np.rad2deg(phi_best):.2f} deg")

    np.savez("recon_params_from_avg.npz",
             A_best=A_best, phi_best=phi_best,
             A_true=float(A_true), phi_true=float(phi_true),
             A_grid1=A_grid1, phi_grid1=phi_grid1, chi_coarse=chi,
             A_grid2=A_grid2, phi_grid2=phi_grid2, chi_fine=chi2,
             Nreal=Nreal, K=K, mu_counts=mu_counts, thin=thin)
    print("Saved: recon_params_from_avg.npz")

    # Plot chi2 (coarse)
    plt.figure(figsize=(9,5))
    extent = [np.rad2deg(phi_grid1[0]), np.rad2deg(phi_grid1[-1]), A_grid1[0], A_grid1[-1]]
    plt.imshow(chi, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="scaled chi2")
    plt.scatter([np.rad2deg(phi_best)], [A_best], marker="x", s=120, label="best-fit")
    plt.scatter([np.rad2deg(phi_true)], [A_true], marker="o", facecolors="none", s=120, label="truth")
    plt.xlabel("phi [deg]")
    plt.ylabel("A")
    plt.title("Reconstruction chi2(A,phi) from averaged data")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compare curves (full)
    full_uv = (t, u12, v12, u23, v23, u31, v31)
    m12, m23, m31, m3 = model_g2_g3(A_best, phi_best, fixed, full_uv)
    th = (t - t[0]) / 3600.0

    plt.figure(figsize=(9,5))
    plt.plot(th, d_g2_12, label="data g2_12")
    plt.plot(th, m12, "--", label="model g2_12")
    plt.plot(th, d_g2_23, label="data g2_23")
    plt.plot(th, m23, "--", label="model g2_23")
    plt.plot(th, d_g2_31, label="data g2_31")
    plt.plot(th, m31, "--", label="model g2_31")
    plt.xlabel("Time [hours]")
    plt.ylabel("g2")
    plt.title("Averaged data vs model (g2)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9,5))
    plt.plot(th, d_g3, label="data g3")
    plt.plot(th, m3, "--", label="model g3")
    plt.xlabel("Time [hours]")
    plt.ylabel("g3")
    plt.title("Averaged data vs model (g3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    t1 = time.perf_counter()
    print(f"\nElapsed seconds: {t1 - t0:.3f}")

i_truth = np.argmin(np.abs(A_grid1 - A_true))
j_truth = np.argmin(np.abs(np.rad2deg(phi_grid1) - np.rad2deg(phi_true)))
chi_truth = chi[i_truth, j_truth]
chi_best = np.min(chi)

print(f"chi2 best  = {chi_best:.6g}")
print(f"chi2 truth = {chi_truth:.6g}")
print(f"ratio truth/best = {chi_truth/chi_best:.3f}")

# hbt_funcs.py
import numpy as np

def seconds_to_hours(t):
    return (t - t[0]) / 3600.0

def wrap_angle_rad(phi):
    return float(np.arctan2(np.sin(phi), np.cos(phi)))

def wrap_deg(phi_deg):
    return ((phi_deg + 180.0) % 360.0) - 180.0

def angdist_deg(a, b):
    return abs(wrap_deg(a - b))

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
    spot *= inside

    I = disk + spot
    I[I < 0] = 0.0

    # flux normalize
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

def model_g2_g3(A, spot_phi, fixed, uv):
    #Forward model: given (A, phi) -> predicted g2_12,g2_23,g2_31,g3_123 time series

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

def local_minima_2d(chi, tol=0.0):
    #Local minima vs 8-neighborhood.
    #periodic boundary in phi; non-periodic in A.
    
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
    #Cluster nearby minima and keep best per cluster.
    mins_sorted = sorted(min_list, key=lambda ij: chi[ij[0], ij[1]])
    kept = []
    used = np.zeros(len(mins_sorted), dtype=bool)

    for idx, (i, j) in enumerate(mins_sorted):
        if used[idx]:
            continue
        A0 = A_grid[i]
        p0 = phi_grid_deg[j]
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

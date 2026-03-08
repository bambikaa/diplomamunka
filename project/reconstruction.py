# run_recon.py
import numpy as np
import matplotlib.pyplot as plt
import time

from hbt_funcs import (
    seconds_to_hours,
    wrap_deg,
    model_g2_g3,
    compute_weights,
    coarse_grid_search,
    local_minima_2d,
    dedupe_minima,
    refine_around_valley,
)

def main():
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

    g = np.load("gamma_earth_rotation.npz")
    u12, v12 = g["u12"][:len(t)], g["v12"][:len(t)]
    u23, v23 = g["u23"][:len(t)], g["v23"][:len(t)]
    u31, v31 = g["u31"][:len(t)], g["v31"][:len(t)]
    full_uv = (t, u12, v12, u23, v23, u31, v31)

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

    thin = 1  
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

    #COARSE reconstruction
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

    #Find all valleys on coarse grid
    chi_min = float(np.min(chi_coarse))
    chi_max = float(np.max(chi_coarse))
    tol = 0.001 * (chi_max - chi_min)  # plateau tolerance

    mins = local_minima_2d(chi_coarse, tol=tol)
    phi_grid1_deg = np.rad2deg(phi_grid1)

    valleys = dedupe_minima(mins, A_grid1, phi_grid1_deg, chi_coarse, dA=0.02, dphi_deg=10.0)
    valleys = sorted(valleys, key=lambda ij: chi_coarse[ij[0], ij[1]])

    print(f"\nLocal-min pixels: {len(mins)}  -> clustered valleys: {len(valleys)}")
    topM = min(10, len(valleys))

    #Refine each valley locally
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

    #Compare truth chi2 on coarse grid 
    i_truth = int(np.argmin(np.abs(A_grid1 - float(A_true))))
    j_truth = int(np.argmin(np.abs(wrap_deg(phi_grid1_deg - np.rad2deg(float(phi_true))))))
    chi_truth = float(chi_coarse[i_truth, j_truth])

    print("\nFinal (best refined valley):")
    print(f"  A_best   = {A_best:.4f}")
    print(f"  phi_best = {np.rad2deg(phi_best):.2f} deg")
    print(f"  chi2 best  = {chi_best:.6g}")
    print(f"  chi2 truth = {chi_truth:.6g}")
    print(f"  ratio truth/best = {chi_truth/chi_best:.6f}")

    #  Save everything 
    np.savez(
        "recon_and_valleys_outputs.npz",
        A_grid1=A_grid1, phi_grid1=phi_grid1, chi_coarse=chi_coarse,
        refined=np.array(refined, dtype=np.float64),
        A_best=A_best, phi_best=phi_best, chi_best=chi_best,
        A_true=float(A_true), phi_true=float(phi_true),
        thin=thin, w2=w2, w3=w3, s2=s2, s3=s3
    )
    print("Saved: recon_and_valleys_outputs.npz")

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

    #Plot chi2 heatmap and mark valleys
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
    plt.savefig("recon_chi2_coarse_with_valleys.png")
    plt.show()

    # Overlay: data vs model for best + truth 
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
    plt.savefig("recon_g3_vs_model.png")
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
    plt.savefig("recon_g2_12_vs_model.png")
    plt.show()


if __name__ == "__main__":
    main()

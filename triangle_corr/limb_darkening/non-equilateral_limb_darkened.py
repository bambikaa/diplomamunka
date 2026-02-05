import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# --- Parameters ---
lam = 416e-9
theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)

u_ud = 0.0
u_ld = 0.6

# baseline range
B = np.linspace(0.0, 150.0, 4000)

# opening angles (degrees)
angles_deg = [30, 60, 90, 120, 150]

def g_ld(B, u):
    x = np.pi * theta * B / lam
    g = np.ones_like(x)

    m = x != 0
    term_ud = 2 * j1(x[m]) / x[m]
    term_mu = 2 * ((np.sin(x[m]) / x[m]) - np.cos(x[m])) / x[m]**2
    g[m] = ((1 - u) * term_ud + u * term_mu) / (1 - u / 3)
    return g

plt.figure(figsize=(10, 6))

for alpha_deg in angles_deg:
    alpha = np.deg2rad(alpha_deg)

    B12 = B
    B23 = B
    B31 = 2 * B * np.cos(alpha / 2)

    # coherence
    g12_ud = g_ld(B12, u_ud)
    g23_ud = g_ld(B23, u_ud)
    g31_ud = g_ld(np.abs(B31), u_ud)

    g12_ld = g_ld(B12, u_ld)
    g23_ld = g_ld(B23, u_ld)
    g31_ld = g_ld(np.abs(B31), u_ld)

    # correlations
    g2_ud = 1 + g12_ud**2
    g2_ld = 1 + g12_ld**2

    g3_ud = (1 + g12_ud**2 + g23_ud**2 + g31_ud**2
               + 2 * g12_ud * g23_ud * g31_ud)

    g3_ld = (1 + g12_ld**2 + g23_ld**2 + g31_ld**2
               + 2 * g12_ld * g23_ld * g31_ld)

    # plot: solid = limb-darkened, dashed = UD
    plt.plot(B, g3_ld, label=fr'$g^{{(3)}}$, $\alpha={alpha_deg}^\circ$')
    plt.plot(B, g3_ud, '--', alpha=0.6)

plt.xlabel("Baseline $B$ [m]")
plt.ylabel(r"$g^{(3)}$")
plt.title(r"$g^{(3)}$ sensitivity to limb darkening for different opening angles")
plt.ylim(0.8, 6.2)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

lam = 416e-9
theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)

B0 = np.linspace(0.0, 150.0, 2000)

def gamma_ud(B):
    x = np.pi * theta * B / lam
    g = np.ones_like(x)
    m = (x != 0)
    g[m] = 2.0 * j1(x[m]) / x[m]
    return g

angles_deg = [30, 60, 90, 120, 150] 
plt.figure(figsize=(9, 6))

for deg in angles_deg:
    alpha = np.deg2rad(deg)
    B31 = 2.0 * B0 * np.cos(alpha / 2.0)

    g12 = gamma_ud(B0)
    g23 = gamma_ud(B0)
    g31 = gamma_ud(np.abs(B31))

    g3_minus_1 = 3.0 * g12**2 + 2.0 * g12 * g23 * g31
    g3 = 1.0 + g3_minus_1  

    plt.plot(B0, g3, label=rf"$\alpha={deg}^\circ$")

plt.xlabel("Baseline $B$ [m]")
plt.ylabel(r"$g^{(3)}(B)$")
plt.title("Three-particle correlation for different opening angles")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("triangle_corr_various_angles.png", dpi=150)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

lam = 416e-9
theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)  

B = np.linspace(0.0, 150.0, 2000)  

def gamma_ud(Bm):
    Bm = np.asarray(Bm, dtype=float)
    x = np.pi * theta * Bm / lam
    g = np.ones_like(x)
    m = (x != 0)
    g[m] = 2.0 * j1(x[m]) / x[m]
    return g


shapes = [
    (1.00, 0.70, 1.20),
    (1.00, 0.85, 1.55),
    (1.00, 0.60, 1.40),
    (1.00, 0.95, 1.30),
    (1.00, 0.75, 1.65),
]


plt.figure(figsize=(9, 6))

for r12, r23, r31 in shapes:
    B12 = r12 * B
    B23 = r23 * B
    B31 = r31 * B

    g12 = gamma_ud(B12)
    g23 = gamma_ud(B23)
    g31 = gamma_ud(B31)

    g3 = 1.0 + g12**2 + g23**2 + g31**2 + 2.0 * g12 * g23 * g31

    plt.plot(B, g3, lw=2, label=rf"$(r_{{12}},r_{{23}},r_{{31}})=({r12:.2f},{r23:.2f},{r31:.2f})$")

plt.xlabel("Baseline scale $B$ [m]")
plt.ylabel(r"$g^{(3)}(B)$")
plt.title("Three-particle correlation (uniform disk)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("triangle_corr_sweep.png", dpi=150)
plt.show()

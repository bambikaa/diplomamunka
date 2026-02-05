import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

lam = 416e-9
theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)

u_ud = 0.0
u_ld = 0.6


B = np.linspace(0.0, 150.0, 4000)

def g_ld(B, u):
    x = np.pi * theta * B / lam
    g = np.ones_like(x)

    m = x != 0
    term_ud = 2 * j1(x[m]) / x[m]
    term_mu = 2 * ((np.sin(x[m]) / x[m]) - np.cos(x[m])) / x[m]**2
    g[m] = ((1 - u) * term_ud + u * term_mu) / (1 - u / 3)
    return g


g_ud = g_ld(B, u_ud)
g_ldisk = g_ld(B, u_ld)


g2_ud = 1 + g_ud**2
g2_ld = 1 + g_ldisk**2

g3_ud = 1 + 3*g_ud**2 + 2*g_ud**3
g3_ld = 1 + 3*g_ldisk**2 + 2*g_ldisk**3


plt.figure(figsize=(10,6))

plt.plot(B, g2_ud, '--', lw=2, label=r'$g^{(2)}$ uniform disk')
plt.plot(B, g2_ld, '-',  lw=2, label=r'$g^{(2)}$ limb-darkened')

plt.plot(B, g3_ud, '--', lw=2, label=r'$g^{(3)}$ uniform disk')
plt.plot(B, g3_ld, '-',  lw=2, label=r'$g^{(3)}$ limb-darkened')

plt.xlabel("Baseline $B$ [m]")
plt.ylabel("Correlation")
plt.title(r"Sensitivity of $g^{(3)}$ vs $g^{(2)}$ to limb darkening"
          "\nEquilateral baseline triangle")
plt.ylim(0.8, 6.2)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("equilateral_limb_darkened_uniform.png", dpi=150)
plt.show()

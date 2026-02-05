import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# ----------------------------
# Fizikai paraméterek
# (csak a skálát adják meg)
# ----------------------------
theta = 1.0e-3        # szögtmérő [rad] – tetszőleges, elméleti
lam = 500e-9          # hullámhossz [m] – tetszőleges, elméleti

# ----------------------------
# Baseline tengely [m]
# ----------------------------
B = np.linspace(0.0, 100.0, 3000)  # méter

# ----------------------------
# gamma(B)
# ----------------------------
x = np.pi * theta * B / lam
gamma = np.ones_like(x)
m = (x != 0)
gamma[m] = 2.0 * j1(x[m]) / x[m]

# ----------------------------
# 3-részecskés korreláció
# (szabályos háromszög)
# ----------------------------
g3 = 1.0 + 3.0 * gamma**2 + 2.0 * gamma**3

# ----------------------------
# Ábra
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(B, g3, lw=2)
plt.xlabel("Baseline $B$ [m]")
plt.ylabel(r"$g^{(3)}(B)$")
plt.title("Three-particle correlation for an equilateral triangle\n(purely mathematical)")
plt.grid(True)
plt.tight_layout()
plt.show()

# opcionális mentés
# plt.savefig("g3_equilateral_baseline_meters.pdf", bbox_inches="tight")

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# -----------------------------
# Parameters
# -----------------------------
lam = 500e-9          # wavelength [m]
theta_mas = 1.0       # uniform disk angular diameter [mas]
Bmax = 400.0
N = 2000

# -----------------------------
# Conversions
# -----------------------------
theta = theta_mas * 1e-3 * (np.pi / 648000.0)

# -----------------------------
# Uniform disk visibility
# -----------------------------
def V_ud(B):
    x = np.pi * theta * B / lam
    V = np.ones_like(x, dtype=float)
    m = x != 0
    V[m] = 2.0 * j1(x[m]) / x[m]
    return V.astype(complex)

B = np.linspace(0.0, Bmax, N)
V = V_ud(B)
Vamp = np.abs(V)

# -----------------------------
# Intensity correlation g2
# -----------------------------
g2 = 1.0 + Vamp**2

# -----------------------------
# Closure phase (equilateral triangle)
# -----------------------------
bispectrum = V * V * V
closure_deg = np.rad2deg(np.angle(bispectrum))

# Map to {0, 180} explicitly
closure_disc = np.zeros_like(closure_deg)
closure_disc[np.real(V) < 0] = 180.0

# Break the line at jumps by inserting NaNs
closure_plot = closure_disc.copy()
jumps = np.where(np.diff(closure_disc) != 0)[0]
closure_plot[jumps + 1] = np.nan

# -----------------------------
# Plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(9, 5.5))

# g2 curve
ax1.plot(
    B, g2,
    label=r"$g^{(2)}(0)=1+|g^{(1)}|^2$",
)
ax1.set_xlabel("Baseline length B (m)")
ax1.set_ylabel(r"Intensity correlation $g^{(2)}(0)$")
ax1.grid(True, alpha=0.3)

# Closure phase (separate color)
ax2 = ax1.twinx()
ax2.plot(
    B, closure_plot,
    linestyle='-',
    linewidth=2,
    label="Closure phase",
    color ='orange'
)
ax2.set_ylabel("Closure phase (degrees)")
ax2.set_ylim(-10, 190)

# Legends
ax1.legend(loc="upper right")
ax2.legend(loc="upper left")

plt.title(
    f"Uniform disk: θ = {theta_mas} mas, λ = {lam*1e9:.0f} nm\n"
    "Baseline-dependent intensity correlation and discrete closure phase"
)

plt.savefig("uniform_disk_closure_phase.png")
plt.tight_layout()
plt.show()

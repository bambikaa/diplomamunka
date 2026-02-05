import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Binary source parameters
# -----------------------------
lam = 500e-9          # wavelength [m]
sep_mas = 1.0         # angular separation [mas]
flux_ratio = 0.5      # secondary / primary

# Convert separation to radians
sep = sep_mas * 1e-3 * (np.pi / 648000.0)

# Binary separation vector (fixed on sky, along x)
s = np.array([sep, 0.0])

# -----------------------------
# Baseline geometry
# Equilateral triangle, fixed orientation
# -----------------------------
B = np.linspace(0.0, 300.0, 1200)   # baseline length [m]

u12 = np.column_stack([B/lam, np.zeros_like(B)])
u23 = np.column_stack([-0.5*B/lam,  np.sqrt(3)/2 * B/lam])
u31 = np.column_stack([-0.5*B/lam, -np.sqrt(3)/2 * B/lam])

# -----------------------------
# Binary visibility
# -----------------------------
def V_binary(u):
    phase = -2j * np.pi * (u @ s)
    return (1.0 + flux_ratio * np.exp(phase)) / (1.0 + flux_ratio)

V12 = V_binary(u12)
V23 = V_binary(u23)
V31 = V_binary(u31)

# -----------------------------
# Closure phase
# -----------------------------
bispectrum = V12 * V23 * V31
closure_phase_deg = np.rad2deg(np.angle(bispectrum))

# Visibility amplitude (example: baseline 12)
Vamp = (V12)

# -----------------------------
# Plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(9, 5.5))

ax1.plot(B, Vamp)
ax1.set_xlabel("Baseline length B (m)")
ax1.set_ylabel(r"Visibility amplitude $|V|$")
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(B, closure_phase_deg, color='orange')
ax2.set_ylabel("Closure phase (degrees)")
ax2.set_ylim(-180, 180)

plt.title(
    f"Binary source: separation = {sep_mas} mas, flux ratio = {flux_ratio}\n"
    "Non-zero closure phase as a function of baseline length"
)

plt.savefig("binary_source_closure_phase.png")
plt.tight_layout()
plt.show()

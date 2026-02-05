import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# -----------------------
# Source + observing setup
# -----------------------
lam = 416e-9          # central wavelength [m]
dlam = 10e-9          # optical bandwidth [m]  (e.g. 10 nm filter)
dt = 1e-9             # effective time bin / timing resolution [s] (e.g. 1 ns)
c = 299792458.0

theta_mas = 1.0       # angular DIAMETER [mas]
theta = theta_mas * 1e-3 * (np.pi / 648000.0)  # mas -> rad (diameter)

u = 0.6             # linear limb-darkening coefficient

# Convert bandwidth to coherence time (order-of-magnitude model)
dnu = c * dlam / (lam**2)     # [Hz]
tau_c = 1.0 / dnu             # [s]  (simple estimate)

c2 = tau_c / dt
c3 = (tau_c / dt)**2

print(f"Δν ≈ {dnu:.3e} Hz,  τc ≈ {tau_c:.3e} s,  c2=τc/Δt ≈ {c2:.3e},  c3 ≈ {c3:.3e}")

# -----------------------
# Baseline scan + geometry
# -----------------------
B = np.linspace(0.0, 150.0, 4000)  # [m]
angles_deg = [30, 60, 90, 120, 150]  # opening angles

def g_limbdark(B, u):
    """
    Linear limb-darkened disk coherence g(B), normalized so g(0)=1.
    (Centrosymmetric -> real.)
    """
    x = np.pi * theta * B / lam
    g = np.ones_like(x)

    m = x != 0
    term_ud = 2.0 * j1(x[m]) / x[m]
    term_mu = 2.0 * ((np.sin(x[m]) / x[m]) - np.cos(x[m])) / (x[m]**2)
    g[m] = ((1.0 - u) * term_ud + u * term_mu) / (1.0 - u / 3.0)
    return g

def g3_meas_from_g(g12, g23, g31, c2, c3):
    # "Measured" third-order correlation with finite time-resolution contrast
    return 1.0 + c2*(g12**2 + g23**2 + g31**2) + 2.0*c3*(g12*g23*g31)

plt.figure(figsize=(10, 6))

for a_deg in angles_deg:
    a = np.deg2rad(a_deg)

    B12 = B
    B23 = B
    B31 = 2.0 * B * np.cos(a/2.0)   # from triangle geometry
    B31 = np.abs(B31)

    g12 = g_limbdark(B12, u=u)
    g23 = g_limbdark(B23, u=u)
    g31 = g_limbdark(B31, u=u)

    g3m = g3_meas_from_g(g12, g23, g31, c2=c2, c3=c3)

    # Plot "excess" in ppm for readability
    plt.plot(B, (g3m - 1.0)*1e6, label=fr'$\alpha={a_deg}^\circ$')

plt.xlabel("Baseline $B$ [m]")
plt.ylabel(r"$(g^{(3)}-1)\times 10^6$")
plt.title("Modeled measurable third-order intensity correlation\n"
          f"λ={lam*1e9:.0f} nm, Δλ={dlam*1e9:.0f} nm, Δt={dt*1e9:.1f} ns")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("triangle_corr_real.png", dpi=150)
plt.show()

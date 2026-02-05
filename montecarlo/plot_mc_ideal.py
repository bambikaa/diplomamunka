import numpy as np
import matplotlib.pyplot as plt

def seconds_to_hours(t):
    return (t - t[0]) / 3600.0

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

print(f"Loaded: {inp}")
print(f"Points: {len(t)} (step={step}), K={K}, mu_counts={mu_counts}")

finite = np.isfinite(cosphi).mean()
print(f"cosphi finite fraction: {finite:.3f}")
print("min(absT) =", np.nanmin(absT), "max(absT) =", np.nanmax(absT))
print("min(g2_12-1) =", np.min(g2_12 - 1.0),
      "min(g2_23-1) =", np.min(g2_23 - 1.0),
      "min(g2_31-1) =", np.min(g2_31 - 1.0))

# -------------------------
# Plot 1: g2 - 1
# -------------------------
plt.figure(figsize=(9, 5))
plt.plot(th, g2_12 - 1.0, label=r"$g^{(2)}_{12}-1$")
plt.plot(th, g2_23 - 1.0, label=r"$g^{(2)}_{23}-1$")
plt.plot(th, g2_31 - 1.0, label=r"$g^{(2)}_{31}-1$")
plt.xlabel("Time [hours]")
plt.ylabel(r"$g^{(2)}-1$")
plt.title(r"IDEAL: Second-order HBT contrast vs time")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Plot 2: g3 - 1
# -------------------------
plt.figure(figsize=(9, 5))
plt.plot(th, g3_123 - 1.0, label=r"$g^{(3)}_{123}-1$")
plt.xlabel("Time [hours]")
plt.ylabel(r"$g^{(3)}-1$")
plt.title(r"IDEAL: Third-order correlation excess vs time")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Plot 3: Re(T)
# -------------------------
plt.figure(figsize=(9, 5))
plt.plot(th, ReT, label=r"$\Re(\gamma_{12}\gamma_{23}\gamma_{31})$")
plt.xlabel("Time [hours]")
plt.ylabel(r"$\Re(T)$")
plt.title(r"IDEAL: Connected 3rd-order term (real triple product)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Plot 4: cos(phi_cl)
# -------------------------
plt.figure(figsize=(9, 5))
plt.plot(th, cosphi, label=r"$\cos(\Phi_{\rm cl})$", marker ='o', markersize=2, linestyle='none')
plt.xlabel("Time [hours]")
plt.ylabel(r"$\cos(\Phi_{\rm cl})$")
plt.title(r"IDEAL: Closure information from 3rd-order HBT")
plt.ylim(-1.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Optional Plot 5: |phi_cl| = arccos(cosphi)
# Note: sign ambiguity remains in pure g3(0) HBT.
# -------------------------
phi_abs = np.arccos(np.clip(cosphi, -1.0, 1.0))

plt.figure(figsize=(9, 5))
plt.plot(th, phi_abs, label=r"$|\Phi_{\rm cl}|=\arccos(\cos\Phi_{\rm cl})$", marker ='o', markersize=2, linestyle='none')
plt.xlabel("Time [hours]")
plt.ylabel("Phase [rad]")
plt.title(r"IDEAL: Closure phase magnitude (sign ambiguous in pure $g^{(3)}(0)$)")
plt.legend()
plt.tight_layout()
plt.show()

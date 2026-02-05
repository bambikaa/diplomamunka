import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import j1

# ----------------------------
# Fizikai paraméterek
# ----------------------------
lam = 416e-9
theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)

B = np.linspace(0.0, 150.0, 2000)

# ----------------------------
# Uniform disk gamma(B)
# ----------------------------
def gamma_ud(Bm):
    Bm = np.asarray(Bm, dtype=float)
    x = np.pi * theta * Bm / lam
    g = np.ones_like(x)
    m = (x != 0)
    g[m] = 2.0 * j1(x[m]) / x[m]
    return g

def g3_curve(r23, r31):
    g12 = gamma_ud(B)
    g23 = gamma_ud(r23 * B)
    g31 = gamma_ud(r31 * B)
    return 1.0 + g12**2 + g23**2 + g31**2 + 2.0 * g12 * g23 * g31

# ----------------------------
# Ábra előkészítése
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

line, = ax.plot(B, g3_curve(0.8, 1.3), lw=2)
text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes,
    ha="left", va="top", fontsize=12
)

ax.set_xlabel("Baseline $B$ [m]")
ax.set_ylabel(r"$g^{(3)}(B)$")
ax.set_title("Three-particle correlation")
ax.grid(True)

# ----------------------------
# Animáció
# ----------------------------
n_frames = 120
r23_vals = np.linspace(0.6, 1.1, n_frames)
r31_vals = np.linspace(1.6, 1.1, n_frames)

def animate(i):
    r23 = r23_vals[i]
    r31 = r31_vals[i]

    line.set_ydata(g3_curve(r23, r31))
    text.set_text(
        rf"$r_{{23}} = {r23:.2f}$" "\n"
        rf"$r_{{31}} = {r31:.2f}$"
    )
    return line, text

ani = animation.FuncAnimation(
    fig, animate, frames=n_frames, blit=True
)

# ----------------------------
# Mentés GIF-be
# ----------------------------
ani.save(
    "triangle_g3_sweep.gif",
    writer="pillow",
    fps=20
)

plt.close()

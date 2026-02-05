import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

r23_0 = 0.8
r31_0 = 1.3

def g3_curve(r23, r31):
    g12 = gamma_ud(B)
    g23 = gamma_ud(r23 * B)
    g31 = gamma_ud(r31 * B)
    return 1.0 + g12**2 + g23**2 + g31**2 + 2.0 * g12 * g23 * g31

g3_init = g3_curve(r23_0, r31_0)

fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(bottom=0.25)

(line,) = ax.plot(B, g3_init, lw=2)

ax.set_xlabel("Baseline scale $B$ [m]")
ax.set_ylabel(r"$g^{(3)}(B)$")
ax.set_title("Interactive three-particle correlation (scalene triangle)")
ax.grid(True)

ax_r23 = plt.axes([0.15, 0.12, 0.65, 0.03])
ax_r31 = plt.axes([0.15, 0.07, 0.65, 0.03])

slider_r23 = Slider(
    ax=ax_r23,
    label=r"$r_{23}=B_{23}/B_{12}$",
    valmin=0.3,
    valmax=1.7,
    valinit=r23_0,
)

slider_r31 = Slider(
    ax=ax_r31,
    label=r"$r_{31}=B_{31}/B_{12}$",
    valmin=0.3,
    valmax=1.7,
    valinit=r31_0,
)

def update(val):
    r23 = slider_r23.val
    r31 = slider_r31.val

    if r23 + 1.0 <= r31 or r31 + r23 <= 1.0 or r31 + 1.0 <= r23:
        line.set_ydata(np.nan * B)
    else:
        line.set_ydata(g3_curve(r23, r31))

    fig.canvas.draw_idle()

slider_r23.on_changed(update)
slider_r31.on_changed(update)

import matplotlib.animation as animation

def animate(i):
    r23 = 0.6 + 0.8 * i / 100
    r31 = 1.6 - 0.8 * i / 100
    line.set_ydata(g3_curve(r23, r31))
    return line,

ani = animation.FuncAnimation(fig, animate, frames=100)
ani.save("triangle_g3.gif", writer="pillow", fps=20)


plt.show()

import numpy as np
from hbt_funcs import bilinear_sample_complex

inp = "./source_image.npz"
d = np.load(inp, allow_pickle=True)

I = d["I"]          
x = d["x"]
y = d["y"]
dx = float(d["dx"])
dy = float(d["dy"])
N = int(d["N"])
lam = float(d["lam"])

# FFT -> visibility V(u,v)
# V(u,v) = ∫ I(x,y) exp(-2π i (u x + v y)) dx dy


# frequency grids
u_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
#print(u_grid[:100])
v_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dy))

F = np.fft.fftshift(np.fft.fft2(I)) * dx * dy  # approximate integral

# normalize so that V(0,0)=1
i0 = N // 2
V00 = F[i0, i0]
F = F / V00

B = 80.0                 # baseline length 
u0 = B / lam          

# time sampling 
hours = 6.0
dt_minutes = 1.0
t = np.arange(0.0, hours * 3600.0 + 0.5, dt_minutes * 60.0)  # seconds

# Idealized rotation
phi_start = 0.0
phi_end = np.deg2rad(180.0)  
phi = phi_start + (phi_end - phi_start) * (t - t[0]) / (t[-1] - t[0])

# three baselines 120° apart 
phi12 = phi + 0.0
phi23 = phi + 2.0 * np.pi / 3.0
phi31 = phi + 4.0 * np.pi / 3.0

# uv points
u12 = u0 * np.cos(phi12)
v12 = u0 * np.sin(phi12)

u23 = u0 * np.cos(phi23)
v23 = u0 * np.sin(phi23)

u31 = u0 * np.cos(phi31)
v31 = u0 * np.sin(phi31)

gamma12 = np.empty_like(t, dtype=np.complex128)
gamma23 = np.empty_like(t, dtype=np.complex128)
gamma31 = np.empty_like(t, dtype=np.complex128)

for k in range(len(t)):
    gamma12[k] = bilinear_sample_complex(u_grid, v_grid, F, u12[k], v12[k])
    gamma23[k] = bilinear_sample_complex(u_grid, v_grid, F, u23[k], v23[k])
    gamma31[k] = bilinear_sample_complex(u_grid, v_grid, F, u31[k], v31[k])

out = "gamma_earth_rotation.npz"
np.savez(
    out,
    t=t,
    phi=phi,
    u0=u0,
    u12=u12, v12=v12, gamma12=gamma12,
    u23=u23, v23=v23, gamma23=gamma23,
    u31=u31, v31=v31, gamma31=gamma31,
    u_grid=u_grid, v_grid=v_grid,
)

print(f"Loaded: {inp}")
print(f"Saved:  {out}")

# check
uv_closure = np.max(np.sqrt((u12 + u23 + u31)**2 + (v12 + v23 + v31)**2))
print("Max |u12+u23+u31, v12+v23+v31| =", uv_closure)

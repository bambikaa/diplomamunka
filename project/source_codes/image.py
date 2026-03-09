import numpy as np
import matplotlib.pyplot as plt

# Parameters (given)
lam = 416e-9          # [m]
dlam = 10e-9          # [m]   
dt = 1e-9             # [s]
c = 299792458.0

theta_mas = 1.0
theta = theta_mas * 1e-3 * (np.pi / 648000.0)  # diameter [rad]

u_ld = 0.6            # limb-darkening coefficient

# Image grid
N = 256
fov_factor = 3.0
fov = fov_factor * theta   # total width [rad]

# pixel coordinates in radians
x = (np.arange(N) - (N - 1) / 2) * (fov / N)
y = (np.arange(N) - (N - 1) / 2) * (fov / N)
X, Y = np.meshgrid(x, y, indexing="xy")
Rxy = np.sqrt(X**2 + Y**2)

dx = fov / N
dy = fov / N

#limb-darkened disk
R_disk = theta / 2.0
disk = np.zeros_like(Rxy)

inside = Rxy <= R_disk
rho = Rxy[inside] / R_disk
mu = np.sqrt(1.0 - rho**2)

disk[inside] = 1.0 - u_ld * (1.0 - mu)  # I0=1

# Gaussian spot
spot_r = 0.5 * R_disk
spot_phi = np.deg2rad(30.0)
x_s = spot_r * np.cos(spot_phi)
y_s = spot_r * np.sin(spot_phi)

sigma_s = 0.10 * R_disk
A = 0.20 

spot = A * np.exp(-((X - x_s)**2 + (Y - y_s)**2) / (2.0 * sigma_s**2))
spot *= inside  # keep spot on the inside of the disk

# total image
I = disk + spot
I[I < 0] = 0.0

# normalize total flux to 1
I /= (I.sum() * dx * dy)

out = "source_image.npz"
np.savez(
    out,
    I=I.astype(np.float64),
    x=x.astype(np.float64),
    y=y.astype(np.float64),
    dx=np.float64(dx),
    dy=np.float64(dy),
    N=np.int32(N),
    fov=np.float64(fov),
    fov_factor=np.float64(fov_factor),
    theta=np.float64(theta),
    theta_mas=np.float64(theta_mas),
    R_disk=np.float64(R_disk),
    u_ld=np.float64(u_ld),
    spot_params=np.array([x_s, y_s, sigma_s, A], dtype=np.float64),
    lam=np.float64(lam),
    dlam=np.float64(dlam),
    dt=np.float64(dt),
    c=np.float64(c),
)

print(f"Saved: {out}")

extent_mas = (np.array([-fov/2, fov/2, -fov/2, fov/2]) * (648000.0/np.pi) * 1e3)
plt.figure(figsize=(6, 5))
plt.imshow(I, origin="lower", extent=extent_mas, interpolation="nearest")
plt.xlabel("x [mas]")
plt.ylabel("y [mas]")
plt.title("Limb-darkened disk + Gaussian spot")
plt.colorbar(label="Intensity (arb.)")
plt.xlim(-0.8, 0.8)
plt.ylim(-0.8, 0.8)
plt.tight_layout()
plt.savefig('image.png')
plt.show()

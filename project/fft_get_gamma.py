import numpy as np
from hbt_funcs import bilinear_sample_complex

def altitude_rad(lat_rad, dec_rad, H_rad):
    return np.arcsin(
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(H_rad)
    )

def source_direction(alpha_rad, delta_rad):
    S = np.array([
        -np.sin(alpha_rad - delta_rad),
        0.0,
        np.cos(alpha_rad - delta_rad)
    ], dtype=float)
    n = np.linalg.norm(S)
    return S / n

def rotation_matrix(axis, phi):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0]
    ], dtype=float)

    I = np.eye(3)
    c = np.cos(phi)
    s = np.sin(phi)

    return c * I + (1.0 - c) * np.outer(axis, axis) + s * K

def uv_basis_from_S(S):
    S = np.asarray(S, dtype=float)
    S = S / np.linalg.norm(S)

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(S, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    e_u = np.cross(S, ref)
    e_u /= np.linalg.norm(e_u)

    e_v = np.cross(S, e_u)
    e_v /= np.linalg.norm(e_v)

    return e_u, e_v

def compute_uv_tracks(a1_0, a2_0, a3_0, alpha_rad, delta_rad, phi_array, lam):
    S = source_direction(alpha_rad, delta_rad)
    e_u, e_v = uv_basis_from_S(S)

    u12 = np.empty(len(phi_array), dtype=float)
    v12 = np.empty(len(phi_array), dtype=float)
    u23 = np.empty(len(phi_array), dtype=float)
    v23 = np.empty(len(phi_array), dtype=float)
    u31 = np.empty(len(phi_array), dtype=float)
    v31 = np.empty(len(phi_array), dtype=float)

    for k, phi in enumerate(phi_array):
        F = rotation_matrix(S, float(phi))

        a1 = F @ a1_0
        a2 = F @ a2_0
        a3 = F @ a3_0

        b12 = a2 - a1
        b23 = a3 - a2
        b31 = a1 - a3

        u12[k] = np.dot(b12, e_u) / lam
        v12[k] = np.dot(b12, e_v) / lam
        u23[k] = np.dot(b23, e_u) / lam
        v23[k] = np.dot(b23, e_v) / lam
        u31[k] = np.dot(b31, e_u) / lam
        v31[k] = np.dot(b31, e_v) / lam

    return u12, v12, u23, v23, u31, v31


inp = "./source_image.npz"
d = np.load(inp, allow_pickle=True)

I = d["I"]
x = d["x"]
y = d["y"]
dx = float(d["dx"])
dy = float(d["dy"])
N = int(d["N"])
lam = float(d["lam"])

u_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
v_grid = np.fft.fftshift(np.fft.fftfreq(N, d=dy))

F = np.fft.fftshift(np.fft.fft2(I)) * dx * dy

# Normalize so that V(0,0)=1
i0 = N // 2
V00 = F[i0, i0]
F = F / V00

B = 80.0  # meters
a1_0 = np.array([0.0, -B / np.sqrt(3.0), 0.0])
a2_0 = np.array([+B / 2.0, +B / (2.0 * np.sqrt(3.0)), 0.0])
a3_0 = np.array([-B / 2.0, +B / (2.0 * np.sqrt(3.0)), 0.0])

alpha_deg = 47.0
delta_deg = 20.0
alpha = np.deg2rad(alpha_deg)
delta = np.deg2rad(delta_deg)


dt_minutes = 1.0
dt_sec = dt_minutes * 60.0

omega = 2.0 * np.pi / (24.0 * 3600.0)

# rise/set hour angle
h_min_deg = 0.0
h_min = np.deg2rad(h_min_deg)

cosH0 = (
    np.sin(h_min) - np.sin(alpha) * np.sin(delta)
) / (np.cos(alpha) * np.cos(delta))
cosH0 = np.clip(cosH0, -1.0, 1.0)
H0 = np.arccos(cosH0)


visible_duration_sec = 2.0 * H0 / omega

n_blocks = int(visible_duration_sec // dt_sec)

t = (-H0 / omega) + (np.arange(n_blocks) + 0.5) * dt_sec
phi = omega * t
alt = altitude_rad(alpha, delta, phi)

u12, v12, u23, v23, u31, v31 = compute_uv_tracks(
    a1_0, a2_0, a3_0,
    alpha, delta,
    phi,
    lam
)

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
    altitude=alt,
    alpha=alpha, delta=delta,
    B=B, lam=lam,
    u12=u12, v12=v12, gamma12=gamma12,
    u23=u23, v23=v23, gamma23=gamma23,
    u31=u31, v31=v31, gamma31=gamma31,
    u_grid=u_grid, v_grid=v_grid,
)

print(f"Loaded: {inp}")
print(f"Saved:  {out}")
print(f"Kept {len(t)} samples")

t_rise = -H0 / omega
t_set = H0 / omega
t_visible = t_set - t_rise

def sec_to_clock(seconds, noon_hour=12):
    total_minutes = int(round(seconds / 60))
    hour = noon_hour + total_minutes // 60
    minute = total_minutes % 60

    hour = hour % 24
    return f"{hour:02d}:{minute:02d}"

print("rise:", sec_to_clock(t_rise))
print("set:", sec_to_clock(t_set))

hours_visible = int(t_visible // 3600)
minutes_visible = int((t_visible % 3600) // 60)

print(f"time spent over the horizon: {hours_visible} h {minutes_visible} min")

uv_closure = np.max(np.sqrt((u12 + u23 + u31) ** 2 + (v12 + v23 + v31) ** 2))
print("Max |u12+u23+u31, v12+v23+v31| =", uv_closure)

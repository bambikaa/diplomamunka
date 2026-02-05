import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Asymmetric source: limb-darkened disk + off-center spot
# -----------------------------
N = 512
FOV_mas = 6.0
theta_diam_mas = 1.0
u_ld = 0.6

spot_flux_frac = 0.05
spot_sigma_mas = 0.08
spot_x_mas = 0.25
spot_y_mas = 0.10

x = np.linspace(-FOV_mas/2, FOV_mas/2, N, endpoint=False)
y = np.linspace(-FOV_mas/2, FOV_mas/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")
R = np.sqrt(X**2 + Y**2)

Rstar_mas = theta_diam_mas / 2.0
inside = (R <= Rstar_mas)

mu = np.zeros_like(R)
mu[inside] = np.sqrt(1.0 - (R[inside]/Rstar_mas)**2)

I_star = np.zeros_like(R)
I_star[inside] = 1.0 - u_ld*(1.0 - mu[inside])
I_star /= I_star.sum()  # normalize star flux to 1

spot = np.exp(-((X-spot_x_mas)**2 + (Y-spot_y_mas)**2) / (2*spot_sigma_mas**2))
spot /= spot.sum()

I = (1.0 - spot_flux_frac)*I_star + spot_flux_frac*spot

# -----------------------------
# 2) FFT -> g(u,v) grid (normalized so g(0,0)=1)
# -----------------------------
mas_to_rad = np.pi / (180.0*3600.0*1000.0)
dx = (FOV_mas / N) * mas_to_rad  # rad per pixel

G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I)))
G /= G[N//2, N//2]  # normalize: g(0,0)=1

fu = np.fft.fftshift(np.fft.fftfreq(N, d=dx))  # cycles/rad
fv = fu.copy()
du = fu[1] - fu[0]
dv = du

def g_uv(u, v):
    """Bilinear interpolation of complex g(u,v) on FFT grid."""
    iu = (u - fu[0]) / du
    iv = (v - fv[0]) / dv
    if iu < 0 or iu > N-1 or iv < 0 or iv > N-1:
        return 0.0 + 0.0j

    i0 = int(np.floor(iu)); j0 = int(np.floor(iv))
    i1 = min(i0 + 1, N-1);  j1 = min(j0 + 1, N-1)
    a = iu - i0
    b = iv - j0

    g00 = G[j0, i0]
    g10 = G[j0, i1]
    g01 = G[j1, i0]
    g11 = G[j1, i1]
    return (1-a)*(1-b)*g00 + a*(1-b)*g10 + (1-a)*b*g01 + a*b*g11

# -----------------------------
# 3) Orientation scan for equilateral triangle
# -----------------------------
lam = 416e-9
B = 80.0  # pick a representative baseline length [m]

base_angles = np.deg2rad([0.0, 120.0, 240.0])
phi_deg = np.linspace(0.0, 360.0, 721)   # fine sampling
phi = np.deg2rad(phi_deg)

Phi = np.zeros_like(phi)  # closure phase [rad]
G3  = np.zeros_like(phi)  # third-order correlation

for k, ph in enumerate(phi):
    dirs = base_angles + ph
    Bx = B*np.cos(dirs)
    By = B*np.sin(dirs)

    u12, v12 = Bx[0]/lam, By[0]/lam
    u23, v23 = Bx[1]/lam, By[1]/lam
    u31, v31 = Bx[2]/lam, By[2]/lam

    g12 = g_uv(u12, v12)
    g23 = g_uv(u23, v23)
    g31 = g_uv(u31, v31)

    T = g12*g23*g31
    Phi[k] = np.angle(T)
    G3[k]  = 1.0 + (abs(g12)**2 + abs(g23)**2 + abs(g31)**2) + 2.0*np.real(T)

# unwrap phase for prettier curve
Phi_unw = np.unwrap(Phi)
Phi_deg = Phi_unw * 180/np.pi

# -----------------------------
# 4) One figure, two y-axes
# -----------------------------
fig, ax1 = plt.subplots(figsize=(11, 6))

ax1.plot(phi_deg, G3 - 1.0, lw=2)
ax1.set_xlabel(r"Triangle rotation angle $\varphi$ [deg]")
ax1.set_ylabel(r"$g^{(3)}(\varphi)-1$")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(phi_deg, Phi_deg, lw=2, linestyle="--")
ax2.set_ylabel(r"Closure phase $\Phi(\varphi)$ [deg]")

plt.title("Orientation dependence for an asymmetric source (equilateral triangle)\n"
          f"Limb-darkened disk + spot, baseline length B={B:.1f} m")
plt.tight_layout()
plt.show()

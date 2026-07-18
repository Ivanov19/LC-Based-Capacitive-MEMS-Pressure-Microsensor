"""
Clamped circular silicon membrane under uniform pressure.
Thin-plate (Kirchhoff) small-deflection theory. SI units internally.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Material & geometry constants (SI) ----
E   = 170e9      # Young's modulus [Pa]
NU  = 0.28       # Poisson ratio
A   = 500e-6     # membrane radius [m]
H   = 10e-6      # thickness [m]
P_MAX = 50e3     # full-scale pressure [Pa]

# ---- Core physics ----
def flexural_rigidity(h, E, nu):
    """D = E h^3 / [12(1-nu^2)]"""
    return E * h**3 / (12 * (1 - nu**2))

def center_deflection(P, a, h, E, nu):
    """w0 = P a^4 / (64 D)"""
    D = flexural_rigidity(h, E, nu)
    return P * a**4 / (64 * D)

def deflection_profile(r, P, a, h, E, nu):
    """w(r) = w0 (1 - r^2/a^2)^2, clamped edge."""
    w0 = center_deflection(P, a, h, E, nu)
    return w0 * (1 - (r / a)**2)**2

# ---- Reporting ----
D = flexural_rigidity(H, E, NU)
w0_fs = center_deflection(P_MAX, A, H, E, NU)
print(f"Flexural rigidity D = {D:.4e} N·m")
print(f"w0 at {P_MAX/1e3:.0f} kPa = {w0_fs*1e6:.4f} µm")

# thin-plate validity flag
limit = H / 5
if w0_fs > limit:
    print(f"WARNING: w0 ({w0_fs*1e6:.2f} µm) exceeds h/5 "
          f"({limit*1e6:.2f} µm) — small-deflection theory unreliable.")
else:
    print(f"OK: w0 within thin-plate regime (< h/5 = {limit*1e6:.2f} µm).")

# ---- Sanity checks ----
# 1) profile at r=0 == center_deflection
assert np.isclose(deflection_profile(0.0, P_MAX, A, H, E, NU),
                  center_deflection(P_MAX, A, H, E, NU))
# 2) linear in P
assert np.isclose(center_deflection(2*P_MAX, A, H, E, NU),
                  2*center_deflection(P_MAX, A, H, E, NU))
# 3) scales as a^4
assert np.isclose(center_deflection(P_MAX, 2*A, H, E, NU),
                  16*center_deflection(P_MAX, A, H, E, NU))
print("Sanity checks passed.")

# ---- Plot 1: center deflection vs pressure ----
P_sweep = np.linspace(0, P_MAX, 100)
w0_sweep = center_deflection(P_sweep, A, H, E, NU)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(P_sweep/1e3, w0_sweep*1e6, color="C0")
ax1.set_xlabel("Pressure [kPa]")
ax1.set_ylabel("Center deflection w₀ [µm]")
ax1.set_title("Center deflection vs pressure")
ax1.grid(alpha=0.3)

# ---- Plot 2: radial profile at max pressure ----
r = np.linspace(-A, A, 200)
w = deflection_profile(np.abs(r), P_MAX, A, H, E, NU)

ax2.plot(r*1e6, w*1e6, color="C3")
ax2.fill_between(r*1e6, w*1e6, alpha=0.15, color="C3")
ax2.set_xlabel("Radial position r [µm]")
ax2.set_ylabel("Deflection w(r) [µm]")
ax2.set_title(f"Radial shape at P = {P_MAX/1e3:.0f} kPa")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/membrane.png", dpi=130)
print("Saved membrane.png")

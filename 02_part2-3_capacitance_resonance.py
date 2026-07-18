"""
Part 2+3: MEMS pressure sensor.
Deflection -> variable-gap capacitance -> LC resonant frequency.
Reuses Part-1 deflection model. SI internally.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ---- Constants (SI) ----
E, NU = 170e9, 0.28
A, H  = 500e-6, 10e-6
P_MAX = 50e3
G0    = 3e-6              # initial gap [m]
EPS0  = 8.854e-12        # F/m
MU0   = 4 * np.pi * 1e-7 # H/m

# Coil
N     = 18   # 20->1.11µH, 19->1.00µH (edge); 18 sits cleanly in range
D_OUT = 4e-3
D_IN  = 1e-3
K1, K2 = 2.34, 2.75      # square-coil Wheeler coeffs

# ---- Part 1 physics (reused) ----
def flexural_rigidity(h, E, nu):
    return E * h**3 / (12 * (1 - nu**2))

def center_deflection(P, a, h, E, nu):
    return P * a**4 / (64 * flexural_rigidity(h, E, nu))

def deflection_profile(r, P, a, h, E, nu):
    w0 = center_deflection(P, a, h, E, nu)
    return w0 * (1 - (r / a)**2)**2

# ---- Part 2: capacitance ----
def capacitance(P, a=A, h=H, E=E, nu=NU, g0=G0):
    """C = eps0 * integral 2*pi*r / (g0 - w(r)) dr, 0..a."""
    w_max = deflection_profile(0.0, P, a, h, E, nu)  # max at center
    if w_max >= g0:
        return np.nan  # plates touch -> invalid
    integrand = lambda r: 2 * np.pi * r / (g0 - deflection_profile(r, P, a, h, E, nu))
    val, _ = quad(integrand, 0, a)
    return EPS0 * val

# ---- Part 3: inductance & resonance ----
def spiral_inductance(n=N, d_out=D_OUT, d_in=D_IN):
    d_avg = (d_out + d_in) / 2
    rho   = (d_out - d_in) / (d_out + d_in)
    return K1 * MU0 * n**2 * d_avg / (1 + K2 * rho)

def resonant_freq(P, L=None):
    if L is None:
        L = spiral_inductance()
    C = capacitance(P)
    if np.isnan(C):
        return np.nan
    return 1 / (2 * np.pi * np.sqrt(L * C))

# ---- Reports & sanity ----
C0 = capacitance(0.0)
C0_flat = EPS0 * np.pi * A**2 / G0
print(f"[P2] C0 = {C0*1e12:.4f} pF | flat-plate = {C0_flat*1e12:.4f} pF")
assert np.isclose(C0, C0_flat, rtol=1e-4), "C0 mismatch"

L = spiral_inductance()
print(f"[P3] L = {L*1e9:.2f} nH")
assert 100e-9 <= L <= 1e-6, "L out of target range"

# sweep
P = np.linspace(0, P_MAX, 100)
C = np.array([capacitance(p) for p in P])
f = np.array([resonant_freq(p, L) for p in P])

valid = ~np.isnan(C)
if not valid.all():
    print(f"[P2] WARNING: plates touch above "
          f"{P[valid].max()/1e3:.1f} kPa — points dropped.")

# sensitivities (linear fit over valid range)
dC = np.polyfit(P[valid]/1e3, C[valid]*1e15, 1)[0]   # fF/kPa
df = np.polyfit(P[valid]/1e3, f[valid]/1e3, 1)[0]     # kHz/kPa
print(f"[P2] C rises {C[valid][0]*1e12:.3f} -> {C[valid][-1]*1e12:.3f} pF")
print(f"[P2] Sensitivity dC/dP = {dC:.2f} fF/kPa")
print(f"[P3] f0 = {f[valid][0]/1e6:.3f} -> {f[valid][-1]/1e6:.3f} MHz")
print(f"[P3] Sensitivity df/dP = {df:.2f} kHz/kPa")
assert f[valid][-1] < f[valid][0], "f0 should decrease with P"
print("Trend OK: f0 decreases as C rises.")

# ---- Plots ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(P[valid]/1e3, C[valid]*1e12, color="C0")
ax1.set(xlabel="Pressure [kPa]", ylabel="Capacitance [pF]",
        title="C vs P (variable gap)")
ax1.grid(alpha=0.3)

ax2.plot(P[valid]/1e3, f[valid]/1e6, color="C3")
ax2.set(xlabel="Pressure [kPa]", ylabel="f₀ [MHz]",
        title="LC resonance vs P")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/sensor.png", dpi=130)
print("Saved sensor.png")

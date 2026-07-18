"""
Part 4: Wireless inductive readout (Takahata-style).
Reader coil coupled to sensor LC; dip in |Z_in| tracks pressure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# ---- Constants (SI) ----
E, NU = 170e9, 0.28
A, H  = 500e-6, 10e-6
P_MAX = 50e3
G0    = 3e-6
EPS0  = 8.854e-12
MU0   = 4 * np.pi * 1e-7
N, D_OUT, D_IN = 18, 4e-3, 1e-3
K1, K2 = 2.34, 2.75
R_S = 5.0    # coil resistance [ohm]
K   = 0.2    # coupling

# ---- Reused physics ----
def flex_rig(h, E, nu): return E*h**3/(12*(1-nu**2))
def center_defl(P,a,h,E,nu): return P*a**4/(64*flex_rig(h,E,nu))
def defl(r,P,a,h,E,nu):
    return center_defl(P,a,h,E,nu)*(1-(r/a)**2)**2

def capacitance(P,a=A,h=H,g0=G0):
    if defl(0.,P,a,h,E,NU) >= g0: return np.nan
    f = lambda r: 2*np.pi*r/(g0-defl(r,P,a,h,E,NU))
    return EPS0*quad(f,0,a)[0]

def spiral_L(n=N,d_out=D_OUT,d_in=D_IN):
    d_avg=(d_out+d_in)/2; rho=(d_out-d_in)/(d_out+d_in)
    return K1*MU0*n**2*d_avg/(1+K2*rho)

def resonant_freq(P,L=None):
    L = L or spiral_L(); C=capacitance(P)
    return np.nan if np.isnan(C) else 1/(2*np.pi*np.sqrt(L*C))

# ---- Part 4: reader impedance ----
def Z_in(freq, P, L_reader=None, L_s=None, k=K, Rs=R_S):
    L_reader = L_reader or spiral_L()
    L_s = L_s or L_reader
    C = capacitance(P)
    if np.isnan(C): return np.nan
    w = 2*np.pi*freq
    M = k*np.sqrt(L_reader*L_s)
    Z_tank = Rs + 1j*w*L_s + 1/(1j*w*C)
    return 1j*w*L_reader + (w**2*M**2)/Z_tank

def find_dip(P, L=None):
    f0 = resonant_freq(P, L)
    if np.isnan(f0): return np.nan
    obj = lambda f: abs(Z_in(f, P, L))
    res = minimize_scalar(obj, bounds=(0.5*f0, 1.5*f0), method='bounded')
    return res.x

# ---- Run: |Z_in| vs freq at 3 pressures ----
L = spiral_L()
print(f"L = {L*1e9:.1f} nH")
pressures = [0, 20e3, 40e3]  # 50kPa touches; 40 is highest valid trio
colors = ["C0","C1","C3"]

plt.figure(figsize=(8,5))
print("\nDip vs analytic f0:")
for P,c in zip(pressures, colors):
    f0 = resonant_freq(P,L)
    if np.isnan(f0):
        print(f"  P={P/1e3:.0f} kPa: plates touch — skipped"); continue
    fr = np.linspace(0.5*f0, 1.5*f0, 2000)
    Z = np.array([abs(Z_in(f,P,L)) for f in fr])
    plt.plot(fr/1e6, Z, c, label=f"{P/1e3:.0f} kPa")
    fd = find_dip(P,L)
    err = 100*(fd-f0)/f0
    print(f"  P={P/1e3:.0f} kPa: dip={fd/1e6:.2f} MHz, f0={f0/1e6:.2f} MHz ({err:+.2f}%)")

plt.xlabel("Frequency [MHz]"); plt.ylabel("|Z_in| [Ω]")
plt.title("Reader impedance — dip shifts left as P rises")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("/home/claude/readout.png", dpi=130)
print("Saved readout.png")

# ---- k sweep: deeper dip? ----
print("\nCoupling k vs dip depth (P=0):")
f0=resonant_freq(0,L); fr=np.linspace(0.5*f0,1.5*f0,1000)
for k in [0.1,0.2,0.4]:
    Z=np.array([abs(Z_in(f,0,L,k=k)) for f in fr])
    depth=Z.max()-Z.min()
    print(f"  k={k}: depth={depth:.1f} Ω")

# ---- Design sweep ----
def sensitivity(a,h,g0):
    """Δf/ΔP (kHz/kPa) over 0..50kPa; linearity R². NaN if invalid."""
    L_ = spiral_L()
    if not (100e-9 <= L_ <= 1e-6): return None
    Ps = np.linspace(0,P_MAX,20); fs=[]
    for P in Ps:
        if defl(0.,P,a,h,E,NU) >= g0: return None  # touch
        C = EPS0*quad(lambda r:2*np.pi*r/(g0-defl(r,P,a,h,E,NU)),0,a)[0]
        fs.append(1/(2*np.pi*np.sqrt(L_*C)))
    fs=np.array(fs)
    slope=np.polyfit(Ps/1e3, fs/1e3, 1)[0]
    # R² of linear fit
    pred=np.polyval(np.polyfit(Ps/1e3,fs/1e3,1), Ps/1e3)
    ss=1-np.sum((fs/1e3-pred)**2)/np.sum((fs/1e3-fs.mean()/1e3)**2)
    return slope, ss

print("\nDesign sweep (a, h, g0):")
best=None
results=[]
for a in [400e-6,500e-6,600e-6]:
    for h in [8e-6,10e-6,12e-6]:
        for g0 in [3e-6,4e-6,5e-6]:
            r=sensitivity(a,h,g0)
            if r is None:
                results.append((a,h,g0,None,None)); continue
            slope,r2=r
            results.append((a,h,g0,slope,r2))
            # want large |slope| AND linear (r2>0.98)
            if r2>0.98 and (best is None or abs(slope)>abs(best[3])):
                best=(a,h,g0,slope,r2)

n_touch=sum(1 for x in results if x[3] is None)
print(f"  {n_touch}/{len(results)} combos invalid (plates touch / L out of range)")
print(f"\nBest (max |Δf/ΔP| with R²>0.98):")
a,h,g0,s,r2=best
print(f"  a={a*1e6:.0f}µm h={h*1e6:.0f}µm g0={g0*1e6:.0f}µm")
print(f"  Δf/ΔP={s:.1f} kHz/kPa, linearity R²={r2:.4f}")

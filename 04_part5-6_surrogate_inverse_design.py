"""
Part 5-6: Surrogate model + inverse design.
Reuses the Part 1-4 pipeline (deflection -> C -> f0 -> sensitivity) unchanged.
"""
import numpy as np, time, warnings
from scipy.integrate import quad
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---- constants ----
E, NU = 170e9, 0.28
EPS0, MU0 = 8.854e-12, 4*np.pi*1e-7
K1, K2 = 2.34, 2.75
P_MAX = 50e3

# ---- Part 1-4 pipeline (unchanged) ----
def flex_rig(h): return E*h**3/(12*(1-NU**2))
def defl(r,P,a,h): return P*a**4/(64*flex_rig(h))*(1-(r/a)**2)**2
def capacitance(P,a,h,g0):
    if defl(0.,P,a,h) >= g0: return np.nan
    return EPS0*quad(lambda r:2*np.pi*r/(g0-defl(r,P,a,h)),0,a)[0]
def spiral_L(n,d_out,d_in=1e-3):
    d_avg=(d_out+d_in)/2; rho=(d_out-d_in)/(d_out+d_in)
    return K1*MU0*n**2*d_avg/(1+K2*rho)

def simulate(a,h,g0,n,d_out):
    """geometry -> (sensitivity kHz/kPa, f0_baseline MHz, linearity R2). NaN if invalid."""
    L=spiral_L(n,d_out)
    if not (100e-9 <= L <= 1e-6): return (np.nan,)*3
    Ps=np.linspace(0,P_MAX,15); fs=[]
    for P in Ps:
        C=capacitance(P,a,h,g0)
        if np.isnan(C): return (np.nan,)*3     # plates touch
        fs.append(1/(2*np.pi*np.sqrt(L*C)))
    fs=np.array(fs)
    slope,int=np.polyfit(Ps/1e3, fs/1e3, 1)          # kHz/kPa
    pred=slope*(Ps/1e3)+int
    r2=1-np.sum((fs/1e3-pred)**2)/np.sum((fs/1e3-fs.mean()/1e3)**2)
    return slope, fs[0]/1e6, r2

# ---- ranges ----
NAMES=["a","h","g0","n","d_out"]
LO=np.array([400e-6, 8e-6, 3e-6, 12, 3e-3])
HI=np.array([600e-6,12e-6, 5e-6, 25, 5e-3])

# ==== STEP 1: data generation ====
def gen_data(N=500, seed=0):
    s=qmc.LatinHypercube(d=5,seed=seed).random(N)
    G=qmc.scale(s,LO,HI); G[:,3]=np.round(G[:,3])   # n integer
    X,Y=[],[]
    for g in G:
        y=simulate(*g)
        if not np.any(np.isnan(y)): X.append(g); Y.append(y)
    return np.array(X), np.array(Y)

print("Generating dataset...")
t=time.time(); X,Y=gen_data(500); tgen=time.time()-t
print(f"  {len(X)} valid / 500 sampled  ({tgen:.1f}s)")
sim_time=tgen/len(X)

hdr="a,h,g0,n,d_out,sensitivity_kHz_kPa,f0_MHz,linearity_R2"
np.savetxt("/home/claude/dataset.csv",np.hstack([X,Y]),delimiter=",",header=hdr,comments="")
print("  saved dataset.csv")

# ==== STEP 2: surrogate ====
Xtr,Xte,Ytr,Yte=train_test_split(X,Y,test_size=0.2,random_state=1)
sx,sy=StandardScaler().fit(Xtr),StandardScaler().fit(Ytr)
mlp=MLPRegressor(hidden_layer_sizes=(64,64,64),max_iter=3000,random_state=1)
mlp.fit(sx.transform(Xtr),sy.transform(Ytr))

Pred=sy.inverse_transform(mlp.predict(sx.transform(Xte)))
outn=["sensitivity","f0","linearity"]
print("\nSurrogate test metrics:")
r2s=[]
for i,o in enumerate(outn):
    r2=r2_score(Yte[:,i],Pred[:,i]); mae=mean_absolute_error(Yte[:,i],Pred[:,i])
    r2s.append(r2); print(f"  {o:12s} R2={r2:.3f}  MAE={mae:.3f}")
if min(r2s)<0.95: print("  WARNING: R2<0.95 on some output — inversion less reliable.")

def surrogate(g):
    return sy.inverse_transform(mlp.predict(sx.transform(np.atleast_2d(g))))[0]

# parity plots
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,o in enumerate(outn):
    ax[i].scatter(Yte[:,i],Pred[:,i],s=14,alpha=0.6)
    lo,hi=Yte[:,i].min(),Yte[:,i].max()
    ax[i].plot([lo,hi],[lo,hi],'r--',lw=1)
    ax[i].set(xlabel="true",ylabel="pred",title=f"{o}  (R²={r2s[i]:.3f})")
    ax[i].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("/home/claude/parity.png",dpi=130)
print("  saved parity.png")

# ==== speed comparison (batch = real use case) ====
B=2000; Xb=np.random.uniform(LO,HI,(B,5))
t=time.time(); sy.inverse_transform(mlp.predict(sx.transform(Xb))); sur_time=(time.time()-t)/B
print(f"\nSpeed (per eval): simulator {sim_time*1e3:.2f} ms | surrogate {sur_time*1e6:.2f} µs "
      f"| {sim_time/sur_time:.0f}x faster (batched)")

# ==== STEP 3: inverse design ====
def inverse(target, idx=0):
    """find geometry so surrogate output[idx] == target. Returns geometry."""
    tr=Y[:,idx]
    flag = "EXTRAPOLATION" if not (tr.min()<=target<=tr.max()) else "in-range"
    obj=lambda g: (surrogate(g)[idx]-target)**2
    best=None; np.random.seed(0)
    for _ in range(25):                          # multistart
        g0=np.random.uniform(LO,HI)
        r=minimize(obj,g0,bounds=list(zip(LO,HI)),method="L-BFGS-B")
        if np.isnan(simulate(*r.x)[0]): continue # reject invalid (plates touch)
        if best is None or r.fun<best.fun: best=r
    return (best.x if best else r.x), flag

print("\nInverse-design case studies (target sensitivity, kHz/kPa):")
cases=[-300,-450,-600]
for tgt in cases:
    g,flag=inverse(tgt,0)
    true=simulate(*g)[0]
    err=abs(true-tgt)/abs(tgt)*100
    gs=f"a={g[0]*1e6:.0f}µm h={g[1]*1e6:.1f}µm g0={g[2]*1e6:.1f}µm n={g[3]:.0f} d_out={g[4]*1e3:.1f}mm"
    print(f"  target {tgt:>5} -> {gs}")
    print(f"           surrogate={surrogate(g)[0]:.1f}  simulator={true:.1f}  err={err:.1f}%  [{flag}]")

print("\nDone.")

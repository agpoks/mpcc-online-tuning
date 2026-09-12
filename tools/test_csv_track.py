import sys, numpy as np
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
# --- reconstruct a Track from the CSV boundary polylines (real-track deployment path) ---
def from_csv(ds=0.1):
    bd=np.genfromtxt(TR/"icra_t2_smooth_boundaries.csv",delimiter=",",names=True)
    ref=np.column_stack([bd["ref_x"],bd["ref_y"]])
    L=np.column_stack([bd["left_x"],bd["left_y"]]); R=np.column_stack([bd["right_x"],bd["right_y"]])
    tang=np.gradient(ref,axis=0); tang/=(np.linalg.norm(tang,axis=1,keepdims=True)+1e-12)
    n=np.column_stack([-tang[:,1],tang[:,0]])   # +normal
    wr=np.einsum('ij,ij->i',L-ref,n)            # +normal wall dist  (left edge)
    wl=-np.einsum('ij,ij->i',R-ref,n)           # -normal wall dist  (right edge)
    t=Track(ref[:,0],ref[:,1],ds=ds,w_left=wl,w_right=wr)
    t.width_vehicle_adjusted=False; t.kv_max=0.55
    return t
tn=Track.icra_t2_smooth()   # npz
tc=from_csv()               # csv
# compare corridor edges from each track's own spline
def edges(t,n=1500):
    s=np.linspace(0,t.length,n,endpoint=False); L=[];R=[]
    for si in s:
        p=np.array(t.pos(float(si))).ravel(); a=float(t.tangent_angle(float(si)))
        nx,ny=-np.sin(a),np.cos(a); wl,wr=t.width(float(si))
        L.append([p[0]+nx*float(wr),p[1]+ny*float(wr)]); R.append([p[0]-nx*float(wl),p[1]-ny*float(wl)])
    return np.array(L),np.array(R),s
Ln,Rn,s=edges(tn); Lc,Rc,_=edges(tc)
print(f"npz  length={tn.length:.3f}  csv length={tc.length:.3f}")
print(f"corridor edge max gap npz-vs-csv:  left {np.hypot(*(Ln-Lc).T).max()*1000:.2f} mm, right {np.hypot(*(Rn-Rc).T).max()*1000:.2f} mm")
print(f"corridor edge MEAN gap:            left {np.hypot(*(Ln-Lc).T).mean()*1000:.2f} mm, right {np.hypot(*(Rn-Rc).T).mean()*1000:.2f} mm")

"""Reference for VALIDATION_PLAN Stage 2.7: isotropic thin annular plate,
clamped on the outer edge, free on the inner (sound-hole) edge.

Morley plate element (scikit-fem) on a structured polar triangulation,
refined until the first modes stop moving. Isotropic E = 10 GPa, nu = 0.3,
h = 3 mm, rho = 500, a = 0.2 m, inner radius 0.05 m.

    python tests/references/annulus_skfem.py   -> tests/references/annulus.json
"""
import json, math, os
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay
import skfem as fem
from skfem.helpers import dd, ddot, trace, eye

E, NU, H, RHO, A, RIN = 10e9, 0.3, 0.003, 500.0, 0.2, 0.05
D = E*H**3/(12*(1-NU**2))


def mesh(nr, nt):
    rs = np.linspace(RIN, A, nr + 1)
    pts = [(0.0, 0.0)]
    for r in rs:
        th = np.linspace(0, 2*np.pi, nt, endpoint=False) + (0.5*2*np.pi/nt if (r - RIN)/(A - RIN)*nr % 2 else 0)
        pts += [(r*math.cos(t), r*math.sin(t)) for t in th]
    P = np.array(pts[1:])
    tri = Delaunay(P).simplices
    c = P[tri].mean(axis=1)
    keep = np.hypot(c[:, 0], c[:, 1]) > RIN*1.0001
    return fem.MeshTri(P.T.copy(), tri[keep].T.copy())


def solve(nr, nt, k=6):
    m = mesh(nr, nt)
    e = fem.ElementTriMorley()
    ib = fem.Basis(m, e)

    @fem.BilinearForm
    def stiff(u, v, w):
        def C(T):
            return E*H**3/(12*(1-NU**2)) * ((1-NU)*T + NU*eye(trace(T), 2))
        return ddot(C(dd(u)), dd(v))

    @fem.BilinearForm
    def mass(u, v, w):
        return RHO*H*u*v

    K = fem.asm(stiff, ib); M = fem.asm(mass, ib)
    outer = m.facets_satisfying(lambda x: np.hypot(x[0], x[1]) > A*0.999)
    fixed = ib.get_dofs(outer).all()
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    vals = eigsh(K[free][:, free], k=k, M=M[free][:, free], sigma=1.0, which='LM')[0]
    return sorted(float(np.sqrt(v)/(2*np.pi)) for v in vals)


if __name__ == '__main__':
    runs = []
    for nr, nt in [(12, 96), (24, 192), (48, 384)]:
        f = solve(nr, nt)
        runs.append({'nr': nr, 'nt': nt, 'freqs': f})
        print(nr, nt, [round(x, 3) for x in f], flush=True)
    out = {'note': __doc__.strip().splitlines()[0], 'E': E, 'nu': NU, 'h': H, 'rho': RHO, 'a': A, 'r_in': RIN,
           'runs': runs, 'freqs': runs[-1]['freqs']}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annulus.json'), 'w'), indent=1)

"""Independent 3D solid finite element reference for a straight brace glued
to a clamped rectangular plate (Stage 6 / Stage 7 stand-in).

Quadratic serendipity hexes (scikit-fem ElementHexS2) on a tensor mesh, with
the plate orthotropic exactly as the ANSYS runs used it (L along x, R along
y, T along z) and the brace isotropic. Fixed support on the plate's four
side faces only, like ANSYS. No beam, flange or footprint assumptions: the
brace is just more solid elements sharing nodes with the plate (perfect
bond, i.e. shared topology).

    python tests/references/solid3d.py --h 2.5
"""
import argparse, json, math, sys
import numpy as np
from scipy.sparse.linalg import eigsh
import skfem as fem
from skfem.helpers import sym_grad

SPRUCE_ANSYS = dict(E1=9.79e9, E2=1.25e9, E3=0.58e9, nu12=0.422, nu13=0.422, nu23=0.422,
                    G12=1.21e9, G13=0.6e9, G23=0.04e9, rho=350.0)


def ortho_C(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, **_):
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -nu12/E1
    S[0, 2] = S[2, 0] = -nu13/E1
    S[1, 2] = S[2, 1] = -nu23/E2
    S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12   # Voigt order xx yy zz yz xz xy
    return np.linalg.inv(S)


def iso_C(E, nu):
    return ortho_C(E, E, E, nu, nu, nu, E/(2*(1+nu)), E/(2*(1+nu)), E/(2*(1+nu)))


def stiffness_form(C):
    @fem.BilinearForm
    def a(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        e = [eu[0, 0], eu[1, 1], eu[2, 2], 2*eu[1, 2], 2*eu[0, 2], 2*eu[0, 1]]
        f = [ev[0, 0], ev[1, 1], ev[2, 2], 2*ev[1, 2], 2*ev[0, 2], 2*ev[0, 1]]
        out = 0
        for i in range(6):
            s = sum(C[i, j]*e[j] for j in range(6))
            out = out + s*f[i]
        return out
    return a


def mass_form(rho):
    @fem.BilinearForm
    def m(u, v, w):
        return rho*(u[0]*v[0] + u[1]*v[1] + u[2]*v[2])
    return m


BRACE_MAT = dict(E=9.79e9, nu=0.4, rho=350.0)
BRIDGE_MAT = dict(E=12e9, nu=0.3, rho=850.0)


def solve(a=0.300, b=0.200, tp=0.0028, boxes=(), h=0.005, nz_plate=1, layer=0.004, n_modes=6,
          plate=SPRUCE_ANSYS):
    """boxes: list of dict(x0, x1, y0, y1, z0, z1, E, nu, rho) in metres,
    glued to the plate (z0 >= tp for braces on the inside face, z1 <= 0 for
    the bridge on the outside face). Same-material boxes may overlap."""
    xs = list(np.linspace(0, a, int(round(a/h)) + 1))
    ys = list(np.linspace(0, b, int(round(b/h)) + 1))
    zs = list(np.linspace(0, tp, nz_plate + 1))
    for bx in boxes:
        xs += [bx['x0'], bx['x1']]; ys += [bx['y0'], bx['y1']]
        nz = max(1, int(math.ceil((bx['z1'] - bx['z0'])/layer)))
        zs += list(np.linspace(bx['z0'], bx['z1'], nz + 1))
    xs, ys, zs = (np.unique(np.round(np.array(v), 9)) for v in (xs, ys, zs))
    m = fem.MeshHex.init_tensor(xs, ys, zs)
    mid = m.p[:, m.t].mean(axis=1)
    owner = np.where((mid[2] > 0) & (mid[2] < tp), 0, -1)
    for k, bx in enumerate(boxes):
        inb = ((mid[0] > bx['x0']) & (mid[0] < bx['x1']) & (mid[1] > bx['y0']) & (mid[1] < bx['y1'])
               & (mid[2] > bx['z0']) & (mid[2] < bx['z1']))
        owner = np.where(inb & (owner < 0), k + 1, owner)
    keep = np.nonzero(owner >= 0)[0]
    m = m.restrict(keep); owner = owner[keep]
    e = fem.ElementVector(fem.ElementHexS2())
    ib = fem.Basis(m, e, elements=np.nonzero(owner == 0)[0], intorder=4)
    K = fem.asm(stiffness_form(ortho_C(**plate)), ib)
    M = fem.asm(mass_form(plate['rho']), ib)
    groups = {}
    for k, bx in enumerate(boxes):
        groups.setdefault((bx['E'], bx['nu'], bx['rho']), []).append(k + 1)
    for (E, nu, rho), ks in groups.items():
        els = np.nonzero(np.isin(owner, ks))[0]
        ibk = fem.Basis(m, e, elements=els, intorder=4)
        K = K + fem.asm(stiffness_form(iso_C(E, nu)), ibk)
        M = M + fem.asm(mass_form(rho), ibk)
    basis = fem.Basis(m, e)
    tol = 1e-9
    # fixed support on the plate's own side faces (not on brace or bridge faces)
    fixed = basis.get_dofs(lambda x: ((np.abs(x[0]) < tol) | (np.abs(x[0] - a) < tol) |
                                      (np.abs(x[1]) < tol) | (np.abs(x[1] - b) < tol))
                           & (x[2] > -tol) & (x[2] < tp + tol)).all()
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    Kf = K[free][:, free]; Mf = M[free][:, free]
    vals = eigsh(Kf, k=n_modes, M=Mf, sigma=(2*math.pi*10)**2, which='LM')[0]
    return sorted(float(np.sqrt(v)/(2*math.pi)) for v in vals), int(len(free))


def brace_box(axis, c, w, hb, s0, s1, tp=0.0028, mat=BRACE_MAT):
    if axis == 'x':
        return dict(x0=s0, x1=s1, y0=c - w/2, y1=c + w/2, z0=tp, z1=tp + hb, **mat)
    return dict(x0=c - w/2, x1=c + w/2, y0=s0, y1=s1, z0=tp, z1=tp + hb, **mat)


def bridge_box(cx, cy, depth, length, hb, mat=BRIDGE_MAT):
    return dict(x0=cx - depth/2, x1=cx + depth/2, y0=cy - length/2, y1=cy + length/2, z0=-hb, z1=0.0, **mat)


# The rectangle cases (mm): plate 300 x 200 x 2.8, clamped on its side faces.
CASES = {
    'bare': [],
    'brace_x': [brace_box('x', 0.100, 0.010, 0.012, 0.020, 0.280)],
    'brace_y': [brace_box('y', 0.150, 0.010, 0.012, 0.020, 0.180)],
    'bridge': [bridge_box(0.150, 0.100, 0.030, 0.150, 0.009)],
    'parallel25': [brace_box('x', 0.0875, 0.010, 0.012, 0.020, 0.280), brace_box('x', 0.1125, 0.010, 0.012, 0.020, 0.280)],
    'cross': [brace_box('x', 0.100, 0.010, 0.012, 0.020, 0.280), brace_box('y', 0.150, 0.010, 0.012, 0.020, 0.180)],
    'brace_bridge': [brace_box('x', 0.070, 0.010, 0.012, 0.020, 0.280), bridge_box(0.150, 0.100, 0.030, 0.150, 0.009)],
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, default=5.0, help='in-plane element size, mm')
    ap.add_argument('--nzp', type=int, default=1)
    ap.add_argument('--cases', default=','.join(CASES))
    a = ap.parse_args()
    out = {}
    for name in a.cases.split(','):
        f, n = solve(h=a.h/1000, boxes=CASES[name], nz_plate=a.nzp)
        out[name] = {'h_mm': a.h, 'nz_plate': a.nzp, 'dofs': n, 'freqs': f}
        print(name, json.dumps(out[name]), flush=True)

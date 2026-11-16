"""Shared helpers for the solver validation suite (see VALIDATION_PLAN.md).

Everything goes through `solve`, so a change to evaluate_design's signature
only has to be absorbed here."""
import json, math, os, sys, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'desktop'))
import engine  # noqa: E402

FIX = os.path.join(HERE, 'fixtures')
GOLD = os.path.join(HERE, 'golden')
OUT = os.path.join(HERE, 'out')
GNS = [34, 50, 80, 120, 160, 200]


# ── Geometry / payload builders ─────────────────────────────────────────
def payload_geo(G, GN, targets=None):
    """The geo dict exactly as the UI's buildGeoPayload sends it."""
    if G.get('bridgePts') and len(G['bridgePts']) >= 3:
        bfx = float(np.mean([p['fx'] for p in G['bridgePts']])); bfy = float(np.mean([p['fy'] for p in G['bridgePts']]))
    else:
        bfx, bfy = G.get('bridge', {'fx': 0.63})['fx'], G.get('bridge', {'fy': 0.5})['fy']
    geo = {
        'Lx': G['Lx'], 'Ly': G['Ly'], 'h': G['h'], 'E': G['E'], 'Ey': G.get('Ey', G['E']), 'Gxy': G.get('Gxy'),
        'nu': G['nu'], 'nuxy': G['nu'], 'rho': G['rho'], 'Gxz': G.get('Gxz'), 'Gyz': G.get('Gyz'),
        'plateShear': G.get('plateShear', True) is not False, 'bcMode': G['bcMode'],
        'braceShearModel': G.get('braceShearModel', 'iso'), 'outline': G['outlinePts'],
        'holeCx': G['holeCx'], 'holeCy': G['holeCy'], 'holeR': G['holeR'],
        'bridgeFx': bfx, 'bridgeFy': bfy, 'loadN': G.get('loadN', 300), 'bridgeTorqueNmm': G.get('bridgeTorqueNmm', 0),
        'GN': GN, 'bridgePts': G.get('bridgePts') or [], 'bridgeLenMm': G.get('bridgeLenMm', 155),
        'bridgeDepthMm': G.get('bridgeDepthMm', 28), 'bridgeStructural': G.get('bridgeStructural', True) is not False,
        'bridgeHeightMm': G.get('bridgeHeightMm', 9), 'bridgeE': G.get('bridgeE', 12e9), 'bridgeRho': G.get('bridgeRho', 850),
        'braceSegMm': 10, 'targetFreqs': list(targets or []),
    }
    if geo['Gxy'] is None:
        geo.pop('Gxy')
    return geo


def load_fixture(name):
    with open(os.path.join(FIX, name + '.json')) as f:
        return json.load(f)


def circle_outline(n=720, cx=0.5, cy=0.5, r=0.48):
    """Outline points (fractions of a square bounding box) for a circle."""
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [{'fx': float(cx + r*np.cos(t)), 'fy': float(cy + r*np.sin(t))} for t in th]


def square_outline(angle_deg=0.0, n_side=60, half=0.30):
    """A square of side 2*half (fraction of the box) rotated about the centre."""
    a = math.radians(angle_deg); c, s = math.cos(a), math.sin(a)
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    pts = []
    for k in range(4):
        x0, y0 = corners[k]; x1, y1 = corners[(k+1) % 4]
        for t in np.linspace(0, 1, n_side, endpoint=False):
            x, y = x0 + (x1-x0)*t, y0 + (y1-y0)*t
            pts.append({'fx': 0.5 + c*x - s*y, 'fy': 0.5 + s*x + c*y})
    return pts


def iso_geo(outline, L, bc, GN, E=10e9, nu=0.3, h=0.003, rho=500.0, shear=False, hole_r=0.0, hole=(0.5, 0.5)):
    """Isotropic test plate in an L x L box, no bridge, tiny load."""
    G = E/(2*(1+nu))
    geo = {'Lx': L, 'Ly': L, 'h': h, 'E': E, 'Ey': E, 'Gxy': G, 'nu': nu, 'nuxy': nu, 'rho': rho,
           'Gxz': G if shear else None, 'Gyz': G if shear else None, 'plateShear': shear,
           'bcMode': bc, 'braceShearModel': 'iso', 'outline': outline,
           'holeCx': hole[0], 'holeCy': hole[1], 'holeR': hole_r,
           'bridgeFx': 0.5, 'bridgeFy': 0.5, 'loadN': 1.0, 'bridgeTorqueNmm': 0.0, 'GN': GN,
           'bridgePts': [], 'bridgeLenMm': 10, 'bridgeDepthMm': 10, 'bridgeStructural': False,
           'braceSegMm': 10, 'targetFreqs': []}
    return geo


def lam2_to_hz(lam2, a, E=10e9, nu=0.3, h=0.003, rho=500.0):
    D = E*h**3/(12*(1-nu**2))
    return lam2*math.sqrt(D/(rho*h))/(2*math.pi*a*a)


def hz_to_lam2(f, a, E=10e9, nu=0.3, h=0.003, rho=500.0):
    return f/lam2_to_hz(1.0, a, E, nu, h, rho)


# ── Solve wrapper ───────────────────────────────────────────────────────
def solve(geo, braces=(), n=10, field=False):
    """All modes (lowest n, above 5 Hz) plus deflection and masses."""
    geo = dict(geo); geo['targetFreqs'] = []
    r = engine.evaluate_design(geo, list(braces), n, want_field=field)
    if not r.get('ok'):
        raise RuntimeError(r.get('reason'))
    modes = r['modes']
    out = {'freqs': [m['freq'] for m in modes], 'strength': [m['strength'] for m in modes],
           'defl': r['deflBridgeMm'], 'massG': r['massG'], 'bridgeMassG': r['bridgeMassG'], 'nNodes': r['nNodes']}
    if field:
        out['shapes'] = [np.asarray(s) for s in r['fieldShapes']]
    return out


def pair_by_mac(shapes_a, shapes_b, min_mac=0.0):
    """Greedy one-to-one pairing of two shape lists by MAC (highest first).
    Returns {index_a: (index_b, mac)}."""
    M = np.array([[engine.mac(a, b) for b in shapes_b] for a in shapes_a])
    pairs = {}; used_a, used_b = set(), set()
    for flat in np.argsort(-M, axis=None):
        i, j = np.unravel_index(flat, M.shape)
        if i in used_a or j in used_b or M[i, j] < min_mac:
            continue
        pairs[int(i)] = (int(j), float(M[i, j])); used_a.add(i); used_b.add(j)
    return pairs


def richardson(gns, fs):
    """Fit f(GN) = f_inf + C*GN^-p over the given grids (least squares on p).
    Returns (f_inf, p). Falls back to p = 1 when the fit is unstable."""
    gns = np.asarray(gns, float); fs = np.asarray(fs, float)
    best = None
    for p in np.linspace(0.5, 4.0, 141):
        A = np.stack([np.ones_like(gns), gns**-p], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, fs, rcond=None)
        r = float(np.sum((A@coef - fs)**2))
        if best is None or r < best[0]:
            best = (r, p, coef[0])
    _, p, finf = best
    if not (0.6 <= p <= 3.9):
        p = 1.0
        A = np.stack([np.ones_like(gns), gns**-p], axis=1)
        finf = np.linalg.lstsq(A, fs, rcond=None)[0][0]
    return float(finf), float(p)


def gci(f_fine, f_coarse, r, p, fs=1.25):
    return fs*abs(f_fine - f_coarse)/abs(f_fine)/(r**p - 1.0)


def ladder_braces(G, gap_mm=25.0, n=7, x0=0.52, h=10.0, w=6.0):
    """Transverse braces about gap_mm apart across the lower bout."""
    Lx = G['Lx']
    out = []
    for k in range(n):
        fx = x0 + k*gap_mm/1000.0/Lx
        out.append(engine.make_brace(fx, 0.1, fx, 0.9, h, w, modulus=9.79e9, density=350.0, shear=0.65e9))
    return engine._clip_braces_to_outline(out, {'outline': G['outlinePts'], 'Lx': G['Lx'], 'Ly': G['Ly'],
                                                'holeCx': G['holeCx'], 'holeCy': G['holeCy'], 'holeR': G['holeR']})

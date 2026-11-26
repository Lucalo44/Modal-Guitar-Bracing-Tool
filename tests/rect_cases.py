"""The tool-side twin of tests/references/solid3d.py's rectangle cases: a
300 x 200 x 2.8 mm clamped spruce plate (ANSYS material) with braces and a
bridge placed exactly where the 3D solid has them."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import engine, solve, pair_by_mac, HERE  # noqa: E402
import numpy as np

A, B = 0.300, 0.200
REF = os.path.join(HERE, 'references', 'solid3d_rect.json')


def rect_outline(n=200):
    pts = []
    for k in range(4):
        for t in np.linspace(0, 1, n, endpoint=False):
            x, y = [(t, 0), (1, t), (1-t, 1), (0, 1-t)][k]
            pts.append({'fx': float(x), 'fy': float(y)})
    return pts


def rect_geo(GN, **flags):
    g = {'Lx': A, 'Ly': B, 'h': 0.0028, 'E': 9.79e9, 'Ey': 1.25e9, 'Gxy': 1.21e9, 'nu': 0.422, 'nuxy': 0.422,
         'rho': 350.0, 'Gxz': 0.6e9, 'Gyz': 0.04e9, 'plateShear': True, 'bcMode': 'clamped',
         'braceShearModel': 'iso', 'outline': rect_outline(), 'holeCx': 0.5, 'holeCy': 0.5, 'holeR': 0.0,
         'bridgeFx': 0.5, 'bridgeFy': 0.5, 'loadN': 1.0, 'bridgeTorqueNmm': 0, 'GN': GN, 'bridgePts': [],
         'bridgeLenMm': 150, 'bridgeDepthMm': 30, 'bridgeStructural': False, 'bridgeHeightMm': 9,
         'bridgeE': 12e9, 'bridgeRho': 850.0, 'braceSegMm': 10, 'targetFreqs': []}
    g.update(flags)
    return g


def brace(axis, c, s0, s1, w=10, hb=12):
    if axis == 'x':
        return engine.make_brace(s0/A, c/B, s1/A, c/B, hb, w, modulus=9.79e9, density=350.0, shear=3.5e9)
    return engine.make_brace(c/A, s0/B, c/A, s1/B, hb, w, modulus=9.79e9, density=350.0, shear=3.5e9)


def bridge(g, cx, cy):
    g = dict(g); g['bridgeStructural'] = True
    g['bridgeFx'], g['bridgeFy'] = cx/A, cy/B
    return g


def case(name, GN, **flags):
    g = rect_geo(GN, **flags)
    if name == 'bare':
        return g, []
    if name == 'brace_x':
        return g, [brace('x', 0.100, 0.020, 0.280)]
    if name == 'brace_y':
        return g, [brace('y', 0.150, 0.020, 0.180)]
    if name == 'bridge':
        return bridge(g, 0.150, 0.100), []
    if name == 'parallel25':
        return g, [brace('x', 0.0875, 0.020, 0.280), brace('x', 0.1125, 0.020, 0.280)]
    if name == 'cross':
        return g, [brace('x', 0.100, 0.020, 0.280), brace('y', 0.150, 0.020, 0.180)]
    if name == 'brace_bridge':
        return bridge(g, 0.150, 0.100), [brace('x', 0.070, 0.020, 0.280)]
    raise KeyError(name)


def reference():
    with open(REF) as f:
        return json.load(f)['cases']


def compare(GN=100, n=6, **flags):
    ref = reference()
    out = {}
    for name, r in ref.items():
        g, br = case(name, GN, **flags)
        t = solve(g, br, n=n)['freqs'][:n]
        out[name] = {'tool': t, 'ref': r['freqs'][:n], 'err': [100*(a/b - 1) for a, b in zip(t, r['freqs'])]}
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gn', type=int, default=100)
    ap.add_argument('--flags', default='')
    a = ap.parse_args()
    flags = {}
    for kv in filter(None, a.flags.split(',')):
        k, v = kv.split('=')
        flags[k] = {'true': True, 'false': False}.get(v, v)
    res = compare(a.gn, **flags)
    for name, r in res.items():
        print(f'{name:13s} ' + ' '.join(f'{e:+6.1f}' for e in r['err']) +
              f'   | mean |err| {np.mean(np.abs(r["err"])):.1f}%')

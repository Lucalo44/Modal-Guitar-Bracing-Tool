"""VALIDATION_PLAN.md, stages 1-9, as one runner.

    python tests/validation.py --stages 1,2,4,5      # fast stages
    python tests/validation.py --stages 3,6,7,8,9    # slow (needs golden/v2.json for 3)

Every check appends {stage, check, measured, limit, pass, detail} to
tests/out/results.json (merged by check name), which tests/test_validation.py
asserts on and tests/make_report.py turns into docs/validation.md."""
import argparse, json, math, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (engine, solve, load_fixture, payload_geo, circle_outline, square_outline, iso_geo,
                     lam2_to_hz, hz_to_lam2, richardson, gci, pair_by_mac, GOLD, OUT, ROOT)  # noqa: E402
import rect_cases  # noqa: E402

RESULTS = os.path.join(OUT, 'results.json')
REC = []


def record(stage, check, measured, limit, ok, detail=''):
    REC.append({'stage': stage, 'check': check, 'measured': measured, 'limit': limit, 'pass': bool(ok), 'detail': detail})
    print(f"[{'PASS' if ok else 'FAIL'}] S{stage} {check}: {measured} (limit {limit}) {detail}", flush=True)


def save():
    os.makedirs(OUT, exist_ok=True)
    old = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {'checks': []}
    stages = {r['stage'] for r in REC}
    keep = [r for r in old['checks'] if r['stage'] not in stages]
    import subprocess
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        commit = '?'
    json.dump({'commit': commit, 'checks': keep + REC, 'extra': {**old.get('extra', {}), **EXTRA}},
              open(RESULTS, 'w'), indent=1)


EXTRA = {}
pct = lambda a, b: 100.0*(a/b - 1.0)
SPRUCE = dict(E=9.79e9, Ey=1.25e9, Gxy=1.21e9, nu=0.422, rho=350.0, Gxz=0.6e9, Gyz=0.04e9)


# ── Stage 1: closed forms on a simply supported orthotropic rectangle ──
def navier(a, b, h, mat, shear, n=6, mmax=8):
    E1, E2, G12, nu = mat['E'], mat['Ey'], mat['Gxy'], mat['nu']
    nyx = nu*E2/E1; den = 1 - nu*nyx
    D11, D22, D12, D66 = E1*h**3/(12*den), E2*h**3/(12*den), nyx*E1*h**3/(12*den), G12*h**3/12
    rho = mat['rho']; k = 5/6
    out = []
    for m in range(1, mmax):
        for q in range(1, mmax):
            al, be = m*math.pi/a, q*math.pi/b
            if not shear:
                w2 = (D11*al**4 + 2*(D12 + 2*D66)*al**2*be**2 + D22*be**4)/(rho*h)
            else:
                Gx, Gy = k*mat['Gxz']*h, k*mat['Gyz']*h
                K = np.array([[Gx*al**2 + Gy*be**2, Gx*al, Gy*be],
                              [Gx*al, D11*al**2 + D66*be**2 + Gx, (D12 + D66)*al*be],
                              [Gy*be, (D12 + D66)*al*be, D66*al**2 + D22*be**2 + Gy]])
                Mm = np.diag([rho*h, rho*h**3/12, rho*h**3/12])
                w2 = min(np.linalg.eigvals(np.linalg.solve(Mm, K)).real)
            out.append(math.sqrt(w2)/(2*math.pi))
    return sorted(out)[:n]


def rect_ss_geo(GN, h, shear):
    g = rect_cases.rect_geo(GN, bcMode='ss', h=h, plateShear=shear)
    return g


def stage1():
    a, b = rect_cases.A, rect_cases.B
    for label, h, shear, tol in [('ss rectangle thin plate', 0.0028, False, 0.3),
                                 ('ss rectangle 2.8mm Mindlin', 0.0028, True, 0.3),
                                 ('ss rectangle 10mm Mindlin', 0.010, True, 1.5)]:
        ref = navier(a, b, h, SPRUCE, shear)
        g = rect_ss_geo(120, h, shear)
        t = solve(g, [], n=6)['freqs'][:6]
        err = [pct(x, y) for x, y in zip(t, ref)]
        record(1, label, f'max |err| {max(map(abs, err)):.2f}%', f'{tol}%', max(map(abs, err)) <= tol,
               'tool ' + ', '.join(f'{x:.1f}' for x in t) + ' | exact ' + ', '.join(f'{x:.1f}' for x in ref))
    # static: central 20x20 mm patch load, thin plate, load-weighted mean deflection over the patch
    h = 0.0028; mat = SPRUCE
    E1, E2, G12, nu = mat['E'], mat['Ey'], mat['Gxy'], mat['nu']
    nyx = nu*E2/E1; den = 1 - nu*nyx
    D11, D22, D12, D66 = E1*h**3/(12*den), E2*h**3/(12*den), nyx*E1*h**3/(12*den), G12*h**3/12
    P, c = 10.0, 0.010
    def wmean():
        s = 0.0
        for m in range(1, 200, 2):
            for q in range(1, 200, 2):
                al, be = m*math.pi/a, q*math.pi/b
                qmn = 16*P/(a*b)/(4*c*c) * math.sin(m*math.pi/2)*math.sin(al*c)/al * math.sin(q*math.pi/2)*math.sin(be*c)/be
                Wmn = qmn/(D11*al**4 + 2*(D12 + 2*D66)*al**2*be**2 + D22*be**4)
                # mean of sin(al x) sin(be y) over the patch
                avg = (math.sin(m*math.pi/2)*2*math.sin(al*c)/(al*2*c))*(math.sin(q*math.pi/2)*2*math.sin(be*c)/(be*2*c))
                s += Wmn*avg
        return s*1000
    ref = wmean()
    worst = 0
    for GN in (34, 80):
        g = rect_ss_geo(GN, h, False)
        g.update(loadN=P, bridgePts=[{'fx': (a/2 + dx)/a, 'fy': (b/2 + dy)/b} for dx, dy in ((-c, -c), (c, -c), (c, c), (-c, c))])
        d = solve(g, [], n=4)['defl']
        worst = max(worst, abs(pct(d, ref)))
    record(1, 'ss rectangle static patch load', f'max |err| {worst:.2f}% (GN 34, 80)', '1%', worst <= 1.0, f'exact {ref:.4f} mm')


# ── Stage 2: exact plate solutions for the edge fixes ──
A_C = 0.2
L_C = 2*A_C/0.96


def circle_modes(bc, GN, nu=0.3, n=8, hole=0.0):
    g = iso_geo(circle_outline(), L_C, bc, GN, nu=nu, hole_r=hole/L_C)
    r = solve(g, [], n=n)
    return [hz_to_lam2(f, A_C, nu=nu) for f in r['freqs']], r


def stage2():
    ref = [10.2158, 21.260, 21.260, 34.877, 34.877, 39.771]
    rows = {}
    for GN in (34, 50, 80, 120):
        lam, _ = circle_modes('clamped', GN)
        rows[GN] = [pct(x, y) for x, y in zip(lam, ref)]
    ok = all(max(map(abs, rows[g])) <= 1.0 for g in (50, 80, 120)) and abs(rows[34][0]) <= 1.0
    record(2, 'clamped circle vs Leissa', f"GN34 f1 {rows[34][0]:+.2f}%, GN50+ max {max(max(map(abs, rows[g])) for g in (50, 80, 120)):.2f}%",
           'f1 1% at GN 34, all 1% at GN>=50', ok, json.dumps({g: [round(e, 2) for e in v] for g, v in rows.items()}))
    EXTRA['clamped_circle'] = rows
    # clamped square, unrotated and rotated
    sq_ref = [35.985, 73.394, 73.394, 108.22, 131.58, 132.20]
    side = 0.6  # square side as a fraction of the box (square_outline half=0.3)
    L = 0.4; a_sq = side*L
    res = {}
    for ang in (0, 30, 45):
        for GN in (50, 80):
            g = iso_geo(square_outline(ang, n_side=120), L, 'clamped', GN)
            f = solve(g, [], n=8)['freqs'][:6]
            res[(ang, GN)] = [hz_to_lam2(x, a_sq) for x in f]
    err0 = [max(abs(pct(x, y)) for x, y in zip(res[(0, GN)], sq_ref)) for GN in (50, 80)]
    record(2, 'clamped square vs Leissa', f'max |err| {max(err0):.2f}% (GN 50, 80)', '1%', max(err0) <= 1.0,
           'GN80 lambda^2 ' + ', '.join(f'{x:.2f}' for x in res[(0, 80)]))
    rot = [max(abs(pct(x, y)) for x, y in zip(res[(ang, GN)], res[(0, GN)])) for ang in (30, 45) for GN in (50, 80)]
    record(2, 'rotated square vs unrotated (staircase)', f'max |diff| {max(rot):.2f}%', '1%', max(rot) <= 1.0)
    # free circle, nu 0.33
    fr = [5.253, 5.253, 9.084, 12.23, 12.23]
    worst, rigid = 0, []
    for GN in (80, 120):
        lam, r = circle_modes('free', GN, nu=0.33)
        worst = max(worst, max(abs(pct(x, y)) for x, y in zip(lam, fr)))
        g = iso_geo(circle_outline(), L_C, 'free', GN, nu=0.33); g['targetFreqs'] = []
        rigid.append(engine.evaluate_design(g, [], 8)['nBelow5Hz'])
    record(2, 'free circle vs Leissa', f'max |err| {worst:.2f}%, modes under 5 Hz {rigid}', '1.5% at GN>=80, exactly 3 rigid',
           worst <= 1.5 and all(x == 3 for x in rigid))
    worst = 0
    for GN in (50, 80, 120):
        lam, _ = circle_modes('ss', GN)
        worst = max(worst, abs(pct(lam[0], 4.935)))
    record(2, 'simply supported circle vs Leissa', f'|err| {worst:.2f}%', '1% at GN>=50', worst <= 1.0)
    # static uniform pressure, clamped circle: w0 = q a^4 / 64 D
    E, nu, h = 10e9, 0.3, 0.003
    D = E*h**3/(12*(1-nu**2)); q = 100.0
    w0 = q*A_C**4/(64*D)*1000
    worst = 0
    for GN in (50, 80, 120):
        g = iso_geo(circle_outline(), L_C, 'clamped', GN)
        e = 0.001/L_C
        g.update(loadN=0.0, pressurePa=q, bridgePts=[{'fx': 0.5 + dx*e, 'fy': 0.5 + dy*e} for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        d = solve(g, [], n=4)['defl']
        worst = max(worst, abs(pct(d, w0)))
    record(2, 'clamped circle static uniform pressure', f'max |err| {worst:.2f}%', '1.5% at GN>=50', worst <= 1.5, f'exact {w0:.4f} mm')
    # annulus: clamped outside, free hole
    ref = json.load(open(os.path.join(os.path.dirname(__file__), 'references', 'annulus.json')))
    worst = 0
    for GN in (80, 120):
        g = iso_geo(circle_outline(), L_C, 'clamped', GN, hole_r=ref['r_in']/L_C)
        f = solve(g, [], n=8)['freqs'][:6]
        worst = max(worst, max(abs(pct(x, y)) for x, y in zip(f, ref['freqs'])))
    record(2, 'annular plate (free hole) vs Morley FE', f'max |err| {worst:.2f}%', '2% at GN>=80', worst <= 2.0)


# ── Stage 3: convergence on real designs (from golden/v2.json) ──
BARE = ('bare_clamped', 'bare_free', 'bare_ss')


def stage3(tag='v2'):
    G = json.load(open(os.path.join(GOLD, tag + '.json')))['fixtures']
    summary = {}
    for name, rec in G.items():
        gns = sorted(int(k) for k in rec)
        if gns[-1] < 200:
            continue
        F = {g: rec[str(g)]['freqs'] for g in gns}
        n = min(len(F[g]) for g in gns)
        # follow modes by order (bare tops: modes don't swap); cap at modes under 400 Hz for bare
        if name in BARE:
            idx = [i for i in range(n) if F[200][i] < 400]
            ch = [abs(pct(F[200][i], F[80][i])) for i in idx]
            record(3, f'{name}: GN 80->200 change, modes < 400 Hz', f'max {max(ch):.2f}%', '2%', max(ch) <= 2.0)
            summary[name] = {'max_change_80_200': max(ch)}
            continue
        idx = list(range(min(3, n)))
        g_list = [g for g in gns if g >= 80]
        gcis, gaps, rev = [], [], 0
        for i in idx:
            fs = [F[g][i] for g in gns]
            finf, p = richardson(g_list, [F[g][i] for g in g_list])
            gcis.append(100*gci(F[200][i], F[160][i], 200/160, p))
            gaps.append(abs(pct(F[200][i], finf)))
            d = np.sign(np.diff(fs)); d = d[d != 0]
            rev = max(rev, int(np.sum(d[1:] != d[:-1])))
        dd = [rec[str(g)]['defl'] for g in (50, 200)]
        dchg = abs(pct(dd[1], dd[0]))
        spread = max((max(F[g][i] for g in gns) - min(F[g][i] for g in gns))/F[200][i]*100 for i in idx)
        b1 = json.load(open(os.path.join(GOLD, 'baseline.json')))['fixtures'].get(name, {})
        spread1 = max((max(b1[str(g)]['freqs'][i] for g in gns) - min(b1[str(g)]['freqs'][i] for g in gns))/b1['200']['freqs'][i]*100
                      for i in idx) if b1 else float('nan')
        ok = max(gcis) < 2.0 and max(gaps) < 1.5 and rev <= 1 and dchg < 3.0
        record(3, f'{name}: convergence (modes 1-3)',
               f'GCI {max(gcis):.2f}%, extrapolation gap {max(gaps):.2f}%, reversals {rev}, deflection 50->200 {dchg:.1f}%',
               'GCI<2%, gap<1.5%, <=1 reversal, defl<3%', ok,
               f'spread over GN 34-200: {spread:.1f}% (v1: {spread1:.1f}%)')
        summary[name] = {'gci': max(gcis), 'gap': max(gaps), 'reversals': rev, 'defl_change': dchg, 'spread': spread, 'spread_v1': spread1}
    EXTRA['stage3'] = summary
    EXTRA['golden_' + tag] = {k: {g: v[g]['freqs'][:6] for g in v} for k, v in G.items()}


# ── Stage 4: invariance and smoothness ──
def stage4():
    fx = load_fixture('tiptest')
    base = None; worst = 0
    for off in [(0, 0), (0.1, 0.1), (0.25, 0.0), (0.5, 0.5)]:
        engine.GRID_OFFSET = off
        f = solve(payload_geo(fx['GEO'], 80), fx['braces'], n=6)['freqs'][:3]
        if base is None:
            base = f
        else:
            worst = max(worst, max(abs(pct(x, y)) for x, y in zip(f, base)))
    engine.GRID_OFFSET = (0.0, 0.0)
    record(4, 'grid alignment shift (Tiptest, GN 80)', f'max {worst:.2f}%', '0.5%', worst <= 0.5)
    # brace sweep on the rectangle
    for GN in (34, 80):
        g = rect_cases.rect_geo(GN)
        dd = max(rect_cases.A, rect_cases.B)/(GN - 1)
        fs = []
        for k in range(31):
            y = 0.090 + k*0.1*dd
            b = rect_cases.brace('x', y, 0.020, 0.280)
            fs.append(solve(g, [b], n=3)['freqs'][:3])
        fs = np.array(fs)
        step = np.max(np.abs(np.diff(fs, axis=0))/fs[:-1])*100
        kink = np.max(np.abs(np.diff(fs, 2, axis=0))/fs[1:-1])*100
        record(4, f'brace sweep in 0.1-cell steps (GN {GN})', f'max step {step:.2f}%, max 2nd difference {kink:.3f}%', '0.3% per step', step <= 0.3)
        EXTRA[f'sweep_{GN}'] = fs.tolist()
    # rotation of a single brace on a square
    L = 0.3
    out = []
    for ang in range(0, 91, 5):
        t = math.radians(ang); c, s = math.cos(t), math.sin(t)
        g = iso_geo(square_outline(0, n_side=120), L, 'clamped', 60, E=9.79e9, nu=0.3, h=0.0028, rho=350)
        b = engine.make_brace(0.5 - 0.22*c, 0.5 - 0.22*s, 0.5 + 0.22*c, 0.5 + 0.22*s, 12, 8, modulus=9.79e9, density=350, shear=3.5e9)
        out.append(solve(g, [b], n=3)['freqs'][:2])
    out = np.array(out)
    sym = np.max(np.abs(out[:9] - out[::-1][:9])/out[:9])*100
    kink = np.max(np.abs(np.diff(out, 2, axis=0))/out[1:-1])*100
    record(4, 'brace rotation 0-90 deg on an isotropic square', f'max 2nd difference {kink:.2f}%, 0/90 symmetry {sym:.2f}%',
           'smooth, no jumps at grid-aligned angles', kink <= 1.0)
    EXTRA['rotation'] = out.tolist()
    # mirror symmetry
    fx = load_fixture('parttest'); G = fx['GEO']
    ax = engine._outline_symmetry_axis_fy(G['outlinePts'])
    mir = []
    for b in fx['braces']:
        m = json.loads(json.dumps(b))
        for key in ('pts',):
            for p_ in m[key]:
                p_['fy'] = 2*ax - p_['fy']
        for key in ('cp', 'cp2'):
            if m.get(key):
                m[key]['fy'] = 2*ax - m[key]['fy']
        mir.append(m)
    Gm = json.loads(json.dumps(G))
    for p_ in Gm['outlinePts']:
        p_['fy'] = 2*ax - p_['fy']
    Gm['holeCy'] = 2*ax - G['holeCy']
    if Gm.get('bridgePts'):
        for p_ in Gm['bridgePts']:
            p_['fy'] = 2*ax - p_['fy']
    if Gm.get('bridge'):
        Gm['bridge']['fy'] = 2*ax - G['bridge']['fy']
    f1 = solve(payload_geo(G, 80), fx['braces'], n=6)['freqs'][:5]
    f2 = solve(payload_geo(Gm, 80), mir, n=6)['freqs'][:5]
    d = max(abs(pct(x, y)) for x, y in zip(f2, f1))
    record(4, 'mirror symmetry (2partTest mirrored)', f'max {d:.2f}%', '0.1%', d <= 0.1,
           'outline is mirrored too, so the grid lands differently; differences are grid alignment')
    # taper and curvature limits
    g = rect_cases.rect_geo(60)
    b1 = engine.make_brace(0.1, 0.5, 0.9, 0.5, 12, 8, modulus=9.79e9, density=350, shear=3.5e9)
    b2 = engine.make_brace(0.1, 0.5, 0.9, 0.5, 12, 8, modulus=9.79e9, density=350, shear=3.5e9, curve_cp=(0.5, 0.5))
    b3 = engine.make_brace(0.1, 0.5, 0.9, 0.5, 12, 8, modulus=9.79e9, density=350, shear=3.5e9, height_tip=12, width_tip=8)
    f1, f2, f3 = (solve(g, [b], n=4)['freqs'][:4] for b in (b1, b2, b3))
    d2 = max(abs(pct(x, y)) for x, y in zip(f2, f1)); d3 = max(abs(pct(x, y)) for x, y in zip(f3, f1))
    record(4, 'curved brace with control point on the chord = straight', f'{d2:.3f}%', '0.1%', d2 <= 0.1)
    record(4, 'equal tip and centre dimensions = untapered', f'{d3:.2e}%', 'exact', d3 <= 1e-9)


# ── Stage 5: limits and consistency ──
def stage5():
    fx = load_fixture('tiptest'); G = fx['GEO']
    g = payload_geo(G, 60)
    gb = dict(g, bridgeStructural=False)   # a bridge block has rotary inertia, which a thin plate doesn't
    a = solve(dict(gb, plateShear=False), fx['braces'], n=6)['freqs'][:5]
    b = solve(dict(gb, Gxz=G['Gxz']*1000, Gyz=G['Gyz']*1000), fx['braces'], n=6)['freqs'][:5]
    d = max(abs(pct(x, y)) for x, y in zip(b, a))
    record(5, 'G_LT, G_RT x1000 -> thin plate (Tiptest braces, no bridge)', f'{d:.3f}%', '0.1%', d <= 0.1)
    a = solve(gb, [], n=6)['freqs'][:5]
    for hgt, lim, fp in ((0.1, 0.5, False), (0.001, 0.5, True)):
        thin = [dict(bb, height=hgt, heightTip=hgt if bb.get('heightTip') is not None else None) for bb in fx['braces']]
        b = solve(dict(gb, braceFootprint=fp), thin, n=6)['freqs'][:5]
        d = max(abs(pct(x, y)) for x, y in zip(b, a))
        record(5, f'brace height {hgt} mm -> bare plate (footprint {"on" if fp else "off"})', f'{d:.2f}%', f'{lim}%', d <= lim,
               'an isotropic 0.1 mm veneer legitimately stiffens spruce across the grain, so the footprint is checked at 0.001 mm')
    gold = json.load(open(os.path.join(GOLD, 'baseline.json')))['fixtures']
    worst = 0
    for name in ('tiptest', 'xbrace', 'bare_clamped', 'bare_free'):
        fxx = load_fixture(name)
        for GN in (34, 80):
            f = solve(dict(payload_geo(fxx['GEO'], GN), solverVersion='v1'), fxx['braces'], n=12)['freqs']
            ref = gold[name][str(GN)]['freqs']
            worst = max(worst, max(abs(x/y - 1) for x, y in zip(f, ref)))
    record(5, "solverVersion 'v1' reproduces the pre-change engine", f'max rel diff {worst:.1e}', '1e-6 (ARPACK stopping on close pairs)', worst <= 1e-6)
    # flange cap: single isolated brace unchanged
    g = rect_cases.rect_geo(60)
    b = [rect_cases.brace('x', 0.100, 0.020, 0.280)]
    a = solve(dict(g, flangeCap=False), b, n=4)['freqs'][:4]; c = solve(dict(g, flangeCap=True), b, n=4)['freqs'][:4]
    d = max(abs(pct(x, y)) for x, y in zip(c, a))
    record(5, 'flange cap leaves an isolated brace unchanged', f'{d:.3f}%', '0.1%', d <= 0.1)
    # two parallel braces swept 100 -> 10 mm
    fs = []
    gaps = [100, 80, 60, 50, 40, 30, 25, 20, 15, 10]
    for gap in gaps:
        br = [rect_cases.brace('x', 0.100 - gap/2000, 0.020, 0.280, w=6), rect_cases.brace('x', 0.100 + gap/2000, 0.020, 0.280, w=6)]
        fs.append([solve(dict(g, flangeCap=cap), br, n=3)['freqs'][0] for cap in (False, True)])
    fs = np.array(fs)
    diff = np.abs(fs[:, 1] - fs[:, 0])/fs[:, 0]*100
    bite = [gp for gp, dlt in zip(gaps, diff) if dlt > 0.05]
    kink = np.max(np.abs(np.diff(fs, 2, axis=0))/fs[1:-1])*100
    record(5, 'parallel braces 100->10 mm: smooth, cap bites only below 2*40h/2', f'cap active at gaps {bite} mm, max 2nd diff {kink:.2f}%',
           f'cap only below {40*2.8:.0f} mm', all(gp < 40*2.8 for gp in bite))
    EXTRA['parallel_sweep'] = {'gaps': gaps, 'f1_nocap_cap': fs.tolist()}
    # mass bookkeeping incl. point masses
    pm = [{'fx': 0.6, 'fy': 0.5, 'massG': 5.0}, {'fx': 0.4, 'fy': 0.3, 'massG': 2.5}]
    g2 = dict(payload_geo(G, 60), pointMasses=pm)
    r = engine.evaluate_design(dict(g2, targetFreqs=[]), fx['braces'], 6)
    bm = sum(engine.braces_to_masses(fx['braces'], G['Lx'], G['Ly'], 10)[2])*1000
    tot_expected = bm + r['bridgeMassG'] + 7.5
    tot = r['massG'] + r['bridgeMassG'] + r['pointMassG']
    record(5, 'mass bookkeeping (braces + bridge + point masses)', f'{tot:.2f} g vs {tot_expected:.2f} g', '0.1%',
           abs(pct(tot, tot_expected)) <= 0.1)
    # point-mass perturbation vs first-order theory: df/f = -1/2 m phi(x)^2, phi mass-normalized
    gp = dict(payload_geo(load_fixture('bare_clamped')['GEO'], 60), targetFreqs=[])
    px, py = 0.62, 0.45
    base = engine.evaluate_design(dict(gp, probePoints=[(px, py)]), [], 5)
    f0 = [m['freq'] for m in base['modes']][:5]
    dd = base['gridMm']/1000.0   # shapes are normalized against mass per area; per node it's per dd^2
    phi = [row[0]/dd for row in base['probe']][:5]
    worst = 0
    for mg in (0.5, 5.0):
        r1 = engine.evaluate_design(dict(gp, pointMasses=[{'fx': px, 'fy': py, 'massG': mg}]), [], 5)
        f1 = [m['freq'] for m in r1['modes']][:5]
        for a_, b_, ph in zip(f1, f0, phi):
            pred = -0.5*(mg/1000)*ph*ph*100
            if abs(pred) > 0.02:
                worst = max(worst, abs(pct(a_, b_)/pred - 1)) if mg == 0.5 else worst
        if mg == 5.0:
            ratio = [pct(a_, b_) for a_, b_ in zip(f1, f0)]
    record(5, 'point mass 0.5 g vs first-order prediction', f'max deviation {worst*100:.1f}%', '10%', worst <= 0.10,
           '5 g shifts: ' + ', '.join(f'{x:+.2f}%' for x in ratio))
    # load linearity
    g0 = dict(payload_geo(G, 50), targetFreqs=[], bridgeTorqueNmm=0.0)
    d1 = engine.evaluate_design(g0, fx['braces'], 4)['deflBridgeMm']
    d2 = engine.evaluate_design(dict(g0, loadN=2*G.get('loadN', 300)), fx['braces'], 4)['deflBridgeMm']
    record(5, 'doubling the load doubles the deflection', f'ratio {d2/d1:.6f}', '2 exactly', abs(d2/d1 - 2) < 1e-6)


# ── Stage 6: independent 3D solid model, same plate (rectangle ladder) ──
def stage6():
    for label, flags in [('v1', {'solverVersion': 'v1'}), ('v2', {})]:
        res = rect_cases.compare(100, **flags)
        EXTRA['rect3d_' + label] = res
        for name, r in res.items():
            e = max(abs(x) for x in r['err'])
            if label == 'v2':
                record(6, f'3D solid rectangle, {name}', f'max |err| {e:.1f}% (mode 1 {r["err"][0]:+.1f}%)',
                       '2% bare / 5% braced', e <= (2.0 if name == 'bare' else 5.0),
                       f'v1 for reference: max |err| {max(abs(x) for x in EXTRA["rect3d_v1"][name]["err"]):.1f}%' if 'rect3d_v1' in EXTRA else '')


# ── Stage 7: ANSYS 3D solid (existing guitar runs) ──
def stage7():
    out = {}
    for name in ('tiptest', 'parttest', 'experimental'):
        fx = load_fixture(name); A = fx['ansys']['freqs']
        row = {'ansys': A}
        for label, flags in [('v2', {}), ('v1', {'solverVersion': 'v1'})]:
            f = solve(dict(payload_geo(fx['GEO'], 120), **flags), fx['braces'], n=10)['freqs']
            row[label] = f
            e = [pct(x, y) for x, y in zip(f[:3], A[:3])]
            record(7, f'{name} vs existing ANSYS run ({label}, GN 120, modes 1-3 by order)', ', '.join(f'{x:+.1f}%' for x in e),
                   '5%, no systematic sign', max(map(abs, e)) <= 5.0 and not (all(x > 0 for x in e) or all(x < 0 for x in e)),
                   fx['ansys'].get('notes', ''))
        out[name] = row
    # bare-plate predictions to compare against an ANSYS bare-plate run
    for bc in ('clamped', 'ss'):
        fx = load_fixture('bare_' + bc)
        out['bare_' + bc] = solve(payload_geo(fx['GEO'], 160), [], n=8)['freqs']
    EXTRA['ansys'] = out


# ── Stage 8: optimizer behaviour (reduced: falcate + novel, fixed seeds) ──
def stage8(iters=400):
    # Targets a falcate design is known to reach: the falcate fixture's own
    # v2 modes 1-3 (GN 100), with a 2.5% tolerance.
    fx = load_fixture('falcate'); G = fx['GEO']
    ref = solve(payload_geo(G, 100), fx['braces'], n=6)['freqs'][:3]
    targets = [{'freq': round(f, 1), 'tol': round(0.025*f, 1), 'weight': 1} for f in ref]
    EXTRA['stage8_targets'] = targets
    def run(seed, pattern):
        p = {'geo': payload_geo(G, 34, [t['freq'] for t in targets]), 'targets': targets, 'deflLimitMm': 3.0, 'massWeight': 0.15,
             'iters': iters, 'patterns': [pattern], 'maxBraces': 6, 'hifiGN': 50, 'displayGN': 100, 'seed': seed,
             'polishEnabled': True, 'minTipWidthMm': 3}
        k, res = engine._solve_pattern_sync(pattern, p)
        return res
    rec = {}
    t0 = time.perf_counter()
    a = run(7, 'falcate'); b = run(7, 'falcate')
    same = a and b and a.get('freqs') == b.get('freqs')
    record(8, 'same seed gives the same design (falcate)', 'identical' if same else 'different', 'identical', bool(same))
    hits, spreads = [], []
    for seed in range(1, 6):
        r = run(seed, 'falcate')
        f = r.get('freqs') or []
        hit = all(abs(x - t['freq']) <= t['tol'] for x, t in zip(f, targets)) if len(f) >= 3 else False
        err = sum(abs(x/t['freq'] - 1) for x, t in zip(f, targets))/len(targets)*100 if len(f) >= 3 else None
        hits.append(hit); spreads.append(err)
        rec[seed] = {'freqs': f, 'meanErrPct': err, 'gridEstimate': r.get('gridEstimate')}
    record(8, 'falcate on reachable targets (+-2.5%), 5 seeds: card within tolerance', f'{sum(hits)}/5', '>= 4/5', sum(hits) >= 4,
           'mean target error per seed: ' + ', '.join(f'{c:.1f}%' for c in spreads if c is not None))
    EXTRA['stage8'] = {'runs': rec, 'seconds': time.perf_counter() - t0, 'iters': iters}


# ── Stage 9: performance, parity, compatibility, robustness ──
def stage9():
    gold = json.load(open(os.path.join(GOLD, 'baseline.json')))
    fx = load_fixture('tiptest')
    times = {}
    for GN in (34, 60, 150):
        g = payload_geo(fx['GEO'], GN, targets=[200, 300, 380, 420, 520, 600])
        ts = []
        for k in range(5 if GN < 150 else 2):
            g2 = dict(g, eigTol=1e-6) if GN < 150 else g
            t = time.perf_counter(); engine.evaluate_design(g2, fx['braces'], 6); ts.append(time.perf_counter() - t)
        times[GN] = sorted(ts)[len(ts)//2]
    base34 = gold['fixtures']['tiptest']['34']['seconds']
    record(9, 'GN 34 evaluation time (Tiptest)', f'{times[34]*1000:.0f} ms vs baseline {base34*1000:.0f} ms', 'no slower', times[34] <= base34*1.02)
    EXTRA['timing'] = {str(k): v for k, v in times.items()}
    EXTRA['timing_baseline'] = {k: gold['fixtures']['tiptest'][k]['seconds'] for k in ('34', '80', '120', '200')}
    # parity: the browser loads desktop/engine.py itself
    html = open(os.path.join(ROOT, 'index.html')).read()
    single = "desktop/engine.py" in html and 'def evaluate_design' not in html
    record(9, 'browser and desktop run the same engine file', 'index.html loads desktop/engine.py' if single else 'separate copy',
           'single source', single)
    # backward compatibility: old saved projects solve
    ok, n = 0, 0
    for path in ['~/Downloads/CADtest2.json', '~/Downloads/234567.json', '~/Downloads/Falcatewinner.json', '~/Downloads/Cardtest.json']:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            continue
        n += 1
        try:
            d = json.load(open(path)); G = d['GEO']
            res = d['GEN']['results']; key = next(k for k, v in res.items() if v and v.get('braces'))
            G = {**{'Gxz': None, 'Gyz': None}, **G}
            r = solve(payload_geo(G, 50), res[key]['braces'], n=6)
            ok += int(len(r['freqs']) >= 3 and all(np.isfinite(r['freqs'])))
        except Exception as exc:
            print('compat fail', path, exc)
    record(9, 'old saved projects load and solve', f'{ok}/{n}', 'all', ok == n and n > 0)
    # robustness: 200 random Novel/Experimental layouts at GN 34
    fx = load_fixture('bare_clamped'); G = dict(fx['GEO'], bridgeStructural=True)
    geo = payload_geo(G, 34, targets=[200, 300, 400])
    random.seed(1); np.random.seed(1)
    bad = 0; low = 0
    for k in range(200):
        nb = random.randint(2, 10)
        br = []
        for _ in range(nb):
            x1, y1, x2, y2 = (random.uniform(0.05, 0.95) for _ in range(4))
            cp = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)) if random.random() < 0.5 else None
            br.append(engine.make_brace(x1, y1, x2, y2, random.uniform(4, 22), random.uniform(3, 12),
                                        curve_cp=cp, height_tip=random.uniform(2, 10), width_tip=random.uniform(3, 6)))
        br = engine._clip_braces_to_outline(br, geo)
        try:
            r = engine.evaluate_design(dict(geo, eigTol=1e-6), br, 3)
            if not r.get('ok') or not all(np.isfinite([m['freq'] for m in r['modes']])):
                bad += 1
            elif r['modes'] and r['modes'][0]['freq'] < 5:
                low += 1
        except Exception as exc:
            bad += 1
    record(9, '200 random layouts at GN 34', f'{bad} failures, {low} modes under 5 Hz', '0, 0', bad == 0 and low == 0)


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8, 9: stage9}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='1,2,4,5')
    a = ap.parse_args()
    for s_ in a.stages.split(','):
        t = time.perf_counter()
        try:
            STAGES[int(s_)]()
        except Exception as exc:
            import traceback; traceback.print_exc()
            record(int(s_), f'stage {s_} crashed', str(exc), '-', False)
        print(f'-- stage {s_}: {time.perf_counter()-t:.0f}s', flush=True)
        save()

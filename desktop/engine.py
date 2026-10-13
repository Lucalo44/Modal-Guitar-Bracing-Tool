# ══════════════════════════════════════════════════════════════════════════
# INVERSE BRACING DESIGN ENGINE -- native (numpy+scipy, no Pyodide/WASM)
#
# This is the SAME physics/optimizer engine as the browser version's
# embedded <script type="py"> block in index.html (copied from there, not
# reimplemented), with only the Pyodide-specific glue swapped out:
#   - the PyScript-only imports -> a small BRIDGE dict of plain Python
#     callbacks, set by whatever's hosting this (app.py).
#   - the JS-export lines at the very end that wired these functions onto
#     the page's window object -> removed; the host imports these
#     functions directly and signals readiness itself.
#   - added `run_all_patterns_parallel`, which runs each requested pattern
#     (xbrace/fan/falcate/novel/novelExperimental/custom) in its OWN OS
#     process via ProcessPoolExecutor -- true parallelism across CPU cores,
#     impossible in the browser build since Pyodide is pinned to the
#     page's single main thread. Every pattern was already fully
#     independent of every other one in the original sequential
#     implementation; this just actually runs them that way.
#
# Plate physics (grid build, brace mass mapping, brace bending-stiffness
# coupling) reuses the same finite-difference biharmonic-plate model as the
# paired bracing visualizer tool. New here: static point-load deflection
# solve, MAC shape comparison against measured node data, four bracing
# pattern encoders, and a simulated-annealing search over each.
# ══════════════════════════════════════════════════════════════════════════
import json, math, random, asyncio, os, time, threading
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import cKDTree

# ── JS bridge shim ──────────────────────────────────────────────────────
# In the browser build, progress/done/error events call straight into the
# page's window object. Here they go through this small table of plain
# Python callbacks instead, so this module has zero dependency on any
# particular UI host (pywebview, a future Tauri/Qt wrapper, a CLI, tests,
# ...). A host wires itself up with set_bridge(...); anything left unset
# is a no-op.
BRIDGE = {
    'onOptimizerProgress': lambda *a, **k: None,
    'onOptimizerTrace': lambda *a, **k: None,
    'onOptimizerDone': lambda *a, **k: None,
    'onOptimizerError': lambda *a, **k: None,
    'onManualEvalDone': lambda *a, **k: None,
    'onManualEvalError': lambda *a, **k: None,
    'onBareModesResult': lambda *a, **k: None,
    'onBareModesError': lambda *a, **k: None,
}

def set_bridge(**callbacks):
    """Register one or more of the callbacks above. Unknown keys are
    rejected loudly rather than silently ignored -- a typo'd callback name
    here would otherwise fail silently and look like a dropped event."""
    for name, fn in callbacks.items():
        if name not in BRIDGE:
            raise KeyError(f'unknown bridge callback: {name!r}')
        BRIDGE[name] = fn

# ── Geometry helpers ────────────────────────────────────────────────────
def points_in_polygon(px, py, poly_pts):
    inside = np.zeros(px.shape, dtype=bool)
    xj, yj = poly_pts[-1]
    for xi, yi in poly_pts:
        cond = (yi > py) != (yj > py)
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        toggle = cond & (px < xint)
        inside = np.where(toggle, ~inside, inside)
        xj, yj = xi, yi
    return inside

def build_grid(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN):
    """Builds three directional second-derivative operators -- Lxx, Lyy,
    Lxy -- rather than a single isotropic Laplacian. An orthotropic
    (grain-direction-dependent) plate's bending energy has THREE
    independent stiffness terms (D11*w,xxxx + 2(D12+2*D66)*w,xxyy +
    D22*w,yyyy), not one isotropic D*grad^4(w); there's no way to recover
    that from squaring a single combined operator the way the old
    isotropic-only solver did (D*L^T*L). Instead each of these three
    pieces gets squared separately when assembling K in evaluate_design
    (D11*Lxx^T Lxx + D22*Lyy^T Lyy + 2(D12+2D66)*Lxy^T Lxy) -- same
    "square it for guaranteed positive-semi-definiteness" principle as
    before, just applied to three simpler, direction-specific building
    blocks instead of one combined one. This was validated before being
    wired in here: matched a known closed-form solution (Navier series)
    for a simply-supported rectangular orthotropic plate to within ~1%
    across the first several modes, confirmed positive-semi-definite (no
    spurious zero or negative-energy modes), and confirmed it reduces to
    the same numbers as the old isotropic solver when given isotropic
    material properties.
    """
    poly_phys = [(p['fx'] * Lx, p['fy'] * Ly) for p in poly_frac]
    xs = [p[0] for p in poly_phys]; ys = [p[1] for p in poly_phys]
    bbx0, bbx1 = min(xs), max(xs); bby0, bby1 = min(ys), max(ys)
    dd = max(bbx1 - bbx0, bby1 - bby0) / (GN - 1)
    nx = round((bbx1 - bbx0) / dd) + 1
    ny = round((bby1 - bby0) / dd) + 1
    poly_local = [(x - bbx0, y - bby0) for x, y in poly_phys]
    hx_local = hole_cx * Lx - bbx0; hy_local = hole_cy * Ly - bby0
    hr = hole_r * min(Lx, Ly)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    X = (ii * dd).ravel(); Y = (jj * dd).ravel()
    poly_only_flat = points_in_polygon(X, Y, poly_local)
    dist_hole = np.hypot(X - hx_local, Y - hy_local)
    inside_flat = poly_only_flat & (dist_hole >= hr)
    in_poly = poly_only_flat.reshape(ny, nx)
    inside = inside_flat.reshape(ny, nx)

    if bc_mode == 'ss':
        # Simply-supported: DOFs are eroded one full ring in from any
        # domain edge (all 8 neighbors, not just the 4 orthogonal ones --
        # the cross-derivative stencil needs diagonal neighbors too) --
        # the excluded ring is where w=0 is effectively imposed, and
        # anything one step beyond THAT uses an antisymmetric ghost
        # reflection through the nearest valid DOF (w_ghost = -w_mirror),
        # the correct zero-curvature condition for a simply-supported
        # edge, not simply zero (that would over-stiffen the edge --
        # confirmed directly: using a plain zero ghost here reproduced
        # frequencies roughly DOUBLE the correct analytic value on a
        # validation rectangle before this was corrected).
        pad = np.zeros((ny + 2, nx + 2), dtype=bool)
        pad[1:-1, 1:-1] = inside
        dof_mask = inside & pad[1:-1, 0:-2] & pad[1:-1, 2:] & pad[0:-2, 1:-1] & pad[2:, 1:-1] \
                          & pad[0:-2, 0:-2] & pad[0:-2, 2:] & pad[2:, 0:-2] & pad[2:, 2:]
    else:
        dof_mask = inside

    node_id = -np.ones((ny, nx), dtype=np.int64)
    nodes_j, nodes_i = np.nonzero(dof_mask)
    N = nodes_i.size
    node_id[nodes_j, nodes_i] = np.arange(N)
    if N < 20:
        return {'N': N, 'nx': nx, 'ny': ny, 'dd': dd}

    def get_or_reflect(i, j, ref_i, ref_j):
        if 0 <= i < nx and 0 <= j < ny and inside[j, i] and node_id[j, i] >= 0:
            return node_id[j, i], 1.0
        if bc_mode == 'ss':
            # antisymmetric reflection through the DOF that requested it
            mi, mj = 2*ref_i - i, 2*ref_j - j
            if 0 <= mi < nx and 0 <= mj < ny and inside[mj, mi] and node_id[mj, mi] >= 0:
                return node_id[mj, mi], -1.0
        return None, 0.0

    def build(stencil):
        rows, cols, vals = [], [], []
        for r in range(N):
            i, j = int(nodes_i[r]), int(nodes_j[r])
            for di, dj, c in stencil:
                nid, sign = get_or_reflect(i+di, j+dj, i, j)
                if nid is not None:
                    rows.append(r); cols.append(nid); vals.append(c*sign)
        K = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        K.sum_duplicates()
        return K

    coef2 = 1.0/(dd*dd)
    Lxx = build([(-1,0,coef2), (0,0,-2*coef2), (1,0,coef2)])
    Lyy = build([(0,-1,coef2), (0,0,-2*coef2), (0,1,coef2)])
    Lxy = build([(1,1,0.25*coef2), (1,-1,-0.25*coef2), (-1,1,-0.25*coef2), (-1,-1,0.25*coef2)])
    return {'Lxx': Lxx, 'Lyy': Lyy, 'Lxy': Lxy, 'N': N, 'dd': dd, 'nx': nx, 'ny': ny,
            'nodes_i': nodes_i, 'nodes_j': nodes_j, 'node_id': node_id, 'bc_mode': bc_mode,
            'inside': inside, 'bbx0': bbx0, 'bby0': bby0}

def _dim_at(center, tip, t):
    """Cross-section dimension at position t in [0,1] along a brace: tip
    value at both ends, center value at the midpoint, parabolic in between.
    Lets width and height taper independently and by different amounts."""
    tip = center if tip is None else tip
    return tip + (center - tip) * 4.0 * t * (1.0 - t)

def _brace_ns(b, Lx, Ly, seg_len_mm):
    """Number of discretization segments for a brace's mass/stiffness
    sampling, sized so each segment is roughly seg_len_mm long (chord-length
    approximation -- endpoint-to-endpoint, not the true arc length for a
    curved brace, but close enough to size the discretization). None or
    non-positive falls back to the old fixed default of 10."""
    if not seg_len_mm or seg_len_mm <= 0:
        return 10
    a, z = b['pts'][0], b['pts'][1]
    chord_mm = math.hypot((z['fx']-a['fx'])*Lx, (z['fy']-a['fy'])*Ly) * 1000.0
    return int(max(4, min(300, round(chord_mm / seg_len_mm))))

def _sample_brace(b, Lx, Ly, NS=10):
    a, z = b['pts'][0], b['pts'][1]
    ax, ay = a['fx'] * Lx, a['fy'] * Ly
    zx, zy = z['fx'] * Lx, z['fy'] * Ly
    btype = b.get('type')
    is_quad = (btype == 'curve') and b.get('cp') is not None
    is_cubic = (btype == 'curve3') and b.get('cp') is not None and b.get('cp2') is not None
    if is_quad:
        cpx, cpy = b['cp']['fx'] * Lx, b['cp']['fy'] * Ly
    if is_cubic:
        c1x, c1y = b['cp']['fx'] * Lx, b['cp']['fy'] * Ly
        c2x, c2y = b['cp2']['fx'] * Lx, b['cp2']['fy'] * Ly
    def eval_pt(t):
        if is_cubic:
            u = 1-t
            x = u**3*ax + 3*u**2*t*c1x + 3*u*t**2*c2x + t**3*zx
            y = u**3*ay + 3*u**2*t*c1y + 3*u*t**2*c2y + t**3*zy
        elif is_quad:
            x = (1-t)**2*ax + 2*(1-t)*t*cpx + t**2*zx
            y = (1-t)**2*ay + 2*(1-t)*t*cpy + t**2*zy
        else:
            x = ax + t*(zx-ax); y = ay + t*(zy-ay)
        return x, y
    pts = []
    for i in range(NS + 1):
        t = i / NS
        x, y = eval_pt(t)
        if is_quad or is_cubic:
            dt = 1.0 / NS
            t0, t1 = max(0.0, t - dt / 2), min(1.0, t + dt / 2)
            x0, y0 = eval_pt(t0); x1, y1 = eval_pt(t1)
            sl = math.hypot(x1 - x0, y1 - y0)
        else:
            sl = math.hypot(zx - ax, zy - ay) / NS
        pts.append((x, y, sl, t))
    return pts

def braces_to_masses(braces, Lx, Ly, seg_len_mm=None):
    xs, ys, ms = [], [], []
    for b in braces:
        wC = b['width']/1000.0; wT = (b['width'] if b.get('widthTip') is None else b['widthTip'])/1000.0
        hC = b['height']/1000.0; hT = (b['height'] if b.get('heightTip') is None else b['heightTip'])/1000.0
        for x, y, sl, t in _sample_brace(b, Lx, Ly, _brace_ns(b, Lx, Ly, seg_len_mm)):
            A = _dim_at(wC, wT, t) * _dim_at(hC, hT, t)
            m = b.get('density', 420) * A * sl
            if m > 0:
                xs.append(min(0.999, max(0.001, x/Lx))*Lx); ys.append(min(0.999, max(0.001, y/Ly))*Ly); ms.append(m)
    return xs, ys, ms

def _bilinear_corners(grid, xs, ys):
    """Bilinear interpolation from the 4 grid nodes surrounding each physical
    point (xs, ys arrays, metres). Returns (ids, W), both shape (n, 4): DOF
    index of each corner (-1 where that corner isn't a DOF) and its weight.

    Corners that aren't DOFs are dropped. Under a simply-supported edge
    that's exact -- those nodes are the w=0 ring -- so the weights are left
    as they are. Under a free edge they're just missing (outside the plate),
    so the remaining weights are renormalized to sum to 1; otherwise a rigid
    translation would read as curvature near the edge and break the plate's
    rigid-body modes."""
    nx, ny, dd = grid['nx'], grid['ny'], grid['dd']
    fx = (np.asarray(xs, dtype=float) - grid['bbx0']) / dd
    fy = (np.asarray(ys, dtype=float) - grid['bby0']) / dd
    i0 = np.clip(np.floor(fx).astype(int), 0, nx-2)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny-2)
    tx = np.clip(fx - i0, 0.0, 1.0); ty = np.clip(fy - j0, 0.0, 1.0)
    ci = np.stack([i0, i0+1, i0, i0+1], axis=1)
    cj = np.stack([j0, j0, j0+1, j0+1], axis=1)
    W = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], axis=1)
    ids = grid['node_id'][cj, ci]
    W = np.where(ids >= 0, W, 0.0)
    if grid.get('bc_mode') != 'ss':
        tot = W.sum(axis=1, keepdims=True)
        W = np.where(tot > 1e-12, W / np.where(tot > 1e-12, tot, 1.0), 0.0)
    return ids, W

def _brace_stations(b, Lx, Ly, spacing):
    """Points along a brace's centreline, evenly spaced in ARC length at
    roughly `spacing` (never fewer than 3). Returns (xy (n,2) in metres,
    t (n,) Bezier parameter for the local cross-section, h = actual spacing)."""
    curved = b.get('type') in ('curve', 'curve3') and b.get('cp') is not None
    dense = _sample_brace(b, Lx, Ly, 48 if curved else 1)  # a straight brace's arc length is exact from its endpoints
    P = np.array([[p[0], p[1]] for p in dense]); T = np.array([p[3] for p in dense])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    L = s[-1]
    if L <= 0:
        return None, None, 0.0
    n = max(3, int(round(L / spacing)) + 1)
    sq = np.linspace(0.0, L, n)
    xy = np.stack([np.interp(sq, s, P[:, 0]), np.interp(sq, s, P[:, 1])], axis=1)
    return xy, np.interp(sq, s, T), L / (n - 1)

def brace_bending_stiffness_matrix(braces, Lx, Ly, grid, seg_len_mm=None):
    """Each brace as an Euler-Bernoulli beam embedded in the plate: bending
    energy 0.5*EI*(w'')^2 integrated along its arc length, with w at any point
    on the brace bilinearly interpolated from the 4 surrounding plate nodes
    (_bilinear_corners).

    Curvature comes from a 3-point second difference between stations spaced
    about one grid spacing apart (_brace_stations) -- deliberately NOT the
    mass-sampling length (seg_len_mm, unused here). Bilinear interpolation is
    only linear within a cell, so stations closer together than the grid
    can't see the plate's curvature; at grid spacing the stencil matches the
    resolution the plate itself is solved at. That makes the result
    independent of the sampling setting, independent of the brace's angle to
    the grid, and smooth as a brace moves -- unlike snapping each sample to
    its nearest node, which skipped samples that landed on the same node and
    divided node-spaced differences by the (much smaller) sample spacing.

    Same per-unit-area scaling as the plate operator (hence /dd^2)."""
    dd = grid['dd']; N = grid['N']
    all_ids, all_g, all_c = [], [], []
    for b in braces:
        wC = b.get('width',3)/1000.0; wT = (b.get('width',3) if b.get('widthTip') is None else b['widthTip'])/1000.0
        hC = b.get('height',9)/1000.0; hT = (b.get('height',9) if b.get('heightTip') is None else b['heightTip'])/1000.0
        modulus = b.get('modulus', 10e9)
        xy, t, hs = _brace_stations(b, Lx, Ly, dd)
        if xy is None:
            continue
        ids, W = _bilinear_corners(grid, xy[:, 0], xy[:, 1])
        a = np.array([1.0, -2.0, 1.0]) / (hs*hs)
        # Interior station k uses stations k-1, k, k+1 -> 12 (node, weight) pairs.
        k = np.arange(1, len(xy)-1)
        g_ids = np.concatenate([ids[k-1], ids[k], ids[k+1]], axis=1)
        g = np.concatenate([a[0]*W[k-1], a[1]*W[k], a[2]*W[k+1]], axis=1)
        # Local cross-section at each station, so a tapered brace is
        # correctly softer near its tips and stiffer through its middle.
        wk = _dim_at(wC, wT, t[k]); hk = _dim_at(hC, hT, t[k])
        EI = modulus * wk * hk**3 / 12.0
        all_ids.append(g_ids); all_g.append(g); all_c.append(EI * hs / (dd*dd))
    if not all_ids:
        return sparse.csr_matrix((N, N))
    g_ids = np.concatenate(all_ids); g = np.concatenate(all_g); coef = np.concatenate(all_c)
    rows = np.broadcast_to(g_ids[:, :, None], g_ids.shape + (12,)).ravel()
    cols = np.broadcast_to(g_ids[:, None, :], g_ids.shape[:1] + (12, 12)).ravel()
    vals = (coef[:, None, None] * g[:, :, None] * g[:, None, :]).ravel()
    keep = (rows >= 0) & (cols >= 0) & (vals != 0)
    K = sparse.coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(N, N)).tocsr()
    K.sum_duplicates()
    return K

def _spread_to_nodes(grid, xs, ys, weights):
    """Like _map_to_nodes, but splits each point's weight bilinearly over its
    4 surrounding nodes instead of dropping it all on the nearest one -- so
    brace mass moves smoothly with the brace rather than jumping node to
    node as the optimizer nudges it."""
    out = np.zeros(grid['N'])
    if not len(xs): return out
    ids, W = _bilinear_corners(grid, xs, ys)
    contrib = W * np.asarray(weights, dtype=float)[:, None]
    ok = ids >= 0
    np.add.at(out, ids[ok], contrib[ok])
    return out

def _map_to_nodes(grid, xs, ys, weights):
    N = grid['N']; out = np.zeros(N)
    if not xs: return out
    node_local = np.stack([grid['nodes_i']*grid['dd'], grid['nodes_j']*grid['dd']], axis=1)
    tree = cKDTree(node_local)
    pts_local = np.stack([np.array(xs)-grid['bbx0'], np.array(ys)-grid['bby0']], axis=1)
    _, idx = tree.query(pts_local)
    np.add.at(out, idx, np.array(weights))
    return out

def sample_field_at_points(grid, node_vals, xs_m, ys_m):
    """Bilinear-sample a nodal solution at arbitrary physical (x,y) points
    (used both for the static load footprint and for comparing simulated
    mode shapes against measured node positions -- MAC)."""
    nx, ny, dd = grid['nx'], grid['ny'], grid['dd']
    field = np.full((ny, nx), np.nan)
    field[grid['nodes_j'], grid['nodes_i']] = node_vals
    filled = np.nan_to_num(field)
    have = (~np.isnan(field)).astype(float)
    out = np.zeros(len(xs_m))
    for k in range(len(xs_m)):
        fx = (xs_m[k]-grid['bbx0'])/dd; fy=(ys_m[k]-grid['bby0'])/dd
        i0,j0 = int(np.floor(fx)), int(np.floor(fy))
        tx,ty = fx-i0, fy-j0
        if i0<0 or j0<0 or i0>=nx-1 or j0>=ny-1:
            i0=max(0,min(nx-2,i0)); j0=max(0,min(ny-2,j0)); tx=max(0,min(1,tx)); ty=max(0,min(1,ty))
        w00=(1-tx)*(1-ty); w10=tx*(1-ty); w01=(1-tx)*ty; w11=tx*ty
        num = (filled[j0,i0]*w00 + filled[j0,i0+1]*w10 + filled[j0+1,i0]*w01 + filled[j0+1,i0+1]*w11)
        den = (have[j0,i0]*w00 + have[j0,i0+1]*w10 + have[j0+1,i0]*w01 + have[j0+1,i0+1]*w11)
        out[k] = num/den if den>1e-9 else 0.0
    return out

def mac(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float((np.dot(a,b)**2) / (np.dot(a,a)*np.dot(b,b)))

# ── Core FEM evaluation: eigenmodes + static deflection under bridge load ──
# ── Geometry validity: keep braces on the actual body and off the hole ──
def brace_hole_violation(braces, hole_cx, hole_cy, hole_r, Lx, Ly, NS=14):
    """Continuous measure of how much brace material occupies the
    soundhole (0 if none does) -- used as a ratchet in the free-topology
    search: a move is only allowed if it doesn't INCREASE this value, so
    the search can still take intermediate steps to escape a design that
    happens to start touching the hole, but can never introduce a new
    violation or make an existing one worse once accepted. This still
    enforces the real constraint (no material in the hole, ever) on any
    converged result -- once violation reaches zero, no move that's
    accepted can ever push it back above zero -- it just isn't a flat
    reject on every touching state along the way, which would otherwise
    trap a design that starts bad with no way to gradually move clear.

    The keep-out radius grows by the brace's own local half-width, not
    just the bare hole radius -- checking the centerline alone lets a WIDE
    brace's actual edge cut into the hole while its centerline still shows
    zero violation, which resize mutations (free to grow width without
    ever moving the centerline) can exploit with nothing else catching it."""
    hx, hy = hole_cx*Lx, hole_cy*Ly
    hr = hole_r*min(Lx, Ly)
    total = 0.0
    for b in braces:
        wC = b.get('width', 3.0)/1000.0
        wT = (b.get('width', 3.0) if b.get('widthTip') is None else b['widthTip'])/1000.0
        for x, y, sl, t in _sample_brace(b, Lx, Ly, NS):
            keepout = hr + _dim_at(wC, wT, t)/2.0
            d = math.hypot(x-hx, y-hy)
            if d < keepout:
                total += (keepout-d)**2
    return total

def brace_geometry_penalty(braces, outline_frac, hole_cx, hole_cy, hole_r, Lx, Ly, NS=14):
    """Penalizes any brace material that falls outside the traced body
    outline, or on/near the soundhole. This is enforced the same way for
    every pattern (X-brace, fan, falcate, novel) via the search objective,
    rather than hard-coding per-pattern clipping -- so it also protects
    against any future pattern that might otherwise wander off the top."""
    if not outline_frac or len(outline_frac) < 3:
        return 0.0
    poly_phys = [(p['fx']*Lx, p['fy']*Ly) for p in outline_frac]
    hx, hy = hole_cx*Lx, hole_cy*Ly
    hr = hole_r*min(Lx, Ly)
    penalty = 0.0
    for b in braces:
        wC = b.get('width', 3.0)/1000.0
        wT = (b.get('width', 3.0) if b.get('widthTip') is None else b['widthTip'])/1000.0
        xs, ys, half_ws = [], [], []
        for x, y, sl, t in _sample_brace(b, Lx, Ly, NS):
            xs.append(x); ys.append(y); half_ws.append(_dim_at(wC, wT, t)/2.0)
        xs_a, ys_a, half_w_a = np.array(xs), np.array(ys), np.array(half_ws)
        inside = points_in_polygon(xs_a, ys_a, poly_phys)
        n_out = int(np.sum(~inside))
        if n_out:
            penalty += n_out * 180.0
        d_hole = np.hypot(xs_a-hx, ys_a-hy)
        # Clearance scales with each sample's own local half-width (see
        # brace_hole_violation) plus the same ~4mm rim allowance as before,
        # not a flat hr+4mm regardless of how wide the brace actually is.
        hole_margin = hr + half_w_a + 0.004
        over_hole = np.clip(hole_margin - d_hole, 0, None)
        if np.any(over_hole > 0):
            penalty += float(np.sum((over_hole/np.maximum(hole_margin,1e-6))**2)) * 160.0
    return penalty

def brace_overlap_penalty(braces, Lx, Ly, NS=10):
    """Penalizes braces whose material runs on top of / alongside another
    brace's material for any real stretch -- redundant stiffness in one
    spot instead of spreading structure (and mass) across the top where it
    can actually do something to the mode shapes. Only the INTERIOR of
    each brace is checked (endpoints excluded): several braces sharing a
    joint -- a fan's rays meeting at one hub, braces meeting at a strut --
    is a deliberate, normal structural choice, not the redundant parallel
    stacking this is meant to catch. A brief single-point crossing (like
    the two arms of an X-brace) still only costs a little either way;
    two braces running alongside each other for real length costs a lot."""
    if len(braces) < 2:
        return 0.0
    samples = []
    for b in braces:
        pts = _sample_brace(b, Lx, Ly, NS)
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        if len(xs) > 2:
            xs, ys = xs[1:-1], ys[1:-1]  # drop the two endpoints
        wC = b.get('width', 3.0); wT = b.get('widthTip', wC)
        w = (wC if wT is None else max(wC, wT)) / 1000.0
        samples.append((xs, ys, w))
    penalty = 0.0
    n = len(samples)
    for i in range(n):
        xi, yi, wi = samples[i]
        for j in range(i+1, n):
            xj, yj, wj = samples[j]
            d = np.hypot(xi[:, None]-xj[None, :], yi[:, None]-yj[None, :])
            min_sep = (wi+wj)/2.0 + 0.003  # ~3mm clearance so edges don't touch
            over = np.clip(min_sep - d, 0, None)
            if np.any(over > 0):
                penalty += float(np.sum((over/max(min_sep,1e-6))**2)) * 15.0
    return penalty

def evaluate_design(geo, braces, n_modes_needed, meas_xy=None, want_field=False):
    Lx, Ly, h = geo['Lx'], geo['Ly'], geo['h']
    rho = geo['rho']
    # Orthotropic (grain-direction-dependent) plate constants. Ex is the
    # along-grain modulus (grain assumed to run along the plate's length,
    # the standard luthiery convention -- this does NOT yet handle a
    # traced body that's rotated relative to true grain direction). Ey,
    # Gxy, nu_xy default to values that make this reduce EXACTLY to the
    # old isotropic model when a caller doesn't supply them (an older
    # saved project, or any payload that predates this) -- Ey=Ex,
    # Gxy=Ex/(2(1+nu)) (the isotropic shear-modulus identity), nu_xy=nu.
    Ex = geo['E']
    nu_xy = geo.get('nuxy', geo['nu'])
    Ey = geo.get('Ey', Ex)
    Gxy = geo.get('Gxy', Ex/(2*(1+nu_xy)))
    nu_yx = nu_xy * Ey / Ex   # reciprocity -- never set independently, or the material isn't physically valid
    denom = max(1e-9, 1 - nu_xy*nu_yx)
    D11 = Ex*h**3/(12*denom)
    D22 = Ey*h**3/(12*denom)
    D12 = nu_yx*D11
    D66 = Gxy*h**3/12
    grid = build_grid(geo['outline'], geo['holeCx'], geo['holeCy'], geo['holeR'], Lx, Ly, geo['bcMode'], geo['GN'])
    N = grid['N']
    if N < 20:
        return {'ok': False, 'reason': 'grid_too_coarse'}

    geom_penalty = brace_geometry_penalty(braces, geo['outline'], geo['holeCx'], geo['holeCy'], geo['holeR'], Lx, Ly)
    geom_penalty += brace_overlap_penalty(braces, Lx, Ly)

    seg_len_mm = geo.get('braceSegMm')
    mx, my, mm = braces_to_masses(braces, Lx, Ly, seg_len_mm)
    braceM = _spread_to_nodes(grid, mx, my, mm) / (grid['dd']**2)
    total_brace_mass_g = sum(mm)*1000.0
    K_brace = brace_bending_stiffness_matrix(braces, Lx, Ly, grid, seg_len_mm)
    # Three separate directional terms, each squared for guaranteed
    # positive-semi-definiteness (same principle the old single-operator
    # isotropic solver relied on, just applied to three simpler
    # direction-specific building blocks instead of one combined one --
    # see build_grid's docstring for the validation this went through
    # before being wired in here).
    Lxx, Lyy, Lxy = grid['Lxx'], grid['Lyy'], grid['Lxy']
    K = D11*(Lxx.T@Lxx) + D22*(Lyy.T@Lyy) + 2*(D12+2*D66)*(Lxy.T@Lxy) + K_brace
    K = (K+K.T)*0.5
    M = sparse.diags(rho*h + braceM)

    # ── Eigenmodes ──
    n_request = min(n_modes_needed+6, N-2)
    sigma = (2*math.pi*1.0)**2
    try:
        vals, vecs = eigsh(K.tocsc(), k=max(1,n_request), M=M.tocsc(), sigma=sigma, which='LM')
    except Exception as exc:
        return {'ok': False, 'reason': f'eigsh_failed:{exc}'}
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    freqs_all = np.sqrt(np.clip(vals,0,None))/(2*math.pi)
    chosen = []
    for idx, f in enumerate(freqs_all):
        if f <= 5: continue
        if any(abs(f-freqs_all[c])/f < 0.005 for c in chosen): continue
        chosen.append(idx)
        if len(chosen) >= n_modes_needed: break
    freqs_out = [float(freqs_all[c]) for c in chosen]

    meas_shapes = None
    if meas_xy is not None and len(meas_xy[0]):
        meas_shapes = []
        for c in chosen:
            shp = sample_field_at_points(grid, vecs[:,c], meas_xy[0], meas_xy[1])
            peak = np.max(np.abs(shp)) or 1.0
            meas_shapes.append((shp/peak).tolist())

    # Coarse dense-grid mode shape, purely for the animated mode-shape
    # visualization (Results panel) -- a regular grid over the
    # plate's bounding box, sampled the same way meas_shapes
    # is (bilinear off the solved eigenvector), just at many points
    # instead of a handful. Kept well below the solve grid's own resolution
    # (this is display only, not a solve) so it stays fast and the JSON
    # payload stays small even with several modes per pattern and several
    # patterns per run. Gated behind want_field (default False) since evaluate_design
    # itself is the hot inner-loop cost function during search -- called
    # hundreds of times per pattern -- and this sampling is only ever
    # needed once, for the design that's actually kept.  Normalized per
    # mode to a peak of ±1 -- the animation applies
    # its own time-varying amplitude on top of this shape in JS, this is
    # only ever the static spatial pattern. Points outside the actual
    # traced outline (or inside the soundhole) aren't masked out here --
    # the rectangular bounding-box grid is simplest to sample and send;
    # the JS renderer clips to the real outline/hole shape at draw time
    # using the same spline path it already draws for the bracing
    # diagram, so nothing outside the body is ever actually shown.
    FIELD_NX, FIELD_NY = 96, 74
    field_shapes = None
    if want_field:
        fxs = np.linspace(0.02, 0.98, FIELD_NX) * Lx
        fys = np.linspace(0.02, 0.98, FIELD_NY) * Ly
        gx, gy = np.meshgrid(fxs, fys)  # shape (FIELD_NY, FIELD_NX)
        gx_flat, gy_flat = gx.ravel(), gy.ravel()
        field_shapes = []
        for c in chosen:
            vals = sample_field_at_points(grid, vecs[:,c], gx_flat, gy_flat)
            peak = np.max(np.abs(vals)) or 1.0
            field_shapes.append((vals/peak).tolist())

    # ── Static deflection under bridge/string load ──
    # Spread the load uniformly over the bridge's actual footprint --
    # traced by hand (a real, possibly irregular shape) if given, else the
    # rectangle from length/depth, else the nearest single node -- rather
    # than concentrating it at a point. A point load lets whichever bit of
    # stiffness happens to sit right there absorb it, which understates
    # how the top actually responds under a real bridge; string tension
    # loads the whole bridge footprint, not one spot on it.
    bx = geo['bridgeFx']*Lx; by = geo['bridgeFy']*Ly
    node_x = grid['nodes_i']*grid['dd']+grid['bbx0']
    node_y = grid['nodes_j']*grid['dd']+grid['bby0']
    bridge_pts = geo.get('bridgePts') or []
    idxs = np.array([], dtype=int)
    if len(bridge_pts) >= 3:
        bridge_poly = [(p['fx']*Lx, p['fy']*Ly) for p in bridge_pts]
        inside = points_in_polygon(node_x, node_y, bridge_poly)
        idxs = np.nonzero(inside)[0]
    if idxs.size == 0:
        half_depth = (geo.get('bridgeDepthMm', 12)/1000.0)/2.0   # along length (fx)
        half_len = (geo.get('bridgeLenMm', 155)/1000.0)/2.0      # along width (fy)
        in_footprint = (np.abs(node_x-bx) <= half_depth) & (np.abs(node_y-by) <= half_len)
        idxs = np.nonzero(in_footprint)[0]
    if idxs.size == 0:
        # Footprint smaller than one grid cell at this resolution (or the
        # bridge landed off the solved region) -- fall back to the single
        # nearest node so this still solves instead of failing.
        node_local = np.stack([node_x, node_y], axis=1)
        tree = cKDTree(node_local)
        _, nearest = tree.query([bx, by], k=1)
        idxs = np.atleast_1d(nearest)
    n_idx = idxs.shape[0]
    # Strings break over the saddle at an angle rather than pulling
    # straight down, which also twists the bridge -- more force presses
    # the neck-side edge of the footprint, less (or an uplift) on the
    # tail-side edge. Modeled as a moment superposed on the uniform load,
    # linearly distributed across the footprint's depth (its extent along
    # the body's length) exactly like combined axial+bending stress on a
    # beam cross-section. This is a deliberately rough approximation --
    # see the UI note next to the torque input.
    node_x_idx = node_x[idxs]
    cx_local = float(np.mean(node_x_idx))
    dx_local = node_x_idx - cx_local
    denom = float(np.sum(dx_local**2))
    # A moment needs force applied at spatially separated points. If the
    # bridge footprint is shallower than the grid spacing, every node in
    # it collapses onto the SAME column and there's nothing to spread a
    # moment across at all. And with only TWO columns, any real-sized
    # torque saturates almost immediately (everything clips to one side,
    # see below) rather than behaving as a smooth, graduated distribution
    # -- so this requires at least three distinct columns to call it
    # meaningfully resolved. Report it either way: torque>0 with little
    # or no visible effect should be explainable, not read as a finding
    # that torque doesn't matter here.
    n_cols = int(np.unique(np.round(node_x_idx / max(grid['dd'],1e-9))).size)
    torque_resolved = bool(n_cols >= 3)
    M = geo.get('bridgeTorqueNmm', 0.0) / 1000.0  # N·mm -> N·m
    base = geo['loadN']/n_idx
    f_i = np.full(n_idx, base) + (M*dx_local/denom if denom > 1e-12 else np.zeros(n_idx))
    # A large enough torque could ask a node to pull UP rather than push
    # down, which this simple superposition can't represent (that needs a
    # proper moment/rotation boundary condition, not just a force) --
    # clip at zero and rescale to preserve the total applied force so the
    # solve stays physically sane instead of going unstable.
    f_i = np.clip(f_i, 0.0, None)
    total_f = float(np.sum(f_i))
    if total_f > 1e-12:
        f_i = f_i * (geo['loadN']/total_f)
    else:
        f_i = np.full(n_idx, base)
    w = f_i / geo['loadN'] if geo['loadN'] else np.full(n_idx, 1.0/n_idx)
    F = np.zeros(N)
    # K is in point-collocation form (D∇⁴w = q, q a pressure in N/m²), same
    # as the brace-mass lumping above -- a concentrated point force has to
    # be converted to an equivalent pressure by dividing by the nodal area
    # (~dd²), or it solves correctly but comes out many orders of magnitude
    # too small (this was silently rounding every deflection to 0.000 mm).
    F[idxs] = f_i / (grid['dd']**2)
    diagK = K.diagonal()
    reg = 1e-8*np.mean(np.abs(diagK))
    Kreg = (K + sparse.diags(np.full(N, reg))).tocsc()
    try:
        defl = spsolve(Kreg, F)
    except Exception:
        defl = np.zeros(N)
    defl_at_bridge = float(np.average(defl[idxs], weights=w))
    defl_max = float(np.max(np.abs(defl))) if N else 0.0

    return {
        'ok': True, 'freqs': freqs_out, 'measShapes': meas_shapes,
        'fieldShapes': field_shapes, 'fieldNx': FIELD_NX, 'fieldNy': FIELD_NY,
        'deflBridgeMm': abs(defl_at_bridge)*1000.0, 'deflMaxMm': abs(defl_max)*1000.0,
        'massG': total_brace_mass_g, 'nNodes': N, 'geomPenalty': geom_penalty,
        'torqueResolved': torque_resolved, 'torqueRequested': abs(geo.get('bridgeTorqueNmm',0.0)) > 1e-9,
    }

# ── Objective ────────────────────────────────────────────────────────────
def objective(result, targets, defl_limit_mm, mass_weight, mass_norm_g=40.0, n_braces=None):
    if not result.get('ok'):
        return 1e6
    freqs = result['freqs']; shapes = result.get('measShapes')
    cost = 0.0
    for i, tgt in enumerate(targets):
        f_sim = freqs[i] if i < len(freqs) else None
        if f_sim is None:
            cost += 50.0*tgt.get('weight',1.0)
            continue
        err = abs(f_sim - tgt['freq'])
        tol = tgt.get('tol',5.0)
        # No flat deadband inside tolerance: a target that's merely
        # "within tolerance" costs less than one that's exactly on target,
        # not the same (zero) as before. Tested empirically against the
        # old all-or-nothing version on a 6-target case -- this roughly
        # doubled how many targets landed within tolerance and halved
        # total error, because the old version gave the search zero
        # incentive to keep refining a mode once it merely cleared the
        # bar, even if doing so would've helped a mode that hadn't.
        inside = min(err, tol)
        outside = max(0.0, err - tol)
        cost += tgt.get('weight',1.0) * ((inside/max(1.0,tgt['freq']))**2 * 40.0 + (outside/max(1.0,tgt['freq']))**2 * 400.0)
        if tgt.get('matchShape') and shapes is not None and i < len(shapes) and tgt.get('measShape') is not None:
            m = mac(shapes[i], tgt['measShape'])
            cost += tgt.get('weight',1.0) * (1.0-m) * 8.0
    over_defl = max(0.0, result['deflBridgeMm']/max(0.05,defl_limit_mm) - 1.0)
    # Deflection is closer to a real structural limit than a soft
    # preference -- weighted much harder than before, with an added cubic
    # term so it escalates steeply rather than staying comparable to a
    # single frequency-matching term even at fairly large violations.
    cost += (over_defl**2) * 900.0 + (over_defl**3) * 700.0
    cost += mass_weight * (result['massG']/mass_norm_g)
    # A free-topology search with a high brace cap will happily creep
    # toward it: a small extra brace barely adds mass, so the mass
    # penalty above gives almost no pushback even when that brace helps
    # a target by a marginal, not meaningful, amount. This adds a direct
    # cost per brace beyond a small free allowance (a real, minimal
    # bracing pattern needs at least a handful of members) -- so the
    # search only keeps a brace when it earns its keep by more than this
    # amount, not just "doesn't actively hurt."
    if n_braces is not None:
        cost += 0.35 * max(0, n_braces - 5)
    cost += result.get('geomPenalty', 0.0)
    return cost

# ── Pattern parameter encoders -> brace list ─────────────────────────────
def _clip_to_outline(fx, fy):
    return clamp01(fx), clamp01(fy)
def clamp01(v): return max(0.02, min(0.98, v))

# Brace material defaults -- overridden per-run from the payload's
# braceModulus/braceDensity so it's actually settable from the UI, without
# threading it through every encoder's parameter list.
BRACE_MATERIAL = {'modulus': 10e9, 'density': 420.0}
# Checked at the top of every search iteration so a run can be stopped
# mid-way (e.g. the person spots a setup mistake) instead of having to
# wait out however many iterations are left.
CANCEL_FLAG = {'stop': False}

def make_brace(x1,y1,x2,y2,height,width,modulus=None,density=None,curve_cp=None,curve_cp2=None,height_tip=None,width_tip=None):
    modulus = BRACE_MATERIAL['modulus'] if modulus is None else modulus
    density = BRACE_MATERIAL['density'] if density is None else density
    # 'curve3' (cubic, two control points) can represent a true S-curve/
    # inflection -- a single quadratic control point can only bow ONE way,
    # so it's structurally incapable of the sweep-in-then-back-out shape
    # falcate bracing actually has. 'curve' (quadratic) still covers the
    # simple one-direction bows used elsewhere (X-brace keystone, fan
    # bridge patch, falcate's own spine).
    btype = 'curve3' if curve_cp2 else ('curve' if curve_cp else 'straight')
    b = {'type':btype,
         'pts':[{'fx':clamp01(x1),'fy':clamp01(y1)},{'fx':clamp01(x2),'fy':clamp01(y2)}],
         'width':max(1.5,width), 'height':max(3.0,height), 'modulus':modulus, 'density':density,
         'widthTip': (max(1.0,width_tip) if width_tip is not None else None),
         'heightTip': (max(2.0,height_tip) if height_tip is not None else None)}
    if curve_cp: b['cp'] = {'fx':clamp01(curve_cp[0]), 'fy':clamp01(curve_cp[1])}
    if curve_cp2: b['cp2'] = {'fx':clamp01(curve_cp2[0]), 'fy':clamp01(curve_cp2[1])}
    return b

# All three fixed-topology templates below (X-brace, fan, falcate) were
# authored and verified against one specific reference soundhole position
# and size -- every (fx,fy) in them is really only correct FOR that hole.
# Changing the hole (a different preset, a custom trace, a resized hole)
# previously left the braces exactly where they were, since the encoders
# never received the hole at all -- only the search parameters. That's
# backwards: the soundhole is the actual anchor real bracing plans are
# built around, so the templates need to move and rescale WITH it, not
# assume it never changes.
#
# Fix: every encoder now takes the real geo dict and builds a small
# closure, hr(fx,fy), that maps a coordinate from the reference hole's
# frame onto wherever the CURRENT hole actually is -- translating by the
# hole-center offset and scaling by the hole-radius ratio. Every position
# in each template is passed through hr() before being used, so the whole
# pattern re-anchors itself to the real hole automatically.
_REF_HOLE_CX, _REF_HOLE_CY, _REF_HOLE_R = 0.382, 0.5, 0.1332
def hole_relative_fn(geo):
    holeCx = geo.get('holeCx', _REF_HOLE_CX); holeCy = geo.get('holeCy', _REF_HOLE_CY)
    holeR = geo.get('holeR', _REF_HOLE_R)
    scale = (holeR / _REF_HOLE_R) if _REF_HOLE_R > 1e-9 else 1.0
    def hr(fx, fy):
        return (holeCx + (fx - _REF_HOLE_CX) * scale, holeCy + (fy - _REF_HOLE_CY) * scale)
    return hr

# hr() above re-anchors a template to the real soundhole, but the templates
# were authored (and their curve bows tuned) against one reference body
# OUTLINE too -- a traced body whose proportions differ from that reference
# can still legitimately place a template's tip past the real outline even
# after the hole re-anchor. brace_geometry_penalty only discourages this via
# the search objective, which does nothing here: X-brace/fan/falcate never
# search position (only cross-section dims), so a penalty that can't change
# any position can't fix it either. Hard-clip instead, specifically for
# these fixed-position templates.
def _bezier_split(ctrl, t):
    """De Casteljau split of a Bezier curve (2=line, 3=quadratic, 4=cubic
    control points) at parameter t. Returns (left, right) control-point
    lists, each spanning the same number of points as ctrl and sharing the
    point at t as their shared endpoint."""
    pts = list(ctrl)
    left, right = [pts[0]], [pts[-1]]
    while len(pts) > 1:
        pts = [((1-t)*pts[i][0]+t*pts[i+1][0], (1-t)*pts[i][1]+t*pts[i+1][1]) for i in range(len(pts)-1)]
        left.append(pts[0]); right.append(pts[-1])
    right.reverse()
    return left, right

def _bezier_subrange(ctrl, t0, t1):
    """Control points for the portion of a Bezier curve spanning [t0,t1]
    of the original, via two De Casteljau splits -- exact for line/
    quadratic/cubic alike, so a clipped brace keeps its real curve shape
    instead of just moving its endpoints and leaving the bow untouched."""
    if t0 <= 0.0 and t1 >= 1.0:
        return ctrl
    if t1 < 1.0:
        ctrl, _ = _bezier_split(ctrl, t1)
        t0 = (t0/t1) if t1 > 1e-9 else 0.0
    if t0 > 0.0:
        _, ctrl = _bezier_split(ctrl, t0)
    return ctrl

def _clip_brace_to_outline(b, outline_frac, Lx, Ly, NS=40):
    """Truncates a brace's endpoints inward along its own path so neither
    end extends past the traced outline, preserving curve shape exactly via
    Bezier subdivision rather than just relocating the endpoints. Returns
    None if the entire brace falls outside (nothing left to keep)."""
    if not outline_frac or len(outline_frac) < 3:
        return b
    poly_phys = [(p['fx']*Lx, p['fy']*Ly) for p in outline_frac]
    pts = _sample_brace(b, Lx, Ly, NS)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    inside = points_in_polygon(xs, ys, poly_phys)
    if inside.all():
        return b
    if not inside.any():
        return None
    idxs = np.where(inside)[0]
    i0, i1 = int(idxs[0]), int(idxs[-1])
    t0 = pts[i0][3] if i0 == 0 else (pts[i0-1][3] + pts[i0][3]) / 2.0
    t1 = pts[i1][3] if i1 == NS else (pts[i1][3] + pts[i1+1][3]) / 2.0
    if t1 - t0 < 0.02:
        return None  # sliver too short to keep as a brace
    a, z = b['pts'][0], b['pts'][1]
    if b.get('type') == 'curve3':
        ctrl = [(a['fx'],a['fy']), (b['cp']['fx'],b['cp']['fy']), (b['cp2']['fx'],b['cp2']['fy']), (z['fx'],z['fy'])]
    elif b.get('type') == 'curve':
        ctrl = [(a['fx'],a['fy']), (b['cp']['fx'],b['cp']['fy']), (z['fx'],z['fy'])]
    else:
        ctrl = [(a['fx'],a['fy']), (z['fx'],z['fy'])]
    ctrl = _bezier_subrange(ctrl, t0, t1)
    nb = dict(b)
    nb['pts'] = [{'fx':clamp01(ctrl[0][0]),'fy':clamp01(ctrl[0][1])}, {'fx':clamp01(ctrl[-1][0]),'fy':clamp01(ctrl[-1][1])}]
    if b.get('type') == 'curve':
        nb['cp'] = {'fx':clamp01(ctrl[1][0]), 'fy':clamp01(ctrl[1][1])}
    elif b.get('type') == 'curve3':
        nb['cp'] = {'fx':clamp01(ctrl[1][0]), 'fy':clamp01(ctrl[1][1])}
        nb['cp2'] = {'fx':clamp01(ctrl[2][0]), 'fy':clamp01(ctrl[2][1])}
    return nb

def _clip_braces_to_outline(braces, geo):
    geo = geo or {}
    outline, Lx, Ly = geo.get('outline'), geo.get('Lx'), geo.get('Ly')
    if not outline or not Lx or not Ly:
        return braces
    out = []
    for b in braces:
        cb = _clip_brace_to_outline(b, outline, Lx, Ly)
        if cb is not None:
            out.append(cb)
    return out

def xbrace_encode(p, geo=None):
    """Fixed X-brace layout taken directly from a real project file the
    user built and exported (not reverse-engineered from a DXF sketch) --
    every position below is the actual data, converted from the paired
    Visualizer's own coordinate convention and re-anchored so the
    soundhole lands exactly where this tool's dreadnought preset expects
    it. This design is NOT bilaterally symmetric -- confirmed again here,
    independently of the earlier DXF-based version: two extra diagonal
    braces run from the treble-upper area to the bass-lower corner with
    no mirrored counterpart, matching what the DXF reference also showed.
    A transverse bar, the main X crossing, a bridge-area patch, and three
    pairs of wing/finger braces at different positions and angles fanning
    from the crossing area toward the lower bout. Only cross-section
    dimensions are searched; every position is fixed."""
    h,w,th,tw,fh,fw,f2h,f2w,f3h,f3w,eh,ew,wTipFrac,hTipFrac = p
    wTip, hTip = w*wTipFrac, h*hTipFrac
    hr = hole_relative_fn(geo or {})
    def hb(x1,y1,x2,y2,height,width,**kw):
        x1,y1 = hr(x1,y1); x2,y2 = hr(x2,y2)
        return make_brace(x1,y1,x2,y2,height,width,**kw)
    braces = [
        hb(0.24,0.80,0.24,0.20,th,tw),                      # transverse bar
        hb(0.76,0.09,0.32,0.79,h,w,height_tip=hTip,width_tip=wTip),  # main X arm 1
        hb(0.76,0.91,0.32,0.21,h,w,height_tip=hTip,width_tip=wTip),  # main X arm 2
        hb(0.53,0.568,0.53,0.436,th,tw),                    # bridge-area patch
        hb(0.407,0.676,0.271,0.602,fh,fw),                  # wing brace A, treble side (nearest the hole)
        hb(0.409,0.322,0.267,0.399,fh,fw),                  # wing brace A, bass side
        hb(0.460,0.846,0.600,0.632,f2h,f2w),                # wing brace B, treble side
        hb(0.603,0.381,0.462,0.158,f2h,f2w),                # wing brace B, bass side
        hb(0.562,0.906,0.671,0.733,f3h,f3w),                # wing brace C, treble side
        hb(0.677,0.272,0.569,0.100,f3h,f3w),                # wing brace C, bass side
        hb(0.691,0.769,0.86,0.10,eh,ew),                    # extra diagonal 1 -- NOT mirrored (real, deliberate asymmetry)
        hb(0.742,0.853,0.91,0.16,eh,ew),                    # extra diagonal 2 -- same direction, NOT mirrored
    ]
    return _clip_braces_to_outline(braces, geo)
XBRACE_BOUNDS = [(6,20),(2,12),(6,20),(3,14),(6,20),(2,10),(6,20),(2,10),(6,20),(2,10),(6,20),(2,10),(0.15,1.0),(0.15,1.0)]

def fan_encode(p, n_braces, geo=None):
    """Fixed classical fan layout, positions re-verified against a
    reference DXF that includes the actual body outline -- cross-checked
    between dreadnought and OM versions of the same plan for consistency.
    A wide closing bar near the neck block, a wide curved bridge-area
    patch arcing just above the hub, N rays fanning from the hub out
    toward the lower bout (one ray reaching almost straight down to the
    tail block), and a small tail brace. Only cross-section dimensions
    are searched; ray geometry and hub position are fixed."""
    h,w,barh,barw,th,tw,wTipFrac,hTipFrac = p
    wTip, hTip = w*wTipFrac, h*hTipFrac
    hr = hole_relative_fn(geo or {})
    hubx, huby = 0.52, 0.5
    fanspread, lenscale = 108.0, 0.42
    def curved(x1,y1,x2,y2,bow_amt,hh,ww):
        x1,y1 = hr(x1,y1); x2,y2 = hr(x2,y2)
        mx,my = (x1+x2)/2, (y1+y2)/2
        dx,dy = x2-x1, y2-y1
        L = math.hypot(dx,dy) or 1e-6
        nx_,ny_ = -dy/L, dx/L
        return make_brace(x1,y1,x2,y2,hh,ww,curve_cp=(mx+nx_*bow_amt,my+ny_*bow_amt))
    def hb(x1,y1,x2,y2,height,width,**kw):
        x1,y1 = hr(x1,y1); x2,y2 = hr(x2,y2)
        return make_brace(x1,y1,x2,y2,height,width,**kw)
    braces = [
        hb(0.19,0.18,0.19,0.82,barh,barw),        # closing bar, near the neck block
        curved(0.52,0.22,0.52,0.78,-0.05,th,tw),   # curved bridge-area patch, just above the hub
    ]
    for i in range(n_braces):
        t = (i/(max(1,n_braces-1))) - 0.5 if n_braces>1 else 0.0
        ang = 90 + t*fanspread
        rad = math.radians(ang)
        ex = hubx + math.sin(rad)*lenscale   # reference-frame endpoint; hb() below
        ey = huby + math.cos(rad)*lenscale   # transforms hub AND tip consistently
        braces.append(hb(hubx,huby,ex,ey,h,w,height_tip=hTip,width_tip=wTip))
    braces.append(hb(0.90,0.47,0.90,0.53,th*0.7,tw*0.7))   # small tail brace
    return _clip_braces_to_outline(braces, geo)
def fan_bounds(n_braces):
    return [(4,20),(2,12),(3,14),(2,10),(3,14),(2,9),(0.12,1.0),(0.12,1.0)]

def falcate_encode(p, geo=None):
    """Fixed falcate layout taken directly from a real project file the
    user built and exported (not reverse-engineered from a DXF sketch) --
    every position and curve control point below is the actual data,
    converted from the paired Visualizer's own coordinate convention
    (confirmed opposite of this tool's: there, fx=width/fy=length) and
    re-anchored so the soundhole lands exactly where this tool's
    dreadnought preset expects it. A main sweep pair (long, single-bow
    quadratic curves), a shorter secondary sweep pair nested alongside
    them, a V-strut-like pair nearer the hole, a mid-body connector pair,
    a central bridge-area patch, and a small tail brace that has no
    mirror partner in the source file (an intentional asymmetry, kept
    rather than forced into a mirror). Only cross-section dimensions are
    searched; every position/curve is fixed."""
    h1,w1,h2,w2,h3,w3,h4,w4,h5,w5,h6,w6,wTipFrac,hTipFrac = p
    w1Tip,h1Tip = w1*wTipFrac, h1*hTipFrac
    w2Tip,h2Tip = w2*wTipFrac, h2*hTipFrac
    hr = hole_relative_fn(geo or {})
    def hb(x1,y1,x2,y2,height,width,curve_cp=None,curve_cp2=None,**kw):
        x1,y1 = hr(x1,y1); x2,y2 = hr(x2,y2)
        if curve_cp is not None: curve_cp = hr(*curve_cp)
        if curve_cp2 is not None: curve_cp2 = hr(*curve_cp2)
        return make_brace(x1,y1,x2,y2,height,width,curve_cp=curve_cp,curve_cp2=curve_cp2,**kw)
    braces = [
        # Main sweep pair -- long, single-bow curves (quadratic, matching
        # the source file's own curve data exactly).
        hb(0.94,0.281,0.258,0.243,h1,w1,curve_cp=(0.795,0.647),height_tip=h1Tip,width_tip=w1Tip),
        hb(0.94,0.721,0.260,0.758,h1,w1,curve_cp=(0.767,0.340),height_tip=h1Tip,width_tip=w1Tip),
        # Secondary sweep pair -- shorter, nested alongside the main one.
        hb(0.855,0.138,0.541,0.207,h2,w2,curve_cp=(0.693,0.561),height_tip=h2Tip,width_tip=w2Tip),
        hb(0.866,0.859,0.544,0.762,h2,w2,curve_cp=(0.700,0.448),height_tip=h2Tip,width_tip=w2Tip),
        # V-strut-like pair, nearer the soundhole.
        hb(0.365,0.329,0.252,0.426,h3,w3),
        hb(0.260,0.579,0.373,0.673,h3,w3),
        # Mid-body connector pair.
        hb(0.685,0.108,0.689,0.324,h4,w4),
        hb(0.686,0.665,0.690,0.885,h4,w4),
        # Central bridge-area patch.
        hb(0.501,0.397,0.503,0.590,h5,w5),
        # Small tail brace -- no mirror in the source file; kept asymmetric.
        hb(0.92,0.421,0.888,0.533,h6,w6),
    ]
    return _clip_braces_to_outline(braces, geo)
FALCATE_BOUNDS = [(6,20),(2,10),(6,20),(2,10),(6,20),(3,14),(6,20),(2,10),(6,20),(3,14),(6,20),(2,10),(0.15,1.0),(0.15,1.0)]


# ── Simulated annealing over a continuous parameter vector ──────────────
async def sa_continuous(encode_fn, bounds, geo, targets, meas_xy, defl_limit_mm, mass_weight, iters, progress_cb, label, x0=None, temp_scale=1.0):
    n = len(bounds)
    x = np.array(x0) if x0 is not None else np.array([random.uniform(lo,hi) for lo,hi in bounds])
    def clip(v): return np.array([min(hi,max(lo,vv)) for vv,(lo,hi) in zip(v,bounds)])
    best_x, best_cost, best_result = x.copy(), 1e18, None
    cur_cost = 1e18
    cur_hole_violation = None  # set on first iteration once braces exist
    best_valid_x, best_valid_cost = None, 1e18
    n_modes = len(targets)
    for it in range(iters):
        if CANCEL_FLAG['stop']:
            break
        T = temp_scale * (0.995**it)
        if it % max(1, iters//150) == 0 or it == iters-1:
            progress_cb(label, it, iters, best_cost, best_result, trace=(cur_cost, T, encode_fn(best_x, geo)))
        step = np.array([(hi-lo) for lo,hi in bounds]) * (0.25*T + 0.02)
        cand = clip(x + np.random.normal(0,1,n)*step) if it>0 else x
        braces = encode_fn(cand, geo)
        # No brace material may ever occupy the soundhole -- enforced as a
        # ratchet, not a flat reject: a move is only rejected here if it
        # would make hole intrusion WORSE than the currently accepted
        # state, so a design that happens to start touching the hole (an
        # unlikely but possible starting point) can still take
        # intermediate steps to move clear, rather than being stuck
        # unable to accept ANY move that still touches the hole even a
        # little less than before. Once violation reaches zero, no
        # accepted move can ever push it back above zero.
        violation = brace_hole_violation(braces, geo['holeCx'], geo['holeCy'], geo['holeR'], geo['Lx'], geo['Ly'])
        if cur_hole_violation is not None and violation > cur_hole_violation + 1e-12:
            if it % max(1,iters//25) == 0 or it==iters-1:
                progress_cb(label, it, iters, best_cost, best_result)
            await asyncio.sleep(0)  # yield every iteration, not just on progress ticks -- keeps the page responsive during long runs
            continue
        result = evaluate_design(geo, braces, n_modes, meas_xy)
        for i,t in enumerate(targets):
            if result.get('ok') and result.get('measShapes') and i < len(result['measShapes']):
                pass
        cost = objective(result, targets, defl_limit_mm, mass_weight)
        accept = cost < cur_cost or random.random() < math.exp(-(cost-cur_cost)/max(1e-6,T*40))
        if accept:
            x, cur_cost, cur_hole_violation = cand, cost, violation
        if cost < best_cost:
            best_cost, best_x, best_result = cost, cand.copy(), result
        if violation <= 1e-12 and cost < best_valid_cost:
            best_valid_cost, best_valid_x = cost, cand.copy()
        if it % max(1,iters//25) == 0 or it==iters-1:
            progress_cb(label, it, iters, best_cost, best_result)
        await asyncio.sleep(0)
    if best_valid_x is not None:
        best_x, best_cost = best_valid_x, best_valid_cost
        best_result = evaluate_design(geo, encode_fn(best_x, geo), n_modes, meas_xy)
    elif best_result is None:
        best_result = evaluate_design(geo, encode_fn(best_x, geo), n_modes, meas_xy)
        best_cost = objective(best_result, targets, defl_limit_mm, mass_weight)
    return best_x, best_cost, best_result, encode_fn(best_x, geo)

def _outline_symmetry_axis_fy(outline_frac):
    """Area-weighted centroid fy of the traced outline -- the best single-
    line proxy for its real bilateral symmetry axis. A hand-traced body
    isn't necessarily centered at fy=0.5 (the middle of the drawing window
    itself) -- a lopsided trace should mirror around ITS OWN center, not
    that fixed constant. Falls back to 0.5 when there's no usable outline
    yet (nothing traced) or a degenerate one (zero/near-zero area)."""
    pts = outline_frac or []
    n = len(pts)
    if n < 3:
        return 0.5
    area2, cy_acc = 0.0, 0.0
    for i in range(n):
        x0, y0 = pts[i]['fx'], pts[i]['fy']
        x1, y1 = pts[(i+1) % n]['fx'], pts[(i+1) % n]['fy']
        cross = x0*y1 - x1*y0
        area2 += cross
        cy_acc += (y0+y1)*cross
    if abs(area2) < 1e-9:
        return 0.5
    cy = cy_acc / (3.0*area2)
    return min(0.9, max(0.1, cy))

# ── Simulated annealing over a variable-length free brace topology ──────
async def sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight, max_braces, iters, progress_cb, label, symmetric=True, warm_braces=None, temp_scale=1.0, materials=None, lock_positions=False, max_height=22.0, max_width=12.0, max_length_mm=None):
    """Unconstrained search has no reason to ever land on a left-right
    symmetric layout by chance -- a symmetric configuration is a
    vanishingly small slice of the full search space, so pure random
    mutation essentially never produces one. Real instrument bracing is
    almost always bilaterally symmetric (asymmetric mode shapes change
    the radiation pattern and skew the bass/treble balance), so when
    symmetric=True the search only ever touches ONE side of the traced
    outline's own symmetry axis (see _outline_symmetry_axis_fy -- NOT
    always fy=0.5; a lopsided trace mirrors around its own geometric
    center, not the center of the drawing window) and every brace's
    mirror image is added automatically at evaluation time -- symmetry is
    guaranteed by construction, not hoped for. max_braces is interpreted
    as the total (mirrored) count.

    materials, when given, is a list of {'modulus','density'} options the
    search can assign per brace (each new brace gets a random pick, and a
    mutation can flip an existing brace to a different one) -- currently
    only passed in for Experimental; plain Novel always uses the
    single global material, unchanged. A mirrored brace always gets the
    SAME material as its mirror partner, not an independent pick, since a
    real soundboard wouldn't have asymmetric material choice on an
    otherwise-symmetric layout.

    lock_positions, when True, restricts every mutation to cross-section
    resizing only -- no adding, removing, repositioning, or bending any
    brace, so the exact layout of a warm-started design (an uploaded
    "Custom" design in particular) stays exactly where it was drawn.
    Requires warm_braces; without an existing layout to lock, there's
    nothing to hold in place.

    max_height / max_width are hard ceilings (mm) on a brace's root
    cross-section -- the search can never sample, mutate, or hold a
    brace above these regardless of what the targets would otherwise
    reward. The existing lower floors (3mm height, 1.5mm width) are
    unchanged; only the ceiling is configurable. max_length_mm, when
    given, is a hard ceiling (mm) on tip-to-tip brace length, enforced
    by rescaling a too-long brace back down around its own midpoint --
    applied after every mutation (not just at creation), the same way
    the curve-bow safety net below re-clamps every iteration rather
    than trusting a one-time check."""
    half_cap = max(1, max_braces//2) if symmetric else max_braces
    sym_axis = _outline_symmetry_axis_fy((geo or {}).get('outline'))
    # Same margins as the old fixed (0.15, 0.47) had relative to 0.5 (0.35
    # far side, 0.03 near the axis) -- just re-centered on the real axis,
    # so behavior is unchanged for a body that IS centered at fy=0.5.
    fy_lo = max(0.03, sym_axis - 0.35) if symmetric else 0.15
    fy_hi = max(fy_lo+0.02, sym_axis - 0.03) if symmetric else 0.85
    min_height, min_width = 4.0, 2.0
    max_height = max(min_height, float(max_height))
    max_width = max(min_width, float(max_width))

    def enforce_length_cap(brace_list):
        """Rescale any brace whose tip-to-tip length exceeds max_length_mm
        back down to the limit, symmetric about its own midpoint so a
        length-mutation can't be undone by always favoring one endpoint."""
        if not max_length_mm:
            return
        Lx_mm, Ly_mm = geo['Lx']*1000.0, geo['Ly']*1000.0
        for b in brace_list:
            p0, p1 = b['pts'][0], b['pts'][1]
            dx, dy = (p1['fx']-p0['fx'])*Lx_mm, (p1['fy']-p0['fy'])*Ly_mm
            L = math.hypot(dx, dy)
            if L > max_length_mm and L > 1e-9:
                scale = max_length_mm / L
                mx, my = (p0['fx']+p1['fx'])/2, (p0['fy']+p1['fy'])/2
                hx, hy = (p1['fx']-p0['fx'])*scale/2.0, (p1['fy']-p0['fy'])*scale/2.0
                p0['fx'], p0['fy'] = clamp01(mx-hx), clamp01(my-hy)
                p1['fx'], p1['fy'] = clamp01(mx+hx), clamp01(my+hy)

    def rand_material():
        if not materials: return None
        return dict(random.choice(materials))

    def mirror_brace(b):
        m = dict(b, pts=[{'fx':pp['fx'], 'fy':clamp01(2*sym_axis-pp['fy'])} for pp in b['pts']])
        if b.get('cp'): m['cp'] = {'fx':b['cp']['fx'], 'fy':clamp01(2*sym_axis-b['cp']['fy'])}
        if b.get('cp2'): m['cp2'] = {'fx':b['cp2']['fx'], 'fy':clamp01(2*sym_axis-b['cp2']['fy'])}
        return m
    def expand(bl):
        if not symmetric: return bl
        out = []
        for b in bl:
            out.append(b); out.append(mirror_brace(b))
        return out

    def rand_brace():
        x1,y1 = random.uniform(0.1,0.9), random.uniform(fy_lo,fy_hi)
        x2,y2 = random.uniform(0.1,0.9), random.uniform(fy_lo,fy_hi)
        h = random.uniform(min_height,max_height); w = random.uniform(min_width,max_width)
        mat = rand_material()
        b = make_brace(x1,y1,x2,y2,h,w, height_tip=h*random.uniform(0.4,1.0), width_tip=w*random.uniform(0.4,1.0),
                        modulus=(mat['modulus'] if mat else None), density=(mat['density'] if mat else None))
        enforce_length_cap([b])
        p0,p1 = b['pts'][0], b['pts'][1]
        mx,my = (p0['fx']+p1['fx'])/2, (p0['fy']+p1['fy'])/2
        dx,dy = p1['fx']-p0['fx'], p1['fy']-p0['fy']; L = math.hypot(dx,dy) or 1e-6
        if random.random() < 0.4:
            nx_,ny_ = -dy/L, dx/L
            bow = random.uniform(-1,1) * min(0.15, L*0.45)
            b['type'] = 'curve'; b['cp'] = {'fx':clamp01(mx+nx_*bow), 'fy':clamp01(my+ny_*bow)}
        return b
    if warm_braces:
        src = warm_braces[::2] if symmetric else warm_braces
        # An uploaded/seed design may have more braces than this run's cap
        # allows -- truncate up front rather than starting from a state
        # that already violates it (the search could only ever remove
        # braces to recover, never add, until it happened to drop below).
        cap_now = half_cap if symmetric else max_braces
        if len(src) > cap_now:
            src = src[:cap_now]
        braces = [dict(b, pts=[dict(pp) for pp in b['pts']],
                       **({'cp':dict(b['cp'])} if b.get('cp') else {}),
                       **({'cp2':dict(b['cp2'])} if b.get('cp2') else {}))
                  for b in src]
        if materials:
            # Warm-started braces may come from a prior single-material run
            # (or a different material set) -- assign each a valid pick
            # from the CURRENT palette rather than keeping a stale value.
            for b in braces:
                mat = rand_material()
                if mat: b['modulus'], b['density'] = mat['modulus'], mat['density']
    else:
        braces = [rand_brace() for _ in range(random.randint(1, max(1,half_cap)))]
    # Whichever path built the starting set (random or warm-started), make
    # sure it actually satisfies today's hard limits before the search even
    # begins -- a warm start from a prior run (or a different limit
    # setting) could otherwise start out already violating a ceiling that
    # was only just dialed down.
    enforce_length_cap(braces)
    for b in braces:
        b['height'] = max(3.0, min(max_height, b['height']))
        b['width'] = max(1.5, min(max_width, b['width']))
        if b.get('heightTip') is not None: b['heightTip'] = max(2.0, min(b['height'], b['heightTip']))
        if b.get('widthTip') is not None: b['widthTip'] = max(1.0, min(b['width'], b['widthTip']))
    n_modes = len(targets)
    def evalb(bl):
        full = expand(bl)
        r = evaluate_design(geo, full, n_modes, meas_xy)
        return objective(r, targets, defl_limit_mm, mass_weight, n_braces=len(full)), r
    cur_cost, cur_result = evalb(braces)
    cur_hole_violation = brace_hole_violation(expand(braces), geo['holeCx'], geo['holeCy'], geo['holeR'], geo['Lx'], geo['Ly'])
    best_braces, best_cost, best_result = [dict(b) for b in braces], cur_cost, cur_result
    # Tracked separately from best_braces/best_cost above: the ratchet
    # only guarantees violation never INCREASES, so an early low-cost
    # candidate recorded while violation was still nonzero could otherwise
    # end up as the returned "best" even after the search finds a fully
    # clean design later with slightly higher cost. The real constraint is
    # "never any material in the hole" on the actual returned result, not
    # just "trending toward less violation" -- so the final return prefers
    # this whenever one exists.
    best_valid_braces, best_valid_cost = (None, 1e18) if cur_hole_violation > 1e-12 else (best_braces, best_cost)
    # Material-flip mutation gets its own slice of the move budget only
    # when a materials palette was actually given -- otherwise the move
    # probabilities below are unchanged from the single-material search.
    mat_move_cutoff = 0.06 if materials else 0.0
    for it in range(iters):
        if CANCEL_FLAG['stop']:
            break
        T = temp_scale*(0.995**it)
        if it % max(1, iters//150) == 0 or it == iters-1:
            progress_cb(label, it, iters, best_cost, best_result, trace=(cur_cost, T, expand(best_braces)))
        cand = [dict(b, pts=[dict(pp) for pp in b['pts']], **({'cp':dict(b['cp'])} if b.get('cp') else {})) for b in braces]
        move = random.random()
        if lock_positions and cur_hole_violation <= 1e-12:
            # Skip straight to the resize branch every time -- pts, cp,
            # and cp2 never get touched, so the layout can't drift from
            # what was drawn, only its cross-section dimensions can. This
            # only applies once the design is actually clear of the
            # soundhole -- "locked" positions still aren't allowed to
            # leave material sitting inside the hole, since that's a
            # physical impossibility no setting should be able to
            # override. If the seed starts with the hole violated,
            # reposition/bend stay available below until it's resolved.
            move = mat_move_cutoff + 0.60
        if materials and move < mat_move_cutoff and cand:
            # Flip one brace to a different material from the palette --
            # this is the actual mixed-material search: position/dimension
            # mutations below are unchanged, this just adds material choice
            # as its own independent variable per brace.
            b = cand[random.randrange(len(cand))]
            mat = rand_material()
            if mat: b['modulus'], b['density'] = mat['modulus'], mat['density']
        elif move < mat_move_cutoff+0.12 and len(cand) < half_cap:
            cand.append(rand_brace())
        elif move < mat_move_cutoff+0.24 and len(cand) > 1:
            cand.pop(random.randrange(len(cand)))
        elif move < mat_move_cutoff+0.55 and cand:
            b = cand[random.randrange(len(cand))]
            pt = b['pts'][random.randrange(2)]
            pt['fx'] = clamp01(pt['fx'] + random.uniform(-0.08,0.08)*T*2+random.uniform(-0.02,0.02))
            pt['fy'] = max(fy_lo, min(fy_hi, pt['fy'] + random.uniform(-0.08,0.08)*T*2+random.uniform(-0.02,0.02)))
        elif move < mat_move_cutoff+0.80 and cand:
            # Resize cross-section -- center and tip dimensions mutate somewhat
            # independently, so width and height can each end up tapered
            # differently (or not at all).
            b = cand[random.randrange(len(cand))]
            b['height'] = max(3.0, min(max_height, b['height'] + random.uniform(-2.5,2.5)))
            b['width'] = max(1.5, min(max_width, b['width'] + random.uniform(-1.5,1.5)))
            hTip = b.get('heightTip') if b.get('heightTip') is not None else b['height']
            wTip = b.get('widthTip') if b.get('widthTip') is not None else b['width']
            b['heightTip'] = max(2.0, min(b['height'], hTip + random.uniform(-2.0,2.0)))
            b['widthTip'] = max(1.0, min(b['width'], wTip + random.uniform(-1.5,1.5)))
        elif cand:
            # Bend or straighten this brace's centerline. The bow cap is
            # proportional to the brace's own chord length, not a fixed
            # absolute value -- a short brace given the same absolute cap
            # as a long one can end up with a bow bigger than its own
            # length, which doesn't read as "curved," it self-intersects
            # into a loop. (The exact same failure mode showed up in the
            # falcate template's spine earlier in this project, for the
            # same underlying reason.)
            b = cand[random.randrange(len(cand))]
            p0,p1 = b['pts'][0], b['pts'][1]
            mx,my = (p0['fx']+p1['fx'])/2, (p0['fy']+p1['fy'])/2
            dx,dy = p1['fx']-p0['fx'], p1['fy']-p0['fy']; L = math.hypot(dx,dy) or 1e-6
            nx_,ny_ = -dy/L, dx/L
            max_bow = min(0.2, L*0.45)
            cur_bow = 0.0
            if b.get('cp'):
                cur_bow = (b['cp']['fx']-mx)*nx_ + (b['cp']['fy']-my)*ny_
            new_bow = max(-max_bow, min(max_bow, cur_bow + random.uniform(-0.06,0.06)))
            if abs(new_bow) < 0.01:
                b['type'] = 'straight'; b.pop('cp', None)
            else:
                b['type'] = 'curve'; b['cp'] = {'fx':clamp01(mx+nx_*new_bow), 'fy':clamp01(my+ny_*new_bow)}
        # Safety net, independent of which move just ran: reposition can
        # shrink a brace's chord without ever touching its cp, so a bow
        # that was proportionate before can become disproportionate to
        # the NEW, shorter chord and loop -- re-clamp every curved brace
        # to its current chord every iteration, not just the one the bend
        # mutation happened to touch this time.
        enforce_length_cap(cand)
        for b in cand:
            if not b.get('cp'):
                continue
            p0,p1 = b['pts'][0], b['pts'][1]
            mx,my = (p0['fx']+p1['fx'])/2, (p0['fy']+p1['fy'])/2
            dx,dy = p1['fx']-p0['fx'], p1['fy']-p0['fy']; L = math.hypot(dx,dy) or 1e-6
            nx_,ny_ = -dy/L, dx/L
            bow = (b['cp']['fx']-mx)*nx_ + (b['cp']['fy']-my)*ny_
            max_bow = min(0.2, L*0.45)
            if abs(bow) > max_bow:
                bow = max(-max_bow, min(max_bow, bow))
                b['cp'] = {'fx':clamp01(mx+nx_*bow), 'fy':clamp01(my+ny_*bow)}
        # No brace material may ever occupy the soundhole -- enforced as a
        # ratchet, not a flat reject (see brace_hole_violation): a move is
        # only rejected here if it would make hole intrusion WORSE than
        # the currently accepted state, checked on the full (mirrored, if
        # symmetric) layout since that's what actually gets built.
        violation = brace_hole_violation(expand(cand), geo['holeCx'], geo['holeCy'], geo['holeR'], geo['Lx'], geo['Ly'])
        if violation > cur_hole_violation + 1e-12:
            if it % max(1,iters//25) == 0 or it==iters-1:
                progress_cb(label, it, iters, best_cost, best_result)
            await asyncio.sleep(0)  # yield every iteration, not just on progress ticks -- keeps the page responsive during long runs
            continue
        cost, result = evalb(cand)
        accept = cost < cur_cost or random.random() < math.exp(-(cost-cur_cost)/max(1e-6,T*40))
        if accept:
            braces, cur_cost, cur_result, cur_hole_violation = cand, cost, result, violation
        if cost < best_cost:
            best_cost, best_braces, best_result = cost, [dict(b) for b in cand], result
        if violation <= 1e-12 and cost < best_valid_cost:
            best_valid_cost, best_valid_braces = cost, [dict(b) for b in cand]
        if it % max(1,iters//25) == 0 or it==iters-1:
            progress_cb(label, it, iters, best_cost, best_result)
        await asyncio.sleep(0)
    def _finalize(bl):
        # The search above only discourages out-of-bounds material via a
        # soft cost penalty (brace_geometry_penalty), same as the hole --
        # it's not a hard reject, so a candidate that pokes past the traced
        # outline can still win if the rest of its score is good enough,
        # especially for a lopsided (non-symmetric) outline where the
        # mirrored half of a brace can land outside even when the searched
        # half doesn't. Hard-clip the actual returned design the same way
        # the fixed-topology patterns already are (via De Casteljau Bezier
        # subdivision, see _clip_braces_to_outline) so what's shown can
        # never have brace material past the border, then re-score against
        # the geometry that's actually being returned rather than the
        # pre-clip one the search explored.
        full = expand(bl)
        clipped = _clip_braces_to_outline(full, geo)
        r = evaluate_design(geo, clipped, n_modes, meas_xy)
        c = objective(r, targets, defl_limit_mm, mass_weight, n_braces=len(clipped))
        return clipped, c, r

    if best_valid_braces is not None:
        return _finalize(best_valid_braces)
    return _finalize(best_braces)


# ── Main entry point called from JS ──────────────────────────────────────
def _progress(label, it, iters, best_cost, best_result, trace=None):
    if trace is not None:
        cur_cost, T, braces = trace
        BRIDGE['onOptimizerTrace'](label, it, iters, float(best_cost), float(cur_cost), float(T),
                                   json.dumps(braces, default=float))
        return
    defl = best_result['deflBridgeMm'] if best_result and best_result.get('ok') else None
    freqs = best_result['freqs'] if best_result and best_result.get('ok') else []
    BRIDGE['onOptimizerProgress'](label, it, iters, float(best_cost),
                                   json.dumps(freqs), (float(defl) if defl is not None else None))

async def run_all_patterns(payload_json):
    try:
        CANCEL_FLAG['stop'] = False
        p = json.loads(payload_json)
        geo = p['geo']
        targets = p['targets']  # [{freq,tol,weight,matchShape,measShape:[...] or None}]
        meas_xy = None
        if p.get('measX') and len(p['measX']):
            meas_xy = (np.array(p['measX']), np.array(p['measY']))
        defl_limit_mm = p['deflLimitMm']; mass_weight = p['massWeight']
        iters = p['iters']; patterns = p['patterns']; fan_n = p.get('fanCount',5)
        max_braces = p.get('maxBraces',6)
        max_braces_exp = p.get('maxBracesExperimental', 20)
        novel_max_height = p.get('novelMaxHeightMm', 22.0)
        novel_max_width = p.get('novelMaxThicknessMm', 12.0)
        novel_max_length = p.get('novelMaxLengthMm')  # None/absent = no cap
        exp_max_height = p.get('novelExpMaxHeightMm', 22.0)
        exp_max_width = p.get('novelExpMaxThicknessMm', 12.0)
        exp_max_length = p.get('novelExpMaxLengthMm')  # None/absent = no cap
        # The experimental pattern always searches asymmetrically with a
        # much higher brace cap, so it has a substantially larger space to
        # cover than "novel" for the same iteration count -- used both for
        # its coarse search and its polish stage below, so it's not
        # shortchanged in either.
        exp_scale = min(3.0, max(1.5, (max_braces_exp / max(2, max_braces)) * 1.6))
        symmetric = p.get('symmetric', True)
        BRACE_MATERIAL['modulus'] = float(p.get('braceModulus', 10e9))
        BRACE_MATERIAL['density'] = float(p.get('braceDensity', 420.0))
        # Mixed-material search is Experimental only, per an explicit
        # request that plain Novel keep using a single material like every
        # other pattern. Two options for now (spruce, carbon fiber); the
        # UI's materials table is the source of truth for their properties.
        # Gated behind its own toggle (default off) -- it was previously
        # always wired up whenever Experimental was selected despite the
        # UI calling it "preview only, not implemented," which left no way
        # to actually opt in or out independent of the pattern itself.
        exp_materials = [
            {'modulus': float(p.get('expMatSpruceE', 10e9)), 'density': float(p.get('expMatSpruceRho', 420.0))},
            {'modulus': float(p.get('expMatCarbonE', 135e9)), 'density': float(p.get('expMatCarbonRho', 1600.0))},
        ] if p.get('enableMultiMaterial', False) else None
        # A design uploaded or picked from the current session, used ONLY
        # by the "custom" pattern below -- it does not feed into or alter
        # "novel" or "novelExperimental" in any way; those always start
        # from a random layout exactly as before, regardless of whether a
        # custom design is also selected. "custom" is a genuinely separate
        # pattern so both can be compared side by side in the same run,
        # rather than one silently changing the other's behavior.
        seed_braces = p.get('seedBraces') or None
        custom_symmetric = False  # always preserve the uploaded design's actual left-right structure; "lock brace positions" is the control that matters here now
        custom_lock_positions = p.get('customLockPositions', False)

        results = {}   # key -> (braces, coarse_cost, coarse_x)
        if 'xbrace' in patterns and not CANCEL_FLAG['stop']:
            coarse_x, cost, _, braces = await sa_continuous(xbrace_encode, XBRACE_BOUNDS, geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'xbrace')
            results['xbrace'] = (braces, cost, coarse_x)
        if 'fan' in patterns and not CANCEL_FLAG['stop']:
            enc = lambda pp, g: fan_encode(pp, fan_n, g)
            coarse_x, cost, _, braces = await sa_continuous(enc, fan_bounds(fan_n), geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'fan')
            results['fan'] = (braces, cost, coarse_x)
        if 'falcate' in patterns and not CANCEL_FLAG['stop']:
            coarse_x, cost, _, braces = await sa_continuous(falcate_encode, FALCATE_BOUNDS, geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'falcate')
            results['falcate'] = (braces, cost, coarse_x)
        if 'novel' in patterns and not CANCEL_FLAG['stop']:
            braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                    max_braces, iters, _progress, 'novel', symmetric,
                                                    max_height=novel_max_height, max_width=novel_max_width,
                                                    max_length_mm=novel_max_length)
            results['novel'] = (braces, cost, None)
        if 'novelExperimental' in patterns and not CANCEL_FLAG['stop']:
            # Same search, same mass penalty keeping it honest -- just with
            # the symmetry requirement and the tight brace-count cap both
            # removed, so it has a substantially larger space to explore
            # than "novel" for the same iteration count. exp_scale (above)
            # compensates, and carries into its polish stage below too.
            # This is also the only pattern given a materials palette --
            # spruce and carbon fiber -- so it can assign each brace its
            # own material rather than one fixed material for everything.
            iters_exp = int(iters * exp_scale)
            braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                    max_braces_exp, iters_exp, _progress, 'novelExperimental',
                                                    symmetric=False, materials=exp_materials,
                                                    max_height=exp_max_height, max_width=exp_max_width,
                                                    max_length_mm=exp_max_length)
            results['novelExperimental'] = (braces, cost, None)
        if 'custom' in patterns and seed_braces and not CANCEL_FLAG['stop']:
            # Optimizes the actual uploaded/selected design -- its own
            # brace count becomes the cap (rather than the Novel-pattern
            # settings above, which have nothing to do with a design that
            # didn't come from this search) so the result stays recognizably
            # the same layout, just re-tuned, rather than being free to grow
            # into something unrelated.
            custom_cap = max(1, len(seed_braces) if not custom_symmetric else (len(seed_braces)+1)//2 * 2)
            braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                    custom_cap, iters, _progress, 'custom',
                                                    symmetric=custom_symmetric, warm_braces=seed_braces,
                                                    lock_positions=custom_lock_positions)
            results['custom'] = (braces, cost, None)

        # ── Polish each coarse winner at full resolution ──
        # The search grid above is deliberately coarse for speed, and a
        # design that's near-optimal there isn't guaranteed to still be
        # optimal on a finer mesh -- deflection in particular is sensitive
        # to grid resolution right at the load point. Re-solving ONCE and
        # just accepting whatever that reveals produces exactly the
        # confusing "final jump" this replaces: instead, warm-start a
        # short follow-up search AT hi-fi resolution from the coarse
        # winner, so the result is actually re-optimized for the grid it's
        # reported on rather than just re-measured on it.
        final = {}
        hifi_geo = dict(geo); hifi_geo['GN'] = p.get('hifiGN', 55)
        display_geo = dict(geo); display_geo['GN'] = p.get('displayGN', 150)
        n_modes = len(targets)
        polish_iters = max(40, min(200, iters//4))
        polish_enabled = p.get('polishEnabled', True)
        ENCODERS = {
            'xbrace': (xbrace_encode, XBRACE_BOUNDS),
            'fan': (lambda pp, g: fan_encode(pp, fan_n, g), fan_bounds(fan_n)),
            'falcate': (falcate_encode, FALCATE_BOUNDS),
        }
        for key, (braces, coarse_cost, coarse_x) in results.items():
            if CANCEL_FLAG['stop']:
                r = evaluate_design(hifi_geo, braces, n_modes, meas_xy, want_field=True)
                final[key] = {'braces': braces, 'result': r}
                continue
            if not polish_enabled:
                # Skip straight to the coarse winner as-is, at hi-fi
                # resolution, with no re-optimization pass -- the final
                # display-resolution re-measurement below still runs
                # regardless, since that's a separate, cheap one-off
                # solve for reporting accuracy, not the (possibly slow)
                # re-optimization this toggle actually controls.
                polished_braces, polished_cost = braces, coarse_cost
                r = evaluate_design(hifi_geo, braces, n_modes, meas_xy, want_field=True)
            elif key == 'novel' or key == 'novelExperimental' or key == 'custom':
                lockp = False
                limit_kwargs = {}
                if key == 'novel':
                    sym, cap, this_polish_iters, mats = symmetric, max_braces, polish_iters, None
                    limit_kwargs = {'max_height': novel_max_height, 'max_width': novel_max_width, 'max_length_mm': novel_max_length}
                elif key == 'novelExperimental':
                    sym, cap, this_polish_iters, mats = False, max_braces_exp, int(polish_iters * exp_scale), exp_materials
                    limit_kwargs = {'max_height': exp_max_height, 'max_width': exp_max_width, 'max_length_mm': exp_max_length}
                else:  # custom
                    sym = custom_symmetric
                    cap = max(1, len(braces) if not sym else (len(braces)+1)//2 * 2)
                    this_polish_iters, mats = polish_iters, None
                    lockp = custom_lock_positions
                polished_braces, polished_cost, r = await sa_free_topology(
                    hifi_geo, targets, meas_xy, defl_limit_mm, mass_weight, cap,
                    this_polish_iters, _progress, key+'-polish', sym, warm_braces=braces, temp_scale=0.15,
                    materials=mats, lock_positions=lockp, **limit_kwargs)
            else:
                enc, bounds = ENCODERS[key]
                # Warm-start the polish search from the coarse winner's own
                # parameter vector -- previously this was discarded (the
                # coarse call site kept only the resulting braces, not the
                # parameters that produced them), so the polish stage was
                # silently starting over from a random point at hi-fi
                # resolution rather than refining the coarse result. That's
                # exactly what produced a cost spike at the start of polish:
                # not the design being "reset" by design, but a real bug
                # throwing away a perfectly good starting point.
                _, polished_cost, r, polished_braces = await sa_continuous(
                    enc, bounds, hifi_geo, targets, meas_xy, defl_limit_mm, mass_weight,
                    polish_iters, _progress, key+'-polish', x0=coarse_x, temp_scale=0.15)
            # One more solve at a MUCH higher resolution, purely for the
            # numbers actually displayed -- a finite-difference plate model
            # like this converges from above (coarse grids overestimate
            # stiffness, so frequencies drop as resolution increases,
            # especially for higher modes with shorter-wavelength shapes
            # that need more grid points to resolve). Doing this only ONCE
            # per pattern with the design already decided -- not during
            # search or polish, where it would run hundreds of times --
            # means better accuracy costs one extra solve, not a slower
            # search.
            if not CANCEL_FLAG['stop'] and display_geo['GN'] > hifi_geo['GN']:
                r_display = evaluate_design(display_geo, polished_braces, n_modes, meas_xy, want_field=True)
                if r_display.get('ok'):
                    r = r_display
                await asyncio.sleep(0)
            # Whichever branch above produced r (a direct call, right above
            # -- already has field data -- or the internal best-result from
            # sa_continuous/sa_free_topology's own search loop, which never
            # requests it since that would slow down every iteration), make
            # sure the design actually being kept has its animation field
            # data before it's packaged up -- one extra solve, same cost
            # class as the display-resolution recompute above, not
            # per-iteration.
            if not CANCEL_FLAG['stop'] and r.get('ok') and not r.get('fieldShapes'):
                field_geo = display_geo if display_geo['GN'] > hifi_geo['GN'] else hifi_geo
                r_field = evaluate_design(field_geo, polished_braces, n_modes, meas_xy, want_field=True)
                if r_field.get('ok'):
                    r['fieldShapes'] = r_field.get('fieldShapes')
                    r['fieldNx'] = r_field.get('fieldNx')
                    r['fieldNy'] = r_field.get('fieldNy')
                await asyncio.sleep(0)
            final[key] = {'braces': polished_braces, 'result': r}
            BRIDGE['onOptimizerProgress'](key, iters, iters, float(polished_cost),
                                           json.dumps(r.get('freqs',[])),
                                           float(r['deflBridgeMm']) if r.get('ok') else None)
            await asyncio.sleep(0)

        out = {}
        for key, fr in final.items():
            r = fr['result']
            out[key] = {
                'braces': fr['braces'],
                'ok': r.get('ok', False),
                'freqs': r.get('freqs', []),
                'measShapes': r.get('measShapes'),
                'fieldShapes': r.get('fieldShapes'),
                'fieldNx': r.get('fieldNx'),
                'fieldNy': r.get('fieldNy'),
                'deflBridgeMm': r.get('deflBridgeMm'),
                'deflMaxMm': r.get('deflMaxMm'),
                'massG': r.get('massG'),
                'torqueResolved': r.get('torqueResolved', True),
                'torqueRequested': r.get('torqueRequested', False),
            }
        BRIDGE['onOptimizerDone'](json.dumps(out), CANCEL_FLAG['stop'])
    except Exception as exc:
        import traceback
        BRIDGE['onOptimizerError'](f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

async def evaluate_manual_design(payload_json):
    """Evaluates a hand-drawn brace layout directly -- one evaluate_design
    call at full display resolution, no search, no iteration. This is what
    lets 'draw manually' skip the optimizer entirely: the braces the user
    actually drew are evaluated as-is, not used as a starting point for
    anything to search from."""
    try:
        p = json.loads(payload_json)
        geo = p['geo']
        targets = p.get('targets') or []
        meas_xy = None
        if p.get('measX') and len(p['measX']):
            meas_xy = (np.array(p['measX']), np.array(p['measY']))
        braces = p.get('braces') or []
        n_modes = max(1, len(targets)) if targets else 8
        r = evaluate_design(geo, braces, n_modes, meas_xy, want_field=True)
        out = {
            'braces': braces,
            'ok': r.get('ok', False),
            'freqs': r.get('freqs', []),
            'measShapes': r.get('measShapes'),
            'fieldShapes': r.get('fieldShapes'),
            'fieldNx': r.get('fieldNx'),
            'fieldNy': r.get('fieldNy'),
            'deflBridgeMm': r.get('deflBridgeMm'),
            'deflMaxMm': r.get('deflMaxMm'),
            'massG': r.get('massG'),
            'torqueResolved': r.get('torqueResolved', True),
            'torqueRequested': r.get('torqueRequested', False),
        }
        BRIDGE['onManualEvalDone'](json.dumps(out))
    except Exception as exc:
        import traceback
        BRIDGE['onManualEvalError'](f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

async def compute_bare_modes(geo_json):
    """Solve the plate with zero braces, so the person can see -- before
    setting targets -- roughly what's actually reachable. Bracing only ADDS
    mass and stiffness; stiffening dominates for realistic brace sizes, so
    frequencies can be pushed UP fairly easily but pushing a mode BELOW its
    bare-plate value takes an unrealistically soft/heavy brace, if it's
    reachable at all. A target below the matching bare mode is a request
    the physics can't really deliver, not something more search will fix."""
    try:
        geo = json.loads(geo_json)
        r = evaluate_design(geo, [], 8)
        BRIDGE['onBareModesResult'](json.dumps(r))
    except Exception as exc:
        import traceback
        BRIDGE['onBareModesError'](f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

def request_cancel_optimizer():
    CANCEL_FLAG['stop'] = True




# ══════════════════════════════════════════════════════════════════════════
# PARALLEL ORCHESTRATION — the actual "multithreading" (really multi-
# PROCESS, see note below) this native build exists for.
#
# run_all_patterns (above) is an exact copy of the browser build's
# sequential orchestrator: every requested pattern is solved one after
# another on a single thread. But every pattern was already fully
# independent of every other one — xbrace/fan/falcate/novel/
# novelExperimental/custom never read or write each other's state, coarse
# stage or polish stage. run_all_patterns_parallel runs each one in its own
# OS process instead, via ProcessPoolExecutor, so they solve concurrently
# across CPU cores. That's real parallelism (no GIL to contend with),
# unlike Python threads — and it's the one thing the browser build
# structurally cannot do: Pyodide pins everything to the page's single
# main JS thread, so even an embarrassingly parallel workload like this
# has always run strictly sequentially there.
#
# Trade-off made for this first experimental version: progress reporting
# is coarse-grained (a pattern is announced as started, then done) rather
# than streaming every search iteration like the sequential browser
# version's progress bar does. Streaming per-iteration progress across a
# process boundary needs its own IPC channel (a multiprocessing.Queue)
# drained continuously by the caller; that's a reasonable next step, not
# included here to keep this first port's surface area small.
# ══════════════════════════════════════════════════════════════════════════

def _solve_pattern_sync(key, p, cancel_event=None, progress_queue=None):
    """Runs in its own worker process (submitted via ProcessPoolExecutor).
    cancel_event, if given, is a multiprocessing.Manager().Event() shared
    with the parent -- a tiny watcher thread mirrors it into this
    process's own CANCEL_FLAG, which sa_continuous/sa_free_topology
    already check every iteration, so Stop works without changing any
    search-loop code. progress_queue, if given, is a
    multiprocessing.Manager().Queue() this process pushes periodic
    (label, it, iters, cost, freqsJson, deflMm) tuples onto -- the same
    shape onOptimizerProgress already expects -- so the UI's progress bar
    and log actually move during the search instead of jumping straight
    from 0 to 100% when the whole pattern finishes."""
    if cancel_event is not None:
        def _watch():
            while not cancel_event.is_set():
                time.sleep(0.1)
            CANCEL_FLAG['stop'] = True
        threading.Thread(target=_watch, daemon=True).start()
    return key, asyncio.run(_solve_pattern_async(key, p, progress_queue))


async def _solve_pattern_async(key, p, progress_queue=None):
    """One pattern's full coarse-search + polish + display-resolution
    pipeline, end to end. This is run_all_patterns's per-key body, pulled
    out unchanged so it can be called standalone (sequentially, by
    run_all_patterns above) or in its own process (by
    run_all_patterns_parallel below) -- same logic either way, not a
    reimplementation."""
    geo = p['geo']
    targets = p['targets']
    meas_xy = None
    if p.get('measX') and len(p['measX']):
        meas_xy = (np.array(p['measX']), np.array(p['measY']))
    defl_limit_mm = p['deflLimitMm']; mass_weight = p['massWeight']
    iters = p['iters']; fan_n = p.get('fanCount', 5)
    max_braces = p.get('maxBraces', 6)
    max_braces_exp = p.get('maxBracesExperimental', 20)
    novel_max_height = p.get('novelMaxHeightMm', 22.0)
    novel_max_width = p.get('novelMaxThicknessMm', 12.0)
    novel_max_length = p.get('novelMaxLengthMm')
    exp_max_height = p.get('novelExpMaxHeightMm', 22.0)
    exp_max_width = p.get('novelExpMaxThicknessMm', 12.0)
    exp_max_length = p.get('novelExpMaxLengthMm')
    exp_scale = min(3.0, max(1.5, (max_braces_exp / max(2, max_braces)) * 1.6))
    symmetric = p.get('symmetric', True)
    BRACE_MATERIAL['modulus'] = float(p.get('braceModulus', 10e9))
    BRACE_MATERIAL['density'] = float(p.get('braceDensity', 420.0))
    exp_materials = [
        {'modulus': float(p.get('expMatSpruceE', 10e9)), 'density': float(p.get('expMatSpruceRho', 420.0))},
        {'modulus': float(p.get('expMatCarbonE', 135e9)), 'density': float(p.get('expMatCarbonRho', 1600.0))},
    ] if p.get('enableMultiMaterial', False) else None
    seed_braces = p.get('seedBraces') or None
    custom_symmetric = False
    custom_lock_positions = p.get('customLockPositions', False)

    def _report_progress(label, it, iters_, best_cost, best_result, trace=None):
        if progress_queue is None:
            return
        if trace is not None:
            cur_cost, T, braces = trace
            try:
                progress_queue.put(('__trace__', label, it, iters_, float(best_cost), float(cur_cost), float(T),
                                    json.dumps(braces, default=float)))
            except Exception:
                pass
            return
        defl = best_result['deflBridgeMm'] if best_result and best_result.get('ok') else None
        freqs = best_result['freqs'] if best_result and best_result.get('ok') else []
        try:
            progress_queue.put((label, it, iters_, float(best_cost), json.dumps(freqs),
                                 (float(defl) if defl is not None else None)))
        except Exception:
            pass  # a progress-reporting hiccup should never take down the actual search

    coarse_x = None
    if key == 'xbrace':
        coarse_x, cost, _, braces = await sa_continuous(xbrace_encode, XBRACE_BOUNDS, geo, targets, meas_xy,
                                                defl_limit_mm, mass_weight, iters, _report_progress, 'xbrace')
    elif key == 'fan':
        enc = lambda pp, g: fan_encode(pp, fan_n, g)
        coarse_x, cost, _, braces = await sa_continuous(enc, fan_bounds(fan_n), geo, targets, meas_xy,
                                                defl_limit_mm, mass_weight, iters, _report_progress, 'fan')
    elif key == 'falcate':
        coarse_x, cost, _, braces = await sa_continuous(falcate_encode, FALCATE_BOUNDS, geo, targets, meas_xy,
                                                defl_limit_mm, mass_weight, iters, _report_progress, 'falcate')
    elif key == 'novel':
        braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                max_braces, iters, _report_progress, 'novel', symmetric,
                                                max_height=novel_max_height, max_width=novel_max_width,
                                                max_length_mm=novel_max_length)
    elif key == 'novelExperimental':
        iters_exp = int(iters * exp_scale)
        braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                max_braces_exp, iters_exp, _report_progress, 'novelExperimental',
                                                symmetric=False, materials=exp_materials,
                                                max_height=exp_max_height, max_width=exp_max_width,
                                                max_length_mm=exp_max_length)
    elif key == 'custom':
        if not seed_braces:
            return None
        custom_cap = max(1, len(seed_braces) if not custom_symmetric else (len(seed_braces)+1)//2 * 2)
        braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                custom_cap, iters, _report_progress, 'custom',
                                                symmetric=custom_symmetric, warm_braces=seed_braces,
                                                lock_positions=custom_lock_positions)
    else:
        return None

    hifi_geo = dict(geo); hifi_geo['GN'] = p.get('hifiGN', 55)
    display_geo = dict(geo); display_geo['GN'] = p.get('displayGN', 150)
    n_modes = len(targets)
    polish_iters = max(40, min(200, iters//4))
    polish_enabled = p.get('polishEnabled', True)
    ENCODERS = {
        'xbrace': (xbrace_encode, XBRACE_BOUNDS),
        'fan': (lambda pp, g: fan_encode(pp, fan_n, g), fan_bounds(fan_n)),
        'falcate': (falcate_encode, FALCATE_BOUNDS),
    }
    if CANCEL_FLAG['stop']:
        polished_braces, polished_cost = braces, cost
        r = evaluate_design(hifi_geo, braces, n_modes, meas_xy, want_field=True)
    elif not polish_enabled:
        polished_braces, polished_cost = braces, cost
        r = evaluate_design(hifi_geo, braces, n_modes, meas_xy, want_field=True)
    elif key in ('novel', 'novelExperimental', 'custom'):
        lockp = False
        limit_kwargs = {}
        if key == 'novel':
            sym, cap, this_polish_iters, mats = symmetric, max_braces, polish_iters, None
            limit_kwargs = {'max_height': novel_max_height, 'max_width': novel_max_width, 'max_length_mm': novel_max_length}
        elif key == 'novelExperimental':
            sym, cap, this_polish_iters, mats = False, max_braces_exp, int(polish_iters * exp_scale), exp_materials
            limit_kwargs = {'max_height': exp_max_height, 'max_width': exp_max_width, 'max_length_mm': exp_max_length}
        else:
            sym = custom_symmetric
            cap = max(1, len(braces) if not sym else (len(braces)+1)//2 * 2)
            this_polish_iters, mats = polish_iters, None
            lockp = custom_lock_positions
        polished_braces, polished_cost, r = await sa_free_topology(
            hifi_geo, targets, meas_xy, defl_limit_mm, mass_weight, cap,
            this_polish_iters, _report_progress, key+'-polish', sym, warm_braces=braces, temp_scale=0.15,
            materials=mats, lock_positions=lockp, **limit_kwargs)
    else:
        enc, bounds = ENCODERS[key]
        _, polished_cost, r, polished_braces = await sa_continuous(
            enc, bounds, hifi_geo, targets, meas_xy, defl_limit_mm, mass_weight,
            polish_iters, _report_progress, key+'-polish', x0=coarse_x, temp_scale=0.15)

    if not CANCEL_FLAG['stop'] and display_geo['GN'] > hifi_geo['GN']:
        r_display = evaluate_design(display_geo, polished_braces, n_modes, meas_xy, want_field=True)
        if r_display.get('ok'):
            r = r_display
    if not CANCEL_FLAG['stop'] and r.get('ok') and not r.get('fieldShapes'):
        field_geo = display_geo if display_geo['GN'] > hifi_geo['GN'] else hifi_geo
        r_field = evaluate_design(field_geo, polished_braces, n_modes, meas_xy, want_field=True)
        if r_field.get('ok'):
            r['fieldShapes'] = r_field.get('fieldShapes')
            r['fieldNx'] = r_field.get('fieldNx')
            r['fieldNy'] = r_field.get('fieldNy')

    return {
        'braces': polished_braces,
        'ok': r.get('ok', False),
        'freqs': r.get('freqs', []),
        'measShapes': r.get('measShapes'),
        'fieldShapes': r.get('fieldShapes'),
        'fieldNx': r.get('fieldNx'),
        'fieldNy': r.get('fieldNy'),
        'deflBridgeMm': r.get('deflBridgeMm'),
        'deflMaxMm': r.get('deflMaxMm'),
        'massG': r.get('massG'),
        'torqueResolved': r.get('torqueResolved', True),
        'torqueRequested': r.get('torqueRequested', False),
    }


# Shared between the API thread (which calls request_cancel_optimizer) and
# whichever thread is running run_all_patterns_parallel -- a second,
# process-safe cancel path alongside the plain CANCEL_FLAG dict the
# sequential/single-process functions above use directly.
REQUEST_CANCEL = {'event': None}

_orig_request_cancel_optimizer = request_cancel_optimizer

def request_cancel_optimizer():
    _orig_request_cancel_optimizer()
    if REQUEST_CANCEL['event'] is not None:
        REQUEST_CANCEL['event'].set()


def run_all_patterns_parallel(payload_json):
    """Drop-in replacement for run_all_patterns (same BRIDGE callbacks, same
    final onOptimizerDone/onOptimizerError payload shape) that runs each
    requested pattern in its own OS process instead of one after another.
    Synchronous/blocking by design -- app.py calls this from a background
    thread, not from the UI/API thread, so the GUI stays responsive while
    this blocks waiting on the worker processes.

    Progress streams in real time via a Manager().Queue() shared with every
    worker: a background thread here drains it continuously and forwards
    each message straight to onOptimizerProgress, so the browser build's
    progress bar/log behavior is preserved rather than jumping from 0% to
    100% only when a whole pattern finishes. Since patterns now run
    concurrently instead of strictly one after another, their progress
    messages interleave on the UI's single shared progress bar -- expect it
    to jump between patterns rather than climb smoothly start to finish,
    which is a real, new side effect of actual parallelism, not a bug."""
    import concurrent.futures as cf
    import multiprocessing as mp

    try:
        p = json.loads(payload_json)
        patterns = [k for k in p.get('patterns', []) if k != 'custom' or p.get('seedBraces')]
        if not patterns:
            BRIDGE['onOptimizerDone'](json.dumps({}), False)
            return

        CANCEL_FLAG['stop'] = False
        manager = mp.Manager()
        cancel_event = manager.Event()
        progress_queue = manager.Queue()
        REQUEST_CANCEL['event'] = cancel_event

        def _dispatch(msg):
            if msg and msg[0] == '__trace__':
                BRIDGE['onOptimizerTrace'](*msg[1:])
            else:
                BRIDGE['onOptimizerProgress'](*msg)

        stop_draining = threading.Event()
        def _drain_progress():
            while not stop_draining.is_set():
                try:
                    msg = progress_queue.get(timeout=0.2)
                except Exception:
                    continue
                try:
                    _dispatch(msg)
                except Exception:
                    pass
        drain_thread = threading.Thread(target=_drain_progress, daemon=True)
        drain_thread.start()

        out = {}
        cancelled = False
        max_workers = min(len(patterns), os.cpu_count() or 4)
        with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_solve_pattern_sync, key, p, cancel_event, progress_queue): key for key in patterns}
            for fut in cf.as_completed(futures):
                key = futures[fut]
                try:
                    _, result = fut.result()
                except Exception as exc:
                    BRIDGE['onOptimizerError'](f'{key}: {type(exc).__name__}: {exc}')
                    continue
                if result is not None:
                    out[key] = result
                    BRIDGE['onOptimizerProgress'](key, 1, 1, 0.0,
                                                   json.dumps(result.get('freqs', [])),
                                                   result.get('deflBridgeMm'))
                if cancel_event.is_set():
                    cancelled = True

        # Stop draining only after every worker has actually finished --
        # the queue can still hold a handful of messages pushed right
        # before a process exited, and those should still reach the UI.
        stop_draining.set()
        drain_thread.join(timeout=1.0)
        while True:
            try:
                msg = progress_queue.get_nowait()
            except Exception:
                break
            try:
                _dispatch(msg)
            except Exception:
                pass

        REQUEST_CANCEL['event'] = None
        BRIDGE['onOptimizerDone'](json.dumps(out), cancelled)
    except Exception as exc:
        import traceback
        BRIDGE['onOptimizerError'](f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

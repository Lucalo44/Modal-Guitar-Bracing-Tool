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
from scipy.sparse.linalg import eigsh, splu, LinearOperator
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

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
    'onOptimizerLog': lambda *a, **k: None,
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

SS_PIN_SPACING = 1.5   # simply-supported: one w=0 pin per this many grid spacings along the outline
SS_PIN_PENALTY = 10.0  # ...each this stiff, relative to the largest diagonal entry of K

def _dilate(mask):
    """mask grown by one node in all 8 directions (callers keep a margin, so no wrap-around)."""
    out = mask.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dj, 0), di, 1)
    return out

# Always pivot on the diagonal: K - sigma*M is (near) positive definite, and
# letting SuperLU pivot off it (threshold 0.001 before) broke the symmetric
# ordering on the cut-cell grid's ghost links, whose diagonals are tiny:
# GN 160 went from 4 s to 195 s with identical results.
FACTOR_PIVOT_THRESH = 0.0
ND_MIN_SIZE = 50000   # unknowns; above this, nested dissection beats SuperLU's MMD ordering

def _bfs_levels(G, start):
    """BFS distance of every node of the connected graph G (csr) from start."""
    lev = np.full(G.shape[0], -1); lev[start] = 0
    front, d = np.array([start]), 0
    while front.size:
        nb = np.unique(G[front].indices)
        nb = nb[lev[nb] < 0]
        d += 1; lev[nb] = d; front = nb
    return lev

def _nested_dissection(A, leaf=64):
    """Fill-reducing ordering for a symmetric-pattern matrix by recursive
    graph bisection. Each piece is split by the BFS level (from a
    pseudo-peripheral node) that halves its node count; the two halves are
    ordered first and that separator level last. On these 2D plate grids
    the fill grows about as n*log(n), where minimum degree (MMD) degrades
    badly past ~50k unknowns: at GN 200 (69k unknowns) it cuts the
    factorization from ~50 s to ~15 s."""
    G = sparse.csr_matrix(A, copy=True); G.data[:] = 1
    G = ((G + G.T) > 0).astype(np.int8).tocsr()
    G.setdiag(0); G.eliminate_zeros()
    order, todo = [], [(np.arange(G.shape[0]), False)]
    # Iterative post-order: a piece is pushed back as "emit its separator"
    # after its two halves, so the separators of outer cuts come out last.
    while todo:
        nodes, is_sep = todo.pop()
        if is_sep or nodes.size <= leaf:
            order.append(nodes); continue
        sub = G[nodes][:, nodes].tocsr()
        ncomp, lab = connected_components(sub, directed=False)
        if ncomp > 1:
            todo.extend((nodes[lab == c], False) for c in range(ncomp))
            continue
        lev = _bfs_levels(sub, int(np.argmax(_bfs_levels(sub, 0))))
        L = int(lev.max())
        if L < 3:
            order.append(nodes); continue
        cnt = np.cumsum(np.bincount(lev, minlength=L+1))
        m = min(max(int(np.searchsorted(cnt, nodes.size/2)), 1), L-1)
        todo.append((nodes[lev == m], True))
        todo.append((nodes[lev > m], False))
        todo.append((nodes[lev < m], False))
    return np.concatenate(order)

class _PermutedLU:
    """splu of P A P^T, solving A x = b."""
    def __init__(self, lu, p):
        self.lu, self.p = lu, p
    def solve(self, b):
        y = self.lu.solve(np.asarray(b)[self.p])
        x = np.empty_like(y); x[self.p] = y
        return x

def _factor_sym(A):
    """Sparse LU of a symmetric matrix, ordered for symmetry and pivoting
    on the diagonal: about half the fill and a third of the time of
    SuperLU's default ordering for these plate systems. Large systems (fine
    display grids) are pre-ordered by nested dissection instead."""
    A = sparse.csc_matrix(A)
    if A.shape[0] >= ND_MIN_SIZE:
        p = _nested_dissection(A)
        lu = splu(A[p][:, p].tocsc(), permc_spec='NATURAL', diag_pivot_thresh=FACTOR_PIVOT_THRESH,
                  options=dict(SymmetricMode=True))
        return _PermutedLU(lu, p)
    return splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=FACTOR_PIVOT_THRESH,
                options=dict(SymmetricMode=True))

EDGE_MODELS = ('cutcell', 'legacy')
CUT_SUBSAMPLE = 6            # cut-cell area fractions from a 6x6 sub-sample of each boundary cell
CUT_GHOST_WEIGHT = 1e-3      # weight of the regularizing rows centred on pure ghost nodes
CUT_MASS_FLOOR = 1e-6
CLAMP_SLOPE_SPACING = 1.5    # clamped: one zero-normal-slope pin per this many grid spacings
CLAMP_SLOPE_OFFSET = 0.45    # ...comparing w at +-this many grid spacings along the outline normal
GRID_OFFSET = (0.0, 0.0)     # test hook (VALIDATION_PLAN Stage 4): shifts the grid origin by this many cells

def build_grid(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN, edge_model='cutcell'):
    """Dispatch to the cut-cell grid (default) or the legacy one. See
    _build_grid_cutcell and _build_grid_legacy."""
    if edge_model == 'legacy':
        return _build_grid_legacy(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN)
    return _build_grid_cutcell(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN)

def _cell_fractions(X, Y, cand, dd, poly_local, hx, hy, hr):
    """Area fraction of each candidate node's cell [x +- dd/2, y +- dd/2]
    that lies inside the outline and outside the sound hole, from a
    CUT_SUBSAMPLE^2 sub-sample of the cell."""
    s = CUT_SUBSAMPLE
    off = (np.arange(s) + 0.5)/s - 0.5
    ox, oy = np.meshgrid(off*dd, off*dd); ox, oy = ox.ravel(), oy.ravel()
    px = (X[cand][:, None] + ox[None, :]).ravel(); py = (Y[cand][:, None] + oy[None, :]).ravel()
    ins = points_in_polygon(px, py, poly_local)
    if hr > 0:
        ins &= np.hypot(px - hx, py - hy) >= hr
    return ins.reshape(-1, s*s).mean(axis=1)

def _build_grid_cutcell(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN):
    """Cut-cell finite-difference plate grid. The plate's energy is
    integrated over the true outline and sound hole instead of over the
    staircase of grid nodes that happen to fall inside them:

      * every node whose cell [x +- dd/2, y +- dd/2] overlaps the plate is
        'active'; its curvature (and shear-strain) rows are weighted by the
        fraction of the cell that's plate, and so is its mass;
      * a ring of ghost nodes around the active set gives every active row
        its full 3x3 stencil. Ghosts carry (almost) no mass, and the only
        rows centred on them are complete second differences at a tiny
        weight (CUT_GHOST_WEIGHT), which vanish on any linear field, so
        they regularize the ghosts without touching rigid-body motion;
      * edges are then imposed on the TRUE outline, never on grid nodes:
          free (sound hole always; outline when bc_mode='free'): nothing --
            the weighted energy alone gives the natural condition;
          simply supported: w = 0 penalty pins on the outline (P);
          clamped: those pins plus zero normal slope, w(x + d n) =
            w(x - d n) with d = CLAMP_SLOPE_OFFSET*dd, at stations every
            CLAMP_SLOPE_SPACING*dd. The difference form fixes the slope
            without also forcing the curvature to zero.

    The legacy grid (_build_grid_legacy) clamped by fixing the nodes just
    outside the outline, which made the plate up to a cell too big (about
    -290/GN percent on the fundamental), and simply deleted curvature rows
    near free edges, which dropped a strip of bending energy (+16% on a
    free top's mode 2 between GN 50 and 200). Same return dict either way."""
    poly_phys = [(p['fx'] * Lx, p['fy'] * Ly) for p in poly_frac]
    xs = [p[0] for p in poly_phys]; ys = [p[1] for p in poly_phys]
    bbx0, bbx1 = min(xs), max(xs); bby0, bby1 = min(ys), max(ys)
    dd = max(bbx1 - bbx0, bby1 - bby0) / (GN - 1)
    margin = 4
    bbx0 -= (margin + GRID_OFFSET[0])*dd; bby0 -= (margin + GRID_OFFSET[1])*dd
    nx = int(round((bbx1 - bbx0) / dd)) + margin + 1
    ny = int(round((bby1 - bby0) / dd)) + margin + 1
    poly_local = [(x - bbx0, y - bby0) for x, y in poly_phys]
    hx_local = hole_cx * Lx - bbx0; hy_local = hole_cy * Ly - bby0
    hr = hole_r * min(Lx, Ly)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    X = (ii * dd).ravel(); Y = (jj * dd).ravel()
    in_poly = points_in_polygon(X, Y, poly_local).reshape(ny, nx)
    in_hole = (np.hypot(X - hx_local, Y - hy_local) < hr).reshape(ny, nx) if hr > 0 else np.zeros_like(in_poly)
    inside = in_poly & ~in_hole
    # Cells that can be cut: any node whose 8-neighbourhood isn't uniformly in or out.
    cand = (_dilate(inside) & _dilate(~inside)).ravel()
    frac = inside.astype(float).ravel()
    if cand.any():
        frac[cand] = _cell_fractions(X, Y, cand, dd, poly_local, hx_local, hy_local, hr)
    frac = frac.reshape(ny, nx)
    active = frac > 0
    unknown = _dilate(active)
    free = ~unknown   # absent nodes: nothing is fixed at a node any more

    node_id = -np.ones((ny, nx), dtype=np.int64)
    nodes_j, nodes_i = np.nonzero(unknown)
    N = nodes_i.size
    node_id[nodes_j, nodes_i] = np.arange(N)
    if int(inside.sum()) < 20:
        return {'N': 0, 'nx': nx, 'ny': ny, 'dd': dd}

    # Rows: every active node (weight = cell fraction) plus every ghost node,
    # at CUT_GHOST_WEIGHT but only where the stencil is complete.
    def complete(stencil):
        ok = np.ones_like(unknown)
        for di, dj, _ in stencil:
            ok &= np.roll(np.roll(unknown, -dj, 0), -di, 1)
        return ok
    sxx = [(-1,0,1.0), (0,0,-2.0), (1,0,1.0)]
    syy = [(0,-1,1.0), (0,0,-2.0), (0,1,1.0)]
    sxy = [(1,1,0.25), (1,-1,-0.25), (-1,1,-0.25), (-1,-1,0.25)]
    ghost_rows = unknown & ~active & complete(sxx) & complete(syy) & complete(sxy)
    centre = active | ghost_rows
    cj, ci = np.nonzero(centre)
    row_w = np.where(active[cj, ci], frac[cj, ci], CUT_GHOST_WEIGHT)
    rows = np.arange(ci.size)
    row_xy = np.stack([ci*dd + bbx0, cj*dd + bby0], axis=1)
    sw = np.sqrt(row_w)
    c2 = 1.0/(dd*dd)
    def op(stencil, scale):
        R, C, V = [], [], []
        for di, dj, c in stencil:
            nid = node_id[cj+dj, ci+di]
            ok = nid >= 0
            R.append(rows[ok]); C.append(nid[ok]); V.append(c*scale*sw[ok])
        return sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                                 shape=(max(ci.size, 1), N)).tocsr()
    Lxx = op(sxx, c2); Lyy = op(syy, c2); Lxy = op(sxy, c2)

    node_frac = frac[nodes_j, nodes_i]
    mass_scale = np.maximum(node_frac, CUT_MASS_FLOOR)
    Xmap, Sx, Sx_ends, wlx = _shear_links_cut(node_id, frac, unknown, dd, 1, 0)
    Ymap, Sy, Sy_ends, wly = _shear_links_cut(node_id, frac, unknown, dd, 0, 1)
    # Pins on the true outline: w = 0 (ss and clamped), plus, for clamped,
    # zero rotation of the cross-section. In a thin plate that's zero
    # normal slope. With plate shear the section rotation is
    # psi = -grad(w) + gamma, so a clamp fixes psi, not the slope: the
    # normal pin is w_n - gamma_n = 0 and a tangential pin gamma_t = 0
    # (w_t is already zero along a pinned edge). Pinning the slope alone
    # leaves the edge free to rotate through shear, which reads softer and
    # softer as the grid refines. The gamma parts go in P_gamma, which
    # evaluate_design only uses when plate shear is on.
    P = None; P_gamma = None
    nlx, nly = int(Sx.shape[0]), int(Sy.shape[0])
    if bc_mode in ('ss', 'clamped'):
        poly = np.array(poly_local + [poly_local[0]])
        seg = poly[1:] - poly[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        def stations(spacing):
            n_pins = max(8, int(np.ceil(s[-1] / (spacing*dd))))
            sq = np.linspace(0.0, s[-1], n_pins, endpoint=False)
            pts = np.stack([np.interp(sq, s, poly[:, 0]), np.interp(sq, s, poly[:, 1])], axis=1)
            k = np.clip(np.searchsorted(s, sq, side='right') - 1, 0, len(seg) - 1)
            t = seg[k] / np.maximum(seg_len[k], 1e-30)[:, None]
            return pts, np.stack([-t[:, 1], t[:, 0]], axis=1), t
        def bil(pts, lmap=None, half=(0.0, 0.0)):
            fx, fy = pts[:, 0]/dd - half[0], pts[:, 1]/dd - half[1]
            m = node_id if lmap is None else lmap
            my, mx = m.shape
            i0 = np.clip(np.floor(fx).astype(int), 0, mx-2); j0 = np.clip(np.floor(fy).astype(int), 0, my-2)
            tx, ty = np.clip(fx - i0, 0, 1), np.clip(fy - j0, 0, 1)
            pci = np.stack([i0, i0+1, i0, i0+1], 1); pcj = np.stack([j0, j0, j0+1, j0+1], 1)
            Wp = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], 1)
            return m[pcj, pci], Wp
        wblocks, gblocks, nrows = [], [], []
        pts, _, _ = stations(SS_PIN_SPACING)
        pid, Wp = bil(pts)
        wblocks.append((pid, Wp)); gblocks.append(None); nrows.append(len(pts))
        if bc_mode == 'ss':
            # Hard simple support (the Navier case): with plate shear the
            # section can't rotate ABOUT the edge's normal either, gamma_t = 0.
            pts, nrm, tng = stations(CLAMP_SLOPE_SPACING)
            d = CLAMP_SLOPE_OFFSET*dd
            gx_id, gx_w = bil(pts, Xmap, (0.5, 0.0)); gy_id, gy_w = bil(pts, Ymap, (0.0, 0.5))
            wblocks.append(None)
            gblocks.append((np.concatenate([gx_id, np.where(gy_id >= 0, gy_id + nlx, -1)], 1),
                            np.concatenate([2*d*tng[:, :1]*gx_w, 2*d*tng[:, 1:]*gy_w], 1)))
            nrows.append(len(pts))
        if bc_mode == 'clamped':
            pts, nrm, tng = stations(CLAMP_SLOPE_SPACING)
            d = CLAMP_SLOPE_OFFSET*dd
            idp, Wpp = bil(pts + d*nrm); idm, Wpm = bil(pts - d*nrm)
            gx_id, gx_w = bil(pts, Xmap, (0.5, 0.0)); gy_id, gy_w = bil(pts, Ymap, (0.0, 0.5))
            # normal rotation: w(+) - w(-) - 2d * n.gamma
            wblocks.append((np.concatenate([idp, idm], 1), np.concatenate([Wpp, -Wpm], 1)))
            gblocks.append((np.concatenate([gx_id, np.where(gy_id >= 0, gy_id + nlx, -1)], 1),
                            np.concatenate([-2*d*nrm[:, :1]*gx_w, -2*d*nrm[:, 1:]*gy_w], 1)))
            nrows.append(len(pts))
            # tangential rotation: 2d * t.gamma
            wblocks.append(None)
            gblocks.append((np.concatenate([gx_id, np.where(gy_id >= 0, gy_id + nlx, -1)], 1),
                            np.concatenate([2*d*tng[:, :1]*gx_w, 2*d*tng[:, 1:]*gy_w], 1)))
            nrows.append(len(pts))
        def stack(blocks, ncols):
            R, C, V = [], [], []; r0 = 0
            for blk, n in zip(blocks, nrows):
                if blk is not None:
                    ids, W = blk; ok = ids >= 0
                    prow = np.broadcast_to(np.arange(n)[:, None] + r0, ids.shape)
                    R.append(prow[ok]); C.append(ids[ok]); V.append(W[ok])
                r0 += n
            if not R:
                return sparse.csr_matrix((r0, ncols))
            return sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(r0, ncols)).tocsr()
        P = stack(wblocks, N)
        P_gamma = stack(gblocks, nlx + nly)

    def links_at(lmap, j, i):
        ok = (j >= 0) & (j < lmap.shape[0]) & (i >= 0) & (i < lmap.shape[1])
        out = np.full(j.shape, -1, dtype=np.int64)
        out[ok] = lmap[j[ok], i[ok]]
        return out
    def link_op(terms, n_cols):
        R, C, V = [], [], []
        for ids, c in terms:
            ok = ids >= 0
            R.append(rows[ok]); C.append(ids[ok]); V.append(c*sw[ok])
        return sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                                 shape=(max(ci.size, 1), n_cols)).tocsr()
    Gxx = link_op([(links_at(Xmap, cj, ci), 1.0/dd), (links_at(Xmap, cj, ci-1), -1.0/dd)], nlx)
    Gyy = link_op([(links_at(Ymap, cj, ci), 1.0/dd), (links_at(Ymap, cj-1, ci), -1.0/dd)], nly)
    q = 0.25/dd
    Gxy = link_op([(links_at(Xmap, cj+1, ci), q), (links_at(Xmap, cj+1, ci-1), q),
                   (links_at(Xmap, cj-1, ci), -q), (links_at(Xmap, cj-1, ci-1), -q)], nlx)
    Gyx = link_op([(links_at(Ymap, cj, ci+1), q), (links_at(Ymap, cj-1, ci+1), q),
                   (links_at(Ymap, cj, ci-1), -q), (links_at(Ymap, cj-1, ci-1), -q)], nly)
    return {'Lxx': Lxx, 'Lyy': Lyy, 'Lxy': Lxy, 'N': N, 'dd': dd, 'nx': nx, 'ny': ny,
            'nodes_i': nodes_i, 'nodes_j': nodes_j, 'node_id': node_id, 'bc_mode': bc_mode,
            'free_mask': free, 'P': P, 'P_gamma': P_gamma, 'mass_scale': mass_scale, 'row_xy': row_xy, 'row_w': row_w,
            'inside': inside, 'frac': frac, 'bbx0': bbx0, 'bby0': bby0, 'edge_model': 'cutcell',
            'nlx': nlx, 'nly': nly, 'Xmap': Xmap, 'Ymap': Ymap, 'Sx': Sx, 'Sy': Sy,
            'link_w': (wlx, wly),
            'Sx_ends': Sx_ends, 'Sy_ends': Sy_ends, 'Gxx': Gxx, 'Gyy': Gyy, 'Gxy': Gxy, 'Gyx': Gyx}

def _shear_links_cut(node_id, frac, unknown, dd, di, dj):
    """Shear-strain links for the cut-cell grid: one per pair of
    neighbouring unknowns with at least one active end. Each carries the
    mean of its two ends' cell fractions as its area weight (floored at
    CUT_GHOST_WEIGHT so ghost-ghost links stay regular). Returns the same
    (lmap, S, ends) as _shear_links plus the weights."""
    ny, nx = node_id.shape
    A = node_id[:ny-dj, :nx-di].ravel(); B = node_id[dj:, di:].ravel()
    fa = frac[:ny-dj, :nx-di].ravel(); fb = frac[dj:, di:].ravel()
    ua = unknown[:ny-dj, :nx-di].ravel(); ub = unknown[dj:, di:].ravel()
    keep = ua & ub & ((fa > 0) | (fb > 0))
    n = int(keep.sum())
    lmap = np.full(keep.size, -1, dtype=np.int64); lmap[keep] = np.arange(n)
    a_ids, b_ids = A[keep], B[keep]
    r = np.arange(n)
    S = sparse.coo_matrix((np.concatenate([np.full(n, -1.0/dd), np.full(n, 1.0/dd)]),
                           (np.concatenate([r, r]), np.concatenate([a_ids, b_ids]))),
                          shape=(n, int(unknown.sum()))).tocsr()
    w = np.maximum(0.5*(fa[keep] + fb[keep]), CUT_GHOST_WEIGHT)
    return lmap.reshape(ny-dj, nx-di), S, (a_ids, b_ids), w

def _build_grid_legacy(poly_frac, hole_cx, hole_cy, hole_r, Lx, Ly, bc_mode, GN):
    """Finite-difference plate grid: curvature operators Lxx, Lyy, Lxy and
    the boundary treatment, for bc_mode 'ss' (simply supported), 'clamped'
    or 'free' along the traced outline. The sound-hole edge is always free.

    Unknowns are the grid nodes inside the outline and outside the sound
    hole. Each curvature row is evaluated at one centre node, and the three
    operators share the same rows so evaluate_design can assemble the FULL
    orthotropic bending energy D11*kxx^2 + 2*D12*kxx*kyy + D22*kyy^2 +
    4*D66*kxy^2 -- including the D12 cross term, which only drops out for
    clamped edges and straight simply-supported ones.

    Edges:
      * free (sound hole always; outline when bc_mode='free'): no row is
        evaluated at a centre whose 3x3 neighbourhood reaches past the edge,
        which leaves the edge's natural (zero moment and shear) condition.
      * clamped: nodes just outside the outline are fixed at w=0, and
        curvature rows are ALSO evaluated at them, which penalizes edge slope.
      * simply supported: a one-node ghost layer outside the outline is left
        free to move, and w=0 is imposed by penalty at points on the TRUE
        outline (bilinear in the ghost/inside nodes; P, SS_PIN_*). Fixing w=0
        at the grid nodes themselves would also stop the staircase corners
        rotating -- on a curved edge that converges to the clamped plate
        instead (Babuska's paradox).
    Ghost nodes carry negligible mass (mass_scale).

    Checked against closed-form fundamentals for an isotropic circular plate
    (Leissa): simply supported within ~1%, clamped within ~2%, completely
    free within ~1% at GN 80-160."""
    poly_phys = [(p['fx'] * Lx, p['fy'] * Ly) for p in poly_frac]
    xs = [p[0] for p in poly_phys]; ys = [p[1] for p in poly_phys]
    bbx0, bbx1 = min(xs), max(xs); bby0, bby1 = min(ys), max(ys)
    dd = max(bbx1 - bbx0, bby1 - bby0) / (GN - 1)
    bbx0 -= 2*dd; bby0 -= 2*dd   # two spare nodes all round for ghost/support nodes and their neighbours
    nx = int(round((bbx1 - bbx0) / dd)) + 3
    ny = int(round((bby1 - bby0) / dd)) + 3
    poly_local = [(x - bbx0, y - bby0) for x, y in poly_phys]
    hx_local = hole_cx * Lx - bbx0; hy_local = hole_cy * Ly - bby0
    hr = hole_r * min(Lx, Ly)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    X = (ii * dd).ravel(); Y = (jj * dd).ravel()
    in_poly = points_in_polygon(X, Y, poly_local).reshape(ny, nx)
    hole = in_poly & (np.hypot(X - hx_local, Y - hy_local) < hr).reshape(ny, nx)
    inside = in_poly & ~hole
    supported = bc_mode in ('ss', 'clamped')
    ghost = (_dilate(inside) & ~in_poly) if bc_mode == 'ss' else np.zeros_like(inside)
    unknown = inside | ghost
    free = hole if supported else (hole | ~in_poly)   # absent nodes beyond a free edge

    node_id = -np.ones((ny, nx), dtype=np.int64)
    nodes_j, nodes_i = np.nonzero(unknown)
    N = nodes_i.size
    node_id[nodes_j, nodes_i] = np.arange(N)
    if N < 20:
        return {'N': N, 'nx': nx, 'ny': ny, 'dd': dd}

    centre = inside.copy()
    if bc_mode == 'clamped':
        centre |= _dilate(inside) & ~in_poly
    centre &= ~_dilate(free)
    cj, ci = np.nonzero(centre)
    rows = np.arange(ci.size)
    row_xy = np.stack([ci*dd + bbx0, cj*dd + bby0], axis=1)
    c2 = 1.0/(dd*dd)
    def op(stencil):
        R, C, V = [], [], []
        for di, dj, c in stencil:
            nid = node_id[cj+dj, ci+di]
            ok = nid >= 0
            R.append(rows[ok]); C.append(nid[ok]); V.append(np.full(ok.sum(), c))
        return sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                                 shape=(max(ci.size, 1), N)).tocsr()
    Lxx = op([(-1,0,c2), (0,0,-2*c2), (1,0,c2)])
    Lyy = op([(0,-1,c2), (0,0,-2*c2), (0,1,c2)])
    Lxy = op([(1,1,0.25*c2), (1,-1,-0.25*c2), (-1,1,-0.25*c2), (-1,-1,0.25*c2)])

    P = None
    if bc_mode == 'ss':
        poly = np.array(poly_local + [poly_local[0]])
        seg = poly[1:] - poly[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        n_pins = max(8, int(np.ceil(s[-1] / (SS_PIN_SPACING*dd))))
        sq = np.linspace(0.0, s[-1], n_pins, endpoint=False)
        pts = np.stack([np.interp(sq, s, poly[:, 0]), np.interp(sq, s, poly[:, 1])], axis=1)
        fx, fy = pts[:, 0]/dd, pts[:, 1]/dd
        i0 = np.clip(np.floor(fx).astype(int), 0, nx-2); j0 = np.clip(np.floor(fy).astype(int), 0, ny-2)
        tx, ty = fx - i0, fy - j0
        pci = np.stack([i0, i0+1, i0, i0+1], 1); pcj = np.stack([j0, j0, j0+1, j0+1], 1)
        Wp = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], 1)
        pid = node_id[pcj, pci]; ok = pid >= 0
        prow = np.broadcast_to(np.arange(n_pins)[:, None], pid.shape)
        P = sparse.coo_matrix((Wp[ok], (prow[ok], pid[ok])), shape=(n_pins, N)).tocsr()
    mass_scale = np.where(inside[nodes_j, nodes_i], 1.0, 1e-6)
    Xmap, Sx, Sx_ends = _shear_links(node_id, inside, unknown, free, bc_mode, dd, 1, 0)
    Ymap, Sy, Sy_ends = _shear_links(node_id, inside, unknown, free, bc_mode, dd, 0, 1)
    def links_at(lmap, j, i):
        ok = (j >= 0) & (j < lmap.shape[0]) & (i >= 0) & (i < lmap.shape[1])
        out = np.full(j.shape, -1, dtype=np.int64)
        out[ok] = lmap[j[ok], i[ok]]
        return out
    def link_op(terms, n_cols):
        R, C, V = [], [], []
        for ids, c in terms:
            ok = ids >= 0
            R.append(rows[ok]); C.append(ids[ok]); V.append(np.full(ok.sum(), c))
        return sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                                 shape=(max(ci.size, 1), n_cols)).tocsr()
    # Derivatives of the shear strains at every curvature row: compact
    # along a link's own direction, and across it with the same 2*dd span
    # as Lxy, so each twist row stays centred on its node.
    nlx, nly = int(Sx.shape[0]), int(Sy.shape[0])
    Gxx = link_op([(links_at(Xmap, cj, ci), 1.0/dd), (links_at(Xmap, cj, ci-1), -1.0/dd)], nlx)
    Gyy = link_op([(links_at(Ymap, cj, ci), 1.0/dd), (links_at(Ymap, cj-1, ci), -1.0/dd)], nly)
    q = 0.25/dd
    Gxy = link_op([(links_at(Xmap, cj+1, ci), q), (links_at(Xmap, cj+1, ci-1), q),
                   (links_at(Xmap, cj-1, ci), -q), (links_at(Xmap, cj-1, ci-1), -q)], nlx)
    Gyx = link_op([(links_at(Ymap, cj, ci+1), q), (links_at(Ymap, cj-1, ci+1), q),
                   (links_at(Ymap, cj, ci-1), -q), (links_at(Ymap, cj-1, ci-1), -q)], nly)
    return {'Lxx': Lxx, 'Lyy': Lyy, 'Lxy': Lxy, 'N': N, 'dd': dd, 'nx': nx, 'ny': ny,
            'nodes_i': nodes_i, 'nodes_j': nodes_j, 'node_id': node_id, 'bc_mode': bc_mode,
            'free_mask': free, 'P': P, 'mass_scale': mass_scale, 'row_xy': row_xy,
            'inside': inside, 'bbx0': bbx0, 'bby0': bby0,
            'nlx': nlx, 'nly': nly, 'Xmap': Xmap, 'Ymap': Ymap, 'Sx': Sx, 'Sy': Sy,
            'Sx_ends': Sx_ends, 'Sy_ends': Sy_ends, 'Gxx': Gxx, 'Gyy': Gyy, 'Gxy': Gxy, 'Gyx': Gyx}

def _shear_links(node_id, inside, unknown, free, bc_mode, dd, di, dj):
    """The grid links one through-thickness shear strain lives on, along one
    direction (di, dj): a link joins neighbouring nodes a and b and counts
    when either end is on the plate. Its other end is either another
    unknown, a clamped support (w = 0), or absent: beyond a free edge or the
    sound hole, and beyond a simply supported ghost node. Returns the link
    id at each link position (-1 where there's none), the (links x N) slope
    operator (w[b] - w[a]) / dd along each link, and each link's end ids
    (b is -1 at a support)."""
    ny, nx = node_id.shape
    A = node_id[:ny-dj, :nx-di].ravel(); B = node_id[dj:, di:].ravel()
    on_plate = (inside[:ny-dj, :nx-di] | inside[dj:, di:]).ravel()
    ua = unknown[:ny-dj, :nx-di].ravel(); ub = unknown[dj:, di:].ravel()
    fa = free[:ny-dj, :nx-di].ravel(); fb = free[dj:, di:].ravel()
    both = on_plate & ua & ub
    to_support = on_plate & (ua != ub) & (bc_mode == 'clamped') & ~(fa | fb)
    keep = both | to_support
    n = int(keep.sum())
    lmap = np.full(keep.size, -1, dtype=np.int64); lmap[keep] = np.arange(n)
    # At a support the link's one unknown end is a, so its slope is -w[a]/dd.
    sgn = np.where(ua, 1.0, -1.0)[keep]
    a_ids = np.where(ua, A, B)[keep]; b_ids = np.where(both, B, -1)[keep]
    r = np.arange(n); keep_b = b_ids >= 0
    S = sparse.coo_matrix((np.concatenate([-sgn/dd, np.full(int(keep_b.sum()), 1.0/dd)]),
                           (np.concatenate([r, r[keep_b]]), np.concatenate([a_ids, b_ids[keep_b]]))),
                          shape=(n, int(unknown.sum()))).tocsr()
    return lmap.reshape(ny-dj, nx-di), S, (a_ids, b_ids)

def _link_bilinear(grid, xs, ys, axis):
    """Bilinear interpolation of a link field (the x-links' strains for
    axis 0, the y-links' for axis 1) at physical points. Links sit halfway
    between their nodes, so their lattice is offset half a cell. Returns
    (ids, W) like _bilinear_corners; absent links count as zero."""
    lmap = grid['Xmap'] if axis == 0 else grid['Ymap']
    dd = grid['dd']; ny, nx = lmap.shape
    fx = (np.asarray(xs, dtype=float) - grid['bbx0']) / dd - (0.5 if axis == 0 else 0.0)
    fy = (np.asarray(ys, dtype=float) - grid['bby0']) / dd - (0.5 if axis == 1 else 0.0)
    i0 = np.clip(np.floor(fx).astype(int), 0, nx-2); j0 = np.clip(np.floor(fy).astype(int), 0, ny-2)
    tx = np.clip(fx - i0, 0.0, 1.0); ty = np.clip(fy - j0, 0.0, 1.0)
    ci = np.stack([i0, i0+1, i0, i0+1], axis=1); cj = np.stack([j0, j0, j0+1, j0+1], axis=1)
    W = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], axis=1)
    ids = lmap[cj, ci]
    return ids, np.where(ids >= 0, W, 0.0)

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

def braces_to_masses(braces, Lx, Ly, seg_len_mm=None, trapezoid=True):
    """Brace mass as point masses along each centreline. With trapezoid
    (v2), a straight brace's two end samples carry half a segment each, as a
    curved brace's already did; v1 gave every one of the NS+1 samples a full
    segment, one segment too much (about 1% extra mass on a long brace)."""
    xs, ys, ms = [], [], []
    for b in braces:
        wC = b['width']/1000.0; wT = (b['width'] if b.get('widthTip') is None else b['widthTip'])/1000.0
        hC = b['height']/1000.0; hT = (b['height'] if b.get('heightTip') is None else b['heightTip'])/1000.0
        straight = not (b.get('type') in ('curve', 'curve3') and b.get('cp') is not None)
        for x, y, sl, t in _sample_brace(b, Lx, Ly, _brace_ns(b, Lx, Ly, seg_len_mm)):
            if trapezoid and straight and (t <= 0.0 or t >= 1.0):
                sl = 0.5*sl
            A = _dim_at(wC, wT, t) * _dim_at(hC, hT, t)
            m = b.get('density', 350) * A * sl
            if m > 0:
                xs.append(min(0.999, max(0.001, x/Lx))*Lx); ys.append(min(0.999, max(0.001, y/Ly))*Ly); ms.append(m)
    return xs, ys, ms

def _bilinear_corners(grid, xs, ys):
    """Bilinear interpolation from the 4 grid nodes surrounding each physical
    point (xs, ys arrays, metres). Returns (ids, W), both shape (n, 4): DOF
    index of each corner (-1 where that corner isn't a DOF) and its weight.

    Corners that aren't DOFs are dropped. A supported (w=0) corner is
    exactly zero, so its weight is just removed. A corner beyond a free edge
    (sound hole, or a free outline) is genuinely absent, so its weight is
    shared out over the remaining corners instead; otherwise a rigid
    translation would read as curvature near the edge."""
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
    lost = np.where(grid['free_mask'][cj, ci], W, 0.0).sum(axis=1, keepdims=True)
    W = np.where(ids >= 0, W, 0.0) / np.maximum(1.0 - lost, 1e-12)
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

BRACE_FLANGE_WIDTH = 40.0   # width of top plate acting with each brace, in plate thicknesses. Chosen against
                            # an independent 3D solid model (tests/references/solid3d.py): single, parallel
                            # and crossing braces on a clamped spruce plate, mean error 1.7% at 40 against
                            # 2.0% at 20. The fundamental is the sensitive mode (about -6% at 20).
BRACE_SUBLINES = 1   # parallel lines each brace's bending is spread over (see brace_stiffness_matrix)
BRACE_SHEAR_FACTOR = 5.0/6.0   # Timoshenko shear correction for a rectangular web
PLATE_SHEAR_FACTOR = 5.0/6.0   # same correction for the plate's through-thickness shear (Reissner-Mindlin)

def brace_shear_modulus(b):
    """Shear modulus a brace resists shear and twist with: the along-grain
    ones (G_LR, G_LT) for wood cut with its grain along the brace, G_12 for
    unidirectional carbon. Braces from older projects carry no 'shear';
    they get a typical E/G for their kind -- about 15 for spruce-like wood,
    about 27 for unidirectional carbon."""
    G = b.get('shear')
    if G:
        return float(G)
    E = float(b.get('modulus', 9.79e9))
    return E / (27.0 if E > 60e9 else 15.0)

def isotropic_shear_modulus(E):
    """G = E/(2(1+nu)) -- what a CAD/FEA package gives a brace that is
    defined as an ordinary isotropic material (nu 0.4 for wood-like, 0.3 for
    carbon-like): about 3.5 GPa for spruce, several times the along-grain
    G_LR of real wood."""
    return float(E) / (2.0 * (1.0 + (0.3 if E > 60e9 else 0.4)))

def _rect_torsion_constant(w, h):
    """Saint-Venant torsion constant of a w x h rectangle (Roark)."""
    a = np.maximum(w, h); c = np.minimum(w, h); r = c / np.maximum(a, 1e-12)
    return a * c**3 * (1.0/3.0 - 0.21*r*(1.0 - r**4/12.0))

def _assemble_rows(ids, g, coef):
    """COO pieces of sum_r coef_r * (g_r . u[ids_r])^2 -- one quadratic
    energy term per row; ids of -1 (not a DOF) are dropped."""
    m = ids.shape[1]
    rows = np.broadcast_to(ids[:, :, None], ids.shape + (m,)).ravel()
    cols = np.broadcast_to(ids[:, None, :], ids.shape[:1] + (m, m)).ravel()
    vals = (coef[:, None, None] * g[:, :, None] * g[:, None, :]).ravel()
    keep = (rows >= 0) & (cols >= 0) & (vals != 0)
    return rows[keep], cols[keep], vals[keep]

def _flange_gaps(braces, Lx, Ly, stations):
    """For each brace's stations (xy, unit normals), the distance along +n
    and along -n to the nearest OTHER brace's centreline (inf where the ray
    meets none). Other braces are sampled every 1 mm and a sample counts as
    on the ray when it lies within 0.75 mm of it."""
    dense = []
    for b in braces:
        xy, _, _ = _brace_stations(b, Lx, Ly, 0.001)
        dense.append(xy if xy is not None else np.zeros((0, 2)))
    out = []
    for i, st in enumerate(stations):
        if st is None:
            out.append(None); continue
        xy, nrm = st
        others = [dense[j] for j in range(len(braces)) if j != i and len(dense[j])]
        if not others:
            out.append((np.full(len(xy), np.inf), np.full(len(xy), np.inf))); continue
        Q = np.concatenate(others)
        d = Q[None, :, :] - xy[:, None, :]                       # (stations, samples, 2)
        sn = d[..., 0]*nrm[:, None, 0] + d[..., 1]*nrm[:, None, 1]
        st_ = np.abs(-d[..., 0]*nrm[:, None, 1] + d[..., 1]*nrm[:, None, 0])
        on = st_ < 0.00075
        plus = np.where(on & (sn > 0), sn, np.inf).min(axis=1)
        minus = np.where(on & (sn < 0), -sn, np.inf).min(axis=1)
        out.append((plus, minus))
    return out

FOOTPRINT_COVER_MAX = 1.0   # the series form is exact (D_p + D_add) on a fully covered row, so only clip

def brace_footprint_terms(braces, Lx, Ly, grid, plate, iso_shear=False, step=0.001, Dmat=None, Gsh=(0.0, 0.0)):
    """Each brace's footprint as a stiff strip across its own width (P0.3).
    A line beam only stiffens bending ALONG the brace; in the real top (and
    in a 3D solid model) the glued footprint is also far stiffer than bare
    plate for bending ACROSS the brace and for through-thickness shear, which
    shortens the panels between braces. The footprint is rasterized every
    `step` (like the bridge) and, on the curvature rows it covers, adds the
    composite plate+brace bending stiffness across the brace (the brace's
    cross-grain modulus about the shared neutral axis, minus the plate's own
    bending, which the plate operator already counts) for the curvature
    across the brace, kappa_nn = n.kappa.n. Along-brace bending is left to
    brace_stiffness_matrix, so nothing is counted twice.

    Returns per-row coefficient arrays (a_xx, a_yy, a_12, a_x12, a_y12) for
    the energy sum_r [a_xx k_xx^2 + a_yy k_yy^2 + a_12 (k_xx k_yy + k_12^2/4...)]
    -- see evaluate_design -- plus the footprint's shear stiffness on each
    x- and y-link, or None without braces."""
    if not braces:
        return None
    tp, Ex, Ey, Gxy, nuxy = plate
    dd = grid['dd']; nrow = grid['row_xy'].shape[0]
    if Dmat is None:
        nyx = nuxy*Ey/Ex; den = 1.0 - nuxy*nyx
        Dmat = (Ex*tp**3/(12*den), Ey*tp**3/(12*den), nyx*Ex*tp**3/(12*den), Gxy*tp**3/12)
    PXs, PYs, Cs, NX, NY, GH, AR = [], [], [], [], [], [], []
    for b in braces:
        xy, t, hs = _brace_stations(b, Lx, Ly, step)
        if xy is None:
            continue
        tan = np.gradient(xy, axis=0); tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-30)[:, None]
        nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        wC = b.get('width', 3)/1000.0; wT = (b.get('width', 3) if b.get('widthTip') is None else b['widthTip'])/1000.0
        hC = b.get('height', 9)/1000.0; hT = (b.get('height', 9) if b.get('heightTip') is None else b['heightTip'])/1000.0
        Eb = float(b.get('modulus', 9.79e9))
        # across the brace: isotropic material -> same E; wood/carbon cut along the brace -> transverse modulus
        Et = Eb if iso_shear else (Eb*0.08 if Eb <= 60e9 else 8e9)
        Gt = isotropic_shear_modulus(Eb) if iso_shear else brace_shear_modulus(b)
        w = _dim_at(wC, wT, t); hb = _dim_at(hC, hT, t)          # per station
        nw = max(1, int(round(float(np.max(w))/step)))
        u = (np.arange(nw) + 0.5)/nw - 0.5                         # across-width sample positions, fraction of w
        c2 = nrm[:, 0]**2
        Ep = 1.0/(c2**2/Ex + (1-c2)**2/Ey + (1.0/Gxy - 2.0*nuxy/Ex)*c2*(1-c2))
        zp, zb = tp/2.0, tp + hb/2.0
        zn = (Ep*tp*zp + Et*hb*zb)/(Ep*tp + Et*hb)
        Dadd = Et*(hb**3/12.0 + hb*(zb - zn)**2) + Ep*tp*(zp - zn)**2
        offs = w[:, None]*u[None, :]                                # (stations, nw)
        PXs.append((xy[:, 0:1] + offs*nrm[:, 0:1]).ravel()); PYs.append((xy[:, 1:2] + offs*nrm[:, 1:2]).ravel())
        area = np.repeat(w/nw*hs, nw)
        Cs.append(np.repeat(Dadd, nw)*area); NX.append(np.repeat(nrm[:, 0], nw)); NY.append(np.repeat(nrm[:, 1], nw))
        GH.append(np.repeat(Gt*hb, nw)*area); AR.append(area)
    if not PXs:
        return None
    PX = np.concatenate(PXs); PY = np.concatenate(PYs); C = np.concatenate(Cs)
    nx_ = np.concatenate(NX); ny_ = np.concatenate(NY); GHa = np.concatenate(GH)
    area = np.concatenate(AR)
    # Series, not parallel: across the brace, the footprint strip and the
    # bare panel beside it share one row's curvature span, so their
    # compliances add. A row a fraction c of whose cell is footprint (strip
    # stiffness D_p + D_add against the plate's D_p in that direction) gets
    #   D_eff = D_p / (1 - c'),  c' = c * D_add / (D_p + D_add),
    # i.e. an extra D_p c'/(1 - c'). Fully covered rows (fine grids) become
    # near-rigid, as they should; a row a coarse grid only half covers gets
    # about twice the plate's stiffness, not a thousand times. c is capped
    # at FOOTPRINT_COVER_MAX.
    cov_s = area/(dd*dd)
    cov = _spread_to_rows(grid, PX, PY, cov_s)
    wsum = np.maximum(cov, 1e-30)
    mom = [_spread_to_rows(grid, PX, PY, cov_s*m)/wsum
           for m in (nx_**4, ny_**4, nx_**2*ny_**2, nx_**3*ny_, nx_*ny_**3)]
    Dadd = _spread_to_rows(grid, PX, PY, cov_s*C/np.maximum(area, 1e-30))/wsum
    Dp = Dmat[0]*mom[0] + Dmat[1]*mom[1] + 2.0*(Dmat[2] + 2.0*Dmat[3])*mom[2]
    c = np.minimum(cov, FOOTPRINT_COVER_MAX)
    cp = c*Dadd/np.maximum(Dp + Dadd, 1e-30)
    Dx = np.where(cov > 0, Dp*cp/(1.0 - cp), 0.0)
    acc = np.stack([Dx*m for m in mom])
    # Through-thickness shear, the same way: footprint G_b h_b in series with
    # the plate's h G on each link, by the fraction of the link it covers.
    cov_n = _spread_to_nodes(grid, PX, PY, cov_s)
    gh_n = _spread_to_nodes(grid, PX, PY, cov_s*GHa/np.maximum(area, 1e-30))
    def link_extra(ends, Gp_h):
        a_, b_ = ends
        bb = np.maximum(b_, 0)
        cl = np.where(b_ >= 0, 0.5*(cov_n[a_] + cov_n[bb]), cov_n[a_])
        gl = np.where(b_ >= 0, 0.5*(gh_n[a_] + gh_n[bb]), gh_n[a_])/np.maximum(cl, 1e-30)
        cl = np.minimum(cl, FOOTPRINT_COVER_MAX)
        cpl = cl*gl/np.maximum(Gp_h + gl, 1e-30)
        return np.where(cl > 0, Gp_h*cpl/(1.0 - cpl), 0.0)
    return {'rows': acc, 'link_gh': (link_extra(grid['Sx_ends'], Gsh[0]), link_extra(grid['Sy_ends'], Gsh[1]))}

def brace_stiffness_matrix(braces, Lx, Ly, grid, seg_len_mm=None, plate=None, iso_shear=False, plate_shear=False, flange_cap=True, bridge=None, flange_width=None, sublines=1):
    """Each brace as a Timoshenko beam embedded in the plate, plus its
    resistance to twisting. Returns (K, n_aux): K is (N+n_shear+n_aux)
    square, where n_shear counts the plate's shear strains (stored right
    after w; 0 without plate shear) and the n_aux extra unknowns after those
    are the braces' shear strains (no mass).

    With plate shear, a brace still follows the plate's deflection w, but
    it twists with the plate's cross-section rotation, the slope across the
    brace minus the plate's shear strain in that direction.

    Bending: energy 0.5*EI*(w'' - gamma')^2 along the arc length, with w at
    any point on the brace bilinearly interpolated from the 4 surrounding
    plate nodes (_bilinear_corners). Shear: 0.5*kappa*G*A*gamma^2, gamma
    living between stations. Eliminating gamma gives the classic Timoshenko
    softening EI/(1 + EI k^2/(kappa G A)) at wavenumber k along the brace --
    noticeable for a tall brace in wood (E/G ~ 15) and more so in carbon
    (~27) -- and as G grows it reduces exactly to the Euler-Bernoulli beam.
    Twist: 0.5*G*J*(d phi/ds)^2, phi the plate's slope across the brace
    (the brace is glued down, so it turns with that slope).

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
    The cross-slope phi is read the same way, from points half a grid
    spacing either side of the centreline.

    A brace is glued ON TOP of the plate, so it bends together with a strip
    of the plate about their shared neutral axis -- a T-section -- not about
    its own centroid. With plate=(h, Ex, Ey, Gxy, nu_xy) the brace's EI is
    that of the T-section (brace + BRACE_FLANGE_WIDTH*h of plate, using the
    plate's modulus along the brace's local direction), minus the strip's
    own bending, which the plate operator already counts. Without that
    offset a tall brace is several times too soft: checked against a 3D
    ANSYS model of an exported design (clamped edge), the fundamental moved
    from 15% low to within 1%. The web (the brace itself) carries the shear.

    Same per-unit-area scaling as the plate operator (hence /dd^2)."""
    dd = grid['dd']; N = grid['N']
    n_shear = grid['nlx'] + grid['nly'] if plate_shear else 0
    parts = []
    n_aux = 0
    # Flange cap (P0.4): each side of a brace's plate flange is at most half
    # the perpendicular gap to the next brace, so closely spaced braces
    # don't each count the same strip of plate.
    st_all = []
    for b in braces:
        xy_, t_, hs_ = _brace_stations(b, Lx, Ly, dd)
        if xy_ is None:
            st_all.append(None); continue
        tg = np.gradient(xy_, axis=0); tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-30)[:, None]
        st_all.append((xy_, np.stack([-tg[:, 1], tg[:, 0]], axis=1)))
    gaps = _flange_gaps(braces, Lx, Ly, st_all) if (flange_cap and plate is not None and len(braces) > 1) else None
    FW = BRACE_FLANGE_WIDTH if flange_width is None else float(flange_width)
    for bi, b in enumerate(braces):
        wC = b.get('width',3)/1000.0; wT = (b.get('width',3) if b.get('widthTip') is None else b['widthTip'])/1000.0
        hC = b.get('height',9)/1000.0; hT = (b.get('height',9) if b.get('heightTip') is None else b['heightTip'])/1000.0
        modulus = b.get('modulus', 9.79e9)
        G = isotropic_shear_modulus(modulus) if iso_shear else brace_shear_modulus(b)
        xy, t, hs = _brace_stations(b, Lx, Ly, dd)
        if xy is None:
            continue
        n = len(xy)
        a = np.array([1.0, -2.0, 1.0]) / (hs*hs)
        # Shear strains between stations: gamma_j sits between stations j and j+1.
        gam = N + n_shear + n_aux + np.arange(n-1)
        n_aux += n-1
        # Interior station k uses stations k-1, k, k+1 -> 12 (node, weight)
        # pairs, plus gamma' from the two strains either side of it.
        k = np.arange(1, n-1)
        # Local cross-section at each station, so a tapered brace is
        # correctly softer near its tips and stiffer through its middle.
        wk = _dim_at(wC, wT, t[k]); hk = _dim_at(hC, hT, t[k])
        EI = modulus * wk * hk**3 / 12.0
        if plate is not None:
            tp, Ex, Ey, Gxy, nuxy = plate
            # plate modulus along the brace's local direction (orthotropic, grain along x)
            d = xy[k+1] - xy[k-1]
            c2 = (d[:, 0]**2) / np.maximum((d**2).sum(axis=1), 1e-30); s2 = 1.0 - c2
            Ep = 1.0 / (c2**2/Ex + s2**2/Ey + (1.0/Gxy - 2.0*nuxy/Ex)*c2*s2)
            bf = FW * tp
            if gaps is not None and gaps[bi] is not None:
                half = 0.5*FW*tp
                gp, gm = gaps[bi][0][k], gaps[bi][1][k]
                bf = np.minimum(half, 0.5*gp) + np.minimum(half, 0.5*gm)
            Ab, Af = wk*hk, bf*tp
            zb, zf = tp + hk/2.0, tp/2.0              # centroids from the plate's underside
            zn = (modulus*Ab*zb + Ep*Af*zf) / (modulus*Ab + Ep*Af)
            EI_T = EI + modulus*Ab*(zb-zn)**2 + Ep*Af*(zf-zn)**2
            if bridge is not None:
                # Where the brace runs over the bridge (on the other face),
                # brace, plate and bridge bend as ONE section. The bridge rows
                # already count the plate+bridge composite over the flange, so
                # the brace adds the three-layer section minus that.
                bpoly, hbr, Ebr = bridge
                over = points_in_polygon(xy[k, 0], xy[k, 1], bpoly)
                if np.any(over):
                    Abr = bf*hbr; zr = -hbr/2.0
                    EA3 = modulus*Ab + Ep*Af + Ebr*Abr
                    zn3 = (modulus*Ab*zb + Ep*Af*zf + Ebr*Abr*zr)/EA3
                    EI3 = (modulus*(wk*hk**3/12.0 + Ab*(zb-zn3)**2) + Ep*(bf*tp**3/12.0 + Af*(zf-zn3)**2)
                           + Ebr*(bf*hbr**3/12.0 + Abr*(zr-zn3)**2))
                    zn2 = (Ep*Af*zf + Ebr*Abr*zr)/(Ep*Af + Ebr*Abr)
                    EI2 = Ep*(bf*tp**3/12.0 + Af*(zf-zn2)**2) + Ebr*(bf*hbr**3/12.0 + Abr*(zr-zn2)**2)
                    EI_T = np.where(over, EI3 - EI2, EI_T)
            EI = EI_T
        # Sub-lines: the bending energy is shared between n_sub parallel lines
        # spread across max(dd, brace width) about the centreline, all on the
        # same shear strains. A single line reads the bilinear field along
        # one track, so a brace sitting on a row of nodes and one halfway
        # between rows bent different plate strips: on a 15 mm grid the
        # frequency rippled about +-2.5% with a period of one cell as a brace
        # moved across it. Two lines half a spread apart remove most of that.
        tg = np.gradient(xy, axis=0); tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-30)[:, None]
        nr = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
        spread = np.maximum(dd, _dim_at(wC, wT, t)) if sublines > 1 else np.zeros(n)
        offs = ((np.arange(sublines) + 0.5)/sublines - 0.5) if sublines > 1 else np.zeros(1)
        for o in offs:
            xs_ = xy + (o*spread)[:, None]*nr
            ids, W = _bilinear_corners(grid, xs_[:, 0], xs_[:, 1])
            g_ids = np.concatenate([ids[k-1], ids[k], ids[k+1], gam[k][:, None], gam[k-1][:, None]], axis=1)
            g = np.concatenate([a[0]*W[k-1], a[1]*W[k], a[2]*W[k+1],
                                np.full((len(k), 1), -1.0/hs), np.full((len(k), 1), 1.0/hs)], axis=1)
            parts.append(_assemble_rows(g_ids, g, EI * hs / (dd*dd) / len(offs)))
        # Shear and twist use the section halfway between stations.
        tm = 0.5*(t[:-1] + t[1:])
        wm = _dim_at(wC, wT, tm); hm = _dim_at(hC, hT, tm)
        parts.append(_assemble_rows(gam[:, None], np.ones((n-1, 1)), BRACE_SHEAR_FACTOR*G*wm*hm * hs / (dd*dd)))
        # Cross-slope at each station, from points dd/2 either side.
        tan = np.gradient(xy, axis=0)
        tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-30)[:, None]
        nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        dl = 0.5*dd
        idsP, WP = _bilinear_corners(grid, xy[:, 0] + dl*nrm[:, 0], xy[:, 1] + dl*nrm[:, 1])
        idsM, WM = _bilinear_corners(grid, xy[:, 0] - dl*nrm[:, 0], xy[:, 1] - dl*nrm[:, 1])
        p_ids = np.concatenate([idsP, idsM], axis=1); p_w = np.concatenate([WP, -WM], axis=1) / (2.0*dl)
        if n_shear:
            # minus the plate's shear strain across the brace, n . (gamma_x, gamma_y)
            gx_ids, gx_w = _link_bilinear(grid, xy[:, 0], xy[:, 1], 0)
            gy_ids, gy_w = _link_bilinear(grid, xy[:, 0], xy[:, 1], 1)
            p_ids = np.concatenate([p_ids, np.where(gx_ids >= 0, gx_ids + N, -1),
                                    np.where(gy_ids >= 0, gy_ids + N + grid['nlx'], -1)], axis=1)
            p_w = np.concatenate([p_w, -nrm[:, :1]*gx_w, -nrm[:, 1:]*gy_w], axis=1)
        j = np.arange(n-1)
        t_ids = np.concatenate([p_ids[j+1], p_ids[j]], axis=1)
        t_w = np.concatenate([p_w[j+1], -p_w[j]], axis=1) / hs
        parts.append(_assemble_rows(t_ids, t_w, G*_rect_torsion_constant(wm, hm) * hs / (dd*dd)))
    size = N + n_shear + n_aux
    if not parts:
        return sparse.csr_matrix((size, size)), 0
    rows = np.concatenate([p[0] for p in parts]); cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
    K.sum_duplicates()
    return K, n_aux

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
HOLE_CLEARANCE_M = 0.001   # minimum gap between any brace's footprint and the sound-hole edge

def _brace_ctrl(b):
    """Bezier control points (fractional coords) for a straight, quadratic or cubic brace."""
    a, z = b['pts'][0], b['pts'][1]
    if b.get('type') == 'curve3' and b.get('cp') and b.get('cp2'):
        return [(a['fx'],a['fy']), (b['cp']['fx'],b['cp']['fy']), (b['cp2']['fx'],b['cp2']['fy']), (z['fx'],z['fy'])]
    if b.get('type') == 'curve' and b.get('cp'):
        return [(a['fx'],a['fy']), (b['cp']['fx'],b['cp']['fy']), (z['fx'],z['fy'])]
    return [(a['fx'],a['fy']), (z['fx'],z['fy'])]

def _brace_points(b, Lx, Ly, t):
    """Centreline points (metres) at Bezier parameters t (array) -- vectorized."""
    c = np.array(_brace_ctrl(b), dtype=float) * np.array([Lx, Ly])
    t = np.asarray(t, dtype=float)[:, None]; u = 1.0 - t
    if len(c) == 4:
        return u**3*c[0] + 3*u**2*t*c[1] + 3*u*t**2*c[2] + t**3*c[3]
    if len(c) == 3:
        return u**2*c[0] + 2*u*t*c[1] + t**2*c[2]
    return u*c[0] + t*c[1]

def _hole_intrusion(b, hole_cx, hole_cy, hole_r, Lx, Ly):
    """(t, intrusion, ds) along a brace, sampled about every 1 mm: intrusion
    is how far (m) the brace's footprint, plus HOLE_CLEARANCE_M, reaches
    into the sound hole at each sample (<= 0 where it's clear)."""
    c = np.array(_brace_ctrl(b), dtype=float) * np.array([Lx, Ly])
    approx_len = float(np.sum(np.hypot(*np.diff(c, axis=0).T)))
    n = int(min(600, max(16, approx_len / 0.001)))
    t = np.linspace(0.0, 1.0, n + 1)
    p = _brace_points(b, Lx, Ly, t)
    ds = np.hypot(*np.gradient(p, axis=0).T)
    wC = b.get('width', 3.0)/1000.0
    wT = (b.get('width', 3.0) if b.get('widthTip') is None else b['widthTip'])/1000.0
    keepout = hole_r*min(Lx, Ly) + _dim_at(wC, wT, t)/2.0 + HOLE_CLEARANCE_M
    d = np.hypot(p[:, 0] - hole_cx*Lx, p[:, 1] - hole_cy*Ly)
    return t, keepout - d, ds

def brace_hole_violation(braces, hole_cx, hole_cy, hole_r, Lx, Ly):
    """How much brace material reaches into the sound hole (0 if none):
    the intrusion depth integrated along each brace, sampled about every
    1 mm and including each brace's local half-width plus HOLE_CLEARANCE_M.
    Used as a ratchet in the searches -- a move may never INCREASE it -- so
    a design that starts out touching the hole can still move clear
    gradually, but nothing accepted can ever add new intrusion. (It used to
    be sampled at only 15 points per brace, which a long brace running past
    the hole could slip between.) The returned design is also trimmed
    exactly at the keep-out circle (_clip_braces_to_outline), so this is a
    search aid, not the final guarantee."""
    if not (hole_r > 0):
        return 0.0
    total = 0.0
    for b in braces:
        _, depth, ds = _hole_intrusion(b, hole_cx, hole_cy, hole_r, Lx, Ly)
        total += float(np.sum(np.clip(depth, 0.0, None) * ds))
    return total

OUTLINE_RASTER_M = 0.0005   # outline is rasterized at this resolution for fast inside/outside lookups
_OUTLINE_MASKS = {}

def _outline_mask(geo):
    """Inside-the-outline raster (bool [ny, nx], plus origin and step), built
    once per outline by scanline fill and cached -- a traced outline can
    have thousands of points, far too many to test against polygon-by-
    polygon on every search move."""
    outline, Lx, Ly = geo['outline'], geo['Lx'], geo['Ly']
    key = (id(outline), len(outline), Lx, Ly)
    if key in _OUTLINE_MASKS:
        return _OUTLINE_MASKS[key]
    poly = np.array([(p['fx']*Lx, p['fy']*Ly) for p in outline], dtype=float)
    step = OUTLINE_RASTER_M
    x0, y0 = poly[:, 0].min() - step, poly[:, 1].min() - step
    nx = int(np.ceil((poly[:, 0].max() - x0)/step)) + 2
    ny = int(np.ceil((poly[:, 1].max() - y0)/step)) + 2
    mask = np.zeros((ny, nx), dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for r in range(ny):
        y = y0 + (r + 0.5)*step
        c = (y1 > y) != (y2 > y)
        if not c.any():
            continue
        xs = np.sort(x1[c] + (y - y1[c])*(x2[c] - x1[c])/(y2[c] - y1[c]))
        for a, b in zip(xs[0::2], xs[1::2]):
            i0 = max(0, int(np.ceil((a - x0)/step - 0.5))); i1 = min(nx - 1, int(np.floor((b - x0)/step - 0.5)))
            if i1 >= i0:
                mask[r, i0:i1+1] = True
    if len(_OUTLINE_MASKS) > 8:
        _OUTLINE_MASKS.clear()
    _OUTLINE_MASKS[key] = (mask, x0, y0, step)
    return _OUTLINE_MASKS[key]

def _inside_outline(geo, x, y):
    mask, x0, y0, step = _outline_mask(geo)
    i = np.floor((np.asarray(x) - x0)/step).astype(int); j = np.floor((np.asarray(y) - y0)/step).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < mask.shape[1]) & (j < mask.shape[0])
    out = np.zeros(i.shape, dtype=bool)
    out[ok] = mask[j[ok], i[ok]]
    return out

def _footprint_inside(b, geo):
    """(t, ds, inside) sampled about every 1 mm along a brace: inside is True
    where the brace's whole width -- centreline and both footprint edges --
    lies within the traced outline."""
    Lx, Ly = geo['Lx'], geo['Ly']
    c = np.array(_brace_ctrl(b), dtype=float) * np.array([Lx, Ly])
    n = int(min(600, max(16, float(np.sum(np.hypot(*np.diff(c, axis=0).T))) / 0.001)))
    t = np.linspace(0.0, 1.0, n + 1)
    p = _brace_points(b, Lx, Ly, t)
    d = np.gradient(p, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(ds, 1e-12)[:, None]
    wC = b.get('width', 3.0)/1000.0
    wT = (b.get('width', 3.0) if b.get('widthTip') is None else b['widthTip'])/1000.0
    hw = (_dim_at(wC, wT, t)/2.0)[:, None]
    inside = (_inside_outline(geo, p[:, 0], p[:, 1])
              & _inside_outline(geo, *(p + nrm*hw).T) & _inside_outline(geo, *(p - nrm*hw).T))
    return t, ds, inside

def brace_outline_violation(braces, geo):
    """Length of brace (m) whose footprint reaches past the traced outline
    (0 if every brace lies wholly inside)."""
    if not geo.get('outline') or len(geo['outline']) < 3:
        return 0.0
    total = 0.0
    for b in braces:
        _, ds, inside = _footprint_inside(b, geo)
        total += float(np.sum(ds[~inside]))
    return total

def bounds_violation(braces, geo):
    """Brace material outside the plate: past the traced outline or into
    the sound hole (0 when there's none). Both searches treat this as a hard
    limit: a move may never increase it, so once a design is in bounds it
    stays in bounds for the rest of the search."""
    return (brace_outline_violation(braces, geo)
            + brace_hole_violation(braces, geo['holeCx'], geo['holeCy'], geo['holeR'], geo['Lx'], geo['Ly']))

def _clip_brace_to_hole(b, geo):
    """Pieces of a brace that stay clear of the sound hole (keep-out = hole
    radius + local half-width + HOLE_CLEARANCE_M), each re-cut from the
    original curve by Bezier subdivision. A brace that crosses the hole
    comes back as two pieces; slivers are dropped."""
    hr = geo.get('holeR') or 0.0
    if not (hr > 0):
        return [b]
    t, depth, _ = _hole_intrusion(b, geo['holeCx'], geo['holeCy'], hr, geo['Lx'], geo['Ly'])
    clear = depth <= 0.0
    if clear.all():
        return [b]
    pieces, i, n = [], 0, len(t)
    while i < n:
        if not clear[i]:
            i += 1; continue
        j = i
        while j + 1 < n and clear[j + 1]:
            j += 1
        t0 = t[i] if i == 0 else t[i] + 1e-9
        t1 = t[j] if j == n - 1 else t[j] - 1e-9
        if t1 - t0 >= 0.02:
            ctrl = _bezier_subrange(_brace_ctrl(b), t0, t1)
            nb = dict(b)
            nb['pts'] = [{'fx': clamp01(ctrl[0][0]), 'fy': clamp01(ctrl[0][1])}, {'fx': clamp01(ctrl[-1][0]), 'fy': clamp01(ctrl[-1][1])}]
            if len(ctrl) >= 3:
                nb['cp'] = {'fx': clamp01(ctrl[1][0]), 'fy': clamp01(ctrl[1][1])}
            if len(ctrl) == 4:
                nb['cp2'] = {'fx': clamp01(ctrl[2][0]), 'fy': clamp01(ctrl[2][1])}
            pieces.append(nb)
        i = j + 1
    return pieces

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
        inside = _inside_outline({'outline': outline_frac, 'Lx': Lx, 'Ly': Ly}, xs_a, ys_a)
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

_GEOM_CACHE = {}
_MODE_REQUEST = {}
GEOM_CACHE_SIZE = 6
_GEOM_KEYS = ('_offset', 'Lx', 'Ly', 'h', 'E', 'Ey', 'Gxy', 'nu', 'nuxy', 'rho', 'Gxz', 'Gyz', 'plateShear', 'bcMode', 'GN',
              'edgeModel', 'bridgePartialComposite', 'bridgeSeries', 'holeCx', 'holeCy', 'holeR', 'bridgeStructural', 'bridgeHeightMm', 'bridgeE', 'bridgeRho',
              'bridgeDepthMm', 'bridgeLenMm', 'bridgeFx', 'bridgeFy')

def _geometry_key(geo):
    o = geo['outline']
    bp = geo.get('bridgePts') or []
    return (GRID_OFFSET, tuple(geo.get(k) for k in _GEOM_KEYS),
            hash(tuple((p['fx'], p['fy']) for p in o)), hash(tuple((p['fx'], p['fy']) for p in bp)))

def _geometry_base(geo):
    """Everything in evaluate_design that depends only on the top, its
    material, the edge, the grid and the bridge -- not on the braces --
    built once and cached (GEOM_CACHE_SIZE most recent geometries), since
    an annealing run solves thousands of designs on the same top. Holds
    the grid, the plate's curvature operators (with the shear terms), its
    bending + shear stiffness K0, its mass M0 (diag mass, rotary inertia,
    bridge), the nodal mass m_w0 and the pin matrix P (shear columns
    included, brace columns added by the caller). Returns None when the
    grid is too coarse."""
    key = _geometry_key(geo)
    hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    Lx, Ly, h = geo['Lx'], geo['Ly'], geo['h']
    rho = geo['rho']
    Ex = geo['E']
    nu_xy = geo.get('nuxy', geo['nu'])
    Ey = geo.get('Ey', Ex)
    Gxy = geo.get('Gxy', Ex/(2*(1+nu_xy)))
    nu_yx = nu_xy * Ey / Ex   # reciprocity -- never set independently, or the material isn't physically valid
    denom = max(1e-9, 1 - nu_xy*nu_yx)
    D11 = Ex*h**3/(12*denom); D22 = Ey*h**3/(12*denom); D12 = nu_yx*D11; D66 = Gxy*h**3/12
    grid = build_grid(geo['outline'], geo['holeCx'], geo['holeCy'], geo['holeR'], Lx, Ly, geo['bcMode'], geo['GN'],
                      edge_model=geo.get('edgeModel', 'cutcell'))
    N = grid['N']
    if N < 20:
        return None
    # Through-thickness (transverse) shear: G_LT in the length-thickness
    # plane, G_RT (rolling shear, tiny in wood) in the width-thickness plane.
    Gxz, Gyz = geo.get('Gxz'), geo.get('Gyz')
    plate_shear = bool(geo.get('plateShear', True)) and bool(Gxz) and bool(Gyz)
    nlx, nly = (grid['nlx'], grid['nly']) if plate_shear else (0, 0)
    n_shear = nlx + nly
    # Bending energy D11 k_xx^2 + 2 D12 k_xx k_yy + D22 k_yy^2 + D66 k_xy^2
    # (k_xy the engineering twist). With plate shear this is a
    # Reissner-Mindlin plate: psi = -grad(w) + gamma, gamma one unknown per
    # grid link after the N nodes; curvatures come from psi, shear energy is
    # kappa*h*(G_xz gamma_x^2 + G_yz gamma_y^2). gamma = 0 is always
    # admissible (no shear locking) and every gamma pattern costs energy (no
    # spurious zero-energy modes).
    Lxx, Lyy, Lxy = grid['Lxx'], grid['Lyy'], grid['Lxy']
    Z = lambda r, c: sparse.csr_matrix((r, c))
    if plate_shear:
        R = Lxx.shape[0]
        Cxx = sparse.hstack([Lxx, -grid['Gxx'], Z(R, nly)], format='csr')
        Cyy = sparse.hstack([Lyy, Z(R, nlx), -grid['Gyy']], format='csr')
        Cxy = sparse.hstack([2*Lxy, -grid['Gxy'], -grid['Gyx']], format='csr')
    else:
        Cxx, Cyy, Cxy = Lxx, Lyy, 2*Lxy
    def bending(d11, d22, d12, d66, S=None):
        W = (lambda A: A) if S is None else (lambda A: S @ A)
        return d11*(Cxx.T@W(Cxx)) + d22*(Cyy.T@W(Cyy)) + d12*(Cxx.T@W(Cyy) + Cyy.T@W(Cxx)) + d66*(Cxy.T@W(Cxy))
    K0 = bending(D11, D22, D12, D66)
    br = _bridge_structure(geo, grid, h, Ex, Ey, Lx, Ly, nu_xy)
    m_bridge = np.zeros(N); bridge_mass_g = 0.0
    if br is not None:
        e11, e22, e12, e66 = (sparse.diags(v) for v in br['rowD'])
        K0 = K0 + (Cxx.T@e11@Cxx + Cyy.T@e22@Cyy + Cxx.T@e12@Cyy + Cyy.T@e12@Cxx + Cxy.T@e66@Cxy)
        m_bridge = br['m_nodes']; bridge_mass_g = br['mass_g']
    m_w0 = rho*h*grid['mass_scale'] + m_bridge
    M0 = sparse.diags(np.concatenate([m_w0, np.zeros(n_shear)]), format='csr')
    if plate_shear:
        cx, cy = (br['link_cover'] if br is not None else (np.zeros(nlx), np.zeros(nly)))
        bgx, bgy = (br['link_gh'] if br is not None else (0.0, 0.0))
        # Cut-cell grids weight each link by the plate area it stands for.
        wlx, wly = grid.get('link_w') or (np.ones(nlx), np.ones(nly))
        K0 = K0 + sparse.block_diag((Z(N, N), sparse.diags(PLATE_SHEAR_FACTOR*(h*Gxz*wlx + bgx)),
                                     sparse.diags(PLATE_SHEAR_FACTOR*(h*Gyz*wly + bgy))), format='csr')
        # Rotary inertia: the cross-sections' rotation psi carries rho*h^3/12
        # per area (plus the bridge block's, about the plate's mid-plane).
        Rx = sparse.hstack([-grid['Sx'], sparse.identity(nlx), Z(nlx, nly)], format='csr')
        Ry = sparse.hstack([-grid['Sy'], Z(nly, nlx), sparse.identity(nly)], format='csr')
        Ib = br['I'] if br is not None else 0.0
        M0 = M0 + Rx.T@sparse.diags(rho*h**3/12.0*wlx + Ib*cx)@Rx + Ry.T@sparse.diags(rho*h**3/12.0*wly + Ib*cy)@Ry
    P = None
    if grid.get('P') is not None:
        Pw = grid['P']; npin = Pw.shape[0]
        if n_shear:
            Pg = grid.get('P_gamma')
            P = sparse.hstack([Pw, Pg if Pg is not None else Z(npin, n_shear)], format='csr')
        else:
            P = Pw
    out = {'grid': grid, 'Ex': Ex, 'Ey': Ey, 'Gxy': Gxy, 'nu_xy': nu_xy, 'plate_shear': plate_shear,
           'nlx': nlx, 'nly': nly, 'Cxx': Cxx, 'Cyy': Cyy, 'Cxy': Cxy, 'K0': K0.tocsr(), 'M0': M0.tocsr(),
           'm_w0': m_w0, 'bridge_mass_g': bridge_mass_g, 'P': P,
           'bridge_section': ([tuple(q) for q in _bridge_footprint(geo, Lx, Ly)],
                              geo.get('bridgeHeightMm', BRIDGE_DEFAULTS['bridgeHeightMm'])/1000.0,
                              float(geo.get('bridgeE', BRIDGE_DEFAULTS['bridgeE']))) if br is not None else None}
    if len(_GEOM_CACHE) >= GEOM_CACHE_SIZE:
        _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
    _GEOM_CACHE[key] = out
    return out

SOLVER_V1 = {'edgeModel': 'legacy', 'braceFootprint': False, 'flangeCap': False, 'bridgePartialComposite': False,
             'braceBridgeSection': False, 'bridgeSeries': False, 'braceFlangeWidth': 20.0, 'braceSublines': 1,
             'braceMassTrapezoid': False}

def solver_geo(geo):
    """geo with the solverVersion preset applied: 'v1' restores the pre-v2
    solver exactly (staircase clamp, deleted free-edge rows, line braces
    with a 20-thickness flange, parallel bridge); anything else is v2.
    v1 overrides any per-feature flags the payload also carries (the UI
    always sends them), so choosing v1 always means exactly v1."""
    if geo.get('solverVersion') == 'v1':
        return {**geo, **SOLVER_V1}
    return geo

def cost_floor(geo, braces, mass_weight, n_braces=None):
    """The part of objective() that's known before any solve: geometry
    penalty, mass penalty and the per-brace cost. Returns (floor,
    geom_penalty) so evaluate_design can reuse the penalty."""
    Lx, Ly = geo['Lx'], geo['Ly']
    gp = brace_geometry_penalty(braces, geo['outline'], geo['holeCx'], geo['holeCy'], geo['holeR'], Lx, Ly)
    gp += brace_overlap_penalty(braces, Lx, Ly)
    _, _, mm = braces_to_masses(braces, Lx, Ly, geo.get('braceSegMm'), trapezoid=geo.get('braceMassTrapezoid', True))
    floor = gp + mass_weight*(sum(mm)*1000.0/40.0)
    if n_braces is not None:
        floor += 0.35*max(0, n_braces - 5)
    return floor, gp

SKIP_ACCEPT_PROB = 1e-6

def can_skip(floor, cur_cost, best_cost, T):
    """P1.4: True when the cost floor alone makes the move both worse than
    the best so far and essentially impossible to accept
    (exp(-dCost/40T) < SKIP_ACCEPT_PROB), so the solve can be skipped."""
    if floor <= best_cost or floor <= cur_cost:
        return False
    return (floor - cur_cost) > -math.log(SKIP_ACCEPT_PROB)*max(1e-6, T*40)

def evaluate_design(geo, braces, n_modes_needed, meas_xy=None, want_field=False, geom_penalty=None):
    geo = solver_geo(geo)
    Lx, Ly, h = geo['Lx'], geo['Ly'], geo['h']
    base = _geometry_base(geo)
    if base is None:
        return {'ok': False, 'reason': 'grid_too_coarse'}
    grid = base['grid']; N = grid['N']
    Ex, Ey, Gxy, nu_xy = base['Ex'], base['Ey'], base['Gxy'], base['nu_xy']
    plate_shear, nlx, nly = base['plate_shear'], base['nlx'], base['nly']
    n_shear = nlx + nly
    Cxx, Cyy, Cxy = base['Cxx'], base['Cyy'], base['Cxy']

    if geom_penalty is None:
        geom_penalty = brace_geometry_penalty(braces, geo['outline'], geo['holeCx'], geo['holeCy'], geo['holeR'], Lx, Ly)
        geom_penalty += brace_overlap_penalty(braces, Lx, Ly)

    seg_len_mm = geo.get('braceSegMm')
    mx, my, mm = braces_to_masses(braces, Lx, Ly, seg_len_mm, trapezoid=geo.get('braceMassTrapezoid', True))
    braceM = _spread_to_nodes(grid, mx, my, mm) / (grid['dd']**2)
    total_brace_mass_g = sum(mm)*1000.0
    # Point masses (accelerometers, sensors): [{fx, fy, massG}], spread onto
    # the 4 surrounding nodes like brace mass.
    pms = geo.get('pointMasses') or []
    point_mass_g = 0.0
    if pms:
        braceM = braceM + _spread_to_nodes(grid, [q['fx']*Lx for q in pms], [q['fy']*Ly for q in pms],
                                           [q['massG']/1000.0 for q in pms]) / (grid['dd']**2)
        point_mass_g = float(sum(q['massG'] for q in pms))
    iso = geo.get('braceShearModel', 'iso') == 'iso'
    K_brace, n_aux = brace_stiffness_matrix(braces, Lx, Ly, grid, seg_len_mm, plate=(h, Ex, Ey, Gxy, nu_xy),
                                            iso_shear=iso, plate_shear=plate_shear,
                                            flange_cap=bool(geo.get('flangeCap', False)),
                                            flange_width=float(geo.get('braceFlangeWidth') or BRACE_FLANGE_WIDTH),
                                            sublines=int(geo.get('braceSublines', BRACE_SUBLINES)),
                                            bridge=base['bridge_section'] if geo.get('braceBridgeSection', True) else None)
    K = base['K0']
    fp = brace_footprint_terms(braces, Lx, Ly, grid, (h, Ex, Ey, Gxy, nu_xy), iso_shear=iso,
                               Gsh=(h*float(geo.get('Gxz') or 0), h*float(geo.get('Gyz') or 0))) \
        if geo.get('braceFootprint', True) else None
    if fp is not None:
        # energy sum_r D_add (n.kappa.n)^2, kappa_nn = nx^2 k_xx + ny^2 k_yy + nx ny k_12 (k_12 = Cxy, engineering twist)
        A1, A2, A3, A4, A5 = (sparse.diags(v) for v in fp['rows'])
        sym = lambda X, Ad, Y: X.T@Ad@Y + Y.T@Ad@X
        K = K + (Cxx.T@A1@Cxx + Cyy.T@A2@Cyy + Cxy.T@A3@Cxy + sym(Cxx, A3, Cyy)
                 + sym(Cxx, A4, Cxy) + sym(Cyy, A5, Cxy))
        if plate_shear:
            fgx, fgy = fp['link_gh']
            K = K + sparse.block_diag((sparse.csr_matrix((N, N)), sparse.diags(PLATE_SHEAR_FACTOR*fgx),
                                       sparse.diags(PLATE_SHEAR_FACTOR*fgy)), format='csr')
    bridge_mass_g = base['bridge_mass_g']
    m_w = base['m_w0'] + braceM
    M = base['M0'] + sparse.diags(np.concatenate([braceM, np.zeros(n_shear)]), format='csr')
    # The braces' shear strains are extra unknowns after the plate's (massless,
    # so they never show up as modes); pad the plate's matrices to match and
    # keep only the first N entries (w) of every solution.
    pad = lambda A: sparse.block_diag((A, sparse.csr_matrix((n_aux, n_aux))), format='csr') if n_aux else A
    K = pad(K) + K_brace
    M = pad(M)
    if base['P'] is not None:
        P = base['P']
        if n_aux:
            P = sparse.hstack([P, sparse.csr_matrix((P.shape[0], n_aux))], format='csr')
        K = K + SS_PIN_PENALTY*float(np.max(K.diagonal())) * (P.T @ P)
    K = (K+K.T)*0.5
    M = (M+M.T)*0.5

    # ── Eigenmodes ──
    # Enough modes to cover the whole target band (up to 1.3x the highest
    # target), not just one per target -- see _select_modes for why every
    # mode in the band matters.
    target_hz = [float(f) for f in (geo.get('targetFreqs') or []) if f and float(f) > 0]
    f_hi = max(target_hz)*1.3 if target_hz else None
    k_max = min(60, N-2)
    n_request = min(max(n_modes_needed+6, 2*n_modes_needed+8), k_max)
    # Start from the mode count the last solve on this geometry needed to
    # pass f_hi (plus two), instead of the generous default.
    req_key = (_geometry_key(geo), n_modes_needed, f_hi)
    if req_key in _MODE_REQUEST:
        n_request = min(max(n_modes_needed + 2, _MODE_REQUEST[req_key] + 2), k_max)
    eig_tol = float(geo.get('eigTol') or 0.0)
    sigma = (2*math.pi*1.0)**2
    Kc, Mc = K.tocsc(), M.tocsc()
    # One factorization of K - sigma*M serves every eigsh call below and,
    # on a supported plate, the static solve too.
    try:
        lu_shift = _factor_sym(Kc - sigma*Mc)
    except Exception as exc:
        return {'ok': False, 'reason': f'factor_failed:{exc}'}
    OPinv = LinearOperator(Kc.shape, matvec=lu_shift.solve, dtype=float)
    while True:
        try:
            # A fixed start vector (not ARPACK's own random one) so the same
            # design always gives bit-identical eigenvalues: with a seed, a
            # search is then reproducible run to run.
            v0 = np.random.RandomState(12345).rand(Kc.shape[0]) + 0.5
            vals, vecs_full = eigsh(Kc, k=max(1,n_request), M=Mc, sigma=sigma, which='LM', OPinv=OPinv, tol=eig_tol, v0=v0)
        except Exception as exc:
            return {'ok': False, 'reason': f'eigsh_failed:{exc}'}
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs_full[:N, order]
        freqs_all = np.sqrt(np.clip(vals,0,None))/(2*math.pi)
        if f_hi is None or freqs_all[-1] >= f_hi or n_request >= k_max:
            break
        n_request = min(2*n_request, k_max)
    if f_hi is not None:
        above = np.nonzero(freqs_all >= f_hi)[0]
        _MODE_REQUEST[req_key] = int(above[0] + 1) if above.size else int(n_request)
        if len(_MODE_REQUEST) > 64:
            _MODE_REQUEST.pop(next(iter(_MODE_REQUEST)))
    # Mass-normalize (phi^T M phi = 1) so mode strengths are comparable.
    m_diag = m_w
    vecs = vecs / np.sqrt(np.einsum('ij,i,ij->j', vecs, m_diag, vecs))[None, :]
    # ── Static deflection under bridge/string load ──
    # The load is spread uniformly over the bridge footprint (traced shape,
    # else the length x depth rectangle), sampled every 1 mm, and each
    # sample's share goes bilinearly onto its 4 surrounding nodes. Deflection
    # is read back through the same interpolation. A bridge is usually
    # shallower (~12 mm) than the search grid's spacing (~15 mm), so putting
    # the load only on nodes that happen to fall inside the footprint
    # made the result jump with grid size -- one column of nodes, or a
    # single point when none fell inside -- reading 30%+ high on the
    # search grid and disagreeing with the polish grid. Sampled this way
    # it matches the exact (Navier) solution on any grid.
    PX, PY = _bridge_samples(geo, Lx, Ly)
    # Strings break over the saddle at an angle rather than pulling
    # straight down, which also twists the bridge -- more force presses
    # the neck-side edge of the footprint, less (or an uplift) on the
    # tail-side edge. Modeled as a moment superposed on the uniform load,
    # linearly distributed across the footprint's depth (its extent along
    # the body's length) exactly like combined axial+bending stress on a
    # beam cross-section. This is a deliberately rough approximation --
    # see the UI note next to the torque input.
    dx_local = PX - float(np.mean(PX))
    denom = float(np.sum(dx_local**2))
    torque_resolved = bool(np.ptp(PX) >= 0.002) if PX.size else False
    M = geo.get('bridgeTorqueNmm', 0.0) / 1000.0  # N·mm -> N·m
    base = geo['loadN']/PX.size
    f_i = np.full(PX.size, base) + (M*dx_local/denom if denom > 1e-12 else np.zeros(PX.size))
    # A large enough torque could ask part of the footprint to pull UP
    # rather than push down, which this simple superposition can't
    # represent -- clip at zero and rescale to preserve the total force.
    f_i = np.clip(f_i, 0.0, None)
    total_f = float(np.sum(f_i))
    if total_f > 1e-12:
        f_i = f_i * (geo['loadN']/total_f)
    else:
        f_i = np.full(PX.size, base)
    w_s = f_i / geo['loadN'] if geo['loadN'] else np.full(PX.size, 1.0/PX.size)
    # Load-weighted footprint as nodal weights (sum 1): the deflection
    # readback and each mode's bridge participation are both w_node . field.
    w_node = _spread_to_nodes(grid, PX, PY, w_s)
    idxs = np.nonzero(w_node)[0]; w = w_node[idxs]
    F = np.zeros(N + n_shear + n_aux)
    # K is in point-collocation form (D∇⁴w = q, q a pressure in N/m²), same
    # as the brace-mass lumping above -- nodal forces become an equivalent
    # pressure by dividing by the nodal area (dd²).
    F[idxs] = geo['loadN'] * w / (grid['dd']**2)
    if geo.get('pressurePa'):
        # Test hook (VALIDATION_PLAN Stage 2.6): a uniform pressure over the
        # plate, weighted by each node's cut-cell area fraction.
        F[:N] += float(geo['pressurePa']) * grid['mass_scale']
    try:
        if grid['bc_mode'] != 'free':
            # K x = F is (K - sigma*M) x = F - sigma*M x: refine from the
            # shifted factorization. sigma is under 1e-4 of the lowest mode's
            # omega^2, so each pass gains about four digits.
            x = lu_shift.solve(F)
            for _ in range(3):
                x = lu_shift.solve(F + sigma*(Mc @ x))
        else:
            # A free plate's K is singular (rigid-body motion), so refinement
            # wouldn't converge. Only enough regularization to keep it
            # solvable: anything larger acts as an elastic foundation whose
            # relative stiffness grows with grid size (1e-8 read 14% low at GN 180).
            reg = 1e-12*np.mean(np.abs(K.diagonal()))
            x = _factor_sym(K + sparse.diags(np.full(K.shape[0], reg))).solve(F)
        defl = x[:N]
    except Exception:
        defl = np.zeros(N)
    defl_at_bridge = float(np.dot(w, defl[idxs]) / max(float(np.sum(w)), 1e-12))
    defl_max = float(np.max(np.abs(defl))) if N else 0.0

    # ── Which modes count ──
    chosen, freqs_out, modes_out, hidden_strong = _select_modes(
        freqs_all, vecs, idxs, w, target_hz, geo.get('targetFreqs') or [], n_modes_needed, f_hi,
        grid['mass_scale'] > 0.5)
    field_modes = [m['idx'] for m in modes_out]

    probe = None
    if geo.get('probePoints'):
        # mass-normalized displacement of each reported mode at given (fx, fy)
        pp = geo['probePoints']
        probe = [sample_field_at_points(grid, vecs[:, m['idx']], [q[0]*Lx for q in pp], [q[1]*Ly for q in pp]).tolist()
                 for m in modes_out]
    meas_shapes = None
    if meas_xy is not None and len(meas_xy[0]):
        meas_shapes = []
        for c in chosen:
            if c is None:
                meas_shapes.append(None); continue
            shp = sample_field_at_points(grid, vecs[:,c], meas_xy[0], meas_xy[1])
            peak = np.max(np.abs(shp)) or 1.0
            meas_shapes.append((shp/peak).tolist())

    # Coarse dense-grid mode shapes, purely for the animated 3D view
    # (Results panel), for EVERY reported mode (modes_out), so modes that
    # weren't matched to a target can be inspected too. Gated behind
    # want_field (default False) since evaluate_design is the search's hot
    # inner loop and this is only needed once, for the design that's kept.
    # Normalized per mode to a peak of +-1; the JS renderer clips to the
    # real outline/hole at draw time.
    FIELD_NX, FIELD_NY = 96, 74
    field_shapes = None
    if want_field:
        fxs = np.linspace(0.02, 0.98, FIELD_NX) * Lx
        fys = np.linspace(0.02, 0.98, FIELD_NY) * Ly
        gx, gy = np.meshgrid(fxs, fys)  # shape (FIELD_NY, FIELD_NX)
        gx_flat, gy_flat = gx.ravel(), gy.ravel()
        field_shapes = []
        for c in field_modes:
            vals = sample_field_at_points(grid, vecs[:,c], gx_flat, gy_flat)
            peak = np.max(np.abs(vals)) or 1.0
            field_shapes.append((vals/peak).tolist())

    return {
        'ok': True, 'freqs': freqs_out, 'measShapes': meas_shapes,
        'modes': [{k: v for k, v in m.items() if k != 'idx'} for m in modes_out], 'hiddenStrong': hidden_strong,
        'fieldShapes': field_shapes, 'fieldNx': FIELD_NX, 'fieldNy': FIELD_NY,
        'deflBridgeMm': abs(defl_at_bridge)*1000.0, 'deflMaxMm': abs(defl_max)*1000.0,
        'massG': total_brace_mass_g, 'bridgeMassG': bridge_mass_g, 'pointMassG': point_mass_g,
        'nNodes': N, 'geomPenalty': geom_penalty, 'gridMm': grid['dd']*1000.0,
        'nBelow5Hz': int(np.sum(freqs_all <= 5.0)), 'probe': probe,
        'torqueResolved': torque_resolved, 'torqueRequested': abs(geo.get('bridgeTorqueNmm',0.0)) > 1e-9,
    }

BRIDGE_DEFAULTS = {'bridgeHeightMm': 9.0, 'bridgeE': 12e9, 'bridgeRho': 850.0, 'bridgeDepthMm': 28.0, 'bridgeLenMm': 155.0}

def _bridge_footprint(geo, Lx, Ly):
    """The bridge's footprint polygon (m): the traced shape if there is one,
    else the length x depth rectangle centred on the bridge point."""
    pts = geo.get('bridgePts') or []
    if len(pts) >= 3:
        return [(p['fx']*Lx, p['fy']*Ly) for p in pts]
    bx, by = geo['bridgeFx']*Lx, geo['bridgeFy']*Ly
    hd = geo.get('bridgeDepthMm', BRIDGE_DEFAULTS['bridgeDepthMm'])/2000.0
    hl = geo.get('bridgeLenMm', BRIDGE_DEFAULTS['bridgeLenMm'])/2000.0
    return [(bx-hd, by-hl), (bx+hd, by-hl), (bx+hd, by+hl), (bx-hd, by+hl)]

def _bridge_samples(geo, Lx, Ly, step=0.001):
    """Points (m) filling the bridge footprint every `step`, for spreading
    its load and its mass onto the grid. Falls back to the bridge point
    itself if the footprint is degenerate."""
    poly = np.array(_bridge_footprint(geo, Lx, Ly))
    gx = np.arange(poly[:, 0].min() + step/2, poly[:, 0].max(), step)
    gy = np.arange(poly[:, 1].min() + step/2, poly[:, 1].max(), step)
    PX, PY = np.meshgrid(gx, gy); PX, PY = PX.ravel(), PY.ravel()
    keep = points_in_polygon(PX, PY, [tuple(p) for p in poly])
    if not np.any(keep):
        return np.array([geo['bridgeFx']*Lx]), np.array([geo['bridgeFy']*Ly])
    return PX[keep], PY[keep]

BRIDGE_SUBSAMPLE = 8

def _spread_to_rows(grid, xs, ys, weights):
    """Bilinear spreading of point weights onto the curvature rows' lattice
    (rows sit on grid nodes). A partition of unity, so a region sampled
    uniformly at density 1/dd^2 gives every row inside it exactly 1 --
    unlike nearest-row binning, whose count noise (+-30%) the series
    stiffness formulas can't tolerate. Weight landing on a node without a
    row is shared out over the corners that have one."""
    nrow = grid['row_xy'].shape[0]
    rm = grid.get('_row_map')
    if rm is None:
        rm = -np.ones((grid['ny'], grid['nx']), dtype=np.int64)
        ri = np.rint((grid['row_xy'][:, 0] - grid['bbx0'])/grid['dd']).astype(int)
        rj = np.rint((grid['row_xy'][:, 1] - grid['bby0'])/grid['dd']).astype(int)
        rm[rj, ri] = np.arange(nrow)
        grid['_row_map'] = rm
    dd = grid['dd']
    fx = (np.asarray(xs, float) - grid['bbx0'])/dd; fy = (np.asarray(ys, float) - grid['bby0'])/dd
    i0 = np.clip(np.floor(fx).astype(int), 0, grid['nx']-2); j0 = np.clip(np.floor(fy).astype(int), 0, grid['ny']-2)
    tx = np.clip(fx - i0, 0, 1); ty = np.clip(fy - j0, 0, 1)
    ci = np.stack([i0, i0+1, i0, i0+1], 1); cj = np.stack([j0, j0, j0+1, j0+1], 1)
    W = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], 1)
    ids = rm[cj, ci]
    W = np.where(ids >= 0, W, 0.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros(nrow)
    ok = ids >= 0
    np.add.at(out, ids[ok], (W*np.asarray(weights, float)[:, None])[ok])
    return out

def _bridge_structure(geo, grid, h, Ex, Ey, Lx, Ly, nuxy_p=0.3):
    """The bridge block as structure: a hardwood layer (bridgeHeightMm thick,
    bridgeE, bridgeRho) bonded over its footprint on the outside face.
    Stiffness is the extra bending stiffness of the plate+bridge section
    about its shared neutral axis (both directions, isotropic wood, nu=0.3),
    added on the curvature rows the footprint covers, weighted by the
    fraction of each row's cell it covers -- so a bridge narrower than the
    grid spacing still counts its true area. Its mass is spread onto the
    nodes the same way. With plate shear, the block also resists
    through-thickness shear (G_b*h_b, isotropic) and adds rotary inertia
    (about the plate's mid-plane), both on the shear links its footprint
    covers.

    Returns None without a structural bridge, else a dict: 'D' the added
    (D11, D22, D12, D66) and 'cover' its per-row weighting (bending is
    assembled by the caller, so it picks up the plate's shear terms too),
    'm_nodes' nodal mass per area, 'mass_g', 'Gh' = G_b*h_b, 'I' the rotary
    inertia per area, and 'link_cover' the footprint fraction on each x- and
    y-link."""
    if not geo.get('bridgeStructural', True):
        return None
    hb = geo.get('bridgeHeightMm', BRIDGE_DEFAULTS['bridgeHeightMm'])/1000.0
    Eb = float(geo.get('bridgeE', BRIDGE_DEFAULTS['bridgeE'])); rhob = float(geo.get('bridgeRho', BRIDGE_DEFAULTS['bridgeRho']))
    if not (hb > 0 and Eb > 0):
        return None
    step = 0.001
    PX, PY = _bridge_samples(geo, Lx, Ly, step)
    if PX.size < 2:
        return None
    dd = grid['dd']; a_each = step*step
    m_nodes = _spread_to_nodes(grid, PX, PY, np.full(PX.size, rhob*hb*a_each)) / (dd*dd)
    mass_g = rhob*hb*a_each*PX.size*1000.0
    nub = 0.3
    Gb = Eb/(2.0*(1.0 + nub))
    poly = np.array(_bridge_footprint(geo, Lx, Ly))
    span_x, span_y = float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))
    partial = geo.get('bridgePartialComposite', True)
    def d_add(Ep, Gpz, span):
        # Composite bending of plate + block about their shared neutral axis:
        # the block's own bending plus EA_eff * z^2 (EA_eff the two layers'
        # axial stiffnesses in series, z the distance between their centroids).
        # The plate operator already counts the plate's own bending.
        EA = Ep*h*Eb*hb/(Ep*h + Eb*hb)
        z = 0.5*(h + hb)
        phi = 1.0
        if partial and Gpz:
            # Partial interaction: the composite force has to pass through the
            # plate's and the block's thickness in shear (interface stiffness
            # k per area), so over a block of finite span it only builds up a
            # development length sqrt(EA/k) in from each end. Averaged over
            # the span: phi = 1 - tanh(b L/2)/(b L/2), b = sqrt(k/EA).
            k = 1.0/((0.5*h)/Gpz + (0.5*hb)/Gb)
            bl = math.sqrt(k/EA)*0.5*span
            phi = 1.0 - math.tanh(bl)/bl if bl > 1e-9 else 0.0
        return Eb*hb**3/12.0 + phi*EA*z*z
    Dx = d_add(Ex, geo.get('Gxz') if geo.get('plateShear', True) else None, span_x)
    Dy = d_add(Ey, geo.get('Gyz') if geo.get('plateShear', True) else None, span_y)
    # Directional series/parallel stiffening. Within one row's cell, plate
    # and block act in SERIES for bending along x wherever a line along x
    # crosses the block's edge (their compliances add), and in PARALLEL
    # between such lines (side-by-side strips share the curvature). So each
    # cell near the block is sub-sampled (BRIDGE_SUBSAMPLE^2): for bending
    # along x, every sub-row with covered fraction f gets
    # D_p/(1 - f*D_add/(D_p + D_add)), and the sub-rows are averaged; the
    # same along y with sub-columns. A fully covered row gets exactly
    # D_p + D_add. A plain area-fraction weighting (in parallel) made a
    # partly covered edge row rigid, so the block read about a cell too big:
    # on a bridge-only rectangle, mode 3 was +10% at GN 100 against a 3D
    # solid model and +2% at GN 160.
    nu_yx = nuxy_p*Ey/Ex; den = 1.0 - nuxy_p*nu_yx
    D11p, D22p = Ex*h**3/(12*den), Ey*h**3/(12*den)
    nrow = grid['row_xy'].shape[0]
    rx, ry = grid['row_xy'][:, 0], grid['row_xy'][:, 1]
    near = np.nonzero((rx > poly[:, 0].min() - dd) & (rx < poly[:, 0].max() + dd) &
                      (ry > poly[:, 1].min() - dd) & (ry < poly[:, 1].max() + dd))[0]
    sbs = BRIDGE_SUBSAMPLE
    off = ((np.arange(sbs) + 0.5)/sbs - 0.5)*dd
    ox, oy = np.meshgrid(off, off)          # [y, x]
    qx = (rx[near][:, None, None] + ox[None]).ravel(); qy = (ry[near][:, None, None] + oy[None]).ravel()
    mask = points_in_polygon(qx, qy, [tuple(q) for q in poly]).reshape(-1, sbs, sbs).astype(float)
    def series(Dp, Da, f):
        cp = f*Da/(Dp + Da)
        return Dp*cp/(1.0 - cp)
    ex = np.zeros(nrow); ey = np.zeros(nrow)
    ex[near] = series(D11p, Dx, mask.mean(axis=2)).mean(axis=1)   # sub-rows (fixed y) along x
    ey[near] = series(D22p, Dy, mask.mean(axis=1)).mean(axis=1)   # sub-columns (fixed x) along y
    exy = np.sqrt(ex*ey)
    Gxz_, Gyz_ = (geo.get('Gxz'), geo.get('Gyz')) if geo.get('plateShear', True) else (None, None)
    gx_r = np.zeros(nrow); gy_r = np.zeros(nrow)
    if Gxz_ and Gyz_:
        gx_r[near] = series(h*Gxz_, Gb*hb, mask.mean(axis=2)).mean(axis=1)
        gy_r[near] = series(h*Gyz_, Gb*hb, mask.mean(axis=1)).mean(axis=1)
    _spread_to_rows(grid, rx[:1], ry[:1], [0.0])   # builds the row map
    rm = grid['_row_map']
    def link_from_rows(ends, vals):
        a_, b_ = ends
        ra = rm[grid['nodes_j'][a_], grid['nodes_i'][a_]]
        va = np.where(ra >= 0, vals[np.maximum(ra, 0)], 0.0)
        bb = np.maximum(b_, 0)
        rb = rm[grid['nodes_j'][bb], grid['nodes_i'][bb]]
        vb = np.where((b_ >= 0) & (rb >= 0), vals[np.maximum(rb, 0)], va)
        return 0.5*(va + vb)
    # Fraction of each node's cell the footprint covers, averaged onto links.
    frac = _spread_to_nodes(grid, PX, PY, np.full(PX.size, a_each)) / (dd*dd)
    def link_cover(ends):
        a, b = ends
        return np.minimum(np.where(b >= 0, 0.5*(frac[a] + frac[np.maximum(b, 0)]), frac[a]), 1.0)
    if not geo.get('bridgeSeries', True):
        # Legacy: full composite D on every row the footprint touches,
        # weighted by its (nearest-row) area fraction, in parallel.
        _, ri_ = cKDTree(grid['row_xy']).query(np.stack([PX, PY], axis=1))
        cov_ = np.bincount(ri_, minlength=nrow) * a_each / (dd*dd)
        Dxy_ = math.sqrt(Dx*Dy)
        ex, ey, exy = cov_*Dx, cov_*Dy, cov_*Dxy_
        lc = (link_cover(grid['Sx_ends']), link_cover(grid['Sy_ends']))
        zc = (h + hb)/2.0
        return {'rowD': (ex, ey, nub*exy, 0.5*(1.0-nub)*exy), 'm_nodes': m_nodes, 'mass_g': mass_g,
                'I': rhob*(hb**3/12.0 + hb*zc*zc), 'link_cover': lc, 'link_gh': (Gb*hb*lc[0], Gb*hb*lc[1])}
    zc = (h + hb)/2.0   # block centroid above the plate's mid-plane
    return {'rowD': (ex, ey, nub*exy, 0.5*(1.0-nub)*exy),
            'm_nodes': m_nodes, 'mass_g': mass_g, 'I': rhob*(hb**3/12.0 + hb*zc*zc),
            'link_cover': (link_cover(grid['Sx_ends']), link_cover(grid['Sy_ends'])),
            'link_gh': (link_from_rows(grid['Sx_ends'], gx_r), link_from_rows(grid['Sy_ends'], gy_r))}

HIDDEN_MODE_RATIO = 0.5   # unmatched modes at least this strong (vs the weakest matched one) count as "hidden"

def _select_modes(freqs_all, vecs, drive_idx, drive_w, target_hz, target_freqs_in_order, n_modes_needed, f_hi, real_mask):
    """Decide which computed modes are reported and which one each target
    is matched to.

    Strength of a mode = its peak in the plate's response to a force on the
    bridge footprint (velocity per unit force at resonance, equal damping
    assumed): bridge participation x plate-average amplitude / omega, with
    mass-normalized modes. That's "how big a peak this mode makes" -- weak
    modes barely show up in a measured or played spectrum.

    With targets: the N strongest modes in the band (above 5 Hz, up to f_hi)
    are paired with the N targets in frequency order. Weak modes between
    them are ignored rather than shifting every pairing above them, and
    every mode in the band is reported (modes_out) so none are hidden.
    Unmatched modes nearly as strong as the matched ones are returned in
    hidden_strong (objective() penalizes them). Without targets: the
    n_modes_needed lowest modes.

    Returns (chosen, freqs_out, modes_out, hidden_strong): chosen[i] / freqs_out[i]
    = mode index / frequency for target i in the caller's target order
    (None if the band had too few modes); modes_out = every reported mode,
    by frequency, as {idx, freq, strength (0-1, vs strongest), target}."""
    valid = [i for i, f in enumerate(freqs_all) if f > 5 and (f_hi is None or f <= f_hi)]
    if not valid:
        return [None]*len(target_hz), [None]*len(target_hz), [], []
    V = vecs[:, valid]
    drive = np.abs(drive_w @ V[drive_idx])
    rms = np.sqrt(np.mean(V[real_mask]**2, axis=0))
    strength = drive * rms / (2*math.pi*freqs_all[valid])
    rel = strength / max(float(strength.max()), 1e-300)

    target_of = {}
    if target_hz:
        n = len(target_freqs_in_order)
        top = sorted(np.argsort(-strength)[:n].tolist(), key=lambda j: freqs_all[valid[j]])
        t_rank = sorted(range(n), key=lambda i: float(target_freqs_in_order[i]))
        chosen = [None]*n
        for r, j in enumerate(top):
            chosen[t_rank[r]] = valid[j]
            target_of[valid[j]] = t_rank[r]
        weakest = min(strength[j] for j in top)
        hidden = [j for j in range(len(valid)) if valid[j] not in target_of
                  and strength[j] >= HIDDEN_MODE_RATIO*weakest]
        hidden_strong = [{'freq': float(freqs_all[valid[j]]), 'rel': float(strength[j]/max(weakest, 1e-300))}
                         for j in hidden]
        reported = list(range(len(valid)))
    else:
        reported = list(range(min(n_modes_needed, len(valid))))
        chosen = [valid[j] for j in reported]
        hidden_strong = []
    freqs_out = [None if c is None else float(freqs_all[c]) for c in chosen]
    modes_out = [{'idx': valid[j], 'freq': float(freqs_all[valid[j]]), 'strength': float(rel[j]),
                  'target': target_of.get(valid[j])} for j in reported]
    return chosen, freqs_out, modes_out, hidden_strong

DEFL_INFEASIBLE_COST = 1e4   # added once a design exceeds the deflection limit
DEFL_SPAN = 3000.0           # ...plus up to this much more the further over it is
DEFL_SCALE = 3.0             # (in multiples of the limit) -- see deflection_cost

def deflection_cost(defl_mm, limit_mm):
    """Bridge deflection over the limit. Any design over the limit costs more
    than any design within it (the DEFL_INFEASIBLE_COST step is larger than
    frequency costs realistically get), so a feasible design always ranks
    first. Beyond that the penalty rises smoothly and levels off (tanh)
    instead of growing without bound: a layout that can't reach the limit
    still feels pressure to deflect less near the limit, but far over it the
    frequency terms still matter, so it gets tuned rather than stalling on
    a deflection term thousands of times larger than everything else."""
    over = max(0.0, defl_mm/max(0.05, limit_mm) - 1.0)
    if over <= 0.0:
        return 0.0
    return DEFL_INFEASIBLE_COST + DEFL_SPAN*math.tanh(over/DEFL_SCALE)

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
        if tgt.get('matchShape') and shapes is not None and i < len(shapes) and shapes[i] is not None and tgt.get('measShape') is not None:
            m = mac(shapes[i], tgt['measShape'])
            cost += tgt.get('weight',1.0) * (1.0-m) * 8.0
    # A strong mode that isn't one of the targets is a peak the design was
    # never asked to have -- costs more the closer it is to the weakest
    # matched (target) mode's strength. Weak in-between modes cost nothing.
    for hm in result.get('hiddenStrong') or []:
        cost += 6.0 * (min(1.0, hm['rel']) - HIDDEN_MODE_RATIO + 0.1)**2
    cost += deflection_cost(result['deflBridgeMm'], defl_limit_mm)
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
BRACE_MATERIAL = {'modulus': 9.79e9, 'density': 350.0, 'shear': 0.65e9}
# Floor (mm) on a brace's width at its tips, overridden per-run from the
# payload's minTipWidthMm. A tip narrower than the grid spacing (about 5.6mm
# at GN 90) is suspected of reading too stiff, since the plate strip working
# with it is wider than the brace itself. The center width shares the floor,
# because a tip never runs wider than its center.
BRACE_LIMITS = {'min_tip_width': 1.0}
# Checked at the top of every search iteration so a run can be stopped
# mid-way (e.g. the person spots a setup mistake) instead of having to
# wait out however many iterations are left.
CANCEL_FLAG = {'stop': False}

def make_brace(x1,y1,x2,y2,height,width,modulus=None,density=None,curve_cp=None,curve_cp2=None,height_tip=None,width_tip=None,shear=None):
    # The shear modulus goes with the modulus: the default material's when
    # the modulus is the default, else whatever the caller's material says
    # (None -> brace_shear_modulus's typical E/G for that kind of material).
    if modulus is None and shear is None:
        shear = BRACE_MATERIAL.get('shear')
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
         'width':max(1.5,BRACE_LIMITS['min_tip_width'],width), 'height':max(3.0,height), 'modulus':modulus, 'density':density,
         'widthTip': (max(BRACE_LIMITS['min_tip_width'],width_tip) if width_tip is not None else None),
         'heightTip': (max(2.0,height_tip) if height_tip is not None else None)}
    if shear: b['shear'] = float(shear)
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

def _clip_brace_to_outline(b, geo):
    """Trims a brace to the longest stretch of it whose whole footprint
    (both edges, not just the centreline) lies inside the traced outline,
    re-cut from the original curve by Bezier subdivision so the curve
    shape is kept exactly. Returns None if no usable stretch is left."""
    t, _, inside = _footprint_inside(b, geo)
    if inside.all():
        return b
    best, i, n = None, 0, len(t)
    while i < n:
        if not inside[i]:
            i += 1; continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        if best is None or (t[j] - t[i]) > (t[best[1]] - t[best[0]]):
            best = (i, j)
        i = j + 1
    if best is None or t[best[1]] - t[best[0]] < 0.02:
        return None   # sliver too short to keep as a brace
    t0 = t[best[0]] if best[0] == 0 else t[best[0]] + 1e-9
    t1 = t[best[1]] if best[1] == n - 1 else t[best[1]] - 1e-9
    ctrl = _bezier_subrange(_brace_ctrl(b), t0, t1)
    nb = dict(b)
    nb['pts'] = [{'fx': clamp01(ctrl[0][0]), 'fy': clamp01(ctrl[0][1])}, {'fx': clamp01(ctrl[-1][0]), 'fy': clamp01(ctrl[-1][1])}]
    if len(ctrl) >= 3:
        nb['cp'] = {'fx': clamp01(ctrl[1][0]), 'fy': clamp01(ctrl[1][1])}
    if len(ctrl) == 4:
        nb['cp2'] = {'fx': clamp01(ctrl[2][0]), 'fy': clamp01(ctrl[2][1])}
    return nb

def _clip_braces_to_outline(braces, geo):
    """Every brace trimmed to the traced outline AND clear of the sound
    hole (see _clip_brace_to_hole) -- the hard guarantee for anything
    returned or evaluated by the fixed-pattern encoders."""
    geo = geo or {}
    outline, Lx, Ly = geo.get('outline'), geo.get('Lx'), geo.get('Ly')
    if not outline or not Lx or not Ly:
        return braces
    out = []
    for b in braces:
        cb = _clip_brace_to_outline(b, geo)
        if cb is not None:
            out.extend(_clip_brace_to_hole(cb, geo))
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
SEARCH_EIG_TOL = 1e-6   # ARPACK tolerance during search and polish; the card's final solve uses machine precision
SKIP_STATS = {'skipped': 0}
SA_T_END = 0.995**900   # temperature at the end of a run, relative to its start

def sa_temperature(it, iters, temp_scale=1.0):
    """Geometric cooling stretched over the whole run: starts at temp_scale
    and reaches temp_scale*SA_T_END on the last iteration, however many
    iterations there are. (A fixed per-iteration decay instead left short
    runs still exploring when they stopped and spent the tail of long runs
    as a plain hill-climb.) SA_T_END matches where the old fixed 0.995 decay
    ended at 900 iterations, the default, so a default run is unchanged."""
    return temp_scale * SA_T_END ** (it / max(1, iters - 1))

async def sa_continuous(encode_fn, bounds, geo, targets, meas_xy, defl_limit_mm, mass_weight, iters, progress_cb, label, x0=None, temp_scale=1.0):
    n = len(bounds)
    x = np.array(x0) if x0 is not None else np.array([random.uniform(lo,hi) for lo,hi in bounds])
    def clip(v): return np.array([min(hi,max(lo,vv)) for vv,(lo,hi) in zip(v,bounds)])
    best_x, best_cost, best_result = x.copy(), 1e18, None
    cur_cost = 1e18
    cur_bounds_violation = None  # set on first iteration once braces exist
    best_valid_x, best_valid_cost = None, 1e18
    n_modes = len(targets)
    geo = dict(geo, eigTol=SEARCH_EIG_TOL)   # P1.1: the search doesn't need machine-precision eigenvalues
    for it in range(iters):
        if CANCEL_FLAG['stop']:
            break
        T = sa_temperature(it, iters, temp_scale)
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
        violation = bounds_violation(braces, geo)
        if cur_bounds_violation is not None and violation > cur_bounds_violation + 1e-12:
            if it % max(1,iters//25) == 0 or it==iters-1:
                progress_cb(label, it, iters, best_cost, best_result)
            await asyncio.sleep(0)  # yield every iteration, not just on progress ticks -- keeps the page responsive during long runs
            continue
        floor, gp = cost_floor(geo, braces, mass_weight)
        if best_result is not None and can_skip(floor, cur_cost, best_cost, T):
            SKIP_STATS['skipped'] += 1
            await asyncio.sleep(0)
            continue
        result = evaluate_design(geo, braces, n_modes, meas_xy, geom_penalty=gp)
        cost = objective(result, targets, defl_limit_mm, mass_weight)
        accept = cost < cur_cost or random.random() < math.exp(-(cost-cur_cost)/max(1e-6,T*40))
        if accept:
            x, cur_cost, cur_bounds_violation = cand, cost, violation
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
    min_tip = BRACE_LIMITS['min_tip_width']
    min_height, min_width = 4.0, max(2.0, min_tip)
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
        # Only ever hand back a brace that's wholly inside the plate (its
        # mirror too, when symmetric) -- the search's bounds ratchet would
        # reject anything else, and the starting layout is built from these.
        b = None
        for _ in range(60):
            b = _rand_brace_once()
            if bounds_violation(expand([b]), geo) <= 0.0:
                return b
        return b

    def _rand_brace_once():
        x1,y1 = random.uniform(0.1,0.9), random.uniform(fy_lo,fy_hi)
        x2,y2 = random.uniform(0.1,0.9), random.uniform(fy_lo,fy_hi)
        h = random.uniform(min_height,max_height); w = random.uniform(min_width,max_width)
        mat = rand_material()
        b = make_brace(x1,y1,x2,y2,h,w, height_tip=h*random.uniform(0.4,1.0), width_tip=w*random.uniform(0.4,1.0),
                        modulus=(mat['modulus'] if mat else None), density=(mat['density'] if mat else None),
                        shear=(mat.get('shear') if mat else None))
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
                if mat: b['modulus'], b['density'], b['shear'] = mat['modulus'], mat['density'], mat.get('shear')
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
        b['width'] = max(1.5, min_tip, min(max_width, b['width']))
        if b.get('heightTip') is not None: b['heightTip'] = max(2.0, min(b['height'], b['heightTip']))
        if b.get('widthTip') is not None: b['widthTip'] = max(min_tip, min(b['width'], b['widthTip']))
    n_modes = len(targets)
    geo = dict(geo, eigTol=SEARCH_EIG_TOL)   # P1.1
    def evalb(bl, floor_gp=None):
        full = expand(bl)
        r = evaluate_design(geo, full, n_modes, meas_xy, geom_penalty=None if floor_gp is None else floor_gp[1])
        return objective(r, targets, defl_limit_mm, mass_weight, n_braces=len(full)), r
    cur_cost, cur_result = evalb(braces)
    cur_bounds_violation = bounds_violation(expand(braces), geo)
    best_braces, best_cost, best_result = [dict(b) for b in braces], cur_cost, cur_result
    # Tracked separately from best_braces/best_cost above: the ratchet
    # only guarantees violation never INCREASES, so an early low-cost
    # candidate recorded while violation was still nonzero could otherwise
    # end up as the returned "best" even after the search finds a fully
    # clean design later with slightly higher cost. The real constraint is
    # "never any material in the hole" on the actual returned result, not
    # just "trending toward less violation" -- so the final return prefers
    # this whenever one exists.
    best_valid_braces, best_valid_cost = (None, 1e18) if cur_bounds_violation > 1e-12 else (best_braces, best_cost)
    # Material-flip mutation gets its own slice of the move budget only
    # when a materials palette was actually given -- otherwise the move
    # probabilities below are unchanged from the single-material search.
    mat_move_cutoff = 0.06 if materials else 0.0
    for it in range(iters):
        if CANCEL_FLAG['stop']:
            break
        T = sa_temperature(it, iters, temp_scale)
        if it % max(1, iters//150) == 0 or it == iters-1:
            progress_cb(label, it, iters, best_cost, best_result, trace=(cur_cost, T, expand(best_braces)))
        cand = [dict(b, pts=[dict(pp) for pp in b['pts']], **({'cp':dict(b['cp'])} if b.get('cp') else {})) for b in braces]
        move = random.random()
        if lock_positions and cur_bounds_violation <= 1e-12:
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
            if mat: b['modulus'], b['density'], b['shear'] = mat['modulus'], mat['density'], mat.get('shear')
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
            b['width'] = max(1.5, min_tip, min(max_width, b['width'] + random.uniform(-1.5,1.5)))
            hTip = b.get('heightTip') if b.get('heightTip') is not None else b['height']
            wTip = b.get('widthTip') if b.get('widthTip') is not None else b['width']
            b['heightTip'] = max(2.0, min(b['height'], hTip + random.uniform(-2.0,2.0)))
            b['widthTip'] = max(min_tip, min(b['width'], wTip + random.uniform(-1.5,1.5)))
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
        violation = bounds_violation(expand(cand), geo)
        if violation > cur_bounds_violation + 1e-12:
            if it % max(1,iters//25) == 0 or it==iters-1:
                progress_cb(label, it, iters, best_cost, best_result)
            await asyncio.sleep(0)  # yield every iteration, not just on progress ticks -- keeps the page responsive during long runs
            continue
        full_c = expand(cand)
        fl = cost_floor(geo, full_c, mass_weight, n_braces=len(full_c))
        if can_skip(fl[0], cur_cost, best_cost, T):
            SKIP_STATS['skipped'] += 1
            if it % max(1,iters//25) == 0 or it==iters-1:
                progress_cb(label, it, iters, best_cost, best_result)
            await asyncio.sleep(0)
            continue
        cost, result = evalb(cand, fl)
        accept = cost < cur_cost or random.random() < math.exp(-(cost-cur_cost)/max(1e-6,T*40))
        if accept:
            braces, cur_cost, cur_result, cur_bounds_violation = cand, cost, result, violation
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

# ── Grid-convergence correction for the polish ───────────────────────────
# The polish grid (hifiGN, ~50-90) is coarse enough to be searched hundreds
# of times, which leaves its modes a few percent off the converged answer
# (the answer a 3D FEA model agrees with). The error isn't one-sided: the
# bilinear spreading of each brace onto the plate nodes and the staircase
# edge both depend on where the braces fall on the grid, so one design's
# modes all rise with refinement (2-7% from GN 60 to 160 on a five-brace
# experimental top) while an x-brace's fall (2-6%), and the higher modes
# move most.
#
# Rather than polish on the fine grid (each solve there is 10-50x slower),
# the coarse winner is solved ONCE on both grids. For target i the ratio
#     k_i = f_fine,i / f_polish,i
# is how far that mode moves from the polish grid to the fine one. The
# polish then aims at f_target,i / k_i (and tolerance / k_i) on the cheap
# grid, so that the same design lands on f_target,i once it is re-solved on
# the fine grid for the card. The bridge deflection limit gets the same
# treatment with the deflection ratio. k_i depends on the design and drifts
# as the polish changes it (a few percent can remain on single modes); the
# fine re-solve at the end reports that residual.
#
# Modes are followed between grids by shape (MAC on the common display
# field), not re-paired by strength: two weak modes can trade places in
# strength between grids, which would pair a target with a different mode,
# give a meaningless ratio, and report a mode the polish never tuned.
GRID_FACTOR_BOUNDS = (0.8, 1.25)
# The fine grid is capped: the Mindlin plate carries about three unknowns per
# node, so GN 200 is ~70k unknowns and a ~2 GB factorization per pattern
# process, and GN 300 (~155k) runs out of memory with patterns in parallel.
MAX_FINE_GN = 200
GRID_TRACK_MAC = 0.5

def _track_modes(r_lo, r_hi, n_targets):
    """For each target slot i: (j_lo, j_hi), indices into r_lo['modes'] and
    r_hi['modes'] (both solved with want_field) of target i's mode on each
    grid. j_hi is None when no fine-grid mode matches it by shape (MAC >
    GRID_TRACK_MAC) with a frequency ratio inside GRID_FACTOR_BOUNDS.
    Each fine-grid mode goes to at most one slot."""
    lo_modes, hi_modes = r_lo.get('modes') or [], r_hi.get('modes') or []
    lo_shapes, hi_shapes = r_lo.get('fieldShapes') or [], r_hi.get('fieldShapes') or []
    lo_f = r_lo.get('freqs') or []
    js, cands = [], []
    for i in range(n_targets):
        j = next((n for n, m in enumerate(lo_modes) if m.get('target') == i), None)
        f_i = lo_f[i] if i < len(lo_f) else None
        if j is None and f_i and lo_modes:   # no target tags: the mode reported for slot i
            j = int(np.argmin([abs(m['freq'] - f_i) for m in lo_modes]))
        js.append(j)
        if j is None or j >= len(lo_shapes) or not (lo_modes[j]['freq'] > 0):
            continue
        for jh, s in enumerate(hi_shapes):
            k = hi_modes[jh]['freq'] / lo_modes[j]['freq']
            if GRID_FACTOR_BOUNDS[0] < k < GRID_FACTOR_BOUNDS[1]:
                c = mac(lo_shapes[j], s)
                if c > GRID_TRACK_MAC:
                    cands.append((c, i, jh))
    # Greedy one-to-one, best MAC first: a close pair whose shapes mix a
    # little between grids (MAC ~0.6-0.7 each way) still pairs up correctly
    # instead of both slots grabbing the same fine-grid mode.
    hit, used = {}, set()
    for c, i, jh in sorted(cands, reverse=True):
        if i not in hit and jh not in used:
            hit[i] = jh; used.add(jh)
    return [(js[i], hit.get(i)) for i in range(n_targets)]

def grid_correction(braces, polish_geo, fine_geo, targets, n_modes, meas_xy, defl_limit_mm):
    """Solve braces on the polish grid and the fine grid and return
    (polish_targets, polish_defl_limit, info): the targets and deflection
    limit to search against on the polish grid, and info = {'factors': k_i
    per target, 'deflFactor', 'hifiGN', 'displayGN'} for the card. A target
    whose mode can't be followed uses the median of the others. Falls back
    to the uncorrected targets if either solve fails."""
    r_lo = evaluate_design(polish_geo, braces, n_modes, meas_xy, want_field=True)
    r_hi = evaluate_design(fine_geo, braces, n_modes, meas_xy, want_field=True)
    if not (r_lo.get('ok') and r_hi.get('ok')):
        return targets, defl_limit_mm, None
    lo_modes, hi_modes = r_lo['modes'], r_hi['modes']
    ratios = [hi_modes[jh]['freq']/lo_modes[j]['freq'] if jh is not None else None
              for j, jh in _track_modes(r_lo, r_hi, len(targets))]
    good = [k for k in ratios if k]
    fallback = float(np.median(good)) if good else 1.0
    ratios = [float(k) if k else fallback for k in ratios]
    polish_targets = [dict(t, freq=t['freq']/k, tol=t.get('tol', 5.0)/k) for t, k in zip(targets, ratios)]
    d_lo, d_hi = r_lo.get('deflBridgeMm') or 0.0, r_hi.get('deflBridgeMm') or 0.0
    d_ratio = min(1.5, max(0.67, d_hi/d_lo)) if d_lo > 1e-9 and d_hi > 0 else 1.0
    info = {'factors': ratios, 'deflFactor': float(d_ratio),
            'hifiGN': int(polish_geo['GN']), 'displayGN': int(fine_geo['GN'])}
    return polish_targets, defl_limit_mm/d_ratio, info

GRID_EST_SAFETY = 1.25   # Roache's safety factor for the grid band

def _richardson3(gns, fs):
    """Observed-order Richardson extrapolation from three grids (ascending
    GN). Returns (f_inf, p, monotone). Falls back to p = 1 on the two finest
    grids when the three don't converge monotonically or p leaves [0.5, 4]."""
    (g1, g2, g3), (f1, f2, f3) = gns, fs
    d1, d2 = f2 - f1, f3 - f2
    def ratio(p):
        return (g1**-p - g2**-p)/(g2**-p - g3**-p)
    p = None; mono = d1*d2 > 0 and abs(d2) < abs(d1)*4
    if mono and abs(d2) > 1e-12:
        target = d1/d2
        lo, hi = 0.5, 4.0
        if (ratio(lo) - target)*(ratio(hi) - target) < 0:
            for _ in range(60):
                mid = 0.5*(lo + hi)
                if (ratio(lo) - target)*(ratio(mid) - target) <= 0: hi = mid
                else: lo = mid
            p = 0.5*(lo + hi)
    if p is None:
        p = 1.0; mono = False
    finf = f3 + d2*g3**-p/(g2**-p - g3**-p)
    return float(finf), float(p), bool(mono)

def grid_estimate(braces, polish_geo, fine_geo, r_lo, r_hi, track, n_modes, meas_xy):
    """P1.5: a third solve between the polish and fine grids, each target's
    mode followed by shape across all three, then Richardson extrapolation
    with the observed order. Per target slot: the extrapolated frequency,
    a +- band (GRID_EST_SAFETY * |f_fine - f_inf|), the order, and whether
    the three grids converged monotonically."""
    gl, gf = int(polish_geo['GN']), int(fine_geo['GN'])
    gm = int(round(math.sqrt(gl*gf)))
    if not (gl < gm < gf):
        return None
    mid_geo = dict(fine_geo, GN=gm)
    r_mid = evaluate_design(mid_geo, braces, n_modes, meas_xy, want_field=True)
    if not r_mid.get('ok'):
        return None
    hs, ms = r_hi.get('fieldShapes') or [], r_mid.get('fieldShapes') or []
    out = {'gns': [gl, gm, gf], 'fine': [], 'extrapolated': [], 'band': [], 'order': [], 'monotone': []}
    for (j, jh) in track:
        if jh is None or j is None or jh >= len(hs):
            for k in ('fine', 'extrapolated', 'band', 'order', 'monotone'):
                out[k].append(None)
            continue
        f_hi = r_hi['modes'][jh]['freq']; f_lo = r_lo['modes'][j]['freq']
        best = max(range(len(ms)), key=lambda q: mac(hs[jh], ms[q])) if ms else None
        if best is None or mac(hs[jh], ms[best]) < GRID_TRACK_MAC:
            for k in ('fine', 'extrapolated', 'band', 'order', 'monotone'):
                out[k].append(None)
            continue
        f_mid = r_mid['modes'][best]['freq']
        finf, p, mono = _richardson3((gl, gm, gf), (f_lo, f_mid, f_hi))
        out['fine'].append(f_hi); out['extrapolated'].append(finf)
        out['band'].append(GRID_EST_SAFETY*abs(f_hi - finf)); out['order'].append(p); out['monotone'].append(mono)
    return out

def fine_resolve(braces, polish_geo, fine_geo, n_modes, meas_xy, estimate=True):
    """The card's re-solve on the fine grid, reporting for each target the
    fine-grid version of the mode the polish tuned (followed by shape)
    rather than whatever the fine grid's own strength pairing picks. A slot
    whose mode can't be followed keeps the strength pairing."""
    r_hi = evaluate_design(fine_geo, braces, n_modes, meas_xy, want_field=True)
    if not r_hi.get('ok'):
        return r_hi
    r_lo = evaluate_design(polish_geo, braces, n_modes, meas_xy, want_field=True)
    if not r_lo.get('ok'):
        return r_hi
    n = len(r_hi.get('freqs') or [])
    track = _track_modes(r_lo, r_hi, n)
    modes, freqs = r_hi['modes'], list(r_hi['freqs'])
    claimed = {jh for _, jh in track if jh is not None}
    meas = list(r_hi['measShapes']) if r_hi.get('measShapes') else None
    # Strength-pairing tags survive only for slots that couldn't be tracked,
    # and only on modes no tracked slot claims.
    for n_i, m in enumerate(modes):
        t = m.get('target')
        if n_i in claimed or (t is not None and t < n and track[t][1] is not None):
            m['target'] = None
    for i, (j, jh) in enumerate(track):
        if jh is None:
            continue
        freqs[i] = modes[jh]['freq']
        modes[jh]['target'] = i
        if meas is not None and r_lo.get('measShapes') and i < len(r_lo['measShapes']):
            meas[i] = r_lo['measShapes'][i]
    r_hi['freqs'] = freqs
    if meas is not None:
        r_hi['measShapes'] = meas
    if estimate:
        try:
            r_hi['gridEstimate'] = grid_estimate(braces, polish_geo, fine_geo, r_lo, r_hi, track, n_modes, meas_xy)
        except Exception:
            r_hi['gridEstimate'] = None
    return r_hi

def _seed_pattern(p, key, stage='coarse'):
    """P1.7: with a 'seed' in the payload, reseed Python's and numpy's RNGs
    per pattern and stage, so a run is reproducible and the sequential
    (browser) and parallel (desktop) pipelines draw the same numbers."""
    seed = p.get('seed')
    if seed is None or seed == '':
        return
    _MODE_REQUEST.clear()   # its guesses depend on what ran before; start every seeded stage from the same state
    import zlib
    v = (int(seed) * 1000003 + zlib.crc32(f'{key}/{stage}'.encode())) % (2**32)
    random.seed(v); np.random.seed(v)

async def run_all_patterns(payload_json):
    try:
        CANCEL_FLAG['stop'] = False
        p = json.loads(payload_json)
        geo = p['geo']
        targets = p['targets']  # [{freq,tol,weight,matchShape,measShape:[...] or None}]
        geo['targetFreqs'] = [float(t['freq']) for t in targets]   # lets evaluate_design match modes to targets by strength
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
        BRACE_MATERIAL['modulus'] = float(p.get('braceModulus', 9.79e9))
        BRACE_MATERIAL['density'] = float(p.get('braceDensity', 350.0))
        BRACE_MATERIAL['shear'] = float(p.get('braceShear', 0.65e9))
        BRACE_LIMITS['min_tip_width'] = max(1.0, float(p.get('minTipWidthMm') or 1.0))
        # Mixed-material search is Experimental only, per an explicit
        # request that plain Novel keep using a single material like every
        # other pattern. Two options for now (spruce, carbon fiber); the
        # UI's materials table is the source of truth for their properties.
        # Gated behind its own toggle (default off) -- it was previously
        # always wired up whenever Experimental was selected despite the
        # UI calling it "preview only, not implemented," which left no way
        # to actually opt in or out independent of the pattern itself.
        exp_materials = [
            {'modulus': float(p.get('expMatSpruceE', 9.79e9)), 'density': float(p.get('expMatSpruceRho', 350.0)), 'shear': float(p.get('expMatSpruceG', 0.65e9))},
            {'modulus': float(p.get('expMatCarbonE', 135e9)), 'density': float(p.get('expMatCarbonRho', 1600.0)), 'shear': float(p.get('expMatCarbonG', 5e9))},
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
            _seed_pattern(p, 'xbrace')
            coarse_x, cost, _, braces = await sa_continuous(xbrace_encode, XBRACE_BOUNDS, geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'xbrace')
            results['xbrace'] = (braces, cost, coarse_x)
        if 'fan' in patterns and not CANCEL_FLAG['stop']:
            _seed_pattern(p, 'fan')
            enc = lambda pp, g: fan_encode(pp, fan_n, g)
            coarse_x, cost, _, braces = await sa_continuous(enc, fan_bounds(fan_n), geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'fan')
            results['fan'] = (braces, cost, coarse_x)
        if 'falcate' in patterns and not CANCEL_FLAG['stop']:
            _seed_pattern(p, 'falcate')
            coarse_x, cost, _, braces = await sa_continuous(falcate_encode, FALCATE_BOUNDS, geo, targets, meas_xy,
                                                    defl_limit_mm, mass_weight, iters, _progress, 'falcate')
            results['falcate'] = (braces, cost, coarse_x)
        if 'novel' in patterns and not CANCEL_FLAG['stop']:
            _seed_pattern(p, 'novel')
            braces, cost, _ = await sa_free_topology(geo, targets, meas_xy, defl_limit_mm, mass_weight,
                                                    max_braces, iters, _progress, 'novel', symmetric,
                                                    max_height=novel_max_height, max_width=novel_max_width,
                                                    max_length_mm=novel_max_length)
            results['novel'] = (braces, cost, None)
        if 'novelExperimental' in patterns and not CANCEL_FLAG['stop']:
            _seed_pattern(p, 'novelExperimental')
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
            _seed_pattern(p, 'custom')
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
        display_geo = dict(geo); display_geo['GN'] = min(int(p.get('displayGN', 150)), MAX_FINE_GN)
        # The card reports a re-solve on the fine display grid. The polish aims at
        # grid-corrected targets (see grid_correction) so that re-solve lands on target.
        use_display_solve = bool(p.get('displayResolve', True)) and display_geo['GN'] > hifi_geo['GN']
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
            polish_targets, polish_defl, grid_info = targets, defl_limit_mm, None
            if polish_enabled and use_display_solve:
                BRIDGE['onOptimizerLog'](f"[{key}] grid correction: solving the coarse winner at GN {hifi_geo['GN']} and GN {display_geo['GN']} (the GN {display_geo['GN']} solve can take a minute)")
                await asyncio.sleep(0)
                polish_targets, polish_defl, grid_info = grid_correction(
                    braces, hifi_geo, display_geo, targets, n_modes, meas_xy, defl_limit_mm)
                if grid_info:
                    BRIDGE['onOptimizerLog'](f"[{key}] grid correction: " + ', '.join(f'{(k-1)*100:+.1f}%' for k in grid_info['factors']) + " per target; polishing toward the shifted targets")
                await asyncio.sleep(0)
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
                _seed_pattern(p, key, 'polish')
                polished_braces, polished_cost, r = await sa_free_topology(
                    hifi_geo, polish_targets, meas_xy, polish_defl, mass_weight, cap,
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
                _seed_pattern(p, key, 'polish')
                _, polished_cost, r, polished_braces = await sa_continuous(
                    enc, bounds, hifi_geo, polish_targets, meas_xy, polish_defl, mass_weight,
                    polish_iters, _progress, key+'-polish', x0=coarse_x, temp_scale=0.15)
            # One more solve at a much higher resolution for the numbers
            # actually displayed. With braces on it this plate converges
            # from below (modes rise a few percent from the polish grid to
            # a converged one, the higher modes most), and the polish above
            # aimed at grid-corrected targets so this solve lands on them.
            # It runs once per pattern, not per iteration.
            if not CANCEL_FLAG['stop'] and use_display_solve:
                BRIDGE['onOptimizerLog'](f"[{key}] re-solving the polished design at GN {display_geo['GN']} for the result card")
                await asyncio.sleep(0)
                r_display = fine_resolve(polished_braces, hifi_geo, display_geo, n_modes, meas_xy)
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
                field_geo = display_geo if use_display_solve else hifi_geo
                r_field = evaluate_design(field_geo, polished_braces, n_modes, meas_xy, want_field=True)
                if r_field.get('ok'):
                    r['fieldShapes'] = r_field.get('fieldShapes')
                    r['fieldNx'] = r_field.get('fieldNx')
                    r['fieldNy'] = r_field.get('fieldNy')
                await asyncio.sleep(0)
            r['gridCorrection'] = grid_info
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
                'modes': r.get('modes'), 'hiddenStrong': r.get('hiddenStrong'), 'bridgeMassG': r.get('bridgeMassG'),
                'torqueResolved': r.get('torqueResolved', True),
                'torqueRequested': r.get('torqueRequested', False),
                'gridCorrection': r.get('gridCorrection'),
                'gridEstimate': r.get('gridEstimate'), 'gridMm': r.get('gridMm'), 'pointMassG': r.get('pointMassG'),
                'seed': p.get('seed'),
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
        geo['targetFreqs'] = [float(t['freq']) for t in targets]
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
                'modes': r.get('modes'), 'hiddenStrong': r.get('hiddenStrong'), 'bridgeMassG': r.get('bridgeMassG'),
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
    geo['targetFreqs'] = [float(t['freq']) for t in targets]
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
    BRACE_MATERIAL['modulus'] = float(p.get('braceModulus', 9.79e9))
    BRACE_MATERIAL['density'] = float(p.get('braceDensity', 350.0))
    BRACE_MATERIAL['shear'] = float(p.get('braceShear', 0.65e9))
    BRACE_LIMITS['min_tip_width'] = max(1.0, float(p.get('minTipWidthMm') or 1.0))
    exp_materials = [
        {'modulus': float(p.get('expMatSpruceE', 9.79e9)), 'density': float(p.get('expMatSpruceRho', 350.0)), 'shear': float(p.get('expMatSpruceG', 0.65e9))},
        {'modulus': float(p.get('expMatCarbonE', 135e9)), 'density': float(p.get('expMatCarbonRho', 1600.0)), 'shear': float(p.get('expMatCarbonG', 5e9))},
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
    _seed_pattern(p, key)
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
    display_geo = dict(geo); display_geo['GN'] = min(int(p.get('displayGN', 150)), MAX_FINE_GN)
    # The card reports a re-solve on the fine display grid. The polish aims at
    # grid-corrected targets (see grid_correction) so that re-solve lands on target.
    use_display_solve = bool(p.get('displayResolve', True)) and display_geo['GN'] > hifi_geo['GN']
    n_modes = len(targets)
    polish_iters = max(40, min(200, iters//4))
    polish_enabled = p.get('polishEnabled', True)
    ENCODERS = {
        'xbrace': (xbrace_encode, XBRACE_BOUNDS),
        'fan': (lambda pp, g: fan_encode(pp, fan_n, g), fan_bounds(fan_n)),
        'falcate': (falcate_encode, FALCATE_BOUNDS),
    }
    def _log(text):
        if progress_queue is not None:
            try:
                progress_queue.put(('__log__', text))
            except Exception:
                pass
    polish_targets, polish_defl, grid_info = targets, defl_limit_mm, None
    if polish_enabled and use_display_solve and not CANCEL_FLAG['stop']:
        _log(f"[{key}] grid correction: solving the coarse winner at GN {hifi_geo['GN']} and GN {display_geo['GN']} (the GN {display_geo['GN']} solve can take a minute)")
        polish_targets, polish_defl, grid_info = grid_correction(
            braces, hifi_geo, display_geo, targets, n_modes, meas_xy, defl_limit_mm)
        if grid_info:
            _log(f"[{key}] grid correction: " + ', '.join(f'{(k-1)*100:+.1f}%' for k in grid_info['factors']) + " per target; polishing toward the shifted targets")
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
        _seed_pattern(p, key, 'polish')
        polished_braces, polished_cost, r = await sa_free_topology(
            hifi_geo, polish_targets, meas_xy, polish_defl, mass_weight, cap,
            this_polish_iters, _report_progress, key+'-polish', sym, warm_braces=braces, temp_scale=0.15,
            materials=mats, lock_positions=lockp, **limit_kwargs)
    else:
        enc, bounds = ENCODERS[key]
        _seed_pattern(p, key, 'polish')
        _, polished_cost, r, polished_braces = await sa_continuous(
            enc, bounds, hifi_geo, polish_targets, meas_xy, polish_defl, mass_weight,
            polish_iters, _report_progress, key+'-polish', x0=coarse_x, temp_scale=0.15)

    if not CANCEL_FLAG['stop'] and use_display_solve:
        _log(f"[{key}] re-solving the polished design at GN {display_geo['GN']} for the result card")
        r_display = fine_resolve(polished_braces, hifi_geo, display_geo, n_modes, meas_xy)
        if r_display.get('ok'):
            r = r_display
    if not CANCEL_FLAG['stop'] and r.get('ok') and not r.get('fieldShapes'):
        field_geo = display_geo if use_display_solve else hifi_geo
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
                'modes': r.get('modes'), 'hiddenStrong': r.get('hiddenStrong'), 'bridgeMassG': r.get('bridgeMassG'),
        'torqueResolved': r.get('torqueResolved', True),
        'torqueRequested': r.get('torqueRequested', False),
        'gridCorrection': grid_info,
        'gridEstimate': r.get('gridEstimate'), 'gridMm': r.get('gridMm'), 'pointMassG': r.get('pointMassG'),
        'seed': p.get('seed'),
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
            elif msg and msg[0] == '__log__':
                BRIDGE['onOptimizerLog'](msg[1])
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

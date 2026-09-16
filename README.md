# Bracing Studio

**Design & Draw, One Tool** — an inverse-modal bracing design tool for guitar/violin soundboards, running entirely client-side in the browser.

Bracing Studio is a single self-contained HTML file (`index.html`) that embeds a full orthotropic-plate physics engine (Python via [Pyodide](https://pyodide.org/), running on WebAssembly) alongside a JavaScript/Canvas UI and a Three.js 3D mode-shape viewer. There is no server-side component and no build step — nothing leaves your computer. Open the file (or serve it statically) and it runs.

## What it does

The tool takes you through four steps, each its own tab:

1. **Setup** — trace or pick a preset plate outline (dreadnought, OM/000, 4/4 violin, or a custom hand-traced body), set thickness/material/orthotropic stiffness properties, place the bridge footprint and load, and optionally place a soundhole.
2. **Targets** *(optional)* — import measured impulse-response data (`.TRF`/`.TRV` and `.AvR`/`.AvC` binary formats, matching the ObieWebApp technical reference layout) to detect resonant peaks and extract mode shapes, or skip straight past this step and hand-type frequency targets, or skip it entirely if you just want to draw and evaluate a design with no targets at all.
3. **Braces** — either:
   - **Auto-generate**: run a simulated-annealing search for one or more bracing patterns (X-brace, fan, falcate, Novel, Experimental, or a Custom uploaded/drawn seed) against your frequency/mode-shape targets, a deflection limit, and a mass penalty; or
   - **Draw manually**: click directly on the traced plate to place straight or curved braces, set per-brace dimensions/material, and get an immediate physics solve with no search involved.
4. **Results** — compare frequencies/mode shapes/deflection across every generated or hand-drawn design side by side, including an animated 3D mode-shape viewer per result.

The plate physics (finite-difference biharmonic orthotropic-plate model, brace mass/stiffness coupling, eigenvalue and static-deflection solves) is identical whether a design comes from the search or from the manual editor, and has been validated against the closed-form Navier series solution for a simply-supported rectangular orthotropic plate (~1% agreement across the first several modes).

## Recent changes

- **Hand-drawn designs now lock brace positions by default.** Using "Send to optimizer to refine…" from the manual brace editor now automatically enables "Lock brace positions" in the Custom pattern settings, since refining a deliberately hand-placed layout should tune cross-section sizing, not relocate braces, unless the user turns it back off.
- **Fixed the Novel/Experimental symmetry axis being hard-coded to the center of the drawing window.** The mirror line used by the symmetric free-topology search is now computed from the actual traced outline's geometric centroid (`_outline_symmetry_axis_fy`, an area-weighted polygon centroid), so a lopsided or off-center traced body gets a symmetry axis that matches its real shape instead of assuming a perfectly centered plate.
- **Fixed fan/X-brace/falcate presets extending past the plate's outline.** These fixed-topology patterns are now hard-clipped to the traced outline using exact Bezier curve subdivision (De Casteljau splitting), rather than relying on a soft search penalty that could never actually move a non-searched, fixed-position brace back inside the boundary.
- **Higher-resolution, controllable 3D mode-shape animation.** The mesh driving the animated 3D viewer went from a 44×34 to a 96×74 grid for a visibly smoother, less "blocky" surface. Each result card also gained **Speed** (0.25×–3×) and **Exaggeration** (0.25×–4×) sliders, with video export automatically compensating its loop timing so a "5 loop" export stays accurate at any speed setting.
- **Fixed the 3D mode-shape plate blending into the background.** The zero-displacement vertex color coincidentally matched the viewer's background color; the neutral color is now a distinct pale spruce tone, and a dark outline + soundhole boundary line was added around the animated surface so the plate edge and soundhole stay visible at every phase of the animation.
- **Performance: the solver no longer freezes page navigation.** The simulated-annealing loops used to yield control back to the browser only on throttled progress-update ticks; they now yield (`await asyncio.sleep(0)`) every iteration, decoupled from the (still throttled) progress callback, so the UI stays responsive while a search runs.
- **Brace discretization is now user-controlled.** A "Brace discretization" slider sets the target beam-segment length (5–50 mm) used when converting a brace into mass/stiffness contributions — shorter segments are more accurate for tapered or curved braces at some speed cost, longer segments are faster. Applies to every generated or hand-drawn design.

## Architecture notes

- Everything lives in `index.html` — HTML, CSS, JS UI/state logic, and the Python physics/optimizer engine (embedded as a `<script type="py">` block) are all in one file, loaded and executed by Pyodide in the browser.
- `fx`/`fy` are the core geometry convention throughout: fractional position along the plate's length and width respectively (0–1), independent of absolute plate size.
- Braces are stored as `line` / `curve` (quadratic) / `curve3` (cubic) control-point structures and discretized into straight segments (`_sample_brace`) for both mass and bending-stiffness contributions.
- Two search strategies: `sa_continuous` (fixed-topology templates — X-brace/fan/falcate — where only cross-section dimensions are search variables) and `sa_free_topology` (Novel/Experimental/Custom — brace count, position, and shape are search variables too).
- The 3D mode-shape viewer is a ported/extended Three.js scene (vertex-colored mesh + animated outline/soundhole/brace line loops), driven by `requestAnimationFrame` and reusing the same sampled displacement field as the 2D results.
- `legacy/` contains earlier standalone designer/visualizer tools this project grew out of; they are not part of the active app.

## Running it

No build step. Serve the directory with any static file server and open `index.html`, e.g.:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/index.html
```

Opening the file directly (`file://`) may not work in all browsers due to module/WASM loading restrictions — a local static server is the reliable path.

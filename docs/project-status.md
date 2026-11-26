# Bracing Studio: project status brief

Prepared 2026-11-25 as source material for a project poster. It covers what the project is, how it got here, where validation stands, and what's left. Numbers come from the repo ([README.md](../README.md), [docs/validation.md](validation.md)) and from the ANSYS comparison runs of 16–17 November 2026. Anything not yet confirmed is marked as open.

**Voice note for the poster.** Write as Luca, in the first person, following the writing guide in `~/.claude/CLAUDE.md`:
- narrate the build in order
- keep sentences plain
- admit failures plainly, then explain the fix
- put concrete numbers in the prose
- use a formal Abstract / Methods / Results / Error / Conclusions block for the technical results

The in-app documentation stays impersonal, but the poster is a project write-up and should be in Luca's voice.

---

## 1. What it is

Bracing Studio designs guitar-top bracing in reverse. You describe the soundboard and the frequencies you want it to ring at, and it searches for a bracing pattern that hits them. The soundboard description covers:
- the traced outline, sound hole and bridge
- the orthotropic wood
- the string load and the bridge-deflection limit

Traditional bracing is designed forwards: pick a pattern, glue it in, tap the top, and see what you got.

- **Physics:** an orthotropic Reissner-Mindlin plate solved by finite differences on a cut-cell grid over the traced outline. Braces are tapered Timoshenko beams glued along straight or curved paths, with a T-section flange, a footprint stiffening term and torsion. The bridge is a hardwood block. The edge is clamped, simply supported or free. The solver returns natural frequencies, mode shapes, mode strength under bridge drive, bridge deflection and brace mass.
- **Search:** simulated annealing over brace size, position, shape and count. Patterns are ordered from least to most freedom:
  1. X-brace, fan and falcate (fixed layout, sizes tuned)
  2. Custom (refines a drawn or uploaded design)
  3. Novel (left-right symmetric)
  4. Experimental (asymmetric, optionally mixing spruce and carbon)

  The search runs on a coarse grid (GN 34), then polishes on a finer grid with a per-mode grid-convergence correction. Each card reports a re-solve at GN 150 plus a Richardson-extrapolated grid estimate.
- **Two builds, one engine:**
  - a browser app ([index.html](../index.html), Python running through Pyodide)
  - a desktop app ([desktop/](../desktop/), pywebview + multiprocessing + STEP export via cadquery-ocp)

  Both load the same [desktop/engine.py](../desktop/engine.py).
- **Workflow:** Setup → Targets (optional tap-test import: `.TRF/.TRV`, `.AvR/.AvC`) → Braces (draw or auto-generate) → Results. Results has animated 3D mode shapes and STL, STEP, template SVG and simulation-info exports. It also has a compare panel for ANSYS mode shapes.

## 2. How it got here (timeline)

| When (2026) | Milestone |
|---|---|
| July | Two separate single-file tools: a target **designer** and a brace **visualizer** (now in [legacy/](../legacy/)). |
| July–Sept | Merged into one tool, "Design & Draw, One Tool". A generated design and a hand-drawn one became the same object, solved by the same engine. |
| 14 Sept | Fixed the page freezing during a search by yielding to the browser every annealing iteration. Fixed the 3D viewer, brace clipping to the outline, and the symmetry axis (now the outline's area-weighted centroid). |
| 16 Sept | Hand-drawn designs lock brace positions by default. First README. |
| 7 Oct | **Desktop build.** Pyodide pins everything to one thread, so each pattern now gets its own CPU process. A four-pattern run at 400 iterations went from 14.9 s to 5.5 s (about 2.7×) on 10 cores. |
| 13 Oct | 3D models, project import/export, **STEP export** for CAD/FEA. |
| 14 Oct | Brace torsion. **First ANSYS check:** 8 of 9 modes within 3%, but mode 3 was 406 Hz in the tool against 359 Hz in ANSYS (13%). The cause was thin-plate theory ignoring spruce's tiny rolling shear. A bending/shear split only recovered 14% of a 21% effect, so the top became a full **Reissner-Mindlin plate** (shear strains on grid links). A symmetric factorization reused for the mode and deflection solves brought search time back to about 2× the old one. |
| 27 Oct | Removed the violin preset; documentation rewrite. |
| 2 Nov | Next ANSYS run had every mode 5–15% *above* the tool. Part of it was self-inflicted: the bridge had the spruce brace material (13.7 g instead of 33.2 g). With that fixed, the remaining 3–11% matched the tool's own grid error. That led to the **grid-corrected polish**: per-target factor k_i = f_fine / f_polish, modes followed between grids by MAC. |
| 3 Nov | "Min tip width" search limit (3 mm default) after narrow, tall brace tips were suspected of being overstiff. Tiptest fundamental gap shrank from −12% to −7%. |
| 5 Nov | Acrylic brace material preset. |
| 15 Nov | **Solver v2** (commit `6a76fcf`): cut-cell edges with a ghost ring on the true outline, brace footprint stiffening with a series rule, a 40-thickness flange, a three-layer brace-over-bridge section, and a nine-stage **validation suite** ([tests/validation.py](../tests/validation.py), report in [docs/validation.md](validation.md)). `solverVersion: 'v1'` reproduces the old solver exactly. |
| 16 Nov | Commit `4e2d961` ("v2"). In the app:<ul><li>an **ANSYS compare panel** (mode exports in, MAC pairing and heatmap out; several files at once; unscoped exports cut to the plate face)</li><li>a simpler **Edit by hand** loop (opens the draw view directly, solve resolution next to Evaluate, result in its own "Hand-drawn solve" section with the ANSYS modes carried over)</li><li>default spruce shear changed to G_LT 1.17 / G_RT 0.1 GPa to match the published material in the ANSYS model</li><li>**bare-plate reference** now solves the top without braces *and* without the bridge</li></ul> |
| 16–17 Nov | **ANSYS isolation campaign** (section 4). |

## 3. Validation against trusted answers

From [docs/validation.md](validation.md) (commit `6a76fcf`, **39 of 56 checks pass**). Most failures are listed known limits (brace-to-grid ripple) or the old ANSYS runs.

1. **Exact solutions.**
   - Simply supported orthotropic rectangle vs Navier: 0.14% (thin plate), 0.22% (2.8 mm Mindlin), 0.53% (10 mm).
   - Clamped, simply supported and free circles vs Leissa: within 1% from GN 50. The clamped fundamental is within 0.66% even at GN 34; the old staircase clamp read 8.5% low.
   - Clamped square: 0.9%.
   - Clamped plate with a free hole vs a Morley FE reference: 0.55%.
2. **Independent 3D solid model** (scikit-fem, quadratic bricks, clamped 300 × 200 mm spruce plate):
   - The bare plate agrees to 0.9%.
   - Single, parallel and crossing braces, a bridge, and a brace over the bridge agree within 2.1–5.0% on modes 1–6.
   - The old v1 line-beam braces read 15–19% low.
3. **Convergence.**
   - Bare tops move under 0.4% from GN 80 to GN 200.
   - Braced tops stay within 1.4–6.6% over GN 34–200; v1 drifted 8–24%.
   - Plots for 10 designs are in [docs/validation/](validation/) as `convergence_*.svg`.
4. **Optimizer.** The same seed gives the same design. Falcate on reachable targets lands within tolerance on 4 of 5 seeds.
5. **Performance.** A GN 34 evaluation takes 53 ms, down from 92 ms. 200 random layouts ran with no failures.

## 4. Where it stands now: the ANSYS isolation campaign (16–17 Nov)

Design under test: `V2test2` (5 curved braces, structural bridge 155 × 28 × 9 mm, clamped, isotropic brace shear). The full braced top showed a scatter the grid couldn't explain:

| ANSYS (MPC contacts) | 207.3 | 332.1 | 415.8 | 471.5 | 553.8 | 640.8 |
|---|---|---|---|---|---|---|
| Tool | 201.7 | 370.6 | 438.4 | 507.1 | 547.2 | 663.6 |
| Tool vs ANSYS | −2.7% | +11.6% | +5.4% | +7.6% | −1.2% | +3.6% |

Shapes match well: MAC 0.92–0.99 on a clean diagonal, so the frequencies are being compared like for like. The tool has one extra mode with no ANSYS partner in the first six.

The model was then taken apart one piece at a time:

| Step | Result | Conclusion |
|---|---|---|
| Re-solve at GN 200 | No movement towards ANSYS | Grid error ruled out |
| ANSYS contacts penalty → MPC | ANSYS dropped about 1% | Contact compliance ruled out |
| **Bare plate** (no braces, no bridge) | ANSYS 111.38 / 202.11 / 227.08 / 313.13 / 329.23 / 351.42 vs tool 111.2 / 201.4 / 226.7 / 312.4 / 328.4 / 351.1 Hz: **within 0.4%** | Plate, clamped edge and material cleared |
| **Plate + bridge** | Within about 2% (mode 2 +2.2%, others ≤1.4%) | Bridge cleared |
| Mass | Tool 183.2 g (plate 128.5 + bridge 33.2 + braces 21.5) vs ANSYS 183.3 g | Mass cleared |
| Brace material | ANSYS braces are isotropic (E 9.79 GPa, ν 0.3) | Grain orientation not involved |
| Brace roll inertia (added to the engine as a test) | Mode 2 −0.2% | Ruled out; reverted |
| Flange width 40 → 5 thicknesses | Mode 2 +16.5% → +12.6% (in the cluster below) | Not the main cause |
| **Brace clusters** | Braces 2 + 3: mostly within 3%. Braces 1 + 4 + 5 (one fused ANSYS body): **mode 2 +16.5%, mode 4 +18.2%** | The error lives in that cluster |
| Per-brace weakening in the tool | Only brace 1 moves modes 2 and 4. ANSYS sits between "brace 1 at ¼ stiffness" and "brace 1 removed" | **Brace 1 is the suspect** |

**Brace 1:**
- a tall, thin blade, 16.8 mm tall at the centre tapering to 4.9 mm, and 3.4 mm wide tapering to 3.0 mm
- its centreline passes 5.2 mm from the sound-hole edge, which leaves about 3.5 mm of plate between the brace and the hole
- ANSYS mode 2 peaks at that stretch of hole edge

Narrow, tall braces have been suspected of reading overstiff since early November. That's why the min-tip-width limit was added.

**Open question:** is the factor-of-four disagreement a real tool error for this brace type, or an ANSYS geometry/bonding problem with brace 1? Nothing confirms either yet.

## 5. What needs to be done

Most urgent first:

1. **Confirm the cluster pairing by MAC.** Delete braces 2 and 3 in the hand editor and solve at GN 160. Load the ANSYS 1 + 4 + 5 run's Directional Deformation (Z) modes into the compare panel. The +16/+18% assumes modes pair in frequency order.
2. **Check brace 1 in ANSYS:**
   - its measured height, width and taper against the tool
   - continuity up to where it meets brace 4
   - the bonded region covering its full 3 mm footprint (Contact Tool → Initial Information: all Closed, zero gap)
   - at least two elements across its width
   - the small cube-shaped body at the base of the large brace, which looks like a trimming leftover
3. **If brace 1 checks out, it's a solver problem.** Add a 3D-solid validation case to stage 6: plate plus one narrow, tall blade (3–4 mm wide, about 17 mm tall), alone and a few millimetres from a free hole edge. Then fix the brace model. Candidates are the effective flange next to a free edge and how a narrow footprint couples to the plate.
4. **Re-validate the grid-corrected polish against ANSYS.** It was added on 2 Nov and has never been checked end to end.
5. **Widen coverage.** Repeat the plate → bridge → brace-cluster ladder on a conventional X-brace and a fan design, so the conclusion isn't one design deep.
6. **Bring the docs up to date.**
   - The [README](../README.md) still says the bare-plate ANSYS run is pending and quotes predictions made with the old 0.6/0.04 shear. That run is now done and matches within 0.4%.
   - Regenerate [docs/validation.md](validation.md) after the brace fix. Its stage 7 still compares against the older ANSYS runs.
7. **Known limits still open** (from the validation report):
   - Brace-to-grid ripple of about 1% per 0.1-cell move at GN 34. The planned fix is a conforming finite-element solve.
   - The brace fundamental reads 2–4% low against the 3D solid.
   - Flat top (no doming), rectangular tapered brace sections, and grain assumed along the body length.
8. **Longer term.**
   - Compare against a physically built and tapped top (the compare panel already has a "measured Hz" column).
   - Package the desktop app as an installer; it currently runs from source.

## 6. Material available for the poster

- Logo: [logofinal.png](../logofinal.png).
- Convergence plots (SVG, modes 1–3, v1 dashed vs v2 solid): [docs/validation/](validation/) `convergence_*.svg`.
- Tables in [docs/validation.md](validation.md), especially stage 6 (3D solid, v2 vs v1 per mode).
- From Luca's own screenshots (not in the repo; ask for them):
  - ANSYS mode shapes for the full top and both brace clusters
  - the app's MAC heatmap (ANSYS modes 1–6 vs tool modes 1–8)
  - ANSYS Tabular Data for the bare-plate, plate + bridge and MPC runs
  - the V2test2 geometry render

Good figure candidates:
1. The isolation ladder as a step chart: plate 0.4% → plate + bridge 2% → full top −3/+12% → cluster 1 + 4 + 5 +16/+18%.
2. Bare-plate tool vs ANSYS bars.
3. The MAC heatmap.
4. v1 vs v2 error against the 3D solid.
5. The 2.7× desktop speed-up.

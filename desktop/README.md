# Bracing Studio — experimental desktop build

This is an experimental native wrapper around the **same HTML/CSS/JS UI** as the browser version (`../index.html`), built to answer one specific question: is the browser/Pyodide architecture actually a hard performance ceiling, and is it worth trading "open a link, it just runs" for "install once, run natively"?

It is **not** a rewrite. `ui/index.html` is a copy of the real app with only the Pyodide-specific glue removed; every UI feature, every physics function, every search algorithm is the same code. The only thing that's different is what runs the Python: instead of Pyodide (Python compiled to WebAssembly, pinned to the browser tab's single main thread), it's a real, native Python process on your machine, hosted in an OS-native webview via [pywebview](https://pywebview.flowrl.com/).

## Why this exists

The browser build has a structural ceiling the HTML/PyScript framing itself doesn't cause, but does lock in: Pyodide runs entirely on the page's one main JS thread, so numpy/scipy run as WASM (slower than native BLAS) and — more importantly — every pattern in "Generate & compare" (X-brace, fan, falcate, Novel, Experimental, Custom) has always had to solve **one after another**, even though they've never depended on each other.

This build fixes that specific thing: `engine.py` adds `run_all_patterns_parallel`, which runs each requested pattern in its own OS process via `ProcessPoolExecutor` — genuine multi-core parallelism, not just concurrency. Measured on this machine (10 cores, 4 patterns, 400 search iterations each): **14.9s sequential → 5.5s parallel**, about 2.7x. Cancel ("Stop") was also verified to actually interrupt in-progress worker processes within ~1s.

## What's here

- `engine.py` — the physics/optimizer engine, copied from index.html's embedded `<script type="py">` block, with Pyodide's `js`/`pyscript.ffi` glue replaced by a plain-Python `BRIDGE` callback table (`set_bridge(...)`), plus the new `run_all_patterns_parallel` / `_solve_pattern_async` / `_solve_pattern_sync` for multiprocess dispatch.
- `app.py` — the pywebview host. Exposes `run_all_patterns`, `evaluate_manual_design`, `compute_bare_modes`, `cancel_optimizer` to the page as `window.pywebview.api.*`, and relays engine callbacks back into the page via `window.evaluate_js(...)`, calling the exact same `onOptimizerProgress`/`onOptimizerDone`/`onOptimizerError`/`onManualEvalDone`/`onManualEvalError`/`onBareModesResult`/`onBareModesError`/`onPyEngineReady` handlers the browser build already defines.
- `ui/index.html` — a copy of `../index.html` with the PyScript bootstrap tags and the embedded Python engine block removed, and the 4 call sites that used to call into Pyodide (`window.pyRunAllPatterns`, `window.pyEvaluateManualDesign`, `window.pyComputeBareModes`, `window.pyCancelOptimizer`) pointed at `window.pywebview.api.*` instead. Nothing else in the file was touched.

## Running it

```bash
cd desktop
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python app.py
```

This opens a native window hosting the UI. "Engine ready" should appear almost immediately (there's no WASM/numpy/scipy download — it's already running natively).

## What's verified vs. not

Verified directly (standalone tests against `engine.py`, not through the actual GUI — see caveat below):
- The engine imports and runs correctly natively (sequential path produces identical results to the browser build's logic).
- `run_all_patterns_parallel` genuinely runs patterns concurrently across processes, with a real, measured speedup.
- Cancel correctly propagates into running worker processes and stops them.

**Not yet verified: the actual GUI.** I could not launch and click through the pywebview window itself in this environment — only the engine and orchestration layer were tested directly (as a library, not through the UI). Before relying on this, actually run `python app.py` and go through a full workflow (trace outline → set targets → generate & compare → results) yourself.

Known rough edges to watch for, inherited from swapping WKWebView (macOS's native webview, which is what pywebview uses here) in for a full Chromium browser:
- **Folder import for measured data** uses the File System Access API (`showDirectoryPicker`) with an existing fallback to a plain `<input webkitdirectory>` picker when that API isn't available. WKWebView doesn't support `showDirectoryPicker`, so it should automatically take the fallback path — untested in the actual window.
- **3D mode-shape video export** uses `canvas.captureStream()` + `MediaRecorder`. Both are commonly available in WKWebView, but this specific combination wasn't exercised here.
- **Progress reporting during "Generate & compare" is coarser** than the browser build in parallel mode: instead of a live per-iteration progress bar per pattern, you'll see each pattern's progress bar jump straight to 100% as that pattern's process finishes, since streaming per-iteration progress across a process boundary needs its own IPC channel that wasn't built for this first pass. `run_all_patterns` (plain sequential, same as the browser build) is still in `engine.py` if you want the old fine-grained behavior — `app.py` just doesn't call it.

## Packaging into a real installable app

Not done here — this is a working prototype, not a distributable build. The natural next step would be [PyInstaller](https://pyinstaller.org/) or [briefcase](https://briefcase.readthedocs.io/) to bundle `app.py`, `engine.py`, and `ui/` into a single executable per OS, so it doesn't need a Python environment installed.

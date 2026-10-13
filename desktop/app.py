"""
Bracing Studio -- experimental native/desktop host.

Wraps the SAME HTML/CSS/JS UI as the browser version (ui/index.html, a copy
of ../index.html with the embedded Pyodide <script type="py"> block and its
PyScript bootstrap <link>/<script> tags removed) in a native OS webview via
pywebview, and swaps Pyodide for engine.py running natively on this
machine: real numpy/scipy (not WASM), and run_all_patterns_parallel
actually spreads independent patterns across CPU cores via
ProcessPoolExecutor -- something the browser build structurally cannot do,
since Pyodide is pinned to the page's single main JS thread.

Run with:
    pip install -r requirements.txt
    python app.py
"""
import asyncio
import base64
import json
import threading
from pathlib import Path

import webview

import engine

UI_PATH = Path(__file__).parent / 'ui' / 'index.html'


class Api:
    """Exposed to the page as window.pywebview.api.<method>(...). Every
    method here returns almost immediately -- the real work runs on a
    background thread, with results delivered back into the page via
    window.evaluate_js(...) calling the SAME onXxx handlers the browser
    build already defines (onOptimizerProgress, onOptimizerDone, ...), so
    index.html's existing callback-handling code needs no changes. Methods
    never block pywebview's call-handling thread, so the window never
    freezes waiting on a long solve."""

    def run_all_patterns(self, payload_json):
        threading.Thread(target=engine.run_all_patterns_parallel, args=(payload_json,), daemon=True).start()
        return True

    def evaluate_manual_design(self, payload_json):
        threading.Thread(target=lambda: asyncio.run(engine.evaluate_manual_design(payload_json)), daemon=True).start()
        return True

    def compute_bare_modes(self, geo_json):
        threading.Thread(target=lambda: asyncio.run(engine.compute_bare_modes(geo_json)), daemon=True).start()
        return True

    def cancel_optimizer(self):
        engine.request_cancel_optimizer()
        return True

    # ── File saving ──
    # pywebview ignores <a download> links by default, so every export goes
    # through a native Save dialog here instead. _window is set by main();
    # the leading underscore keeps pywebview from exposing it to the page.
    _window = None

    def _ask_save_path(self, filename):
        save = webview.FileDialog.SAVE if hasattr(webview, 'FileDialog') else webview.SAVE_DIALOG  # pywebview 6 / 5
        res = self._window.create_file_dialog(save, save_filename=filename)
        if not res:
            return None
        return res if isinstance(res, str) else res[0]

    def save_file(self, filename, data_b64):
        """Save base64-encoded bytes from the page. Returns the chosen path,
        or None if the person cancelled."""
        path = self._ask_save_path(filename)
        if path:
            Path(path).write_bytes(base64.b64decode(data_b64))
        return path

    def export_step(self, payload_json, filename):
        """Build a solved design as STEP solids (see cad_export.py) and write
        it, plus the page's simulation-info JSON alongside it as
        <name>_simulation.json. Returns {ok, path, bodies} or {ok: False, error}."""
        try:
            import cad_export  # optional dependency -- only needed here
        except ImportError:
            return {'ok': False, 'error': 'STEP export needs the OpenCascade kernel: run  pip install cadquery-ocp  in this environment, then restart the app.'}
        path = self._ask_save_path(filename)
        if not path:
            return {'ok': False, 'cancelled': True}
        try:
            payload = json.loads(payload_json)
            bodies = cad_export.export_step(payload, path)
            sim_path = Path(path).with_name(Path(path).stem + '_simulation.json')
            sim_path.write_text(json.dumps(payload['simulationInfo'], indent=2))
            return {'ok': True, 'path': path, 'simPath': str(sim_path), 'bodies': bodies}
        except Exception as exc:
            return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def _js_literal(value):
    """JSON-encode a single bridge-callback argument for interpolation
    into an evaluate_js call -- every value crossing this bridge is
    already a plain string/number/bool/None (the same JSON-safe payloads
    index.html's onXxx handlers were always written to receive), so
    json.dumps is exact and injection-safe."""
    return json.dumps(value)


def main():
    window_ref = {}

    def bridge_call(fn_name, *args):
        window = window_ref.get('window')
        if window is None:
            return
        js_args = ', '.join(_js_literal(a) for a in args)
        window.evaluate_js(f'{fn_name}({js_args})')

    engine.set_bridge(
        onOptimizerProgress=lambda *a: bridge_call('onOptimizerProgress', *a),
        onOptimizerTrace=lambda *a: bridge_call('onOptimizerTrace', *a),
        onOptimizerDone=lambda *a: bridge_call('onOptimizerDone', *a),
        onOptimizerError=lambda *a: bridge_call('onOptimizerError', *a),
        onManualEvalDone=lambda *a: bridge_call('onManualEvalDone', *a),
        onManualEvalError=lambda *a: bridge_call('onManualEvalError', *a),
        onBareModesResult=lambda *a: bridge_call('onBareModesResult', *a),
        onBareModesError=lambda *a: bridge_call('onBareModesError', *a),
    )

    api = Api()
    window = webview.create_window(
        'Bracing Studio (experimental desktop build)',
        url=str(UI_PATH),
        js_api=api,
        width=1440, height=940, min_size=(1000, 700),
    )
    window_ref['window'] = window
    api._window = window

    def on_loaded():
        # No WASM/numpy/scipy download here -- the engine is already
        # running natively, so "ready" fires as soon as the page has.
        window.evaluate_js('onPyEngineReady()')

    window.events.loaded += on_loaded
    webview.start()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # no-op on macOS/Linux; needed if this is ever frozen into a Windows .exe
    main()

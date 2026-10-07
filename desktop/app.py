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
        onOptimizerDone=lambda *a: bridge_call('onOptimizerDone', *a),
        onOptimizerError=lambda *a: bridge_call('onOptimizerError', *a),
        onManualEvalDone=lambda *a: bridge_call('onManualEvalDone', *a),
        onManualEvalError=lambda *a: bridge_call('onManualEvalError', *a),
        onBareModesResult=lambda *a: bridge_call('onBareModesResult', *a),
        onBareModesError=lambda *a: bridge_call('onBareModesError', *a),
    )

    window = webview.create_window(
        'Bracing Studio (experimental desktop build)',
        url=str(UI_PATH),
        js_api=Api(),
        width=1440, height=940, min_size=(1000, 700),
    )
    window_ref['window'] = window

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

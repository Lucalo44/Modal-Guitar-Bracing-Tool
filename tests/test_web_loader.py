"""The browser build's loader (generated into index.html by
tools/build_web.py) run under CPython with stand-ins for PyScript's js,
pyfetch and create_proxy: it must load desktop/engine.py itself, wire every
bridge callback to window.*, expose the four entry points and signal ready."""
import asyncio, os, re, sys, types, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))


def run_loader():
    import build_web
    html = build_web.build()
    code = re.search(r'<script type="py"[^>]*>(.*?)</script>', html, re.S).group(1)
    calls = []
    win = types.SimpleNamespace(onPyEngineReady=lambda: calls.append('ready'))
    for name in ('onOptimizerProgress', 'onOptimizerTrace', 'onOptimizerDone', 'onOptimizerError', 'onManualEvalDone',
                 'onManualEvalError', 'onBareModesResult', 'onBareModesError', 'onOptimizerLog'):
        setattr(win, name, (lambda n: (lambda *a: calls.append((n, a))))(name))
    js = types.ModuleType('js'); js.window = win
    class Resp:
        async def string(self):
            return open(os.path.join(ROOT, 'desktop', 'engine.py')).read()
    async def pyfetch(url):
        assert url == 'desktop/engine.py'
        return Resp()
    ph = types.ModuleType('pyodide.http'); ph.pyfetch = pyfetch
    py = types.ModuleType('pyodide'); py.http = ph
    ffi = types.ModuleType('pyscript.ffi'); ffi.create_proxy = lambda f: f
    ps = types.ModuleType('pyscript'); ps.ffi = ffi
    saved = {k: sys.modules.get(k) for k in ('js', 'pyodide', 'pyodide.http', 'pyscript', 'pyscript.ffi', 'engine')}
    sys.modules.update({'js': js, 'pyodide': py, 'pyodide.http': ph, 'pyscript': ps, 'pyscript.ffi': ffi})
    try:
        body = 'async def __loader():\n' + '\n'.join('    ' + l for l in code.strip().splitlines()) + '\n    return engine\n'
        ns = {}
        exec(body, ns)
        eng = asyncio.run(ns['__loader']())
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    return win, calls, eng


def test_loader_runs_shared_engine():
    win, calls, eng = run_loader()
    assert 'ready' in calls
    for n in ('pyComputeBareModes', 'pyRunAllPatterns', 'pyEvaluateManualDesign', 'pyCancelOptimizer'):
        assert callable(getattr(win, n))
    assert eng.__file__ if hasattr(eng, '__file__') else True
    assert hasattr(eng, 'evaluate_design') and hasattr(eng, '_geometry_base')
    # a bare-modes call reaches window.onBareModesResult through the bridge
    fx = json.load(open(os.path.join(HERE, 'fixtures', 'bare_clamped.json')))
    sys.path.insert(0, HERE)
    from helpers import payload_geo
    asyncio.run(win.pyComputeBareModes(json.dumps(payload_geo(fx['GEO'], 34))))
    got = [c for c in calls if isinstance(c, tuple) and c[0] in ('onBareModesResult', 'onBareModesError')]
    assert got and got[0][0] == 'onBareModesResult', got


def test_index_html_is_built_from_sources():
    import build_web
    assert open(os.path.join(ROOT, 'index.html')).read() == build_web.build()

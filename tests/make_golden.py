"""Stage 0: freeze frequencies, deflection, masses and timings for every
fixture at every GN, on whatever engine is currently checked out.

    python tests/make_golden.py [--tag NAME] [--gns 34,50,...]

Writes tests/golden/<tag>.json. Run on the pre-change commit with
--tag baseline; later runs with another tag are compared against it."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import engine, load_fixture, payload_geo, solve, ladder_braces, GNS, GOLD  # noqa: E402

FIXTURES = ['bare_clamped', 'bare_free', 'bare_ss', 'xbrace', 'fan', 'falcate', 'experimental',
            'parttest', 'tiptest', 'ladder']


def fixture_case(name):
    if name == 'ladder':
        fx = load_fixture('bare_clamped'); G = dict(fx['GEO']); G['bridgeStructural'] = True
        return G, ladder_braces(G)
    fx = load_fixture(name)
    return fx['GEO'], fx['braces']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--gns', default=','.join(map(str, GNS)))
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    gns = [int(g) for g in a.gns.split(',')]
    names = a.only.split(',') if a.only else FIXTURES
    path = os.path.join(GOLD, a.tag + '.json')
    out = json.load(open(path)) if os.path.exists(path) else {'fixtures': {}, 'timing': {}}
    for name in names:
        G, braces = fixture_case(name)
        rec = out['fixtures'].setdefault(name, {})
        for gn in gns:
            t0 = time.perf_counter()
            r = solve(payload_geo(G, gn), braces, n=12)
            dt = time.perf_counter() - t0
            rec[str(gn)] = {'freqs': r['freqs'], 'strength': r['strength'], 'defl': r['defl'],
                            'massG': r['massG'], 'bridgeMassG': r['bridgeMassG'], 'seconds': dt}
            print(f'{name:14s} GN {gn:3d}  {dt:6.2f}s  ' + ' '.join(f'{f:7.1f}' for f in r['freqs'][:6]), flush=True)
            json.dump(out, open(path, 'w'), indent=1)
    json.dump(out, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()

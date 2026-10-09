"""Integration checks for independent CPU jobs. Usage: python test-cpu-jobs.py ECM"""
import argparse
import math
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('ecm', type=Path)
args = parser.parse_args()
exe = args.ecm.resolve()
checks = 0

def check(ok, message):
    global checks
    assert ok, message
    checks += 1

def records(path):
    if not path.exists():
        return []
    return [dict(re.findall(r'(\w+)=([^;]*);', line))
            for line in path.read_text().splitlines() if line.strip()]

def numeric(rows):
    keys = ('METHOD', 'N', 'B1', 'X', 'Y', 'SIGMA', 'A', 'PARAM')
    return [tuple((k, (r.get(k, '0') if k == 'PARAM' and r.get('METHOD') == 'ECM' else r.get(k))) for k in keys) for r in rows if r.get('N') != '1']

with tempfile.TemporaryDirectory(prefix='ecm-cpu-jobs-') as tmp:
    root = Path(tmp)
    serial = 0

    def run(opts, text='', save=True):
        global serial
        serial += 1
        path = root / f'{serial}.save'
        cmd = [str(exe), *opts[:-2]]
        if save:
            cmd += ['-save', str(path)]
        cmd += opts[-2:]
        r = subprocess.run(cmd, input=text, cwd=root, text=True, capture_output=True, timeout=180)
        check(r.returncode in (0, 2, 6, 8, 10, 14), (cmd, r.returncode, r.stdout, r.stderr))
        (root / f'{serial}.log').write_text(r.stdout + r.stderr)
        return r, path, records(path)

    numbers = [2**p - 1 for p in (1279, 127, 607, 521, 127, 607, 521, 127)]
    text = '\n'.join(map(str, numbers)) + '\n'
    methods = [(['-pm1', '-x0', '3'], 'P-1'), (['-pp1', '-x0', '3'], 'P+1')]
    methods += [(['-sigma', f'{p}:11'], 'ECM') for p in range(4)]
    for opts, method in methods:
        a, _, ra = run([*opts, '1000', '0'], text)
        b, _, rb = run(['-threads', '4', *opts, '1000', '0'], text)
        check(numeric(ra) == numeric(rb), ('stage1', opts, numeric(ra), numeric(rb)))
        check([int(r['N']) for r in rb] == numbers, ('input order', opts))
        check([int(n) for n in re.findall(r'Input number is (\d+)', b.stdout)] == numbers, 'console order')
        check(a.returncode == b.returncode, 'return code')

    # P-1 residues independently checked using Python modular exponentiation.
    exponent = math.lcm(*range(1, 1001))
    _, _, rows = run(['-threads', '8', '-pm1', '-x0', '3', '1000', '0'], text)
    check(all(int(r['X'], 16) == pow(3, exponent, n) for r, n in zip(rows, numbers)), 'P-1 pow oracle')

    # Mixed methods and shuffled input sizes when extending a resume file.
    mixed = root / 'mixed.save'
    for opts, _ in methods[:3]:
        _, p, _ = run([*opts, '100', '0'], text)
        with mixed.open('a') as f:
            f.write(p.read_text())
    _, _, ra = run(['-resume', str(mixed), '1000', '0'])
    _, _, rb = run(['-threads', '4', '-resume', str(mixed), '1000', '0'])
    check(numeric(ra) == numeric(rb), 'mixed resume ordering/residues')

    # Stage 2, both transform paths, including different Fermat moduli.
    stage2text = text + '\n'.join(str(2**p + 1) for p in (256, 512, 1024)) + '\n'
    for method in ('-pm1', '-pp1', '-ecm'):
        opts = ['-sigma', '0:11'] if method == '-ecm' else [method, '-x0', '3']
        for ntt in ('-ntt', '-no-ntt'):
            a, _, ra = run([*opts, ntt, '1000', '100000'], stage2text)
            b, _, rb = run(['-threads', '4', *opts, ntt, '1000', '100000'], stage2text)
            check(numeric(ra) == numeric(rb), ('stage2 saves', method, ntt))
            check(re.findall(r'Factor found in step \d: \d+', a.stdout) ==
                  re.findall(r'Factor found in step \d: \d+', b.stdout), ('stage2 factors', method, ntt))

    # Multiple curves on one candidate, with distinct random seeds.
    _, _, rows = run(['-threads', '4', '-c', '12', '1000', '0'], str(2**521 - 1) + '\n')
    check(len(rows) == 12, 'curve count')
    check(len({r['SIGMA'] for r in rows}) == 12, 'distinct random curves')
    for row in rows:
        _, _, replay = run(['-sigma', row.get('PARAM', '0') + ':' + row['SIGMA'], '1000', '0'], row['N'] + '\n')
        check(numeric([row]) == numeric(replay), 'random curve serial replay')

    # Automatic batch parametrization must be written explicitly in saves.
    _, _, rows = run(['-threads', '4', '-c', '4', '-modmuln', '1000', '0'],
                    str((2**127 - 1) * (2**521 - 1)) + '\n')
    check(len(rows) == 4 and all(r.get('PARAM') == '1' for r in rows), 'automatic curve metadata')
    for row in rows:
        _, _, replay = run(['-sigma', row['PARAM'] + ':' + row['SIGMA'], '1000', '0'], row['N'] + '\n')
        check(numeric([row]) == numeric(replay), 'automatic curve serial replay')

    # GWNUM handles and sieves must be independent across concurrent curves.
    for iteration in range(3):
        opts = ['-force-gwnum', '-sigma', '0:11']
        a, _, ra = run([*opts, '1000', '0'], text)
        b, _, rb = run(['-threads', '8', *opts, '1000', '0'], text)
        check(numeric(ra) == numeric(rb), ('GWNUM', iteration))
        check("Using gwnum fft's" in b.stdout, 'GWNUM exercised')

    # Deduplicate factors across already-running curves of the same candidate.
    candidate = 101 * (2**127 - 1)
    _, _, seed = run(['-pm1', '-x0', '3', '2', '0'], str(candidate) + '\n')
    repeated = root / 'repeated.save'
    _, p, _ = run(['-pm1', '-x0', '3', '2', '0'], str(candidate) + '\n')
    repeated.write_text(p.read_text() * 16)
    a, _, _ = run(['-threads', '4', '-resume', str(repeated), '100', '0'])
    check(a.stdout.count('Factor found in step 1: 101') == 1, 'duplicate resume factors')
    check(a.returncode == 14, 'factor exit status')

    # Brent logging and save labels retain input order.
    log = root / 'BrentFactors.log'
    lines = [f'2 {12700 * i}-L {candidate}' for i in range(1, 9)]
    a, _, _ = run(['-threads', '4', '-pm1', '-x0', '3', '100', '0'], '\n'.join(lines) + '\n')
    check([line.split(';')[0] for line in log.read_text().splitlines()] ==
          [f'2 {12700 * i}-L 101' for i in range(1, 9)], 'Brent log order')
    check('Sigma: Not Applicable' in log.read_text(), 'P-1 sigma wording')

    # -one stops a repeated candidate even when its cofactor is composite.
    composite = 101 * 1000003 * (2**127 - 1)
    _, p, _ = run(['-pm1', '-x0', '3', '2', '0'], str(composite) + '\n')
    repeated.write_text(p.read_text() * 20)
    a, _, rows = run(['-threads', '4', '-one', '-resume', str(repeated), '100', '0'])
    check(a.stdout.count('Factor found in step 1: 101') == 1 and len(rows) == 1, '-one resume')
    check(int(rows[0]['N']) == composite // 101, 'saved composite cofactor')
    a, _, rows = run(['-threads', '4', '-one', '-c', '0', '-sigma', '0:6', '100', '0'],
                     str(candidate) + '\n')
    check(a.stdout.count('Factor found in step 1: 101') == 1, 'unbounded curves stop on factor')

    # Rational initialization factors are collected and saved in the same order.
    a, _, _ = run(['-threads', '4', '-pm1', '-x0', '1/101', '100', '0'],
                  (str(candidate) + '\n') * 4)
    check(a.stdout.count('Factor found in step 1: 101') == 4, 'rational initialization')

    # An unsupported GPU checkpoint after valid records errors but drains
    # previously dispatched records without changing their order.
    _, p, original = run(['-pm1', '-x0', '3', '100', '0'], text)
    broken = root / 'broken.save'
    broken.write_text(p.read_text() + 'METHOD=P-1; B1=100; N=17; X=3; GPU_B1=1000; GPU_BITS=1; GPU_BASE=3;\n')
    out = root / 'error.save'
    r = subprocess.run([str(exe), '-threads', '4', '-resume', str(broken),
                        '-save', str(out), '1000', '0'], cwd=root, text=True,
                       capture_output=True, timeout=60)
    check(r.returncode == 1, 'unsupported resume error')
    check([row['N'] for row in records(out)] == [row['N'] for row in original], 'error drain order')

    # A library error must not emit an invalid stage-1 residue or send its
    # diagnostic to stdout in quiet mode.
    out = root / 'invalid.save'
    r = subprocess.run([str(exe), '-q', '-threads', '4', '-param', '1', '-mpzmod',
                        '-save', str(out), '100', '0'], input=text, cwd=root,
                       text=True, capture_output=True, timeout=60)
    check(r.returncode == 1 and 'Error' in r.stderr, 'worker error handling')
    check(not records(out), 'failed calculations are not saved')

    # -savea appends in order and -threads 1 keeps serial behavior.
    _, p, original = run(['-threads', '1', '-pm1', '-x0', '3', '1000', '0'], text)
    a, _, _ = run(['-threads', '4', '-pm1', '-x0', '3', '-savea', str(p), '1000', '0'], text, save=False)
    check(numeric(records(p)) == numeric(original) * 2, 'append order')
    for bad in ('0', '-1', '1.5', 'abc', '1025', '4junk'):
        r = subprocess.run([str(exe), '-threads', bad, '100', '0'], cwd=root, capture_output=True, text=True)
        check(r.returncode == 1 and '-threads requires' in r.stderr, ('bad thread count', bad))

print(f'PASS {checks} CPU job checks', flush=True)

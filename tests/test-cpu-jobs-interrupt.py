"""Validate ordered partial saves using the test-only signal wrapper."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('ecm', type=Path)
p.add_argument('interrupter', type=Path)
a = p.parse_args()
exe, interrupter = a.ecm.resolve(), a.interrupter.resolve()

def records(path):
    return [dict(re.findall(r'(\w+)=([^;]*);', row)) for row in path.read_text().splitlines()]

with tempfile.TemporaryDirectory(prefix='cpu-interrupt-') as directory:
    root = Path(directory)
    n = 2**1279 - 1
    # Distinct comments allow checking order even when the same N is repeated.
    for name, opts in [('pm1', ['-pm1', '-x0', '3']), ('pp1', ['-pp1', '-x0', '3']),
                       ('ecm', ['-sigma', '0:11', '-force-no-gwnum']),
                       ('gwnum', ['-sigma', '0:11', '-force-gwnum'])]:
        source, partial = root / (name + '.input'), root / (name + '.partial')
        source.write_text('\n'.join(str(2**power - 1) for power in
                                    (1279, 2203, 607, 521, 2281, 127, 2203, 1279, 607)) + '\n')
        run = subprocess.run([str(interrupter), '-threads', '4', *opts, '-inp', str(source),
                              '-save', str(partial), '3000000', '0'], cwd=root,
                             text=True, capture_output=True, timeout=120)
        assert run.returncode == 143, (name, run.returncode, run.stdout, run.stderr)
        rows = records(partial)
        assert 4 <= len(rows) <= 9, (name, len(rows), run.stdout, run.stderr)
        assert [r['N'] for r in rows] == source.read_text().splitlines()[:len(rows)], (name, 'order')
        assert any(float(r['B1']) < 3000000 for r in rows), (name, 'no partial records')
        # Completed and unstarted residues can be replayed from their bound.
        # Traditional CPU checkpoints may already contain powers of small
        # primes up to the original target, so intermediate B1 is a progress
        # marker, not necessarily the exact LCM exponent for that bound.
        # Exercise those records via both serial and parallel continuation.
        for i, row in enumerate(rows):
            if 1 < float(row['B1']) < 3000000:
                continue
            checkfile = root / f'{name}-{i}.check'
            run = subprocess.run([str(exe), *opts, '-save', str(checkfile), row['B1'], '0'],
                                 input=row['N'] + '\n', cwd=root, text=True,
                                 capture_output=True, timeout=120)
            assert run.returncode == 0, (name, i, run.stdout, run.stderr)
            assert records(checkfile)[0]['X'] == row['X'], (name, i, 'partial residue')
        for threads in (1, 4):
            out = root / f'{name}-{threads}.continued'
            run = subprocess.run([str(exe), '-threads', str(threads), '-resume', str(partial),
                                  '-save', str(out), '3000000', '0'], cwd=root,
                                 capture_output=True, text=True, timeout=240)
            assert run.returncode == 0, (name, threads, run.stdout, run.stderr)
        key = lambda path: [(r['N'], r['B1'], r['X'], r.get('SIGMA')) for r in records(path)]
        assert key(root / f'{name}-1.continued') == key(root / f'{name}-4.continued'), name
        print(f'PASS {name}: {len(rows)} ordered partial saves and serial/parallel continuation', flush=True)

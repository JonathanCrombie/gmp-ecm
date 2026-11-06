"""Exercise actual adaptive batch boundaries, ordering, residues and fixed override.

python tests/test-gpu-adaptive.py build-msvc/bin/ecm.exe [interrupter.exe]
All input/save/checkpoint/log files are temporary. No timing assumptions.
"""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

def lucas(a, e, n):
    x, y = 2, a
    for bit in bin(e)[2:]:
        cross = (x*y-a) % n
        if bit == '1':
            x, y = cross, (y*y-2) % n
        else:
            x, y = (x*x-2) % n, cross
    return x

def records(path):
    return [dict(re.findall(r'(\w+)=([^;]*);', line)) for line in path.read_text().splitlines()]

def semantic(rows):
    return [{k: r[k] for k in ('METHOD', 'B1', 'N', 'X', 'X0', 'COMMENT')} for r in rows]

source = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='ecm-adaptive-') as directory:
    root = Path(directory)
    exe = root / source.name
    shutil.copy2(source, exe)
    for dll in source.parent.glob('*.dll'):
        shutil.copy2(dll, root / dll.name)

    def run(method, options, text='', bound=100):
        p = subprocess.run([str(exe), '-'+method, '-gpu', '-color', 'never',
                            '-x0', '3', *map(str, options), str(bound), '0'],
                           input=text, text=True, capture_output=True, cwd=root, timeout=180)
        assert p.returncode == 0, (p.returncode, p.stdout[-3000:], p.stderr)
        return p.stdout

    def capacity(output):
        return int(re.search(r'auto batch: capacity (\d+)', output)[1])

    # Known Mersenne primes; both fixtures survive the small stage-1 bounds.
    small = ((1 << 127)-1)*((1 << 61)-1)
    large = ((1 << 521)-1)*((1 << 607)-1)
    exponent = math.lcm(*range(1, 101))
    for method in ('pm1', 'pp1'):
        p = subprocess.run([str(exe), '-'+method, '-gpu', '-color', 'never', '100', '0'],
                           input=str((1 << 13000)*((1 << 127)-1)), text=True,
                           capture_output=True, cwd=root, timeout=30)
        assert p.returncode in (2,6,8,10,14) and 'Factor found in step 1: 2' in p.stdout, (p.stdout, p.stderr)
        expected = {n: pow(3, exponent, n) if method=='pm1' else lucas(3, exponent, n)
                    for n in (small, large)}
        assert all(math.gcd(x-(1 if method=='pm1' else 2), n)==1 for n, x in expected.items())
        low = capacity(run(method, [], str(small)))
        high = capacity(run(method, [], str(large)))
        # Force a width transition after more small inputs have been admitted
        # than fit in a wide batch, plus full batches and a non-block-sized tail.
        middle = min(low-1, high+3)
        ns = [small]*middle + [large]*(high+5) + [small]*(low+7)
        text = '\n'.join(f'2 {500+i}-M {n}' for i, n in enumerate(ns))
        (root/'inputs.txt').write_text(text)
        output = run(method, ['-inp', 'inputs.txt', '-save', 'auto.save', '-chkpnt', 'auto.chk'])
        rows = records(root/'auto.save')
        assert len(rows)==len(ns), (len(rows), len(ns))
        assert [r['COMMENT'] for r in rows]==[f'LABEL=2 {500+i}-M' for i in range(len(ns))]
        assert [int(r['N']) for r in rows]==ns
        assert all(int(r['X'], 0)==expected[n] and not any(k.startswith('GPU_') for k in r)
                   for r, n in zip(rows, ns))
        batches = list(map(int, re.findall(r'GPU P[+-]1 batch: (\d+) inputs', output)))
        assert sum(batches)==len(ns) and len(batches)>=3, batches
        fixed = run(method, ['-gpubatch', 317, '-inp', 'inputs.txt', '-save', 'fixed.save'])
        assert 'GPU auto batch' not in fixed
        assert semantic(rows)==semantic(records(root/'fixed.save'))
        # Continuation batches must retain residue and label order too.
        run(method, ['-resume', 'auto.save', '-save', 'continued-auto.save'], bound=200)
        run(method, ['-gpubatch', 317, '-resume', 'fixed.save', '-save', 'continued-fixed.save'], bound=200)
        assert semantic(records(root/'continued-auto.save'))==semantic(records(root/'continued-fixed.save'))
        assert len(records(root/'continued-auto.save'))==len(ns)
        for filename in ('auto.save', 'auto.chk', 'fixed.save', 'continued-auto.save', 'continued-fixed.save'):
            assert '\x1b' not in (root/filename).read_text()
            (root/filename).unlink()
        print(f'PASS {method}: {len(ns)} ordered inputs, batch sizes {batches}; fixed/auto residues and continuation agree')
        if len(sys.argv)>2:
            interrupted = root/'interrupter.exe'
            shutil.copy2(Path(sys.argv[2]).resolve(), interrupted)
            count = low*2+3
            stop_exp = math.lcm(*range(1, 1001))
            final = pow(3, stop_exp, small) if method=='pm1' else lucas(3, stop_exp, small)
            assert math.gcd(final-(1 if method=='pm1' else 2), small)==1
            text = '\n'.join(f'2 {500+i}-M {small}' for i in range(count))
            for batch in (1, 2):
                p = subprocess.run([str(interrupted), '-'+method, '-gpu', '-x0', '3',
                                    '-save', 'stopped.save', '-chkpnt', 'stopped.chk', '1000', '0'],
                                   input=text, text=True, capture_output=True, cwd=root,
                                   env=dict(os.environ, ECM_TEST_STOP_BATCH=str(batch)), timeout=180)
                assert p.returncode==143, (p.returncode, p.stdout[-3000:], p.stderr)
                stopped = records(root/'stopped.save')
                assert stopped and len(stopped)<=count
                read_count = int(re.search(r'stopped after reading (\d+) inputs', p.stdout+p.stderr)[1])
                assert len(stopped)==read_count, ('lost buffered inputs', len(stopped), read_count)
                assert [r['COMMENT'] for r in stopped]==[f'LABEL=2 {500+i}-M' for i in range(len(stopped))]
                checkpoint_rows = records(root/'stopped.chk')
                assert checkpoint_rows
                assert [r['COMMENT'] for r in checkpoint_rows]==[r['COMMENT'] for r in stopped[-len(checkpoint_rows):]]
                partials = [r for r in stopped if 'GPU_BITS' in r]
                assert partials and any(0<int(r['GPU_BITS'])<stop_exp.bit_length() for r in partials)
                for r in partials:
                    prefix = stop_exp >> (stop_exp.bit_length()-int(r['GPU_BITS']))
                    value = pow(3, prefix, small) if method=='pm1' else lucas(3, prefix, small)
                    assert int(r['X'], 0)==value
                    if method=='pp1':
                        assert int(r['GPU_NEXT'], 0)==lucas(3, prefix+1, small)
                run(method, ['-resume', 'stopped.save', '-save', 'recovered.save'], bound=1000)
                recovered = records(root/'recovered.save')
                assert len(recovered)==len(stopped)
                assert [r['COMMENT'] for r in recovered]==[r['COMMENT'] for r in stopped]
                assert all(int(r['X'], 0)==final for r in recovered)
                run(method, ['-resume', 'stopped.chk', '-save', 'checkpoint-recovered.save'], bound=1000)
                checkpoint_recovered = records(root/'checkpoint-recovered.save')
                assert [r['COMMENT'] for r in checkpoint_recovered]==[r['COMMENT'] for r in checkpoint_rows]
                assert all(int(r['X'], 0)==final for r in checkpoint_recovered)
                for filename in ('stopped.save', 'stopped.chk', 'recovered.save', 'checkpoint-recovered.save'):
                    (root/filename).unlink()
            print(f'PASS {method}: adaptive CLI interruption in batches 1 and 2, coherent checkpoints and recovery')
print('PASS adaptive GPU integration')

"""GPU P+1 arithmetic, save/resume, CPU handoff and CLI integration checks.

python tests/test-gpu-pp1.py build-msvc/bin/ecm.exe [--batch-data ../Test_Batches]
Uses companion matrices as an independent oracle. All saves/logs are temporary.
"""
import argparse
from functools import lru_cache
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
parser = argparse.ArgumentParser()
parser.add_argument('ecm', type=Path)
parser.add_argument('--batch-data', type=Path)
parser.add_argument('--interrupter', type=Path, help='CLI built with gpu-pp1-interrupt.c')
args = parser.parse_args()
rng = random.Random(20261010)
checks = 0


def check(ok, message):
    global checks
    assert ok, message
    checks += 1


@lru_cache(None)
def exponent(bound):
    return math.lcm(*range(1, bound + 1))


def matrix_mul(a, b, n):
    return ((a[0]*b[0]+a[1]*b[2]) % n, (a[0]*b[1]+a[1]*b[3]) % n,
            (a[2]*b[0]+a[3]*b[2]) % n, (a[2]*b[1]+a[3]*b[3]) % n)


def lucas_pair(a, e, n):
    r, m = (1, 0, 0, 1), (a % n, n-1, 1, 0)
    while e:
        if e & 1:
            r = matrix_mul(r, m, n)
        m = matrix_mul(m, m, n)
        e >>= 1
    return (r[0]+r[3]) % n, (a*r[0]+r[1]-r[2]) % n


def numbers(widths, bound=100, base=3):
    result = []
    for bits in widths:
        while True:
            n = (1 << (bits-1)) | rng.getrandbits(bits-1) | 1
            if math.gcd(lucas_pair(base, exponent(bound), n)[0]-2, n) == 1:
                result.append(n)
                break
    return result


def records(path):
    return [dict(re.findall(r'(\w+)=([^;]*);', line))
            for line in path.read_text().splitlines() if line and not line.startswith('#')]


def partial(n, a, bound, bits, start=1, label=''):
    e = exponent(bound) // exponent(start)
    prefix = e >> (e.bit_length()-bits)
    x, y = lucas_pair(a, prefix, n)
    mod = 4294967291
    checksum = start * (n % mod) * (x % mod)
    for field in (a % mod, bound & 0xffffffff, bound >> 32,
                  bits & 0xffffffff, bits >> 32, y % mod):
        checksum = checksum*65599 + field
    text = (f'METHOD=P+1; B1={start}; N={n}; X={hex(x)}; GPU_B1={bound}; '
            f'GPU_BITS={bits}; GPU_BASE={hex(a)}; GPU_NEXT={hex(y)}; '
            f'CHECKSUM={checksum % mod}; X0=0x3;')
    if label:
        text += f' COMMENT=LABEL={label};'
    return text + '\n'


with tempfile.TemporaryDirectory(prefix='ecm-gpu-pp1-') as directory:
    root = Path(directory)
    source = args.ecm.resolve()
    exe = root / source.name
    shutil.copy2(source, exe)
    for dll in source.parent.glob('*.dll'):
        shutil.copy2(dll, root / dll.name)
    env = os.environ.copy()
    if os.name == 'nt':
        env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')

    def run(options, inputs='', error=False):
        p = subprocess.run([str(exe), *map(str, options)], input=inputs,
                           text=True, capture_output=True, cwd=root, env=env, timeout=180)
        check(p.returncode == 1 if error else p.returncode in (0, 2, 6, 8, 10, 14),
              (options, p.returncode, p.stdout, p.stderr))
        return p.stdout + p.stderr

    def check_residues(file, ns, bound, bases=None, powers=None):
        rows = records(root / file)
        check(len(rows) == len(ns), (file, len(rows), len(ns)))
        for i, (row, n) in enumerate(zip(rows, ns)):
            a = 3 if bases is None else bases[i]
            e = exponent(bound) if powers is None else powers[i]
            check(int(row['N']) == n, ('input/save order', file))
            check(int(row['X'], 0) == lucas_pair(a, e, n)[0], ('Lucas residue', file, n.bit_length()))
            check(row['METHOD'] == 'P+1' and float(row['B1']) == bound and
                  not any(k.startswith('GPU_') for k in row), ('completed save format', file))
        return rows

    widths = [31, 32, 33, 127, 128, 255, 256, 257, 511, 512, 513, 767, 768,
              769, 1023, 1024, 1025, 1536, 1537, 2048, 2049, 3072, 3073,
              4096, 4097, 6144, 6145, 8192, 8193, 12288]
    ns = numbers(widths)
    run(['-pp1', '-cgbn', '-gpubatch', 3, '-x0', 3, '-save', 'widths.save', 100, 0],
        '\n'.join(map(str, ns)))
    check_residues('widths.save', ns, 100)
    ns = numbers([64, 256, 512])
    out = run(['-pp1', '-gpu', '-x0', 3, '-save', 'mixed.save', 100, 0], '\n'.join(map(str, ns)))
    check('CGBN<512,4>' in out and 'GPU P+1' in out, out)
    check_residues('mixed.save', ns, 100)
    ns = numbers([511]*65)
    out = run(['-pp1', '-gpu', '-gpubatch', 33, '-x0', 3, '-save', 'tail.save', 100, 0], '\n'.join(map(str, ns)))
    check('33 numbers' in out and '32 numbers' in out, out)
    check_residues('tail.save', ns, 100)

    # CPU/GPU continuation without the N^2-1 preload, including source order.
    ns = numbers([127, 255, 512, 769], 1000)
    text = '\n'.join(map(str, ns))
    run(['-pp1', '-gpu', '-x0', 3, '-save', 'b100.save', 100, 0], text)
    for mode in ([], ['-gpu']):
        name = 'gpu1000.save' if mode else 'cpu1000.save'
        run(['-pp1', *mode, '-resume', 'b100.save', '-save', name, 1000, 0])
        check_residues(name, ns, 1000)
    run(['-pp1', '-x0', 3, '-save', 'cpu100.save', 100, 0], text)
    run(['-pp1', '-gpu', '-resume', 'cpu100.save', '-save', 'cpu-gpu.save', 1000, 0])
    check_residues('cpu-gpu.save', ns, 1000)

    # Independently constructed partial states: zero bits, mid-launch states,
    # and a fully consumed exponent still encoded as a partial record.
    e = exponent(1000)
    checkpoint = ''.join(partial(n, 3, 1000, bits, label=f'2 {127+i}-M')
                         for i, (n, bits) in enumerate(zip(ns, [0, 17, 17, e.bit_length()])))
    (root / 'partial.save').write_text(checkpoint)
    run(['-pp1', '-gpu', '-gpubatch', 2, '-resume', 'partial.save',
         '-save', 'finished.save', '-chkpnt', 'snapshot.save', 1000, 0])
    rows = check_residues('finished.save', ns, 1000)
    check([r['COMMENT'] for r in rows] == [f'LABEL=2 {127+i}-M' for i in range(4)], 'resume labels')
    run(['-pp1', '-resume', 'partial.save', 1000, 0], error=True)
    run(['-pm1', '-gpu', '-resume', 'partial.save', 1000, 0], error=True)
    run(['-pp1', '-gpu', '-resume', 'partial.save', 1001, 0], error=True)
    for bad in (checkpoint.replace('GPU_BITS=17;', 'GPU_BITS=18;', 1),
                re.sub(r'GPU_NEXT=[^;]+;', '', checkpoint, count=1),
                re.sub(r'GPU_NEXT=[^;]+;', 'GPU_NEXT=0x0;', checkpoint, count=1),
                checkpoint.replace('METHOD=P+1;', 'METHOD=P-1;', 1),
                checkpoint.replace('GPU_BITS=0;', 'GPU_BITS=99999999999999999;', 1)):
        (root / 'bad.save').write_text(bad)
        run(['-pp1', '-gpu', '-resume', 'bad.save', 1000, 0], error=True)

    # CPU residues and independent oracle agree when N^2-1 preloads apply.
    # Compare printed pre-factor residues so whole-N or prime factors cannot
    # hide arithmetic by removing an input's save record.
    preloads = [(1 << 61)-1, ((1 << 31)-1)*((1 << 61)-1), 1000003*1000033,
                ((1 << 43)+19)*((1 << 45)+27)]
    for n in preloads:
        prime = n == (1 << 61)-1
        for go in (1, 7):
            power = exponent(10000)*go*(1 if prime else n*n-1)
            expected = lucas_pair(3, power, n)[0]
            for mode in ([], ['-gpu']):
                out = run(['-pp1', *mode, '-v', '-v', '-x0', 3, '-go', go, 10000, 0], str(n))
                check(re.findall(r'^x=(\d+)$', out, re.M) == [str(expected)], ('preload', n, go, out))
        # Extending a completed save applies N^2-1 again, like CPU pp1_stage1.
        x = lucas_pair(3, exponent(100), n)[0]
        (root / 'preload.save').write_text(f'METHOD=P+1; B1=100; N={n}; X={hex(x)}; X0=0x3;\n')
        power = (exponent(10000)//exponent(100))*(1 if prime else n*n-1)
        for mode in ([], ['-gpu']):
            out = run(['-pp1', *mode, '-v', '-v', '-resume', 'preload.save', 10000, 0])
            check(re.findall(r'^x=(\d+)$', out, re.M) == [str(lucas_pair(x, power, n)[0])], ('preload continuation', out))

    # Here applying the preload twice changes the answer, so a partial resume
    # cannot pass accidentally because the residue has already saturated to 2.
    n = preloads[-1]
    for start in (1, 100):
        x = 3 if start == 1 else lucas_pair(3, exponent(start), n)[0]
        prepared = lucas_pair(x, n*n-1, n)[0]
        e = exponent(10000)//exponent(start)
        expected = lucas_pair(prepared, e, n)[0]
        check(expected != lucas_pair(prepared, e*(n*n-1), n)[0], 'nontrivial preload fixture')
        (root / 'preloaded-partial.save').write_text(partial(n, prepared, 10000, 17, start))
        out = run(['-pp1', '-gpu', '-v', '-v', '-resume', 'preloaded-partial.save', 10000, 0])
        check(re.findall(r'^x=(\d+)$', out, re.M) == [str(expected)], 'do not repeat preload on partial resume')

    # Rational and N-dependent preloads, and B1=1 identity.
    n = (1 << 127)-1
    for bound, x0, go in ((1, '3', '1'), (100, '2/7', '1'), (100, '3', 'N+1')):
        name = f'rational-{bound}-{go}.save'
        out = run(['-pp1', '-gpu', '-v', '-v', '-x0', x0, '-go', go, '-save', name, bound, 0], str(n))
        a = 2*pow(7, -1, n) % n if x0 == '2/7' else 3
        power = exponent(bound)*(n+1 if go == 'N+1' else 1)
        expected = lucas_pair(a, power, n)[0]
        check(re.findall(r'^x=(\d+)$', out, re.M) == [str(expected)], ('rational/go', out))
        if math.gcd(expected-2, n) == 1:
            check_residues(name, [n], bound, [a], [power])
        else:
            check(f'Factor found in step 1: {n}' in out, out)

    # A zero completed residue must not silently select a random seed.
    (root / 'zero.save').write_text(f'METHOD=P+1; B1=100; N={n}; X=0x0; X0=0x3;\n')
    for mode in ([], ['-gpu']):
        name = 'zero-gpu.save' if mode else 'zero-cpu.save'
        run(['-pp1', *mode, '-resume', 'zero.save', '-save', name, 100, 0])
        check_residues(name, [n], 100, [0], [1])

    # CPU stage 2 (NTT and non-NTT), pipelined batches, and Brent logging order.
    n = 1000039*((1 << 127)-1)
    for ntt in ('-ntt', '-no-ntt'):
        for mode in ([], ['-gpu']):
            out = run(['-pp1', *mode, ntt, '-x0', 4, 23, 1087], str(n))
            check('Factor found in step 2: 1000039' in out, out)
    out = run(['-pp1', '-gpu', '-gpubatch', 1, '-x0', 4, 23, 1087],
              '\n'.join(f'2 {127+i}-M {n}' for i in range(3)))
    check(out.count('Factor found in step 2: 1000039') == 3, 'CPU stage 2 while next GPU batch runs')
    entries = (root / 'BrentFactors.log').read_text().splitlines()[-3:]
    check([s.split(';')[0] for s in entries] == [f'2 {127+i}-M 1000039' for i in range(3)], 'factor log ordering')
    check(all('X0: 0x4;' in s and 'Sigma:' not in s for s in entries), 'P+1 starting base')
    out = run(['-pp1', '-gpu', '-x0', 4, 11, 0], str(43*((1 << 127)-1)))
    check('Factor found in step 1: 43' in out, out)
    out = run(['-pp1', '-gpu', '-x0', '2/7', 100, 0], str(7*((1 << 127)-1)))
    check('Factor found in step 1: 7' in out, out)
    out = run(['-pp1', '-gpu', '-chkpnt', 'even.save', 3000000000, 0], '6\n10\n')
    check(out.count('Factor found in step 1: 2') == 2, 'even inputs bypass huge exponent')
    run(['-pp1', '-gpu', 100, 0], '3\n5\n7\n')

    # Replay the actual randomly selected starting values.
    ns = [(1 << 127)-1, (1 << 521)-1]
    run(['-pp1', '-gpu', '-save', 'random.save', 100, 0], '\n'.join(map(str, ns)))
    rows = records(root / 'random.save')
    check_residues('random.save', ns, 100, [int(r['X0'], 0) for r in rows])
    for options in (['-gpubatch', 0], ['-gpucurves', 32], ['-c', 2], ['-threads', 2]):
        run(['-pp1', '-gpu', *options, 100, 0], error=True)
    run(['-pp1', '-gpu', '-resume', 'b100.save', '-savea', './b100.save', 1000, 0], error=True)
    run(['-pp1', '-gpu', 100, 0], str((1 << 12288)+1), error=True)

    if args.interrupter:
        interrupter = root / 'interrupt.exe'
        shutil.copy2(args.interrupter.resolve(), interrupter)
        n = (1 << 521)-1
        expected = lucas_pair(3, exponent(1000), n)[0]
        check(math.gcd(expected-2, n) == 1, 'signal fixture survives stage 1')
        count = 111
        text = '\n'.join(f'2 {500+i}-M {n}' for i in range(count))
        for batch in (1, 2):
            interrupted = f'interrupted-{batch}.save'
            snapshot = f'interrupted-{batch}.checkpoint'
            signal_env = dict(env, ECM_TEST_STOP_BATCH=str(batch))
            p = subprocess.run([str(interrupter), '-pp1', '-gpu', '-x0', '3',
                                '-gpubatch', '37', '-save', interrupted,
                                '-chkpnt', snapshot, '1000', '0'],
                               input=text, text=True, capture_output=True, cwd=root,
                               env=signal_env, timeout=180)
            check(p.returncode == 143, ('signal exit', p.returncode, p.stdout, p.stderr))
            rows = records(root / interrupted)
            check(37 <= len(rows) <= count, ('buffered records', len(rows)))
            check([r['COMMENT'] for r in rows] == [f'LABEL=2 {500+i}-M' for i in range(len(rows))], 'signal save order')
            partials = [r for r in rows if 'GPU_BITS' in r]
            check(any(0 < int(r['GPU_BITS']) < exponent(1000).bit_length() for r in partials), 'actually interrupted CUDA work')
            for r in partials:
                prefix = exponent(1000) >> (exponent(1000).bit_length()-int(r['GPU_BITS']))
                check((int(r['X'], 0), int(r['GPU_NEXT'], 0)) == lucas_pair(3, prefix, n), 'signal snapshot Lucas pair')
            finished = f'signal-complete-{batch}.save'
            run(['-pp1', '-gpu', '-gpubatch', 19, '-resume', interrupted, '-save', finished, 1000, 0])
            check_residues(finished, [n]*len(rows), 1000)
            # Checkpoint can omit already reported jobs; everything it keeps
            # must be coherent and independently resumable.
            snaprows = records(root / snapshot)
            snapfinished = f'signal-checkpoint-complete-{batch}.save'
            run(['-pp1', '-gpu', '-gpubatch', 23, '-resume', snapshot, '-save', snapfinished, 1000, 0])
            check_residues(snapfinished, [n]*len(snaprows), 1000)

    if args.batch_data:
        for i, path in enumerate(sorted(args.batch_data.resolve().glob('*.txt'))):
            lines = []
            with path.open() as f:
                for line in f:
                    n = int(line.split()[-1])
                    if math.gcd(lucas_pair(3, exponent(100), n)[0]-2, n) == 1:
                        lines.append(line.strip())
                    if len(lines) == 32:
                        break
            name = f'data-{i}.save'
            run(['-pp1', '-gpu', '-gpubatch', 13, '-x0', 3, '-save', name, 100, 0], '\n'.join(lines))
            rows = check_residues(name, [int(s.split()[-1]) for s in lines], 100)
            check([r['COMMENT'][6:] for r in rows] == [' '.join(s.split()[:-1]) for s in lines], 'batch-data labels')

print(f'PASS: {checks} GPU P+1 arithmetic, preload, batching, stage-2, save/resume and input checks')

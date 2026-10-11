"""Original P-/+1 fractions in display, logs, saves and CPU/GPU checkpoints.

python tests/test-x0-rational.py path/to/ecm [--gpu] [--interrupter path/to/cli]
All input, output and executables live in a temporary directory.
"""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('ecm', type=Path)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--interrupter', type=Path)
args = parser.parse_args()
checks = 0


def check(ok, message):
    global checks
    assert ok, message
    checks += 1


def residue(method, base, power, n):
    if method == '-pm1':
        return pow(base, power, n)
    # Independent companion-matrix oracle for V_power(base).
    def mul(a, b):
        return ((a[0]*b[0]+a[1]*b[2]) % n, (a[0]*b[1]+a[1]*b[3]) % n,
                (a[2]*b[0]+a[3]*b[2]) % n, (a[2]*b[1]+a[3]*b[3]) % n)
    r, m = (1, 0, 0, 1), (base, n-1, 1, 0)
    while power:
        if power & 1:
            r = mul(r, m)
        m = mul(m, m)
        power >>= 1
    return (r[0]+r[3]) % n


def records(path):
    return [dict(re.findall(r'(\w+)=([^;]*);', line))
            for line in path.read_text().splitlines() if line]


with tempfile.TemporaryDirectory(prefix='ecm-rational-x0-') as directory:
    root = Path(directory)
    exe = root / args.ecm.name
    shutil.copy2(args.ecm.resolve(), exe)
    for dll in args.ecm.resolve().parent.glob('*.dll'):
        shutil.copy2(dll, root / dll.name)
    env = os.environ.copy()
    if os.name == 'nt':
        env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    modes = [[], ['-threads', 4]] + ([['-gpu', '-gpubatch', 2]] if args.gpu else [])
    log = root / 'BrentFactors.log'

    def run(options, text='', error=False):
        if '-save' in options:
            saved = Path(options[options.index('-save')+1])
            check(saved.parent == root, 'test output stays in its temporary directory')
            saved.unlink(missing_ok=True)
        p = subprocess.run([str(exe), *map(str, options)], cwd=root, env=env,
                           input=text, capture_output=True, text=True, timeout=120)
        check(p.returncode == 1 if error else p.returncode in (0, 2, 6, 8, 10, 14),
              (options, p.returncode, p.stdout, p.stderr))
        return p.stdout + p.stderr

    prime = (1 << 127)-1
    seeds = ['2/7', '6/5', '4/14', '-2/7']
    for method in ('-pm1', '-pp1'):
        offset = 1 if method == '-pm1' else 2
        for index, mode in enumerate(modes):
            mixed = []
            for seed_index, seed in enumerate(seeds):
                num, den = map(int, seed.split('/'))
                base = num*pow(den, -1, prime) % prime
                power = math.lcm(*range(1, 11))
                expected = residue(method, base, power, prime)
                check(math.gcd(expected-offset, prime) == 1, ('fixture', method, seed))
                saved = root / f'{method}-{index}-{seed_index}.save'
                out = run([method, *mode, '-x0', seed, '-save', saved, 10, 0], str(prime))
                check(f'x0={seed}' in out, ('fresh console', out))
                rows = records(saved)
                check(len(rows) == 1 and rows[0]['X0'] == seed, ('fresh save', rows))
                check(int(rows[0]['X'], 0) == expected, ('fresh arithmetic', rows))
                mixed.extend(saved.read_text().splitlines())
                # CPU checkpoints use a different writer from ordinary saves.
                if '-gpu' not in mode:
                    checkpoint = root / f'{method}-{index}-{seed_index}.checkpoint'
                    run([method, *mode, '-x0', seed, '-chkpnt', checkpoint, 10, 0], str(prime))
                    paths = list(root.glob(checkpoint.name + '*'))
                    check(len(paths) == 1, ('checkpoint files', paths))
                    check(records(paths[0])[0]['X0'] == seed, ('checkpoint seed', paths))
                    again = root / 'checkpoint-resumed.save'
                    out = run([method, '-resume', paths[0], '-save', again, 10, 0])
                    check(records(again)[0]['X0'] == seed, 'checkpoint resumed seed')
                # Original fraction must survive a preload as well.
                preloaded = root / 'preloaded.save'
                run([method, *mode, '-x0', seed, '-go', 7, '-save', preloaded, 10, 0], str(prime))
                rows = records(preloaded)
                check(rows[0]['X0'] == seed and int(rows[0]['X'], 0) ==
                      residue(method, base, power*7, prime), ('preload', rows))
                out = run([method, *mode, '-x0', seed, 100, 0], f'2 12700-M {101*prime}')
                check('Factor found in step 1: 101' in out and
                      f'X0: {seed};' in log.read_text().splitlines()[-1], ('stage-1 log', out))
                interval = root / 'interval.save'
                out = run([method, *mode, '-x0', seed, '-save', interval, '10-100', 0], str(prime))
                row = records(interval)[0]
                check(row['X0'] == seed and f'x0={seed}' in out and int(row['X'], 0) ==
                      residue(method, base, math.lcm(*range(1, 101))//power, prime), ('B1 interval', row, out))

            # Mixed fractions (including an unreduced one), old hex and absent
            # metadata exercise per-job state and resetting optional fields.
            mixed.append(re.sub(r'X0=[^;]+;', 'X0=0x36238f4d;', mixed[0]))
            mixed.append(re.sub(r'X0=[^;]+;', '', mixed[0]))
            source = root / f'{method}-{index}-mixed.save'
            source.write_text('\n'.join(mixed)+'\n')
            for dest_index, dest in enumerate(modes):
                finished = root / f'{method}-{index}-{dest_index}-resumed.save'
                out = run([method, *dest, '-resume', source, '-save', finished, 100, 0])
                rows = records(finished)
                check([r['X0'] for r in rows] == seeds+['0x36238f4d', '0x0'], ('resume seeds', rows))
                for i, seed in enumerate(seeds):
                    num, den = map(int, seed.split('/'))
                    expected = residue(method, num*pow(den, -1, prime) % prime,
                                       math.lcm(*range(1, 101)), prime)
                    check(int(rows[i]['X'], 0) == expected, ('resumed arithmetic', rows[i]))
                    check(f'x0={seed}' in out, ('resumed console', out))

            # Denominator factors still have a known original fraction.
            for seed in ('2/7', '4/14'):
                out = run([method, *mode, '-x0', seed, 100, 0], f'2 12700-M {7*prime}')
                check(f'Factor found in step 1: 7' in out and f'X0: {seed};' in log.read_text().splitlines()[-1], out)
            run([method, *mode, '-x0', '2/0', 10, 0], str(prime), error=True)
            for bad in ('2/0', 'word', '2/7/3'):
                source.write_text(mixed[0].replace('X0=2/7;', f'X0={bad};')+'\n')
                run([method, *mode, '-resume', source, 100, 0], error=True)

        # Known stage-2 factors; fractions equivalent to the established bases
        # verify that unreduced originals are not replaced with integers.
        seed, factor, cofactor, b1, b2 = (('14/7', 1000033, 1000003, 100, 10000)
                                         if method == '-pm1' else ('20/5', 1000039, prime, 23, 1087))
        for index, mode in enumerate(modes):
            saved = root / f'{method}-{index}-stage2.save'
            run([method, *mode, '-x0', seed, '-save', saved, b1, 0], f'2 12700-L {factor*cofactor}')
            for dest in modes:
                out = run([method, *dest, '-resume', saved, b1, b2])
                check(f'Factor found in step 2: {factor}' in out, out)
                check(f'X0: {seed};' in log.read_text().splitlines()[-1], 'stage-2 original fraction')

        if args.gpu:
            # A partial ladder stores modular GPU operands independently of
            # its compact original seed. Check both algorithms and endpoints.
            exponent = math.lcm(*range(1, 101))
            modulus = 4294967291
            for seed in ('2/7', '6/5'):
                num, den = map(int, seed.split('/'))
                base = num*pow(den, -1, prime) % prime
                for bits in (0, 17, exponent.bit_length()):
                    prefix = exponent >> (exponent.bit_length()-bits)
                    x = residue(method, base, prefix, prime)
                    fields = [base % modulus, 100, 0, bits, 0]
                    next_field = ''
                    if method == '-pp1':
                        next_x = residue(method, base, prefix+1, prime)
                        next_field = f' GPU_NEXT={hex(next_x)};'
                        fields.append(next_x % modulus)
                    checksum = (prime % modulus)*(x % modulus)
                    for field in fields:
                        checksum = checksum*65599+field
                    source, finished = root / 'partial.save', root / 'partial-completed.save'
                    source.write_text(f'METHOD={"P-1" if method == "-pm1" else "P+1"}; '
                                      f'B1=1; N={prime}; X={hex(x)}; GPU_B1=100; GPU_BITS={bits}; '
                                      f'GPU_BASE={hex(base)};{next_field} CHECKSUM={checksum % modulus}; X0={seed};\n')
                    run([method, '-gpu', '-resume', source, '-save', finished, 100, 0])
                    row = records(finished)[0]
                    check(row['X0'] == seed and int(row['X'], 0) == residue(method, base, exponent, prime),
                          ('partial resume', row))

    if args.interrupter:
        interrupter = root / 'interrupt.exe'
        shutil.copy2(args.interrupter.resolve(), interrupter)
        n = (1 << 521)-1
        for batch in (1, 2):
            seed = '2/7' if batch == 1 else '6/5'
            num, den = map(int, seed.split('/'))
            base = num*pow(den, -1, n) % n
            expected = residue('-pp1', base, math.lcm(*range(1, 1001)), n)
            check(math.gcd(expected-2, n) == 1, 'signal fixture')
            saved, checkpoint = root / f'signal-{batch}.save', root / f'signal-{batch}.checkpoint'
            p = subprocess.run([str(interrupter), '-pp1', '-gpu', '-x0', seed,
                                '-gpubatch', '3', '-save', str(saved), '-chkpnt', str(checkpoint), '1000', '0'],
                               input='\n'.join(f'2 {500+i}-M {n}' for i in range(11)), cwd=root,
                               env=dict(env, ECM_TEST_STOP_BATCH=str(batch)), text=True, capture_output=True, timeout=120)
            check(p.returncode == 143, ('signal', p.returncode, p.stdout, p.stderr))
            for source in (saved, checkpoint):
                rows = records(source)
                check(any(int(r.get('GPU_BITS', 0)) > 0 for r in rows), 'partial CUDA work')
                check(all(r['X0'] == seed for r in rows), ('partial seed', rows))
                finished = root / 'signal-completed.save'
                run(['-pp1', '-gpu', '-gpubatch', 2, '-resume', source, '-save', finished, 1000, 0])
                final = records(finished)
                check(len(final) == len(rows) and all(r['X0'] == seed and int(r['X'], 0) == expected
                                                      for r in final), ('partial resume', final))

print(f'PASS: {checks} rational X0 checks' + (' including GPU' if args.gpu else ''))

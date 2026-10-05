"""Run with ECM, a separately supplied grouporder executable, and optional stub.

python tests/test-brent-logging.py /path/to/ecm /path/to/grouporder [--stub PATH]
The test copies executables into a temporary directory; no SEA files are used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('ecm', type=Path)
p.add_argument('helper', type=Path)
p.add_argument('--stub', type=Path)
p.add_argument('--gpu', action='store_true')
args = p.parse_args()
exe_source, helper_source = args.ecm.resolve(), args.helper.resolve()
checks = 0


def check(value, message):
    global checks
    assert value, message
    checks += 1


with tempfile.TemporaryDirectory(prefix='ECM Brent logging ') as directory:
    root = Path(directory)
    binary = root / 'bin with spaces'
    binary.mkdir()
    work = root / 'work'
    work.mkdir()
    exe = binary / exe_source.name
    shutil.copy2(exe_source, exe)
    for dll in exe_source.parent.glob('*.dll'):
        shutil.copy2(dll, binary)
    helper = binary / ('grouporder.exe' if os.name == 'nt' else 'grouporder')
    env = os.environ.copy()
    env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32') if os.name == 'nt' else '/usr/bin:/bin'
    env.pop('GROUPORDER_DATA_DIR', None)
    log = work / 'BrentFactors.log'
    n = 101 * (2**127 - 1)
    source = f'2 12700-L {n}'

    def run(options, text=source, expected=True):
        before = log.read_text().splitlines() if log.exists() else []
        result = subprocess.run([str(exe), *options], input=text + '\n', cwd=work,
                                env=env, text=True, capture_output=True, timeout=120)
        output = result.stdout + result.stderr
        check(result.returncode in (0, 2, 6, 8, 10, 14), output)
        after = log.read_text().splitlines() if log.exists() else []
        check(after[:len(before)] == before, 'Existing records were overwritten')
        entries = after[len(before):]
        check(bool(entries) == expected, output)
        check(re.findall(r'^Adding Log Entry --> "([^\r\n]*)"$', output, re.M) == entries,
              'Screen and file records differ: ' + output)
        if '-q' in options:
            check('Adding Log Entry' not in result.stdout, result.stdout)
        return output, entries

    _, entries = run(['-sigma', '0:6', '100', '0'])
    check(entries == ['2 12700-L 101; Sigma: 0:6; B1=100, B2=0'], entries)
    # Supplied helper beside ECM, while the working directory is elsewhere.
    shutil.copy2(helper_source, helper)

    def verify_order(entry):
        match = re.fullmatch(r'(.+?) (\d+); Sigma: ([0-3]):(\d+); B1=\d+, B2=\w+; '
                             r'Group Order: (\d+) = ([0-9 *^]+)', entry)
        check(match is not None, entry)
        label, factor, param, sigma, order, factored = match.groups()
        result = subprocess.run([str(helper), factor, param, sigma], cwd=work,
                                env=env, capture_output=True, text=True, timeout=120)
        check(result.returncode == 0, result.stderr)
        check(re.search(r'Group Order:\s*(\d+)', result.stdout)[1] == order, entry)
        check(re.search(r'Factored:\s*([^\r\n]+)', result.stdout)[1].strip() == factored, entry)
        product = 1
        for term in factored.split(' * '):
            parts = term.split('^')
            product *= int(parts[0]) ** (int(parts[1]) if len(parts) == 2 else 1)
        check(product == int(order), entry)
        return label, int(factor), int(param), int(sigma)

    for sigma in ('0:6', '1:11', '2:11', '3:18446744073709551615'):
        _, entries = run(['-sigma', sigma, '300', '0'])
        for entry in entries:
            verify_order(entry)
    output, entries = run(['300'])
    for entry in entries:
        verify_order(entry)
        using = re.search(r'Using B1=300, B2=(\d+).*sigma=([0-3]):(\d+)', output)
        check(using is not None and f'B2={using[1]};' in entry, output)
        check(f'Sigma: {using[2]}:{using[3]};' in entry, output)
    _, entries = run(['-q', '-sigma', '0:6', '100', '0'])
    verify_order(entries[0])
    run(['-sigma', '0:6', '100', '0'], str(n), False)
    run(['-sigma', '3:671088640', '3', '0'], source, False)
    # Composite factors and non-ECM methods do not have an elliptic group order.
    _, entries = run(['-sigma', '0:6', '300', '0'], f'2 12700-M {n * 103}')
    check('10403;' in entries[0] and 'Group Order:' not in entries[0], entries)
    for method in ('-pm1', '-pp1'):
        output, entries = run([method, '-x0', '3', '100'])
        check(all('Group Order:' not in e and 'Sigma: unavailable' in e for e in entries), entries)
        bound = re.search(r'Using B1=100, B2=(?:\d+-)?(\d+)', output)[1]
        check(all(f'B2={bound}' in e for e in entries), entries)
    _, entries = run(['-sigma', '0:6', '100'], '3 5+M 122')
    check(entries == ['3 5+M 2; Sigma: unavailable; B1=100, B2=auto'], entries)
    _, entries = run(['-sigma', '0:6', '100'], source + '\n3 5+M 122')
    check(len(entries) == 2 and entries[1] ==
          '3 5+M 2; Sigma: unavailable; B1=100, B2=auto', entries)
    verify_order(entries[0])
    large_bound = '123456789012345678901234567890'
    _, entries = run(['100', large_bound], '3 5+M 122')
    check(entries[0].endswith('B2=' + large_bound), entries)
    _, entries = run(['-A', '1/7', '1'], '2 12-M 273')
    check(entries == ['2 12-M 7; Sigma: unavailable; B1=1, B2=auto'], entries)
    resume_n = 1000003 * (2**127 - 1)
    run(['-sigma', '3:18446744073709551615', '-save', 'stage1.save', '10', '0'],
        f'2 127000254+M {resume_n}', False)
    output, entries = run(['-resume', 'stage1.save', '10', '20000'], '')
    check('Factor found in step 2: 1000003' in output, output)
    check(entries[0].startswith('2 127000254+M 1000003;'), entries)
    verify_order(entries[0])
    # Discover the supplied helper using PATH as well.
    pathdir = root / 'helper on PATH'
    pathdir.mkdir()
    shutil.move(str(helper), pathdir / helper.name)
    env['PATH'] = str(pathdir) + os.pathsep + env['PATH']
    _, entries = run(['-sigma', '0:6', '100', '0'])
    check('Group Order:' in entries[0], entries)
    env['PATH'] = env['PATH'].split(os.pathsep, 1)[1]
    shutil.move(str(pathdir / helper.name), helper)

    config = subprocess.run([str(exe), '-printconfig'], cwd=work, env=env,
                            text=True, capture_output=True, check=True).stdout
    if 'Included GWNUM header files version' in config:
        for sigma in ('0:6', '3:18446744073709551615'):
            output, entries = run(['-force-gwnum', '-sigma', sigma, '100', '0'],
                                  f'2 52100-L {101 * (2**521 - 1)}')
            check("Using gwnum fft's" in output, output)
            for entry in entries:
                verify_order(entry)

    if args.gpu:
        gpu = ['-gpu', '-gpucurves', '32', '-sigma', '3:18446744073709551584']
        for bounds in (['100', '0'], ['10', '20000']):
            output, entries = run([*gpu, *bounds], f'2 127000254-L {resume_n}')
            original = [(int(f), int(s)) for f, s in re.findall(
                r'GPU: factor (\d+) found in Step [12].*?-sigma 3:(\d+)', output)]
            for entry in entries:
                _, factor, param, sigma = verify_order(entry)
                check(18446744073709551584 <= sigma <= 18446744073709551615, entry)
                check(any(f % factor == 0 and s == sigma for f, s in original), output)
        # Different curves can return overlapping composite factors. Check
        # attribution after splitting/sorting, rather than using batch sigma+i.
        output, entries = run([*gpu, '10', '0'], f'2 12700-M {n * 103 * 107}')
        check(len(entries) >= 2, output)
        original = [(int(f), int(s)) for f, s in re.findall(
            r'GPU: factor (\d+) found in Step [12].*?-sigma 3:(\d+)', output)]
        for entry in entries:
            _, factor, _, sigma = verify_order(entry)
            check(any(f % factor == 0 and s == sigma for f, s in original), output)

    if args.stub:
        shutil.copy2(args.stub.resolve(), helper)
        for mode in ('failure', 'malformed', 'oversize'):
            env['ECM_TEST_HELPER_MODE'] = mode
            _, entries = run(['-sigma', '0:6', '100', '0'])
            check(entries[0].endswith('; Group Order: unavailable'), entries)
        del env['ECM_TEST_HELPER_MODE']
    helper.unlink()
    # Concurrent appenders must not lose or interleave records.
    before = len(log.read_text().splitlines())
    def concurrent(_):
        return subprocess.run([str(exe), '-q', '100', '0'], input='3 5+M 122\n',
                              cwd=work, env=env, text=True, capture_output=True, timeout=30)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(concurrent, range(24)))
    check(all(r.returncode == 14 for r in results), results)
    entries = log.read_text().splitlines()[before:]
    check(entries == ['3 5+M 2; Sigma: unavailable; B1=100, B2=0'] * 24, entries)
    # A failed log append does not change factoring success or claim a write.
    log.unlink()
    log.mkdir()
    result = subprocess.run([str(exe), '100', '0'], input='3 5+M 122\n', cwd=work,
                            env=env, text=True, capture_output=True, timeout=30)
    check(result.returncode == 14 and 'cannot append' in result.stderr, result)
    check('Adding Log Entry' not in result.stdout, result.stdout)
    check(not any(x.name in ('data', 'seadata') for x in root.rglob('*')), 'Unexpected SEA directory')

print(f'PASS: {checks} Brent logging checks' + (' including GPU' if args.gpu else ''))

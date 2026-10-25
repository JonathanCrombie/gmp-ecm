"""CLI color regression checks: python tests/test-color.py ECM [--gpu].

POSIX also tests real pseudo-terminals and mixed terminal/pipe destinations.
All factoring and log writes take place in an isolated temporary directory.
"""
import argparse
import errno
import os
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('ecm', type=Path)
p.add_argument('--gpu', action='store_true')
args = p.parse_args()
exe = args.ecm.resolve()
ansi = re.compile(r'\x1b\[[0-9;]*m')
checks = 0


def check(ok, detail):
    global checks
    assert ok, detail
    checks += 1


def plain(text):
    return ansi.sub('', text)


with tempfile.TemporaryDirectory(prefix='ecm-colors-') as tmp:
    root = Path(tmp)
    env = os.environ.copy()
    env.pop('NO_COLOR', None)
    env['TERM'] = 'xterm-256color'
    serial = 0

    def run(opts, text='', extra_env=None):
        global serial
        serial += 1
        work = root / str(serial)
        work.mkdir()
        result = subprocess.run([str(exe), *opts], cwd=work, env=env | (extra_env or {}),
                                input=text, text=True, capture_output=True, timeout=120)
        check(result.returncode in (0, 1, 2, 6, 8, 10, 14), result)
        return result, work

    prime = (1 << 127) - 1
    hard = prime * ((1 << 521) - 1)
    data = f'2 12700-L {101*prime}\n2 12701-L {103*prime}\n2 12702-L {hard}\n'
    modes = [[], ['-threads', '4']]
    methods = [['-sigma', '0:6'], ['-pm1', '-x0', '3'], ['-pp1', '-x0', '3']]
    for method in methods:
        variants = modes + ([['-gpu', '-gpubatch', '2']] if args.gpu and method[0] != '-sigma' else [])
        for mode in variants:
            records = []
            factors = []
            for color in ('auto', 'never', 'always'):
                r, work = run(['-color', color, *mode, *method, '-save', 'residues.save', '100', '0'], data)
                check(r.returncode != 1, r.stderr)
                if color == 'always':
                    check('\x1b[36m' in r.stdout and '\x1b[92m' in r.stdout, r.stdout)
                    check(re.search(r'\x1b\[36m\*+ Factor found in step \d: \x1b\[92m\d+\x1b\[36m\n', r.stdout), r.stdout)
                    check(re.search(r'\x1b\[36mStep 1', r.stdout), r.stdout)
                else:
                    check('\x1b' not in r.stdout + r.stderr, (color, r))
                log = (work / 'BrentFactors.log').read_text()
                saved = (work / 'residues.save').read_text()
                check('\x1b' not in log + saved, (log, saved))
                check('METHOD=' in saved and str(hard) in saved, saved)
                records.append(re.sub(r'TIME=[^;]*;', '', saved))
                factors.append(re.findall(r'Factor found in step \d: \d+', plain(r.stdout)))
                check(re.findall(r'Input number is (\d+)', plain(r.stdout)) ==
                      [str(101*prime), str(103*prime), str(hard)], r.stdout)
                # Reading a colored run's save must work in serial and worker modes.
                resumed, _ = run(['-color', color, *mode, *method,
                                  '-resume', str(work / 'residues.save'), '100', '0'])
                check(resumed.returncode != 1 and 'Input number is' in plain(resumed.stdout), resumed)
            check(records[0] == records[1] == records[2], records)
            check(factors[0] == factors[1] == factors[2], factors)
            quiet, _ = run(['-color', 'always', '-q', *mode, *method, '100', '0'], data)
            check('\x1b' not in quiet.stdout and re.fullmatch(r'[0-9\s]+', quiet.stdout), quiet)

    for opts in (['-color'], ['-color', 'invalid'], ['-color', 'always', '-threads', '0']):
        r, _ = run(opts)
        check(r.returncode == 1, r)
        if 'always' in opts:
            check('\x1b[31m' in r.stderr and '\x1b[0m' in r.stderr, r)
    for color in ('always', 'never', 'auto'):
        r, _ = run(['-color', color, '-pm1', '-x0', '3', '-go', '1', '-go', '2', '10', '0'], str(hard))
        check('Warning' in r.stderr and ('\x1b[33m' in r.stderr) == (color == 'always'), r)
        r, _ = run(['-color', color, '-pm1', '-x0', '3', '10', '0'], str(hard), {'NO_COLOR': '1'})
        check(('\x1b' in r.stdout) == (color == 'always'), r)

    # Library diagnostic produced within a worker must retain its final color.
    for mode in ([], ['-threads', '4']):
        r, _ = run(['-color', 'always', *mode, '-sigma', '0:1', '10', '0'], str(hard))
        check(r.returncode == 1 and '\x1b[31m' in r.stderr, r)

    # Initialization factors use a separate path from ordinary stage results.
    r, work = run(['-color', 'always', '-v', '-pm1', '-x0', '2/7', '10', '0'],
                  f'2 12700-L {7*prime}')
    check('\x1b[33mWarning:' in r.stdout and
          re.search(r'\x1b\[36m\*+ Factor found in step 1: \x1b\[92m7\x1b\[36m\n', r.stdout), r)
    check('\x1b' not in (work / 'BrentFactors.log').read_text(), work)
    large = str((1 << 4000) - 17)
    r, _ = run(['-color', 'always', '-pm1', '-x0', '3', '1', '0'], large)
    check(f'Input number is {large[:10]}...{int(large[-10:])} ({len(large)} digits)' in plain(r.stdout), r)
    r, work = run(['-color', 'always', '-pm1', '-x0', '3', '-chkpnt', 'checkpoint', '100', '0'], str(hard))
    check('METHOD=P-1;' in (work / 'checkpoint').read_text() and
          '\x1b' not in (work / 'checkpoint').read_text(), r)
    for opts, enabled in (([], False), (['-color', 'always', '-color', 'never'], False),
                          (['-color', 'never', '-color', 'always'], True)):
        r, _ = run([*opts, '-pm1', '-x0', '3', '10', '0'], str(hard))
        check(('\x1b' in r.stdout) == enabled, r)

    if os.name != 'nt':
        import pty

        def terminal(stream, opts, extra_env=None, text=''):
            master, slave = pty.openpty()
            try:
                streams = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                streams[stream] = slave
                proc = subprocess.Popen([str(exe), *opts], cwd=root,
                                        env=env | (extra_env or {}), stdin=subprocess.PIPE, **streams)
                os.close(slave)
                slave = None
                out, err = proc.communicate(text.encode(), timeout=30)
                chunks = []
                while True:
                    try:
                        data = os.read(master, 8192)
                        if not data:
                            break
                        chunks.append(data)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        break
                return b''.join(chunks).decode().replace('\r\n', '\n'), (err if stream == 'stdout' else out).decode()
            finally:
                os.close(master)
                if slave is not None:
                    os.close(slave)

        for settings, enabled in (({}, True), ({'NO_COLOR': ''}, True), ({'NO_COLOR': '1'}, False),
                                  ({'TERM': 'dumb'}, False), ({'TERM': ''}, False)):
            out, err = terminal('stdout', ['-color', 'auto', '-pm1', '-x0', '3', '10', '0'], settings, str(hard))
            check(('\x1b[36m' in out) == enabled and '\x1b' not in err, (settings, out, err))
        out, err = terminal('stderr', ['-color', 'auto', '-threads', '0'])
        check('\x1b[31m' in out and '\x1b' not in err, (out, err))
        out, err = terminal('stdout', ['-color', 'never', '-pm1', '-x0', '3', '10', '0'], text=str(hard))
        check('\x1b' not in out + err, out)

print(f'PASS {checks} color, stream, quiet-mode, save and logging checks')

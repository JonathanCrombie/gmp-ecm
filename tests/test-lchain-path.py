"""Windows integration test: python tests/test-lchain-path.py build-msvc/bin/ecm.exe"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('ecm', type=Path)
args = parser.parse_args()
if os.name != 'nt':
    parser.error('The executable-directory fallback is Windows-specific.')
source = args.ecm.resolve()
checks = 0


def check(condition, detail):
    global checks
    assert condition, detail
    checks += 1


with tempfile.TemporaryDirectory(prefix='ECM Lucas lookup ') as directory:
    root = Path(directory)
    # Exercise spaces and a directory name outside the Windows ANSI code page.
    binary = root / 'bin with spaces \u6f22'
    binary.mkdir()
    work = root / 'working directory'
    work.mkdir()
    for item in [source, source.parent / 'LucasChainGen.exe',
                 *source.parent.glob('*.dll')]:
        shutil.copy2(item, binary / item.name)
    exe = binary / source.name
    chain = 'Lchain_codes.dat'
    env = os.environ.copy()
    env['PATH'] = str(binary) + os.pathsep + env.get('PATH', '')
    env.pop('GMPECM_DATADIR', None)

    data = {}
    for bound in (100, 1000):
        generator_work = root / str(bound)
        generator_work.mkdir()
        result = subprocess.run([str(binary / 'LucasChainGen.exe'), '-B1', str(bound),
                                 '-nT', '1'], cwd=generator_work, env=env,
                                capture_output=True, text=True, timeout=30)
        check(result.returncode == 0, result)
        data[bound] = (generator_work / chain).read_bytes()
        check(bool(data[bound]) and len(data[bound]) % 8 == 0, bound)
    check(b'\x1a' in data[1000], 'Missing binary-mode EOF-byte coverage')

    def run(mode, expected, via_path=False):
        save = work / 'residue.save'
        save.unlink(missing_ok=True)
        command = [str(exe), mode, '-sigma', '0:42', '-save', save.name, '1000', '0']
        if via_path:
            # Let the child shell search its PATH. CreateProcess otherwise uses
            # the parent's search path when locating an unqualified application.
            command = [os.environ['COMSPEC'], '/d', '/c', source.name, *command[1:]]
        result = subprocess.run(
            command,
            input=str(2**1279 - 1) + '\n', cwd=work, env=env,
            capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        check(result.returncode == 0, output)
        check(("Using gwnum fft's" in output) == (mode == '-force-gwnum'), output)
        check(('file failed to open' in output) == (expected == 'missing'), output)
        check(('Reached Lchain_codes.dat EOF' in output) == (expected == 'short'), output)
        check(save.is_file() and not (binary / save.name).exists(), 'Working directory changed')
        record = dict(re.findall(r'(?:^|;)\s*(\w+)=([^;]*)', save.read_text()))
        return tuple(record[field] for field in ('N', 'B1', 'X', 'CHECKSUM'))

    for mode in ('-force-no-gwnum', '-force-gwnum'):
        # No file: ordinary PRAC remains available.
        baseline = run(mode, 'missing')
        # Fallback beside the executable, including invocation through PATH.
        (binary / chain).write_bytes(data[1000])
        for via_path in (False, True):
            check(run(mode, 'complete', via_path) == baseline, 'Fallback residue mismatch')
        # A shorter working-directory file wins over the complete fallback.
        (work / chain).write_bytes(data[100])
        check(run(mode, 'short') == baseline, 'Working-directory precedence mismatch')
        (work / chain).unlink()
        # An inaccessible local entry is not treated as an absent file.
        (work / chain).mkdir()
        check(run(mode, 'missing') == baseline, 'Unexpected fallback after local open error')
        (work / chain).rmdir()
        (binary / chain).unlink()
        # A complete local file also works with no executable-directory copy.
        (work / chain).write_bytes(data[1000])
        check(run(mode, 'complete') == baseline, 'Local residue mismatch')
        (work / chain).unlink()

print(f'PASS: {checks} Windows Lucas-chain lookup checks (CPU ECM and GWNUM)')

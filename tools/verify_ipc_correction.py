"""Verify the exact two-H100 IPC correction record; never execute CUDA here."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
DATASET = '2026-11-19-h100-ipc'
SOURCE = 'docs/part1/code/17_ipc_mem_handle.cu'
SOURCE_SHA = '0a6c761e152bb3d0015b3f96bfb93cf7fa61764b90cde1f23aafdf11539a6388'
OUTPUT = ('Consumer: 4096 / 4096 exact payload values correct.\n'
          'PASS: two-process IPC memory, 4096 nonuniform values checked.\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    public = root / 'docs/validation' / DATASET
    integrity = json.loads((public / 'integrity.json').read_text())
    inventory = integrity['files']
    actual = {p.name for p in public.iterdir() if p.is_file()} - {'integrity.json'}
    require(set(inventory) == actual, 'IPC evidence inventory differs')
    for name, digest in inventory.items():
        require(Path(name).name == name and sha(public / name) == digest, 'Changed IPC evidence: ' + name)
    result = json.loads((public / 'results.json').read_text())
    require(result['status'] == 'PASS' and result['source'] == SOURCE
            and result['source_sha256'] == SOURCE_SHA and result['arch'] == 'sm_90'
            and result['normal_runs'] == 20 and result['memcheck_runs'] == 2
            and result['payload_values_per_run'] == 4096, 'Unexpected IPC result scope')
    require(sha(root / SOURCE) == SOURCE_SHA and sha(public / Path(SOURCE).name) == SOURCE_SHA,
            'Current IPC source differs from tested revision')
    uuids = result['gpu_uuids']
    require(len(uuids) == 2 and len(set(uuids)) == 2
            and all(re.fullmatch(r'GPU-[0-9a-f-]{36}', value) for value in uuids), 'Invalid GPU identities')
    metadata = {'gpu', 'topology', 'nvcc', 'gcc', 'os', 'kernel', 'sanitizer-version'}
    executions = {f'{direction}-{n:02}' for direction in ('forward', 'reverse') for n in range(1, 11)}
    memchecks = {'forward-memcheck', 'reverse-memcheck'}
    commands = json.loads((public / 'commands.json').read_text())
    names = [row['name'] for row in commands]
    require(len(names) == 30 and len(set(names)) == 30
            and set(names) == metadata | executions | memchecks | {'compile'}, 'Incomplete IPC commands')
    previous = None
    for row in commands:
        name = row['name']; log = public / row['log']
        require(row['log'] == name + '.log' and row['exit_code'] == 0 and not row['timed_out']
                and sha(log) == row['log_sha256'], 'Failed/changed IPC command: ' + name)
        raw = log.read_text()
        require(raw.startswith('$ ' + shlex.join(row['command']) + '\n'), 'IPC command/log mismatch: ' + name)
        start, finish = (datetime.fromisoformat(row[key]) for key in ('started_utc', 'finished_utc'))
        require(start.utcoffset() == timezone.utc.utcoffset(start)
                and finish.utcoffset() == timezone.utc.utcoffset(finish)
                and row['started_utc'].startswith('2026-11-19T') and start <= finish
                and (previous is None or previous <= start), 'Invalid IPC timestamp ordering')
        previous = finish
        output = raw.split('\n', 1)[1]
        if name in executions | memchecks:
            direction = name.split('-')[0]
            expected_devices = uuids if direction == 'forward' else uuids[::-1]
            require(row['environment'] == {'CUDA_VISIBLE_DEVICES': ','.join(expected_devices),
                                          'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}, 'Incorrect device-role mapping')
            require(OUTPUT in output, 'Incomplete IPC payload check: ' + name)
            if name in executions:
                require(row['command'] == ['./example'] and output == OUTPUT, 'Unexpected IPC execution')
            else:
                require(row['command'] == ['compute-sanitizer', '--tool', 'memcheck', '--target-processes',
                                          'all', '--error-exitcode', '3', './example'], 'Incorrect IPC memcheck')
                counts = re.findall(r'ERROR SUMMARY: (\d+) errors?', output)
                require(bool(counts) and all(value == '0' for value in counts), 'IPC memcheck errors')
        elif name == 'compile':
            require(row['command'] == ['nvcc', '-std=c++17', '-O2', '-arch=sm_90',
                                      '17_ipc_mem_handle.cu', '-o', 'example'], 'Incorrect IPC compile command')
    gpu = (public / 'gpu.log').read_text()
    require(all(value in gpu for value in uuids) and gpu.count('NVIDIA H100 80GB HBM3') == 2
            and '580.105.08' in gpu and 'NV18' in (public / 'topology.log').read_text()
            and 'V12.8.93' in (public / 'nvcc.log').read_text()
            and '13.3.0' in (public / 'gcc.log').read_text()
            and 'Ubuntu 24.04.5' in (public / 'os.log').read_text(), 'IPC hardware/tool identity differs')
    report_path = root / 'docs/validation-h100-ipc.md'
    require(sha(report_path) == integrity['summary_sha256'], 'IPC report differs from inventory')
    report = report_path.read_text()
    chapter = (root / 'docs/part1/07-cuda-inter-process-communication.md').read_text()
    require(SOURCE_SHA in report and SOURCE_SHA in chapter and OUTPUT.rstrip() in report
            and OUTPUT.rstrip() in chapter and 'Correction verified on November 19' in chapter,
            'Book omits current IPC source/result')
    require(all(row['started_utc'] in report for row in commands if row['name'] in executions | memchecks),
            'IPC report omits execution dates')
    require((public / 'forward-memcheck.log').read_text().split('\n', 1)[1].rstrip() in report,
            'IPC report changes the captured memcheck output')
    return result


if __name__ == '__main__':
    verify()
    print('PASS: exact IPC revision, two H100 role mappings, 20 payload checks and two clean memchecks.')

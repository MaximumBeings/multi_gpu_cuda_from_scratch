#!/usr/bin/env python3
"""Publish complete, current-hash execution evidence as an immutable architecture dataset."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {'.cu', '.cpp', '.c'}
SUCCESS = {'validation_pass', 'api_probe_pass', 'executed_unverified'}
ARCHITECTURES = {'h100': 'sm_90', 'a100': 'sm_80', 'b200': 'sm_100'}
FENCES = re.compile(r'^(`{3,})([^\n]*)\n(.*?)^\1[ \t]*$', re.M | re.S)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    return {str(p.relative_to(root)): p for p in sorted(root.glob('docs/**/code/*')) if p.suffix in SUFFIXES}


def strip_command(log):
    """Display normalization only: omit the command header and trailing line whitespace."""
    output = log.split('\n', 1)[1] if log.startswith('$ ') and '\n' in log else log
    return '\n'.join(line.rstrip() for line in output.splitlines())


def normalize(command, source):
    filename = Path(source).name
    return [filename if arg.endswith('/' + source) or arg.endswith('/' + filename)
            else './example' if arg.endswith('/example') else arg for arg in command]


def category(row):
    if row['status'] == 'validation_pass':
        return 'Numerical/transport validation (recorded scope)'
    if row['status'] == 'api_probe_pass':
        return 'API validation (recorded scope)'
    if not row.get('cuda') and not row.get('dependencies'):
        return 'Host program/model (exit-only)'
    return 'API/host demonstration (exit-only)'


def supplements(root, public, label=None):
    """Render only facts supported by retained raw logs and checked supplemental manifests."""
    prefix = 'validation/' + public.name
    lines = ['## Captured system and supplemental checks', '']
    def linked(path, title):
        return f'[{title}]({prefix}/{path.relative_to(public).as_posix()})'
    probes = sorted(public.glob('systems/*/gpus.log'))
    if probes:
        gpu_lines = [line for line in probes[0].read_text().splitlines() if line.startswith(('NVIDIA', 'Tesla', 'GeForce'))]
        if gpu_lines:
            names = collections.Counter(line.split(',')[0].strip() for line in gpu_lines)
            description = ', '.join(f'{count}× {name}' for name, count in names.items())
            fields = gpu_lines[0].split(',')
            driver = fields[-2].strip() if len(fields) >= 4 else 'see log'
            memory = fields[-1].strip() if len(fields) >= 4 else 'see log'
            lines += [f'GPU inventory: **{description}**. Reported driver: **{driver}**; memory per listed device: **{memory}**. ' + linked(probes[0], 'GPU inventory log') + '.', '']
    topology = sorted(public.glob('systems/*/topology.log'))
    if topology:
        routes = sorted(set(re.findall(r'\bNV\d+\b', topology[0].read_text())))
        if routes:
            lines += [f'Topology query reports **{", ".join(routes)}** peer paths. This identifies the recorded topology, not measured bandwidth. ' + linked(topology[0], 'Topology log') + '.', '']
    version_paths = [(public / '120_toolchain_version_check/run.log', r'GNU-compatible host compiler macros: ([0-9.]+)', 'Host compiler macros'),
                     (public / '120_toolchain_version_check/run.log', r'CUDA compiler: ([0-9.]+)', 'CUDA compiler'),
                     (public / '121_multi_gpu_stack_version_check/run.log', r'NCCL_MAJOR = (\d+), NCCL_MINOR = (\d+), NCCL_PATCH = (\d+)', 'NCCL headers'),
                     (public / '121_multi_gpu_stack_version_check/run.log', r'OMPI_MAJOR_VERSION = (\d+), OMPI_MINOR_VERSION = (\d+), OMPI_RELEASE_VERSION = (\d+)', 'Open MPI headers'),
                     (public / 'hello_nvshmem/run.log', r'NVSHMEM ([0-9.]+)', 'NVSHMEM')]
    for path, pattern, title in version_paths:
        if path.is_file():
            match = re.search(pattern, path.read_text())
            if match:
                lines += [f'{title}: **{".".join(match.groups())}**. ' + linked(path, 'Captured version output') + '.', '']
    for manifest_path in sorted(public.glob('systems/*/metadata/memcheck/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        names = []
        for test in manifest['tests']:
            log = manifest_path.parent / test['log']
            if (test.get('process_exit') != 0 or test.get('reported_errors') != 0
                    or sha(root / test['source']) != test['source_sha256']
                    or not re.search(r'^=+\s*ERROR SUMMARY: 0 errors\s*$', log.read_text(), re.M)):
                raise ValueError('Invalid supplemental memcheck evidence: ' + test['source'])
            names.append(linked(log, Path(test['source']).name))
        lines += ['Compute Sanitizer memcheck completed with exit zero and **ERROR SUMMARY: 0 errors** for ' + ', '.join(names) + '. These checks cover the recorded executions, not every possible race or asynchronous protocol. ' + linked(manifest_path, 'Memcheck manifest') + '.', '']
    for manifest_path in sorted(public.glob('systems/*/metadata/setup/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        makefile = manifest_path.parent / 'Makefile.txt'
        if (manifest.get('compile_exit') != 0 or sha(makefile) != manifest['makefile_sha256']
                or sha(root / 'docs/appendix/code/Makefile') != manifest['makefile_sha256']):
            raise ValueError('Setup Makefile supplemental evidence mismatch.')
        lines += ['The captured Appendix A Makefile build completed with exit zero. ' + linked(makefile, 'Exact Makefile') + ' · ' + linked(manifest_path.parent / 'build.log', 'Build output') + ' · ' + linked(manifest_path, 'Build manifest') + '. This checks the five build/link recipes; program execution is recorded separately in the main manifest.', '']
    for manifest_path in sorted(public.glob('systems/*/metadata/eight-device/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        folder = manifest_path.parent
        source_name = manifest['source']
        if Path(source_name).name != source_name or source_name != 'validate_eight_gpus.cu':
            raise ValueError('Unexpected eight-device source filename.')
        if (label != 'a100' or manifest.get('arch') != 'sm_80' or manifest.get('status') != 'PASS'
                or sha(root / 'tools/validate_eight_gpus.cu') != manifest['source_sha256']
                or sha(folder / source_name) != manifest['source_sha256']):
            raise ValueError('Eight-device architecture, status or source provenance mismatch.')
        for stage in ('compile', 'run'):
            item = manifest[stage]
            if Path(item['log']).name != item['log']:
                raise ValueError('Unsafe eight-device log filename.')
            if item.get('exit_code') != 0 or item.get('timed_out') or sha(folder / item['log']) != item['log_sha256']:
                raise ValueError('Eight-device ' + stage + ' evidence failed integrity/status checks.')
        system_log = folder / 'system-info.log'
        if sha(system_log) != manifest['system_sha256']:
            raise ValueError('Eight-device system information hash mismatch.')
        hardware = [line for line in system_log.read_text().splitlines() if line.startswith('NVIDIA')]
        if len(hardware) != 8 or any(not re.match(r'^NVIDIA A100[^,]*, [^,]+, 8\.0,', line) for line in hardware):
            raise ValueError('Eight-device system record must identify eight A100 CC 8.0 GPUs.')
        output = (folder / manifest['run']['log']).read_text()
        devices = re.findall(r'^GPU ([0-7]): NVIDIA A100[^\n]+ CC=8\.0 ', output, re.M)
        if len(devices) != 8 or set(devices) != set('01234567'):
            raise ValueError('Eight-device runtime GPU inventory is incomplete.')
        pairs = re.findall(r'^PEER_COPY ([0-7])->([0-7]): PASS;[^\n]*; errors=0\s*$', output, re.M)
        expected_pairs = {(str(a), str(b)) for a in range(8) for b in range(8) if a != b}
        if len(pairs) != 56 or set(pairs) != expected_pairs:
            raise ValueError('Eight-device peer copies must cover all 56 unique ordered pairs.')
        markers = ['ALL_PEER_PAIRS', 'ALL_REDUCE', 'ALL_GATHER', 'REDUCE_SCATTER', 'GROUPED_ALL_TO_ALL']
        for marker in markers:
            if len(re.findall(r'^' + marker + r': PASS;[^\n]*errors=0\s*$', output, re.M)) != 1:
                raise ValueError('Eight-device collective/peer marker missing: ' + marker)
        if not re.search(r'^ALL_PEER_PAIRS: PASS; pairs=56;', output, re.M):
            raise ValueError('Eight-device pair summary count mismatch.')
        generations = re.findall(r'^HALO_GENERATION_([0-9]+): PASS;[^\n]*errors=0\s*$', output, re.M)
        if len(generations) != 3 or set(generations) != {'1', '2', '3'}:
            raise ValueError('Eight-device halo must pass all three generations.')
        if not re.search(r'^EIGHT_GPU_AUDIT: PASS; numerical_errors=0; cleanup=PASS\s*$', output, re.M):
            raise ValueError('Eight-device overall passing oracle is missing.')
        if re.search(r'(?<![A-Za-z_])(?:numerical_)?errors=[1-9][0-9]*', output) or re.search(r': FAIL\b', output):
            raise ValueError('Eight-device log reports errors.')
        lines += ['## Additional eight-GPU single-node audit', '',
            f'A separate harness passed on **eight A100-SXM4-40GB GPUs (CC 8.0)** at `{manifest["timestamp_utc"]}` UTC: **56 unique ordered peer-copy pairs**, **four NCCL collectives** (AllReduce, AllGather, ReduceScatter and grouped All-to-All), and **three halo generations**. Every numerical check reported zero errors; cleanup passed. This supplements the 145 book examples, many of which intentionally use only two or four devices.', '',
            linked(folder / source_name, 'Exact harness source') + ' · ' + linked(manifest_path, 'Execution manifest') + ' · ' + linked(folder / manifest['compile']['log'], 'Compiler output') + ' · ' + linked(folder / manifest['run']['log'], 'Runtime output') + ' · ' + linked(system_log, 'Eight-device system/topology') + '.', '',
            'Scope: single-node correctness with eight distinct CUDA-visible devices. This is not a multi-node RDMA test, bandwidth/latency benchmark or B200 result.', '']
    return lines


def execution_time(row):
    started = row.get('run_started_utc') or row.get('run', {}).get('started_utc')
    if started:
        return f'Executed UTC: `{started}`'
    return f'Execution record UTC ({row["timestamp_scope"]}): `{row["captured_utc"]}`'


def capture_block(row, dataset, label, relative):
    name = Path(row['source']).name
    output = row['_output'].rstrip('\n')
    fence = '`' * max(3, max([len(x) for x in re.findall(r'`+', output)] + [0]) + 1)
    commands = shlex.join(normalize(row['compile_command'], row['source'])) + '\n' + shlex.join(normalize(row['run_command'], row['source']))
    scope = category(row)
    return '\n\n' + '\n'.join([
        f'<!-- execution:start:{label}:{name} -->',
        f'> **Compiled and executed — {dataset}: {scope}.** Hardware: {row["evidence_system"].get("visible_gpu_count", "recorded")} {label.upper()} GPUs. {execution_time(row)}.', '',
        f'<!-- execution:metadata Source SHA-256: {row["sha256"]}; recorded UTC ({row["timestamp_scope"]}): {row["captured_utc"]} -->', '',
        f'[Exact tested source]({relative}/{name}) · [Compiler log]({relative}/compile.log) · [Execution log]({relative}/run.log). Compile exit: `{row["compile"]["exit_code"]}`; run exit: `{row["run"]["exit_code"]}`.', '',
        'Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.', '',
        '```bash', commands, '```', '',
        '**Sample output**', '', fence + 'text', output, fence, '',
        f'<!-- execution:end:{label}:{name} -->', ''])


def retire_obsolete_output(text):
    text = re.sub(r'\n*<!-- execution:provenance -->\n!!! info "Execution provenance"\n\n(?:    [^\n]*\n)+\n*', '\n\n', text)
    return re.sub(r'\n*<details\b[^>]*>\s*<summary>Historical[^<]*</summary>.*?</details>\s*', '\n\n', text, flags=re.S | re.I)


def locate(text, name, body):
    matches = [m for m in FENCES.finditer(text) if m[3].strip() == body.strip()]
    if len(matches) == 1:
        return matches[0]
    named = []
    for match in FENCES.finditer(text):
        filenames = re.findall(r'\b[A-Za-z_0-9]+\.(?:cu|cpp|c)\b', match[3])
        if filenames and filenames[0] == name and re.search(r'\bint\s+main\s*\(', match[3]):
            named.append(match)
    if len(named) != 1:
        raise ValueError(f'Expected one unambiguous complete source fence for {name}; found {len(named)}.')
    return named[0]


def prepare(inputs, dataset, label, root, temporary):
    current = sources(root)
    if len(current) != 145:
        raise ValueError(f'Expected 145 standalone sources, found {len(current)}.')
    candidates = {}
    environments = {}
    for index, directory in enumerate(inputs):
        directory = directory.resolve()
        system = json.loads((directory / 'system.json').read_text())
        if system.get('arch') != ARCHITECTURES[label]:
            raise ValueError(f'{label} architecture mismatch in {directory}.')
        run_id = f'run-{index + 1}'
        environments[run_id] = system
        for original in json.loads((directory / 'results.json').read_text()):
            rel = original['source']
            if rel in current and sha(current[rel]) == original['sha256']:
                row = dict(original)
                # Combined datasets can already identify their original system/run.
                row['run_id'] = original.get('run_id', run_id)
                row['evidence_system'] = original.get('evidence_system', system)
                if isinstance(row['evidence_system'], str):
                    row['evidence_system'] = json.loads((directory / row['evidence_system']).read_text())
                row['captured_utc'] = original.get('timestamp_utc') or original.get('captured_utc') or row['evidence_system']['utc']
                row['timestamp_scope'] = original.get('timestamp_scope') or ('program start' if original.get('timestamp_utc') else 'run-batch start; per-program time unrecorded')
                candidates[rel] = (row, directory)
    if set(candidates) != set(current):
        raise ValueError('Missing current-source evidence: ' + ', '.join(sorted(set(current) - set(candidates))))
    public = temporary / dataset
    public.mkdir()
    records = []
    for rel, (row, directory) in sorted(candidates.items()):
        name, stem = Path(rel).name, Path(rel).stem
        captured = directory / stem / name
        if not captured.is_file() or sha(captured) != row['sha256']:
            raise ValueError(f'Captured source mismatch: {rel}')
        if row.get('compile', {}).get('exit_code') != 0 or row.get('compile', {}).get('timed_out'):
            raise ValueError(f'Compilation did not succeed: {rel}')
        if row.get('status') not in SUCCESS or row.get('run', {}).get('exit_code') != 0 or row.get('run', {}).get('timed_out'):
            raise ValueError(f'No successful executed status: {rel}: {row.get("status")}')
        folder = public / stem
        folder.mkdir()
        for filename in (name, 'compile.log', 'run.log'):
            shutil.copyfile(directory / stem / filename, folder / filename)
        row['_output'] = strip_command((folder / 'run.log').read_text())
        records.append(row)
    # Preserve all supplied environment records and their raw probe outputs.
    for index, directory in enumerate(inputs):
        target = public / 'systems' / f'run-{index + 1}'
        target.mkdir(parents=True)
        for item in directory.iterdir():
            if item.is_file() and item.suffix in {'.json', '.log', '.txt', '.cu'}:
                shutil.copyfile(item, target / item.name)
        for metadata_name in ('systems', 'metadata'):
            metadata = directory / metadata_name
            if metadata.is_dir():
                for item in metadata.rglob('*'):
                    if item.is_file() and item.suffix in {'.json', '.log', '.txt', '.cu'}:
                        copied = target / metadata_name / item.relative_to(metadata)
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(item, copied)
    (public / 'environments.json').write_text(json.dumps(environments, indent=2) + '\n')
    docs = {}
    for row in records:
        rel = row['source']
        source = current[rel]
        name, body = source.name, source.read_text()
        options = []
        for page in source.parent.parent.glob('*.md'):
            text = retire_obsolete_output(docs.get(page, page.read_text()))
            if name in text or any(m[3].strip() == body.strip() for m in FENCES.finditer(text)):
                try:
                    match = locate(text, name, body)
                except ValueError:
                    continue
                options.append((page, text, match))
        if len(options) != 1:
            raise ValueError(f'Expected one chapter for {name}; found {len(options)}.')
        page, text, match = options[0]
        # Replace only this architecture's generated record; retain other datasets.
        marker = re.compile(r'\n*<!-- execution:start:' + re.escape(label + ':' + name) + r' -->.*?<!-- execution:end:' + re.escape(label + ':' + name) + r' -->\n*', re.S)
        text = marker.sub('\n', text)
        match = locate(text, name, body)
        canonical = match[1] + match[2] + '\n' + body.rstrip('\n') + '\n' + match[1]
        after = text[match.end():]
        # Current captured output replaces obsolete sample output. Source fences,
        # other architecture records and explanatory prose are retained.
        boundary = re.search(r'^#{2,6} |^!!!|^```(?:cpp|cuda|c\+\+|c)\s*$', after, re.M)
        limit = boundary.start() if boundary else len(after)
        old = next((m for m in FENCES.finditer(after[:limit]) if m[2].strip() in {'', 'text', 'console'}), None)
        if old and '<!-- execution:start:' not in after[:old.start()]:
            after = after[:old.start()] + after[old.end():]
            after = re.sub(r'^\s*(?:Genuinely compiled[^\n]*|Locked output[^\n]*|Sample output[^\n]*)\n', '\n', after)
        relative = Path(os.path.relpath(root / 'docs/validation' / dataset / Path(rel).stem, page.parent)).as_posix()
        text = text[:match.start()] + canonical + capture_block(row, dataset, label, relative) + after
        docs[page] = text
        row['page'] = str(page.relative_to(root))
        del row['_output']
    (public / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    counts = collections.Counter(category(row) for row in records)
    summary = ['# ' + label.upper() + ' execution validation', '',
               f'Dataset: **{dataset}**. All **145 current sources compiled and executed with exit status zero**. The categories below describe evidence scope; an API probe or a host model is not a successful numerical GPU workload.', '',
               '| Evidence category | Programs |', '| --- | ---: |']
    summary += [f'| {key} | {value} |' for key, value in sorted(counts.items())]
    summary += ['', f'[Execution manifest](validation/{dataset}/results.json) · [System records](validation/{dataset}/environments.json) · [Integrity inventory](validation/{dataset}/integrity.json). Each program’s chapter contains normalized commands, timestamp scope, source hash and captured output.', '',
                'Rows without a per-program timestamp explicitly use their recorded run-batch start UTC. Hardware and tool metadata remain attached to each original run; combining matching sources from reruns does not redate earlier executions.', '',
                'This dataset covers the captured hardware and single-node configurations. Host-only models validate their own modeled arithmetic, not real network performance. It does not establish multi-node behavior, failure recovery under physical faults, or benchmark throughput. H100, A100 and B200 evidence are published separately. A100 execution does not establish B200 compatibility or results.', '',
                '## Programs', '', '| Source | Evidence category | UTC | Output |', '| --- | --- | --- | --- |']
    for row in records:
        filename = Path(row['source']).name
        summary.append(f'| [{filename}]({row["page"][5:]}) | {category(row)} | `{row["captured_utc"]}` | [Log](validation/{dataset}/{Path(filename).stem}/run.log) |')
    programs_heading = summary.index('## Programs')
    summary[programs_heading:programs_heading] = supplements(root, public, label)
    summary_text = '\n'.join(summary) + '\n'
    integrity = {'dataset': dataset, 'label': label, 'source_count': 145,
                 'files': {str(p.relative_to(public)): sha(p) for p in sorted(public.rglob('*')) if p.is_file()},
                 'summary_sha256': hashlib.sha256(summary_text.encode()).hexdigest()}
    (public / 'integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')
    return public, docs, summary_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--dataset', required=True, help='Fresh immutable directory name, e.g. 2026-11-18-h100.')
    parser.add_argument('--label', choices=list(ARCHITECTURES), default='h100')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.dataset):
        parser.error('Dataset name must be a plain directory name.')
    destination = ROOT / 'docs/validation' / args.dataset
    summary = ROOT / 'docs' / f'validation-{args.label}.md'
    if destination.exists():
        parser.error('Dataset already exists; immutable evidence is never overwritten.')
    with tempfile.TemporaryDirectory(prefix='book2-publication-') as directory:
        try:
            prepared, docs, summary_text = prepare(args.input, args.dataset, args.label, ROOT, Path(directory))
            from verify_execution import verify
            errors = verify(ROOT, prepared, docs, summary_text)
            if errors:
                raise ValueError('; '.join(errors))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        shutil.copytree(prepared, destination)
        for page, text in docs.items():
            page.write_text(text)
        summary.write_text(summary_text)
        config = ROOT / 'mkdocs.yml'
        configuration = config.read_text()
        if summary.name not in configuration:
            anchor = '  - Getting Started: getting-started.md\n'
            if anchor not in configuration:
                raise ValueError('Add the generated validation summary to the MkDocs navigation.')
            configuration = configuration.replace(anchor, anchor + f'  - {args.label.upper()} Execution Validation: {summary.name}\n')
            config.write_text(configuration)
    print(f'Published 145 current-source records to {destination}; summary {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

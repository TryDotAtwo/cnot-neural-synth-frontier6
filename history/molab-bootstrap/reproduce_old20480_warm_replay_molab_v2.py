"""Foreground Molab reproduction of historical warm replay; no training.

Run only after live process inventory. Requires an isolated own source snapshot,
including code/, structured_field/ and the preserved historical run directory.
No benchmark, external solver or existing output is loaded as a teacher.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def check_word(record):
    rows = [1 << i for i in range(len(record.rows))]
    for control, target in record.certificate:
        if not (0 <= control < len(rows) and 0 <= target < len(rows)) or control == target:
            raise ValueError('illegal physical gate')
        rows[target] ^= rows[control]
    if tuple(rows) != record.rows or record.upper != len(record.certificate):
        raise ValueError('full encoder certificate mismatch')
    for control, target in reversed(record.certificate):
        rows[target] ^= rows[control]
    if rows != [1 << i for i in range(len(rows))]:
        raise ValueError('full reduction certificate mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--own-source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler-pins', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/marimo').is_dir():
        raise RuntimeError('Historical generation must run in Molab')
    root = args.own_source_root.resolve()
    run = root / 'structured_field/runs/replay_base_seed1700_v4'
    if args.output.exists():
        raise FileExistsError(args.output)
    pins = json.loads((run / 'source_hashes.json').read_bytes())
    for name, expected in pins.items():
        if hashlib.sha256((root / 'structured_field' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned source drift: ' + name)
    compiler_pins = json.loads(args.compiler_pins.read_bytes())
    required = {'code/shared_synth.py', 'code/stage16_reference.py',
                'code/global_gradient.py', 'code/exact_basis.py'}
    if set(compiler_pins) != required:
        raise ValueError('explicit complete compiler dependency pins required')
    for name, expected in compiler_pins.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('compiler dependency drift: ' + name)
    words_dir = args.output.with_suffix('.words')
    words_dir.mkdir(parents=False, exist_ok=False)
    sys.path.insert(0, str(root))
    import numpy as np
    from structured_field.data import exact_records, trajectory_records
    from structured_field.basis_transitions import build_macros
    from structured_field.gf2 import digest
    cfg = json.loads((run / 'config.json').read_bytes())
    seed = cfg['seed']
    started = time.perf_counter()
    validation, exact_train = [], {}
    for n in (3, 4):
        records = exact_records(n)
        order = np.random.default_rng(seed + n).permutation(len(records))
        cut = min(32, len(records) // 4)
        validation.extend(records[int(i)] for i in order[:cut])
        exact_train[n] = [records[int(i)] for i in order[cut:]][:2048]
    for n in cfg['sizes']:
        if n >= 8:
            macros = build_macros(n, count=2, seed=seed + 90000 + n, wide=True)
            validation.extend(trajectory_records(n, 6, seed=seed + 90000 + n, macros=macros))
    if [r.manifest() for r in validation] != json.loads((run / 'validation_manifest.json').read_bytes()):
        raise ValueError('historical validation manifest differs')
    for record in validation:
        check_word(record)
    def persist_words(name, records):
        path = words_dir / name
        with path.open('x', encoding='utf-8') as stream:
            json.dump([{'rows': list(r.rows), 'encoder': r.certificate,
                        'declared_exact': r.exact, 'provenance': r.provenance}
                       for r in records], stream)
        return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    validation_words = persist_words('validation.json', validation)
    forbidden = {digest(r.rows) for r in validation}
    evidence = []
    for path in sorted(run.glob('replay_*.json')):
        generation = int(path.stem.split('_')[1])
        pools = {n: [r for r in records if digest(r.rows) not in forbidden]
                 for n, records in exact_train.items()}
        for n in cfg['sizes']:
            if n >= 8:
                replay_seed = seed + 1000 * generation + n
                macros = build_macros(n, count=4, seed=replay_seed, wide=True)
                pool = trajectory_records(n, cfg['trajectories_per_size'], seed=replay_seed, macros=macros)
                pools[n] = [r for r in pool if digest(r.rows) not in forbidden]
        records = [r for pool in pools.values() for r in pool]
        if [r.manifest() for r in records] != json.loads(path.read_bytes()):
            raise ValueError('historical replay manifest differs: ' + path.name)
        for record in records:
            check_word(record)
        row = {'generation': generation, 'programs_verified': len(records),
               'physical_cnot_count': sum(len(r.certificate) for r in records),
               'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        row['full_words'] = persist_words(path.name, records)
        evidence.append(row)
        print(json.dumps(row), flush=True)
    report = {'schema': 'old20480_warm_replay_reproduction', 'generations': evidence,
              'compiler_dependency_pins': compiler_pins,
              'compiler_pin_scope': 'current audited dependencies; historical generator correspondence proven only if manifests match',
              'validation_words': validation_words,
              'validation_programs': len(validation), 'seconds': time.perf_counter() - started,
              'scope': 'Exact saved manifest correspondence and full certificates; not model execution, weight ancestry or independent exact-distance proof'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

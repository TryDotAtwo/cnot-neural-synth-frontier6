"""Source-gated v13 root audit using one read-only transaction including WAL.

The logical snapshot hash covers actual selected rows, not just the main DB
file. A live partial snapshot is never called a complete sealed holdout.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from ..data.stream_solver_dataset_v13 import (
    SCHEMA, SIZES, canonical, data_source_hashes, source_hashes, audit_split_units)
from ..data.stream_reader_v13 import decode_record


def audit(path, *, require_complete=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    try:
        connection.execute('BEGIN')
        metadata = dict(connection.execute('SELECT key,value FROM metadata'))
        manifest = json.loads(metadata['manifest'])
        if manifest['schema'] != SCHEMA or manifest['dataset_source_sha256'] != data_source_hashes():
            raise ValueError('v13 dataset source drift')
        if manifest['compiler_source_sha256'] != source_hashes():
            raise ValueError('own compiler source drift')
        rows = connection.execute('''SELECT matrix_hash,matrix_json,record_json,
            audit_json,suffix_offsets_json,root_digest,split,n,family,epoch,position
            FROM roots ORDER BY split,epoch,position''').fetchall()
        limits = connection.execute('SELECT COUNT(*) FROM generation_limits').fetchone()[0]
    finally:
        connection.close()
    records = []
    counts = Counter()
    families = Counter()
    population = Counter()
    snapshot = hashlib.sha256(canonical(manifest).encode())
    certificates = 0
    for row in rows:
        matrix_hash, identity, raw, producer_audit, offsets, digest, split, n, family, epoch, position = row
        expected = hashlib.sha256(canonical((identity,raw,producer_audit,offsets)).encode()).hexdigest()
        if digest != expected or matrix_hash != hashlib.sha256(identity.encode('ascii')).hexdigest():
            raise ValueError('stored root digest drift')
        record = decode_record(raw)
        if record.split != split or record.state.n != n or canonical(record.state.key) != identity:
            raise ValueError('stored root matrix/split drift')
        records.append(record)
        counts[(split,n)] += 1
        families[(split,family)] += 1
        population[(split,family,n)] += 1
        certificates += len(record.verified_words)
        snapshot.update(canonical(row).encode())
        snapshot.update(b'\n')
    expected = {'dev':manifest['expected_dev_roots'],'test':manifest['expected_test_roots']}
    actual = {split:sum(count for (owner,_),count in counts.items() if owner==split)
              for split in expected}
    seal = json.loads(metadata['holdout_seal']) if 'holdout_seal' in metadata else None
    complete = actual == expected and seal is not None
    if require_complete:
        if expected != {'dev':1792,'test':1792} or not complete:
            raise ValueError('complete normative1792 dev/test seal required')
        if any(counts[(split,n)] != 128 for split in expected for n in SIZES):
            raise ValueError('normative per-width holdout population drift')
        from ..data.stream_solver_dataset_v13 import FAMILIES
        if any(population[(split,family,n)] != 32
               for split in expected for family in FAMILIES for n in SIZES):
            raise ValueError('normative per-family per-width holdout population drift')
        if any(seal[split+'_roots'] != actual[split] for split in expected):
            raise ValueError('seal root count drift')
    return {'schema':'own_v13_readonly_full_root_audit_v1',
        'read_scope':'read_only_transaction_including_committed_WAL',
        'logical_snapshot_sha256':snapshot.hexdigest(), 'root_count':len(records),
        'certificates_replayed':certificates, 'source_verified':True,
        'counts':{f'{split}:{n}':count for (split,n),count in sorted(counts.items())},
        'families':{f'{split}:{family}':count for (split,family),count in sorted(families.items())},
        'complete_sealed_holdout':complete, 'generation_limit_count':limits,
        'split_units':audit_split_units(records),
        'scope':'root provenance/full-word replay; model exposure and full forbidden-bank audits separate'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    print(canonical(audit(args.db,require_complete=args.require_complete)),flush=True)


if __name__=='__main__':
    main()

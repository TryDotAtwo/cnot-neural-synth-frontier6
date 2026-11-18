"""Run one full-model RTX 3060 epoch with four sampled Qhat Bellman neighbors.

Sampling uses two lowest-cost Qhat actions and two deterministic exploration
actions. The original model, exact feature contexts, and optimizer remain intact.
Private detailed results stay on Vast; public status contains allowlisted gates.
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(os.environ.get('CNOT103_QHAT4_ROOT', '/workspace/cnot103-full3060'))
RUN_TAG = os.environ.get('CNOT103_QHAT4_RUN_TAG', 'manual')
if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', RUN_TAG):
    raise ValueError('invalid Qhat4 run tag')
OUT = ROOT / f'batched-probe-v18-qhat4-{RUN_TAG}'
V84_INPUT = ROOT / 'longrun-v84-20261117-r2' / 'longrun-input-v84'
SOURCE_ARCHIVE = V84_INPUT / 'proposal-source.tar.gz'
SOURCE_RECEIPT = V84_INPUT / 'proposal-source-receipt.json'
EXPECTED_SOURCE_SHA256 = '404004ac816b159705d904664f815f31ac8e2aee59d33fd9b3672af1eeb7ac4f'
EXPECTED_TEMPLATE_SHA256 = 'e8bd14165c8176fd602953f8db2fa93c38757e1b1610d24f5ff3e3af03120a94'
SRC = ROOT / f'qhat4-source-v84b-{RUN_TAG}'
TARGET = ROOT / f'qhat4-target-v84b-{RUN_TAG}'
RUN_CREATED = False
PUBLIC_STATUS = {
    'schema': 'qhat4_epoch_status_v1',
    'experiment': 'not_started',
    'epoch': 'not_started',
    'model_contract_gate': 'not_evaluated',
    'runtime_gate': 'not_evaluated',
    'qhat4_quality_gate': 'not_evaluated',
}


def write_public_status():
    if not RUN_CREATED:
        return
    tmp = OUT / 'public-status.json.tmp'
    tmp.write_text(json.dumps(PUBLIC_STATUS, sort_keys=True) + '\n')
    tmp.replace(OUT / 'public-status.json')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    global RUN_CREATED
    OUT.mkdir(exist_ok=False)
    RUN_CREATED = True
    PUBLIC_STATUS['experiment'] = 'running'
    write_public_status()
    template_path = ROOT / 'full-template.json'
    assert sha(template_path) == EXPECTED_TEMPLATE_SHA256
    original = json.loads(template_path.read_bytes())
    source_receipt = json.loads(SOURCE_RECEIPT.read_bytes())
    assert source_receipt['sha256'] == EXPECTED_SOURCE_SHA256
    assert source_receipt['bytes'] == SOURCE_ARCHIVE.stat().st_size
    assert sha(SOURCE_ARCHIVE) == EXPECTED_SOURCE_SHA256
    assert not SRC.exists() and not TARGET.exists()
    with tarfile.open(SOURCE_ARCHIVE, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names))
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        inv = {x['path']: x for x in json.load(tar.extractfile('inventory.json'))}
        assert set(inv) == {m.name.removeprefix('native/') for m in members if m.name != 'inventory.json'}
        SRC.mkdir()
        for m in members:
            if m.name == 'inventory.json':
                continue
            data = tar.extractfile(m).read()
            item = inv[m.name.removeprefix('native/')]
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            dst = SRC / m.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists(): assert dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest() == item['sha256']
            else: dst.write_bytes(data)
    spec = importlib.util.spec_from_file_location('followup', ROOT / 'followup.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    f.ROOT, f.OUT, f.NATIVE, f.MIXED = ROOT, OUT, ROOT, ROOT / 'mixed'
    f.rows = []
    f.machine = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
                                         '--format=csv,noheader'], text=True).strip()
    assert '3060' in f.machine and '8.6' in f.machine
    f.env = os.environ.copy()
    f.env.update(PATH=str(Path.home()/'.cargo/bin')+':'+f.env.get('PATH',''),
        LIBTORCH_USE_PYTORCH='1', CUDACXX='/usr/local/cuda/bin/nvcc', CUDA_PATH='/usr/local/cuda',
        CUDAToolkit_ROOT='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='8.6', CNOT103_CUDA_ARCHITECTURES='86',
        CARGO_BUILD_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2', CARGO_TARGET_DIR=str(TARGET),
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    )
    f.env['LIBTORCH'] = subprocess.check_output(['python', '-c',
        'import os,torch; print(os.path.dirname(torch.__file__))'], text=True).strip()
    f.env['LD_LIBRARY_PATH'] = f.env['LIBTORCH']+'/lib:/usr/local/cuda/lib64:'+f.env.get('LD_LIBRARY_PATH','')
    for key in ('NVIDIA_TF32_OVERRIDE', 'CNOT103_STAGE_PROFILE', 'CNOT103_FEATURE_STATUS_AUDIT',
                'CNOT103_BF16_BENCHMARK_PROBE', 'CNOT103_BENCHMARK_EVAL_SPEC',
                'CNOT103_BELLMAN_RECALL_AUDIT'):
        f.env.pop(key, None)
    def save():
        f.write(OUT/'summary.json', {'schema':'qhat4_exact_context_3060_epoch_v1', 'gpu':f.machine,
            'source_sha256':EXPECTED_SOURCE_SHA256, 'template_sha256':EXPECTED_TEMPLATE_SHA256,
            'rows':f.rows,
            'scope':'durable full-model one-epoch experiment, all 102 input slots and exact base feature contexts, original optimizer, Qhat sample-4 Bellman (two lowest predicted costs plus two deterministic exploration actions), batch4; dev is not test'})
    f.save = save
    save()
    cargo = str(Path.home()/'.cargo/bin/cargo')
    if not Path(cargo).exists():
        folder, code, wall, _ = f.command('install-rust', ['sh','-c',
            'curl --fail --silent --show-error --location --retry 5 https://sh.rustup.rs | sh -s -- -y --profile minimal'],
            f.env, 900, True)
        row = {'name':'install-rust', 'returncode':code, 'wall_seconds':wall,
               'complete':code == 0 and Path(cargo).exists()}
        if not row['complete']:
            row['error_tail'] = (folder/'stderr').read_text(errors='replace')[-2500:]
        f.rows.append(row); save()
        if not row['complete']: return
    manifest = str(SRC/'native/Cargo.toml')
    # Durable trainer uses only its manifest profile, no benchmark flags.
    for key in list(f.env):
        if key.startswith('CNOT103_') and key != 'CNOT103_CUDA_ARCHITECTURES': f.env.pop(key, None)
    assert 'CNOT103_TOP_ONLY_PROBE' not in f.env
    cmd=[cargo,'build','--locked','--release','--features','cuda','--manifest-path',manifest,
         '--bin','cnot103-train','--bin','cnot103-search']
    folder,code,wall,_=f.command('build-v84-full-model-qhat4',cmd,f.env,1200)
    row={'name':'build-v84-full-model-qhat4','returncode':code,'wall_seconds':wall,'complete':code==0}
    if code: row['error_tail']=(folder/'stderr').read_text(errors='replace')[-5000:]
    f.rows.append(row);save()
    if code: return
    bins=TARGET/'release'
    def config(data, attempts, batch, sample, cache=False):
        cfg = json.loads(json.dumps(original))
        dataset = ROOT/data/'dataset.json'
        cfg['dataset'] = {'path':str(dataset), 'sha256':sha(dataset)}
        cfg['resume'] = False
        cfg['training'].update(attempts=attempts, batch_size=batch, checkpoint_every_attempts=attempts)
        cfg['objective'].update(sampled_neighbor_count=sample, proposal_top_only=False, semi_gradient=False,
                                share_training_wire_projections_min_width=1,
                                prevalidated_device_closure_audit=True, cache_root_f08=cache)
        cfg['objective'].pop('sequential_batch_gradients', None)
        assert cfg['model'] == original['model'] and cfg['model']['structural_blocks'] == 8
        assert cfg['feature_contexts'] == original['feature_contexts']
        assert cfg['optimizer'] == original['optimizer']
        assert cfg['objective']['teacher_weight'] == original['objective']['teacher_weight']
        assert cfg['objective']['proposal_top_only'] is False and cfg['objective']['semi_gradient'] is False
        assert cfg['objective']['sampled_neighbor_count'] == 4 and cfg['training']['batch_size'] == 4
        return cfg
    dataset = ROOT/'full10000-dev32/dataset.json'
    receipt = json.loads((ROOT/'full10000-dev32/receipt.json').read_bytes())
    assert sha(dataset)==receipt['sha256']=='eb71744c2afcef922596bc9956abfd8390a661273fa6461dd277a04fd85e9b86'
    def durable_config(data, attempts, ckpt, resume=False, target=None):
        cfg=config(data,attempts,4,4,cache=True)
        cfg['resume']=resume
        cfg['training']['checkpoint_every_attempts']=attempts
        cfg['training']['fast_cuda']={'batch4_max_width':108,'batch2_max_width':144}
        if target is not None: cfg['training']['target_completed_epochs']=target
        cfg['checkpoint_dir']=str(ckpt)
        assert cfg['model'] == original['model']
        assert cfg['feature_contexts'] == original['feature_contexts']
        return cfg
    def checkpoint_info(directory, report):
        pointer=json.loads((directory/'last.json').read_bytes())
        assert pointer==report['checkpoint']
        path=directory/pointer['file']
        assert path.is_file() and sha(path)==pointer['sha256']
        with path.open('rb') as stream:
            assert stream.read(8)==b'C103QCP1'
            size=int.from_bytes(stream.read(8),'little')
            assert 0<size<16*1024*1024
            header=json.loads(stream.read(size))
        assert header['contract']['cuda_execution']=={'id':'grouped_bf16_fused_adamw_cpu_backup_v1',
            'batch_size':4,'batch4_max_width':108,'batch2_max_width':144}
        assert header['progress']==report['end_progress']
        return {'sha256':pointer['sha256'],'bytes':path.stat().st_size,'path':str(path),
                'sampler_cursor':header['sampler']['cursor'],'sampler_epochs':header['sampler']['epochs']}
    def parse_stage_profile(path):
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return None
        profiles = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get('schema') == 'cnot103_stage_profile_v1':
                profiles.append(value)
        if len(profiles) != 1:
            return None
        profile = profiles[0]
        number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        stages = profile.get('stages')
        if (not number(profile.get('wall_ms')) or not number(profile.get('unattributed_ms'))
                or not isinstance(stages, dict) or not stages):
            return None
        clean_stages = {}
        for name, stage in stages.items():
            if (not isinstance(name, str) or not isinstance(stage, dict)
                    or not isinstance(stage.get('calls'), int) or isinstance(stage.get('calls'), bool)
                    or stage['calls'] < 0 or not number(stage.get('exclusive_ms'))
                    or not number(stage.get('inclusive_ms'))):
                return None
            clean_stages[name] = {
                'calls': stage['calls'],
                'exclusive_ms': stage['exclusive_ms'],
                'inclusive_ms': stage['inclusive_ms'],
            }
        chunk_start = profile.get('candidate_chunk_start_index')
        if not isinstance(chunk_start, int) or isinstance(chunk_start, bool) or chunk_start < 0:
            return None
        chunk_times = profile.get('candidate_chunk_seconds')
        if not isinstance(chunk_times, list) or not all(number(value) for value in chunk_times):
            return None
        chunk_probe = profile.get('candidate_chunk_probe')
        if chunk_probe is not None:
            if (not isinstance(chunk_probe, dict)
                    or not isinstance(chunk_probe.get('limit'), int)
                    or isinstance(chunk_probe.get('limit'), bool) or chunk_probe['limit'] < 0
                    or not isinstance(chunk_probe.get('completed'), int)
                    or isinstance(chunk_probe.get('completed'), bool) or chunk_probe['completed'] < 0):
                return None
            chunk_probe = {'limit': chunk_probe['limit'], 'completed': chunk_probe['completed']}
        return {
            'schema': 'cnot103_stage_profile_v1',
            'wall_ms': profile['wall_ms'],
            'unattributed_ms': profile['unattributed_ms'],
            'candidate_chunk_probe': chunk_probe,
            'candidate_chunk_start_index': chunk_start,
            'candidate_chunk_seconds': chunk_times,
            'stages': clean_stages,
        }

    def train(label,cfg,timeout,expect_success=True,run_env=None,capture_stage_profile=False):
        manifests=OUT/'durable-manifests';manifests.mkdir(exist_ok=True)
        manifest_path=manifests/(label+'.json');f.write(manifest_path,cfg)
        command_env = f.env if run_env is None else run_env
        folder,code,wall,samples=f.command(label,[str(bins/'cnot103-train'),str(manifest_path)],command_env,timeout,True)
        row={'name':label,'returncode':code,'wall_seconds':wall,'manifest_sha256':sha(manifest_path),
             'binary_sha256':sha(bins/'cnot103-train'),
             'gpu_peak_sampled_mib':max((s['memory_mib'] for s in samples),default=None),
             'gpu_mean_sampled_util':sum(s['util_percent'] for s in samples)/len(samples) if samples else None}
        if capture_stage_profile:
            profile = parse_stage_profile(folder/'stderr')
            row['stage_profile_status'] = 'captured' if profile is not None else 'missing_or_invalid'
            if profile is not None:
                row['stage_profile'] = profile
        stderr=(folder/'stderr').read_text(errors='replace')
        if not expect_success:
            row['complete']=code!=0 and 'checkpoint contract differs' in stderr.lower()
            row['error_tail']=stderr[-1000:]
        elif code==0:
            report=json.loads((folder/'stdout').read_bytes())
            assert report['feature_model_inputs']==102 and report['exposure_audit']['strict_unseen']
            assert report['dataset_sha256']==cfg['dataset']['sha256']
            assert report['training_runtime']=='cuda_qhat_sampled_bellman_v1'
            row.update(complete=True,report=report,checkpoint_info=checkpoint_info(Path(cfg['checkpoint_dir']),report))
        else:
            row['complete']=False;row['error_tail']=stderr[-5000:]
        f.rows.append(row);save()
        return row,manifest_path,folder

    def dev_search(label,training_manifest,training_folder,shortlist):
        cfg={'schema':'cnot103_dev_search_v1',
             'training_manifest':{'path':str(training_manifest),'sha256':sha(training_manifest)},
             'training_result':{'path':str(training_folder/'stdout'),'sha256':sha(training_folder/'stdout')},
             'dev_indices':receipt['added_dev_indices'], 'head':'qhat',
             'search':{'beam_width':4,'shortlist':shortlist,'max_depth':8,
                       'max_expanded':64,'inference_batch':8}}
        path=OUT/'durable-manifests'/(label+'.json');f.write(path,cfg)
        folder,code,wall,samples=f.command(label,[str(bins/'cnot103-search'),str(path)],f.env,1200,True)
        row={'name':label,'returncode':code,'wall_seconds':wall,'manifest_sha256':sha(path)}
        if code==0:
            report=json.loads((folder/'stdout').read_bytes())
            answers=report['rows']; assert len(answers)==32 and all(a['dimension']==8 for a in answers)
            for a in answers: f.replay(a)
            row.update(complete=True,solved=sum(a['status']=='solved' for a in answers),total=len(answers),
                       solved_indices=[a['dev_index'] for a in answers if a['status']=='solved'],
                       checkpoint_sha256=report['checkpoint']['sha256'])
        else: row.update(complete=False,error_tail=(folder/'stderr').read_text(errors='replace')[-5000:])
        f.rows.append(row);save();return row

    full=OUT/'checkpoint-full'
    epoch,epoch_manifest,epoch_folder=train('full-qhat4-epoch1',
        durable_config('full10000-dev32',10000,full,target=1), 4200)
    if not epoch['complete']:
        PUBLIC_STATUS.update(experiment='failed', epoch='failed')
        write_public_status()
        raise RuntimeError('the full 10,000-row epoch did not complete')
    report=epoch['report'];progress=report['epoch_progress']
    assert report['model_parameters']==136421807 and report['train_rows']==10000
    assert progress['completed']==1 and progress['cursor']==10000 and progress['target_reached']
    assert report['exposure_audit']['label_count']==10000
    epoch['measured_epoch_wall_seconds']=epoch['wall_seconds']
    epoch['projected_days_100_equal_epochs']=epoch['wall_seconds']*100/86400
    PUBLIC_STATUS.update(epoch='complete', model_contract_gate='pass',
        runtime_gate='pass' if epoch['wall_seconds'] <= 1728 else 'fail')
    write_public_status()
    save()
    qhat4=dev_search('dev32-qhat4-shortlist4',epoch_manifest,epoch_folder,4)
    top8=dev_search('dev32-qhat4-shortlist8',epoch_manifest,epoch_folder,8)
    all56=dev_search('dev32-qhat4-all56',epoch_manifest,epoch_folder,56)
    if not qhat4['complete'] or not top8['complete'] or not all56['complete']:
        PUBLIC_STATUS.update(experiment='failed')
        write_public_status()
        raise RuntimeError('one or more development searches did not complete')
    quality_gate = all56['solved'] >= 16 and qhat4['solved'] >= all56['solved'] - 1
    PUBLIC_STATUS.update(experiment='complete',
        qhat4_quality_gate='pass' if quality_gate else 'fail')
    write_public_status()
    profile_label = 'stage-profile-qhat4-64-updates'
    profile_env = dict(f.env)
    profile_env['CNOT103_STAGE_PROFILE'] = '1'
    try:
        profile_row, _, _ = train(
            profile_label,
            durable_config('full10000-dev32',64,OUT/'checkpoint-stage-profile-64'),
            600,
            run_env=profile_env,
            capture_stage_profile=True,
        )
        if not profile_row.get('complete'):
            profile_row['stage_profile_status'] = 'run_failed'
            save()
    except Exception:
        if not any(row.get('name') == profile_label for row in f.rows):
            f.rows.append({
                'name': profile_label,
                'complete': False,
                'stage_profile_status': 'run_failed',
            })
            try:
                save()
            except Exception:
                pass
    f.status('finished')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        if RUN_CREATED:
            PUBLIC_STATUS['experiment'] = 'failed'
            write_public_status()
        raise
    finally:
        if RUN_CREATED:
            (OUT/'finished.txt').write_text(str(time.time())+'\n')


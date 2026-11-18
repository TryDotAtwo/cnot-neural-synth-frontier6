"""Bounded GitHub Actions controller for one existing Vast RTX 3060 instance.

Only allowlisted run gates are written to Actions output. Billing rows,
training logs, checkpoints, and development-set counts stay off Actions logs.
"""
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

INSTANCE_LABELS_BY_ID = {
    61081857: {
        'cnot103-longrun-original-3060-v84',
        'cnot103-longrun-original-3060-v84-r2',
    },
    60932086: {
        'cnot103-original-full-rtx3060-20261116-v3',
    },
}
SOURCE_INSTANCE_ID = 61081857
OWNED_IDS = {
    60488802, 60491053, 60509722, 60522214, 60585616, 60586973, 60588069,
    60621205, 60622406, 60626596, 60642345, 60646895, 60654590, 60658156,
    60660708, 60667533, 60674887, 60694337, 60695154, 60729505, 60745475,
    60749603, 60917228, 60919132, 60932086, 61081857,
}
REQUESTED_INSTANCE_ID = os.environ.get('VAST_INSTANCE_ID', '61081857').strip()
SELECTED_INSTANCE_ID = os.environ.get('VAST_SELECTED_INSTANCE_ID', '').strip()
SELECTED_INSTANCE_LABEL = os.environ.get('VAST_SELECTED_INSTANCE_LABEL', '').strip()
if SELECTED_INSTANCE_ID:
    if not re.fullmatch(r'[0-9]{1,20}', SELECTED_INSTANCE_ID):
        raise ValueError('invalid selected existing instance id')
    INSTANCE_ID = int(SELECTED_INSTANCE_ID)
    if INSTANCE_ID not in OWNED_IDS:
        raise ValueError('selected instance is outside the owned inventory')
    if SELECTED_INSTANCE_LABEL:
        if not re.fullmatch(r'(?i)cnot103[a-z0-9_.-]{1,120}', SELECTED_INSTANCE_LABEL):
            raise ValueError('invalid selected existing instance label')
        if INSTANCE_ID in INSTANCE_LABELS_BY_ID and SELECTED_INSTANCE_LABEL not in INSTANCE_LABELS_BY_ID[INSTANCE_ID]:
            raise ValueError('selected existing instance label does not match its fixed identity')
        INSTANCE_LABELS = {SELECTED_INSTANCE_LABEL}
    elif INSTANCE_ID in INSTANCE_LABELS_BY_ID:
        INSTANCE_LABELS = INSTANCE_LABELS_BY_ID[INSTANCE_ID]
    else:
        raise ValueError('selected existing instance label is missing')
elif REQUESTED_INSTANCE_ID == 'auto':
    INSTANCE_ID = 0
    INSTANCE_LABELS = set()
else:
    try:
        INSTANCE_ID = int(REQUESTED_INSTANCE_ID)
    except ValueError:
        raise ValueError('invalid fixed RTX 3060 instance') from None
    if INSTANCE_ID not in INSTANCE_LABELS_BY_ID:
        raise ValueError('unsupported fixed RTX 3060 instance')
    INSTANCE_LABELS = INSTANCE_LABELS_BY_ID[INSTANCE_ID]
SOURCE_INSTANCE_LABELS = INSTANCE_LABELS_BY_ID[SOURCE_INSTANCE_ID]
VAST_ROOT = pathlib.Path(os.environ.get('CNOT103_QHAT4_ROOT', '/workspace/cnot103-full3060'))
SOURCE_ROOT = pathlib.Path('/workspace/cnot103-full3060')
TRAINER = '.github/scripts/train_qhat4_epoch.py'
EXPECTED_DATASET_SHA256 = 'eb71744c2afcef922596bc9956abfd8390a661273fa6461dd277a04fd85e9b86'
EXPECTED_FULL_TEMPLATE_SHA256 = 'e8bd14165c8176fd602953f8db2fa93c38757e1b1610d24f5ff3e3af03120a94'
EXPECTED_PROPOSAL_SOURCE_SHA256 = '404004ac816b159705d904664f815f31ac8e2aee59d33fd9b3672af1eeb7ac4f'
VAST_API = 'https://console.vast.ai/api/v0'


class GateError(RuntimeError):
    pass


def env_value(name):
    value = os.environ.get(name, '').strip()
    if not value:
        raise GateError('required runner configuration is missing')
    return value


def run_tag():
    value = f'{env_value("GITHUB_RUN_ID")}-{env_value("GITHUB_RUN_ATTEMPT")}'
    if not re.fullmatch(r'[0-9]{1,20}-[0-9]{1,10}', value):
        raise GateError('workflow run identifier is invalid')
    return value


def vast(args, timeout=60):
    key = env_value('VAST_API_KEY')
    try:
        proc = subprocess.run(
            ['vastai', '--api-key', key, *args], capture_output=True,
            timeout=timeout, check=False,
        )
        if proc.returncode:
            raise GateError('Vast query did not succeed')
        return json.loads(proc.stdout.decode('utf-8'))
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, json.JSONDecodeError):
        raise GateError('Vast query could not be verified') from None


def start_response_summary(status, payload):
    if not isinstance(payload, dict):
        return {
            'start_api_http_status': status,
            'start_api_success': 'unknown',
            'start_api_error': 'unparsed',
            'start_api_message_present': 'unknown',
        }, False

    error_code = payload.get('error')
    known_errors = {
        'auth_error', 'invalid_args', 'label_too_long', 'missing_billing',
        'no_such_instance', 'not_found', 'rate_limit_exceeded',
    }
    safe_error = (
        error_code if isinstance(error_code, str) and
        (error_code in known_errors or re.fullmatch(r'[A-Za-z0-9_]{1,64}', error_code))
        else 'other' if error_code else 'none'
    )
    message = payload.get('msg') or payload.get('detail')
    summary = {
        'start_api_http_status': status,
        'start_api_success': 'true' if payload.get('success') is True else 'false' if payload.get('success') is False else 'unknown',
        'start_api_error': safe_error,
        'start_api_message_present': 'true' if isinstance(message, str) and bool(message.strip()) else 'false',
    }
    return summary, payload.get('success') is True


def vast_start_instance(timeout=75):
    key = env_value('VAST_API_KEY')
    request = urllib.request.Request(
        f'{VAST_API}/instances/{INSTANCE_ID}/',
        data=json.dumps({'state': 'running'}).encode('utf-8'),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        method='PUT',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise GateError('Vast start request was rejected')
            body = response.read()
            try:
                payload = json.loads(body.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            summary, accepted = start_response_summary(response.status, payload)
            print('start_api_result=' + json.dumps(summary, sort_keys=True, separators=(',', ':')))
            write_output(summary)
            if not accepted:
                raise GateError('Vast start response did not confirm success')
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
            payload = json.loads(body.decode('utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        summary, _ = start_response_summary(error.code, payload)
        print('start_api_result=' + json.dumps(summary, sort_keys=True, separators=(',', ':')))
        write_output(summary)
        if 400 <= error.code < 500:
            raise GateError('Vast start request was rejected') from None
        raise GateError('Vast start request timed out or was not confirmed') from None
    except (urllib.error.URLError, TimeoutError):
        raise GateError('Vast start request timed out or was not confirmed') from None


def get_instance(instance_id=None, expected_labels=None):
    ident = INSTANCE_ID if instance_id is None else int(instance_id)
    labels = INSTANCE_LABELS if expected_labels is None else expected_labels
    rows = vast(['show', 'instances', '--raw'])
    if not isinstance(rows, list):
        raise GateError('Vast instance list has an unknown shape')
    matches = [x for x in rows if str(x.get('id')) == str(ident)]
    if len(matches) != 1 or matches[0].get('label') not in labels:
        raise GateError('the expected existing instance was not found exactly once')
    return matches[0]


def copy_container_path(source_id, source_path, destination_id, destination_path, timeout=600):
    try:
        proc = subprocess.run(
            ['vastai', '--api-key', env_value('VAST_API_KEY'), 'copy',
             f'C.{source_id}:{source_path}', f'C.{destination_id}:{destination_path}'],
            capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise GateError('the pinned inactive-instance data copy did not finish') from None
    if proc.returncode:
        raise GateError('the pinned inactive-instance data copy did not pass')


def select_existing_target():
    global INSTANCE_ID, INSTANCE_LABELS
    if REQUESTED_INSTANCE_ID != 'auto' or SELECTED_INSTANCE_ID:
        return
    rows = vast(['show', 'instances', '--raw'])
    if not isinstance(rows, list):
        raise GateError('Vast instance list has an unknown shape')
    source_rows = [row for row in rows if isinstance(row, dict)
                   and str(row.get('id')) == str(SOURCE_INSTANCE_ID)
                   and row.get('label') in SOURCE_INSTANCE_LABELS]
    if len(source_rows) != 1:
        raise GateError('the pinned source instance was not found exactly once')
    source = source_rows[0]
    if source.get('actual_status') not in ('stopped', 'exited') or source.get('intended_status') != 'stopped':
        raise GateError('the pinned source instance is not stopped')
    rejected_start_ids = {60932086, SOURCE_INSTANCE_ID}
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            ident = int(row.get('id'))
        except (TypeError, ValueError):
            continue
        label = row.get('label')
        gpu = row.get('gpu_name')
        if (ident not in OWNED_IDS or ident in rejected_start_ids
                or not isinstance(label, str)
                or not re.fullmatch(r'(?i)cnot103[a-z0-9_.-]{1,120}', label)
                or not isinstance(gpu, str) or '3060' not in gpu.casefold()):
            continue
        if row.get('actual_status') not in ('stopped', 'exited') or row.get('intended_status') != 'stopped':
            continue
        rate = row.get('dph_total')
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            continue
        same_machine = (source.get('machine_id') is not None
                        and row.get('machine_id') is not None
                        and str(source.get('machine_id')) == str(row.get('machine_id')))
        source_host, target_host = source.get('host_id'), row.get('host_id')
        source_location, target_location = source.get('geolocation'), row.get('geolocation')
        same_local_network = (
            source_host is not None and target_host is not None
            and str(source_host) == str(target_host)
            and isinstance(source_location, str) and isinstance(target_location, str)
            and source_location.strip().casefold() == target_location.strip().casefold()
        )
        priority = 0 if same_machine else 1 if same_local_network else 2
        candidates.append((priority, float(rate), ident, label))
    skip_rank_raw = os.environ.get('VAST_AUTO_TARGET_SKIP_RANK', '').strip()
    if skip_rank_raw:
        if not re.fullmatch(r'[0-9]{1,2}', skip_rank_raw):
            raise GateError('automatic target skip rank is invalid')
        skip_rank = int(skip_rank_raw)
        candidates.sort()
        if 1 <= skip_rank <= len(candidates):
            del candidates[skip_rank - 1]
        else:
            candidates.clear()
    if not candidates:
        print('target_selection=none')
        raise GateError('no unattempted stopped owned RTX 3060 instance is available')
    _, _, ident, label = min(candidates)
    INSTANCE_ID = ident
    INSTANCE_LABELS = {label}
    envfile = os.environ.get('GITHUB_ENV')
    if envfile:
        with open(envfile, 'a', encoding='utf-8') as stream:
            stream.write(f'VAST_SELECTED_INSTANCE_ID={ident}\n')
            stream.write(f'VAST_SELECTED_INSTANCE_LABEL={label}\n')
    print('target_selection=selected existing_rtx3060=1')


def prepare_fallback_inputs():
    if INSTANCE_ID == SOURCE_INSTANCE_ID:
        return
    target = get_instance()
    source = get_instance(SOURCE_INSTANCE_ID, SOURCE_INSTANCE_LABELS)
    for row in (source, target):
        if row.get('actual_status') not in ('stopped', 'exited') or row.get('intended_status') != 'stopped':
            raise GateError('source and destination must both be stopped for file copy')
    source_machine = source.get('machine_id')
    target_machine = target.get('machine_id')
    same_machine = (source_machine is not None and target_machine is not None
                    and str(source_machine) == str(target_machine))
    source_host, target_host = source.get('host_id'), target.get('host_id')
    source_location, target_location = source.get('geolocation'), target.get('geolocation')
    same_local_network = (
        source_host is not None and target_host is not None
        and str(source_host) == str(target_host)
        and isinstance(source_location, str) and isinstance(target_location, str)
        and source_location.strip().casefold() == target_location.strip().casefold()
    )
    if same_machine:
        transfer_mode = 'same_machine'
    elif same_local_network:
        transfer_mode = 'same_local_network'
    else:
        transfer_mode = 'internet'
        rate = target.get('dph_total')
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            raise GateError('cross-host transfer budget rate is unavailable')
        if project_charges() + 3.5 * float(rate) + 5.0 >= 20.0:
            raise GateError('budget gate lacks cross-host transfer reserve')
    print(f'input_copy_preflight=pass transfer_mode={transfer_mode}')
    expected_root = pathlib.Path(f'/workspace/qhat4-recovery-{run_tag()}')
    if VAST_ROOT != expected_root:
        raise GateError('fallback input path is not this workflow run directory')

    copy_units = [
        ('/workspace/cnot103-full3060/full10000-dev32', f'{VAST_ROOT}/'),
        ('/workspace/cnot103-full3060/full-template.json', f'{VAST_ROOT}/full-template.json'),
        ('/workspace/cnot103-full3060/followup.py', f'{VAST_ROOT}/followup.py'),
        ('/workspace/cnot103-full3060/longrun-v84-20261117-r2/longrun-input-v84', f'{VAST_ROOT}/'),
    ]
    for source_path, destination_path in copy_units:
        copy_container_path(SOURCE_INSTANCE_ID, source_path, INSTANCE_ID, destination_path)
    if transfer_mode == 'internet':
        rate = float(target['dph_total'])
        if project_charges() + 3.5 * rate + 1.5 >= 20.0:
            raise GateError('budget gate after cross-host transfer did not pass')
    print(f'input_copy=accepted units={len(copy_units)} transfer_mode={transfer_mode}')

def get_api_instance():
    request = urllib.request.Request(
        f'{VAST_API}/instances/',
        headers={'Authorization': f'Bearer {env_value("VAST_API_KEY")}',
                 'Accept': 'application/json',
                 'User-Agent': 'TheQuantumProject-GitHub-Actions/1.0'},
        method='GET',
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        raise GateError('Vast instance status could not be verified') from None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get('instances'), list):
        rows = payload['instances']
    else:
        raise GateError('Vast instance response has an unknown shape')
    matches = [x for x in rows if isinstance(x, dict) and str(x.get('id')) == str(INSTANCE_ID)]
    if len(matches) != 1:
        raise GateError('the expected existing instance was not found exactly once')
    return matches[0]


def ssh_endpoint(instance):
    host = instance.get('public_ipaddr')
    ports = instance.get('ports')
    entries = ports.get('22/tcp') if isinstance(ports, dict) else None
    port = entries[0].get('HostPort') if isinstance(entries, list) and entries and isinstance(entries[0], dict) else None
    host_parts = host.split('.') if isinstance(host, str) else []
    valid_host = len(host_parts) == 4 and all(
        part.isdigit() and 0 <= int(part) <= 255 for part in host_parts
    )
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 0
    if not valid_host or not 1 <= port <= 65535:
        return None
    return host, port


def project_charges():
    start = dt.date(2026, 11, 13)
    end = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    args = [
        'show', 'invoices-v1', '--charges', '--charge-type', 'instance',
        '--start-date', start.isoformat(), '--end-date', end.isoformat(),
        '--limit', '100', '--raw',
    ]
    records, token = [], None
    for _ in range(100):
        page_args = args + (['--next-token', token] if token else [])
        page = vast(page_args, timeout=70)
        if not isinstance(page, dict) or not isinstance(page.get('results'), list):
            raise GateError('Vast billing response has an unknown shape')
        records.extend(page['results'])
        token = page.get('next_token')
        if token is None or token == '':
            break
        if not isinstance(token, str):
            raise GateError('Vast billing pagination token is invalid')
    else:
        raise GateError('Vast billing pagination did not finish')

    total = 0.0
    for row in records:
        description = str(row.get('description') or '')
        match = re.match(r'^Instance (\d+) Charges(?:\s|$)', description)
        label = str((row.get('metadata') or {}).get('label') or '').lower()
        if not ((match and int(match.group(1)) in OWNED_IDS) or label.startswith('cnot103')):
            continue
        amount = row.get('amount')
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise GateError('Vast billing row has an invalid amount')
        total += float(amount)
    return total


def write_output(values):
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            for key, value in values.items():
                if '\n' in str(value) or '\r' in str(value):
                    raise GateError('invalid workflow output')
                stream.write(f'{key}={value}\n')


def preflight():
    configured_id = env_value('VAST_INSTANCE_ID')
    if configured_id == 'auto':
        if INSTANCE_ID not in OWNED_IDS or INSTANCE_ID in {0, SOURCE_INSTANCE_ID}:
            raise GateError('no authorized existing target was selected')
        if SELECTED_INSTANCE_ID and int(SELECTED_INSTANCE_ID) != INSTANCE_ID:
            raise GateError('selected existing instance id changed unexpectedly')
    elif int(configured_id) != INSTANCE_ID:
        raise GateError('configured instance id does not match the authorized instance')
    instance = get_instance()
    if instance.get('actual_status') not in ('exited', 'stopped') or instance.get('intended_status') != 'stopped':
        raise GateError('instance is not in the expected stopped state')
    rate = instance.get('dph_total')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise GateError('instance hourly rate is unavailable')
    spent = project_charges()
    if spent + 3.5 * float(rate) + 1.5 >= 20.0:
        raise GateError('the $20 Vast budget gate did not pass')
    return instance


def attach_ephemeral_key(pubkey):
    ident = INSTANCE_ID
    request = urllib.request.Request(
        f'{VAST_API}/instances/{ident}/ssh',
        data=json.dumps({'ssh_key': pubkey.strip()}).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {env_value("VAST_API_KEY")}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status < 200 or response.status >= 300:
                raise GateError('ephemeral SSH key attachment was not confirmed')
            response.read()
    except (urllib.error.URLError, TimeoutError):
        raise GateError('ephemeral SSH key attachment was not confirmed') from None


def key_paths():
    runner_temp = pathlib.Path(env_value('RUNNER_TEMP'))
    return runner_temp / 'vast-qhat4-ed25519', runner_temp / 'vast-qhat4-known-hosts'


def generate_key():
    private, _ = key_paths()
    if private.exists() or private.with_suffix('.pub').exists():
        raise GateError('temporary SSH key path already exists')
    try:
        subprocess.run(
            ['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', 'github-vast-qhat4-epoch', '-f', str(private)],
            check=True, capture_output=True, timeout=20,
        )
        pubkey = private.with_suffix('.pub').read_text(encoding='ascii').strip()
        if not pubkey.startswith('ssh-ed25519 '):
            raise GateError('temporary SSH key was invalid')
        return pubkey
    except (OSError, subprocess.SubprocessError):
        raise GateError('temporary SSH key could not be generated') from None


def mark_started():
    envfile = os.environ.get('GITHUB_ENV')
    if envfile:
        with open(envfile, 'a', encoding='utf-8') as stream:
            stream.write('VAST_STARTED_BY_WORKFLOW=true\n')


def mark_key_attached():
    envfile = os.environ.get('GITHUB_ENV')
    if envfile:
        with open(envfile, 'a', encoding='utf-8') as stream:
            stream.write('VAST_EPHEMERAL_KEY_ATTACHED=true\n')


def start():
    select_existing_target()
    preflight()
    prepare_fallback_inputs()
    pubkey = generate_key()
    try:
        vast_start_instance()
    except GateError as error:
        if 'timed out' in str(error):
            try:
                state = get_instance()
                if state.get('intended_status') == 'running':
                    mark_started()
            except GateError:
                pass
        raise
    mark_started()
    deadline = time.monotonic() + 15 * 60
    allowed_states = {
        'running', 'stopped', 'starting', 'stopping', 'scheduling',
        'loading', 'created', 'offline', 'unknown', 'exited', 'error',
    }
    last_actual = 'unknown'
    last_intended = 'unknown'
    last_endpoint = 'not_checked'
    while time.monotonic() < deadline:
        state = get_instance()
        actual = state.get('actual_status')
        intended = state.get('intended_status')
        last_actual = actual.strip().lower() if isinstance(actual, str) and actual.strip().lower() in allowed_states else 'other'
        last_intended = intended.strip().lower() if isinstance(intended, str) and intended.strip().lower() in allowed_states else 'other'
        if state.get('actual_status') == 'running' and state.get('intended_status') == 'running':
            endpoint = ssh_endpoint(get_api_instance())
            last_endpoint = 'available' if endpoint else 'unavailable'
            if endpoint:
                mark_key_attached()
                attach_ephemeral_key(pubkey)
                write_output({'budget_gate': 'pass', 'start_gate': 'pass'})
                print('start_gate=pass')
                return
        time.sleep(10)
    diagnostic = {
        'start_actual_status': last_actual,
        'start_intended_status': last_intended,
        'start_ssh_endpoint': last_endpoint,
    }
    write_output(diagnostic)
    print('start_timeout=' + json.dumps(diagnostic, sort_keys=True, separators=(',', ':')))
    raise GateError('the existing instance did not start within the bounded window')


def ssh_command(host, port, remote='bash -s'):
    private, known_hosts = key_paths()
    return [
        'ssh', '-i', str(private), '-p', str(port),
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', f'UserKnownHostsFile={known_hosts}',
        f'root@{host}', remote,
    ]


def remote_run(script, state):
    endpoint = ssh_endpoint(get_api_instance())
    if not endpoint:
        raise GateError('instance SSH endpoint is unavailable')
    host, port = endpoint
    for attempt in range(12):
        try:
            proc = subprocess.run(
                ssh_command(host, port), input=script.encode('utf-8'),
                capture_output=True, timeout=25, check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.decode('utf-8', errors='strict').strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        if attempt < 11:
            time.sleep(5)
    raise GateError('remote preflight did not pass')


def run_experiment(state):
    tag = run_tag()
    expected_root = SOURCE_ROOT if INSTANCE_ID == SOURCE_INSTANCE_ID else pathlib.Path(f'/workspace/qhat4-recovery-{tag}')
    if VAST_ROOT != expected_root:
        raise GateError('experiment root does not match the authorized instance and run')
    pubkey = pathlib.Path(str(key_paths()[0]) + '.pub').read_text(encoding='ascii').strip()
    pre = r'''set -eu
ROOT=__QHAT4_ROOT__
ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}"
ID="${ID#C.}"
test "$ID" = "__INSTANCE_ID__"
test -n "${CONTAINER_API_KEY:-}"
if pgrep -x cnot103-train >/dev/null 2>&1; then exit 31; fi
python3 - <<'PY'
import hashlib, json
from pathlib import Path
r=Path('__QHAT4_ROOT__')
v84=r/'longrun-input-v84'
required=['followup.py','full-template.json',
          'full10000-dev32/dataset.json','full10000-dev32/receipt.json',
          'longrun-input-v84/payload-installed.json',
          'longrun-input-v84/proposal-source.tar.gz',
          'longrun-input-v84/proposal-source-receipt.json']
assert all((r/x).is_file() for x in required)
receipt=json.loads((r/'full10000-dev32/receipt.json').read_bytes())
dataset=r/'full10000-dev32/dataset.json'
assert hashlib.file_digest(dataset.open('rb'),'sha256').hexdigest() == 'eb71744c2afcef922596bc9956abfd8390a661273fa6461dd277a04fd85e9b86'
assert receipt['sha256'] == 'eb71744c2afcef922596bc9956abfd8390a661273fa6461dd277a04fd85e9b86'
template=r/'full-template.json'
assert hashlib.file_digest(template.open('rb'),'sha256').hexdigest() == '__TEMPLATE_SHA256__'
source=v84/'proposal-source.tar.gz'
source_receipt=json.loads((v84/'proposal-source-receipt.json').read_bytes())
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest() == '__SOURCE_SHA256__'
assert source_receipt['sha256'] == '__SOURCE_SHA256__'
assert source_receipt['bytes'] == source.stat().st_size
assert sum((r/x).stat().st_size for x in required) <= 256 * 1024 * 1024
assert not (r/'batched-probe-v18-qhat4-__TAG__').exists()
assert not (r/'train_qhat4_epoch_v18-__TAG__.py').exists()
assert not (r/'driver-console-v18-qhat4-__TAG__.log').exists()
assert not (r/'qhat4-source-v84b-__TAG__').exists()
assert not (r/'qhat4-target-v84b-__TAG__').exists()
print('remote_preflight=pass')
PY
'''
    pre = pre.replace('__TEMPLATE_SHA256__', EXPECTED_FULL_TEMPLATE_SHA256)
    pre = pre.replace('__SOURCE_SHA256__', EXPECTED_PROPOSAL_SOURCE_SHA256)
    pre = pre.replace('__TAG__', tag)
    pre = pre.replace('__QHAT4_ROOT__', str(VAST_ROOT))
    pre = pre.replace('__INSTANCE_ID__', str(INSTANCE_ID))
    pre_result = remote_run(pre, state)
    if pre_result != 'remote_preflight=pass':
        raise GateError('remote preflight response was invalid')

    arm = r'''set -eu
ROOT=/workspace/cnot103-full3060
ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}"; ID="${ID#C.}"
test "$ID" = "__INSTANCE_ID__"
test -n "${CONTAINER_API_KEY:-}"
mkdir -p "$ROOT/longrun-v84-state-$ID"
PUBKEY=''' + repr(pubkey) + r'''
WATCH="$ROOT/longrun-v84-state-$ID/qhat4-watchdog-__TAG__.py"
PIDFILE="$ROOT/longrun-v84-state-$ID/qhat4-watchdog-__TAG__.pid"
LOGFILE="$ROOT/longrun-v84-state-$ID/qhat4-watchdog-__TAG__.log"
test ! -e "$WATCH"
test ! -e "$PIDFILE"
cat > "$WATCH" <<'PY'
import json, os, time, urllib.request
from pathlib import Path
ident='__INSTANCE_ID__'
pubkey=''' + repr(pubkey) + r'''
time.sleep(10800)
auth=Path('/root/.ssh/authorized_keys')
if auth.exists():
    target=pubkey.split()[1]
    lines=[line for line in auth.read_text().splitlines() if len(line.split())<2 or line.split()[1]!=target]
    tmp=auth.with_suffix('.authorized_keys.qhat4.tmp')
    tmp.write_text('\n'.join(lines)+'\n' if lines else '')
    tmp.chmod(0o600)
    tmp.replace(auth)
key=os.environ.get('CONTAINER_API_KEY','')
if not key:
    raise SystemExit(2)
for attempt in range(20):
    try:
        request=urllib.request.Request('https://console.vast.ai/api/v0/instances/'+ident+'/',
            data=json.dumps({'state':'stopped'}).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='PUT')
        with urllib.request.urlopen(request,timeout=35) as response:
            response.read()
        break
    except Exception:
        if attempt == 19:
            raise
        time.sleep(20)
state=Path('/workspace/cnot103-full3060/longrun-v84-state-'+ident)
for name in ('qhat4-watchdog-__TAG__.pid','qhat4-watchdog-__TAG__.py'):
    try:
        (state/name).unlink()
    except FileNotFoundError:
        pass
PY
chmod 600 "$WATCH"
nohup python3 "$WATCH" </dev/null > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
chmod 600 "$PIDFILE"
printf '%s\n' 'watchdog=armed'
'''
    arm = arm.replace('__TAG__', tag)
    arm = arm.replace('__INSTANCE_ID__', str(INSTANCE_ID))
    if remote_run(arm, state) != 'watchdog=armed':
        raise GateError('bounded remote watchdog was not confirmed')

    api_state = get_api_instance()
    endpoint = ssh_endpoint(api_state)
    if not endpoint:
        raise GateError('instance SSH endpoint is unavailable')
    host, port = endpoint
    script_path = pathlib.Path(TRAINER)
    if not script_path.is_file():
        raise GateError('trainer script is unavailable on the runner')
    payload = script_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    remote_trainer = f'train_qhat4_epoch_v18-{tag}.py'
    upload = f"set -eu; ROOT={str(VAST_ROOT)}; DEST=\"$ROOT/{remote_trainer}\"; test ! -e \"$DEST\"; umask 077; cat > \"$DEST\"; chmod 600 \"$DEST\"; echo \"$(sha256sum \"$DEST\" | cut -d' ' -f1)\""
    proc = subprocess.run(
        ssh_command(host, port, "bash -c " + repr(upload)),
        input=payload, capture_output=True, timeout=90, check=False,
    )
    if proc.returncode or proc.stdout.decode('ascii', errors='ignore').strip() != digest:
        raise GateError('trainer script upload hash did not match')

    run_script = r'''set +e
export CNOT103_QHAT4_RUN_TAG=__TAG__
export CNOT103_QHAT4_ROOT=__QHAT4_ROOT__
cd __QHAT4_ROOT__
timeout --signal=TERM --kill-after=90s 165m python3 -u train_qhat4_epoch_v18-__TAG__.py > driver-console-v18-qhat4-__TAG__.log 2>&1
rc=$?
python3 - <<'PY'
import json
from pathlib import Path
p=Path('__QHAT4_ROOT__/batched-probe-v18-qhat4-__TAG__/public-status.json')
allowed={'experiment','epoch','model_contract_gate','runtime_gate','qhat4_quality_gate'}
if not p.is_file():
    print('run_status=missing')
else:
    obj=json.loads(p.read_bytes())
    status={k:obj.get(k,'not_evaluated') for k in allowed}
    print('run_status='+json.dumps(status,sort_keys=True,separators=(',',':')))
PY
exit "$rc"
'''
    run_script = run_script.replace('__TAG__', tag)
    run_script = run_script.replace('__QHAT4_ROOT__', str(VAST_ROOT))
    try:
        proc = subprocess.run(
            ssh_command(host, port), input=run_script.encode('utf-8'),
            capture_output=True, timeout=170 * 60, check=False,
        )
    except subprocess.TimeoutExpired:
        raise GateError('bounded remote training command timed out') from None
    out = proc.stdout.decode('utf-8', errors='strict').strip()
    if not out.startswith('run_status=') or '\n' in out:
        raise GateError('remote run returned an invalid status record')
    raw = out.removeprefix('run_status=')
    if raw != 'missing':
        status = json.loads(raw)
        required = {
            'experiment': {'complete', 'failed', 'running', 'not_started'},
            'epoch': {'complete', 'failed', 'running', 'not_started'},
            'model_contract_gate': {'pass', 'fail', 'not_evaluated'},
            'runtime_gate': {'pass', 'fail', 'not_evaluated'},
            'qhat4_quality_gate': {'pass', 'fail', 'not_evaluated'},
        }
        if set(status) != set(required) or any(status[k] not in values for k, values in required.items()):
            raise GateError('remote status did not match the allowlist')
        write_output({k: status[k] for k in required})
        print('run_status=' + json.dumps(status, sort_keys=True, separators=(',', ':')))
        success = proc.returncode == 0 and all(status[k] == 'pass' for k in (
            'model_contract_gate', 'runtime_gate', 'qhat4_quality_gate')) and status['experiment'] == 'complete'
        if not success:
            raise GateError('one or more experiment gates failed')
        return
    print('run_status=missing')
    raise GateError('remote status file was not produced')


def remove_ephemeral_key(state):
    pubpath = key_paths()[0].with_suffix('.pub')
    if not pubpath.is_file():
        return False
    pubkey = pubpath.read_text(encoding='ascii').strip()
    script = r'''set -eu
python3 - <<'PY'
from pathlib import Path
pubkey=''' + repr(pubkey) + r'''
auth=Path('/root/.ssh/authorized_keys')
if auth.exists():
    target=pubkey.split()[1]
    lines=[line for line in auth.read_text().splitlines() if len(line.split())<2 or line.split()[1]!=target]
    tmp=auth.with_suffix('.authorized_keys.qhat4.tmp')
    tmp.write_text('\n'.join(lines)+'\n' if lines else '')
    tmp.chmod(0o600)
    tmp.replace(auth)
state=Path('/workspace/cnot103-full3060/longrun-v84-state-__INSTANCE_ID__')
pidfile=state/'qhat4-watchdog-__TAG__.pid'
if pidfile.is_file():
    try:
        import os, signal
        os.kill(int(pidfile.read_text().strip()),signal.SIGTERM)
    except (OSError,ValueError):
        pass
for path in (pidfile,state/'qhat4-watchdog-__TAG__.py'):
    try:
        path.unlink()
    except FileNotFoundError:
        pass
print('ephemeral_key_removed=pass')
PY
'''
    script = script.replace('__TAG__', run_tag())
    script = script.replace('__INSTANCE_ID__', str(INSTANCE_ID))
    try:
        return remote_run(script, state) == 'ephemeral_key_removed=pass'
    except GateError:
        return False


def stop_if_owned():
    started = os.environ.get('VAST_STARTED_BY_WORKFLOW') == 'true'
    key_attached = os.environ.get('VAST_EPHEMERAL_KEY_ATTACHED') == 'true'
    if not started and not key_attached:
        print('instance_stop=not_owned')
        write_output({'instance_stop': 'not_owned', 'ephemeral_key_cleanup': 'not_needed'})
        return
    state = None
    try:
        state = get_instance()
    except GateError:
        pass
    removed = remove_ephemeral_key(state) if state and state.get('actual_status') == 'running' else False
    if started:
        try:
            vast(['stop', 'instance', str(INSTANCE_ID)], timeout=60)
        except GateError:
            request = urllib.request.Request(
                f'{VAST_API}/instances/{INSTANCE_ID}/',
                data=json.dumps({'state': 'stopped'}).encode('utf-8'),
                headers={'Authorization': f'Bearer {env_value("VAST_API_KEY")}',
                         'Content-Type': 'application/json'},
                method='PUT',
            )
            try:
                with urllib.request.urlopen(request, timeout=40) as response:
                    response.read()
            except (urllib.error.URLError, TimeoutError):
                print('instance_stop=unverified')
                write_output({'instance_stop': 'unverified', 'ephemeral_key_cleanup': 'pass' if removed else 'unverified'})
                return
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                current = get_instance()
                if current.get('actual_status') in ('exited', 'stopped') and current.get('intended_status') == 'stopped':
                    print('instance_stop=verified')
                    write_output({'instance_stop': 'verified', 'ephemeral_key_cleanup': 'pass' if removed else 'unverified'})
                    return
            except GateError:
                pass
            time.sleep(10)
        print('instance_stop=unverified')
        write_output({'instance_stop': 'unverified', 'ephemeral_key_cleanup': 'pass' if removed else 'unverified'})
        return
    print('instance_stop=not_owned')
    write_output({'instance_stop': 'not_owned', 'ephemeral_key_cleanup': 'pass' if removed else 'unverified'})


def cleanup_runner_key():
    private, known_hosts = key_paths()
    for path in (private, private.with_suffix('.pub'), known_hosts):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def failure_category(error):
    message = str(error).lower()
    if 'same host or local network' in message or 'local network' in message:
        return 'copy_network'
    if 'fallback input path' in message or 'experiment root' in message:
        return 'fallback_path'
    if 'data copy' in message or ' copy ' in message:
        return 'data_copy'
    if 'vast query' in message:
        return 'vast_query'
    if 'budget' in message or 'billing' in message:
        return 'budget'
    if 'vast start' in message:
        return 'vast_start'
    if 'ssh' in message:
        return 'ssh'
    if 'remote' in message or 'dataset' in message or 'trainer' in message:
        return 'remote_preflight'
    if 'instance' in message or 'state' in message:
        return 'instance_state'
    return 'operational'


def main():
    if len(sys.argv) != 2:
        raise GateError('invalid operation')
    op = sys.argv[1]
    if op == 'start':
        start()
    elif op == 'run':
        if REQUESTED_INSTANCE_ID == 'auto' and not SELECTED_INSTANCE_ID:
            raise GateError('auto-selected target is missing for the training step')
        state = get_instance()
        if state.get('actual_status') != 'running' or state.get('label') not in INSTANCE_LABELS:
            raise GateError('the authorized instance is not running')
        if not ssh_endpoint(get_api_instance()):
            raise GateError('instance SSH endpoint is unavailable')
        run_experiment(state)
    elif op == 'cleanup':
        try:
            stop_if_owned()
        finally:
            cleanup_runner_key()
    else:
        raise GateError('invalid operation')


if __name__ == '__main__':
    try:
        main()
    except GateError as error:
        print(f'vast_qhat4_gate=blocked category={failure_category(error)}')
        raise SystemExit(1)


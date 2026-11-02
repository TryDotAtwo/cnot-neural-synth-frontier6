import marimo._code_mode as cm
cell_code = '''def run_frontier61_v16_full_root_audit():
    import hashlib,io,json,os,requests,select,subprocess,sys,time,zipfile
    from pathlib import Path,PurePosixPath
    root=Path('/marimo/frontier61_recovery_v16')
    assert root.is_dir(), 'verified v16 source missing'
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            stat=(p/'stat').read_text();state=stat[stat.rfind(')')+2:].split()[0];cmd=(p/'cmdline').read_bytes()
            if state!='Z' and (b'research_v3' in cmd or b'frontier61_recovery' in cmd):active.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not active,'existing Frontier process requires ownership audit'
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
    assert gpu.returncode==0 and not gpu.stdout.strip(),'existing GPU process requires ownership audit'
    audit_url='https://raw.githubusercontent.com/TryDotAtwo/cnot-neural-synth-frontier61/b58543a9ea2ec8abf319342a6070d941a68e6fb2/audits/v16/audit_readonly_stream_v13_r2.py'
    response=requests.get(audit_url,timeout=(10,None))
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest()=='325b66c28d452cc3e5b51828faade84abac92d131f9f56838f16a8da7475dac7', 'root audit source drift'
    audit_path=root/'research_v3/cli/audit_readonly_stream_v13.py'
    if audit_path.exists():
        assert audit_path.read_bytes()==response.content, 'existing root auditor differs'
    else:
        audit_path.write_bytes(response.content)
    environment=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-B','-m','research_v3.cli.audit_readonly_stream_v13','--db',str(root/'full_holdout_v16.sqlite'),'--require-complete']
    assert (root/'full_holdout_v16.sqlite').is_file(), 'complete dev database missing'
    import sqlite3
    with sqlite3.connect((root/'full_holdout_v16.sqlite').as_uri()+'?mode=ro',uri=True) as _db:
        assert _db.execute("SELECT COUNT(*) FROM roots WHERE split='dev'").fetchone()[0]==1792, 'dev generation incomplete'
        assert _db.execute("SELECT COUNT(*) FROM roots WHERE split='test'").fetchone()[0]==1792, 'test generation incomplete'
        assert _db.execute("SELECT value FROM metadata WHERE key='holdout_seal'").fetchone() is not None, 'sealed holdout required'
    started=time.monotonic()
    audit_result=None
    with (root/'full_root_audit.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps({'stage':'NATIVE_V16_START','pid':process.pid}),flush=True)
        while True:
            ready,_,_=select.select([process.stdout],[],[],10)
            if ready:
                line=process.stdout.readline()
                if line:
                    log.write(line);log.flush();print(line.rstrip(),flush=True)
                    try:
                        candidate=json.loads(line)
                        if candidate.get('schema')=='own_v13_readonly_full_root_audit_v1':audit_result=candidate
                    except (ValueError,AttributeError):pass
                elif process.poll() is not None:break
            else:print(json.dumps({'stage':'NATIVE_V16_HEARTBEAT','pid':process.pid,'elapsed':time.monotonic()-started}),flush=True)
        returncode=process.wait()
    with sqlite3.connect((root/'full_holdout_v16.sqlite').as_uri()+'?mode=ro',uri=True) as _db:
        saved_seal=_db.execute("SELECT value FROM metadata WHERE key='holdout_seal'").fetchone()
    seal=None if saved_seal is None else json.loads(saved_seal[0])
    if returncode==0:
        assert seal is not None and seal['dev_roots']==1792 and seal['test_roots']==1792, 'successful exit without complete holdout seal'
        assert audit_result is not None and audit_result['complete_sealed_holdout'] and audit_result['root_count']==3584, 'full root audit result missing'
        (root/'full_root_audit.json').write_text(json.dumps(audit_result,sort_keys=True),encoding='utf8')
    receipt={'stage':'NATIVE_V16_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'root_audit':audit_result,'production_training_launched':False}
    (root/'full_root_audit_terminal.json').write_text(json.dumps(receipt,sort_keys=True),encoding='utf8')
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v16_full_root_audit_receipt=run_frontier61_v16_full_root_audit()
frontier61_v16_full_root_audit_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v16_full_root_audit',hide_code=False)
    print('DURABLE_NATIVE_V16_CELL',cid,flush=True)
    ctx.run_cell(cid)

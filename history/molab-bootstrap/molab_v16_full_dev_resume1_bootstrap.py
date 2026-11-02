import marimo._code_mode as cm
cell_code = '''def run_frontier61_v16_full_dev_resume1():
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
    environment=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-B','-m','research_v3.data.stream_solver_dataset_v13','--db',str(root/'full_holdout_v16.sqlite'),'--split','dev','--epoch','0','--count','1792','--expected-dev-roots','1792','--expected-test-roots','1792','--seed-base','3700','--stage17-root',str(root)]
    assert (root/'full_holdout_v16.sqlite').is_file(), 'audited partial dev database missing'
    import sqlite3
    with sqlite3.connect((root/'full_holdout_v16.sqlite').as_uri()+'?mode=ro',uri=True) as _db:
        assert _db.execute("SELECT COUNT(*) FROM roots WHERE split='dev'").fetchone()[0]==335, 'partial root count changed; audit again'
        assert not _db.execute("SELECT 1 FROM metadata WHERE key='holdout_seal'").fetchone(), 'holdout already sealed'
        prior=tuple(_db.execute("SELECT position,root_digest FROM roots WHERE split='dev' ORDER BY position"))
    started=time.monotonic()
    with (root/'full_dev_resume1.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps({'stage':'NATIVE_V16_START','pid':process.pid}),flush=True)
        while True:
            ready,_,_=select.select([process.stdout],[],[],10)
            if ready:
                line=process.stdout.readline()
                if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                elif process.poll() is not None:break
            else:print(json.dumps({'stage':'NATIVE_V16_HEARTBEAT','pid':process.pid,'elapsed':time.monotonic()-started}),flush=True)
        returncode=process.wait()
    with sqlite3.connect((root/'full_holdout_v16.sqlite').as_uri()+'?mode=ro',uri=True) as _db:
        after=tuple(_db.execute("SELECT position,root_digest FROM roots WHERE split='dev' AND position<335 ORDER BY position"))
        assert prior==after, 'prior root content changed'
        final_counts=dict(_db.execute('SELECT split,COUNT(*) FROM roots GROUP BY split'))
    print(json.dumps({'prior335_root_digests_preserved':True,'root_counts':final_counts}),flush=True)
    receipt={'stage':'NATIVE_V16_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'report_exists':(root/'full_dev_generation.json').exists(),'production_training_launched':False}
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v16_full_dev_resume1_receipt=run_frontier61_v16_full_dev_resume1()
frontier61_v16_full_dev_resume1_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v16_full_dev_resume1',hide_code=False)
    print('DURABLE_NATIVE_V16_CELL',cid,flush=True)
    ctx.run_cell(cid)

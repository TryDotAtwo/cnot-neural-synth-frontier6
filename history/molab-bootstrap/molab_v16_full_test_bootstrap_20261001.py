import marimo._code_mode as cm
cell_code = '''def run_frontier61_v16_full_test_generation():
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
    command=[sys.executable,'-B','-m','research_v3.data.stream_solver_dataset_v13','--db',str(root/'full_holdout_v16.sqlite'),'--split','test','--epoch','0','--count','1792','--expected-dev-roots','1792','--expected-test-roots','1792','--seed-base','3700','--stage17-root',str(root/'historical_stage17_own_v1')]
    assert (root/'full_holdout_v16.sqlite').is_file(), 'complete dev database missing'
    import sqlite3
    with sqlite3.connect((root/'full_holdout_v16.sqlite').as_uri()+'?mode=ro',uri=True) as _db:
        assert _db.execute("SELECT COUNT(*) FROM roots WHERE split='dev'").fetchone()[0]==1792, 'dev generation incomplete'
        assert _db.execute("SELECT COUNT(*) FROM roots WHERE split='test'").fetchone()[0]==0, 'test generation needs explicit resume audit'
    started=time.monotonic()
    with (root/'full_test_generation.log').open('x') as log:
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
    receipt={'stage':'NATIVE_V16_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'report_exists':(root/'full_test_generation.json').exists(),'production_training_launched':False}
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v16_full_test_receipt=run_frontier61_v16_full_test_generation()
frontier61_v16_full_test_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v16_full_test_generation',hide_code=False)
    print('DURABLE_NATIVE_V16_CELL',cid,flush=True)
    ctx.run_cell(cid)

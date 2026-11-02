import marimo._code_mode as cm
cell_code = '''def run_frontier61_v12_guarded_panel():
    import hashlib,io,json,os,requests,select,subprocess,sys,time,zipfile
    from pathlib import Path,PurePosixPath
    root=Path('/marimo/frontier61_recovery_v12')
    assert root.is_dir()
    manifest=json.loads((root/'source_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
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
    command=[sys.executable,'-B','-m','research_v3.cli.audit_guarded_native_panel','--report',str(root/'guarded_panel_112.json')]
    started=time.monotonic()
    with (root/'guarded_panel_112.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps({'stage':'NATIVE_V12_START','pid':process.pid}),flush=True)
        while True:
            ready,_,_=select.select([process.stdout],[],[],10)
            if ready:
                line=process.stdout.readline()
                if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                elif process.poll() is not None:break
            else:print(json.dumps({'stage':'NATIVE_V12_HEARTBEAT','pid':process.pid,'elapsed':time.monotonic()-started}),flush=True)
        returncode=process.wait()
    receipt={'stage':'NATIVE_V12_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'report_exists':(root/'guarded_panel_112.json').exists(),'production_training_launched':False}
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v12_native_receipt=run_frontier61_v12_guarded_panel()
frontier61_v12_native_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v12_guarded_panel_112',hide_code=False)
    print('DURABLE_PANEL_V12_CELL',cid,flush=True)
    ctx.run_cell(cid)

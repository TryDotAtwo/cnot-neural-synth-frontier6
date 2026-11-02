import marimo._code_mode as cm
cell_code = '''def run_frontier61_v14_native_diagnostic():
    import hashlib,io,json,os,requests,select,subprocess,sys,time,zipfile
    from pathlib import Path,PurePosixPath
    root=Path('/marimo/frontier61_recovery_v14')
    data=requests.get('https://raw.githubusercontent.com/TryDotAtwo/cnot-neural-synth-frontier61/b99bc80a8ff6ac1d1159c34ed046ca78d0568a1c/packages/v14_recovery_candidate_v14_20261001.zip',timeout=(10,None))
    data.raise_for_status()
    payload=data.content
    assert hashlib.sha256(payload).hexdigest()=='8e30ad2376d9d72cc4fc6b6ba380fb4be2aa6f9f80b26928370bd28b55c6e0a0'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('source_manifest.json'))
        for name,expected in manifest['files'].items():
            relative=PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected
        root.mkdir(exist_ok=False)
        for name in archive.namelist():
            relative=PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            target=root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out:out.write(archive.read(name))
    print(json.dumps({'stage':'RESTORE_V14_VERIFIED','files':len(manifest['files'])}),flush=True)
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
    command=[sys.executable,'-B','-m','research_v3.cli.audit_native_resource_words','--baseline-report','/marimo/frontier61_recovery_v12/guarded_panel_112.json','--report',str(root/'resource_words_native.json')]
    started=time.monotonic()
    with (root/'resource_words_native.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps({'stage':'NATIVE_V14_START','pid':process.pid}),flush=True)
        while True:
            ready,_,_=select.select([process.stdout],[],[],10)
            if ready:
                line=process.stdout.readline()
                if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                elif process.poll() is not None:break
            else:print(json.dumps({'stage':'NATIVE_V14_HEARTBEAT','pid':process.pid,'elapsed':time.monotonic()-started}),flush=True)
        returncode=process.wait()
    receipt={'stage':'NATIVE_V14_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'report_exists':(root/'resource_words_native.json').exists(),'production_training_launched':False}
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v14_native_receipt=run_frontier61_v14_native_diagnostic()
frontier61_v14_native_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v14_native_exposure_diagnostic',hide_code=False)
    print('DURABLE_NATIVE_V14_CELL',cid,flush=True)
    ctx.run_cell(cid)

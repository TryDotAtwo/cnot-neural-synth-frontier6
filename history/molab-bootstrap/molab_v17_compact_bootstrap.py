import marimo._code_mode as cm
cell_code = '''def run_frontier61_v17_native_diagnostic():
    import hashlib,io,json,os,requests,select,subprocess,sys,time,zipfile
    from pathlib import Path,PurePosixPath
    root=Path('/marimo/frontier61_recovery_v17')
    data=requests.get('https://raw.githubusercontent.com/TryDotAtwo/cnot-neural-synth-frontier61/7767783c21a3f0d64d44b9c2272cc5aa264f2e6c/packages/v13_recovery_candidate_v17_20261001.zip',timeout=(10,None))
    data.raise_for_status()
    payload=data.content
    assert hashlib.sha256(payload).hexdigest()=='9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866'
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
    print(json.dumps({'stage':'RESTORE_V17_VERIFIED','files':len(manifest['files'])}),flush=True)
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
    command=[sys.executable,'-B','-m','research_v3.cli.audit_native_compact_onehop','--report',str(root/'compact_onehop_native.json')]
    started=time.monotonic()
    with (root/'compact_onehop_native.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps({'stage':'NATIVE_V17_START','pid':process.pid}),flush=True)
        while True:
            ready,_,_=select.select([process.stdout],[],[],10)
            if ready:
                line=process.stdout.readline()
                if line:log.write(line);log.flush();print(line.rstrip(),flush=True)
                elif process.poll() is not None:break
            else:print(json.dumps({'stage':'NATIVE_V17_HEARTBEAT','pid':process.pid,'elapsed':time.monotonic()-started}),flush=True)
        returncode=process.wait()
    receipt={'stage':'NATIVE_V17_TERMINAL','returncode':returncode,'seconds':time.monotonic()-started,'report_exists':(root/'compact_onehop_native.json').exists(),'production_training_launched':False}
    print(json.dumps(receipt),flush=True)
    return receipt
frontier61_v17_native_receipt=run_frontier61_v17_native_diagnostic()
frontier61_v17_native_receipt
'''
async with cm.get_context() as ctx:
    cid=ctx.create_cell(cell_code,name='v17_compact_onehop_diagnostic',hide_code=False)
    print('DURABLE_NATIVE_V17_CELL',cid,flush=True)
    ctx.run_cell(cid)

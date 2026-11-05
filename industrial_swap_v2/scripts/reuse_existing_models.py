"""Record and validate user-provided official HF snapshots without modifying them."""
import json
from pathlib import Path
from industrial_swap_v2.core.context import sha256, write_json
base=Path(__file__).resolve().parents[1]
cache=Path('/home/kxliu/.cache/huggingface/hub')
revisions={
    'microsoft/TRELLIS.2-4B':'af44b45f2e35a493886929c6d786e563ec68364d',
    'microsoft/TRELLIS-image-large':'25e0d31ffbebe4b5a97464dd851910efc3002d96',
    'facebook/dinov3-vitl16-pretrain-lvd1689m':'ea8dc2863c51be0a264bab82070e3e8836b02d51',
    'briaai/RMBG-2.0':'5df4c9c76d8170882c34f6986e848ee07fd0ba43'}
deps={}; provenance={}
for repo,revision in revisions.items():
    snapshot=cache/('models--'+repo.replace('/','--'))/'snapshots'/revision
    if not snapshot.is_dir():
        # Sparse decoder was already downloaded in the new project cache.
        snapshot=base/'models/huggingface/hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision
    files={}
    for path in snapshot.rglob('*'):
        if path.is_file():
            if repo=='microsoft/TRELLIS-image-large' and not path.name.startswith('ss_dec_conv3d_16l8_fp16'):
                continue
            digest=sha256(path)
            if len(path.resolve().name)==64:
                assert digest==path.resolve().name, ('Cached LFS object hash mismatch',path.name)
            files[path.relative_to(snapshot).as_posix()]={'bytes':path.stat().st_size,'sha256':digest}
    if not any(name.endswith('.safetensors') for name in files):
        raise RuntimeError(f'No readable weights in {repo}')
    provenance[repo]={'snapshot_path':str(snapshot),'revision':revision,'read_only_reuse':True,'files':files}
    if repo=='microsoft/TRELLIS.2-4B':
        pipeline=json.loads((snapshot/'pipeline.json').read_text())
        for name,path in pipeline['args']['models'].items():
            if path.startswith('ckpts/'):
                assert path+'.json' in files and path+'.safetensors' in files, ('Incomplete main snapshot',name)
        write_json(base/'records/trellis_model.json',{'model':repo,'revision':revision,'pipeline':pipeline,
            'snapshot_path':str(snapshot),'download_complete':True,'reused_existing_user_cache':True})
    else:
        deps[repo]={'revision':revision,'snapshot_path':str(snapshot),'status':'complete','read_only_reuse':True}
    print(repo,revision,'validated',len(files),'files',flush=True)
write_json(base/'records/trellis_dependencies.json',deps)
write_json(base/'records/reused_model_integrity.json',provenance)

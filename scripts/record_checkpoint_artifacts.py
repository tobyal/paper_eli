"""Record official checkpoint artifacts without modifying or reserializing them."""
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['ROOT'])
records={}
for method in ['gaussianeditor','dge','dreamcatalyst','editsplat']:
    dest=root/'outputs'/method
    paths=list(dest.glob('**/last.ckpt')) if method!='editsplat' else list(dest.glob('point_cloud/iteration_33060/chkpnt33060.pth'))
    assert len(paths)==1,(method,paths)
    path=paths[0]
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    item={'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'format':'Official saved checkpoint; not converted'}
    manifest_path=dest/'artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['official_checkpoint']=item
    manifest_path.write_text(json.dumps(manifest,indent=2))
    records[method]=item
(root/'records/official_checkpoints.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))

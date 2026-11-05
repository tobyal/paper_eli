"""Verify completed-stage receipts on the server or a downloaded local copy."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--stage',choices=['all','m1','m2','m3','m4'],default='all')
args=p.parse_args(); base=args.base.resolve(); report={}
for stage in ['m1','m2','m3','m4'] if args.stage=='all' else [args.stage]:
    receipt=json.loads((base/'records'/f'{stage}_status.json').read_text(encoding='utf-8'))
    assert receipt['status']=='complete',(stage,receipt['status'])
    for item in receipt['artifacts']:
        relative=item['path'].replace('\\','/').split('/industrial_swap_v2/',1)[1]
        path=(base/relative).resolve(); path.relative_to(base)
        assert path.stat().st_size==item['bytes'],('Size mismatch',path)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
        assert digest.hexdigest()==item['sha256'],('SHA256 mismatch',path)
    report[stage]={'status':'verified','artifacts':len(receipt['artifacts'])}
    print(stage,'verified',len(receipt['artifacts']),'artifacts',flush=True)
(base/'records/artifact_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

"""Pin official TRELLIS.2 snapshots and their documented dependent weights."""
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

base = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(base/'models/huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
cache = base/'models/huggingface/hub'
api = HfApi()
repo = 'microsoft/TRELLIS.2-4B'
info = api.model_info(repo, revision='af44b45f2e35a493886929c6d786e563ec68364d')
pipeline_path = hf_hub_download(repo, 'pipeline.json', revision=info.sha, cache_dir=cache)
pipeline = json.loads(Path(pipeline_path).read_text())
record = {'model':repo, 'revision':info.sha, 'pipeline':pipeline, 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(base/'records/trellis_model.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
path = snapshot_download(repo, revision=info.sha, cache_dir=cache, max_workers=4)
record['snapshot_path'] = path
record['download_complete'] = True
(base/'records/trellis_model.json').write_text(json.dumps(record,indent=2))
print('TRELLIS snapshot ready',path,flush=True)

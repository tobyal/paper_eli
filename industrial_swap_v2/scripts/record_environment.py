import importlib
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base / "third_party/TRELLIS.2"))
modules = ("torch", "flash_attn", "nvdiffrast", "nvdiffrec_render", "cumesh", "flex_gemm", "o_voxel", "trellis2.pipelines")
loaded = {}
for name in modules:
    module = importlib.import_module(name)
    loaded[name] = {"file": module.__file__, "version": getattr(module, "__version__", None)}
commits = {}
for path in (base / "third_party").iterdir():
    if (path / ".git").exists():
        commits[path.name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
record = {"python": sys.version, "modules": loaded, "upstream_commits": commits}
distributions = {}
for name in ('torch','torchvision','transformers','flash-attn','nvdiffrast','nvdiffrec_render','cumesh','flex_gemm','o_voxel','trimesh'):
    distribution = importlib.metadata.distribution(name)
    direct = distribution.read_text('direct_url.json')
    distributions[name] = {'version':distribution.version,'direct_url':json.loads(direct) if direct else None}
record['installed_distributions'] = distributions
fingerprints = {}
for name in ('nvdiffrast','nvdiffrec_render','cumesh','flex_gemm','o_voxel'):
    package = Path(loaded[name]['file']).parent
    fingerprints[name] = {}
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in ('.so','.py','.cpp','.cu','.h','.cuh'):
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
            fingerprints[name][str(path.relative_to(package))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
record['extension_package_fingerprints'] = fingerprints
record['legacy_extension_source_commit_note'] = 'The reused installation records /tmp/extensions directories which no longer exist. Those historical build commits cannot be recovered from its metadata; package versions and full installed source/binary hashes are recorded instead. Do not claim an exact source revision for those reused binaries.'
check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
record['pip_check'] = {'exit_code':check.returncode,'output':check.stdout,
    'note':'Inherited Jupyter packages are not part of this inference pipeline; imports above are checked independently.'}
(base / "records/environment.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)

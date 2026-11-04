"""Package logs, scripts, exact metadata and real previews; large PLYs stay on server."""
import json,os,tarfile
from pathlib import Path
root=Path(os.environ['ROOT'])
methods=['gaussianeditor','dge','dreamcatalyst','editsplat']
for method in methods:
 assert (root/f'records/{method}.reviewed').exists(), method+' needs visual review'
 assert (root/f'outputs/{method}/artifact_manifest.json').exists()
assert (root/'records/final_comparison.json').exists()
assert (root/'outputs/comparison/four_methods_multiview.mp4').exists()
files=[]
for folder in ['scripts','records','logs']:
 files.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
for name in ['README_reproduction.txt','REPORT.txt']:
 files.append(root/name)
for method in methods:
 dest=root/'outputs'/method
 files += [dest/'artifact_manifest.json',dest/'model_check.json',dest/'quality_preview.png',dest/'common_render/multiview.mp4']
 files.extend(dest.glob('**/configs/*.yaml'))
 files.extend(p for p in [dest/'args.json',dest/'train_frozen.py',dest/'imagereward_ranking.json'] if p.exists())
files += [root/'outputs/reconstruction/common_render/multiview.mp4']
test_names=json.loads((root/'records/camera_split.json').read_text())['test']
for method in ['reconstruction']+methods:
 render=root/'outputs'/method/'common_render'
 files.extend(p for p in [render/'cameras.json',render/'model.json'] if p.exists())
 for name in test_names:
  files.append(render/'wide'/Path(name).with_suffix('.png'))
 for frame in [61,91,151,161,281]:
  files.append(render/'wide'/f'frame_{frame:06d}.png')
for name in test_names:
 files.append(root/'outputs/evaluation_masks/wide'/Path(name).with_suffix('.png'))
 files.append(root/'outputs/reconstruction/common_render/wide'/Path(name).with_suffix('.npz'))
files.extend(p for p in (root/'outputs/comparison').glob('*') if p.is_file())
destination=root/'reproduction_results.tar.gz'
with tarfile.open(destination,'w:gz',compresslevel=3) as archive:
 for path in sorted(set(files)):
  archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,'included_files':len(set(files)),'large_models':'PLY/checkpoints and pretrained weights remain in server outputs/models directories; locations and SHA256 are in manifests'},indent=2))

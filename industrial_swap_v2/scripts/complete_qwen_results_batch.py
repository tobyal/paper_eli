"""Import named manual Qwen results and use the existing durable M2/M3/M4 worker.

No generation API is exposed here. Each stage is explicitly advanced after review.
Run on ubuntuA6000 with the webapp interpreter, from the repository root.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from industrial_swap_v2.core.context import write_json
from industrial_swap_v2.webapp import service as v, store as s


def initialize(folder, source_manifest):
    destination = folder / 'manifest.json'
    if destination.exists():
        return json.loads(destination.read_text())
    source = json.loads(source_manifest.read_text())
    catalog = {i['key']: i for i in source['items']}
    paths = sorted((folder / 'inputs').glob('case_*.png'))
    if not paths:
        raise ValueError('No named manual Qwen results')
    result = {'batch': folder.name, 'created_at': s.now(), 'source_manifest': str(source_manifest),
              'qwen_api_allowed': False, 'm4_iterations': 250, 'items': []}
    # Validate every mapping before creating any experiment.
    for path in paths:
        item = catalog[path.stem]
        conditions = json.loads((Path(item['output_directory']) / 'conditions.json').read_text())
        if conditions.get('spatial_method') != 'adaptive_visible_v1':
            raise ValueError('Expected current adaptive M2 input: ' + path.stem)
        camera = json.loads((Path(item['output_directory']) / 'anchor_camera.json').read_text())
        with Image.open(path) as image:
            if abs(image.width/image.height-camera['width']/camera['height']) > 1e-6:
                raise ValueError('Anchor canvas aspect mismatch: ' + path.stem)
        parent = v.experiment(item['experiment_id'])
        heads = s.unpack(parent['heads'])
        if heads.get('materials') != item['run_id']:
            raise ValueError('Prepared experiment no longer points to declared materials')
        for name in ('m1', 'materials'):
            if not v.verify_outputs(v.run(heads[name])):
                raise ValueError('Input integrity failed: ' + path.stem + '/' + name)
    for path in paths:
        item = catalog[path.stem]
        identifier = v.create(folder.name + ' / ' + path.stem, copy_from=item['experiment_id'])
        exp = v.experiment(identifier)
        heads = s.unpack(exp['heads'])
        camera = json.loads((Path(item['output_directory']) / 'anchor_camera.json').read_text())
        metadata = {'camera_aligned': True, 'camera_key': v.camera_key(camera),
                    'scene_sha': exp['scene_sha'], 'source_case_key': path.stem,
                    'source_preparation_id': item['preparation_id'],
                    'alignment_evidence': 'User-supplied full square canvas, named for supplied Qwen input folder; no crop',
                    'manual_qwen_result': True}
        aid = s.register(path, 'edited_scene', name=path.name, source='user_upload',
                         model='qwen-image-2.1-pro (user-generated)', prompt=item['description'],
                         metadata=metadata, experiment=identifier, role='m2')
        v.select(identifier, aid, 'm2')
        entry = {'key': path.stem, 'label': item['label'], 'description': item['description'],
                 'experiment_id': identifier, 'source_experiment_id': item['experiment_id'],
                 'source_case_id': item['case_id'], 'source_preparation_id': item['preparation_id'],
                 'input_path': str(path), 'input_sha256': s.sha(path), 'asset_id': aid,
                 'anchor': item['anchor'], 'source_heads': heads, 'runs': {},
                 'web_url': '/#experiment/' + identifier}
        result['items'].append(entry)
        write_json(destination, result)
    return result


def advance(folder, manifest, stage, keys):
    for item in manifest['items']:
        if keys and item['key'] not in keys:
            continue
        if stage in item['runs']:
            continue
        action = {'m2': 'reuse', 'm3': 'trellis', 'm4': 'refine'}[stage]
        parameters = {'iterations': manifest['m4_iterations']} if stage == 'm4' else {}
        item['runs'][stage] = v.submit(item['experiment_id'], stage, action,
                                      folder.name + ':' + item['key'] + ':' + stage,
                                      parameters=parameters)
        write_json(folder / 'manifest.json', manifest)


def status(folder, manifest):
    for item in manifest['items']:
        states = {}
        for stage, rid in item['runs'].items():
            value = v.run(rid)
            states[stage] = {'id': rid, 'status': value['status'], 'error': value['error'],
                             'folder': value['folder'], 'log_path': value['log_path'],
                             'output': s.unpack(value['outputs']).get('directory')}
        item['states'] = states
        print(json.dumps({'key': item['key'], 'experiment_id': item['experiment_id'],
                          'stages': {k: z['status'] for k, z in states.items()},
                          'errors': {k: z['error'] for k,z in states.items() if z['error']}}), flush=True)
    write_json(folder / 'manifest.json', manifest)


def import_reviewed_sam(folder, manifest, keys, review_name='sam_review'):
    if not keys:
        raise ValueError('Explicit visually reviewed case keys are required')
    for item in manifest['items']:
        if item['key'] not in keys or item.get('reviewed_foreground_asset') or item.get('reviewed_sam_asset'):
            continue
        review = folder/review_name/item['key']
        record = json.loads((review/'selection.json').read_text())
        if record['source_sha256'] != item['input_sha256']:
            raise ValueError('SAM image source mismatch')
        record.update(review_status='visually_reviewed', review_at=s.now())
        write_json(review/'selection.json', record)
        metadata = s.unpack(s.asset(item['asset_id'])['metadata'])
        metadata.update(sam_review=str(review/'selection.json'), source_asset=item['asset_id'],
                        source_image_sha256=item['input_sha256'], sam_called=record.get('sam_called',True),
                        foreground_extraction=record['method'])
        aid = s.register(review/'object_rgba.png', 'object_rgba',
                         name=item['key']+' reviewed foreground', source='foreground_segmentation',
                         model=record.get('model','SAM ViT-H + OpenCLIP'), prompt=item['description'], metadata=metadata,
                         experiment=item['experiment_id'], role='m2')
        v.select(item['experiment_id'], aid, 'm2')
        if item['runs'].get('m2'):
            item.setdefault('prior_m2_runs', []).append(item['runs'].pop('m2'))
        item['reviewed_foreground_asset'] = aid
        item['sam_review_directory'] = str(review)
        item['runs']['m2'] = v.submit(item['experiment_id'], 'm2', 'reuse',
                                     folder.name+':'+item['key']+':m2-reviewed-sam')
        write_json(folder/'manifest.json', manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['init', 'status', 'm2', 'm3', 'm4', 'import-sam', 'watch-m3'])
    p.add_argument('--folder', required=True, type=Path)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--keys', nargs='*')
    p.add_argument('--sam-review-root', default='sam_review', choices=['sam_review','sam_review_v2','rembg_review'])
    args = p.parse_args()
    folder = args.folder.resolve()
    folder.relative_to(s.BASE / 'outputs')
    s.init()
    if args.action == 'watch-m3':
        while True:
            manifest=json.loads((folder/'manifest.json').read_text())
            remaining=[i for i in manifest['items'] if 'm3' not in i['runs']]
            if not remaining:
                status(folder,manifest)
                return
            for item in remaining:
                if not (item.get('reviewed_sam_asset') or item.get('reviewed_foreground_asset')):
                    raise ValueError('Foreground must be visually reviewed before automatic M3 scheduling')
                if v.run(item['runs']['m2'])['status']=='completed':
                    advance(folder,manifest,'m3',[item['key']])
                    print(item['key'],'M3 submitted after reviewed foreground import',flush=True)
                elif v.run(item['runs']['m2'])['status']=='failed':
                    raise RuntimeError('Reviewed foreground import failed: '+item['key'])
            time.sleep(10)
    if args.action == 'init':
        if not args.source_manifest:
            p.error('--source-manifest is required for init')
        manifest = initialize(folder, args.source_manifest)
    else:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if args.action in ('m2', 'm3', 'm4'):
            advance(folder, manifest, args.action, args.keys)
        elif args.action == 'import-sam':
            import_reviewed_sam(folder, manifest, args.keys, args.sam_review_root)
    status(folder, manifest)


if __name__ == '__main__':
    main()

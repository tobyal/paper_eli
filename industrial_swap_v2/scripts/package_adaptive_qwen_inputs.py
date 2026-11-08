"""Prepare the approved 30 adaptive M2 variants and export Qwen-only inputs."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import yaml
from ..core.context import read_json,write_json
from ..core.m2_space import VERSION,signature
from ..webapp import store as s,service as v,m2_cases

INPUTS={'01_scene_rgb.png':'anchor.png','02_position_hint.png':'position_hint.png',
        '03_background_depth.png':'depth_background.png','prompt.txt':'generation_prompt.txt'}


def submit(folder):
    catalog=read_json(s.BASE/'configs/m2_case_description_options.json')
    assert len(catalog['cases'])==10 and all(len(c['options'])==3 for c in catalog['cases'])
    manifest=read_json(folder/'manifest.json') if (folder/'manifest.json').exists() else {
        'batch':folder.name,'spatial_method':VERSION,'created_at':s.now(),
        'scope':'30 Qwen-only input sets, no image-generation request','items':[]}
    for case in catalog['cases']:
        row,case_folder=m2_cases.case(case['case_id'])
        for option in case['options']:
            key=f"case_{case['case_number']:02d}_{option['id']}"
            if any(i['key']==key for i in manifest['items']):continue
            expected=m2_cases.config_for_case(row,case_folder,option['description'])
            expected['m2'].update(avp_review_fallback=True,avp_review_policy='expanded_observed_canvas_v1')
            result=None
            # Reuse only an actual, identical adaptive preparation, never the old batch.
            prior=s.rows('''SELECT p.id,p.run_id FROM case_m2_preparations p JOIN stage_runs r ON r.id=p.run_id
                WHERE p.case_id=:case AND p.replacement=:text AND r.status='completed' ORDER BY p.created_at DESC''',
                case=case['case_id'],text=option['description'])
            hashes={n:s.sha(case_folder/'outputs/m1'/n) for n in
                ('edit_region.json','target.json','target_indices.npy','target_gaussians.ply','background_gaussians.ply')}
            for previous in prior:
                run=v.run(previous['run_id']);out=v.stage_dir(run['id']);condition=read_json(out/'conditions.json')
                if (condition.get('spatial_method')==VERSION and condition.get('condition_signature')==signature(expected)
                    and condition.get('input_sha256')==hashes and v.verify_outputs(run)):
                    result=m2_cases.detail(previous['id']);break
            reused=result is not None
            if result is None:
                result=m2_cases.prepare(case['case_id'],m2_cases.Preparation(replacement_text=option['description'],
                    request_id=folder.name+'-'+key,avp_review_fallback=True))
            manifest['items'].append({'key':key,'case_number':case['case_number'],'case_id':case['case_id'],
                'variant':option['id'],'label':option['label'],'description':option['description'],
                'preparation_id':result['id'],'run_id':result['run_id'],'experiment_id':result['experiment_id'],
                'reused_existing_adaptive_result':reused})
            write_json(folder/'manifest.json',manifest)
            print(key,result['status'],'existing' if reused else 'new',flush=True)
    assert len(manifest['items'])==30


def status(folder):
    manifest=read_json(folder/'manifest.json');counts={};running=[];failed=[]
    for item in manifest['items']:
        run=v.run(item['run_id']);item.update(status=run['status'],error=run['error'])
        counts[run['status']]=counts.get(run['status'],0)+1
        if run['status']=='running':
            log=Path(run['log_path']); tail=log.read_text(errors='replace').splitlines()[-1:] if log.exists() else []
            running.append({'key':item['key'],'tail':tail})
        if run['status']=='failed':failed.append({'key':item['key'],'error':run['error'],'log':run['log_path']})
    manifest.update(counts=counts,updated_at=s.now());write_json(folder/'manifest.json',manifest)
    print({'counts':counts,'running':running,'failed':failed},flush=True)
    return manifest


def export(folder):
    manifest=status(folder)
    assert manifest['counts']=={'completed':30},'All 30 real M2 preparations must complete before packaging'
    file_hashes={};review=[]
    for item in manifest['items']:
        run=v.run(item['run_id']);source=v.stage_dir(run['id']);condition=read_json(source/'conditions.json')
        cfg=yaml.safe_load(Path(run['config_path']).read_text())
        assert condition['spatial_method']==VERSION and condition['condition_signature']==signature(cfg)
        assert cfg['m2']['replacement_text']==item['description'] and cfg['m2']['preparation_only']
        assert v.verify_outputs(run)
        row,case=m2_cases.case(item['case_id'])
        audit=folder/'audits'/(item['key']+'.json')
        subprocess.run([str(s.ROOT/'envs/gsle/bin/python'),'-m','industrial_swap_v2.scripts.audit_m2_case_inputs',
            '--config',run['config_path'],'--case',str(case),'--report',str(audit)],cwd=s.ROOT,check=True)
        assert not list((Path(run['folder'])/'records').glob('qwen*lock'))
        assert s.one('SELECT external_api FROM generation_records WHERE run_id=:id',id=run['id'])['external_api']==0
        destination=folder/'qwen_only'/item['key'];destination.mkdir(parents=True,exist_ok=True)
        for name,source_name in INPUTS.items():
            shutil.copy2(source/source_name,destination/name)
            digest=s.sha(source/source_name);assert s.sha(destination/name)==digest
            file_hashes[item['key']+'/'+name]=digest
        assert set(p.name for p in destination.iterdir())==set(INPUTS)
        item.update(output_directory=str(source),anchor=condition['camera']['name'],
            mask_pixels=condition['mask_area'],mask_quality=condition['mask_report']['quality_status'],
            anchor_quality=condition['avp_selection'].get('quality_status','passed'),
            web_url=f"/m2-cases?case={item['case_id']}&preparation={item['preparation_id']}")
        if item['mask_quality']=='manual_review_required' or item['anchor_quality']=='manual_review_required':review.append(item['key'])
        write_json(folder/'manifest.json',manifest)
        print('AUDITED',item['key'],item['anchor'],'pixels',item['mask_pixels'],flush=True)
    archive=folder/'qwen_only_adaptive_30.zip'
    if archive.exists():raise FileExistsError('Do not overwrite an already exported bundle')
    temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in sorted(file_hashes):z.write(folder/'qwen_only'/relative,relative)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None and len(z.namelist())==120
        assert set(z.namelist())==set(file_hashes)
        for name,digest in file_hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    temporary.rename(archive)
    manifest.update(finished_at=s.now(),qwen_called=False,trellis_called=False,training_executed=False,
        archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=s.sha(archive),
        archived_files=120,archived_input_sha256=file_hashes,manual_review_inputs=review)
    write_json(folder/'manifest.json',manifest)
    print('PACKAGED',archive,archive.stat().st_size,'bytes','review',review,flush=True)


def watch(folder):
    previous=None
    while True:
        manifest=read_json(folder/'manifest.json');assert len(manifest['items'])==30
        counts={};progress=[]
        for item in manifest['items']:
            run=v.run(item['run_id']);counts[run['status']]=counts.get(run['status'],0)+1
            if run['status']=='running':
                try:lines=Path(run['log_path']).read_text(errors='replace').splitlines()
                except FileNotFoundError:lines=[]
                tail=lines[-1] if lines else 'Preparing isolated inputs'
                if tail.startswith('{'):tail='Starting the M2 input-only adapter'
                progress.append((item['key'],tail))
        current=(counts,progress)
        if current!=previous:
            print(time.strftime('%H:%M:%S'),{'counts':counts,'progress':progress},flush=True)
            previous=current
        if counts.get('failed'):
            status(folder);raise RuntimeError('Inspect failed preparations before packaging')
        if counts=={'completed':30}:
            export(folder);return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','export','watch']);p.add_argument('--folder',type=Path,required=True)
    a=p.parse_args();folder=a.folder.resolve();folder.relative_to(s.BASE);folder.mkdir(parents=True,exist_ok=True)
    globals()[a.action](folder)

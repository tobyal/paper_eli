"""Expose the verified historical robot as case00 without executing any model.

Keeps the nine-case manifest, audit and ZIP unchanged. Imported M4 files are
read-only references; a separate combined manifest/package records provenance.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from industrial_swap_v2.core.context import read_json, write_json
from industrial_swap_v2.webapp import service as v, store as s

SOURCE_EXPERIMENT = '9b0eeee5840d453e8f21e1763544a6cb'
SOURCE_M3 = '173dd1e87749494db5f66ab1287d37d7'
HISTORICAL_RUN = 'outputs/m4_dual_mask/run_resume2000_to10000_save2000_20261106_173926'


def link(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        assert destination.resolve() == source, destination
    else:
        destination.symlink_to(source)


def import_case(batch):
    root = s.BASE / HISTORICAL_RUN
    report = read_json(root / 'm4/summary.json')
    verification = read_json(root / 'run_verification.json')
    assert verification['target_step'] == 10000 and verification['executed_steps'] == 8000
    assert verification['source_inputs_unchanged'] and verification['previous_run_files_unchanged']
    native = v.stage_dir(SOURCE_M3)
    expected = report['candidate_sha256']
    for path in (root/'m3/object_world.ply', native/'object_world.ply', root/'m4/object_fixed.ply'):
        assert s.sha(path) == expected['object'], path
    assert s.sha(report['candidate_scene']) == expected['scene']
    assert s.sha(report['candidate_background']) == expected['background']
    for path, checksum in read_json(root/'source_preservation.json').items():
        assert s.sha(path) == checksum, path

    folder = batch/'case00_historical'
    folder.mkdir(exist_ok=True)
    registration = folder/'registration.json'
    if registration.exists():
        state = read_json(registration)
    else:
        eid = v.create('case00 · 黄色六轴机械臂 · 已完成 10000 步', copy_from=SOURCE_EXPERIMENT)
        heads = s.unpack(v.experiment(SOURCE_EXPERIMENT)['heads'])
        heads = {k: heads[k] for k in ('m1', 'm2', 'materials') if k in heads}
        heads['m3'] = SOURCE_M3
        with s.transaction() as conn:
            s.sql(conn, 'UPDATE experiments SET heads=:h WHERE id=:id', h=s.dump(heads), id=eid)
        state = {'experiment_id': eid, 'run_id': s.uid(), 'source_run': str(root), 'heads': heads}
        write_json(registration, state)

    names = ['summary.json', 'scene_final.ply', 'scene_candidate_step10000.ply', 'object_fixed.ply',
             'background_optimization/background_optimized.ply',
             'background_optimization/background_candidate_step10000.ply']
    names += [str(p.relative_to(root/'m4')) for sub in ('renders', 'validation')
              for p in (root/'m4'/sub).glob('*') if p.is_file()]
    for name in names:
        link(root/'m4'/name, folder/'m4'/name)
    shutil.copy2(root/'config.yaml', folder/'config.yaml')
    (folder/'import.log').write_text('Read-only import of verified historical M4. No jobs queued; no model invoked.\n'
                                   + str(root)+'\n', encoding='utf-8')
    rid = state['run_id']; eid = state['experiment_id']
    previous = s.one('SELECT * FROM stage_runs WHERE id=:id', id=rid)
    if not previous:
        inputs = {k: value for k, value in state['heads'].items() if k in ('m1', 'm2', 'm3')}
        with s.transaction() as conn:
            s.sql(conn, '''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,
                parameters,config_path,folder,log_path,created_at,request_id)
                VALUES(:id,:e,'m4','historical','importing',:sig,:inputs,:params,:cfg,:folder,:log,:t,:req)''',
                id=rid, e=eid, sig=s.digest(verification), inputs=s.dump(inputs),
                params=s.dump({'iterations':10000,'historical_import':True,'new_training_steps':0}),
                cfg=str(folder/'config.yaml'), folder=str(folder), log=str(folder/'import.log'),
                t=s.now(), req=batch.name+':case00:historical-m4')
            s.sql(conn, '''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,
                external_api,status,created_at) VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'importing',:t)''',
                id=s.uid(), e=eid, r=rid, model='Historical dual-mask background optimization, import only',
                prompt=v.experiment(eid)['replacement_text'], inputs=s.dump(inputs), t=s.now())
    if not previous or previous['status'] != 'completed':
        v.finish(rid, folder/'m4', seconds=0)
    m2 = v.stage_dir(state['heads']['m2'])
    item = {'key':'case00', 'label':'黄色六轴工业机械臂（历史案例）',
            'description':v.experiment(eid)['replacement_text'], 'experiment_id':eid,
            'source_experiment_id':SOURCE_EXPERIMENT, 'source_run':str(root),
            'input_path':str(m2/'edited_scene.png'), 'input_sha256':s.sha(m2/'edited_scene.png'),
            'anchor':read_json(m2/'anchor_camera.json').get('name', 'avp_099'),
            'historical_import':True, 'postprocess_status':'completed',
            'alignment':{'source':str(native/'transform.json'), 'object_world_sha256':expected['object'],
                         'changed_on_import':False},
            'training_budget_note':'Historical total 10000 steps (8000 resumed from 2000); other cases 250.'}
    write_json(batch/'historical_cases.json', {'items':[item], 'new_model_calls':0})
    write_json(batch/'reviews/case00.json', {
        'object_world_sha256':expected['object'], 'historical_import':True,
        'quality_notes_zh':'复用此前已验证的 10,000 步结果，本次未重新训练或对齐。训练预算与其余 250 步案例不同。',
        'source_verification':str(root/'run_verification.json'), 'candidate_guard_passed':False})

    preview = batch/'results/case00'
    preview.mkdir(parents=True, exist_ok=True)
    panels = [('Fixed object', native/'renders/anchor_object.png'),
              ('M3 / original background', root/'m4/renders/anchor_before.png'),
              ('M4 actual 10000 / guard FAIL', root/'m4/renders/anchor_candidate_step10000.png'),
              ('Guarded final / selected step 0', root/'m4/renders/anchor_after.png')]
    board = Image.new('RGB', (2400,660), 'white'); draw=ImageDraw.Draw(board)
    for index,(label,path) in enumerate(panels):
        im=Image.open(path).convert('RGB'); resized=ImageOps.contain(im,(600,620))
        board.paste(resized,(index*600+(600-resized.width)//2,40))
        draw.text((index*600+10,12),label,fill='black')
    board.save(preview/'stages_same_camera.jpg', quality=94)
    for name in ('summary.json', 'validation/supervision_union_masks.png', 'validation/anchor_comparison.png'):
        shutil.copy2(root/'m4'/name, preview/Path(name).name)
    for path in (root/'m4/snapshots/iteration_010000/renders').glob('*comparison.png'):
        shutil.copy2(path, preview/path.name)
    write_json(folder/'import_verification.json', {
        'status':'verified_read_only_import', 'object_sha256':expected['object'],
        'candidate_scene_sha256':expected['scene'], 'source_inputs_sha256_verified':True,
        'original_models_modified':False, 'new_training_steps':0, 'qwen_calls':0, 'trellis_calls':0,
        'source_verification_sha256':s.sha(root/'run_verification.json'),
        'historical_summary_sha256':s.sha(root/'m4/summary.json'),
        'selected_step':report['selected_step'], 'actual_step':10000,
        'source_run':str(root), 'experiment_id':eid, 'm4_run_id':rid})
    print({'case':'case00', 'experiment':eid, 'run':rid, 'source':str(root)}, flush=True)


def bundle(batch):
    manifest=read_json(batch/'final_manifest_with_case00.json')
    assert len(manifest['items'])==10 and manifest['items'][0]['key']=='case00'
    original=batch/(batch.name+'_previews.zip')
    original_hash=s.sha(original)
    target=batch/'industrial_cases10_with_case00_previews.zip'
    root=s.BASE/HISTORICAL_RUN
    readme='''10 个案例：case00 是此前黄色六轴机械臂，其余为本批 9 个新设备。
case00 总迭代 10000 步（从 2000 步继续 8000 步），其余案例为 250 步，不能当作相同训练预算。
本次仅导入已有结果，没有重新调用 Qwen、TRELLIS.2 或训练 M4。
case00 实际 10000 步候选未通过既有环境保护检查；保护规则选出的 scene_final.ply 为第 0 步。
检查页同时提供实际候选和保护规则结果，不应混淆。
final_manifest_with_case00.json 是当前十案例索引；nine_case_original/ 保留原九案例报告和来源。
大体积 PLY/GLB 仍在服务器，检查页可下载。case00 多视角检查图复用已保存结果，未生成新视频。
检查页：http://127.0.0.1:18086/static/batches/qwen9_completion_20261107/index.html#case00
'''
    (batch/'RESULTS_WITH_CASE00_ZH.txt').write_text(readme, encoding='utf-8')
    old_metadata={'audit_report.json','manifest.json','final_manifest.json','postprocess.json','adaptations.json',
                  'RESULTS_README.txt','RESULTS_ZH.txt'}
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as new:
        for info in old.infolist():
            name='nine_case_original/'+info.filename if info.filename in old_metadata else info.filename
            if info.filename.startswith('overview/'):
                name='nine_case_original/'+info.filename
            with old.open(info) as src, new.open(name,'w') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
        for name in ('RESULTS_WITH_CASE00_ZH.txt','final_manifest_with_case00.json','historical_cases.json'):
            new.write(batch/name,name)
        for path in (batch/'review').glob('*with_case00.jpg'):
            new.write(path,'overview/'+path.name)
        for path in (batch/'results/case00').iterdir():
            new.write(path,'case00/'+path.name)
        for name in ('registration.json','import_verification.json'):
            new.write(batch/'case00_historical'/name,'case00/'+name)
        for name in ('run_verification.json','source_preservation.json','config.yaml','REPORT.txt'):
            new.write(root/name,'case00/provenance/'+name)
        for name in ('import_robot_case00.py','collect_qwen_batch_results.py'):
            new.write(s.BASE/'scripts'/name,'current_scripts/'+name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        count=len(archive.namelist())
        assert len(set(archive.namelist()))==count
    assert s.sha(original)==original_hash
    record={'path':str(target),'sha256':s.sha(target),'bytes':target.stat().st_size,'entries':count,
            'zip_crc_passed':True,'original_nine_case_bundle_unchanged':True,
            'original_nine_case_bundle_sha256':original_hash,'cases':10}
    write_json(batch/'bundle_with_case00_verification.json',record)
    print(record,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True,type=Path)
    parser.add_argument('--bundle-only',action='store_true');args=parser.parse_args()
    path=args.batch.resolve();path.relative_to(s.BASE/'outputs')
    bundle(path) if args.bundle_only else import_case(path)

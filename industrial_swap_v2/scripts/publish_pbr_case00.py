"""Audit and publish completed PBR case00; no generation or optimization calls."""
import argparse
import html
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import quote
import numpy as np
import yaml
from industrial_swap_v2.core.context import read_json,write_json,sha256
from industrial_swap_v2.core.scene import read_vertices
from industrial_swap_v2.core.pbr_math import sh_basis
from industrial_swap_v2.webapp import store as s,service as v


def file_url(rid,name):return '/api/runs/'+rid+'/file?name='+quote(name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    root=args.run.resolve();root.relative_to(s.BASE/'outputs');m3=root/'m3';m4=root/'m4'
    for stage in ('m3','m4'):
        receipt=read_json(root/'records'/f'{stage}_status.json');assert receipt['status']=='complete'
        for artifact in receipt['artifacts']: assert sha256(artifact['path'])==artifact['sha256']
    originals=read_json(root/'source_preservation.json')
    for path,checksum in originals.items(): assert sha256(path)==checksum,path
    a=read_json(m3/'pbr_training_summary.json');b=read_json(m4/'summary.json')
    validation=read_json(m3/'pbr_views/dataset_validation.json')
    assert validation['train_count']==30 and validation['validation_count']==6
    sampling=[json.loads(line) for line in (m3/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in sampling]==list(range(1,a['iterations']+1))
    forbidden={r['name'] for r in read_json(m3/'pbr_views/cameras.json')['views'] if r['split']=='validation'}
    assert not any(r['camera'] in forbidden for r in sampling)
    assert sha256(m4/'object_world.ply')==sha256(m4/'object_fixed.ply')
    local=read_vertices(m3/'object_refined_local.ply');world=read_vertices(m4/'object_world.ply')
    assert len(local)==len(world)==200000
    pose=read_json(m4/'transform.json');R=np.asarray(pose['rotation']);t=np.asarray(pose['translation']);scale=pose['scale']
    xyz=lambda v:np.stack([v[n] for n in ('x','y','z')],1)
    np.testing.assert_allclose(xyz(world),xyz(local)@R.T*scale+t,atol=2e-6)
    rng=np.random.default_rng(42);ids=rng.integers(len(local),size=100)
    directions=rng.normal(size=(100,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    coeff=lambda v:np.concatenate((np.stack([v[f'f_dc_{i}'] for i in range(3)],1)[:,:,None],
        np.stack([v[f'f_rest_{i}'] for i in range(45)],1).reshape(-1,3,15)),axis=-1)
    old=np.einsum('ncj,nj->nc',coeff(local[ids]),sh_basis(directions@R))
    new=np.einsum('ncj,nj->nc',coeff(world[ids]),sh_basis(directions))
    np.testing.assert_allclose(old,new,atol=2e-6)
    for cache in (m4/'multiview_supervision/cache').glob('*.npz'):
        x=np.load(cache);assert np.array_equal(x['environment_mask'],~(x['old_mask']|x['new_mask']))
        assert np.array_equal(x['overlap'],x['old_mask']&x['new_mask'])
    del local,world
    report={'case':'case00','status':'completed_actual_runs','m3_iterations':a['iterations'],'m3_best_step':a['best_step'],
        'm3_validation_baseline':a['baseline']['validation'],'m3_validation_best':a['best']['validation'],
        'm4_iterations':b['checks']['training_steps'],'m4_selected_step':b['selected_step'],
        'm4_candidate_guard_passed':b['candidate_guard_passed'],'m4_baseline':b['baseline'],
        'm4_candidate':b['candidate'],'m4_checks':b['checks'],'m3_seconds':a['seconds'],
        'old_sources_sha256_unchanged':True,'m3_validation_cameras_never_trained':True,
        'sh_rotation_color_error_max':float(abs(old-new).max()),'dual_mask_union_verified':True,
        'qwen_calls':0,'trellis_calls':0,'other_cases_executed':0,
        'blender':read_json(m3/'pbr_views/cameras.json')['blender_version'],
        'depth_calibration':read_json(m3/'pbr_views/ray_depth_calibration.json'),
        'source_run':str(root)}
    write_json(root/'audit_report.json',report)
    registration=root/'web_registration.json'
    if registration.exists(): state=read_json(registration)
    else:
        provenance=read_json(root/'provenance.json')
        eid=v.create('case00 · PBR 多视角 Gaussian / 双 Mask 场景融合',copy_from=provenance['source_experiment'])
        cfg=yaml.safe_load((root/'config.yaml').read_text())
        heads={k:val for k,val in provenance['source_heads'].items() if k in ('m1','m2','materials')}
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET config=:cfg,heads=:h WHERE id=:id',cfg=s.dump(cfg),h=s.dump(heads),id=eid)
        state={'experiment_id':eid,'runs':{}}
        write_json(registration,state)
    for stage in ('m3','m4'):
        if stage in state['runs']: continue
        rid=s.uid();exp=v.experiment(state['experiment_id']);heads=s.unpack(exp['heads'])
        inputs={k:heads[k] for k in ('m1','m2','m3') if k in heads and k!=stage}
        with s.transaction() as conn:
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
                config_path,folder,log_path,created_at,request_id)
                VALUES(:id,:e,:stage,'historical','importing',:sig,:inputs,:params,:cfg,:folder,:log,:t,:req)''',
                id=rid,e=state['experiment_id'],stage=stage,sig=s.digest(read_json(root/'records'/f'{stage}_status.json')),
                inputs=s.dump(inputs),params=s.dump({'iterations':a['iterations'] if stage=='m3' else b['checks']['training_steps']}),
                cfg=str(root/'config.yaml'),folder=str(root),log=str(root/'logs'/f'{stage}.log'),t=s.now(),req=root.name+':verified-'+stage)
            s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,external_api,status,created_at)
                VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'importing',:t)''',id=s.uid(),e=state['experiment_id'],r=rid,
                model='PBR RGB+Depth Gaussian' if stage=='m3' else 'Dual mask scene integration',
                prompt=exp['replacement_text'],inputs=s.dump(inputs),t=s.now())
        v.finish(rid,root/stage,seconds=a['seconds'] if stage=='m3' else read_json(m4/'background_optimization/summary.json')['seconds'])
        state['runs'][stage]=rid;write_json(registration,state)
    assets={}
    for path in (root/'previews').iterdir():
        if path.suffix in ('.jpg','.png','.mp4'):
            assets[path.name]=s.register(path,'video' if path.suffix=='.mp4' else 'preview',
                source='verified_pbr_case00_export',experiment=state['experiment_id'],trusted=True)
    def asset(name):return '/api/assets/'+assets[name]+'/file'
    rid3=state['runs']['m3'];rid4=state['runs']['m4']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>case00 · PBR Gaussian</title><style>body{max-width:1600px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;background:#f3f5f7;color:#182533}section{background:white;padding:24px;margin:20px 0;border-radius:12px}img,video{width:100%}a{color:#1259b0;margin-right:14px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd4df;padding:8px;text-align:left}</style>'''
    page+='<h1>case00 · PBR 多视角设备 Gaussian → 双 Mask 场景融合</h1><p>复用机械臂原 GLB，工厂 HDRI：autoshop_01_4k.hdr。仅运行 case00；无 Qwen / TRELLIS 新生成。</p>'
    page+=f'<p><a href="/#experiment/{state["experiment_id"]}">实验与素材</a><a href="/viewer?experiment={state["experiment_id"]}">保护规则选出场景的漫游</a></p>'
    page+=f'<section><h2>M3：36 视角 PBR 监督，4,000 步，最佳 {a["best_step"]} 步</h2><p>30 个训练视角、6 个独立验证视角；仅 RGB + Depth 两项损失。下面每行依次为 PBR 目标、初始化高斯、优化高斯。</p><table><tr><th>验证指标</th><th>初始化</th><th>最佳模型</th></tr>'
    for label,key in [('PSNR (dB)','psnr'),('设备前景 PSNR (dB)','foreground_psnr'),('归一化深度 MAE','depth_mae_relative'),('轮廓 IoU','silhouette_iou')]:
        page+=f'<tr><td>{label}</td><td>{a["baseline"]["validation"][key]:.5f}</td><td>{a["best"]["validation"][key]:.5f}</td></tr>'
    page+=f'</table><img src="{file_url(rid3,"renders/validation_comparison.jpg")}"><a href="{file_url(rid3,"object_refined_local.ply")}&download=true">优化后的独立设备 PLY</a><a href="{file_url(rid3,"checkpoints/best.pth")}&download=true">最佳 checkpoint</a><a href="{file_url(rid3,"trellis_asset.glb")}&download=true">原始 PBR GLB</a><video controls preload="none" src="{file_url(rid3,"renders/pbr_gaussian_orbit_1.mp4")}"></video></section>'
    page+=f'<section><h2>M4：固定设备，局部背景优化 {b["checks"]["training_steps"]} 步</h2><p>实际末步候选环境检查：{"通过" if b["candidate_guard_passed"] else "未通过"}；默认 scene_final.ply 选择第 {b["selected_step"]} 步。选中 0 步表示保留原背景，不能当作末步优化场景。实际候选另行公开。</p><p>四列依次：独立设备、插入后未优化、实际末步候选、保护规则场景。</p>'
    page+=f'<img src="{asset("anchor_stages.jpg")}">'
    for name in ('scene_final.ply','scene_candidate_step250.ply','object_world.ply','background_optimization/background_optimized.ply'):
        page+=f'<a href="{file_url(rid4,name)}&download=true">{name.split("/")[-1]}</a>'
    page+=f'<p><a href="{file_url(rid4,"summary.json")}">M4 逐视角与冻结检查报告</a><a href="{file_url(rid4,"validation/supervision_union_masks.png")}">双 Mask 监督</a></p>'
    for name in ('orbit_insertion_actual_guarded.mp4','trajectory_insertion_actual_guarded.mp4'):
        page+=f'<video controls preload="none" src="{asset(name)}"></video>'
    for name in sorted(assets):
        if name.endswith('.jpg') and name!='anchor_stages.jpg':page+=f'<img loading="lazy" src="{asset(name)}">'
    page+='</section><p>限制：SH3 学习固定 HDRI 下的外观，不等于可任意重新打光的 PBR 材质；原始未观测区域和合法前景不能由背景透明度优化完全恢复。历史实验均保留。</p></html>'
    dest=s.BASE/'webapp/static/batches'/root.name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'index.html').write_text(page,encoding='utf-8');(root/'results_index.html').write_text(page,encoding='utf-8')
    state.update(page='/static/batches/'+root.name+'/index.html',preview_assets=assets)
    write_json(registration,state)
    bundle=root/'case00_pbr_previews_and_supervision.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for folder in ('previews','logs'):
            for path in (root/folder).rglob('*'):
                if path.is_file():archive.write(path,str(path.relative_to(root)))
        for path in (m3/'pbr_views').rglob('*'):
            if path.is_file() and 'cache' not in path.parts:archive.write(path,str(path.relative_to(root)))
        for path in (m3/'renders').glob('*'):
            if path.suffix in ('.jpg','.png','.mp4'):archive.write(path,str(path.relative_to(root)))
        for name in ('config.yaml','source_preservation.json','provenance.json','audit_report.json','web_registration.json',
            'm3/pbr_training_summary.json','m3/training.jsonl','m3/evaluations.json','m3/baseline_metrics.json',
            'm4/summary.json','m4/integration.json','m4/transform.json','m4/validation/supervision_union_masks.png'):
            archive.write(root/name,name)
        code=['core/pbr_math.py','core/pbr_gaussian_optimizer.py','core/pbr_pipeline.py','core/scene_integration.py',
              'scripts/blender_pbr_views.py','scripts/validate_pbr_calibration.py','scripts/prepare_pbr_case00.py',
              'scripts/render_pbr_scene_results.py','scripts/publish_pbr_case00.py']
        for name in code:archive.write(s.BASE/name,'code/'+name)
        archive.write(s.BASE/'scripts/setup_pbr.sh','code/scripts/setup_pbr.sh')
        for name in ('source_snapshot.tar.gz','implementation_provenance.json','blender_interface_verification.json',
                     'blender_pbr_views.executed.py','live_web_verification.json'):
            if (root/name).is_file(): archive.write(root/name,name)
    with zipfile.ZipFile(bundle) as archive:assert archive.testzip() is None
    write_json(root/'delivery.json',{'bundle':str(bundle),'sha256':sha256(bundle),'bytes':bundle.stat().st_size,
        'page':state['page'],'experiment':state['experiment_id'],'model_files_on_server':True,
        'sources_unchanged':True})
    print(read_json(root/'delivery.json'),flush=True)


if __name__=='__main__':main()

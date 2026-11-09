"""Register a completed alignment/M4 run without altering its source experiment."""
import argparse
import html
from pathlib import Path
from urllib.parse import quote
import zipfile
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256
from industrial_swap_v2.webapp import service as v,store as s


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--label',default='case02')
    p.add_argument('--page-name')
    a=p.parse_args();ctx=Context(a.config);label=html.escape(a.label);page_name=a.page_name or ctx.out.name
    out=ctx.stage('m4');report=read_json(out/'alignment/comparison_report.json');summary=read_json(out/'summary.json')
    receipt=read_json(ctx.records/'m4_status.json');assert receipt['status']=='complete'
    for key in ('object_file_sha256_unchanged','object_scene_rows_bitwise_identical','unselected_background_bitwise_identical','original_sources_sha256_unchanged'):
        assert summary['checks'][key],key
    record_path=ctx.out/'web_registration.json'
    if record_path.exists():record=read_json(record_path)
    else:
        source=read_json(ctx.out/'provenance.json')
        old=s.one("SELECT experiment_id FROM stage_runs WHERE config_path=:p AND stage='m4' AND status='completed' ORDER BY created_at DESC LIMIT 1",p=source['source_config'])
        if not old:raise ValueError('Source experiment is not registered')
        eid=v.create(a.label+' · Gaussian 空间重配准 · M4 2000',copy_from=old['experiment_id'])
        heads=s.unpack(v.experiment(eid)['heads']);heads.pop('m4',None)
        rid=s.uid();inputs={k:heads[k] for k in ('m1','m2','m3')}
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET config=:c,heads=:h WHERE id=:id',c=s.dump(ctx.cfg),h=s.dump(heads),id=eid)
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
                config_path,folder,log_path,created_at,request_id)
                VALUES(:id,:e,'m4','historical','importing',:sig,:inputs,:params,:cfg,:folder,:log,:t,:req)''',
                id=rid,e=eid,sig=s.digest(receipt),inputs=s.dump(inputs),params=s.dump(ctx.cfg['m4']),cfg=str(ctx.config_path),
                folder=str(ctx.out),log=str(ctx.logs/'m4.log'),t=s.now(),req=str(ctx.out)+':registered_m4')
            s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,external_api,status,created_at)
                VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'importing',:t)''',id=s.uid(),e=eid,r=rid,
                model='Rasterized Gaussian registration + dual-mask M4',prompt=ctx.cfg['m2']['replacement_text'],inputs=s.dump(inputs),t=s.now())
        record={'experiment_id':eid,'m4_run':rid};write_json(record_path,record)
    rid=record['m4_run']
    if v.run(rid)['status']!='completed':v.finish(rid,out)
    url=lambda name:'/api/runs/'+rid+'/file?name='+quote(name)
    def picture(name):return '<img loading="lazy" src="'+url('alignment/'+name)+'">'
    before=report['baseline'];after=report['selected'];steps=summary['checks']['training_steps']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>case02 Gaussian 空间配准</title><style>body{font:16px/1.7 system-ui;max-width:1700px;margin:30px auto;background:#f4f6f8;color:#172334;padding:20px}section{padding:22px;background:white;margin:20px 0;border-radius:12px}img{width:100%;height:auto}a{display:inline-block;margin:6px 20px 6px 0;color:#1259ad}table{border-collapse:collapse}td,th{padding:7px 20px;border-bottom:1px solid #ddd}</style>'''
    page=page.replace('<title>case02 Gaussian 空间配准</title>','<title>'+label+' Gaussian 空间配准</title>')
    page+='<h1>'+label+'：真实 Gaussian 渲染空间配准</h1><p>复用同一 M1、M2 和 M3；只优化四个空间参数，之后固定设备执行 M4。原实验保留。</p>'
    page+=f'<table><tr><th>指标</th><th>旧变换</th><th>新变换</th></tr><tr><td>Anchor Mask IoU</td><td>{before["mask_iou"]:.4f}</td><td>{after["mask_iou"]:.4f}</td></tr>'
    for label,key in [('统一尺度','scale'),('Yaw / 度','yaw_degrees')]:
        page+=f'<tr><td>{label}</td><td>{report["pose_comparison"]["old"][key]:.4f}</td><td>{report["pose_comparison"]["new"][key]:.4f}</td></tr>'
    page+='</table>'
    adaptation=report['space'].get('space_override')
    if adaptation:
        page+=f'<p>经用户确认，本次 M4 生成空间高度由 {adaptation["old_height"]:.4f} 调整为 {adaptation["new_height"]:.4f}，上限仍在原 M1 编辑空间内；底面、占地和原 M2 文件保持不变。</p>'
    page+=f'<p>最大生成框越界比例 {after["max_volume_overflow_fraction"]:.2%}（按对应轴长度归一化，采用稳健高斯支持范围）；底部支撑高度固定。M4 实际 {steps} 步，环境检查'+('通过' if summary['candidate_guard_passed'] else '未通过')+f'；保护规则选择第 {summary["selected_step"]} 步。实际末步场景单独提供。</p>'
    page+='<nav>'
    for name,label in [('transform.json','新空间变换'),('object_world.ply','新设备世界 Gaussian'),('scene_initial.ply','新位姿初始场景'),(f'scene_candidate_step{steps}.ply',f'实际 {steps} 步场景'),('scene_final.ply','保护规则最终场景'),('alignment/comparison_report.json','完整指标与来源'),('alignment/anchor_depth_z.npy','Anchor 相机 Z 深度')]:
        page+=f'<a href="{url(name)}&download=true">{label}</a>'
    page+=f'<a href="/#experiment/{record["experiment_id"]}">实验与日志</a><a href="/viewer?experiment={record["experiment_id"]}">场景漫游</a></nav>'
    for title,names in [('Qwen / 旧位姿 / 新位姿',['reference_old_new.jpg']),('Mask 叠加：红色高斯 Alpha，绿色 Qwen Mask',['old_overlay.png','new_overlay.png']),('同背景的新旧插入结果',['anchor_scene_old_new.jpg']),('新位姿 M4：初始 / 实际 2000 / 保护规则',['anchor_m4_comparison.jpg'])]:
        page+='<section><h2>'+title+'</h2>'+''.join(picture(n) for n in names)+'</section>'
    for i,view in enumerate(report['other_views']):
        page+='<section><h2>原始配准视角 '+view['name']+'</h2>'+picture(view['comparison'])+picture(f'view_{i:02d}_m4_comparison.jpg')+'</section>'
    page+='</html>';folder=s.BASE/'webapp/static/batches'/page_name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'index.html').write_text(page,encoding='utf-8')
    archive=ctx.out/'case02_alignment_results.zip' if a.label=='case02' else ctx.out/'alignment_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in ('config.yaml','provenance.json','source_preservation.json','web_registration.json','m4_height_adaptation.json','m4/transform.json','m4/integration.json','m4/summary.json','logs/alignment_retry.log','logs/m4.log'):
            file=ctx.out/name
            if file.exists():z.write(file,name)
        for file in (out/'alignment').iterdir():
            if file.suffix in ('.png','.jpg','.json'):z.write(file,'alignment/'+file.name)
        for name in ('core/alignment.py','core/gaussian_registration.py','core/scene_integration.py','core/m2_space.py',
            'scripts/prepare_reference_alignment.py','scripts/inspect_reference_alignment.py','scripts/publish_reference_alignment.py','tests/test_gaussian_registration.py'):
            z.write(s.BASE/name,'code/'+name)
    write_json(ctx.out/'delivery.json',{'page':'/static/batches/'+page_name+'/index.html','archive':str(archive),
        'sha256':sha256(archive),'bytes':archive.stat().st_size,**record})
    print(read_json(ctx.out/'delivery.json'))


if __name__=='__main__':main()

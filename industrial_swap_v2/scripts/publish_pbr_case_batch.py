"""Compact live gallery and result package for the existing ten PBR cases."""
import argparse
import csv
import html
from pathlib import Path
from urllib.parse import quote
import zipfile
import yaml
from PIL import Image,ImageDraw,ImageOps
from industrial_swap_v2.core.context import read_json,write_json,sha256
from industrial_swap_v2.webapp import service as v,store as s


def url(rid,name):return '/api/runs/'+rid+'/file?name='+quote(name)


def register(item,published):
    key=item['key'];root=Path(item['folder'])
    if key not in published:
        source=read_json(root/'provenance.json');eid=v.create(key+' · '+item['label']+' · PBR / M4 2000',copy_from=item['experiment_id'])
        cfg=yaml.safe_load((root/'config.yaml').read_text());heads={k:value for k,value in source['source_heads'].items() if k in ('m1','m2','materials')}
        if item.get('reuse_m3_run'):heads['m3']=item['reuse_m3_run']
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET config=:c,heads=:h WHERE id=:id',c=s.dump(cfg),h=s.dump(heads),id=eid)
        published[key]={'experiment':eid,'runs':{'m3':item['reuse_m3_run']} if item.get('reuse_m3_run') else {}}
    record=published[key]
    for stage in ('m3','m4'):
        if stage in record['runs']:continue
        receipt=read_json(root/'records'/f'{stage}_status.json');assert receipt['status']=='complete'
        request=root.parent.name+':'+key+':'+stage
        prior=s.one('SELECT id,status FROM stage_runs WHERE request_id=:r',r=request)
        if prior and prior['status']=='completed':record['runs'][stage]=prior['id'];continue
        rid=prior['id'] if prior else s.uid();exp=v.experiment(record['experiment']);heads=s.unpack(exp['heads'])
        inputs={k:heads[k] for k in ('m1','m2','m3') if k in heads and k!=stage}
        if not prior:
            with s.transaction() as conn:
                s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
                    config_path,folder,log_path,created_at,request_id)
                    VALUES(:id,:e,:stage,'historical','importing',:sig,:inputs,:params,:cfg,:folder,:log,:t,:req)''',
                    id=rid,e=record['experiment'],stage=stage,sig=s.digest(receipt),inputs=s.dump(inputs),
                    params=s.dump({'iterations':4000 if stage=='m3' else 2000}),cfg=str(root/'config.yaml'),
                    folder=str(root),log=str(root/'logs'/f'{stage}.log'),t=s.now(),req=request)
                s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,external_api,status,created_at)
                    VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'importing',:t)''',id=s.uid(),e=record['experiment'],r=rid,
                    model='PBR RGB+Depth Gaussian' if stage=='m3' else 'Dual-mask background 2000 steps',
                    prompt=exp['replacement_text'],inputs=s.dump(inputs),t=s.now())
        v.finish(rid,root/stage,seconds=item['stages'].get(stage,{}).get('seconds',0))
        record['runs'][stage]=rid
    if 'preview_assets' not in record:
        record['preview_assets']={}
        for path in (root/'previews').iterdir():
            if path.suffix in ('.png','.jpg','.mp4'):
                record['preview_assets'][path.name]=s.register(path,'video' if path.suffix=='.mp4' else 'preview',
                    name=key+' '+path.name,source='pbr_2000_completed_batch',experiment=record['experiment'],trusted=True)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True,type=Path);parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args();batch=args.batch.resolve();state=read_json(batch/'batch.json')
    pubfile=batch/'published.json';published=read_json(pubfile) if pubfile.exists() else {}
    completed=sum(x['status']=='completed' for x in state['items']);failed=sum(x['status']=='failed' for x in state['items'])
    cards=[];rows=[];size=(1440,520*((len(state['items'])+2)//3))
    actual=Image.new('RGB',size,'white');guarded=Image.new('RGB',size,'white')
    adraw=ImageDraw.Draw(actual);gdraw=ImageDraw.Draw(guarded)
    for index,item in enumerate(state['items']):
        key=item['key'];root=Path(item['folder']);title=html.escape(key+' · '+item['label'])
        x=index%3*480;y=index//3*520
        if item['status']!='completed':
            status={'pending':'等待执行','running':'运行中','failed':'执行失败'}[item['status']]
            detail=status+' · '+item.get('active_stage','')
            progress=root/'m3/training_progress.json'
            if item.get('active_stage')=='m3' and progress.exists():detail+=f' · {read_json(progress)["step"]}/4000'
            if item.get('active_stage')=='m4':
                path=root/'m4/background_optimization/progress.json'
                if path.exists():detail+=' · '+str(read_json(path).get('step',''))+'/2000'
            cards.append(f'<section id="{key}"><h2>{title}</h2><p>{html.escape(detail)}</p></section>')
            adraw.text((x+8,y+10),key+' | '+item['status'],fill='black');continue
        r=register(item,published);write_json(pubfile,published)
        a=read_json(root/'m3/pbr_training_summary.json');b=read_json(root/'m4/summary.json')
        assert a['iterations']==4000 and b['checks']['training_steps']==2000
        for check in ('object_file_sha256_unchanged','object_scene_rows_bitwise_identical','unselected_background_bitwise_identical','original_sources_sha256_unchanged'):
            assert b['checks'][check],(key,check)
        row={'case':key,'label':item['label'],'m3_reused':item['reuse_pbr_m3'],'m3_best_step':a['best_step'],
            'foreground_psnr_before':a['baseline']['validation']['foreground_psnr'],
            'foreground_psnr_after':a['best']['validation']['foreground_psnr'],
            'depth_mae_before':a['baseline']['validation']['depth_mae_relative'],
            'depth_mae_after':a['best']['validation']['depth_mae_relative'],
            'm4_steps':2000,'guard_passed':b['candidate_guard_passed'],'selected_step':b['selected_step'],
            'anchor_visibility_before':b['baseline']['anchor']['visibility'],
            'anchor_visibility_after':b['candidate']['anchor']['visibility'],
            'outside_union_mae':b['candidate']['anchor']['outside_union_mae'],
            'experiment':r['experiment'],'folder':str(root)}
        rows.append(row);rid3=r['runs']['m3'];rid4=r['runs']['m4']
        asset=lambda name:'/api/assets/'+r['preview_assets'][name]+'/file'
        for board,draw,name,label in ((actual,adraw,'anchor_candidate_step2000.png','actual 2000 | guard '+('PASS' if b['candidate_guard_passed'] else 'FAIL')),
            (guarded,gdraw,'anchor_after.png',f'guarded step {b["selected_step"]}')):
            im=Image.open(root/'m4/renders'/name).convert('RGB');board.paste(ImageOps.contain(im,(480,480)),(x,y+40));draw.text((x+8,y+10),key+' | '+label,fill='black')
        content=f'<section id="{key}"><h2>{title}</h2><p>M3：'+('复用已完成 PBR 模型' if item['reuse_pbr_m3'] else '已完成 4,000 步 PBR 训练')
        content+=f'，最佳 {a["best_step"]} 步；验证前景 PSNR {row["foreground_psnr_before"]:.2f} → {row["foreground_psnr_after"]:.2f} dB。</p>'
        content+=f'<p>M4 已完成 2,000 步；末步检查：{"通过" if b["candidate_guard_passed"] else "未通过"}，默认最终场景选择第 {b["selected_step"]} 步。下图四列：固定设备 / 插入后未优化 / 实际 2,000 步 / 保护规则场景。</p>'
        content+=f'<img loading="lazy" src="{asset("anchor_stages.jpg")}"><nav><a href="/#experiment/{r["experiment"]}">实验与日志</a>'
        for name,label in [('scene_candidate_step2000.ply','实际 2000 步场景 PLY'),('scene_final.ply','保护规则场景 PLY'),('object_world.ply','设备世界 PLY')]:
            content+=f'<a href="{url(rid4,name)}&download=true">{label}</a>'
        content+=f'<a href="{url(rid3,"object_refined_local.ply")}&download=true">设备局部 PLY</a><a href="{url(rid3,"renders/validation_comparison.jpg")}">PBR / 初始 / 优化高斯对比</a><a href="{url(rid4,"summary.json")}">M4 指标</a><a href="/viewer?experiment={r["experiment"]}">保护规则场景漫游</a></nav>'
        for name in ('orbit_insertion_actual_guarded.mp4','trajectory_insertion_actual_guarded.mp4'):
            content+=f'<video controls preload="none" src="{asset(name)}"></video>'
        cards.append(content+'</section>')
    report={'batch':batch.name,'completed':completed,'failed':failed,'total':len(state['items']),
        'm3_steps':4000,'m4_steps':2000,'qwen_calls':0,'trellis_calls':0,'items':rows,
        'candidate_guard_passed':sum(r['guard_passed'] for r in rows)}
    write_json(batch/'results_summary.json',report)
    if rows:
        with (batch/'results_summary.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    actual.save(batch/'actual_2000_overview.jpg');guarded.save(batch/'guarded_overview.jpg')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10 案例 PBR / M4 2000</title><style>body{font:16px/1.65 system-ui;background:#f3f5f7;color:#182533;max-width:1700px;margin:32px auto;padding:20px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}img{width:100%}video{width:49%;margin-top:16px}a{display:inline-block;color:#1259b0;margin:7px 14px 5px 0}@media(max-width:700px){video{width:100%}}</style>'''
    if completed+failed<len(state['items']):page+='<meta http-equiv="refresh" content="30">'
    page+=f'<h1>10 个已有案例 · PBR Gaussian + 2,000 步局部背景优化</h1><p>完成 {completed}/10；失败 {failed}。GLB、参考图像和已有空间位置均复用。case00 的 PBR M3 已完成，直接复用；其余沿用相同 36 视角、4,000 步 M3 配置。</p><p>默认最终场景经过环境保护检查；实际 2,000 步候选单独提供。第 0 步表示保留输入背景。旧实验不覆盖，不调用 Qwen 或 TRELLIS.2。</p>'
    page+=''.join(cards)+'</html>'
    dest=s.BASE/'webapp/static/batches'/batch.name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'index.html').write_text(page,encoding='utf-8');(batch/'results_index.html').write_text(page,encoding='utf-8')
    if args.bundle:
        assert completed==len(state['items']) and not failed
        path=batch/'pbr_10cases_m4_2000_previews.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name in ('batch.json','published.json','results_summary.json','results_summary.csv','actual_2000_overview.jpg','guarded_overview.jpg'):
                z.write(batch/name,name)
            for item in state['items']:
                root=Path(item['folder']);key=item['key']
                for p in (root/'previews').iterdir():
                    if p.is_file():z.write(p,key+'/previews/'+p.name)
                for name in ('config.yaml','provenance.json','m3/pbr_training_summary.json','m3/pbr_views/cameras.json',
                    'm3/renders/validation_comparison.jpg','m4/summary.json','m4/transform.json',
                    'm4/validation/supervision_union_masks.png','logs/m3.log','logs/m4.log'):
                    z.write(root/name,key+'/'+name)
            for name in ('run_pbr_case_batch.py','publish_pbr_case_batch.py','render_pbr_scene_results.py'):
                z.write(s.BASE/'scripts'/name,'scripts/'+name)
        write_json(batch/'delivery.json',{'zip':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),
            'page':'/static/batches/'+batch.name+'/index.html','completed':completed})
    print({'completed':completed,'failed':failed,'page':'/static/batches/'+batch.name+'/index.html'},flush=True)


if __name__=='__main__':main()

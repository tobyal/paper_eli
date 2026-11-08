"""Queue the approved 10 x 3 descriptions; export only verified prepared inputs."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from industrial_swap_v2.webapp import store as s,service as v,m1_cases,m2_cases
from industrial_swap_v2.core.context import read_json,write_json

FILES=(*v.MATERIALS,'depth.npz','generation_inputs.json')


def submit(folder):
    catalog=read_json(s.BASE/'configs/m2_case_description_options.json')
    assert len(catalog['cases'])==10 and all(len(c['options'])==3 for c in catalog['cases'])
    for excluded in catalog.get('archived_cases',[]):
        with s.transaction() as conn:
            s.sql(conn,'INSERT OR IGNORE INTO m1_case_archives VALUES(:id,:reason,:t)',id=excluded['case_id'],
                  reason='User excluded duplicate case 11 from the 30-input batch.',t=s.now())
    manifest=read_json(folder/'manifest.json') if (folder/'manifest.json').exists() else {
        'batch':folder.name,'created_at':s.now(),'scope':'Qwen inputs only; no generation API',
        'targets':10,'variants_per_target':3,'excluded_case':11,'items':[]}
    for case in catalog['cases']:
        for option in case['options']:
            key=f"case_{case['case_number']:02d}_{option['id']}"
            result=m2_cases.prepare(case['case_id'],m2_cases.Preparation(
                replacement_text=option['description'],request_id=folder.name+'-'+key))
            if not any(r['key']==key for r in manifest['items']):
                manifest['items'].append({'key':key,'number':len(manifest['items'])+1,
                    'case_number':case['case_number'],'case_id':case['case_id'],'variant':option['id'],
                    'label':option['label'],'description':option['description'],'original_text':case['original_text'],
                    'preparation_id':result['id'],'experiment_id':result['experiment_id'],'run_id':result['run_id']})
            with s.transaction() as conn:
                s.sql(conn,'UPDATE experiments SET name=:name WHERE id=:id',
                      name=key+' · '+option['label'],id=result['experiment_id'])
            write_json(folder/'manifest.json',manifest)
            print(key,result['status'],result['run_id'],flush=True)
    assert len(manifest['items'])==30
    return manifest


def status(folder):
    manifest=read_json(folder/'manifest.json');counts={}
    for item in manifest['items']:
        run=v.run(item['run_id']);item['status']=run['status'];item['error']=run['error']
        counts[run['status']]=counts.get(run['status'],0)+1
    manifest['counts']=counts;manifest['updated_at']=s.now()
    write_json(folder/'manifest.json',manifest)
    print(json.dumps({'counts':counts,'problems':[{'key':i['key'],'error':i['error']} for i in manifest['items'] if i['status']=='failed']},ensure_ascii=False),flush=True)
    return manifest


def retry_for_review(folder,number):
    """Keep failed attempts; explicitly create input-only versions flagged for review."""
    manifest=read_json(folder/'manifest.json')
    selected=[i for i in manifest['items'] if i['case_number']==number]
    if len(selected)!=3:raise ValueError('Expected one target with three descriptions')
    for item in selected:
        item.setdefault('prior_attempts',item.pop('prior_failed_preparations',[]))
        previous=v.run(item['run_id'])
        if previous['status']!='failed' and not item.get('prior_attempts'):
            raise ValueError('Review retry requires an actual failed preparation')
        result=m2_cases.prepare(item['case_id'],m2_cases.Preparation(replacement_text=item['description'],
            request_id=folder.name+'-'+item['key']+'-review2',avp_review_fallback=True))
        if result['id']!=item['preparation_id']:
            item['prior_attempts'].append({**{k:item[k] for k in ('preparation_id','run_id','experiment_id')},'status':previous['status'],'reason':'Retain prior attempt; use supported observed camera poses for final generation inputs.'})
            item.update(preparation_id=result['id'],run_id=result['run_id'],experiment_id=result['experiment_id'])
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET name=:name WHERE id=:id',
                  name=item['key']+' · '+item['label']+' [Anchor review]',id=result['experiment_id'])
        write_json(folder/'manifest.json',manifest)
        print('REVIEW RETRY',item['key'],result['status'],flush=True)


def export(folder):
    manifest=status(folder)
    assert manifest['counts']=={'completed':30},'Export requires all 30 actual inputs completed'
    catalog={c['case_id']:c for c in read_json(s.BASE/'configs/m2_case_description_options.json')['cases']}
    for item in manifest['items']:
        run=v.run(item['run_id']);source=v.stage_dir(run['id']);case,_folder=m2_cases.case(item['case_id'])
        out=folder/item['key'];out.mkdir(exist_ok=True)
        env=os.environ.copy();env['SWAP_DISABLE_QWEN']='1'
        subprocess.run([str(s.ROOT/'envs/gsle/bin/python'),'-m','industrial_swap_v2.scripts.audit_m2_case_inputs',
            '--config',run['config_path'],'--case',case['folder'],'--report',str(out/'audit.json')],
            cwd=s.ROOT,env=env,check=True)
        conditions=read_json(source/'conditions.json')
        for name in FILES:shutil.copy2(source/name,out/name)
        assert not (Path(run['folder'])/'records/qwen_generation_once.lock').exists()
        generation=s.one('SELECT external_api FROM generation_records WHERE run_id=:id',id=run['id'])
        assert generation['external_api']==0
        item.update(output_directory=str(source),anchor=conditions['camera']['name'],
                    anchor_quality=conditions['avp_selection'].get('quality_status','passed'),
                    anchor_visibility=conditions['avp_selection']['selected']['visibility'],
                    design_note=catalog[item['case_id']]['design_note'],
                    condition_cache=conditions.get('condition_cache'),audit='passed',
                    image_sha256={n:s.sha(out/n) for n in ('anchor.png','position_hint.png','depth.png','depth_background.png','edit_mask.png','anchor_camera.json')})
        write_json(out/'case_info.json',item)
        with zipfile.ZipFile(folder/(item['key']+'.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.iterdir()):z.write(path,arcname=path.name)
        print('EXPORTED',item['key'],item['anchor'],flush=True)
        write_json(folder/'manifest.json',manifest)
    for n in range(1,11):
        group=[i for i in manifest['items'] if i['case_number']==n]
        assert group[0]['image_sha256']==group[1]['image_sha256']==group[2]['image_sha256']
        assert len({i['description'] for i in group})==3
    manifest.update(result='30 inputs prepared and audited; no Qwen API called',finished_at=s.now(),
                    all_audits_passed=True,same_camera_per_target_verified=True,qwen_api_called=False,
                    manual_review_inputs=[i['key'] for i in manifest['items'] if i['anchor_quality']=='manual_review_required'])
    write_json(folder/'manifest.json',manifest)
    columns=['number','key','case_id','variant','label','original_text','description','anchor','anchor_quality','anchor_visibility','audit','preparation_id','experiment_id','run_id','output_directory']
    with (folder/'cases_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(manifest['items'])
    review(folder,manifest)
    (folder/'README.txt').write_text('30 Qwen input sets = 10 confirmed M1 targets x 3 replacement descriptions.\n'
        'Case 11 was archived as requested. No Qwen, SAM, TRELLIS.2 or M4 was run.\n\n'
        'For each case_XX_A/B/C folder, upload these three images to Qwen in order:\n'
        '1. anchor.png\n2. position_hint.png\n3. depth_background.png\n'
        'Then use generation_prompt.txt. depth.png is for inspection only.\n'
        'The black area of the background depth image is unknown depth, not a black physical surface.\n'
        'Keep the full square image canvas and the same camera in the generated RGB output.\n'
        'Use the corresponding Web experiment to upload the generated image later.\n'
        'A/B/C use exactly the same camera and image conditions for their target; only descriptions differ.\n'
        'Each folder includes the numerical depth, editing mask, camera, provenance and audit result.\n'
        'Manual-review input sets (below unchanged AVP visibility threshold): '+', '.join(manifest['manual_review_inputs'])+'\n',encoding='utf-8')
    with zipfile.ZipFile(folder/'qwen_inputs_30.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for item in manifest['items']:
            for path in sorted((folder/item['key']).iterdir()):z.write(path,arcname=str(path.relative_to(folder)))
            z.write(folder/(item['key']+'.zip'),arcname=item['key']+'.zip')
        for name in ('manifest.json','cases_summary.csv','index.html','overview.jpg','README.txt'):z.write(folder/name,arcname=name)
    print('COMPLETE',folder,'archive_bytes',(folder/'qwen_inputs_30.zip').stat().st_size,flush=True)


def review(folder,manifest):
    from PIL import Image,ImageDraw,ImageFont
    esc=html.escape;sections=[]
    overview=Image.new('RGB',(1280,10*290),'white');draw=ImageDraw.Draw(overview)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for number in range(1,11):
        group=[i for i in manifest['items'] if i['case_number']==number];a=group[0]
        images=[]
        for k,(name,label) in enumerate([('anchor.png','Anchor RGB'),('position_hint.png','Position guide'),('depth.png','Original depth'),('depth_background.png','Background depth')]):
            path=folder/a['key']/name;im=Image.open(path).convert('RGB');im.thumbnail((270,250));x=k*320
            overview.paste(im,(x+20,(number-1)*290+30));draw.text((x+15,(number-1)*290+7),f'{number:02d} {label}',fill='black',font=font)
            images.append(f'<figure><a href="{a["key"]}/{name}"><img loading="lazy" src="{a["key"]}/{name}"></a><figcaption>{label}</figcaption></figure>')
        cards=[]
        for item in group:
            prompt=(folder/item['key']/'generation_prompt.txt').read_text()
            cards.append(f'<article><h3>{esc(item["key"])} · {esc(item["label"])}</h3><p lang="en">{esc(item["description"])}</p>'
                f'<a href="{item["key"]}.zip">下载此组完整输入 ZIP</a> · <a href="{item["key"]}/generation_prompt.txt">Prompt TXT</a> · '
                f'<a href="/m2-cases?case={item["case_id"]}&preparation={item["preparation_id"]}">在 M2 查看</a> · '
                f'<a href="/#experiment/{item["experiment_id"]}">后续上传生成图</a>'
                f'<details><summary>完整 Qwen Prompt</summary><pre>{esc(prompt)}</pre><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent).then(()=>this.textContent=\'已复制\')">复制 Prompt</button></details></article>')
        warning=(f'<p style="background:#fff0cf;padding:12px;color:#704500"><strong>需人工检查：</strong>最佳候选可见贡献 {a["anchor_visibility"]:.1%}，低于原 90% 门槛。门槛未降低，此视角未标记为 AVP 合格。</p>' if a['anchor_quality']=='manual_review_required' else '')
        sections.append(f'<section><h2>目标 {number:02d} · {esc(a["original_text"])}</h2>{warning}<p>Anchor: {esc(a["anchor"])}。A/B/C 共用经校验的同视角条件，分别使用不同替换描述。</p><p>'+esc(a.get('design_note',''))+'</p><div class="images">'+''.join(images)+'</div><div class="cards">'+''.join(cards)+'</div></section>')
    overview.save(folder/'overview.jpg',quality=90)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>30 组 Qwen 生成输入</title><style>body{font:15px/1.6 system-ui;background:#edf3f5;color:#19323c;margin:0}main{max-width:1440px;margin:auto;padding:24px}section,header{background:white;border-radius:12px;padding:22px;margin-bottom:22px}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{padding:16px;background:#f3f8f8;border-radius:8px}a{color:#087b70}pre{white-space:pre-wrap;font:13px/1.6 monospace}button{padding:8px;border:1px solid #aaa;background:white;cursor:pointer}@media(max-width:850px){.cards{grid-template-columns:1fr}.images{grid-template-columns:1fr 1fr}}</style><main><header><h1>30 组 Qwen 生成输入</h1><p>10 个原设备 × 每个 3 种替换方案。原第 11 条重复定位已归档。全部输入准备和对齐校验完成，未调用 Qwen、SAM、TRELLIS.2 或 M4。</p><p><a href="qwen_inputs_30.zip">下载全部 30 组 ZIP</a> · <a href="cases_summary.csv">下载汇总 CSV</a> · <a href="overview.jpg">10 个视角条件总览</a> · <a href="/m2-cases">M2 案例页面</a></p><p>提交 Qwen 的图像顺序：Anchor RGB → Position guide → Background depth。Original depth 只用于人工检查。每组目录另有相机、数值深度、Mask、完整 Prompt 和校验报告。</p></header>'''+''.join(sections)+'</main></html>'
    warning_keys=', '.join(manifest.get('manual_review_inputs',[])) or '无'
    page=page.replace('<p>提交 Qwen 的图像顺序',f'<p><strong>Anchor 需人工检查的输入组：</strong>{esc(warning_keys)}。下方单独说明可见度门槛问题。</p><p>提交 Qwen 的图像顺序')
    (folder/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','export','review']);p.add_argument('--batch',default='m2_30_20261107');p.add_argument('--case-number',type=int);a=p.parse_args()
    if not a.batch.replace('_','').isalnum():raise ValueError('Invalid batch name')
    s.init();m1_cases.init();m2_cases.init();folder=s.DATA/'m2_batches'/a.batch;folder.mkdir(parents=True,exist_ok=True)
    if a.action=='review':retry_for_review(folder,a.case_number)
    else:{'submit':submit,'status':status,'export':export}[a.action](folder)

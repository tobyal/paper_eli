"""Compact overview of the completed Gaussian registration experiments."""
import argparse
import csv
import html
from pathlib import Path
import zipfile
from PIL import Image,ImageDraw,ImageOps
from industrial_swap_v2.core.context import read_json,write_json,sha256
from industrial_swap_v2.webapp import store as s


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);p.add_argument('--bundle',action='store_true');a=p.parse_args()
    batch=a.batch.resolve();state=read_json(batch/'batch.json');rows=[];cards=[]
    board=Image.new('RGB',(1440,520*((len(state['items'])+2)//3)),'white');draw=ImageDraw.Draw(board)
    for index,item in enumerate(state['items']):
        root=Path(item['folder']);key=item['key'];title=html.escape(key+' · '+item['label'])
        x=index%3*480;y=index//3*520
        if item['status']!='completed':
            cards.append('<section><h2>'+title+'</h2><p>'+html.escape(item['status']+' '+item.get('active_stage',''))+'</p></section>')
            draw.text((x+8,y+10),key+' | '+item['status'],fill='black');continue
        r=read_json(root/'m4/alignment/comparison_report.json');d=read_json(root/'delivery.json');m=read_json(root/'m4/summary.json')
        row={'case':key,'label':item['label'],'iou_before':r['baseline']['mask_iou'],'iou_after':r['selected']['mask_iou'],
            'scale_before':r['pose_comparison']['old']['scale'],'scale_after':r['selected']['scale'],
            'yaw_before':r['pose_comparison']['old']['yaw_degrees'],'yaw_after':r['selected']['yaw_degrees'],
            'volume_overflow':r['selected']['max_volume_overflow_fraction'],'m4_steps':m['checks']['training_steps'],
            'candidate_guard_passed':m['candidate_guard_passed'],'selected_step':m['selected_step'],
            'height_adapted':bool(r['space'].get('space_override')),
            'page':d['page'],'folder':str(root),'case02_reused':item['reused_case02']}
        rows.append(row);rid=d['m4_run']
        cards.append(f'<section><h2>{title}</h2><p>Mask IoU {row["iou_before"]:.3f} → {row["iou_after"]:.3f}；M4 {row["m4_steps"]} 步；保护规则选择 {row["selected_step"]} 步。</p><a href="{d["page"]}">新旧位姿、多视角对比与 PLY 下载</a><img loading="lazy" src="/api/runs/{rid}/file?name=alignment/reference_old_new.jpg"><img loading="lazy" src="/api/runs/{rid}/file?name=alignment/anchor_m4_comparison.jpg"></section>')
        image=Image.open(root/'m4/renders/anchor_candidate_step2000.png').convert('RGB')
        board.paste(ImageOps.contain(image,(480,480)),(x,y+40));draw.text((x+8,y+10),f'{key} | IoU {row["iou_before"]:.3f} -> {row["iou_after"]:.3f}',fill='black')
    completed=len(rows);failed=sum(x['status']=='failed' for x in state['items'])
    report={'completed':completed,'failed':failed,'total':len(state['items']),'items':rows,'qwen_calls':0,'trellis_calls':0}
    write_json(batch/'results_summary.json',report);board.save(batch/'actual_2000_overview.jpg',quality=93)
    if rows:
        with (batch/'results_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaussian 配准案例总览</title><style>body{font:16px/1.7 system-ui;max-width:1700px;margin:30px auto;padding:20px;background:#f4f6f8;color:#182533}section{background:white;padding:22px;border-radius:12px;margin:20px 0}img{width:100%}a{color:#1259ad}</style>'''
    if completed+failed<len(state['items']):page+='<meta http-equiv="refresh" content="30">'
    page+=f'<h1>真实 Gaussian 空间配准 · M4 2000 步</h1><p>完成 {completed}/{len(state["items"])}；失败 {failed}。case02 复用已验证的新配准结果，其余 9 例使用同一方法。旧实验保留，M1–M3 不重跑。</p><p>实际末步候选与环境保护场景分别提供。配准失败不会擅自放宽阈值或继续训练。</p>'+''.join(cards)+'</html>'
    folder=s.BASE/'webapp/static/batches'/batch.name;folder.mkdir(parents=True,exist_ok=True);(folder/'index.html').write_text(page,encoding='utf-8')
    if a.bundle:
        path=batch/'registration_all_cases_results.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name in ('batch.json','results_summary.json','results_summary.csv','actual_2000_overview.jpg'):z.write(batch/name,name)
            for item in state['items']:
                root=Path(item['folder'])
                if item['status']=='completed':
                    delivery=read_json(root/'delivery.json');z.write(delivery['archive'],item['key']+'/alignment_results.zip')
            for name in ('run_registration_batch.py','publish_registration_batch.py'):z.write(s.BASE/'scripts'/name,'scripts/'+name)
        write_json(batch/'delivery.json',{'archive':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'page':'/static/batches/'+batch.name+'/index.html','completed':completed,'failed':failed})
    print({'completed':completed,'failed':failed},flush=True)


if __name__=='__main__':main()

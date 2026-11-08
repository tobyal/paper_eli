"""Build a read-only result index from actual completed Web/CLI stage records."""
import argparse
import html
from pathlib import Path
from urllib.parse import quote
from PIL import Image,ImageDraw,ImageOps
from industrial_swap_v2.core.context import read_json,write_json
from industrial_swap_v2.webapp import store as s,service as v


def file_url(rid,name,download=False):
    return '/api/runs/'+rid+'/file?name='+quote(name)+('&download=true' if download else '')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args()
    batch=a.batch.resolve();manifest=read_json(batch/'manifest.json')
    post=read_json(batch/'postprocess.json') if (batch/'postprocess.json').exists() else {'items':{}}
    result={'batch':batch.name,'generated_at':s.now(),'qwen_calls':0,'requested_cases':len(manifest['items']),'items':[]}
    supplemental=batch/'historical_cases.json'
    historical=read_json(supplemental)['items'] if supplemental.exists() else []
    sources=historical+manifest['items'];result['requested_cases']=len(sources)
    cards=[];board=Image.new('RGB',(1440,520*((len(sources)+2)//3)),'white');draw=ImageDraw.Draw(board)
    candidates_board=Image.new('RGB',board.size,'white');candidate_draw=ImageDraw.Draw(candidates_board)
    for j,source in enumerate(sources):
        item=dict(source);key=item['key'];heads=s.unpack(v.experiment(item['experiment_id'])['heads'])
        item['web_url']='/#experiment/'+item['experiment_id'];item['states']={}
        for stage in ('m2','m3','m4'):
            if stage not in heads:continue
            value=v.run(heads[stage]);item['states'][stage]={'id':value['id'],'status':value['status'],
                'folder':value['folder'],'output':s.unpack(value['outputs']).get('directory')}
        item['postprocess_status']=source.get('postprocess_status') if source.get('historical_import') else post['items'].get(key,{}).get('status','waiting_m3')
        ready=item['postprocess_status']=='completed' and 'm4' in item['states']
        item['complete']=ready
        title=html.escape(key+' · '+item['label']);links=[f'<a href="{item["web_url"]}">实验 / 日志 / 素材</a>']
        images=[];detail='尚未完成：'+html.escape(item['postprocess_status'])
        if ready:
            m3=item['states']['m3'];m4=item['states']['m4'];report=read_json(Path(m4['output'])/'summary.json')
            item['m4_report']=report
            item['quality_review']=read_json(batch/'reviews'/(key+'.json'))
            alignment=Path(m3['output'])/'reference_alignment.json'
            item['alignment']=read_json(alignment) if alignment.exists() else source.get('alignment')
            item['source_input_rgb_unchanged']=s.sha(item['input_path'])==item['input_sha256']
            steps=report['checks']['training_steps'];candidate_name=f'scene_candidate_step{steps}.ply'
            before=('renders/anchor_before.png',m4['id']) if source.get('historical_import') else ('renders/anchor_after.png',m3['id'])
            for label,name,rid in [('独立设备高斯','renders/anchor_object.png',m3['id']),('M3 场景',*before),(f'M4 实际 {steps:,} 步候选',f'renders/anchor_candidate_step{steps}.png',m4['id']),('M4 保护规则选出的最终场景','renders/anchor_after.png',m4['id'])]:
                images.append(f'<figure><img loading="lazy" src="{file_url(rid,name)}"><figcaption>{label}</figcaption></figure>')
            links += [f'<a href="{file_url(m4["id"],"scene_final.ply",True)}">最终场景 PLY</a>',
                f'<a href="{file_url(m3["id"],"object_world.ply",True)}">设备 PLY</a>',
                f'<a href="{file_url(m3["id"],"trellis_asset.glb",True)}">原始 GLB</a>',
                f'<a href="/viewer?experiment={item["experiment_id"]}">保护规则选出场景的漫游</a>',
                f'<a href="{file_url(m4["id"],"validation/anchor_comparison.png")}">实际 {steps:,} 步对比</a>',
                f'<a href="{file_url(m4["id"],candidate_name,True)}">{steps:,} 步候选 PLY</a>',
                f'<a href="{file_url(m4["id"],"validation/supervision_union_masks.png")}">双 Mask 监督</a>']
            detail=f'实际执行 {report["checks"]["training_steps"]} 步；最终选择第 {report["selected_step"]} 步。末轮环境检查：'+('通过' if report['candidate_guard_passed'] else '未通过，候选单独保留。')
            if item['quality_review'].get('quality_notes_zh'):
                detail+=' '+html.escape(item['quality_review']['quality_notes_zh'])
            im=Image.open(Path(m4['output'])/'renders/anchor_after.png').convert('RGB')
            x=j%3*480;y=j//3*520;board.paste(ImageOps.contain(im,(480,480)),(x,y+40))
            draw.text((x+8,y+7),key+f' | final selected step {report["selected_step"]}',fill='black')
            actual=Image.open(Path(m4['output'])/f'renders/anchor_candidate_step{steps}.png').convert('RGB')
            candidates_board.paste(ImageOps.contain(actual,(480,480)),(x,y+40))
            candidate_draw.text((x+8,y+7),key+f' | actual {steps} | guard '+('PASS' if report['candidate_guard_passed'] else 'FAIL'),fill='black')
            preview=batch/'results'/item.get('result_subdir',key)
            if (preview/'render_receipt.json').exists():
                for kind in ('orbit','trajectory'):
                    aid=s.register(preview/(kind+'_m3_m4.mp4'),'video',name=key+' M3/M4 '+kind,source='verified_final_render',
                        experiment=item['experiment_id'],role='preview',metadata={'case_key':key,'selected_step':report['selected_step']})
                    links.append(f'<a href="/api/assets/{aid}/file">{kind} M3/M4 视频</a>')
                if (preview/'candidate_video_receipt.json').exists():
                    for kind in ('orbit','trajectory'):
                        aid=s.register(preview/(kind+'_m3_actual250_final.mp4'),'video',name=key+' M3/actual250/guarded '+kind,
                            source='verified_candidate_video',experiment=item['experiment_id'],role='preview',
                            metadata={'case_key':key,'selected_step':report['selected_step'],'panels':['M3','actual250','guarded_final']})
                        links.append(f'<a href="/api/assets/{aid}/file">{kind} 含实际 250 步的三列对比</a>')
        else:
            x=j%3*480;y=j//3*520;draw.text((x+8,y+8),key+' | '+item['postprocess_status'],fill='black')
        cards.append(f'<section id="{key}"><h2>{title}</h2><p>{detail}</p><div class="images">'+''.join(images)+'</div><nav>'+' · '.join(links)+'</nav></section>')
        result['items'].append(item)
    result['completed_cases']=sum(i['complete'] for i in result['items'])
    write_json(batch/('final_manifest_with_case00.json' if historical else 'final_manifest.json'),result)
    board.save(batch/'review'/('final_overview_with_case00.jpg' if historical else 'final_overview.jpg'))
    candidates_board.save(batch/'review'/('actual_steps_overview_with_case00.jpg' if historical else 'actual_250_overview.jpg'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>9 个设备替换结果</title><style>body{font:16px/1.65 system-ui;max-width:1800px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#17212c}section{background:white;padding:20px;border-radius:12px;margin:20px 0}h1,h2{line-height:1.3}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%;height:auto}nav a{display:inline-block;margin:5px 8px 5px 0;color:#145bb6}figcaption{text-align:center}small{color:#536474}@media(max-width:700px){.images{grid-template-columns:1fr}}</style>'''
    page=page.replace('<title>9 个设备替换结果</title>',f'<title>{len(sources)} 个设备替换结果</title>')
    historical_note='case00 复用已完成的机械臂 10,000 步续训结果；其余 9 例各 250 步，训练预算不同。此次整理未运行模型。' if historical else 'M4 固定新设备，局部背景优化 250 步；没有调用 Qwen 或 DI-SDS。'
    page+=f'<h1>{len(sources)} 个工业设备替换案例</h1><p>完成 {result["completed_cases"]}/{len(sources)}。{historical_note}</p><p>“最终场景”遵守现有环境保护规则。选中第 0 步表示保留输入背景；真实训练候选另行提供，不能与通过检查的最终结果混用。</p>'
    page+='<small>03B、05B 的 SAM 分割不完整，使用已有 RMBG-2.0 提取前景。M3 增加实际高斯 RGB 对齐来消除轮廓对称引起的朝向歧义。金属/PBR 光照在当前基色 SH0 转换中会弱化。</small>'+''.join(cards)
    destination=s.BASE/'webapp/static/batches'/batch.name;destination.mkdir(parents=True,exist_ok=True)
    (destination/'index.html').write_text(page,encoding='utf-8');(batch/'results_index.html').write_text(page,encoding='utf-8')
    print({'completed':result['completed_cases'],'page':'/static/batches/'+batch.name+'/index.html'},flush=True)


if __name__=='__main__':main()

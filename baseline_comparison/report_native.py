"""Offline/HTTP inspection pages for actual DGE and paper-derived GSLE outputs."""
import argparse,html,json,time,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
CASES=['case00','case02','case04','case07','case10']
SECTIONS={
 'dge':[
  ('Native localization',['original_view','localization_2d_mask','localization_2d_overlay','localization_multiview','localization_3d_mask','localization_selected_gaussians','localization_full_scene']),
  ('Actual multi-view 2D editing at steps 0 and 500',['original_multiview','edited_2d_views','edited_2d_views_step500','edited_2d_comparison']),
  ('Gaussian fitting: 0 / 500 / 1000 steps',['gaussian_optimization_initial','gaussian_optimization_middle','scene_final','scene_final_crop','final_multiview','final_comparison'])],
 'gsle':[
  ('Text-only LangSplat localization + explicit connected-instance adapter',['original_view','language_query_heatmap','localization_candidates','localization_selected_mask','localization_selected_gaussians','localization_3d_box','scene_without_target']),
  ('AVP: literal supplement Algorithm 1',['avp_candidates','avp_scores','avp_selected_anchor','avp_anchor_with_bbox','avp_depth','avp_inpainting_mask']),
  ('Depth-conditioned inpainting + SAM',['inpainting_input','inpainting_mask','inpainting_depth','inpainting_result','foreground_mask','foreground_rgba','foreground_white']),
  ('Coarse 3D-SDS: 0 / 3500 / 6500',['coarse_initial','coarse_middle','coarse_final','coarse_multiview']),
  ('DI-SDS: initialization / 1000 / 2000',['refine_initial','refine_middle','refine_final','refine_multiview','coarse_refine_comparison','training_losses']),
  ('Final scene',['scene_final','scene_final_crop','final_comparison','final_multiview'])]}
HEAD='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:24px auto;padding:20px;background:#f4f6f8;color:#172435}section{background:white;padding:20px;margin:20px 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}figure{margin:0}img{max-width:100%;height:auto}a{color:#1262a7}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#fff1cd;padding:12px}code{overflow-wrap:anywhere}</style>'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def overview(root,method,rows):
    board=Image.new('RGB',(5*320,len(rows)*352),'white');draw=ImageDraw.Draw(board)
    names=['original_view','localization_full_scene','edited_2d_primary','scene_final','scene_final_crop'] if method=='dge' else ['avp_selected_anchor','avp_anchor_with_bbox','inpainting_result','scene_final','scene_final_crop']
    labels=['Original','Own localization','Actual 2D editing' if method=='dge' else 'Actual inpainting','Final Gaussian','Own edit-region crop']
    for i,case in enumerate(rows):
        for j,(name,label) in enumerate(zip(names,labels)):
            x,y=j*320,i*352;draw.text((x+5,y+6),case+' | '+label,fill='black');p=root/case/(name+'.png')
            if p.exists():board.paste(ImageOps.contain(Image.open(p).convert('RGB'),(320,320)),(x,y+28))
            else:draw.text((x+15,y+130),'Not produced',fill='gray')
    board.save(root/'all_cases_overview.png')


def coarse_refine_overview(root,rows):
    names=['foreground_white','coarse_final','refine_final','scene_final']
    labels=['Generated foreground','Coarse 6500','DI-SDS 2000','Final composed scene']
    board=Image.new('RGB',(4*320,len(rows)*352),'white');draw=ImageDraw.Draw(board)
    for i,case in enumerate(rows):
        for j,(name,label) in enumerate(zip(names,labels)):
            x,y=j*320,i*352;draw.text((x+5,y+6),case+' | '+label,fill='black')
            path=root/case/(name+'.png')
            if path.exists():board.paste(ImageOps.contain(Image.open(path).convert('RGB'),(320,320)),(x,y+28))
            else:draw.text((x+15,y+130),'Not produced',fill='gray')
    board.save(root/'coarse_refine_all_cases.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',action='store_true');a=p.parse_args()
    delivery=BASE/'native_dge_gsle_review';delivery.mkdir(exist_ok=True)
    portal=HEAD+'<title>DGE / GSLE Native</title><h1>DGE / GSLE 原生完整流程实验</h1><p>统一输入仅为原始场景、完整 COLMAP、A/B 文本。执行完成不代表设备替换成功。</p><p>本页为实际输出的静态快照；耗时为共享 GPU 上的实测墙钟时间，包含预处理和导出，不是独占硬件性能测试。</p>'
    all_rows=[]
    for method in ['dge','gsle']:
        root=BASE/(method+'_native');root.mkdir(exist_ok=True);summaries=[]
        for case in CASES:
            folder=root/case
            if not (folder/'config.json').exists():summaries.append(dict(case=case,status='not_run'));continue
            status=read(folder/'records/status.json',{});prompt=read(folder/'prompts.json',{})
            quality=read(folder/'records/visual_review.json',{});localization=read(folder/'records/localization.json',{})
            if method=='dge' and localization.get('review_cameras'):
                primary=localization['review_cameras'][0]['index']
                source=folder/f'edited/step_0000/{primary:04d}.png'
                if source.exists():
                    import shutil
                    shutil.copy2(source,folder/'edited_2d_primary.png')
            row=dict(case=case,status=status.get('status','prepared'),active_stage=status.get('active_stage'),
                 execution_success=status.get('status')=='complete',A=prompt.get('A'),B=prompt.get('B'),quality=quality,
                 localization=localization,final_ply=str(folder/'final_scene.ply'),timing=status,verification=read(folder/'records/verification.json',{}))
            summaries.append(row)
            all_rows.append(dict(method=method,**row))
            page=HEAD+f'<title>{method.upper()} {case}</title><p><a href="../index.html">返回案例列表</a></p><h1>{method.upper()} Native · {case}</h1>'
            page+='<p>状态：'+html.escape(row['status'])+'</p><p>A：'+html.escape(prompt.get('A',''))+'</p><p>B：'+html.escape(prompt.get('B',''))+'</p>'
            if status.get('seconds') is not None:page+=f'<p>实测总耗时：{status["seconds"]/60:.1f} 分钟（共享 GPU，含预处理和导出）。</p>'
            if method=='gsle':page+='<p class="note">GSLE 为论文复现，并非作者官方实现。独立三维连通实例选择属于明确记录的适配；AVP 按补充材料 Algorithm 1 选择。背景冻结，Coarse 6500 步 + DI-SDS 2000 步。</p>'
            else:page+='<p class="note">官方 DGE：随机 20 视角、追踪阈值 0.8、1000 步；第 0 和 500 步实际执行多视角编辑。没有独立 Delete/Add。</p>'
            if quality:page+='<p class="note">实际检查：'+html.escape(quality.get('observations',str(quality)))+'</p>'
            page+='<p><a href="config.json">配置</a> · <a href="prompts.json">完整 Prompt</a> · <a href="records/status.json">运行记录</a> · <a href="records/localization.json">定位来源</a></p>'
            if row['verification']:page+='<p><a href="records/verification.json">输出与原生流程验证：'+html.escape(row['verification'].get('status',''))+'</a></p>'
            if quality:page+='<p><a href="records/visual_review.json">实际视觉检查记录</a></p>'
            page+='<p>服务器最终 PLY：<code>'+html.escape(row['final_ply'])+'</code></p>'
            for title,names in SECTIONS[method]:
                page+='<section><h2>'+title+'</h2><div class="grid">'
                for name in names:
                    if (folder/(name+'.png')).exists():page+=f'<figure><a href="{name}.png"><img loading="lazy" src="{name}.png"></a><figcaption>{name}.png</figcaption></figure>'
                    else:page+='<p>尚未输出：'+name+'</p>'
                page+='</div></section>'
            (folder/'index.html').write_text(page+'</html>',encoding='utf-8')
        (root/'summary.json').write_text(json.dumps(dict(updated_at=time.time(),method=method,cases=summaries),ensure_ascii=False,indent=2),encoding='utf-8')
        overview(root,method,CASES)
        if method=='gsle':coarse_refine_overview(root,CASES)
        index=HEAD+f'<title>{method.upper()} Native</title><p><a href="../index.html">两种方法总入口</a></p><h1>{method.upper()} Native · 五案例</h1>'
        index+='<p>执行完成 '+str(sum(x.get('execution_success',False) for x in summaries))+'/5。质量结论见每例检查记录。</p><a href="summary.json">完整摘要 / PLY 路径</a><p><a href="all_cases_overview.png"><img src="all_cases_overview.png"></a></p>'
        if method=='gsle':index+='<h2>二维前景 → Coarse → DI-SDS → 完整场景</h2><p><a href="coarse_refine_all_cases.png"><img src="coarse_refine_all_cases.png"></a></p>'
        for row in summaries:
            title=f'<a href="{row["case"]}/index.html">{row["case"]}</a>' if (root/row['case']/'index.html').exists() else row['case']
            index+='<section><h2>'+title+'</h2><p>'+html.escape(row['status'])+'</p><p>'+html.escape(row.get('quality',{}).get('observations',''))+'</p></section>'
        (root/'index.html').write_text(index+'</html>',encoding='utf-8')
        portal+=f'<section><h2><a href="{method}_native/index.html">{method.upper()} Native</a></h2><p><a href="{method}_native/all_cases_overview.png"><img src="{method}_native/all_cases_overview.png"></a></p></section>'
    (delivery/'summary.json').write_text(json.dumps(dict(updated_at=time.time(),experiments=all_rows),ensure_ascii=False,indent=2),encoding='utf-8')
    portal+='<p><a href="summary.json">十组实验完整运行状态、质量记录与服务器 PLY 路径</a></p>'
    (delivery/'index.html').write_text(portal+'</html>',encoding='utf-8')
    if a.package:
        archive=BASE/'native_dge_gsle_review.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.write(delivery/'index.html','index.html')
            z.write(delivery/'summary.json','summary.json')
            for method in ['dge','gsle']:
                root=BASE/(method+'_native')
                for name in ['index.html','summary.json','all_cases_overview.png']:z.write(root/name,f'{method}_native/{name}')
                if method=='gsle':z.write(root/'coarse_refine_all_cases.png',f'{method}_native/coarse_refine_all_cases.png')
                for case in CASES:
                    folder=root/case
                    if not folder.exists():continue
                    files=[p for p in folder.iterdir() if p.is_file() and p.suffix in ('.png','.json','.html')]
                    for sub in ['records','logs']:
                        for suffix in ['*.json','*.jsonl','*.log','*.patch','*.txt']:files+=list((folder/sub).rglob(suffix))
                    if method=='gsle':
                        for sub in ['localization','avp','inpainting','foreground','coarse','refine','final']:
                            files+=list((folder/sub).glob('*.json'))+list((folder/sub).glob('*.jsonl'))
                    for f in sorted(set(files)):z.write(f,str(f.relative_to(BASE)))
                for f in (BASE/('native_'+method)).glob('*'):
                    if f.suffix in ('.py','.md'):z.write(f,'code/'+f' native_{method}'.strip()+'/'+f.name)
            for name in ['verify_native.py','report_native.py','run_native_remaining.py','native_status.py']:z.write(BASE/name,'code/'+name)
        print(archive,archive.stat().st_size,flush=True)
    print('Updated actual native inspection pages',flush=True)


if __name__=='__main__':main()

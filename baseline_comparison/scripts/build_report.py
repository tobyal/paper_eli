"""Build a portable, artifact-linked report from actual run records."""
import json,csv,html,time,os
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison';OUT.mkdir(exist_ok=True)
spec=json.loads((BASE/'prompts.json').read_text());rows=[];sections=[]
review=json.loads((OUT/'qualitative_review.json').read_text()).get('cases',{}) if (OUT/'qualitative_review.json').exists() else {}
def link(path,label):
 return '<a href="'+html.escape(os.path.relpath(path,OUT).replace(os.sep,'/'))+'">'+html.escape(label)+'</a>'
for task in spec['cases']:
 case=task['case'];tr=[]
 for method in ['gaussianeditor','dge','gsle']:
  folder=BASE/method/case;status=folder/'records/status.json';data=json.loads(status.read_text()) if status.exists() else {'status':'not_started'}
  artifact=OUT/case/method/'artifact.json';record=json.loads(artifact.read_text()) if artifact.exists() else {}
  successful=data['status']=='training_complete' and (folder/'scene_final.ply').exists()
  state='training_complete' if successful else data['status']
  measured=sum(v.get('seconds',0) for v in data['stages'].values()) if data.get('stages') else data.get('seconds',0)
  row={'case':case,'method':method,'status':state,'active_stage':data.get('active_stage',''),'active_stage_seconds':round(measured,2),'wall_seconds':data.get('seconds',''),'scene_ply':str(folder/'scene_final.ply') if successful else '', 'gaussians':record.get('gaussians',''),'sha256':record.get('sha256','')}
  row['qualitative_outcome']=review.get(case,{}).get(method,{}).get('outcome','not_yet_reviewed')
  row['timing_note']='Stage seconds sum the latest saved stage records only; wall seconds include waiting and retries. Cached stages completed in earlier attempts may be absent from the stage sum.'
  if method=='gaussianeditor' and case=='case00':row['timing_note']+=' Final Add reused this baseline attempt\'s generated asset; the 112.65-second Add record covers placement recovery, not the full Add generation.'
  rows.append(row)
  links=[link(status,'Run record')] if status.exists() else []
  links+=[link(folder/'task.json','Actual prompts'),link(folder/'logs','Logs')]
  if successful:links+=[link(folder/'scene_final.ply','Final PLY')]
  wall=data.get('seconds');timing=(str(round(wall/60,1))+' min wall' if isinstance(wall,(int,float)) else 'Wall not recorded')+'<br>'+str(round(measured/60,1))+' min latest stage records'
  tr.append('<tr><td>'+method+'</td><td>'+state+'</td><td title="'+html.escape(row['timing_note'])+'">'+timing+'</td><td>'+' · '.join(links)+'</td></tr>')
 note='' if task['ours_B_exact_match'] else '<p class="warn">Ours historical B differs in placement wording; see full manifest. Baseline A/B are exactly the new requested texts.</p>'
 if not task.get('ours_guard_passed',False):note+='<p>Ours displays the existing actual 2000-step candidate requested for comparison; its historical guard did not pass. It is not replaced by a fallback scene.</p>'
 task['ours_scene']=str(BASE/'inputs'/case/'ours_scene.ply')
 note+='<p>'+link(OUT/case/'comparison_full.png','Lossless full-frame PNG')+' · '+link(OUT/case/'comparison_crops.png','Lossless common-crop PNG')+'</p>'
 if review.get(case):
  note+='<details><summary>Observed quality and limitations</summary><ul>'+''.join('<li><b>'+html.escape(method)+':</b> '+html.escape(q['observations'])+'</li>' for method,q in review[case].items())+'</ul></details>'
 if (OUT/case/'gsle_stages.jpg').exists():note+='<details><summary>GSLE native reference / coarse / DI-SDS diagnostics</summary><p>These are GSLE\'s own anchor previews, separate from the common comparison cameras. Losses are against its generated reference, not cross-method quality metrics.</p><a href="'+case+'/gsle_stages.png"><img src="'+case+'/gsle_stages.jpg"></a></details>'
 sections.append('<section id="'+case+'"><h2>'+case+' — '+html.escape(task['label'])+'</h2><p><b>A:</b> '+html.escape(task['A'])+'</p><p><b>B:</b> '+html.escape(task['B'])+'</p>'+note+'<table><tr><th>Method</th><th>Actual status</th><th>Recorded duration</th><th>Files</th></tr>'+''.join(tr)+'</table><p>'+link(OUT/case/'cameras.json','Identical original camera parameters')+' · '+link(Path(task['ours_scene']),'Ours actual 2000-step scene')+'</p><a href="'+case+'/comparison_full.jpg"><img src="'+case+'/comparison_full.jpg" loading="lazy"></a><details><summary>Identical target crops (three views)</summary><a href="'+case+'/comparison_crops.jpg"><img src="'+case+'/comparison_crops.jpg" loading="lazy"></a></details></section>')
with (OUT/'experiments.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'experiments.json').write_text(json.dumps(rows,indent=2))
complete=sum(r['status']=='training_complete' for r in rows)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Industrial replacement baseline comparison</title><style>body{font:16px/1.55 system-ui;margin:28px auto;max-width:1500px;padding:0 22px;background:#f7f8fa;color:#1b2530}section{background:white;border:1px solid #ddd;border-radius:10px;padding:20px;margin:24px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#155dba}nav{display:flex;gap:20px}.warn{background:#fff4cf;padding:10px}code{font-size:13px}summary{cursor:pointer;font-weight:bold}</style><h1>Industrial equipment replacement: 5 cases × 3 baselines</h1>'''
page+=f'<p>Saved final training artifacts: <b>{complete}/15</b>. Updated {time.strftime("%Y-%m-%d %H:%M:%S %Z")}.</p>'
page+='<p>Original | GaussianEditor | DGE | GSLE | Ours. All columns use identical original calibrated cameras and identical crops. Training completion does not imply successful object replacement; inspect the actual rendered geometry.</p><p class="warn">GSLE is the existing paper-derived reproduction, not official public GSLE code. Existing Ours outputs are reused; no Qwen or TRELLIS is invoked in baseline runs. Missing results are explicitly labeled; superseded diagnostics are excluded.</p>'
if any('degenerated' in q.get('gsle',{}).get('outcome','') for q in review.values()):page+='<p class="warn">The current GSLE reproduction exhibits optimization degeneration. Its failures are retained as actual outputs and must not be treated as evidence of the original paper\'s best achievable quality. Saturated regions are actual RGB renders of final PLYs, not segmentation overlays. All qualitative observations are specific to this implementation/configuration and seed.</p>'
page+='<p>Wall durations include waiting and retries. Stage sums include only the latest saved record of each stage, and can omit generation reused from earlier attempts (notably GaussianEditor case00 placement recovery). They are not complete training-time measurements. GPUs are shared and jobs overlap, so these operational records must not be used as a controlled speed benchmark.</p>'
if (OUT/'overview_primary.jpg').exists():page+='<p>'+link(OUT/'overview_primary.png','Five-case overview (lossless PNG)')+'</p><a href="overview_primary.png"><img src="overview_primary.jpg"></a>'
page+='<nav>'+''.join('<a href="#'+t['case']+'">'+t['case']+'</a>' for t in spec['cases'])+'</nav><p>'+link(BASE/'prompts.json','Unified prompt / provenance manifest')+' · '+link(OUT/'experiments.csv','Experiment CSV')+'</p>'+''.join(sections)
page+='<p>'+link(OUT/'reproduction_code_and_records.zip','Exact executed code, cumulative patches and run records')+' · '+link(OUT/'qualitative_review.json','Qualitative review record')+'</p>'
(OUT/'index.html').write_text(page+'</html>',encoding='utf-8');print('Report',complete,'/ 15 saved results',flush=True)

"""Package real weak DI-SDS previews and records; never include source symlinks."""
from pathlib import Path
import csv
import io
import json
import zipfile
from PIL import Image,ImageDraw,ImageChops,ImageOps

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison/gsle_weak_disds')
cases=['case00','case02','case04','case07','case10']
rows=[]; sections=[]
overview=Image.new('RGB',(1536,544*len(cases)),'white')
for i,case in enumerate(cases):
    folder=ROOT/case
    status=json.loads((folder/'records/status.json').read_text())
    assert status['status']=='complete',case
    v=json.loads((folder/'records/verification.json').read_text())
    assert v['iterations']==1000
    training=[json.loads(line) for line in (folder/'refine/training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in training]==list(range(1,1001)),case
    # Presentation only: the same crop of the actual Gaussian renders in each pair.
    crop_sheet=Image.new('RGB',(1024,544*3),'white')
    crop_draw=ImageDraw.Draw(crop_sheet)
    crop_records=[]
    for j in range(1,4):
        pair=[Image.open(folder/f'{stage}_view_{j}.png').convert('RGB') for stage in ('coarse','weak')]
        differences=[ImageChops.difference(img,Image.new('RGB',img.size,'white')).convert('L').point(lambda x:255 if x>5 else 0) for img in pair]
        box=ImageChops.lighter(*differences).getbbox() or (0,0,pair[0].width,pair[0].height)
        box=(max(0,box[0]-8),max(0,box[1]-8),min(pair[0].width,box[2]+8),min(pair[0].height,box[3]+8))
        crop_records.append(dict(camera=j,crop=list(box)))
        for k,img in enumerate(pair):
            thumb=ImageOps.contain(img.crop(box),(512,512))
            x,y=k*512,(j-1)*544
            crop_sheet.paste(thumb,(x+(512-thumb.width)//2,y+32+(512-thumb.height)//2))
            crop_draw.text((x+8,y+8),f'{"Coarse" if k==0 else "Weak DI-SDS"} / original camera {j} / same crop',fill='black')
    crop_sheet.save(folder/'multiview_crops.png')
    (folder/'records/display_crops.json').write_text(json.dumps(crop_records,indent=2))
    page=folder/'index.html'
    content=page.read_text()
    if 'multiview_crops.png' not in content:
        content=content.replace('<h2>Server outputs</h2>','<h2>Same crops of original camera renders</h2><a href="multiview_crops.png"><img src="multiview_crops.png"></a><h2>Server outputs</h2>')
        page.write_text(content)
    rows.append(dict(case=case,steps=v['iterations'],gaussians=v['gaussians'],
        frozen_geometry_unchanged=all(v['frozen_parameters_bitwise_equal'].values()),
        alpha_max_abs_change=max(x['alpha_max_abs_change'] for x in v['views']),
        coarse_rgb_mse=v['anchor_rgb_mse']['coarse'],weak_rgb_mse=v['anchor_rgb_mse']['weak'],
        mean_color_change=v['color_rgb_mean_abs_change'],training_seconds=sum(r['seconds'] for r in training),
        current_run_wall_seconds=status['seconds'],
        preceding_pilot_wall_seconds=json.loads((ROOT/'pilot_case04_500/records/status.json').read_text())['seconds'] if case=='case04' else 0,
        object_ply=v['outputs']['object_world'],scene_ply=v['outputs']['scene']))
    image=Image.open(folder/'comparison.png').convert('RGB')
    overview.paste(image,(0,i*544))
    ImageDraw.Draw(overview).text((700,i*544+20),case,fill='black')
    sections.append(f'<section><h2>{case}</h2><a href="{case}/index.html">View steps 100 / 500 / 1000, scenes, multiview and verification</a><a href="{case}/comparison.png"><img src="{case}/comparison.png"></a></section>')
overview.save(ROOT/'overview.png')
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2))
csvbuf=io.StringIO();writer=csv.DictWriter(csvbuf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(ROOT/'summary.csv').write_text(csvbuf.getvalue())
(ROOT/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>GSLE Weak DI-SDS</title><style>body{font:17px/1.6 system-ui;max-width:1550px;margin:auto;padding:24px;background:#f4f6f8}img{max-width:100%}section{background:white;margin:24px 0;padding:20px}</style><h1>GSLE Weak DI-SDS / five actual runs</h1><p>Coarse 6500 / original DI-SDS 2000 / weak DI-SDS 1000. Independent conservative experiment: color only, LR 0.001, SDS weight 0.0001. Geometry, opacity and background are frozen. No Coarse or input regeneration.</p><p><a href="summary.csv">CSV summary</a> | <a href="summary.json">Metrics and server PLY paths</a> | <a href="pilot_case04_500/index.html">case04 pilot 500</a></p>'+''.join(sections))
archive=ROOT/'review.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['index.html','overview.png','summary.json','summary.csv']:
        z.write(ROOT/name,name)
    for label in cases+['pilot_case04_500']:
        folder=ROOT/label
        files=[x for x in folder.iterdir() if x.is_file() and x.suffix in ('.png','.html','.json')]
        for sub in ['records','refine/previews']:
            files += [x for x in (folder/sub).rglob('*') if x.is_file() and x.suffix in ('.png','.json','.patch')]
        for x in files:z.write(x,x.relative_to(ROOT))
print(archive,archive.stat().st_size)

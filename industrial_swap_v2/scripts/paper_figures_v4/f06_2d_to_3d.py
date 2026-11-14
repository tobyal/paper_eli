"""Read-only Case02/04/07 2D-to-3D presentation. No model generation/training.

CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python .../f06_2d_to_3d.py --render
python3 .../f06_2d_to_3d.py --assemble
"""
import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[3]
V3=ROOT/'industrial_swap_v2/outputs/paper_figures_v3'
V4=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
OUT=V4/'F06_2d_to_3d'
OLD=V3/'F06_five_cases'
CASES=['case02','case04','case07']
NAMES=['Roller Conveyor','Industrial Chiller','Storage Cabinet']
SIZE=(1600,1000)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def record(path):
    p=Path(path)
    return dict(path=str(p),resolved=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def bounds(mask):
    y,x=np.where(mask)
    assert len(x)>0
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]


def crop_box(bbox,size,fraction=.82):
    x0,y0,x1,y1=bbox
    w,h=size
    # Integer 8:5 crops allow exactly uniform scaling to 1600x1000.
    k=math.ceil(max((x1-x0)/(8*fraction),(y1-y0)/(5*fraction)))
    k=min(k,w//8,h//5)
    cw,ch=8*k,5*k
    assert cw>=x1-x0 and ch>=y1-y0,'Full equipment does not fit the requested crop aspect'
    left=min(max(round((x0+x1-cw)/2),0),w-cw)
    top=min(max(round((y0+y1-ch)/2),0),h-ch)
    assert left<=x0 and top<=y0 and left+cw>=x1 and top+ch>=y1
    return [left,top,left+cw,top+ch]


def save_panel(src,box,path):
    im=src.crop(box)
    assert im.width/im.height==1.6
    im=im.resize(SIZE,getattr(Image,'Resampling',Image).LANCZOS)
    im.save(path,dpi=(300,300))


def render():
    import torch
    sys.path.insert(0,str(ROOT))
    from baseline_comparison.matched_rendering import render as r
    c=r.c
    (OUT/'panels').mkdir(parents=True,exist_ok=True)
    (OUT/'renders').mkdir(exist_ok=True)
    old_manifest=json.loads((OLD/'case_manifest.json').read_text())
    rows={row['case']:row for row in old_manifest['cases']}
    task_file=ROOT/'baseline_comparison/prompts.json'
    task_rows={row['case']:row for row in json.loads(task_file.read_text())['cases']}
    records={}
    with torch.no_grad():
        for case in CASES:
            row=rows[case]
            asset_path=Path(row['m3_asset']['path'])
            qwen_path=Path(row['qwen']['path'])
            transform_path=Path(row['world_object']['path']).parent/'transform.json'
            tf=json.loads(transform_path.read_text())
            assert sha(asset_path)==row['m3_asset']['sha256']==tf['source_local_sha256']
            assert sha(qwen_path)==row['qwen']['sha256']
            source_hash=sha(asset_path)
            qwen=Image.open(qwen_path).convert('RGB')
            anchor=c.Camera.from_record(json.loads(Path(row['anchor_camera']['path']).read_text()))
            # Match the native saved Qwen resolution before any presentation crop.
            anchor=anchor.resize(*qwen.size)
            mask_path=Path(tf.get('reference_mask',str(Path(row['anchor_camera']['path']).parent/'object_mask.png')))
            mask=np.array(Image.open(mask_path).convert('L'))>127
            assert mask.shape==(qwen.height,qwen.width)
            mask_box=bounds(mask)
            qbox=crop_box(mask_box,qwen.size)
            qfraction=max((mask_box[2]-mask_box[0])/(qbox[2]-qbox[0]),
                          (mask_box[3]-mask_box[1])/(qbox[3]-qbox[1]))
            qpanel=OUT/'panels'/(case+'_2d_reference.png')
            save_panel(qwen,qbox,qpanel)
            assert mask[qbox[1]:qbox[3],qbox[0]:qbox[2]].sum()==mask.sum()

            # world = scale * Q * local + b. Transform only the camera, never the Gaussians.
            scale=float(tf['scale'])
            Q=np.asarray(tf['rotation'],dtype=np.float64)
            b=np.asarray(tf['translation'],dtype=np.float64)
            assert np.max(np.abs(Q.T@Q-np.eye(3)))<1e-5 and np.linalg.det(Q)>.999
            base=anchor.resize(2048,2048)
            local_R=base.R@Q
            local_t=(base.R@b+base.t)/scale
            local=c.Camera(local_R,local_t,2048,2048,base.fx,base.fy,base.cx,base.cy,case+'_anchor_in_asset_coordinates')
            asset=c.GaussianData.load(asset_path)
            points=asset.xyz.cpu().numpy()[::100]
            world=scale*(points@Q.T)+b
            uv_world,z=base.project(world)
            uv_local,zlocal=local.project(points)
            valid=(z>.01)&(zlocal>.01)
            projection_error=float(np.max(np.abs(uv_world[valid]-uv_local[valid])))
            assert projection_error<.03,projection_error
            # Expand the image plane while preserving pose and focal lengths, to avoid clipping.
            pad=512
            wide=c.Camera(local.R,local.t,3072,3072,local.fx,local.fy,
                          local.cx+pad,local.cy+pad,local.name+'_expanded_canvas')
            rgb=c.render(wide,asset,background=torch.ones(3,device='cuda'),aux=False)['rgb']
            raw=OUT/'renders'/(case+'_asset_white.png')
            c.save_image(raw,rgb)
            alpha=c.raster(wide,asset,torch.ones_like(asset.xyz))[0].cpu().numpy()
            Image.fromarray(np.uint16(np.clip(alpha,0,1)*65535)).save(OUT/'renders'/(case+'_asset_alpha.png'))
            im=Image.open(raw).convert('RGB')
            content=(alpha>.001)|np.any(np.asarray(im)<254,axis=2)
            abox=bounds(content)
            assert abox[0]>0 and abox[1]>0 and abox[2]<3072 and abox[3]<3072,'Asset touches wide-canvas edge'
            crop=crop_box(abox,im.size,qfraction)
            retained=float(alpha[crop[1]:crop[3],crop[0]:crop[2]].sum()/alpha.sum())
            assert retained>.999,retained
            apanel=OUT/'panels'/(case+'_3d_gaussian.png')
            save_panel(im,crop,apanel)
            factor=SIZE[0]/(crop[2]-crop[0])
            displayed_cam=c.Camera(wide.R,wide.t,*SIZE,wide.fx*factor,wide.fy*factor,
                (wide.cx-crop[0])*factor,(wide.cy-crop[1])*factor,wide.name+'_display_crop')
            assert sha(asset_path)==source_hash,'Read-only M3 asset changed'
            # Record why the old generic presentation view was replaced.
            old_R=np.asarray(row['m3_camera']['R'])
            rotation_difference=float(np.degrees(np.arccos(np.clip((np.trace(old_R@local.R.T)-1)/2,-1,1))))
            records[case]=dict(qwen=record(qwen_path),reference_mask=record(mask_path),
                m3_asset=record(asset_path),m3_gaussian_count=len(asset.xyz),
                saved_transform=record(transform_path),saved_transform_data=tf,
                anchor_camera_source=record(row['anchor_camera']['path']),
                qwen_anchor_camera=anchor.record(),asset_camera_local=wide.record(),
                asset_display_camera_local=displayed_cam.record(),
                camera_conversion='R_local = R_world @ Q; t_local = (R_world @ translation + t_world) / scale',
                previous_F06_asset_camera=row['m3_camera'],
                camera_orientation_difference_from_prior_F06_degrees=rotation_difference,
                projection_conversion_max_error_pixels=projection_error,
                qwen_mask_bbox=mask_box,qwen_crop_pixels=qbox,reference_mask_fully_retained=True,
                asset_silhouette_bbox=abox,asset_crop_pixels=crop,retained_asset_alpha_mass=retained,
                target_normalized_extent=qfraction,
                qwen_panel=record(qpanel),asset_panel=record(apanel),raw_asset_render=record(raw),
                source_ply_modified=False,source_qwen_modified=False,
                case07_accepted_yaw_correction_reused=(case=='case07'),
                prompt_record=dict(A=task_rows[case]['A'],B=task_rows[case]['B']))
            print(case,'READ-ONLY RENDER COMPLETE',projection_error,retained,flush=True)
            del asset,rgb,alpha
            gc.collect();torch.cuda.empty_cache()
    dump(OUT/'render_sources.json',dict(cases=records,old_F06_manifest=record(OLD/'case_manifest.json'),
         prompt_manifest=record(task_file),gaussian_render_calls=3,optimization_steps=0,
         qwen_calls=0,trellis_runs=0,render_background='white',asset_geometry_unchanged=True))


def assemble():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from figure_f05_revised import FONT_DIR  # Registers the existing shared embedded fonts.
    import subprocess
    sources=json.loads((OUT/'render_sources.json').read_text())
    protected=[p for root in [V3,V4] for p in root.glob('*/figure_*') if p.is_file() and p.parent!=OUT]
    before={str(p):sha(p) for p in protected}
    width=180/25.4*72
    margin,labelw,labelgap,colgap,header,rowgap=3.,66.,3.,5.,20.,5.
    x0=margin+labelw+labelgap
    cellw=(width-x0-margin-colgap)/2
    cellh=cellw/1.6
    height=2*margin+header+3*cellh+2*rowgap
    pdf=OUT/'figure_F06_revised.pdf'
    c=canvas.Canvas(str(pdf),pagesize=(width,height),pageCompression=1)
    c.setTitle('Text-guided 2D-to-3D generation of industrial equipment')
    c.setAuthor('paper_eli - existing Qwen references and unchanged M3 Gaussian assets')
    for j,text in enumerate(['2D Generated Reference','3D Gaussian Asset']):
        c.setFont('PaperSans',8.4);c.setFillColorRGB(.06,.06,.06)
        c.drawCentredString(x0+j*(cellw+colgap)+cellw/2,height-margin-9,text)
    for i,(case,name) in enumerate(zip(CASES,NAMES)):
        y=height-margin-header-cellh-i*(cellh+rowgap)
        c.setFont('PaperSans',7.5);c.setFillColorRGB(.06,.06,.06)
        assert pdfmetrics.stringWidth(name,'PaperSans',7.5)<labelw
        c.drawString(margin,y+cellh/2+3,'Case'+case[-2:])
        c.drawString(margin,y+cellh/2-8,name)
        for j,suffix in enumerate(['2d_reference','3d_gaussian']):
            x=x0+j*(cellw+colgap)
            p=OUT/'panels'/(case+'_'+suffix+'.png')
            assert Image.open(p).size==SIZE
            c.drawImage(str(p),x,y,cellw,cellh)
            if j==1:
                c.setStrokeColorRGB(.91,.91,.91);c.setLineWidth(.22)
                c.rect(x,y,cellw,cellh,fill=0,stroke=1)
    c.showPage();c.save()
    stem=OUT/'figure_F06_revised'
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(pdf),str(stem)],check=True)
    png=stem.with_suffix('.png');im=Image.open(png);im.save(png,dpi=(300,300))
    ImageOps.contain(im,(1700,1800)).save(OUT/'figure_F06_revised_preview.jpg',quality=95)
    subprocess.run(['pdftoppm','-singlefile','-r','150','-png',str(pdf),str(OUT/'figure_F06_180mm_proof')],check=True)
    caption=('Fig. 6. Text-guided 2D-to-3D generation of industrial equipment. '
        'Generated 2D references and their corresponding 3D Gaussian assets are shown for three '
        'representative industrial equipment categories: a roller conveyor, an industrial chiller, '
        'and an automated storage cabinet. The pairs illustrate corresponding characteristic '
        'equipment structures in the scene-conditioned 2D references and independent 3D Gaussian '
        'representations, including rollers and support legs, ventilation and control components, '
        'and storage drawers. The original Qwen outputs are cropped to retain each complete '
        'reference-mask region and a limited amount of factory context. White-background images '
        'are new read-only renders of the unchanged saved M3 PBR-trained Gaussians. Their cameras '
        'are derived from the saved Anchor camera and accepted object-to-world registration, '
        'including the existing Case07 yaw correction, then reframed uniformly for comparable '
        'subject occupancy. Geometry, Gaussian parameters, textures, and source references are '
        'not edited; genuine appearance and reconstruction differences are retained. No Qwen, '
        'TRELLIS.2 generation, or Gaussian optimization was rerun.\n')
    (OUT/'caption.txt').write_text(caption,encoding='utf-8')
    sources['layout']=dict(width_mm=180,height_mm=height/72*25.4,rows=3,columns=2,
        cases=CASES,pixels=list(im.size),dpi=300,panel_canvas=SIZE,cell_mm=[cellw/72*25.4,cellh/72*25.4],
        font='DejaVu Sans embedded vector text',header_font_pt=8.4,case_font_pt=7.5,
        title_inside=False,original_scene_or_integrated_scene_shown=False)
    sources['script']=record(Path(__file__))
    dump(OUT/'sources.json',sources)
    assert before=={str(p):sha(p) for p in protected},'Previous figure changed'
    dump(OUT/'validation.json',dict(old_figures_unchanged=True,protected_sha256=before,
        png=record(png),pdf=record(pdf),six_panels=True,render_sources_only=True,
        all_m3_ply_hashes_unchanged=True,all_reference_masks_fully_within_crops=True))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>F06 2D-to-3D</title>
<style>body{font-family:Arial,sans-serif;max-width:1300px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}</style>
<h1>F06 · 工业设备 2D-to-3D 展示</h1><p>Case02 输送机、Case04 冷水机、Case07 存储柜。使用原 Qwen 结果与原 M3 Gaussian；仅裁剪、只读渲染及排版。</p>
<p><a href="figure_F06_revised.png">300 DPI PNG</a> · <a href="figure_F06_revised.pdf">PDF</a> · <a href="panels/">六张独立子图</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">来源与相机</a> · <a href="../F06_2d_to_3d.zip">完整包</a></p>
<img src="figure_F06_revised_preview.jpg">''',encoding='utf-8')
    with ZipFile(V4/'F06_2d_to_3d.zip','w',ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(output=str(OUT),pixels=list(im.size),width_mm=180,height_mm=height/72*25.4)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--assemble',action='store_true')
    a=parser.parse_args()
    if a.render:render()
    if a.assemble:assemble()

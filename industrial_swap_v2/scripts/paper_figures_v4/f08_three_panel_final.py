"""F08: three independent registered-photo views and a horizontal count chart.

Only existing Gaussian selections are rasterized for continuous display weights.
No localization, model generation, geometry changes or correctness audits run.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
SOURCE=BASE/'F08_localization_final'
OUT=BASE/'F08_final_relayout'
DATA=BASE/'F08_localization_revised'
CASE=DATA/'cases/60c31fa1a02c4a09b8b57494a1d1455d'
SCALE=3
COLOR=np.array([217.,120.,120.])/255.
GAIN=.60
TARGET_CROP=[585,298,770,483]
WRONG_CROP=[685,225,870,410]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def picture(a):return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))

def protected():
    files=[p for p in SOURCE.rglob('*') if p.is_file()]
    files += [CASE/'reference_mask.png',CASE/'reference_photo.jpg',
        CASE/'photo_assisted_indices.npy',CASE/'text_only_indices.npy',
        CASE/'view_0/camera.json',CASE/'native_text/camera.json']
    return {str(p):sha(p) for p in files}

def render(gpu):
    for folder in ['panels','renders','code','inputs']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    os.environ.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=str(gpu),GSLE_OUTPUT=str(OUT/'display_adapter'),
        GSLE_RECORDS=str(OUT/'display_records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(ROOT/'gsle'))
    import torch
    import common as c
    snapshot=protected()
    src=read(DATA/'sources.json')['scene'];scene_path=Path(src['path'])
    assert sha(scene_path)==src['sha256']
    scene=c.GaussianData.load(scene_path)
    records=[]
    for key,view,method in [('photo_guided','view_0','photo_assisted'),('text_only','native_text','text_only')]:
        cam=c.Camera.from_record(read(CASE/view/'camera.json'));hi=cam.resize(cam.width*SCALE,cam.height*SCALE)
        ids_path=CASE/(method+'_indices.npy');ids=np.load(ids_path)
        photo_path=CASE/'reference_photo.jpg' if key=='photo_guided' else ROOT/'data/industrial/images'/cam.name
        if not photo_path.exists():photo_path=CASE/view/'photograph.png'
        photo=Image.open(photo_path).convert('RGB')
        rgb=np.asarray(photo.resize((hi.width,hi.height),Image.Resampling.LANCZOS))/255.
        with torch.no_grad():
            flags=torch.zeros_like(scene.xyz);flags[torch.as_tensor(ids,device='cuda')]=1
            weight=c.raster(hi,scene,flags)[0].cpu().numpy()
        alpha=GAIN*weight[...,None]
        # True alpha-composited Gaussian contribution, never a thresholded mask.
        composite=rgb*(1-alpha)+COLOR*alpha
        picture(composite).save(OUT/'renders'/f'{key}_full.png',dpi=(300,300))
        picture(rgb).save(OUT/'renders'/f'{key}_registered_photo.png',dpi=(300,300))
        np.save(OUT/'renders'/f'{key}_contribution.npy',weight.astype(np.float16))
        records.append(dict(key=key,method=method,selected_gaussians=len(ids),indices=str(ids_path),indices_sha256=sha(ids_path),
            camera=cam.record(),display_camera=hi.record(),photograph=str(photo_path),photograph_sha256=sha(photo_path),
            photograph_resolution=list(photo.size),color_hex='#D97878',display_overlay_gain=GAIN,
            full_scene_depth_occlusion_retained=True,geometry_or_opacity_modified=False))
        print('Rendered existing selection:',key,len(ids),'at',cam.name,flush=True)
    assert protected()==snapshot and sha(scene_path)==src['sha256']
    shutil.copy2(CASE/'reference_mask.png',OUT/'inputs/intended_target_mask.png')
    shutil.copy2(SOURCE/'instance_selection.csv',OUT/'instance_selection.csv')
    write(OUT/'render_manifest.json',dict(scene=src,records=records,original_files_sha256=snapshot,
        actual_Gaussian_contribution=True,annotation='Original input-reference SAM denotes intended target; it is not a new evaluation GT.'))

def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    from matplotlib.transforms import Bbox
    manifest=read(OUT/'render_manifest.json')
    assert protected()==manifest['original_files_sha256']
    with (OUT/'instance_selection.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==9 and len({r['physical_target_id'] for r in rows})==9
    counts=[sum(int(r[k]) for r in rows) for k in ['text_only_correct','photo_guided_correct']]
    assert counts==[3,9]
    mask=np.asarray(Image.open(OUT/'inputs/intended_target_mask.png'))>127
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    w,h=180.,108.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    specs=[('a_specified_target','photo_guided_registered_photo.png',TARGET_CROP,True,'(a) Specified Target'),
           ('b_text_only','text_only_full.png',WRONG_CROP,False,'(b) Text-only Localization'),
           ('c_photo_guided','photo_guided_full.png',TARGET_CROP,True,'(c) Photo-guided Localization (Ours)')]
    image_axes=[]
    for col,(name,filename,crop,cyan,title) in enumerate(specs):
        x=1.5+59.5*col;y=43.;pw=58.
        ax=axis(x,y,pw,pw);image_axes.append(ax)
        image=Image.open(OUT/'renders'/filename)
        ax.imshow(image,extent=[-.5,937.5,527.5,-.5],interpolation='lanczos')
        if cyan:ax.contour(mask,levels=[.5],colors=['#00cfe7'],linewidths=.95,antialiased=True)
        ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5);ax.axis('off')
        text(x,104.3,title,ha='left',va='center',fontsize=7.7)
        if col:text(x+pw/2,40.2,'Wrong Instance' if col==1 else 'Correct Instance',ha='center',va='center',fontsize=7.4)
        image.crop(tuple(v*SCALE for v in crop)).save(OUT/'panels'/f'{name}_image.png',dpi=(300,300))
    fig.legend(handles=[Line2D([0],[0],color='#D97878',lw=3.8,label='Selected Gaussians'),
        Line2D([0],[0],color='#00cfe7',lw=1.5,label='Specified Target')],loc='center',bbox_to_anchor=(.5,35.8/h),
        ncol=2,frameon=False,fontsize=6.7,columnspacing=1.5,handlelength=1.6)
    text(1.5,31,'(d) Correct vs. Incorrect Instance Selection',ha='left',va='center',fontsize=8.1)
    correct='#196f86';incorrect='#d9a277'
    fig.legend(handles=[Patch(facecolor=correct,label='Correct'),Patch(facecolor=incorrect,label='Incorrect')],
        loc='center',bbox_to_anchor=(155/w,31/h),ncol=2,frameon=False,fontsize=6.1,
        handlelength=1.2,handletextpad=.4,columnspacing=1.2)
    ax=axis(36,11,139,16.5)
    values=np.array(counts)/9.
    ax.barh([1,0],values,height=.62,color=correct,zorder=3)
    ax.barh([1,0],1-values,left=values,height=.62,color=incorrect,zorder=3)
    ax.text(1/6,1,'3/9 (33.3%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(2/3,1,'6/9 (66.7%)',ha='center',va='center',fontsize=7.1,color='#3f2b20',fontweight='bold')
    ax.text(.5,0,'9/9 (100%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(.985,0,'Incorrect: 0/9 (0%)',ha='right',va='center',fontsize=5.8,color='white')
    ax.set_yticks([1,0],['Text-only','Photo-guided (Ours)']);ax.tick_params(axis='y',length=0,labelsize=7.1,pad=5)
    ax.set_ylim(-.55,1.55);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(axis='x',labelsize=6.1,length=2,pad=2)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#777777')
    ax.grid(axis='x',color='#ededed',zorder=0)
    text(1.5,2.3,'9 unique equipment targets',ha='left',va='center',fontsize=5.9,color='#555555')
    text(178.5,2.3,'Preliminary AI visual audit',ha='right',va='center',fontsize=5.9,color='#555555')
    fig.savefig(OUT/'figure_F08_final.png',dpi=300);fig.savefig(OUT/'figure_F08_final.pdf')
    for col,(name,*_) in enumerate(specs):
        bounds=[.8+59.5*col,38,60.2+59.5*col,107.]
        fig.savefig(OUT/'panels'/f'{name}.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in bounds)),dpi=300)
    fig.savefig(OUT/'panels/d_instance_statistics.png',bbox_inches=Bbox.from_extents(.5/25.4,.3/25.4,179.5/25.4,33.5/25.4),dpi=300)
    plt.close(fig)
    caption=('Fig. 8. Reference-photo-guided instance localization in an industrial scene. '
        '(a) The registered reference photograph of the user-specified M1-10 target, outlined in cyan. '
        '(b) The actual instance selected by text-only LangSplat, shown from its own saved registered camera '
        '(frame_000015), because it lies outside the main reference view. '
        '(c) Photo-guided localization of the specified target, using exactly the same photograph, camera '
        '(frame_000342) and crop as (a). Soft red (#D97878) is a semi-transparent composite of the actual '
        'full-scene alpha-composited selected-Gaussian contribution, not a filled binary mask. '
        'Original Gaussian opacity, covariance, geometry, selection indices and scene occlusion are retained. '
        '(d) Correct and incorrect instance selections over nine unique physical equipment targets. '
        'Text-only: 3/9 correct (33.3%) and 6/9 incorrect (66.7%); photo-guided: 9/9 correct (100%) and '
        '0/9 incorrect (0%). M1-04 and M1-09 are grouped as the same physical cooler. '
        'These are unchanged preliminary AI visual-audit judgments; independent human adjudication remains '
        'pending. Both settings share the same per-task text and frozen LangSplat fields. Some saved '
        'descriptions, including M1-10, imperfectly describe the user-selected equipment, so description '
        'mismatch is a confound in addition to instance ambiguity. All targets belong to one industrial '
        'scene. No localization, generation or optimization is rerun for this visualization.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    write(OUT/'sources.json',dict(previous_F08_directory=str(SOURCE),previous_sources=read(SOURCE/'sources_revised.json'),
        revision='Three independent registered-photo panels plus horizontal correct/incorrect stacked bars',
        render_manifest=manifest,figure_mm=[w,h],dpi=300,vector_PDF_text_and_contours=True,
        qualitative_case='M1-10',m1_case_id=CASE.name,shared_ac_camera='frame_000342.jpg',b_camera='frame_000015.jpg',
        ac_crop=TARGET_CROP,b_crop=WRONG_CROP,no_insets=True,qualitative_panel_mm=[58,58],
        Gaussian_color='#D97878',overlay_gain=GAIN,actual_continuous_gaussian_weights=True,
        instance_selection_source=str(SOURCE/'instance_selection.csv'),instance_selection_sha256=sha(OUT/'instance_selection.csv'),
        correct_counts=counts,incorrect_counts=[9-n for n in counts],unique_targets=9,
        independent_human_adjudication=False,physical_target_rows=rows,
        original_camera_selection_geometry_and_statistics_unchanged=True))
    assert protected()==manifest['original_files_sha256']
    assert sha(OUT/'instance_selection.csv')==sha(SOURCE/'instance_selection.csv')
    write(OUT/'validation.json',dict(source_files_sha256_unchanged=True,CSV_byte_identical=True,
        same_a_c_photograph_camera_and_crop=True,b_uses_actual_saved_wrong_instance_camera=True,
        true_gaussian_projection=True,no_binary_mask_overlay=True,no_insets=True,
        correct_counts=counts,incorrect_counts=[9-n for n in counts],unique_targets=9,
        no_localization_generation_training_or_scoring_rerun=True))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - Final relayout</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08 · 三图与横向实例选择统计</h1><p>指定目标 → Text-only 实际误选实例 → Photo-guided 正确实例。无嵌套小窗；统计沿用 9 个去重目标的初步 AI 视觉核验。</p>
<p><a href="figure_F08_final.png">300 DPI PNG</a> · <a href="figure_F08_final.pdf">PDF</a> · <a href="panels/">独立子图</a> · <a href="instance_selection.csv">原值 CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F08_localization_final/index_revised.html">上一版</a></p>
<img src="figure_F08_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('F08 final relayout exported; source results and audit CSV unchanged.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','render','figure'],default='all');p.add_argument('--gpu',default='2');a=p.parse_args()
    if a.stage in ['all','render']:render(a.gpu)
    if a.stage in ['all','figure']:figure()

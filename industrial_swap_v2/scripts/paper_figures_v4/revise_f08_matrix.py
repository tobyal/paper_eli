"""F08 visual-only revision: larger real splat views and a case-wise audit matrix.

Reads the existing deduplicated instance_selection.csv verbatim. Re-renders
only saved Gaussian selections/cameras for display; no localization or scoring.
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
OUT=BASE/'F08_localization_final'
DATA=BASE/'F08_localization_revised'
CASE=DATA/'cases/60c31fa1a02c4a09b8b57494a1d1455d'
RENDERS=OUT/'renders_revised'
PANELS=OUT/'panels_revised'
SCALE=3
TINT=.60
MAIN_CROP=[180,80,908,510]
TARGET_CROP=[585,298,770,483]
WRONG_CROP=[643,205,867,429]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def protected_files():
    names=['figure_F08.png','figure_F08.pdf','caption.txt','sources.json','instance_selection.csv','index.html']
    files=[OUT/x for x in names]
    files+=list((OUT/'panels').glob('*.png'))
    files+=[CASE/'reference_mask.png',CASE/'text_only_indices.npy',CASE/'photo_assisted_indices.npy',
            CASE/'view_0/camera.json',CASE/'native_text/camera.json',
            Path(__file__).with_name('f08_instance_audit.json')]
    return {str(p):sha(p) for p in files}

def render(gpu):
    os.environ.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=str(gpu),GSLE_OUTPUT=str(OUT/'display_adapter'),
        GSLE_RECORDS=str(OUT/'display_records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(ROOT/'gsle'))
    import torch
    import common as c
    RENDERS.mkdir(exist_ok=True)
    snapshot=protected_files()
    scene_info=read(DATA/'sources.json')['scene'];path=Path(scene_info['path'])
    assert sha(path)==scene_info['sha256']
    scene=c.GaussianData.load(path)
    views=[('a_main','view_0','text_only'),('b_main','view_0','photo_assisted'),
           ('wrong_instance','native_text','text_only')]
    records=[]
    for key,view,method in views:
        cam=c.Camera.from_record(read(CASE/view/'camera.json'))
        hi=cam.resize(cam.width*SCALE,cam.height*SCALE)
        index_path=CASE/(method+'_indices.npy');ids=np.load(index_path)
        with torch.no_grad():
            colors=scene.colors(hi.center)
            ti=torch.as_tensor(ids,device='cuda')
            colors[ti]=(1-TINT)*colors[ti]+TINT*colors.new_tensor([1.,0.,0.])
            pixels=c.raster(hi,scene,colors)
            c.save_image(RENDERS/(key+'.png'),pixels)
        records.append(dict(key=key,method=method,camera=cam.record(),display_camera=hi.record(),
            selected_gaussians=len(ids),indices=str(index_path),indices_sha256=sha(index_path),
            previous_red_tint=.5,new_red_tint=TINT,opacity_or_geometry_modified=False))
        print('Rendered display:',key,len(ids),'unchanged selected Gaussians',flush=True)
    assert protected_files()==snapshot and sha(path)==scene_info['sha256']
    write(RENDERS/'manifest.json',dict(scene=scene_info,scene_unchanged=True,source_hashes=snapshot,
        records=records,render_scale=SCALE,display='Actual full-scene Gaussian rasterization, selected SH color mixed with red; original opacity, covariance, geometry and depth compositing retained.'))


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox

    PANELS.mkdir(exist_ok=True)
    source=read(OUT/'sources.json');render_manifest=read(RENDERS/'manifest.json')
    assert protected_files()==render_manifest['source_hashes']
    with (OUT/'instance_selection.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    assert len(records)==9 and len({r['physical_target_id'] for r in records})==9
    matrix=np.array([[int(r[k]) for r in records] for k in ['text_only_correct','photo_guided_correct']])
    assert matrix.shape==(2,9) and matrix.sum(1).tolist()==[3,9]
    assert [r['physical_target_id'] for r in records]==['P01','P02','P03','P04','P05','P06','P07','P08','P10']
    labels=['/'.join(x.replace('M1-','') for x in r['m1_labels'].split(';')) for r in records]
    mask=np.asarray(Image.open(CASE/'reference_mask.png'))>127
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    w,h=180.,104.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    # Contour the original target mask on its own native grid, but draw the
    # line as an antialiased vector, without enlarging a binary raster edge.
    def image_view(ax,im,crop,show_target=False):
        ax.imshow(im,extent=[-.5,937.5,527.5,-.5],interpolation='lanczos')
        if show_target:
            ax.contour(mask,levels=[.5],colors=['#00cfe7'],linewidths=.95,antialiased=True)
        ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5)
        ax.set_xticks([]);ax.set_yticks([])
    scene_w=87.;scene_h=scene_w*(MAIN_CROP[3]-MAIN_CROP[1])/(MAIN_CROP[2]-MAIN_CROP[0])
    axes=[]
    for i,(key,title) in enumerate([('a_main','(a) Text-only Localization'),('b_main','(b) Photo-guided Localization')]):
        x=1.5+90*i;y=45.5
        ax=axis(x,y,scene_w,scene_h);axes.append(ax)
        im=Image.open(RENDERS/(key+'.png'))
        image_view(ax,im,MAIN_CROP,True)
        for spine in ax.spines.values():spine.set_visible(False)
        text(x,y+scene_h+3.4,title,ha='left',va='center',fontsize=8.4)
        # All three zoom windows are 23.5-mm squares, with identical borders
        # and labels placed inside a small dark strip.
        locations=[(x+scene_w-24.5,y+scene_h-24.5,im,TARGET_CROP,
            'Intended target' if i==0 else 'Correct target',True)]
        if i==0:
            locations.append((x+1.0,y+1.0,Image.open(RENDERS/'wrong_instance.png'),WRONG_CROP,'Wrong instance',False))
        for zx,zy,zim,box,label,has_mask in locations:
            ia=axis(zx,zy,23.5,23.5);image_view(ia,zim,box,has_mask)
            for spine in ia.spines.values():spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.8)
            ia.text(.035,.965,label,transform=ia.transAxes,va='top',ha='left',fontsize=5.8,color='white',
                bbox=dict(facecolor='#171717',alpha=.82,pad=1.3,edgecolor='none'))
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.8,label='Selected Gaussians'),
        Line2D([0],[0],color='#00cfe7',lw=1.5,label='Intended target')],loc='center',bbox_to_anchor=(.5,41.4/h),
        ncol=2,frameon=False,fontsize=6.8,columnspacing=1.7,handlelength=1.7)
    # The matrix entries are copied directly from the existing deduplicated CSV.
    # No new audit, correctness rule or synthetic case result is introduced.
    text(1.5,36,'(c) Case-wise Correct Instance Selection',ha='left',va='center',fontsize=8.4)
    correct='#196f86';incorrect='#d89a68'
    fig.legend(handles=[Line2D([0],[0],marker='s',lw=0,color=correct,markersize=5,label='Correct'),
        Line2D([0],[0],marker='s',lw=0,color=incorrect,markersize=5,label='Incorrect')],loc='center',
        bbox_to_anchor=(156/w,36/h),ncol=2,frameon=False,fontsize=6.1,handletextpad=.4,columnspacing=1.)
    for col,label in enumerate(labels):text(40+12*col,31.1,label,ha='center',va='center',fontsize=7.1)
    text(162,31.1,'Summary',ha='center',va='center',fontsize=7.1)
    for row,(name,cy) in enumerate([('Text-only',25.),('Photo-guided',15.)]):
        text(31.5,cy,name,ha='right',va='center',fontsize=7.5)
        for col,ok in enumerate(matrix[row]):
            cx=40+12*col
            fig.patches.append(Rectangle(((cx-4)/w,(cy-4)/h),8/w,8/h,
                facecolor=correct if ok else incorrect,edgecolor='none',transform=fig.transFigure))
            if ok:
                px=[cx-1.9,cx-.45,cx+2.05];py=[cy-.05,cy-1.45,cy+1.65]
                fig.lines.append(Line2D(np.array(px)/w,np.array(py)/h,transform=fig.transFigure,
                    color='white',lw=1.25,solid_capstyle='round',solid_joinstyle='round'))
            else:
                for ys in [[cy-1.5,cy+1.5],[cy+1.5,cy-1.5]]:
                    fig.lines.append(Line2D(np.array([cx-1.5,cx+1.5])/w,np.array(ys)/h,
                        transform=fig.transFigure,color='white',lw=1.25,solid_capstyle='round'))
        n=int(matrix[row].sum());percent=f'{n/9*100:.1f}%' if n!=9 else '100%'
        text(162,cy,f'{n}/9  ({percent})',ha='center',va='center',fontsize=7.6,fontweight='bold',color=correct if row else '#8e552b')
    text(178.5,4.7,'Preliminary AI visual audit',ha='right',va='center',fontsize=5.9,color='#555555')
    fig.savefig(OUT/'figure_F08_revised.png',dpi=300);fig.savefig(OUT/'figure_F08_revised.pdf')
    boxes={'a_text_only':[.5,44.5,89.5,103.], 'b_photo_guided':[90.5,44.5,179.5,103.],
           'c_instance_matrix':[.5,2.5,179.5,39.], 'legend':[47.,38.,134.,44.5]}
    for name,box in boxes.items():
        fig.savefig(PANELS/(name+'.png'),bbox_inches=Bbox.from_extents(*(v/25.4 for v in box)),dpi=300)
    plt.close(fig)
    # Export unlabelled crops as replaceable figure assets too.
    for key in ['a_main','b_main']:
        Image.open(RENDERS/(key+'.png')).crop(tuple(v*SCALE for v in MAIN_CROP)).save(PANELS/(key+'_scene.png'),dpi=(300,300))
    for name,key,box in [('a_target','a_main',TARGET_CROP),('b_target','b_main',TARGET_CROP),('wrong_instance','wrong_instance',WRONG_CROP)]:
        Image.open(RENDERS/(key+'.png')).crop(tuple(v*SCALE for v in box)).save(PANELS/(name+'_detail.png'),dpi=(300,300))
    caption=(OUT/'caption.txt').read_text(encoding='utf-8').strip()
    caption=caption.replace('(c) Correct-instance counts are','(c) The case-wise selection matrix shows one column per unique physical target; teal check marks denote correct selections and orange crosses denote incorrect selections. Correct-instance counts are')
    caption+=' The 04/09 column denotes their shared physical target, not two independent devices. Only display framing, inset styling, contour width and the red display tint (0.50 to 0.60) are changed; selected Gaussian parameters, indices, camera poses, the audit CSV and all judgments are unchanged.'
    (OUT/'caption_revised.txt').write_text(caption+'\n',encoding='utf-8')
    write(OUT/'sources_revised.json',dict(previous_sources=source,revision='Larger shared-camera Gaussian views and 2 x 9 case-wise instance-selection matrix',
        render_manifest=render_manifest,figure_mm=[w,h],dpi=300,main_crop=MAIN_CROP,
        intended_target_detail_crop=TARGET_CROP,wrong_instance_detail_crop=WRONG_CROP,
        inset_size_mm=[23.5,23.5],GT_contour_linewidth_points=.95,previous_contour_linewidth_points=.6,
        matrix_source=str(OUT/'instance_selection.csv'),matrix_source_sha256=sha(OUT/'instance_selection.csv'),
        matrix_rows=['Text-only','Photo-guided'],matrix_columns=labels,
        matrix_physical_target_ids=[r['physical_target_id'] for r in records],matrix_values=matrix.tolist(),
        matrix_cases=records,correct_counts=matrix.sum(1).tolist(),physical_targets=9,
        human_adjudicated=False,audit_label='Preliminary AI visual audit',
        old_figure_files_preserved=True,original_data_and_judgments_unchanged=True))
    assert protected_files()==render_manifest['source_hashes']
    write(OUT/'validation_revised.json',dict(original_files_sha256_unchanged=True,matrix_matches_saved_CSV=True,
        deduplicated_targets=9,correct_counts=matrix.sum(1).tolist(),text_correct_columns=[labels[i] for i in np.where(matrix[0]==1)[0]],
        same_main_camera_and_crop=True,insets_same_physical_size=True,real_gaussian_rendering=True,
        no_localization_training_or_evaluation_rerun=True))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_revised.pdf'),str(OUT/'pdf_proof_revised')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - Instance selection matrix</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08 · 逐目标实例选择矩阵</h1><p>9 个去重物理目标；Text-only 3/9，Photo-guided 9/9。04/09 合并为同一设备。仍为 AI 初步视觉核验，数据不变。</p>
<p><a href="figure_F08_revised.png">300 DPI PNG</a> · <a href="figure_F08_revised.pdf">PDF</a> · <a href="panels_revised/">独立子图</a> · <a href="instance_selection.csv">原始判断 CSV</a> · <a href="caption_revised.txt">Caption</a> · <a href="sources_revised.json">Sources</a> · <a href="index.html">旧版</a></p><img src="figure_F08_revised.png">'''
    (OUT/'index_revised.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('F08 revised: actual 2 x 9 matrix, counts 3/9 and 9/9, original figure/data untouched.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['render','figure','all'],default='all');p.add_argument('--gpu',default='2');a=p.parse_args()
    if a.stage in ['render','all']:render(a.gpu)
    if a.stage in ['figure','all']:figure()

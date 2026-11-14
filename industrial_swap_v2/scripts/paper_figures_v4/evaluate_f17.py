"""Common-camera Gaussian localization evaluation against supplied AI drafts.

Scores every nonempty supplied mask for transparency; primary estimates exclude
the pre-identified wrong-instance draft and cameras used by either native tracer.
Missing annotations remain NA. Selection is never fitted to evaluation masks.
"""
import argparse, csv, hashlib, io, json, os, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
SRC=BASE/'F08_localization_revised'
OUT=BASE/'F17_localization_comparison'
METHODS=['GaussianEditor','DGE','Ours']
WRONG='M1-06_frame_000202'
THRESHOLD=.05

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def csv_save(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def iou(a,b):
    assert a.shape==b.shape
    union=int((a|b).sum());return float((a&b).sum()/union) if union else None
def name(x):return Path(x).stem

def baseline_sources(number,method):
    if number in [3,6,5]:
        d=OUT/'native_localization'/method/('query_05' if number==5 else 'query_03')
        assert (d/'complete.json').exists(),f'Incomplete native localization: {d}'
    else:
        case={2:'case02',4:'case04',7:'case07',9:'case04',10:'case10'}.get(number)
        if case is None:return None
        d=ROOT/'baseline_comparison'/('gaussianeditor_native' if method=='GaussianEditor' else 'dge_native')/case
    ids=d/('localization/selected_indices.npy' if method=='GaussianEditor' else 'target_indices.npy')
    meta=read(d/'records/localization.json')
    camrows=meta.get('sampled_cameras',meta.get('segmentation_cameras',[]))
    cams=[name(x['name'] if isinstance(x,dict) else x) for x in camrows]
    assert cams
    prompts=read(d/('prompts.json' if (d/'prompts.json').exists() else 'config.json'))
    prompt=prompts.get('A',prompts.get('prompt'))
    return dict(indices=ids,directory=d,localization_record=meta,used_cameras=cams,prompt=prompt)

def overlay(rgb,gt,pred=None):
    a=np.array(rgb.convert('RGB'),copy=True)
    if pred is not None:a[pred]=np.uint8(.55*a[pred]+.45*np.array([255,30,30]))
    edge=binary_dilation(gt,iterations=1)^binary_erosion(gt,iterations=1)
    a[edge]=[0,210,230]
    return Image.fromarray(a)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='2');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    for d in ['ground_truth_masks','projected_masks','panels','photos','selection_indices','code']:(OUT/d).mkdir(exist_ok=True)
    os.environ.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,GSLE_OUTPUT=str(OUT/'render_adapter'),GSLE_RECORDS=str(OUT/'records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(ROOT/'gsle'))
    import torch
    import common as c
    index=read(SRC/'case_index.json')['cases'];cases={x['m1_case_id']:x for x in index}
    tasks=read(SRC/'evaluation_views.json')['views']
    package=SRC/'with_8_masks/inputs/F08_heldout_masks_AI_draft.zip'
    scene_path=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    scene_hash=sha(scene_path)
    assert scene_hash=='1d5cc6a5b58e7d99fbefd54ba6de7ab351ee23b884b2fcacba11720906b5bf7a'
    scene=c.GaussianData.load(scene_path);N=len(scene.xyz)
    sources={};selections={};validation=[]
    # Require identical Text A for the evaluated physical targets; source cohort
    # has two confirmed descriptions of P04, of which M1-04 is canonical here.
    for rec in index:
        num=rec['catalog_number'];label=rec['label']
        if num==9:continue
        selections[label]={};sources[label]={}
        for method in METHODS:
            if method=='Ours':
                pth=SRC/'cases'/rec['m1_case_id']/'photo_assisted_indices.npy'
                src=dict(indices=pth,used_cameras=[name(rec['reference_camera']['name'])],prompt=rec['description'])
            else:
                src=baseline_sources(num,method)
                if src is None:continue
                pth=src['indices']
                assert src['prompt']==rec['description'],(label,method,src['prompt'],rec['description'])
            ids=np.load(pth);assert ids.ndim==1 and len(np.unique(ids))==len(ids)
            assert len(ids)==0 or (ids.min()>=0 and ids.max()<N)
            dest=OUT/'selection_indices'/f'{label}_{method}.npy';shutil.copy2(pth,dest)
            selections[label][method]=ids
            sources[label][method]={k:str(v) if isinstance(v,Path) else v for k,v in src.items()}
            sources[label][method].update(sha256=sha(pth),gaussians=len(ids))
    write(OUT/'selection_sources.json',sources)
    byview=[];views=[]
    with zipfile.ZipFile(package) as z:
        for task in tasks:
            label=task['label'];rec=cases[task['m1_case_id']];tid=task['task_id']
            prefix='annotation_tasks/'+tid+'/'
            review=json.loads(z.read(prefix+'review.json'));camdict=json.loads(z.read(prefix+'camera.json'))
            assert review['task_id']==tid and not review['derived_from_method_prediction']
            assert review['reviewer_type']=='ai' and review['reviewed'] is False
            rawimage=z.read(prefix+'image.png');rawmask=z.read(prefix+'mask.png')
            assert hashlib.sha256(rawimage).hexdigest()==task['image_sha256']==review['image_sha256']
            assert name(camdict['name'])!=name(task['input_camera'])
            for key in ['R','t','width','height','fx','fy','cx','cy']:
                assert np.allclose(camdict[key],task['camera'][key],rtol=0,atol=1e-7)
            gt=np.asarray(Image.open(io.BytesIO(rawmask)))>127
            assert gt.shape==(camdict['height'],camdict['width'])==(528,938)
            d=OUT/'ground_truth_masks'/tid;d.mkdir(exist_ok=True)
            for f in ['mask.png','review.json','polygon.json','camera.json']:(d/f).write_bytes(z.read(prefix+f))
            nonempty=bool(gt.any());canonical=label!='M1-09'
            used={m:name(camdict['name']) in sources.get(label,{}).get(m,{}).get('used_cameras',[]) for m in METHODS}
            reasons=[]
            if not nonempty:reasons.append('missing_nonempty_independent_mask')
            if tid==WRONG:reasons.append('wrong_physical_instance_in_AI_draft')
            if any(used.values()):reasons.append('camera_used_for_native_localization')
            if not canonical:reasons.append('duplicate_physical_target_P04')
            if len(selections.get(label,{}))!=3:reasons.append('missing_native_selection')
            primary=not reasons
            views.append(dict(**task,ai_draft_nonempty=nonempty,primary_included=primary,exclusion_reasons=reasons,
                camera_used_for_localization=used,annotation_review=review,GT_sha256=hashlib.sha256(rawmask).hexdigest()))
            if nonempty:
                (OUT/'photos'/f'{tid}.png').write_bytes(rawimage)
                image=Image.open(io.BytesIO(rawimage))
                overlay(image,gt).save(OUT/'panels'/f'{tid}_GT.png')
                cam=c.Camera(R=np.array(camdict['R']),t=np.array(camdict['t']),width=camdict['width'],height=camdict['height'],
                    fx=camdict['fx'],fy=camdict['fy'],cx=camdict['cx'],cy=camdict['cy'],name=camdict['name'])
            for method in METHODS:
                score=None;inter=None;union=None;predpixels=None;path=None
                if nonempty and method in selections.get(label,{}):
                    ids=selections[label][method]
                    colors=torch.zeros_like(scene.xyz);colors[torch.as_tensor(ids,device='cuda')]=1
                    with torch.no_grad():contribution=c.raster(cam,scene,colors)[0].cpu().numpy()
                    pred=contribution>THRESHOLD;score=iou(pred,gt);inter=int((pred&gt).sum());union=int((pred|gt).sum());predpixels=int(pred.sum())
                    path=OUT/'projected_masks'/f'{tid}_{method}.png'
                    Image.fromarray(np.uint8(pred)*255).save(path)
                    overlay(image,gt,pred).save(OUT/'panels'/f'{tid}_{method}.png')
                    if method=='Ours':
                        old=np.asarray(Image.open(task['predicted_masks']['photo_assisted']))>127
                        validation.append(dict(task_id=tid,ours_matches_previous_mask=bool(np.array_equal(pred,old)),different_pixels=int((pred!=old).sum())))
                    print(tid,method,'IoU',round(score,6),'primary',primary,flush=True)
                byview.append(dict(physical_target_id='P04' if label=='M1-09' else 'P'+label[-2:],m1_case_id=task['m1_case_id'],label=label,
                    task_id=tid,camera=camdict['name'],method=method,primary_included=primary,primary_heldout_iou=score if primary else None,
                    supplied_draft_iou=score,intersection_pixels=inter,union_pixels=union,prediction_pixels=predpixels,GT_pixels=int(gt.sum()),
                    exclusion_reason=';'.join(reasons),annotation_status='AI polygon draft; not human reviewed',
                    camera_used_for_localization=used[method],prediction_path=str(path) if path else None))
    bytarget=[];summaries=[]
    for rec in index:
        if rec['catalog_number']==9:continue
        label=rec['label']
        for method in METHODS:
            rows=[x for x in byview if x['label']==label and x['method']==method]
            valid=[x['primary_heldout_iou'] for x in rows if x['primary_included']]
            supplied=[x['supplied_draft_iou'] for x in rows if x['supplied_draft_iou'] is not None]
            bytarget.append(dict(physical_target_id='P'+label[-2:],label=label,m1_case_id=rec['m1_case_id'],description=rec['description'],method=method,
                heldout_iou=float(np.mean(valid)) if valid else None,valid_views=len(valid),
                all_supplied_drafts_iou=float(np.mean(supplied)) if supplied else None,
                supplied_views=len(supplied),status='preliminary_AI_draft' if valid else 'NA_missing_eligible_GT'))
    common=[label for label in selections if all(any(r['label']==label and r['method']==m and r['heldout_iou'] is not None for r in bytarget) for m in METHODS)]
    for m in METHODS:
        vals=[r['heldout_iou'] for r in bytarget if r['method']==m and r['label'] in common]
        allvals=[r['all_supplied_drafts_iou'] for r in bytarget if r['method']==m and r['all_supplied_drafts_iou'] is not None]
        summaries.append(dict(method=m,mean_heldout_iou=float(np.mean(vals)),common_valid_targets=len(vals),
            common_valid_views=sum(x['primary_included'] and x['method']==m for x in byview),
            total_distinct_targets=9,missing_eligible_targets=9-len(vals),annotation_status='preliminary_AI_draft_not_human_adjudicated',
            all_8_drafts_sensitivity_mean=float(np.mean(allvals)),aggregation='mean within target, then unweighted mean across common targets'))
    csv_save(OUT/'iou_per_view.csv',byview);csv_save(OUT/'iou_per_target.csv',bytarget);csv_save(OUT/'iou_summary.csv',summaries)
    write(OUT/'metrics.json',dict(per_view=byview,per_target=bytarget,summary=summaries,common_targets=common))
    write(OUT/'evaluation_views.json',dict(views=views))
    write(OUT/'sources.json',dict(source_scene=str(scene_path),scene_sha256=scene_hash,scene_gaussians=N,
        annotation_package=str(package),annotation_package_sha256=sha(package),mask_threshold=THRESHOLD,resolution=[938,528],
        prediction='Full original-scene alpha-composited contribution of actual selected Gaussians > 0.05; unselected Gaussians retain geometry and opacity but carry zero color.',
        display='Original registered photograph; red prediction alpha 0.45; cyan AI-draft contour; shared per-row crop.',
        no_generation_or_scene_optimization=True,simulated_table_used=False,selection_sources='selection_sources.json',
        primary_exclusion_policy='Pre-identified wrong-instance draft and any native-localization camera excluded; no exclusion based on IoU.',
        wrong_instance_task=WRONG,wrong_instance_reason='Draft outlines central-aisle cooler, whereas M1-06 designates rack-side cooler at extreme left edge; see original reference and multi-view audit.',
        mean_rule='Views within target, then targets; not treating multiple views as independent scenes.',
        guidance_difference='GE/DGE receive identical Text A only; Ours additionally uses registered reference-photo instance information.',
        heldout_scope='Not used for per-target reference assistance or the sampled native tracing. Images belong to the original scene/language-field training capture.',
        repeated_text='M1-03 and M1-06 share exactly the same Text A, so GE and DGE reuse the same native text-only selection. Ours retains the distinct confirmed instances.'))
    assert sha(scene_path)==scene_hash
    write(OUT/'validation.json',dict(original_scene_unchanged=True,common_cameras=True,mask_threshold_identical=True,
        original_AI_masks_unchanged=True,ours_previous_mask_checks=validation,common_targets=common,
        excluded_nonempty_tasks=[dict(task_id=v['task_id'],reasons=v['exclusion_reasons']) for v in views if v['ai_draft_nonempty'] and not v['primary_included']]))
    for f in ['evaluate_f17.py','f17_native_localize.py']:shutil.copy2(Path(__file__).with_name(f),OUT/'code'/f)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()

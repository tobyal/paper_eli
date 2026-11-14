"""Read independent annotations only; keep missing IoU null, export provenance/tables.

No GPU/model calls. Annotation masks must be independently human reviewed, with
camera and image hashes matching evaluation_views.json. Empty/not-visible views
are excluded. This script never creates a ground-truth mask from predictions.
"""
import json,hashlib,shutil,html,zipfile,csv
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F08_localization_revised'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_export(path,rows):
    # Flat scientific records; CSV export is not exposed by bundled artifact-tool.
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else ('NA' if v is None else v) for k,v in row.items()})

def evaluate_mask(task):
    review=Path(task['review_path']);mask=Path(task['mask_path'])
    if not review.exists():return 'pending_independent_human_annotation',None
    r=read(review)
    if not r.get('reviewed') or not r.get('reviewer') or r.get('reviewer_type')!='human':return 'not_human_reviewed',None
    if r.get('task_id')!=task['task_id'] or r.get('image_sha256')!=task['image_sha256']:raise ValueError('Annotation identity mismatch '+task['task_id'])
    if r.get('visible') is False:return 'not_visible',None
    if r.get('visible') is not True:return 'visibility_unresolved',None
    if r.get('mask_origin') not in ('independent_manual','independent_sam_human_corrected'):return 'nonindependent_mask_rejected',None
    if r.get('derived_from_method_prediction') is not False:return 'mask_independence_unconfirmed',None
    if not mask.exists():return 'mask_missing',None
    gt=np.asarray(Image.open(mask).convert('L'))>127
    if gt.shape!=(task['camera']['height'],task['camera']['width']):raise ValueError('Mask/camera resolution mismatch')
    if not gt.any():return 'empty_annotation_requires_visibility_review',None
    if task['camera']['name']==task['input_camera']:raise ValueError('Input reference cannot be held out')
    out={}
    for method,p in task['predicted_masks'].items():
        pred=np.asarray(Image.open(p).convert('L'))>127
        assert pred.shape==gt.shape
        out[method]=float((pred&gt).sum()/(pred|gt).sum())
    return 'valid',out

def main():
    index=read(OUT/'case_index.json');cases=index['cases'];assert len(cases)==10
    audit=read(Path(__file__).with_name('f08_instance_audit.json'));judgements={x['m1_case_id']:x for x in audit['cases']}
    assert set(judgements)=={x['m1_case_id'] for x in cases}
    shutil.copy2(Path(__file__).with_name('f08_instance_audit.json'),OUT/'instance_audit.json')
    views=read(OUT/'evaluation_views.json');heldout=[]
    for task in views['views']:
        status,values=evaluate_mask(task);task['annotation_status']=status
        for method in ['text_only','photo_assisted']:
            heldout.append(dict(m1_case_id=task['m1_case_id'],label=task['label'],camera=task['camera']['name'],method=method,
                heldout_iou=None if values is None else values[method],status=status,mask_path=task['mask_path'],review_path=task['review_path']))
        # Explicit pending template, not a review or an annotation.
        template=dict(task_id=task['task_id'],image_sha256=task['image_sha256'],reviewed=False,reviewer='',reviewer_type='human',
            visible=None,mask_origin='independent_manual',derived_from_method_prediction=False,notes='')
        save(Path(task['review_path']).with_name('review_template.json'),template)
    metrics=[];instances=[]
    for row in cases:
        cid=row['m1_case_id'];j=judgements[cid]
        vals={m:[r['heldout_iou'] for r in heldout if r['m1_case_id']==cid and r['method']==m and r['status']=='valid'] for m in ['text_only','photo_assisted']}
        assert len(vals['text_only'])==len(vals['photo_assisted'])
        metrics.append(dict(m1_case_id=cid,label=row['label'],target_description=row['description'],
            text_only_correct_instance=j['text_only_correct'],photo_assisted_correct_instance=j['photo_assisted_correct'],
            text_only_heldout_iou=float(np.mean(vals['text_only'])) if vals['text_only'] else None,
            photo_assisted_heldout_iou=float(np.mean(vals['photo_assisted'])) if vals['photo_assisted'] else None,
            number_valid_evaluation_views=len(vals['text_only']),
            reference_sam_agreement_text_only=row['input_sam_diagnostic']['text_only']['iou'],
            reference_sam_agreement_photo_assisted=row['input_sam_diagnostic']['photo_assisted']['iou'],
            physical_target_group=j['physical_target_group'],instance_review_type=audit['reviewer_type'],notes=j['notes']))
        instances.append(dict(m1_case_id=cid,label=row['label'],target_description=row['description'],
            text_only_correct_instance=j['text_only_correct'],photo_assisted_correct_instance=j['photo_assisted_correct'],
            physical_target_group=j['physical_target_group'],reference_camera=row['reference_camera']['name'],
            text_only_native_camera=row['text_native_camera']['name'],text_center=row['text_center'],photo_center=row['photo_center'],
            center_distance_world_units=row['center_distance'],gaussian_membership_iou=row['shared_gaussian_iou'],
            review_type=audit['reviewer_type'],independent_human_adjudication=False,notes=j['notes'],review_sheet=row['review_sheet']))
    summary={}
    for method in ['text_only','photo_assisted']:
        key=method+'_correct_instance';valid=[r for r in metrics if r[key] is not None];correct=sum(bool(r[key]) for r in valid)
        groups={}
        for r in metrics:groups.setdefault(r['physical_target_group'],[]).append(r[key])
        # Both queries for repeated target must succeed, conservatively, for group success.
        group_valid=[v for v in groups.values() if all(x is not None for x in v)]
        target_means=[r[method+'_heldout_iou'] for r in metrics if r[method+'_heldout_iou'] is not None]
        summary[method]=dict(correct=correct,valid_cases=len(valid),case_selection_rate=correct/len(valid) if valid else None,
            unique_physical_targets=len(group_valid),unique_correct=sum(all(v) for v in group_valid),
            unique_target_rate=sum(all(v) for v in group_valid)/len(group_valid),
            heldout_mean_iou=float(np.mean(target_means)) if target_means else None,heldout_valid_cases=len(target_means))
    # Existing confirmed case list is retained. Expose additional physical duplication.
    a=next(x for x in cases if x['catalog_number']==4);b=next(x for x in cases if x['catalog_number']==9)
    ia=np.load(a['photo_indices_source']);ib=np.load(b['photo_indices_source']);n=len(np.intersect1d(ia,ib))
    duplicate=dict(case_ids=[a['m1_case_id'],b['m1_case_id']],labels=['M1-04','M1-09'],intersection=n,
        membership_iou=n/len(np.union1d(ia,ib)),smaller_set_coverage=n/min(len(ia),len(ib)),
        evidence='Same central-aisle cooler viewed from opposite directions, verified in photographs and common 3D coordinates.')
    summary.update(confirmed_case_count=10,unique_physical_target_count=9,duplicate_target=duplicate,
        excluded_archived_case=index['excluded'],review_type=audit['reviewer_type'],
        independent_human_instance_review=False,independent_human_masks=sum(t['annotation_status']=='valid' for t in views['views']),
        annotation_tasks=len(views['views']),retrained_models=False,
        study_limitations=['Confirmed successful M1 cohort, not an unselected benchmark.',
            'Ten confirmed M1 tasks correspond to nine distinct physical targets in one scene.',
            'Correct-instance labels are an AI visual audit with original user confirmations, not independent human adjudication.',
            'Stored descriptions for M1-07, M1-08 and M1-10 imperfectly describe the selected equipment; unchanged in both conditions.',
            'Held-out views were not input to instance selection but were available to original scene/language-field training and historical inspection.',
            'No independent held-out IoU is reported until reviewed target masks are supplied.'])
    views['annotation_count']=summary['independent_human_masks'];save(OUT/'evaluation_views.json',views)
    save(OUT/'metrics.json',dict(summary=summary,per_case=metrics,instance_selection=instances,heldout_views=heldout))
    save(OUT/'csv_tables.json',{'localization_metrics_10cases':metrics,'instance_selection_results':instances,'heldout_iou_results':heldout})
    for name,records in [('localization_metrics_10cases',metrics),('instance_selection_results',instances),('heldout_iou_results',heldout)]:
        csv_export(OUT/(name+'.csv'),records)
    # Verify input preservation and same native Case10 localization.
    for row in cases:
        assert sha(row['photo_indices_source'])==row['photo_indices_sha256']
        assert sha(OUT/'cases'/row['m1_case_id']/'photo_assisted_indices.npy')==row['photo_indices_sha256']
    last=cases[-1];native=ROOT/'baseline_comparison/gsle_native/case10/target_indices.npy'
    assert np.array_equal(np.load(native),np.load(OUT/'cases'/last['m1_case_id']/'text_only_indices.npy'))
    save(OUT/'validation.json',dict(ten_case_ids_unique=True,archived_11_excluded=True,source_photo_indices_unchanged=True,
        case10_text_only_matches_existing_native=True,heldout_cameras_exclude_reference=True,
        missing_iou_is_null=True,no_prediction_derived_ground_truth=True,gaussian_rendering_threshold=.05))
    instructions='''Independent held-out annotation task\n\nFor each task: image.png is the registered photograph at the exact saved camera resolution. target_identity_reference.jpg identifies the intended equipment in the ORIGINAL INPUT view; it is not a test mask.\nDraw only the visible pixels of that same physical target on image.png. Do not include foreground occluders and do not use either method prediction as a starting mask. Export a single-channel binary mask.png (0 background, 255 target), same resolution. Independent SAM masks may be manually corrected, but must not be seeded from method projections.\nCopy review_template.json to review.json. Supply reviewer, reviewed=true, visible=true/false, mask_origin=independent_manual or independent_sam_human_corrected. Leave derived_from_method_prediction=false only if true. If hidden/unidentifiable, visible=false and explain; do not label a guessed mask.\nDo not use input SAM or a projected method mask as ground truth. No annotation masks are included in this pending package.\nAfter annotation: run summarize_f08_ten.py and figure_f08_ten.py; they never run M2/M3/M4 or models.\n'''
    (OUT/'annotation_tasks/README.txt').write_text(instructions)
    with zipfile.ZipFile(OUT/'heldout_annotation_tasks.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in (OUT/'annotation_tasks').rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    cards=[]
    for row in cases:
        j=judgements[row['m1_case_id']];cards.append(f'<section><h2>{row["label"]} · {html.escape(row["description"])}</h2><p>M1 ID: {row["m1_case_id"]}<br>Text-only: {j["text_only_correct"]}; Photo-assisted: {j["photo_assisted_correct"]}</p><p>{html.escape(j["notes"])}</p><a href="cases/{row["m1_case_id"]}/review_sheet.jpg"><img loading="lazy" src="cases/{row["m1_case_id"]}/review_sheet.jpg"></a></section>')
    page='''<!doctype html><meta charset="utf-8"><title>F08 localization audit</title><style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:20px;background:#f4f5f7}section{background:white;padding:20px;margin:20px 0}img{max-width:100%}a{color:#1756a9}</style><h1>F08 · 十条 M1 案例定位核验</h1><p>纯文本 4/10；照片辅助 10/10。当前为基于原始照片、多视角投影和用户既有确认的 AI 核验，尚非独立人工复核。M1-04 / M1-09 是同一物理设备：去重为 3/9 与 9/9。独立测试 Mask 尚未提供，Held-out IoU 为 NA，未使用输入 SAM 代替。</p><p><a href="figure_F08_revised.png">高清 PNG</a> · <a href="figure_F08_revised.pdf">PDF</a> · <a href="heldout_annotation_tasks.zip">20 个独立视角待标注包</a> · <a href="localization_metrics_10cases.csv">逐案例 CSV</a> · <a href="caption.txt">Caption</a></p><img src="figure_F08_revised.png">'''+''.join(cards)
    (OUT/'index.html').write_text(page,encoding='utf-8')
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

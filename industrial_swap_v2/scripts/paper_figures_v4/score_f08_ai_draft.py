"""Score a supplied AI mask draft without relabeling it human ground truth.

Uses only saved predictions; no Gaussian re-rendering or model calls. Nonempty
unreviewed AI masks are scored as provisional agreement. Empty masks stay NA.
The official human-annotation scorer and previous F08 outputs are not altered.
"""
import argparse, copy, hashlib, io, json, shutil
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from summarize_f08_ten import csv_export

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F08_localization_revised'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iou(pred,mask):return float(np.logical_and(pred,mask).sum()/np.logical_or(pred,mask).sum())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=BASE/'with_8_masks');args=parser.parse_args()
    out=args.output.resolve();out.relative_to(BASE);out.mkdir(exist_ok=True)
    package=args.package.resolve();package_hash=sha(package)
    base_hashes={name:sha(BASE/name) for name in ['figure_F08_revised.png','figure_F08_revised.pdf','metrics.json','evaluation_views.json']}
    data=copy.deepcopy(read(BASE/'metrics.json'));tasks=read(BASE/'evaluation_views.json')['views'];byview=[];task_records=[]
    with zipfile.ZipFile(package) as z:
        assert len(z.namelist())==len(set(z.namelist())), 'Duplicate ZIP member'
        for task in tasks:
            prefix='annotation_tasks/'+task['task_id']+'/'
            review=json.loads(z.read(prefix+'review.json'))
            camera=json.loads(z.read(prefix+'camera.json'))
            assert review['task_id']==task['task_id']
            assert review['image_sha256']==task['image_sha256']
            assert hashlib.sha256(z.read(prefix+'image.png')).hexdigest()==task['image_sha256']
            assert camera['name']==task['camera']['name']!=task['input_camera']
            for key in ('R','t','width','height','fx','fy','cx','cy'):
                assert np.allclose(camera[key],task['camera'][key],rtol=0,atol=1e-7),key
            assert review['reviewer_type']=='ai' and review['reviewed'] is False
            assert review['derived_from_method_prediction'] is False
            raw=z.read(prefix+'mask.png');arr=np.asarray(Image.open(io.BytesIO(raw)))
            assert arr.ndim==2 and arr.shape==(camera['height'],camera['width'])
            assert set(np.unique(arr)).issubset({0,255})
            mask=arr>127;nonempty=bool(mask.any())
            if nonempty:
                assert review['visible'] is True and review['mask_origin']=='independent_ai_polygon_draft'
            status='ai_draft_scored_not_human_validated' if nonempty else 'empty_draft_excluded'
            ad=out/'supplied_annotations'/task['task_id'];ad.mkdir(parents=True,exist_ok=True)
            for name in ['mask.png','review.json','polygon.json','camera.json']:(ad/name).write_bytes(z.read(prefix+name))
            # Original submitted review metadata remains byte-for-byte unchanged.
            warning=('Suspected wrong physical instance: draft outlines central-aisle cooler; intended rack-side cooler is at the left image edge. Scored exactly as supplied, no exclusion.'
                if task['task_id']=='M1-06_frame_000202' else review.get('notes',''))
            task_records.append(dict(task_id=task['task_id'],m1_case_id=task['m1_case_id'],label=task['label'],camera=camera,
                status=status,draft_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(raw).hexdigest(),
                image_sha256=task['image_sha256'],review=review,warning=warning))
            for method,p in task['predicted_masks'].items():
                path=Path(p)
                if not path.is_file():path=ROOT/Path(p).relative_to('/mnt/sda/ubuntu/lkx/papers/paper_eli')
                pred=np.asarray(Image.open(path))>127;assert pred.shape==mask.shape
                score=iou(pred,mask) if nonempty else None
                byview.append(dict(m1_case_id=task['m1_case_id'],label=task['label'],task_id=task['task_id'],camera=camera['name'],method=method,
                    ai_draft_mask_iou=score,independent_human_heldout_iou=None,status=status,
                    intersection_pixels=int((pred&mask).sum()) if nonempty else None,
                    union_pixels=int((pred|mask).sum()) if nonempty else None,
                    draft_mask_pixels=int(mask.sum()),prediction_pixels=int(pred.sum()),
                    prediction_path=str(path),prediction_sha256=sha(path),draft_mask_sha256=hashlib.sha256(raw).hexdigest(),
                    human_reviewed=False,warning=warning))
        (out/'annotation_package_README.txt').write_bytes(z.read('README.txt'))
    for row in data['per_case']:
        vals={m:[v['ai_draft_mask_iou'] for v in byview if v['m1_case_id']==row['m1_case_id'] and v['method']==m and v['ai_draft_mask_iou'] is not None] for m in ['text_only','photo_assisted']}
        row['number_ai_draft_evaluation_views']=len(vals['text_only'])
        assert len(vals['text_only'])==len(vals['photo_assisted'])
        for method in vals:
            row[method+'_ai_draft_iou']=float(np.mean(vals[method])) if vals[method] else None
            # Formal independent-human metrics stay unchanged and unavailable.
            assert row[method+'_heldout_iou'] is None
    summary=data['summary'];summary.update(evaluation_mode='ai_draft',ai_draft_views=sum(x['draft_pixels']>0 for x in task_records),
        ai_draft_cases=sum(x['number_ai_draft_evaluation_views']>0 for x in data['per_case']),
        empty_draft_views=sum(x['draft_pixels']==0 for x in task_records),input_archive_sha256=package_hash,
        input_archive=str(package),independent_human_masks=0,
        mask_policy='User requested provisional scoring of supplied AI drafts; no masks corrected, excluded by IoU, or promoted to human ground truth.',
        flagged_task='M1-06_frame_000202')
    for method in ['text_only','photo_assisted']:
        vals=[r[method+'_ai_draft_iou'] for r in data['per_case'] if r[method+'_ai_draft_iou'] is not None]
        summary[method]['ai_draft_mean_iou']=float(np.mean(vals));summary[method]['ai_draft_evaluated_cases']=len(vals)
    data['ai_draft_view_results']=byview
    save(out/'metrics.json',data);save(out/'summary.json',summary);save(out/'evaluation_views_ai_draft.json',task_records)
    percase=[{k:v for k,v in r.items() if k in ['m1_case_id','label','target_description','text_only_ai_draft_iou','photo_assisted_ai_draft_iou','number_ai_draft_evaluation_views','text_only_heldout_iou','photo_assisted_heldout_iou']} for r in data['per_case']]
    csv_export(out/'ai_draft_iou_per_case.csv',percase);csv_export(out/'ai_draft_iou_per_view.csv',byview)
    for name in ['case_index.json','instance_audit.json']:shutil.copy2(BASE/name,out/name)
    sources=read(BASE/'sources.json');sources.update(ai_draft_archive=dict(path=str(package),sha256=package_hash),
        existing_figure_directory=str(BASE),draft_annotation_review='All supplied reviewed=false / reviewer_type=ai metadata preserved.',
        frozen_existing_artifacts=base_hashes,no_models_or_rerenders=True,methods_share_exact_test_cameras=True)
    save(out/'sources.json',sources)
    assert base_hashes=={name:sha(BASE/name) for name in base_hashes}
    save(out/'validation.json',dict(source_images_and_cameras_match=True,original_figure_and_metrics_unchanged=True,
        nonempty_masks=summary['ai_draft_views'],empty_masks_excluded=summary['empty_draft_views'],human_review_records_modified=False,
        formal_independent_iou_remains_null=True,mean_rule='per-view IoU, then mean within each target, then across the four scored targets'))
    for r in percase:
        if r['number_ai_draft_evaluation_views']:print(json.dumps(r),flush=True)
    print('Macro means',summary['text_only']['ai_draft_mean_iou'],summary['photo_assisted']['ai_draft_mean_iou'],flush=True)

if __name__=='__main__':main()

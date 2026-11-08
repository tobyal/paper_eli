"""CPU-only integrity audit and preview packaging of ten pre-generation cases."""
import argparse
import itertools
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
from PIL import Image
from ..core.context import Context,read_json,write_json,sha256
from ..core.scene import read_vertices
from ..core.workflow import validate_workflow


def exact_subset(vertices,indices,output):
    saved=read_vertices(output)
    assert len(saved)==len(indices)
    for start in range(0,len(indices),65536):
        assert saved[start:start+65536].tobytes()==vertices[indices[start:start+65536]].tobytes(),output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args();batch=args.batch.resolve()
    cases=sorted(batch.glob('case_*/case.json'))
    assert len(cases)==10,('Expected ten completed preparations',len(cases))
    first=Context(cases[0].parent/'config.yaml');batch.relative_to(first.base)
    visual_notes=read_json(batch/'visual_review_notes.json')['cases'] if (batch/'visual_review_notes.json').exists() else {}
    source=first.source(first.cfg['project']['source_ply']);source_hash=sha256(source)
    assert source_hash==first.cfg['project']['source_sha256']
    vertices=read_vertices(source);sets={};rows=[];checks=[];files=[]
    prohibited=('edited_scene.png','object_rgba.png','object_mask.png','generation.json',
        'api_request.json','api_response.json','qwen_generation_once.lock')
    for path in cases:
        folder=path.parent;case=read_json(path);ctx=Context(folder/'config.yaml');validate_workflow(ctx.cfg)
        target=read_json(folder/'localization/target.json')
        assert target['fresh_semantic_query'] and not target['manual_roi_used']
        assert not case['qwen_calls'] and not case['trellis_calls'] and not case['m3_executed'] and not case['m4_executed']
        assert not (folder/'m3').exists() and not (folder/'m4').exists()
        assert not any(p.name in prohibited or p.name.startswith('qwen_attempt_') for p in folder.rglob('*'))
        ids=np.load(folder/'m1/target_indices.npy');keep=np.load(folder/'m1/keep_indices.npy')
        assert len(ids)==case['target_count']
        assert np.array_equal(np.sort(np.concatenate((ids,keep))),np.arange(len(vertices)))
        assert sha256(folder/'m1/target_indices.npy')==case['target_indices_sha256']
        assert case['target_indices_sha256']==sha256(batch/case['candidate']/'target_indices.npy')
        exact_subset(vertices,ids,folder/'m1/target_gaussians.ply')
        exact_subset(vertices,keep,folder/'m1/background_gaussians.ply')
        sets[case['case_id']]=ids
        camera=read_json(folder/'m2/anchor_camera.json');size=(camera['width'],camera['height'])
        plan=read_json(folder/'m2/generation_plan.json')
        for name in ('anchor.png','position_hint.png','edit_mask.png','depth.png','depth_background.png'):
            assert Image.open(folder/'m2'/name).size==size
        for name in ('old_device_mask.png','localization_overlay.png','background_render.png'):
            assert Image.open(folder/'m1'/name).size==size
        assert all(Path(p).is_file() and Path(p).parent==folder/'m2' for p in plan['images'])
        assert len(plan['images'])==3 and target['text'] in plan['prompt'] and case['replacement'] in plan['prompt']
        edit=np.asarray(Image.open(folder/'m2/edit_mask.png'))>127
        depth=np.asarray(Image.open(folder/'m2/depth_background.png'))
        assert (depth[edit]==0).all(), 'Unknown editing depth must be zero'
        assert read_json(folder/'records/m2_status.json')['status']=='conditions_prepared_api_not_called'
        # Keep a real photograph distinct from the synthesized Anchor render.
        photo=ctx.root/'data/industrial/images'/target['reference_camera']['name']
        shutil.copy2(photo,folder/'localization/source_photo.jpg')
        refmask=np.asarray(Image.open(folder/'localization/reference_mask.png'))>127
        ref=Image.open(photo).convert('RGB').resize((refmask.shape[1],refmask.shape[0]),Image.Resampling.LANCZOS)
        overlay=np.asarray(ref).copy();overlay[refmask]=.4*overlay[refmask]+.6*np.array([255,30,30])
        Image.fromarray(overlay).save(folder/'localization/source_photo_overlay.jpg',quality=94)
        thresholds=read_json(folder/'avp_settings.json') if (folder/'avp_settings.json').exists() else read_json(ctx.root/'gsle/config.json')['avp']
        note=visual_notes.get(case['case_id'],case['review_note']+' 删除后背景残留需人工检查；本次未作背景修复。')
        row={'case_id':case['case_id'],'original_device':case['name_zh'],'m1_english_semantic':target['text'],
            'target_gaussian_count':len(ids),'reference_camera':target['reference_camera']['name'],
            'anchor_view':camera['name'],'anchor_visibility':case['anchor']['visibility'],
            'reference_sam_iou':case['quality']['reference_iou'],'projection_spill':case['quality']['projection_spill'],
            'avp_min_visibility':thresholds['min_visibility'],
            'replacement_prompt':case['replacement'],'quality_check':'prepared_pending_human_review',
            'quality_notes':note,'output_path':str(folder),'config_path':str(folder/'config.yaml'),
            'review_path':str(folder/'review_sheet.jpg'),'qwen_prompt_path':str(folder/'m2/generation_prompt.txt')}
        rows.append(row)
        checks.append({'case_id':case['case_id'],'partition_complete':True,'target_bytes_equal_source':True,
            'background_bytes_equal_source':True,'conditions_aligned':True,'qwen_not_invoked':True,
            'fresh_semantic_query':True,'source_photo':str(photo),'source_photo_sha256':sha256(photo),
            'target_indices_sha256':case['target_indices_sha256']})
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.yaml','.txt','.png','.jpg') and 'avp' not in p.relative_to(folder).parts:
                files.append(p)
        print('AUDITED',case['case_id'],len(ids),'lossless source partition',flush=True)
    overlaps=[]
    for a,b in itertools.combinations(sets,2):
        intersection=len(np.intersect1d(sets[a],sets[b],assume_unique=True))
        iou=intersection/(len(sets[a])+len(sets[b])-intersection)
        overlap=intersection/min(len(sets[a]),len(sets[b]))
        overlaps.append({'a':a,'b':b,'intersection':intersection,'iou':iou,'overlap_smaller':overlap})
        assert overlap<.2,('Possible duplicated physical device',a,b,overlap)
    assert len({x['target_indices_sha256'] for x in checks})==10
    provenance={}
    for directory in sorted(batch.glob('survey*')):
        if not directory.is_dir():continue
        provenance[directory.name]=read_json(directory/'semantic_provenance.json')
        files.extend((directory/'semantic_provenance.json',directory/'queries.json'))
    audit={'status':'preparation_integrity_passed_human_review_pending','source_sha256':source_hash,
        'source_gaussian_count':len(vertices),'cases':checks,'pairwise_target_overlap':overlaps,
        'source_checkpoint_order_verified':all(all(level['source_parameters_identical'] for level in p['levels']) for p in provenance.values()),
        'semantic_provenance':provenance,'qwen_calls':0,'trellis_calls':0,'m3_executed':False,'m4_executed':False,
        'limitations':['Reference IoU is against an automatic SAM proposal, not manually annotated ground truth.',
            'Localization fills the existing semantic seed bounding box, preserving floor-semantic points; some remnants or neighbor points may remain.',
            'Black area in the Qwen background-depth condition denotes unknown depth, not completed background.',
            'All ten cases require human acceptance before image generation.']}
    write_json(batch/'preparation_audit.json',audit);write_json(batch/'cases_summary_rows.json',rows)
    files.extend((batch/'preparation_audit.json',batch/'cases_summary_rows.json',batch/'interface_tests.log'))
    with tarfile.open(batch/'preview_inputs.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(batch)))
    print('10 cases validated; max pairwise target overlap',max(x['overlap_smaller'] for x in overlaps),flush=True)


if __name__=='__main__':main()

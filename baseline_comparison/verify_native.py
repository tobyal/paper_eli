"""Lightweight execution/provenance verification of completed native baselines."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
DGE_IMAGES=['original_view','localization_2d_mask','localization_2d_overlay','localization_multiview',
 'localization_3d_mask','localization_selected_gaussians','localization_full_scene','original_multiview',
 'edited_2d_views','edited_2d_comparison','gaussian_optimization_initial','gaussian_optimization_middle',
 'scene_final','scene_final_crop','final_multiview','final_comparison']
GSLE_IMAGES=['original_view','language_query_heatmap','localization_candidates','localization_selected_mask',
 'localization_selected_gaussians','localization_3d_box','scene_without_target','avp_candidates','avp_scores',
 'avp_selected_anchor','avp_anchor_with_bbox','avp_depth','avp_inpainting_mask','inpainting_input','inpainting_mask',
 'inpainting_depth','inpainting_result','foreground_mask','foreground_rgba','foreground_white',
 'coarse_initial','coarse_middle','coarse_final','coarse_multiview','refine_initial','refine_middle','refine_final',
 'refine_multiview','scene_final','scene_final_crop','final_comparison','final_multiview','coarse_refine_comparison']


def main():
 p=argparse.ArgumentParser();p.add_argument('method',choices=['dge','gsle']);p.add_argument('case');a=p.parse_args()
 out=BASE/(a.method+'_native')/a.case;status=json.loads((out/'records/status.json').read_text());assert status['status']=='complete'
 cfg=json.loads((out/'config.json').read_text());prompt=json.loads((out/'prompts.json').read_text())
 original=next(x for x in json.loads((BASE/'prompts.json').read_text())['cases'] if x['case']==a.case)
 assert all(prompt[k]==original[k] for k in ['A','B'])
 for key in ['target_record','instance_geometry','reference_sam_mask','reference_image','reference_sam_ids','ours_anchor','ours_scene']:
  assert key not in cfg and key not in prompt,key
 images=DGE_IMAGES if a.method=='dge' else GSLE_IMAGES
 for name in images:
  with Image.open(out/(name+'.png')) as im:im.verify()
 indices=np.load(out/'target_indices.npy');assert len(indices)>0 and len(indices)==len(np.unique(indices))
 vertex=PlyData.read(str(out/'final_scene.ply'))['vertex'].data
 for name in vertex.dtype.names:assert np.isfinite(vertex[name]).all(),name
 result=dict(status='passed',required_images=len(images),target_gaussians=len(indices),final_gaussians=len(vertex),
   exact_A_B_preserved=True,no_ours_spatial_inputs_in_config=True,final_ply=str(out/'final_scene.ply'))
 if a.method=='dge':
  commit=cfg['upstream_commit'];repo=ROOT/'repos/DGE';code=BASE/'vendor/DGE_native'
  for name in ['threestudio/utils/sam.py','threestudio/models/guidance/dge_guidance.py','threestudio/data/gs_load.py','configs/dge.yaml']:
   assert subprocess.check_output(['git','show',commit+':'+name],cwd=repo)==(code/name).read_bytes(),name
  loc=json.loads((out/'records/localization.json').read_text());final=json.loads((out/'records/final.json').read_text())
  assert loc['threshold']==.8 and loc['total_cameras']==350 and len(loc['sampled_cameras'])==20
  assert final['iterations']==1000
  assert json.loads((out/'records/middle_iteration.json').read_text())['completed_optimizer_steps']==500
  assert len(list((out/'edited/step_0000').glob('[0-9]*.png')))==20
  assert len(list((out/'edited/step_0500').glob('[0-9]*.png')))==20
  result.update(upstream_core_byte_identical=True,iterations=1000,native_threshold=.8,middle_actual_step=500)
 else:
  query=json.loads((out/'localization/query.json').read_text());assert query['reference_image'] is None and query['prototype'] is None
  assert len(query['levels'])==3 and all(x['source_parameters_identical'] for x in query['levels'])
  target=json.loads((out/'localization/target.json').read_text());region=json.loads((out/'localization/edit_region.json').read_text())
  assert region['bbox_corners']==target['bbox_corners']
  for stage,steps,batch in [('coarse',6500,6),('refine',2000,4)]:
   effective=json.loads((out/stage/'effective_config.json').read_text());assert effective['iterations']==steps and effective['effective_batch']==batch
   last=json.loads((out/stage/'training.jsonl').read_text().splitlines()[-1]);assert last['step']==steps
  composition=json.loads((out/'final/composition.json').read_text());assert composition['retained_background_bitwise_identical']
  result.update(independent_text_localization=True,generation_bbox_equals_localization=True,coarse_steps=6500,refine_steps=2000,
      retained_background_bitwise_identical=True,avp_scoring=json.loads((out/'avp/anchor.json').read_text())['scoring'])
 (out/'records/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

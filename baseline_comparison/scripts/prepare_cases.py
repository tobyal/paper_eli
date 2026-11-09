"""Record shared instances/prompts and immutable inputs for 15 baseline runs."""
import json, hashlib, shutil
from pathlib import Path
import numpy as np
import yaml

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE=ROOT/'baseline_comparison'
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=read(BASE/'prompts.json')
source=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
spec.update(source_scene=str(source),source_scene_sha256=sha(source),colmap_source=str(ROOT/'data/edit_train'),
    gsle_implementation='Existing paper-derived reproduction, not an official public GSLE implementation',seed=42)
for row in spec['cases']:
    key=row['case']
    ours=ROOT/'industrial_swap_v2/outputs'/({'case00':'registration_all_cases_v1/case00','case02':'case02_raster_alignment_v1'}.get(key,'qwen_reselected_04_07_10_m4_2000_v1/'+key))
    target=read(ours/'m1/target.json');summary=read(ours/'m4/summary.json')
    assert target['text']==row['A']
    row.update(ours_source=str(ours),target_record=str((ours/'m1/target.json').resolve()),
        reference_image=target['reference_camera']['name'],reference_camera=target['reference_camera'],
        reference_sam_level=target['selected_level'],reference_sam_ids=target.get('selected_sam_ids',[target.get('reference_proposal',{}).get('feature_index')]),
        ours_scene=summary['candidate_scene'],ours_guard_passed=summary['candidate_guard_passed'],
        ours_anchor=str((ours/'m2/anchor_camera.json').resolve()))
    if key=='case10':
        refined=ROOT/'industrial_swap_v2/outputs/qwen_reselected_04_07_10_final_v3/case10'
        row['ours_scene']=read(refined/'m4/summary.json')['candidate_scene']
    if key=='case07':row['ours_scene']=str(ROOT/'industrial_swap_v2/outputs/case07_yaw_adjust_v1/scene_rotated_step2000.ply')
    cfg=yaml.safe_load((ours/'config.yaml').read_text())
    row['ours_saved_B']=cfg['m2']['replacement_text']
    row['ours_B_exact_match']=row['B']==row['ours_saved_B']
    shared=BASE/'inputs'/key;shared.mkdir(parents=True,exist_ok=True)
    low=np.asarray(target['bbox_low']);high=np.asarray(target['bbox_high'])
    write(shared/'target_geometry.json',{'center':((low+high)/2).tolist(),'half_extents':((high-low)/2).tolist(),'basis':np.eye(3).tolist(),
        'usage':'Rank native semantic candidates to select the confirmed physical instance; never substitute a baseline mask.'})
    row['instance_geometry']=str(shared/'target_geometry.json')
    # Preserve the user-confirmed SAM reference for GSLE instance association.
    sam=np.load(ROOT/'data/gsle_langsplat/language_features'/(Path(row['reference_image']).stem+'_s.npy'))
    mask=np.isin(sam[row['reference_sam_level']],row['reference_sam_ids']);np.save(shared/'reference_sam_mask.npy',mask)
    row['reference_sam_mask']=str(shared/'reference_sam_mask.npy')
    row['background_text']='An industrial factory interior with continuous concrete floor, existing yellow floor markings, surrounding equipment, metal carts and storage containers, matching the original perspective and lighting, no object in the removed area.'
    short={'case00':'a yellow six-axis industrial robotic arm on a square steel base with a gripper',
        'case02':'a stainless-steel roller conveyor with blue side rails and four square legs',
        'case04':'a compact white and light-gray industrial chiller with a round dark grille and blue control panel',
        'case07':'a white and gray automated parts cabinet with blue drawers and a central dispensing hatch',
        'case10':'a slim light-gray industrial tool cabinet with blue drawers and caster wheels'}[key]
    row['gaussianeditor_refine_text']='Make it '+short+'.'
    for method in spec['methods']:
        folder=BASE/method/key;folder.mkdir(parents=True,exist_ok=True)
        (folder/'logs').mkdir(exist_ok=True);(folder/'records').mkdir(exist_ok=True)
        write(folder/'task.json',row)
    cfg=read(ROOT/'gsle/config.json');cfg.update(target_text=row['A'],replacement_text=row['B'],reference_image=row['reference_image'],
        reference_sam_level=row['reference_sam_level'],reference_sam_ids=row['reference_sam_ids'],reference_sam_mask=row['reference_sam_mask'])
    cfg['inpainting']['negative_text']='multiple separate replacement objects, disassembled machinery, floating object, disconnected parts'
    cfg['foreground_prompt_mode']='generic_box_clip'
    write(BASE/'gsle'/key/'config.json',cfg)
write(BASE/'prompts.json',spec)
print('Prepared',len(spec['cases']),'cases; 15 independent experiment directories')

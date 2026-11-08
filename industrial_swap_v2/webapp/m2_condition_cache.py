"""Reuse only matching geometry, replacement text and versioned spatial conditions."""
from pathlib import Path
import shutil
import yaml
from . import store as s, service as v
from ..core.context import read_json,write_json
from ..core.m2_space import VERSION, signature

CONDITION_FILES=('anchor.png','position_hint.png','depth.png','depth_background.png',
                 'edit_mask.png','depth.npz','anchor_camera.json','conditions.json',
                 'generation_region.json','mask_report.json','spatial_depth.npz',
                 'old_visible_mask.png','old_clean_mask.png','extension_proposal.png',
                 'protected_foreground_mask.png','generation_bbox_mask.png')
M1_FILES=('edit_region.json','target.json','target_indices.npy','target_gaussians.ply','background_gaussians.ply')


def settings(cfg):
    return {'scene':cfg['project']['source_sha256'],'seed':cfg['project']['seed'],
            'spatial_signature':signature(cfg),
            'm2':{k:val for k,val in cfg['m2'].items() if k not in ('prompt_override','api_key_file')},
            'render':cfg['render']}


def reuse_conditions(cfg,out,m1):
    case_id=cfg['m2'].get('confirmed_m1_case_id')
    if not case_id or out.exists():return None
    candidates=s.rows('''SELECT r.* FROM stage_runs r JOIN case_m2_preparations p ON p.run_id=r.id
        WHERE p.case_id=:case AND r.status='completed' AND r.action='prepare' ORDER BY r.created_at''',case=case_id)
    if not candidates:return None
    hashes={name:s.sha(m1/name) for name in M1_FILES}
    for row in candidates:
        previous=yaml.safe_load(Path(row['config_path']).read_text())
        if settings(previous)!=settings(cfg) or not v.verify_outputs(row):continue
        source=v.stage_dir(row['id']);conditions=read_json(source/'conditions.json')
        if conditions.get('spatial_method')!=VERSION or conditions.get('condition_signature')!=signature(cfg):continue
        if (conditions.get('m1_case_id')!=case_id or conditions.get('input_sha256')!=hashes or
            conditions.get('source_scene_sha256')!=cfg['project']['source_sha256']):continue
        if not all((source/n).is_file() for n in CONDITION_FILES):continue
        if s.sha(s.ROOT/cfg['project']['source_ply'])!=cfg['project']['source_sha256']:
            raise ValueError('Original scene SHA256 changed before condition reuse')
        out.mkdir(parents=True)
        for name in CONDITION_FILES:
            if name!='conditions.json':shutil.copy2(source/name,out/name)
        if (source/'avp_work').is_dir():shutil.copytree(source/'avp_work',out/'avp_work')
        anchor=Path(conditions['source_anchor'])
        if anchor.is_relative_to(source):conditions['source_anchor']=str(out/anchor.relative_to(source))
        conditions['condition_cache']={'source_run':row['id'],'same_confirmed_m1_case':True,
            'reason':'Identical replacement text, spatial algorithm/settings, source geometry and camera policy',
            'copied_image_sha256':{n:s.sha(out/n) for n in CONDITION_FILES if n!='conditions.json'}}
        write_json(out/'conditions.json',conditions)
        return row['id']
    return None

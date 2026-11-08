"""Official workflow settings and explicit migration of NEW experiment configs."""
import copy
from pathlib import Path
import yaml

METHOD='dual_mask_local_background'
SCHEME='old_new_union_v2'


def supervision_settings(cfg):
    """M4 owns supervision; the M3 key is read-only compatibility for old configs."""
    return cfg['m4'].get('multiview_supervision',cfg['m3'].get('visibility_initialization',{}))


def validate_workflow(cfg,stage='m4'):
    if cfg['m3'].get('method')=='pbr_multiview_gaussian':
        pbr=cfg['m3'].get('pbr',{})
        if not pbr.get('hdri') or not pbr.get('blender'):
            raise ValueError('PBR M3 requires configured HDRI and Blender')
        train=pbr.get('training',{})
        if train.get('sh_degree')!=3 or train.get('iterations',0)<1:
            raise ValueError('PBR training requires SH3 and positive iterations')
    if cfg['m3'].get('object_occlusion_filter',{}).get('enabled',False):
        raise ValueError('Deletion filtering is archived; create a new official experiment')
    if stage=='m3': return
    masks=supervision_settings(cfg)
    if not masks.get('enabled') or masks.get('scheme')!=SCHEME:
        raise ValueError('Official M4 requires per-view old_new_union_v2 masks')
    local=cfg['m4'].get('local_background',{})
    if cfg['m4'].get('method')!=METHOD or not local.get('enabled'):
        raise ValueError('Historical M4 config is read-only; use a new official experiment or --source-config')
    if type(local.get('iterations')) is not int or not 200<=local['iterations']<=10000:
        raise ValueError('M4 supports 200–10000 local background steps')
    save=cfg['m4'].get('save_every',0)
    if type(save) is not int or save<0 or (save and save%local.get('evaluate_every',50)):
        raise ValueError('save_every must be zero or a positive multiple of evaluate_every')


def current_config(source):
    """Migrate settings for a NEW run; never rewrite stored configs or results."""
    defaults=yaml.safe_load((Path(__file__).parents[1]/'configs/default.yaml').read_text(encoding='utf-8'))
    cfg=copy.deepcopy(source)
    cfg.setdefault('m2',{}).setdefault('spatial_conditions',copy.deepcopy(defaults['m2']['spatial_conditions']))
    previous=supervision_settings(cfg)
    union=previous.get('scheme')==SCHEME
    for key in ('object_occlusion_filter','output_type','use_original_scene_transform','visibility_initialization'):
        cfg['m3'].pop(key,None)
    masks=copy.deepcopy(defaults['m4']['multiview_supervision'])
    if union:
        masks.update(previous); masks.pop('edge_dilation_fraction',None)
    masks.update(enabled=True,scheme=SCHEME)
    old=cfg['m4']; cfg['m4']=copy.deepcopy(defaults['m4'])
    if 'integration' in old: cfg['m4']['integration']=copy.deepcopy(old['integration'])
    cfg['m4']['multiview_supervision']=masks
    if old.get('method')==METHOD or (union and old.get('background_only')):
        local=copy.deepcopy(old.get('local_background',{}))
        for key in ('outside_preserve_weight','candidate_max_opacity_change'): local.pop(key,None)
        cfg['m4']['local_background'].update(local)
        cfg['m4']['minimum_free_gpu_gib']=old.get('minimum_free_gpu_gib',cfg['m4']['minimum_free_gpu_gib'])
        if 'save_every' in old: cfg['m4']['save_every']=old['save_every']
    validate_workflow(cfg)
    return cfg

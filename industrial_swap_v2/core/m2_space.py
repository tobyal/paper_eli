"""CPU geometry and explicit structural priors for M2 replacement space.

English text selects a documented shape prior, not a metric size prediction.
Unknown descriptions conservatively retain the measured old geometry.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from .context import read_json, sha256

VERSION = 'adaptive_visible_v1'
DEFAULTS = dict(method=VERSION, geometry_quantile=.01, covariance_support_sigma=1.,
    covariance_cap_fraction=.04, space_padding_fraction=.025,
    floor_snap_fraction=.12, max_height_ratio=1.65, min_height_ratio=.75,
    old_alpha_threshold=.02, minimum_visible_fraction=.08,
    component_area_fraction=.002, hole_area_fraction=.06, edge_fraction=.015,
    depth_margin_fraction=.02, depth_std_fraction=.12, background_alpha_threshold=.8,
    depth_cdf_bins=33, max_front_contribution=.15, protected_contribution=.5)

# (phrases identifying the main device, width/depth factors, height / footprint).
PROFILES = {
    'articulated_arm': (('robot arm', 'robotic arm', 'articulated robot'), (.85,.85,1.25)),
    'scara': (('scara',), (.95,.95,.8)),
    'enclosed_machine': (('cnc', 'machining center', 'machining centre', 'enclosed machine'), (.98,.95,1.)),
    'cabinet': (('control cabinet', 'electrical cabinet', 'equipment cabinet', 'cabinet', 'locker'), (.92,.92,1.6)),
    'gantry_station': (('inspection station', 'screwdriving station', 'gantry', 'press', 'assembly station'), (.98,.95,1.1)),
    'conveyor': (('conveyor', 'roller transfer',), (1.,.9,.45)),
    'cylindrical_unit': (('dust collector', 'air compressor', 'receiver tank', 'cylindrical tank'), (.85,.85,1.5)),
    'cart_table': (('workbench', 'worktable', 'trolley', 'cart', 'mobile platform'), (.95,.95,.7)),
}


def settings(cfg):
    result = {**DEFAULTS, **cfg.get('m2', {}).get('spatial_conditions', {})}
    if result['method'] != VERSION:
        raise ValueError('Unknown M2 spatial conditions method')
    if not 0 <= result['geometry_quantile'] < .1:
        raise ValueError('geometry_quantile must be in [0, 0.1)')
    return result


def signature(cfg):
    m2 = cfg['m2']
    data = {'algorithm': VERSION, 'settings': settings(cfg),
            'replacement_text': m2['replacement_text'].strip(),
            'source_sha256': cfg['project']['source_sha256'],
            'anchor': {k:m2.get(k) for k in ('reuse_anchor','existing_anchor','anchor_resolution',
                       'avp_observed_fallback','avp_review_fallback')}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def semantic_profile(description):
    # Attributes (e.g. a lower cabinet under a gantry) cannot override the head device.
    head = re.split(r'\b(?:with|featuring|equipped)\b|[.;]', description.lower(), maxsplit=1)[0]
    matches = []
    for name, (phrases, factors) in PROFILES.items():
        for phrase in phrases:
            match = re.search(r'\b' + re.escape(phrase) + r'\b', head)
            if match:
                matches.append((match.start(), -len(phrase), name, phrase, factors))
    if not matches:
        return {'name':'geometry_fallback', 'matched_phrase':None, 'factors':[1.,1.,None],
                'confidence':'unknown', 'basis':'No recognized head device; retain measured dimensions'}
    _, _, name, phrase, factors = min(matches)
    return {'name':name, 'matched_phrase':phrase, 'factors':list(factors),
            'confidence':'rule_based_prior', 'basis':'Explicit structural proportion prior; not metric size inference'}


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = (np.cumsum(w) - .5*w) / w.sum()
    return float(np.interp(q, cumulative, v))


def corners(low, high):
    return np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])


def build_generation_region(vertices, target, description, cfg):
    from scipy.spatial.transform import Rotation
    xyz = np.column_stack([vertices[k] for k in ('x','y','z')]).astype(float)
    opacity = 1 / (1 + np.exp(-np.clip(vertices['opacity'], -30, 30)))
    scales = np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(float))
    quat = np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).astype(float)
    valid = (np.isfinite(xyz).all(1) & np.isfinite(scales).all(1) & np.isfinite(quat).all(1)
             & (np.linalg.norm(quat,axis=1)>1e-8) & (opacity>.01))
    xyz, opacity, scales, quat = xyz[valid], opacity[valid], scales[valid], quat[valid]
    if len(xyz)<32:
        raise ValueError('Not enough valid selected target Gaussians to estimate replacement space')
    up = np.asarray(target['up'],float); up /= np.linalg.norm(up)
    origin = np.average(xyz, axis=0, weights=opacity)
    reference = target['reference_camera']
    toward = -np.asarray(reference['R']).T @ np.asarray(reference['t']) - origin
    front = toward - up * np.dot(toward,up)
    if np.linalg.norm(front)<1e-6:
        raise ValueError('Reference camera cannot define a horizontal device frame')
    front /= np.linalg.norm(front)
    right = np.cross(up,front); right /= np.linalg.norm(right)
    axes = np.stack((right,up,front),axis=1)
    local = (xyz-origin) @ axes
    q = cfg['geometry_quantile']
    center_low = np.array([weighted_quantile(local[:,i],opacity,q) for i in range(3)])
    center_high = np.array([weighted_quantile(local[:,i],opacity,1-q) for i in range(3)])
    size = max(center_high-center_low)
    if size<=1e-8:
        raise ValueError('Degenerate target geometry')
    rotations = Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
    # Sigma along the generation frame axes, including anisotropic covariance.
    alignment = np.einsum('nji,jk->nik',rotations,axes)
    support = np.sqrt(np.sum(alignment**2 * scales[:,:,None]**2,axis=1))
    support = np.minimum(support*cfg['covariance_support_sigma'],size*cfg['covariance_cap_fraction'])
    low = np.array([weighted_quantile(local[:,i]-support[:,i],opacity,q) for i in range(3)])
    high = np.array([weighted_quantile(local[:,i]+support[:,i],opacity,1-q) for i in range(3)])
    old_sides = np.maximum(high-low,size*.03)
    middle = (low+high)/2
    contact = float(low[1]); snap = {'source':'observed_target_bottom','snapped':False}
    plane = target.get('up_estimation',{})
    if 'normal' in plane and 'offset' in plane:
        normal = np.asarray(plane['normal'],float)
        dot = float(normal@up)
        if dot>.98:
            horizontal = origin + axes @ np.array([middle[0],0,middle[2]])
            floor_y = float((-plane['offset']-normal@horizontal)/dot)
            gap = contact-floor_y
            snap.update(floor_local_y=floor_y,observed_bottom_gap=gap)
            if abs(gap)<=cfg['floor_snap_fraction']*max(old_sides):
                contact=floor_y; snap.update(source='existing_m1_floor_plane',snapped=True)
    profile = semantic_profile(description)
    fw, fd, ratio = profile['factors']
    new_sides = old_sides.copy()
    new_sides[[0,2]] *= [fw,fd]
    if ratio is not None:
        new_sides[1] = np.clip(ratio*max(new_sides[[0,2]]),
            old_sides[1]*cfg['min_height_ratio'],old_sides[1]*cfg['max_height_ratio'])
    new_sides *= 1 + 2*cfg['space_padding_fraction']
    new_low = middle-new_sides/2
    new_low[1] = contact
    new_high = new_low+new_sides
    local_center = (new_low+new_high)/2
    center = origin+axes@local_center
    box = corners(new_low,new_high)@axes.T+origin
    frame = {'axes':axes.tolist(),'center':center.tolist(),'scale':1.,
             'local_bbox_low':(-new_sides/2).tolist(),'local_bbox_high':(new_sides/2).tolist(),
             'convention':'world = local @ axes.T + center; x right, y up, z toward M1 reference'}
    return {'schema_version':VERSION,'text':target['text'],'replacement_text':description,
        'bbox_corners':box.tolist(),'generation_bbox':box.tolist(),'center':center.tolist(),
        'up':up.tolist(),'reference_camera':reference,'object_scale':float(max(new_sides)),
        'world_transform':frame,'semantic_profile':profile,'contact':snap,
        'geometry':{'valid_gaussians':int(len(xyz)),'origin':origin.tolist(),'axes':axes.tolist(),
            'old_local_low':low.tolist(),'old_local_high':high.tolist(),
            'old_side_lengths':old_sides.tolist(),'new_local_low':new_low.tolist(),
            'new_local_high':new_high.tolist(),'new_side_lengths':new_sides.tolist(),
            'bounds':'opacity-weighted quantiles of centers +/- capped directional covariance support'},
        'settings':cfg,'generation_region_rule':'Measured target geometry + structural prior; preserve contact; bounded dimension change'}


def generation_frame(ctx):
    """M3 consumes the exact M2 frame; old experiments keep their historical frame."""
    directory = ctx.stage('m2')
    metadata = read_json(directory/'conditions.json') if (directory/'conditions.json').exists() else {}
    if metadata.get('spatial_method') == VERSION:
        path = directory/'generation_region.json'
        if not path.is_file() or sha256(path)!=metadata['generation_region_sha256']:
            raise ValueError('M2 generation space missing or changed; prepare a new M2 version')
        return read_json(path)['world_transform']
    return read_json(ctx.stage('m1')/'manifest.json')['world_transform']

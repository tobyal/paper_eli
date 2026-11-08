"""Validated continuation of an immutable actual-iteration M4 snapshot."""
import hashlib
import json
from pathlib import Path
import torch
import yaml
from .context import read_json, sha256


def training_signature(config, visibility_signature):
    return hashlib.sha256(json.dumps({'config':config,'visibility':visibility_signature},sort_keys=True).encode()).hexdigest()


def load_resume(ctx, config, manifest):
    spec=ctx.cfg['m4'].get('resume')
    if not spec: return None
    checkpoint=ctx.path(spec['checkpoint']); source_config=ctx.path(spec['source_config'])
    if sha256(checkpoint)!=spec['checkpoint_sha256'] or sha256(source_config)!=spec['source_config_sha256']:
        raise ValueError('Resume source changed since experiment initialization')
    receipt=read_json(checkpoint.parent/'snapshot.json')
    if receipt['files_sha256'].get(checkpoint.name)!=spec['checkpoint_sha256']:
        raise ValueError('Resume checkpoint does not match immutable snapshot')
    if receipt['object_sha256']!=manifest['source_hashes']['object'] or receipt['source_background_sha256']!=manifest['source_hashes']['background']:
        raise ValueError('Resume snapshot uses a different background or device')
    previous=yaml.safe_load(source_config.read_text(encoding='utf-8'))
    from .local_background_optimizer import optimizer_config
    old=optimizer_config(previous['m4']['local_background'])
    if {k:v for k,v in old.items() if k!='iterations'}!={k:v for k,v in config.items() if k!='iterations'}:
        raise ValueError('Resume may change total iterations and save interval, not optimization settings')
    if previous['project']['seed']!=ctx.cfg['project']['seed']:
        raise ValueError('Resume seed changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    required={'state','optimizer','step','signature','rng','remaining_view_order','torch_rng','cuda_rng','best_state','best_step'}
    if not required.issubset(saved): raise ValueError('Use a full iteration snapshot with optimizer and sampling states')
    if saved['signature']!=training_signature(old,manifest['signature']):
        raise ValueError('Resume supervision or source configuration mismatch')
    if saved['step']!=receipt['step'] or not 0<saved['step']<config['iterations']:
        raise ValueError('Total iterations must exceed the saved actual iteration')
    return saved


def equal_state(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equal_state(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):
        return isinstance(right,type(left)) and len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def restore_training_state(saved,model,optimizer,rng,training_view_count):
    # Check frozen buffers before loading: loading must never change permissions,
    # source opacity or geometry-dependent bounds to fit an incompatible run.
    for name,value in model.named_buffers():
        if name not in saved['state'] or not equal_state(value,saved['state'][name]):
            raise ValueError('Resume frozen buffer mismatch: '+name)
    order=list(saved['remaining_view_order'])
    if len(order)!=len(set(order)) or any(not 0<=i<training_view_count for i in order):
        raise ValueError('Invalid remaining training-camera order')
    if len(saved['cuda_rng'])!=torch.cuda.device_count():
        raise ValueError('Resume CUDA device count changed')
    model.load_state_dict(saved['state'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    assert equal_state(model.state_dict(),saved['state'])
    assert equal_state(optimizer.state_dict(),saved['optimizer'])
    rng.bit_generator.state=saved['rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return order

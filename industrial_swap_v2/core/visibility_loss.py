"""Differentiable object contribution; legal foreground remains excluded."""
import torch
import torch.nn.functional as F


def masked_l1(prediction,target,weight):
    if weight.ndim==2: weight=weight[None]
    return ((prediction-target).abs()*weight).sum()/(weight.sum()*prediction.shape[0]).clamp_min(1)


def object_contribution(common,camera,background,obj):
    joint=common.concatenate(background,obj)
    colors=torch.cat((torch.zeros((len(background.xyz),3),device=obj.xyz.device),
                      torch.ones((len(obj.xyz),3),device=obj.xyz.device)))
    return common.raster(camera,joint,colors)[0:1]


def visibility_loss(common,camera,background,obj,alone_alpha,weight):
    visible=object_contribution(common,camera,background,obj)
    # Only penalize missing contribution in the explicitly trusted region.
    return (F.relu(alone_alpha-visible)*weight).sum()/weight.sum().clamp_min(1),visible

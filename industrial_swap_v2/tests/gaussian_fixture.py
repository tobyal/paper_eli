"""Tiny CPU alpha compositing fixture."""
from types import SimpleNamespace
import torch

class Common:
    @staticmethod
    def GaussianData(xyz,log_scale,rotation,opacity_logit,sh):
        return SimpleNamespace(xyz=xyz,log_scale=log_scale,rotation=rotation,opacity_logit=opacity_logit,sh=sh)
    @staticmethod
    def concatenate(bg,obj):
        return SimpleNamespace(opacity_logit=torch.cat((bg.opacity_logit,obj.opacity_logit)))
    @staticmethod
    def raster(camera,data,colors):
        alpha=data.opacity_logit.sigmoid()[:,0]
        trans=torch.cumprod(torch.cat((torch.ones(1),1-alpha[:-1])),0)
        return ((alpha*trans)[:,None]*colors).sum(0)[:,None,None]

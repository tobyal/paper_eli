"""Optimization permissions for the independent, color-only DI-SDS experiment."""
import torch

FROZEN = ("xyz", "log_scale", "rotation", "opacity_logit")


def color_only_optimizer(obj, cfg):
    for name in obj.fields:
        param = getattr(obj, name)
        param.requires_grad_(name == "color_logit")
        param.grad = None
    return torch.optim.Adam(
        [{"params": [obj.color_logit], "name": "color_logit", "lr": cfg["color_lr"]}],
        lr=0.0, eps=cfg["eps"],
    )


def gradient_norms(obj):
    norms = {}
    for name in obj.fields:
        param = getattr(obj, name)
        if name in FROZEN:
            assert not param.requires_grad and param.grad is None, name
            norms[name] = 0.0
        else:
            assert param.requires_grad and param.grad is not None, name
            norms[name] = float(param.grad.norm())
    return norms

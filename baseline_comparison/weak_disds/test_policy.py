"""Small CPU test: frozen fields receive neither gradients nor Adam updates."""
import torch
from policy import color_only_optimizer, gradient_norms, FROZEN

class Dummy(torch.nn.Module):
    fields=FROZEN+("color_logit",)
    def __init__(self):
        super().__init__()
        for field in self.fields:
            setattr(self,field,torch.nn.Parameter(torch.ones(4,3)))

obj=Dummy()
before={k:getattr(obj,k).detach().clone() for k in obj.fields}
optimizer=color_only_optimizer(obj,{"color_lr":0.001,"eps":1e-15})
sum(getattr(obj,k).square().mean() for k in obj.fields).backward()
norms=gradient_norms(obj)
optimizer.step()
assert all(torch.equal(before[k],getattr(obj,k)) for k in FROZEN)
assert all(norms[k]==0 for k in FROZEN)
assert not torch.equal(before["color_logit"],obj.color_logit)
assert [g["name"] for g in optimizer.param_groups]==["color_logit"]
assert len(optimizer.state)==1
print("PASS: only color gets a gradient, optimizer state, and an update")

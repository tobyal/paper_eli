"""CUDA graph execution of the unchanged LangSplat AE optimizer update."""

import torch
import torch.nn.functional as F


def ae_loss(model, data):
    output = model.decode(model.encode(data))
    return (output - data).square().mean() + 0.001 * (
        1 - F.cosine_similarity(output, data, dim=0).mean()
    )


class AEGraph:
    def __init__(self, model, optimizer, example):
        self.model, self.optimizer = model, optimizer
        self.input = example.clone()
        self.graph = torch.cuda.CUDAGraph()
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer_state = {
            p: {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
            for p, state in optimizer.state.items()
        }
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                ae_loss(model, self.input).backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad():
            for key, value in model.state_dict().items():
                value.copy_(model_state[key])
            for parameter, state in optimizer.state.items():
                previous = optimizer_state.get(parameter)
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if previous is None:
                            value.zero_()
                        else:
                            value.copy_(previous[key])
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            optimizer.zero_grad(set_to_none=True)
            self.loss = ae_loss(model, self.input)
            self.loss.backward()
            optimizer.step()

    def step(self, batch):
        if batch.shape == self.input.shape:
            self.input.copy_(batch)
            self.graph.replay()
            return self.loss.detach()
        self.optimizer.zero_grad(set_to_none=True)
        loss = ae_loss(self.model, batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

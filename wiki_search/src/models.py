import functools
from typing import Literal

import torch
from sentence_transformers import SentenceTransformer


def _find_leafs(model: torch.nn.Module, leafs: list[torch.nn.Module]):
    """
    Find all the leaf modules of the passed model.
    """

    children = list(model.children())
    if len(children) == 0:
        leafs.append(model)
    else:
        for child in children:
            _find_leafs(child, leafs)

    return leafs


class GPUMover:
    def __init__(self, gpu: str = "cuda:0"):

        self.gpu = gpu

    @functools.singledispatchmethod
    def _to_gpu(self, module: torch.nn.Module):
        module.to(self.gpu)

    @_to_gpu.register
    def _(self, module: torch.nn.Linear):

        module.weight = torch.nn.Parameter(module.weight.clone())
        if module.bias is not None:
            module.bias = torch.nn.Parameter(module.bias.clone())

        module.to(self.gpu)

    @_to_gpu.register
    def _(self, module: torch.nn.Embedding):

        module.weight = torch.nn.Parameter(module.weight.clone())
        module.to(self.gpu)

    def __call__(self, model: torch.nn.Module):

        leafs = _find_leafs(model, [])

        for leaf in leafs:
            self._to_gpu(leaf)

        self._to_gpu(model)


def sentence_transformer(
    device: Literal["cpu", "cuda:0"] = "cuda:0", download_model: bool = False
):

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        local_files_only=not download_model,
    )
    if "cuda" in device:
        GPUMover()(model)

    model.eval().requires_grad_(False)
    return model

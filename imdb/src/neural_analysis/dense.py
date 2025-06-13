import torch
import pandas as pd

from typing import override

from src.neural_analysis.dataset import ImdbDataSet as BaseDataSet
from src.data_handling import SparseNumericTestData


class ImdbDataSet(BaseDataSet):

    @override
    def transform_word_indices(
        self, data: SparseNumericTestData, indices: torch.Tensor
    ):
        return torch.tensor(
            list(map(data.indices_to_weight, indices)),
            dtype=torch.float32,
            device=self.device,
        )


class Model(torch.nn.Module):

    def __init__(self, num_used_words, embedding_dim,   *args, dtype = None, device = None, **kwargs):
        super().__init__(*args, **kwargs)


        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        dtype = dtype or torch.float32
        self.word_vecs = torch.nn.Linear(
            in_features=num_used_words,
            out_features=embedding_dim,
            bias=False,
            device=device,
            dtype=dtype
        )
        # standard initialisation
        torch.nn.init.constant_(self.word_vecs.weight, 1/torch.numel(self.word_vecs.weight))

        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(
                out_features=4,
                bias=True,
                device=device,
                dtype=dtype
            ),
            torch.nn.LazyLinear(
                out_features=2,
                bias=False,
                device=device,
                dtype=dtype
            ),
            torch.nn.LogSoftmax(dim=1)
        )

    @property
    def embeddings(self):
        return self.word_vecs.weight.detach().clone().T
    
    @embeddings.setter
    def embeddings(self, x):
        self.word_vecs.weight.data = torch.nn.parameter.Parameter(x.T)

    def forward(self, x):
        return self.net.forward(self.word_vecs(x.to(dtype=self.word_vecs.weight.dtype)))

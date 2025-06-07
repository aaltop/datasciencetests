import torch
import pandas as pd

from src.data_handling import StandardLengthTestData


class ImdbDataSet(torch.utils.data.TensorDataset):

    def __init__(self, data: StandardLengthTestData, start_row=0, end_row: int|None = None, device = None):
        '''
        Arguments:
            dtype:
                should be of integer dtype suitable for torch tensors
        '''

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        end_row = len(data) if end_row is None else end_row
        data = data[start_row:end_row]

        # create context mappings
        contexts, indices = zip(*data.context_and_indices)
        context_categorical = pd.Categorical(contexts)
        self.context = context_categorical.codes
        self.context_mapping = dict(zip(context_categorical.categories, self.context))

        # needs to be int64 for nll loss, seemingly
        # could also just directly calculate though, which would
        # allow using at least int32
        self.context = torch.tensor(self.context, dtype=torch.int64, device=device)
        
        words = torch.tensor(indices, dtype=torch.int32, device=self.device)
        self.words = words

    def __len__(self):

        return len(self.context)
    
    def __getitem__(self, idx):

        return self.words[idx], self.context[idx]

    def numeric_context_to_word(self):

        return list(map(lambda val: self.context_mapping[str(int(val))], self.context))
    

class EmbeddingLinear(torch.nn.Module):
    def __init__(self, embed_dim: int, sentence_length: int, num_classes: int, *args, device = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.sentence_length = sentence_length
        self.num_classes = num_classes

        # reduce along the embedding dimension
        self.word_reduction = torch.nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True,
            device=device
        )

        # reduce along the sentence dimension
        # not particularly useful, as the idea is that
        # the passed-in vector also encodes positional data.
        # So change to using something more appropriate at some point
        self.sentence_reduction = torch.nn.Linear(
            in_features=sentence_length,
            out_features=num_classes,
            bias=False,
            device=device
        )

    def forward(self, x):
        reduced_sentences = self.word_reduction(x).reshape(-1, self.sentence_length)
        return self.sentence_reduction(reduced_sentences)

class Model(torch.nn.Module):

    def __init__(self, num_used_words: int, embedding_dim: int, sentence_length: int, *args, device = None, **kwargs):
        super().__init__(*args, **kwargs)

        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.word_vecs = torch.nn.Embedding(
            num_used_words,
            embedding_dim,
            padding_idx=0,
            device=device
        )
        # standard initialisation
        with torch.no_grad():
            torch.nn.init.constant_(self.word_vecs.weight, 1/torch.numel(self.word_vecs.weight))
            self.word_vecs.weight[0] = 0.0

        self.net = torch.nn.Sequential(
            EmbeddingLinear(
                embedding_dim,
                sentence_length,
                2,
                device=device
            ),
            torch.nn.LogSoftmax(dim=1)
        )

    @property
    def embeddings(self):
        return self.word_vecs.weight.detach().clone()
    
    @embeddings.setter
    def embeddings(self, x):
        self.word_vecs.weight.data = torch.nn.parameter.Parameter(x)

    def forward(self, x):
        return self.net.forward(self.word_vecs(x.to(dtype=torch.int32)))
import torch
import pandas as pd


from src.data_handling import SparseNumericTestData



class ImdbDataSet(torch.utils.data.TensorDataset):

    def __init__(self, data: SparseNumericTestData, start_row=0, end_row: int|None = None, device = None):
        '''
        Arguments:
            dtype:
                should be of integer dtype suitable for torch tensors
        '''

        end_row = len(data) if end_row is None else end_row
        data = data[start_row:end_row]

        # create context mappings
        contexts, indices = zip(*data.context_and_indices)
        context_categorical = pd.Categorical(contexts)
        self.context = context_categorical.codes
        self.context_mapping = dict(zip(context_categorical.categories, self.context))

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # needs to be int64 for nll loss, seemingly
        # could also just directly calculate though, which would
        # allow using at least int32
        self.context = torch.tensor(self.context, dtype=torch.int64, device=device)
        
        words = torch.tensor(list(map(data.indices_to_weight, indices)), dtype=torch.float32, device=self.device)
        self.words = words

    def __len__(self):

        return len(self.context)
    
    def __getitem__(self, idx):

        return self.words[idx], self.context[idx]

    def numeric_context_to_word(self):

        return list(map(lambda val: self.context_mapping[str(int(val))], self.context))
    

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
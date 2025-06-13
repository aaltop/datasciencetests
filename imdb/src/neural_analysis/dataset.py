import torch
import pandas as pd

from src.data_handling import SparseNumericTestData


class ImdbDataSet(torch.utils.data.TensorDataset):

    def __init__(
        self,
        data: SparseNumericTestData,
        start_row=0,
        end_row: int | None = None,
        device=None,
    ):
        """
        Arguments:
            dtype:
                should be of integer dtype suitable for torch tensors
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        end_row = len(data) if end_row is None else end_row
        data = data[start_row:end_row]

        # create context mappings
        contexts, indices = zip(*data.context_and_indices)
        context_categorical = pd.Categorical(contexts)
        self.context = context_categorical.codes
        self.context_mapping = dict(enumerate(context_categorical.categories))

        # needs to be int64 for nll loss, seemingly
        # could also just directly calculate though, which would
        # allow using at least int32
        self.context = torch.tensor(self.context, dtype=torch.int64, device=device)

        self.words = self.transform_word_indices(data, indices)

    def transform_word_indices(
        self, data_handler: SparseNumericTestData, indices: torch.Tensor
    ):
        """
        Given the data handler from which `indices` come from, transform
        `indices` to a suitable form to be used by the dataset.
        """
        return torch.tensor(indices, dtype=torch.int32, device=self.device)

    def __len__(self):

        return len(self.context)

    def __getitem__(self, idx):

        return self.words[idx], self.context[idx]

    def numeric_context_to_word(self):

        return list(map(lambda val: self.context_mapping[int(val)], self.context))

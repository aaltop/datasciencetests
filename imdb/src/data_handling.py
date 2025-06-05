from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

import pandas as pd

from pathlib import Path
from collections.abc import Iterable

from .string_processing import StringProcessor
from .context_counter import ContextCounter


processor = StringProcessor(EnglishStemmer().stem, stopwords.words("english"))


def process_training_data(
    words: Iterable[str], context: Iterable[str], train_data_path: str
) -> ContextCounter:
    """
    Process the raw training data of `words` and `context` and write the result to
    `train_data_path`. `context` should include one value for each
    string of words in `words`, so matching indices.
    """

    # get and process the given words
    processed_words = map(lambda val: list(processor.process(val)), words)

    # match the processed words with the contexts
    counter = ContextCounter()
    counter.add_each(context, processed_words)

    # save train data
    train_data_path = Path(train_data_path)
    with open(train_data_path, "w", encoding="utf-8") as f:
        f.write(counter.to_csv())

    return counter


def process_test_data(
    words: Iterable[str], context: Iterable[str], test_data_path: str
):
    """
    Perform stemming on `words`, writing the stemmed words and their
    contexts to `test_data_path`.

    Format (csv):

    - Header of "context,words"
    - rows of context word and the words themselves separated by spaces
    """

    processed_words = map(lambda val: list(processor.process(val)), words)

    def create_row(context, words):

        return f"{context},{' '.join(words)}"

    test_data_path = Path(test_data_path)
    with open(test_data_path, "w", encoding="utf-8") as f:

        print("context,words", file=f)
        for line in map(create_row, context, processed_words):
            print(line, file=f)


def test_data_to_numeric_full(
    test_context: Iterable[str],
    test_words: Iterable[str],
    train_words: Iterable[str],
    numeric_test_data_path: str,
):
    """
    Create a table from the test data as created by `process_test_data()`
    and write it to `numeric_test_data_path`. This table allows easy computation
    of predicted context (based on train data) against the actual context.
    The format is (as parquet)

    - columns of "__context__" (for the context the words appear in), and the words in `train_words`
        - "__context__" associates the data point with a given context
        - each of the columns after "__context__" contain a boolean (0 or 1)
        for whether the column's word found in the data.

    Arguments:
        test_words:
            Each item is words delimited by spaces.
    """

    numeric_test_data_path = Path(numeric_test_data_path)

    # take a context (e.g. positive vs. negative) and
    # the associated words, return context and a boolean for each word
    # for whether it was found in the train data
    def to_numeric(words):
        uniq = set(words.split())
        num_words = [1 if train_word in uniq else 0 for train_word in train_words]
        return num_words

    cols = ["__context__"] + list(train_words)
    assert len(cols) == len(set(cols))

    df1 = pd.DataFrame()
    df1["__context__"] = test_context
    
    df2 = pd.DataFrame(
        columns=train_words,
        data=map(to_numeric, test_words),
    )

    df = df1.join(df2)

    with open(numeric_test_data_path, "wb") as f:
        df.to_parquet(path=f)

def train_words_dict(train_words: Iterable[str]):

    return {"<unk>": 0} | {train_word: i+1 for i, train_word in enumerate(train_words)}

class SparseNumericTestData:
    '''
    Attributes:
        train_words_dict:
            dictionary of index, word -pairs, where the words are
            words seen during training (and the extra catch-all "\<unk\>" word).
            The indices match those in `context_and_indices`.
        context_and_indices:
            list of tuples of (context, indices) data point pairs, where `context` is
            a string denoting the context (e.g. positive, negative) of the data point,
            and `indices` is a list of indices that map to words using
            `train_words_dict`.
    '''

    def __init__(self, train_words: Iterable[str], context_and_indices: list[tuple[str, list[int]]]):

        self.train_words_dict = {key: value for key, value in enumerate(train_words)}
        self.context_and_indices = context_and_indices

    def indices_to_words(self, indices: Iterable[int]):
        return [self.train_words_dict[i] for i in indices]
    
    def indices_to_weight(self, indices: Iterable[int]):
        '''
        Create a list `x` of length `len(self.train_words_dict)` where,
        for each element `i` in `indices`, `x[i]` is equal to the number of
        times that `i` is in `indices`.
        '''
        ret = [0]*len(self.train_words_dict)
        for idx in indices:
            ret[idx] += 1
        return ret
    
    @property
    def context_and_words(self):
        return [(context, self.indices_to_words(indices)) for context, indices in self.context_and_indices]
    
    @property
    def contexts(self):
        return [context for context, _ in self.context_and_indices]
    
    def __getitem__(self, idx):
        train_words = self.train_words_dict.values()
        return SparseNumericTestData(train_words, self.context_and_indices[idx])
    
    def __len__(self):
        return len(self.context_and_indices)

class SparseNumericTestDataIO:
    '''
    Writes and reads test data in a sparse numeric format.
    '''

    def __init__(
        self,
        test_context: Iterable[str],
        test_words: Iterable[str],
        train_words: Iterable[str],
    ):
        '''
        Arguments:
            test_words:
                Each item is words delimited by spaces.
        '''
        self.test_context = test_context
        self.test_words = test_words
        self.train_words_dict = train_words_dict(train_words)

    def to_indices(self, words: Iterable[str]):
        return [str(self.train_words_dict.get(word, 0)) for word in words]
    
    def row(self, context, words: Iterable[str]):
        return context + " " + " ".join(self.to_indices(words)) + "\n"

    def write(self, path: Path):
        """
        Indexise the test data as created by `process_test_data()`
        and write it to `path`. The format is the following:

        ```
        *train words*
        ==data==
        *data*
        ```

        Before the data itself, the used train words are printed
        in the order they appear in `self.train_words_dict`. The header '==data==' indicates
        the start of the data part. The data is written
        as rows of data points. Each row consists of a context word and
        a number of indices, all delimited by spaces. The indices correspond
        to words in `self.train_words_dict`, where a word found in `self.test_words` is matched
        with the same word's index in `self.train_words_dict`.
        """

        with open(path, "w", encoding="utf-8") as f:
            print(*self.train_words_dict.keys(), sep="\n", file=f)
            print("==data==", file=f)
            for context, words in zip(self.test_context, self.test_words):
                f.write(self.row(context, words.split()))

    @staticmethod
    def read(path: Path) -> SparseNumericTestData:
        '''
        Read the data as written by `self.write()`.
        '''

        training_words = []
        context_and_indices = []
        with open(path, "r", encoding="utf-8") as f:
            while (row := f.readline().strip()) != "==data==":
                training_words.append(row)
            while (row := f.readline().strip()) != "":
                context, *idx = row.split()
                context_and_indices.append((context, list(map(int, idx))))
        
        return SparseNumericTestData(training_words, context_and_indices)
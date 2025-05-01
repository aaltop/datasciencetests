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
    set of words in `words`, so matching indices.
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


def test_data_to_numeric(
    test_context: Iterable[str],
    test_words: Iterable[str],
    train_words: Iterable[str],
    numeric_test_data_path: str,
):
    """
    Create a table from the test data as written by `process_test_data()`
    written to `numeric_test_data_path`. This table allows easy computation
    of predicted context (based on train data) against the actual context.
    The format is

    - Pre-header of {context_word_id},{context_word} pairs
    - CSV header of the words in `train_words`
    - The data, of {context_word_id},{...word_indices}
        - {word_indices} the indices of words in `train_words` that
        appeared in the given test data point's words.
    """

    numeric_test_data_path = Path(numeric_test_data_path)

    # take a context (e.g. positive vs. negative) and
    # the associated words, return context and a boolean for each word
    # for whether it was found in the train data
    def to_numeric(context, words):
        num_words = [1 if train_word in words else 0 for train_word in train_words]
        return [context] + num_words

    cols = ["__context__"] + list(train_words)
    assert len(cols) == len(set(cols))
    df = pd.DataFrame(
        columns=cols,
        data=map(to_numeric, test_context, test_words),
    )
    with open(numeric_test_data_path, "wb") as f:
        df.to_parquet(path=f)

# doing basic python "as much as possible" for the fun of it

from pathlib import Path
from collections.abc import Iterable

from .string_processing import StringProcessor
from .context_counter import ContextCounter

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

processor = StringProcessor(EnglishStemmer().stem, stopwords.words("english"))

def process_training_data(words:Iterable[str], context:Iterable[str], train_data_path:str) -> ContextCounter:
    '''
    Process the raw training data of `words` and `context` and write the result to 
    `train_data_path`. `context` should include one value for each
    set of words in `words`, so matching indices.
    '''

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

def process_test_data(words: Iterable[str], context: Iterable[str], test_data_path:str):
    '''
    Perform stemming on `words`, writing the stemmed words and their
    contexts to `test_data_path`.

    Format (csv):
    
    - Header of "context,words"
    - rows of context word and the words themselves separated by spaces
    '''

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
    numeric_test_data_path:str
):
    '''
    Create a table from the test data as written by `process_test_data()`
    written to `numeric_test_data_path`. This table allows easy computation
    of predicted context (based on train data) against the actual context.
    The format is

    - Pre-header of {context_word_id},{context_word} pairs
    - CSV header of "context_id" followed by each of the words in `train_words`
    - The data, of {context_word_id},{...word_flags}
        - {word_flags} denote whether a word in `train_words` appeared in
        in the given test data point's words.
    '''


    numeric_test_data_path = Path(numeric_test_data_path)
    context_id_to_word = dict(enumerate(set(test_context)))
    context_word_to_id = {v: k for k, v in context_id_to_word.items()}

    # take a context (e.g. positive vs. negative) and
    # the associated words, return context and a boolean for each word
    # for whether it was found in the train data
    def to_numeric(context, words):

        num_words = ["1" if train_word in words else "0" for train_word in train_words]
        num_words = ",".join(num_words)
        return f"{context_word_to_id[context]},{num_words}"


    with open(numeric_test_data_path, "w", encoding="utf-8") as f:
        
        # pre-header (it's stupid, but whatever)
        print(",".join(map(lambda val: ",".join([str(val[0]),val[1]]), context_id_to_word.items())), file=f)

        # csv header
        print(f"context_id,{','.join(train_words)}", file=f)

        for line in map(to_numeric, test_context, test_words):
            print(line, file=f)

def main():

    import pandas as pd

    data = pd.read_csv("../data/imdb.csv")

    train_data_path = "../data/train_context_counts.csv"
    # raw_train_data = data.iloc[:25000]
    # process_training_data(raw_train_data.review, raw_train_data.sentiment, train_data_path)
    train_data = pd.read_csv(train_data_path)
    # print("Train data done")

    test_data_path = "../data/test_data.csv"
    # raw_test_data = data.iloc[25000:]
    # process_test_data(raw_test_data.review, raw_test_data.sentiment, "../data/test_data.csv")
    # print("Test data done")

    test_data = pd.read_csv(test_data_path)
    print(test_data)

    numeric_test_data_path = "../data/numeric_test_data.csv"
    test_data_to_numeric(
        test_data.context,
        test_data.words,
        train_data.word[:100], # probably don't want to take all words, just take some of the most usual
        numeric_test_data_path
    )
    print("Numeric test data done")

    numeric_test_data = pd.read_csv(numeric_test_data_path, skiprows=1)
    print(numeric_test_data)
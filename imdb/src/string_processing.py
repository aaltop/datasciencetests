import string
import abc

class AbstractStringProcessor(abc.ABC):


    @abc.abstractmethod
    def stopword_filter(self, _str:str) -> str:
        '''
        Filter out any stopwords. Run first due to stopwords possibly
        containing punctuation.
        '''

    @abc.abstractmethod
    def filter_char(self, _str:str) -> str:
        '''
        Filter out unwanted characters from `_str`, replacing
        with character that is suitable for processing with `self.to_words()`.
        '''

    @abc.abstractmethod
    def to_words(self, _str:str):
        '''
        Split `_str` into an iterator of words.
        '''

    @abc.abstractmethod
    def pre_filter(self, word:str):
        '''
        Return True if `word` passes filter, otherwise False. Called
        before `self.transform()`.
        '''

    @abc.abstractmethod
    def transform(self, word:str):
        '''
        Process `word` into a suitable format. Called after `self.filter()`.
        '''

    @abc.abstractmethod
    def post_filter(self, word:str):
        '''
        Return True if `word` passes filter, otherwise False. Called
        after `self.transform()`.
        '''

    def process(self, _str:str):
        '''
        Process `_str` in to an iterator of words.
        '''

        # TODO: might want to change it so that filters and the others
        # are only run under some circumstance
        pre_filtered = filter(self.pre_filter, self.to_words(self.filter_char(self.stopword_filter(_str))))
        return filter(self.post_filter, map(self.transform, pre_filtered))


class StringProcessor(AbstractStringProcessor):

    filtered_char = string.punctuation+string.digits

    def __init__(self, stemmer, stopwords):
        '''
        `stemmer` is a callable that performs stemming of a word.
        `stopwords` is a list of words to filter out as stopwords;
        this filtering is run first and last.
        '''


        self.stemmer = stemmer
        self.stopwords = stopwords

    def stopword_filter(self, _str: str):
        
        if (self.stopwords is None):
            return _str

        words_list = _str.split()
        return " ".join(
            filter(
                lambda word: not (word in self.stopwords),
                words_list
            )
        )

    def filter_char(self, _str):

        return "".join(
            map(lambda char: " " if char in self.filtered_char else char, _str)
        )
    
    def to_words(self, _str):

        return _str.split()
    
    def pre_filter(self, word):

        return (
            (len(word) >= 2) 
            and not (word in ("br"))
        )
    
    def post_filter(self, word):

        return not (word in self.stopwords)
    
    def transform(self, word):

        return self.stemmer(word)
    

if __name__ == "__main__":

    from pathlib import Path

    import pandas as pd
    import nltk.stem.snowball as snw
    from nltk.corpus import stopwords

    data_path = Path("../data/imdb.csv")
    imdb_data = pd.read_csv(data_path)
    imdb_data["positive"] = imdb_data.sentiment.map({"positive":1, "negative":0})
    del imdb_data["sentiment"]
    sample = imdb_data.review.sample(1).iat[0]
    processor = StringProcessor(snw.EnglishStemmer().stem, stopwords.words("english"))
    print(" ".join(processor.process(sample)))
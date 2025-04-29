from src.string_processing import StringProcessor
from src.context_counter import ContextCounter

from nltk.stem.snowball import EnglishStemmer

# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords

processor = StringProcessor(
        EnglishStemmer().stem,
        stopwords.words("english")
)

counter = ContextCounter()

# from src.analyse import analyse

import pandas as pd

idat = pd.read_csv("./data/imdb.csv")
trunc = idat.iloc[:100,:]
print(trunc)
trunc["proc_rev"] = trunc["review"].map(lambda val: list(processor.process(val)))
counter.add_each(trunc.sentiment, trunc.proc_rev)
print(counter)

# counter = analyse(dat.iloc[:10,:], "sentiment", "review")

# dat = pd.read_csv("./data/train_context_counts.csv")
# dat = dat[dat["total"] > 1000]
# dat["pos_neg"] = dat["positive"] - dat["negative"]
# dat["pos_neg_norm"] = dat["pos_neg"]/dat["total"]
# sorted_dat = dat.sort_values("pos_neg_norm")
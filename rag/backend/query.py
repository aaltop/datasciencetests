from functools import partial
from io import StringIO

import polars as pl
import torch

import src.env

from src.models import sentence_transformer
from src.filesystem import fs
from src.gui import Search


def main():

    model = sentence_transformer()
    # TODO: Move ID and embedding to separate table, keep only that actively
    # loaded, while looking up the IDs from another table for actual content? 
    df = pl.read_parquet(fs.encodings_output / "python_docs_text.parquet")
    encodings = df[:, "encoding"].to_torch()

    def query_docs(query:str):

        query_embed = torch.tensor(model.encode_query(query), dtype=torch.float64)
        sim = model.similarity(query_embed, encodings)[0]
        top = torch.topk(sim, k=10)
        top_df = df[list(top.indices.tolist()), :]

        buf = StringIO()
        _print = partial(print, file=buf)
        sep = "-="*10 + "-"
        for i in range(len(top_df)):
            _print(sep)
            _print(f"{i+1}, file {top_df[i, "file"]}:")
            _print(sep)
            _print(top_df[i, "text"])
        
        return buf.getvalue()

    gui = Search(query_docs)
    gui.run()


if __name__ == "__main__":

    main()
from pathlib import Path

import polars as pl
import torch

import src.env

from src.models import sentence_transformer
from src.filesystem import fs
from src.gui import Search, TextLocation


class SemanticSearch(Search):

    def __init__(self):

        self.model = sentence_transformer()
        # TODO: Move ID and embedding to separate table, keep only that actively
        # loaded, while looking up the IDs from another table for actual content? 
        self.df = (
            pl.read_parquet(fs.encodings_output / "python_docs_text.parquet")
        )
        self.encodings = self.df[:, "encoding"].to_torch()

    def submit_search(self, query: str) -> list[TextLocation]:
        query_embed = torch.tensor(self.model.encode_query(query), dtype=torch.float64)
        sim = self.model.similarity(query_embed, self.encodings)[0]
        top = torch.topk(sim, k=10)
        top_df = self.df[list(top.indices.tolist()), :]
        top_df = top_df.with_columns(
            # make filenames relative to the docs directory so they're not so
            # long
            pl.col("file").map_elements(lambda x: str(Path(x).relative_to(fs.text_data.absolute())), return_dtype=str),
            # add one to row, as tkinter Text does it like that (in the df,
            # they're zero-indexed)
            pl.col("start_row").add(1),
            pl.col("end_row").add(1)
        )
        return list(map(TextLocation, top_df.to_dicts()))

    def get_file(self, file_name: str) -> str:
        absolute_path = fs.text_data / file_name
        return absolute_path.read_text()

def main():

    SemanticSearch().run()


if __name__ == "__main__":

    main()
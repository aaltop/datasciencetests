import abc
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict
import time
import itertools

import polars as pl
from sentence_transformers import SentenceTransformer

import src.env

from src.filesystem import fs
from src.io import TerminalPrinter


class FileContent(TypedDict):
    '''Represents content on a single row and given columns in a "file".'''
    row: int
    cols: tuple[int, int]
    text: str

class AbstractTextFile(abc.ABC):

    @abc.abstractmethod
    def content(self) -> list[FileContent]:
        """
        Returns the text content of the file.
        """


class TextFile(AbstractTextFile):

    def __init__(self, path: Path):
        self.path = path

    def content(self) -> list[FileContent]:
        with open(self.path) as f:
            return [FileContent(row=row, cols=(0, len(line)), text=line) for row, line in enumerate(f.readlines())]

class TextChunker:
    '''
    Create overlapping chunks of text suitable for use in calculating
    vector embeddings.
    '''

    def __init__(self, chunk_size:int):
        '''
        Arguments:
            chunk_size:
                The approximate size of each chunk (how many characters
                to include in the chunk).
        '''

        self.chunk_size = chunk_size

    def chunk(self, file: AbstractTextFile) -> Iterable[list[FileContent]]:

        content = file.content()
        text_length = [len(cont['text']) for cont in content]
        i = 0
        curr_start_idx = 0
        texts_length = 0
        contents: list[FileContent] = []
        while curr_start_idx < len(text_length):
            if (len(contents) < 1 or (texts_length < self.chunk_size)) and i < len(text_length):
                texts_length += len(content[i]['text'])
                contents.append(content[i])
                i += 1
                continue
            else:
                yield contents

                # start next chunk from roughly halfway between start and end
                # of this chunk, or one beyond the end if the chunk
                # had only one piece of text
                curr_start_idx = (curr_start_idx + i) // 2 + 1
                i = curr_start_idx
                if i >= len(text_length):
                    break

                texts_length = 0
                contents = []


class TextLocation(TypedDict):
    file: str
    start_row: int
    end_row: int

class Encoding(TextLocation):
    text: str
    encoding: list[float]



def encode(files: list[Path], name: str, model: SentenceTransformer):
    '''
    Encode the contents of the text files in `files`, creating
    a set of parquet tables and a combined version of these, using `name`
    for names, in the output/ directory.
    '''
    model_embed_dimension = model.get_embedding_dimension()

    files: list[TextFile] = [TextFile(f) for f in files]
    # model does have get_max_seq_length, but didn't really seem
    # like that was the actual length it truncates at. The output
    # still seemed to change beyond that, though I suppose I don't
    # know whether its a hard truncate. Anyway, use a bigger chunk size
    # because of this.
    chunker = TextChunker(200)

    encodings = []
    start_time = time.time()
    printer = TerminalPrinter()
    start_idx = 0
    current_chunks = 0
    max_chunks = 10_000
    to_write = len(files)
    print()
    output_dir = fs.encodings_output / name
    output_dir.mkdir(exist_ok=True)
    for i, file in enumerate(files):
        printer.clear()
        printer.print(f"{str(file.path)}")
        printer.print(f"{i+1}/{to_write}, {round(time.time() - start_time, 2)} seconds elapsed")

        chunks = list(chunker.chunk(file))
        current_chunks += len(chunks)
        printer.print(f"Chunks: {len(chunks)}")

        texts = ["".join(ch["text"] for ch in chunk) for chunk in chunks]
        # TODO: could pass a set batch each time, should be faster,
        # right now it might pass only a few chunks of text for very
        # small files
        enc = model.encode_document(texts)

        rows = [(chunk[0]["row"], chunk[-1]["row"]) for chunk in chunks]
        file_name = str(file.path.absolute())
        enc_rows = [Encoding(file=file_name, start_row=r[0], end_row=r[1], text=t, encoding=e) for r, e, t in zip(rows, enc.tolist(), texts)]
        encodings += enc_rows

        # periodically write the encodings to file to not use up
        # memory too much (also write at the end)
        if current_chunks > max_chunks or i+1 == to_write:
            df = pl.DataFrame(encodings)
            encodings = []
            df.write_parquet(output_dir / f"{start_idx}_{i}.parquet")
            df = None
            start_idx = i+1
            current_chunks = 0

    # combine the created tables
    df = pl.concat(map(pl.read_parquet, output_dir.rglob("*.parquet")))
    # set schema, namely to the get encodings into array form for better
    # performance and easier transformation to torch.tensor
    df = pl.DataFrame(
        df,
        schema=dict(
            file=str,
            start_row=int,
            end_row=int,
            text=str,
            encoding=pl.Array(pl.Float64, model_embed_dimension)
        )
    )
    df.write_parquet(fs.encodings_output / f"{name}.parquet")
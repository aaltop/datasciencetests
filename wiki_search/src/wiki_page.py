from typing import TypedDict


class IndexedPage(TypedDict):
    """
    A wiki page to be indexed.
    """

    title: str
    text: str


type Embedding = list[float]


class EmbeddingIndexedPage(IndexedPage):
    """
    A wiki page with a semantic embedding to be indexed.
    """

    embedding: Embedding


class PageDetails(IndexedPage):
    """
    A Parsed page from a wikipedia XML dump.
    """

    start_line: int
    end_line: int


class EmbeddingPageDetails(PageDetails):
    embedding: Embedding

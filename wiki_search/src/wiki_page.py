from typing import TypedDict


class IndexedPage(TypedDict):
    """
    A wiki page to be indexed.
    """

    title: str
    text: str


type Embedding = list[float]


class EmbeddingMixin(TypedDict):
    text_embedding: Embedding


class EmbeddingIndexedPage(IndexedPage, EmbeddingMixin):
    """
    A wiki page with a semantic embedding to be indexed.
    """


class PageDetails(IndexedPage):
    """
    A Parsed page from a wikipedia XML dump.
    """

    start_line: int
    end_line: int


class EmbeddingPageDetails(PageDetails, EmbeddingMixin): ...

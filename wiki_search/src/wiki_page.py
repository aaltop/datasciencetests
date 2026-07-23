from typing import TypedDict


class IndexedPage(TypedDict):
    """
    A wiki page to be indexed.
    """

    title: str
    text: str


class PageDetails(IndexedPage):
    """
    A Parsed page from a wikipedia XML dump.
    """

    start_line: int
    end_line: int

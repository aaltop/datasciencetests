"""
Data transformation utilities.
"""

from src.strings import NestedDelimitedFinder


def _process_squareb_contents(article_name: str):
    _split = article_name.split("|")
    if len(_split) > 1:
        return _split[1]

    # could be more sophisticated, but should be fine for now
    # used to remove stuff like "[[Category:<whatever>]]"
    if ":" in article_name:
        return ""

    return article_name


_delimited_wavy = NestedDelimitedFinder(r"\{\{", r"\}\}", left="{{", right="}}")
_delimited_square = NestedDelimitedFinder(r"\[\[", r"\]\]", left="[[", right="]]")


def clean_wiki_text(text: str):
    """
    Return a cleaned wiki page.
    """

    text = "".join(
        section.contained_content
        for section in _delimited_wavy.find(text)
        if not section.delimited
    )

    text = "".join(
        _process_squareb_contents(section.contained_content)
        if section.delimited
        else section.contained_content
        for section in _delimited_square.find(text)
    )

    return text.strip()

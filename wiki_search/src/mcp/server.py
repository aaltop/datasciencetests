import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path().absolute()))

from mcp.server import MCPServer
from src.opensearch import rest
from src.strings import NestedDelimitedFinder

server = MCPServer("opensearch")


INDEX_NAME = "semantic_wiki_pages"


wavy = re.compile("{{.+?({{.+}})?.+?}}")
file_link = re.compile(r"\[\[.+?:.+?\]\]")
article_link = re.compile(r"\[\[(?P<text>.+?)\]\]")
renamed_article_link = re.compile(r"\[\[(.+?\|)?(?P<text>.+?)\]\]")


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


def _clean_text(text: str):

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

    text = text.strip().split("\n")[0]

    return text


@server.tool()
def search_wiki(query: str):
    """
    Use a semantic query to search for appropriate wiki pages.

    Args:
        query:
            The query to use to search for wiki pages.

    Returns:
        A list of objects with keys "title" and "text" which
        represent the recommended pages.
    """

    with rest.default_rest() as api:
        response = api.search_index(INDEX_NAME, semantic_query=query)

    data = response.json()
    hits: list[dict] = sorted(
        data["hits"]["hits"], key=lambda x: x["_score"], reverse=True
    )

    title_and_text = [
        dict(title=hit["_source"]["title"], text=_clean_text(hit["_source"]["text"]))
        for hit in hits
    ]
    return json.dumps(title_and_text)


if __name__ == "__main__":
    server.run("streamable-http", port=8110)

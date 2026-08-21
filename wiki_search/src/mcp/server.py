import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path().absolute()))

from mcp.server import MCPServer
from src.opensearch import rest
from src.transform import clean_wiki_text

server = MCPServer("opensearch")


INDEX_NAME = "semantic_wiki_pages"


@server.tool()
def sqrt(x: float):
    """
    Calculate the square root of a value.

    Args:
        x:
            The value for which to calculate the square root.

    Returns:
        The square root of the passed value.
    """

    return math.sqrt(x)


# add a few extra tools for the sake of testing
@server.tool()
def natlog(x: float):
    """
    Calculate the natural logarithm of a value.

    Args:
        x:
            The value for which to calculate the natural logarithm.

    Returns:
        The natural logarithm of the passed value.
    """

    return math.log(x)


@server.tool()
def search_wiki(query: str):
    """
    Use a semantic query to search for appropriate wiki pages.

    Args:
        query:
            The query to use to search for wiki pages.

    Returns:
        A list of objects (in JSON) with keys "title" and "text" which
        represent the recommended pages, or a string indicating service
        unavailability.
    """

    with rest.default_rest() as api:
        response = api.search_index(INDEX_NAME, semantic_query=query)

    if not response.ok:
        return "Search not available"

    data = response.json()
    hits: list[dict] = sorted(
        data["hits"]["hits"], key=lambda x: x["_score"], reverse=True
    )

    title_and_text = [
        dict(
            title=hit["_source"]["title"],
            text=clean_wiki_text(hit["_source"]["text"]).split("\n")[0],
        )
        for hit in hits
    ]
    return json.dumps(title_and_text)


if __name__ == "__main__":
    server.run("streamable-http", port=8110)

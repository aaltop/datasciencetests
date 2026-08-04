import json
import sys
from pathlib import Path

sys.path.append(str(Path().absolute()))

from mcp.server import MCPServer
from src.opensearch import rest

server = MCPServer("opensearch")


INDEX_NAME = "semantic_wiki_pages"


@server.tool()
def search_wiki(query: str):
    """
    Use a semantic query to search for appropriate wiki pages.

    Returns a list of objects with keys "title" and "text" which
    represent the recommended pages.
    """

    with rest.default_rest() as api:
        response = api.search_index(INDEX_NAME, semantic_query=query)

    data = response.json()
    hits: list[dict] = sorted(
        data["hits"]["hits"], key=lambda x: x["_score"], reverse=True
    )

    title_and_text = [
        dict(title=hit["_source"]["title"], text=hit["_source"]["text"]) for hit in hits
    ]
    return json.dumps(title_and_text)


if __name__ == "__main__":
    server.run("streamable-http", port=8110)

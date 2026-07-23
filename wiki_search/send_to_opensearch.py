import json
import time
import tomllib as toml

import requests
import urllib3

from src.ansi import ANSI
from src.filesystem import FileSystem
import src.opensearch as opensearch
from src.wiki_page import PageDetails, IndexedPage
from src.env import Env

fs = FileSystem()
env = Env(fs.root / "env.toml")
# making requests to locally served thing where we don't want to check
# the ssl, so disable the warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PageBulkRequestCreator(
    opensearch.BaseBulkRequestCreator[PageDetails, IndexedPage]
):
    def action_document_pairs(self):

        for doc in self.docs:
            action = opensearch.ActionCreator.create(
                _index="wiki_pages", _id=f"{doc['start_line']}-{doc['end_line']}"
            )
            yield action, IndexedPage(title=doc["title"], text=doc["text"])


def main():

    pages_shards = sorted(
        fs.parsed_pages_dir.iterdir(), key=lambda x: int(str(x.stem).split("_")[0])
    )

    with open(pages_shards[0], "r") as f:
        first: list[PageDetails] = json.load(f)

    request_creator = PageBulkRequestCreator(first)

    pages_to_send = len(first)

    start_time = time.time()
    clear_move_to_start = ANSI().erase_in_line("entire_line").cursor_to_column()
    with opensearch.create_session(
        user_pass=("admin", env.opensearch_password), verify_cert=False
    ) as s:
        api = opensearch.REST(f"https://localhost:{env.opensearch_rest_port}", s)
        print(api.delete_index("wiki_pages").text)

        for total_pages_sent, request_batch in request_creator.create_bulk_requests():
            response = api.bulk(request_batch)
            if not response.ok:
                raise Exception(response.text)
            print(
                "{ansi}Pages sent: {pages_fraction} time elapsed (s): {elapsed_time}".format(
                    ansi=clear_move_to_start,
                    pages_fraction=total_pages_sent / pages_to_send,
                    elapsed_time=time.time() - start_time,
                ),
                end="",
                flush=True,
            )


if __name__ == "__main__":
    main()

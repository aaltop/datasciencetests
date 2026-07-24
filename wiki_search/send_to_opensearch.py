import json
from pathlib import Path
import time
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

WIKI_PAGES_INDEX = "wiki_pages"

class PageBulkRequestCreator(
    opensearch.BaseBulkRequestCreator[PageDetails, IndexedPage]
):
    def action_document_pairs(self):

        for doc in self.docs:
            action = opensearch.ActionCreator.create(
                _index=WIKI_PAGES_INDEX, _id=f"{doc['start_line']}-{doc['end_line']}"
            )
            yield action, IndexedPage(title=doc["title"], text=doc["text"])

def send_bulk_requests(wiki_page_files: list[Path], delete_index: bool = False):

    files_to_process = len(wiki_page_files)
    clear_move_to_start = ANSI().erase_in_line("entire_line").cursor_to_column()

    start_time = time.time()
    with opensearch.create_session(
        user_pass=("admin", env.opensearch_password), verify_cert=False
    ) as s:
        api = opensearch.REST(f"https://localhost:{env.opensearch_rest_port}", s)
        if delete_index:
            api.delete_index(WIKI_PAGES_INDEX)

        # go over each file of wiki pages
        for current_file_index, file in enumerate(wiki_page_files):
            with open(file, "r") as f:
                docs: list[PageDetails] = json.load(f)
            
            pages_to_send = len(docs)
            request_creator = PageBulkRequestCreator(docs=docs)

            # send pages from file to OpenSearch
            for total_pages_sent, request_batch in request_creator.create_bulk_requests():
                response = api.bulk(request_batch)
                if not response.ok:
                    raise Exception(response.text)
                print(
                    "{ansi}Processing file {files_fraction} pages sent: {pages_fraction} time elapsed (s): {elapsed_time}".format(
                        ansi=clear_move_to_start,
                        files_fraction=f"{current_file_index+1}/{files_to_process}",
                        pages_fraction=total_pages_sent / pages_to_send,
                        elapsed_time=time.time() - start_time,
                    ),
                    end="",
                    flush=True,
                )

def main():

    pages_shards = sorted(
        fs.parsed_pages_dir.iterdir(), key=lambda x: int(str(x.stem).split("_")[0])
    )

    send_bulk_requests(pages_shards)


if __name__ == "__main__":
    main()

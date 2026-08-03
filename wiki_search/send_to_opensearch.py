import json
import time
from collections import namedtuple
from collections.abc import Callable, Iterable, Iterator, Sized
from pathlib import Path

import torch

from opensearch import rest
from src.ansi import ANSI
from src.env import Env
from src.filesystem import FileSystem
from src.opensearch import bulk_request
from src.wiki_page import (
    EmbeddingIndexedPage,
    EmbeddingMixin,
    EmbeddingPageDetails,
    IndexedPage,
    PageDetails,
)

fs = FileSystem()
env = Env(fs.root / "env.toml")


class BulkRequestCreatorWithIndex[T](bulk_request.BaseBulkRequestCreator[T]):
    index: str


class PageBulkRequestCreator(BulkRequestCreatorWithIndex[IndexedPage]):
    index = "wiki_pages"

    def __init__(self, docs: Iterable[PageDetails]):

        self.docs = docs

    def action_document_pairs(self):

        for doc in self.docs:
            action = bulk_request.ActionCreator.create(
                _index=self.index, _id=f"{doc['start_line']}-{doc['end_line']}"
            )
            yield action, IndexedPage(title=doc["title"], text=doc["text"])


class EmbeddingPageBulkRequestCreator(
    BulkRequestCreatorWithIndex[EmbeddingIndexedPage]
):
    index = "semantic_wiki_pages"

    def __init__(self, docs: Iterable[EmbeddingPageDetails]):

        self.docs = docs

    def action_document_pairs(self):

        for doc in self.docs:
            action = bulk_request.ActionCreator.create(
                _index=self.index,
                _id=f"{doc['start_line']}-{doc['end_line']}",
            )
            yield (
                action,
                EmbeddingIndexedPage(
                    title=doc["title"],
                    text=doc["text"],
                    text_embedding=doc["text_embedding"],
                ),
            )


class SizedIterable[IterableType](Iterable[IterableType], Sized):
    pass


class Response:
    """
    Partial interface to requests.Response.
    """

    ok: bool
    text: str


def send_bulk_requests[T](
    file_iterable: SizedIterable[list[T]],
    request_creator_factory: Callable[[Iterable[T]], BulkRequestCreatorWithIndex[T]],
    rest_api: rest.BaseREST[Response],
    delete_index: bool = False,
):

    files_to_process = len(file_iterable)
    clear_move_to_start = ANSI().erase_in_line("entire_line").cursor_to_column()

    start_time = time.time()
    if delete_index:
        index = request_creator_factory([]).index
        rest_api.delete_index(index)

    # go over each file
    for current_file_index, docs in enumerate(file_iterable):
        pages_to_send = len(docs)
        request_creator = request_creator_factory(docs)

        # send pages from file to OpenSearch
        for (
            total_pages_sent,
            request_batch,
        ) in request_creator.create_bulk_requests():
            response = rest_api.bulk(request_batch)
            if not response.ok:
                raise Exception(response.text)
            print(
                "{ansi}Processing file {files_fraction} pages sent: {pages_fraction} time elapsed (s): {elapsed_time}".format(
                    ansi=clear_move_to_start,
                    files_fraction=f"{current_file_index + 1}/{files_to_process}",
                    pages_fraction=total_pages_sent / pages_to_send,
                    elapsed_time=time.time() - start_time,
                ),
                end="",
                flush=True,
            )


def send_page_details(parsed_pages_files: list[Path]):
    """
    Send PageDetails from the parsed pages files to the OpenSearch API.
    """

    with rest.default_rest() as api:
        send_bulk_requests(
            PageDetailsIterable(parsed_pages_files),
            PageBulkRequestCreator,
            api,
        )


def send_embedding_page_details(parsed_pages_files: list[Path]):
    """
    Send EmbeddedPageDetails from the parsed pages files and corresponding
    embedding files to the OpenSearch API.
    """

    with rest.default_rest() as api:
        send_bulk_requests(
            EmbeddingPageDetailsIterable(parsed_pages_files),
            EmbeddingPageBulkRequestCreator,
            api,
        )


class PageDetailsIterable(SizedIterable[list[PageDetails]]):
    """
    Iterates of the parsed pages files.
    """

    def __init__(self, parsed_pages_files: list[Path]):

        self.parsed_pages_files = parsed_pages_files

    def _page_details_gen(self):

        for file in self.parsed_pages_files:
            with open(file, "r") as f:
                docs: list[PageDetails] = json.load(f)

            yield docs

    def __iter__(self) -> Iterator[list[PageDetails]]:
        return self._page_details_gen()

    def __len__(self):
        return len(self.parsed_pages_files)


class EmbeddingPageDetailsIterable(SizedIterable[list[EmbeddingPageDetails]]):
    """
    Iterates over the parsed pages files and corresponding embedding files.
    """

    def __init__(self, parsed_pages_files: list[Path]):

        self.parsed_pages_files = list(
            filter(self._has_matching_embedding_file, parsed_pages_files)
        )

    def _has_matching_embedding_file(self, parsed_pages_file: Path):

        return fs.page_embeddings_file(parsed_pages_file).exists()

    def _embedding_page_details_gen(self):

        for file in self.parsed_pages_files:
            docs: list[EmbeddingPageDetails] | None = combine_page_and_embedding(file)
            # docs shouldn't be None because of the filter in __init__,
            # but ensure anyway (technically could be None if the combine function
            # changes, of course)
            if docs is None:
                raise ValueError("Docs should not be None")

            yield docs

    def __iter__(self) -> Iterator[list[EmbeddingPageDetails]]:

        return self._embedding_page_details_gen()

    def __len__(self):

        return len(self.parsed_pages_files)


def combine_page_and_embedding(
    parsed_pages_file: Path,
) -> list[EmbeddingPageDetails] | None:
    """
    Combine the parsed pages and their embeddings.
    """

    embedding_file = fs.page_embeddings_file(parsed_pages_file)
    if not embedding_file.exists():
        return

    embeddings: dict[str, torch.Tensor] = torch.load(embedding_file, weights_only=False)
    embeddings_with_list = {title: tens.tolist() for title, tens in embeddings.items()}
    pages: list[PageDetails] = json.loads(parsed_pages_file.read_text())

    return [
        page | EmbeddingMixin(text_embedding=embeddings_with_list[page["title"]])
        for page in pages
    ]


def test_embedding_send():

    pages_shards = fs.get_parsed_pages_files()

    send_bulk_requests(
        EmbeddingPageDetailsIterable(pages_shards[:1]),
        EmbeddingPageBulkRequestCreator,
        MockREST(),
    )


def test_pages_send():

    pages_shards = fs.get_parsed_pages_files()

    send_bulk_requests(
        PageDetailsIterable(pages_shards[:1]),
        PageBulkRequestCreator,
        MockREST(),
    )


MockResponse = namedtuple("Response", ["ok", "text"])


class MockREST(rest.BaseREST):
    """
    Mock the OpenSearch API for testing.
    """

    def bulk(self, body: str):
        print(body)
        return MockResponse(True, "")

    def delete_index(self, index: str):
        print("Delete index:", index)
        return MockResponse(True, "")


def main():

    pages_shards = sorted(
        fs.parsed_pages_dir.iterdir(), key=lambda x: int(str(x.stem).split("_")[0])
    )

    send_embedding_page_details(pages_shards)


if __name__ == "__main__":
    main()

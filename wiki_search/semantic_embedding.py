import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src import filesystem
from src.wiki_page import PageDetails

fs = filesystem.FileSystem()


def get_pages(parsed_pages_file: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Returns:
        Tuple of (texts, titles) of pages.
    """

    with open(parsed_pages_file, "r") as f:
        docs: list[PageDetails] = json.load(f)

    texts, titles = zip(*[(doc["text"], doc["title"]) for doc in docs])
    return texts, titles


def save_embeddings(
    titles: list[str], encodings: torch.Tensor, page_embedding_file: Path
):
    """
    Save the embeddings of pages using torch.save.

    Saved in a dictionary format which matches each element in `titles`
    to each first-index tensor in `encodings`.
    """

    title_and_embedding = {titles[i]: encodings[i] for i in range(len(titles))}
    torch.save(title_and_embedding, page_embedding_file)


def main():

    start_time = time.time()
    download_model = False

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda:0",
        local_files_only=not download_model,
    )

    # print("time to load model:", time.time() - start_time)
    start_time = time.time()
    page_files = fs.get_parsed_pages_files()
    files_to_process = 5
    files_processed = 0
    for i, page_file in enumerate(page_files):
        page_embeddings_file = fs.page_embeddings_file(page_file)
        if page_embeddings_file.exists():
            print(f"Embedding file '{page_embeddings_file}' already exists, skipping")
            continue

        print(
            f"Processing file {files_processed + 1}/{files_to_process} ({i + 1}/{len(page_files)})"
        )
        texts, titles = get_pages(page_file)
        encodings = model.encode([text[:500] for text in texts], show_progress_bar=True)
        save_embeddings(titles, encodings, page_embeddings_file)
        files_processed += 1
        if files_processed >= files_to_process:
            break
    print("finish encoding, elapsed_time:", time.time() - start_time)
    # print(encoded.shape)


if __name__ == "__main__":
    main()

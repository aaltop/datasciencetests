from sentence_transformers.sentence_transformer.model import SentenceTransformer


from typing import Literal

from sentence_transformers import SentenceTransformer


def sentence_transformer(
    device: Literal["cpu", "cuda:0"] = "cuda:0", download_model: bool = False
) -> SentenceTransformer:

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device,
        local_files_only=not download_model,
    )

    model.eval().requires_grad_(False)
    return model

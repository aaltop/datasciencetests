from sentence_transformers.cross_encoder.model import CrossEncoder


from sentence_transformers.sentence_transformer.model import SentenceTransformer


from typing import Literal

from sentence_transformers import SentenceTransformer, CrossEncoder


def sentence_transformer(
    device: Literal["cpu", "cuda:0"] = "cuda:0", download_model: bool = False
) -> SentenceTransformer:

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
        local_files_only=not download_model,
        similarity_fn_name="euclidean"
    )

    model.eval().requires_grad_(False)
    return model

def cross_encoder(device: Literal["cpu", "cuda:0"] = "cuda:0", download_model: bool = False) -> CrossEncoder:

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
        local_files_only=not download_model,
    )
    model.eval().requires_grad_(False)
    return model
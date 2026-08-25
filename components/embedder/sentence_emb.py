from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(model_name=DEFAULT_MODEL):
    model = getattr(load_model, "model", None)
    if model is None:
        load_model.model = {  # type: ignore
            # Load the model and processor (which will handle image pre-processing)
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
            "model": AutoModel.from_pretrained(model_name),
        }
    return load_model.model  # type: ignore


def _mean_pooling(model_output, attention_mask):
    """
    Mean-pool token embeddings while ignoring padding tokens.
    """
    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * mask, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed_embeddings / summed_mask


def embed_text(filepath, state):
    """
    Embed the text contained in `filepath` using a Hugging Face model.

    The model URI is expected at:

        state.META["cfg"]["embeddings"]["model"]

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to the UTF-8 text file.

    state : object
        Object containing the configuration under `state.META`.

    Returns
    -------
    list[float]
        L2-normalized embedding vector for the complete text.
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    model_uri = state.META["cfg"]["embeddings"].get("model", DEFAULT_MODEL)
    emb_size = state.META["cfg"]["embeddings"].get("size", 256)

    text = filepath.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"File contains no text: {filepath}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = load_model(model_uri)["model"].to(device)
    tokenizer = load_model(model_uri)["tokenizer"]

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=emb_size,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output = model(**encoded)

        embedding = _mean_pooling(
            output,
            encoded["attention_mask"],
        )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1,
        )

    return embedding[0].cpu().tolist()


if __name__ == "__main__":
    # Minimal standalone example.
    #
    # Good default model:
    #   sentence-transformers/all-MiniLM-L6-v2

    class State:
        META = {
            "cfg": {
                "embeddings": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        }

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()

    embedding = embed_text(args.filepath, State())

    print(f"Embedding dimension: {len(embedding)}")
    print(embedding)
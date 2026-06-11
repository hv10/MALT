import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

def load_model(state, model_name="microsoft/resnet50"):
    model = getattr(load_model, "model", None)
    if model is None:
        load_model.model = {
            # Load the model and processor (which will handle image pre-processing)
            "processor": AutoProcessor.from_pretrained(model_name),
            "model": AutoModel.from_pretrained(model_name),
        }
    return load_model.model


def embed_image(filepath):
    """
    Embeds an image as a vector using a Hugging Face model specified by its name.
    """
    image = Image.open(filepath).convert("RGB")
    inputs = load_model.model["processor"](images=image, return_tensors="pt")
    with torch.no_grad():  # forward pass
        outputs = load_model.model["model"](**inputs)

    if hasattr(outputs, "pooler_output"):  # extr. emb
        embedding = outputs.pooler_output.squeeze()
    else:
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.numpy()

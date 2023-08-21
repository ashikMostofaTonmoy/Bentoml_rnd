"""This module saves a Keras model to BentoML."""

from pathlib import Path

# from tensorflow import keras
from sentence_transformers import SentenceTransformer, util
import torch
import bentoml


def load_model_and_save_it_to_bento(model_name: Path) -> None:
    """Loads a sentencetransformer model from disk and saves it to BentoML."""
    
    model = SentenceTransformer(model_name)

    bento_model = bentoml.pytorch.save_model("sentence_trn", model)
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    # load_model_and_save_it_to_bento(Path("model"))
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    load_model_and_save_it_to_bento(model_name)
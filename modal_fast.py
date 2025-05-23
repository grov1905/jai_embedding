#modal_fast.py
import modal
import logging
import os
from modal import App

app = App("jai-embedding-app")  # Usa la clase App directamente
logger = logging.getLogger(__name__)

image = modal.Image.debian_slim().pip_install("sentence-transformers", "torch")


@app.function(
    image=image,
    keep_warm=int(os.getenv("MODAL_MIN_CONTAINERS", "0")),
    gpu="L4",
    timeout=30
)
def fast_embedding(texts: list[str], model: str):
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model, device="cuda")
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64, 
            convert_to_tensor=False
        ).tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise
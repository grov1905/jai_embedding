import modal
import logging
import torch
import os
app = modal.App("jai-embedding-app")

logger = logging.getLogger(__name__)

KEEP_WARM = int(os.getenv("MODAL_KEEP_WARM", "0"))  # "1" en producción


def get_device():
    """Auto-detects available hardware"""
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    elif torch.cuda.is_available():  # NVIDIA GPU
        return "cuda"
    return "cpu"

image = modal.Image.debian_slim().pip_install(
    "sentence-transformers",
    "torch"
)

@app.function(
    image=image,
    keep_warm=KEEP_WARM,
    gpu="L4",
    timeout=30
)
def fast_embedding(texts: list[str], model: str):
    from sentence_transformers import SentenceTransformer

    try:
        device = get_device()

        model = SentenceTransformer(
            model,
            device=device)
        
        return model.encode(texts,
                            normalize_embeddings=True,  # <- Normalización aquí
                            batch_size=64, 
                            convert_to_tensor=False
                            ).tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise
#modal_fast.py
import modal
import logging
#import os
import modal


app = modal.App(
    name="jai-embedding-app"  # Solo el nombre es parámetro válido en el constructor
)
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentence-transformers==2.6.1",  # Versión fija
        "torch==2.2.1"  # Versión CUDA
    )
)

@app.function(
    image=image,
    min_containers=0,
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
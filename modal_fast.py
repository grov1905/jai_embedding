#modal_fast.py
import modal
import logging
#import os
import modal


app = modal.App(
    name="jai-embedding-app"  # Solo el nombre es parámetro válido en el constructor
)
logger = logging.getLogger(__name__)

# Creamos una imagen "blindada" para garantizar la compatibilidad
# Forzamos Python 3.11, que tiene un soporte más maduro para estas librerías.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Un conjunto de versiones conocidas por trabajar bien juntas
        "torch==2.2.2",
        "transformers==4.38.2",
        "sentence-transformers==2.7.0",
        "numpy~=1.26.4",  # Forzamos una versión < 2.0 y estable
    )
)

@app.function(
    image=image,
    min_containers=0,
    gpu="L4",
    timeout=180
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
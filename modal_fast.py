#modal_fast.py
import modal
import logging
#import os
import modal


app = modal.App(
    name="jai-embedding-app"  # Solo el nombre es parámetro válido en el constructor
)
logger = logging.getLogger(__name__)

# Corregimos la imagen para evitar conflictos de versiones
image = (
    modal.Image.debian_slim(python_version="3.11")  # Es buena práctica fijar la versión de Python
    .pip_install(
        "sentence-transformers==2.7.0", # Una versión reciente y estable
        "torch==2.3.0", # Compatible con la anterior
        "numpy<2.0"  # ¡Esta es la corrección clave! Forzamos una versión compatible.
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
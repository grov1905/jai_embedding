# embedder.py
import modal
from typing import List
import logging
import torch
from sentence_transformers import SentenceTransformer


app = modal.App("jai-embedder")

image = modal.Image.debian_slim().pip_install(
    "sentence-transformers", "torch"
)
logger = logging.getLogger(__name__)

def get_device():
    """Auto-detects available hardware"""
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    elif torch.cuda.is_available():  # NVIDIA GPU
        return "cuda"
    return "cpu"

@app.function(image=image, timeout=60)
def generate_embeddings(texts: List[str], embedding_model: str) -> List[List[float]]:

    try:
        device = get_device()
        model = SentenceTransformer(embedding_model, device=device)
        model.encode_kwargs = {'normalize_embeddings': True, 'batch_size': 32 if device == "cuda" else 8}
        embeddings = model.encode(texts, convert_to_tensor=False, **model.encode_kwargs)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise
# Añade esto al final de tu embedder.py
@app.local_entrypoint()
def main(texts: str, model: str = "BAAI/bge-small-en-v1.5"):
    """
    Uso:
    modal run embedder.py --texts '["Hola", "Mundo"]' --model "BAAI/bge-small-en-v1.5"
    """
    import ast
    try:
        # Convierte el string a lista de forma segura
        lista_textos = ast.literal_eval(texts)
        if not isinstance(lista_textos, list):
            raise ValueError("El parámetro 'texts' debe ser una lista en formato string")
        
        # Genera los embeddings
        embeddings = generate_embeddings.remote(lista_textos, model)
        print(embeddings)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
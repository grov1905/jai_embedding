from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal
import logging
from typing import List
import os

# Configura las credenciales de Modal (muy importante en Railway)
# Asegúrate de haber configurado las variables de entorno en Railway:
# MODAL_TOKEN_ID y MODAL_TOKEN_SECRET
if "MODAL_TOKEN_ID" not in os.environ or "MODAL_TOKEN_SECRET" not in os.environ:
    raise RuntimeError("Las variables de entorno de Modal (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET) no están configuradas.")


app = FastAPI()

logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    texts: List[str]  # Lista de chunks
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Modelo por defecto


@app.post("/api/v1/embeddings/generate")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        # Busca la función ya desplegada en Modal
        # "jai-embedding-app" es el nombre de tu app en Modal
        # "fast_embedding" es el nombre de la función dentro de esa app
        f = modal.Function.from_name("jai-embedding-app", "fast_embedding")

        # Llama a la función remota de forma asíncrona
        result = await f.remote.aio(request.texts, request.embedding_model)
        
        return {"embeddings": result}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
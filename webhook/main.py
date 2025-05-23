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


@app.post("/generate-embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        # Ya no pasamos el nombre del modelo aquí, porque está definido en la clase de Modal
        # 1. Busca la clase desplegada
        embedding_model_cls = modal.Cls.from_name("jai-embedding-app", "EmbeddingModel")
        # 2. Crea una instancia remota
        model = embedding_model_cls()
        # 3. Llama al método .generate()
        result = await model.generate.remote.aio(request.texts)
        
        return {"embeddings": result}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# webhook/main.py - VERSIÓN CORREGIDA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal
import logging
from typing import List
import os

# Configura las credenciales de Modal
if "MODAL_TOKEN_ID" not in os.environ or "MODAL_TOKEN_SECRET" not in os.environ:
    raise RuntimeError("Las variables de entorno de Modal (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET) no están configuradas.")

app = FastAPI()
logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    texts: List[str]
    embedding_model: str = "BAAI/bge-large-en-v1.5"

@app.post("/api/v1/embeddings/generate")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        logger.info(f"Procesando solicitud de embeddings para {len(request.texts)} textos")
        
        # 1. Busca la clase desplegada
        embedding_model_cls = modal.Cls.from_name("jai-embedding-app", "EmbeddingModel")
        
        # 2. OPCIÓN A: Usar context manager (RECOMENDADO)
        with embedding_model_cls() as model:
            result = await model.generate.remote.aio(request.texts)
        
        # 2. OPCIÓN B: Alternativa sin context manager
        # model = embedding_model_cls()
        # result = await model.generate.remote.aio(request.texts)
        
        logger.info(f"Embeddings generados exitosamente para {len(request.texts)} textos")
        return {"embeddings": result}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
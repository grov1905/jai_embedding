# webhook/main.py - VERSIÓN FINAL CORREGIDA
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
        
        # 2. Crear instancia remota (Modal ejecuta setup() automáticamente)
        model = embedding_model_cls()
        
        # 3. Generar embeddings
        result = await model.generate.remote.aio(request.texts)
        
        logger.info(f"Embeddings generados exitosamente para {len(request.texts)} textos")
        return {"embeddings": result}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Endpoint de salud para verificar que el servicio está funcionando."""
    try:
        # Verificar conexión con Modal
        embedding_model_cls = modal.Cls.from_name("jai-embedding-app", "EmbeddingModel")
        model = embedding_model_cls()
        health_status = await model.health_check.remote.aio()
        
        return {
            "status": "healthy",
            "modal_connection": "ok",
            "model_status": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "error": str(e)
        }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal
import logging
from typing import List

app = FastAPI()
logger = logging.getLogger(__name__)

# Configuración
MAX_DIRECT_PROCESSING = 50  # Máximo chunks para procesamiento inmediato

class EmbeddingRequest(BaseModel):
    texts: List[str]  # Lista de chunks
    model: str = "BAAI/bge-large-en-v1.5"  # Modelo por defecto

# Clients Modal
embedding_app = modal.App("jai-embedding-app")

@app.post("/generate-embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        # Pequeñas cantidades: procesamiento inmediato
        #if len(request.texts) <= MAX_DIRECT_PROCESSING:
            fn = embedding_app.function("fast_embedding")
            result = await fn.remote.aio(request.texts, request.model)
            return {"method": "immediate", "embeddings": result}
        
        # Grandes volúmenes: encolar
        #embedding_app.queue.put({
        #    "texts": request.texts,
        #    "model": request.model,
        #    "batch_id": f"batch_{len(request.texts)}_{hash(tuple(request.texts))}"
        #})
        #return {
        #    "method": "queued",
        #    "batch_id": f"batch_{len(request.texts)}_{hash(tuple(request.texts))}",
        #    "estimated_time": f"{len(request.texts)*0.01:.2f} seconds"
        #}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
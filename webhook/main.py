#webhook/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal
import logging
from typing import List
#from dotenv import load_dotenv
#import os
from modal import App  # Importa igual que en modal_fast.py

#load_dotenv()  # Carga variables desde .env

app = FastAPI()
# Clients Modal
embedding_app = modal.App("jai-embedding-app")

logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    texts: List[str]  # Lista de chunks
    model: str = "BAAI/bge-large-en-v1.5"  # Modelo por defecto


@app.post("/generate-embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        fn = embedding_app.function("fast_embedding")
        if not fn:
            raise HTTPException(status_code=500, detail="Function not found")
        result = await fn.remote.aio(request.texts, request.model)
        return {"embeddings": result}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


# Configuración
#MAX_DIRECT_PROCESSING = 50  # Máximo chunks para procesamiento inmediato


#@app.post("/generate-embeddings")
#async def generate_embeddings(request: EmbeddingRequest):
 #   try:
        # Pequeñas cantidades: procesamiento inmediato
        #if len(request.texts) <= MAX_DIRECT_PROCESSING:
  #          fn = embedding_app.function("fast_embedding")
  #          result = await fn.remote.aio(request.texts, request.model)
  #          #return {"method": "immediate", "embeddings": result}
  #          return {"embeddings": result}
    
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
   # except Exception as e:
   #     logger.error(f"Error processing request: {str(e)}")
   #     raise HTTPException(status_code=500, detail=str(e))
    
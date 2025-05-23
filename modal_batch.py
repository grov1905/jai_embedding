import modal
from modal import Queue, SharedVolume
import numpy as np
from sentence_transformers import SentenceTransformer

app = modal.App("jai-embedding-app")
queue = Queue.from_name("embedding-queue")
volume = SharedVolume().persist("model-cache")

image = modal.Image.debian_slim().pip_install(
    "sentence-transformers",
    "torch",
    "numpy"
)

@app.function(
    image=image,
    gpu="A10G",
    shared_volumes={"/cache": volume},
    timeout=3600,
    concurrency_limit=10
)
def batch_embedding(batch: dict):

    
    # Carga el modelo (con cache persistente)
    model = SentenceTransformer(
        batch["model"],
        cache_folder="/cache/models",
        device="cuda"
    )
    
    # Procesamiento optimizado por chunks
    embeddings = model.encode(
        batch["texts"],
        normalize_embeddings=True,  # <- Normalización aquí
        batch_size=128,
        convert_to_tensor=False
    )
    
    # Guarda resultados en el volumen compartido
    output_path = f"/cache/results/{batch['batch_id']}.npy"
    np.save(output_path, embeddings, allow_pickle=False)
    
    return {"batch_id": batch["batch_id"], "saved_path": output_path}

@app.function(schedule=modal.Period(seconds=10))
def process_queue():
    while queue.len() > 0:
        batch = queue.get()
        batch_embedding.remote(batch)
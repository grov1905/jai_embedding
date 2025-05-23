import modal
import logging
from sentence_transformers import SentenceTransformer

app = modal.App(
    name="jai-embedding-app"
)
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.38.2",
        "sentence-transformers==2.7.0",
        "numpy~=1.26.4",
    )
)

# Convertimos la función en una clase para cargar el modelo una sola vez
@app.cls(
    image=image,
    timeout=180,
    gpu="L4",
    # Mantiene el contenedor "caliente" por 120 segundos después de la última petición
    # para evitar el cold start en llamadas seguidas.
    container_idle_timeout=90 
)
class EmbeddingModel:
    def __enter__(self):
        # Esta parte se ejecuta UNA SOLA VEZ cuando el contenedor arranca.
        logger.info("Cargando el modelo en la GPU...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        logger.info("Modelo cargado exitosamente.")

    @modal.method()
    def generate(self, texts: list[str]):
        # Este método se llama en cada petición y reutiliza el modelo ya cargado.
        logger.info(f"Generando embeddings para {len(texts)} textos.")
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            convert_to_tensor=False
        ).tolist()
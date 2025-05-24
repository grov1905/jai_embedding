# Aplicación de embedding con modal.fast
# modal_fast.py - VERSIÓN CORREGIDA
import modal
import logging
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

app = modal.App(
    name="jai-embedding-app"
)

logger = logging.getLogger(__name__)

def download_model():
    logger.info("Iniciando descarga del modelo...")
    try:
        snapshot_download(
            "BAAI/bge-large-en-v1.5",
            local_dir="/pretrained_model"
        )
        logger.info("Descarga del modelo completada exitosamente.")
    except Exception as e:
        logger.error(f"Error en la descarga del modelo: {e}")
        raise

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.38.2",
        "sentence-transformers==2.7.0",
        "numpy~=1.26.4",
        "huggingface-hub",
    )
    .run_function(download_model)
)

@app.cls(
    image=image,
    timeout=300,
    gpu="L4",
    scaledown_window=90
)
class EmbeddingModel:
    def __enter__(self):  # ✅ CORREGIDO: Sintaxis correcta
        """Este método se ejecuta UNA SOLA VEZ cuando el contenedor arranca."""
        logger.info(">>>> Entrando al método __enter__ para inicializar EmbeddingModel <<<<")
        try:
            # Verifica si el directorio existe y contiene algo
            import os
            if not os.path.exists("/pretrained_model") or not os.listdir("/pretrained_model"):
                logger.error("El directorio /pretrained_model no existe o está vacío.")
                raise RuntimeError("Modelo pre-entrenado no encontrado o directorio vacío.")
            
            logger.info("Cargando el modelo SentenceTransformer desde /pretrained_model...")
            self.model = SentenceTransformer("/pretrained_model", device="cuda")
            logger.info("Modelo SentenceTransformer cargado exitosamente en la GPU.")
            return self  # ✅ IMPORTANTE: Retornar self
            
        except Exception as e:
            logger.critical(f"ERROR CRÍTICO: Fallo al cargar el modelo en __enter__: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):  # ✅ NUEVO: Método requerido
        """Método requerido para el protocolo de context manager."""
        logger.info("Saliendo del context manager de EmbeddingModel")
        # Aquí podrías hacer limpieza si fuera necesario
        pass

    @modal.method()
    def generate(self, texts: list[str]):
        try:
            if not hasattr(self, 'model') or self.model is None:
                logger.error("self.model no está inicializado. Esto indica un fallo en __enter__.")
                raise RuntimeError("El modelo de embeddings no está listo.")
            
            logger.info(f"Generando embeddings para {len(texts)} textos utilizando el modelo cargado.")
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,
                convert_to_tensor=False
            ).tolist()
            
            logger.info(f"Embeddings generados exitosamente para {len(texts)} textos.")
            return embeddings
            
        except RuntimeError as e:
            logger.error(f"Error de tiempo de ejecución en generate: {e}")
            raise
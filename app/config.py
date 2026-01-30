import os

class Config:
    # Rutas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "best_model.pt"))
    
    # MLOps
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "") # Vacío = no loguear o local
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "XRay_Production_Inference")
    
    # Configuración de Imagen
    IMGSZ = 224
    
    # Configuración CLAHE (Debe ser igual al entrenamiento)
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)
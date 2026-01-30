from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

# --- BLOQUES DE DATOS ---

class PredictionData(BaseModel):
    label: str = Field(..., description="Etiqueta predicha (Normal o Anomaly).")
    confidence: float = Field(..., description="Nivel de confianza de la predicción (0-1).")
    class_id: int = Field(..., description="ID numérico de la clase interna del modelo.")

class ExplainabilityData(BaseModel):
    heatmap_base64: str = Field(..., description="Solo el mapa de calor (colores) en Base64.")
    overlay_base64: str = Field(..., description="Imagen original + mapa de calor en Base64.")
    description: str = Field(default="Grad-CAM generado usando la última capa convolucional.", description="Metodología usada.")

class PerformanceData(BaseModel):
    preprocess_time_ms: float = Field(..., description="Tiempo en decodificar y aplicar CLAHE.")
    inference_time_ms: float = Field(..., description="Tiempo de inferencia del modelo YOLO.")
    explainability_time_ms: float = Field(0.0, description="Tiempo generando GradCAM (0 si está desactivado).")
    total_latency_ms: float = Field(..., description="Tiempo total de procesamiento en el backend.")
    model_used: str = Field(..., description="Nombre del archivo del modelo (.pt).")

# --- OUTPUT FINAL ---

class XRayOutput(BaseModel):
    """
    Estructura de respuesta de la API.
    """
    filename: str = Field(..., description="Nombre del archivo procesado.")
    prediction: PredictionData
    performance: PerformanceData
    # Hacemos este campo OPCIONAL. Si no piden GradCAM, esto será null en el JSON.
    explainability: Optional[ExplainabilityData] = None

# --- INPUT (Solo si decides usar Base64 en la entrada) ---

class XRayInput(BaseModel):
    """
    Usar este esquema SOLO si envías JSON raw. 
    Si usas UploadFile (recomendado), este esquema no se usa en el endpoint.
    """
    image_base64: str = Field(..., description="Cadena Base64 de la imagen.")

    # Configuración estilo Pydantic v2
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
            }
        }
    )
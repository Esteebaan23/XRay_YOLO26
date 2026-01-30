import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import mlflow
from ultralytics import YOLO
from app.config import Config
from app.schemas import (
    XRayOutput, 
    PredictionData, 
    PerformanceData, 
    ExplainabilityData
)

class XRayService:
    def __init__(self):
        print(f"🔄 Cargando modelo desde: {Config.MODEL_PATH}...")
        try:
            self.yolo = YOLO(Config.MODEL_PATH)
            self.model = self.yolo.model.to("cpu").eval()
            self.names = self.yolo.names
            self.target_layer = self._find_last_conv(self.model)
            
            self.clahe = cv2.createCLAHE(
                clipLimit=Config.CLAHE_CLIP, 
                tileGridSize=Config.CLAHE_GRID
            )
            
            if Config.MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            
            print("✅ Servicio X-Ray Inicializado correctamente en CPU.")
            
        except Exception as e:
            print(f"❌ Error crítico inicializando el servicio: {e}")
            raise e

    def _find_last_conv(self, module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No se encontró capa Conv2d para Grad-CAM")
        return last

    def _clahe_rgb(self, img_rgb):
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _unwrap_logits(self, out):
        if isinstance(out, torch.Tensor): return out
        if isinstance(out, (list, tuple)): return out[0]
        return out

    def _image_to_base64(self, img_numpy_rgb):
        img_bgr = cv2.cvtColor(img_numpy_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, image_bytes, filename: str, include_gradcam: bool = False) -> XRayOutput:
        t_start = time.time()
        
        # --- ETAPA 1: PREPROCESAMIENTO ---
        t0 = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("El archivo corrupto o formato no soportado.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_clahe = self._clahe_rgb(img_rgb)
        img_resized = cv2.resize(img_clahe, (Config.IMGSZ, Config.IMGSZ), interpolation=cv2.INTER_AREA)
        
        x = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to("cpu")
        
        t1 = time.time()
        time_preprocess = (t1 - t0) * 1000

        # --- ETAPA 2: INFERENCIA ---
        prediction_label = ""
        confidence = 0.0
        class_id = -1
        explainability_data = None
        time_explain = 0.0
        time_inference = 0.0
        
        if not include_gradcam:
            with torch.no_grad():
                out = self.model(x)
                logits = self._unwrap_logits(out)
                probs = torch.softmax(logits, dim=1)[0]
                
                class_id = int(torch.argmax(probs))
                prediction_label = self.names[class_id]
                # CORRECCIÓN: Usamos .item() para evitar el warning
                confidence = probs[class_id].item()
            
            t2 = time.time()
            time_inference = (t2 - t1) * 1000
            
        else:
            t_ex_start = time.time()
            
            activations = None
            gradients = None
            
            def fwd_hook(module, input, output):
                nonlocal activations
                activations = output
            def bwd_hook(module, grad_in, grad_out):
                nonlocal gradients
                gradients = grad_out[0]

            x.requires_grad_(True)
            h1 = self.target_layer.register_forward_hook(fwd_hook)
            h2 = self.target_layer.register_full_backward_hook(bwd_hook)
            
            self.model.zero_grad()
            out = self.model(x)
            logits = self._unwrap_logits(out)
            
            probs = torch.softmax(logits, dim=1)[0]
            class_id = int(torch.argmax(probs))
            prediction_label = self.names[class_id]
            # CORRECCIÓN: Usamos .item() aquí también
            confidence = probs[class_id].item()
            
            score = logits[:, class_id].sum()
            score.backward()
            
            h1.remove()
            h2.remove()
            
            t_inference_done = time.time()
            time_inference = (t_inference_done - t_ex_start) * 1000
            
            if activations is not None and gradients is not None:
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=False)
                cam = torch.relu(cam)[0]
                
                cam = cam.detach().numpy()
                cam -= cam.min()
                cam /= (cam.max() + 1e-6)
                
                cam_resized = cv2.resize(cam, (Config.IMGSZ, Config.IMGSZ))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                overlay = (0.6 * img_resized + 0.4 * heatmap_rgb).astype(np.uint8)
                
                explainability_data = ExplainabilityData(
                    heatmap_base64=self._image_to_base64(heatmap_rgb),
                    overlay_base64=self._image_to_base64(overlay),
                    description=f"Grad-CAM on layer {self.target_layer.__class__.__name__}"
                )
            
            t2 = time.time()
            time_explain = (t2 - t_inference_done) * 1000

        # --- ETAPA 3: FINAL ---
        total_latency = (time.time() - t_start) * 1000
        
        self._log_to_mlflow(prediction_label, confidence, total_latency, include_gradcam)

        return XRayOutput(
            filename=filename,
            prediction=PredictionData(
                label=prediction_label,
                confidence=confidence,
                class_id=class_id
            ),
            performance=PerformanceData(
                preprocess_time_ms=round(time_preprocess, 2),
                inference_time_ms=round(time_inference, 2),
                explainability_time_ms=round(time_explain, 2),
                total_latency_ms=round(total_latency, 2),
                model_used=Config.MODEL_PATH
            ),
            explainability=explainability_data
        )

    def _log_to_mlflow(self, label, conf, latency, gradcam_flag):
        if Config.MLFLOW_TRACKING_URI:
            try:
                with mlflow.start_run(run_name="inference_req", nested=True):
                    mlflow.log_metric("confidence", conf)
                    mlflow.log_metric("latency_total_ms", latency)
                    mlflow.log_param("predicted_class", label)
            except Exception:
                pass
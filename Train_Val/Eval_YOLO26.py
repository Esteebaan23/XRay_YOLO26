import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import fiftyone as fo
from fiftyone import ViewField as F


# =========================
# CONFIG
# =========================
MODEL_PATH = r"/home/STUDENTS/hel0057/Downloads/XRay/exported_models/YOLO26n-cls_clahe_best.pt"
TEST_DIR = r"/home/STUDENTS/hel0057/Downloads/XRay/test"

# Labels
FOLDER_MAP = {"anomaly": 0, "normal": 1}
ID2NAME = {0: "anomaly", 1: "normal"}

# Inferencia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 224

# CLAHE
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
_CLAHE = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

# Grad-CAM
CAM_TARGET = "pred"          # "pred" o "true"
ONLY_MISCLASSIFIED = False   # True => Grad-CAM solo errores
MAX_CAM_SAMPLES = None       # None => todos

# Outputs
OUT_DIR = Path("./eval_outputs_yolo26n")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Cambié el nombre de la carpeta de salida para reflejar que ya no es Side-by-Side
CAM_OUT_DIR = OUT_DIR / "gradcam_overlays"
CAM_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# DATASET INDEXING
# =========================
def list_images(test_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    items = []
    for folder, lab in FOLDER_MAP.items():
        folder_path = Path(test_dir) / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"No existe: {folder_path}")
        paths = []
        for e in exts:
            paths.extend(glob.glob(str(folder_path / e)))
        for p in paths:
            items.append((p, lab))
    return items


# =========================
# PREPROCESSING
# =========================
def clahe_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        out = _CLAHE.apply(rgb.astype(np.uint8))
        return np.stack([out, out, out], axis=-1)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = _CLAHE.apply(l)
    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb2


def preprocess_image(path: str, imgsz: int):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No pude leer: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = clahe_rgb(rgb)

    rgb_resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return rgb, rgb_resized, x


# =========================
# UTILS
# =========================
def unwrap_logits(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                return item
        raise TypeError(f"No Tensor found: {[type(i) for i in out]}")
    if isinstance(out, dict):
        for k in ["logits", "pred", "output"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise TypeError(f"No logits in dict. keys={list(out.keys())}")
    raise TypeError(f"Unsupported output type: {type(out)}")


def find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No encontré capa Conv2d.")
    return last


# =========================
# GRAD-CAM
# =========================
def gradcam_for_image(model_torch: nn.Module, x: torch.Tensor, target_class: int, conv_layer: nn.Module):
    activations = None
    gradients = None

    def fwd_hook(_, __, out):
        nonlocal activations
        activations = out

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = conv_layer.register_forward_hook(fwd_hook)
    h2 = conv_layer.register_full_backward_hook(bwd_hook)

    model_torch.zero_grad(set_to_none=True)

    out = model_torch(x)
    logits = unwrap_logits(out)
    score = logits[:, target_class].sum()
    score.backward()

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Error capturando hooks.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=False)
    cam = torch.relu(cam)
    cam = cam[0]

    cam -= cam.min()
    cam /= (cam.max() + 1e-6)
    return cam.detach().cpu().numpy()


def overlay_cam_on_rgb(rgb_resized: np.ndarray, cam_01: np.ndarray):
    """
    Crea el overlay: Imagen Original + Mapa de Calor.
    """
    H, W = rgb_resized.shape[:2]
    cam = cv2.resize(cam_01, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Convertir mapa de calor a RGB
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    
    # Mezcla ponderada
    overlay = (0.55 * rgb_resized + 0.45 * heat).astype(np.uint8)
    return overlay


# =========================
# MAIN
# =========================
def main():
    print(f"Device: {DEVICE}")
    items = list_images(TEST_DIR)
    print(f"Test images: {len(items)}")

    yolo = YOLO(MODEL_PATH)
    model_torch = yolo.model
    model_torch.to(DEVICE).eval()

    conv_layer = find_last_conv(model_torch)
    print("Grad-CAM target conv:", conv_layer)

    y_true, y_pred = [], []
    records = []

    # 1) Inferencia
    print("Ejecutando inferencia...")
    for path, lab in items:
        _, rgb_resized, x = preprocess_image(path, IMGSZ)
        x = x.to(DEVICE)

        with torch.no_grad():
            out = model_torch(x)
            logits = unwrap_logits(out)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

        y_true.append(lab)
        y_pred.append(pred)

        records.append({
            "path": path,
            "true": lab,
            "pred": pred,
            "conf": conf
        })

    # 2) Métricas
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\nAccuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)

    # 3) Generar Grad-CAM (Solo Overlay)
    print("\nGenerando Overlays y Dataset FiftyOne...")
    
    dataset_name = "YOLO26n_XRay_Overlays"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    
    ds = fo.Dataset(dataset_name)
    ds.add_group_field("group", default="original")

    samples_to_add = []
    cam_count = 0

    for rec in records:
        path = rec["path"]
        true_lab = rec["true"]
        pred_lab = rec["pred"]
        conf_val = rec["conf"]

        true_name = ID2NAME[int(true_lab)]
        pred_name = ID2NAME[int(pred_lab)]
        is_correct = (true_name == pred_name)

        # Determinar si generamos CAM
        generate_cam = True
        if ONLY_MISCLASSIFIED and is_correct:
            generate_cam = False
        if MAX_CAM_SAMPLES is not None and cam_count >= MAX_CAM_SAMPLES:
            generate_cam = False

        overlay_path_str = None

        if generate_cam:
            rgb_orig, rgb_resized, x = preprocess_image(path, IMGSZ)
            x = x.to(DEVICE)
            x.requires_grad_(True)

            target = int(true_lab) if CAM_TARGET == "true" else int(pred_lab)

            cam_01 = gradcam_for_image(model_torch, x, target, conv_layer)
            
            # --- CAMBIO PRINCIPAL AQUI ---
            # Solo obtenemos el overlay, NO hacemos concatenación side-by-side
            overlay = overlay_cam_on_rgb(rgb_resized, cam_01)

            stem = Path(path).stem
            cam_path = CAM_OUT_DIR / f"{stem}_cam.png"
            
            # Guardamos usando BGR (opencv standard)
            cv2.imwrite(str(cam_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            overlay_path_str = str(cam_path)
            cam_count += 1

        # --- FIFTYONE SAMPLES ---
        group = fo.Group()

        # 1. Sample ORIGINAL (Imagen limpia)
        s_orig = fo.Sample(filepath=path, group=group.element("original"))
        s_orig["true_label"] = fo.Classification(label=true_name)
        s_orig["pred_label"] = fo.Classification(label=pred_name, confidence=conf_val)
        s_orig["confidence"] = conf_val  # Campo numérico para ordenar en grid
        s_orig["is_correct"] = is_correct
        samples_to_add.append(s_orig)

        # 2. Sample GRADCAM (Solo Overlay)
        if overlay_path_str and os.path.exists(overlay_path_str):
            s_cam = fo.Sample(filepath=overlay_path_str, group=group.element("gradcam"))
            # Copiamos la info importante para verla en esta vista también
            s_cam["true_label"] = fo.Classification(label=true_name)
            s_cam["pred_label"] = fo.Classification(label=pred_name, confidence=conf_val)
            s_cam["confidence"] = conf_val
            s_cam["is_correct"] = is_correct
            samples_to_add.append(s_cam)

    ds.add_samples(samples_to_add)

    # Vistas
    ds.save_view("✅ Correct", ds.match(F("is_correct") == True))
    ds.save_view("❌ Wrong", ds.match(F("is_correct") == False))

    print(f"Overlays generados: {cam_count}")
    print(f"Dataset creado: {dataset_name}")
    
    print("Abriendo App...")
    session = fo.launch_app(ds)
    session.wait()

if __name__ == "__main__":
    main()
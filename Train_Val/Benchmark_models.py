import os
import time
import glob
import numpy as np
import pandas as pd

import cv2
from PIL import Image

import matplotlib.pyplot as plt
import mlflow
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score


# =========================
# 1) CONFIG
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(CURRENT_DIR, "exported_models")  # donde guardaste *_clahe_best.pt
MODELS_TO_BENCHMARK = {
    "YOLO26n-cls_CLAHE_best": os.path.join(MODELS_DIR, "YOLO26n-cls_clahe_best.pt"),
    "YOLO26m-cls_CLAHE_best": os.path.join(MODELS_DIR, "YOLO26m-cls_clahe_best.pt"),
    "YOLO26x-cls_CLAHE_best": os.path.join(MODELS_DIR, "YOLO26x-cls_clahe_best.pt"),
}

TEST_DATA_PATH = r"/XRay/test"  # contiene anomaly/ y normal/

FOLDER_MAP = {"anomaly": 0, "normal": 1}
TARGET_NAMES = ["anomaly", "normal"]

DEVICE = 0
IMGSZ = 224
BATCH = 64
WARMUP_ITERS = 10

# CLAHE params (¡igual que en training!)
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
_CLAHE = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

# MLflow
MLFLOW_DB_PATH = os.path.join(CURRENT_DIR, "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "Xrays_YOLO26_Benchmarking_Test_CLAHE"


# =========================
# 2) MLFLOW
# =========================
def setup_mlflow():
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"📡 MLFlow conectado a: {tracking_uri}")

def safe_start_run(run_name: str):
    mlflow.end_run()
    return mlflow.start_run(run_name=run_name)


# =========================
# 3) CLAHE ON-THE-FLY
# =========================
def clahe_rgb_numpy(rgb: np.ndarray) -> np.ndarray:
    """Aplica CLAHE en luminancia (LAB). Entrada/Salida: RGB uint8."""
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

def load_and_preprocess(path: str) -> np.ndarray:
    """Lee imagen, convierte a RGB uint8, aplica CLAHE, retorna RGB array."""
    # cv2 lee en BGR
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No pude leer imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = clahe_rgb_numpy(rgb)
    return rgb


# =========================
# 4) DATASET LOADING
# =========================
def load_test_dataset(base_path: str) -> pd.DataFrame:
    data = []
    base_path = os.path.abspath(base_path)

    for folder_name, label in FOLDER_MAP.items():
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ No se encontró la carpeta: {folder_path}")

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp')
        images = []
        for e in exts:
            images.extend(glob.glob(os.path.join(folder_path, e)))

        for img_path in images:
            data.append({"path": img_path, "true_label": label})

    df = pd.DataFrame(data)
    print(f"📂 Test cargado: {len(df)} imágenes encontradas en {base_path}")
    return df


# =========================
# 5) METRICS + PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(TARGET_NAMES))
    plt.xticks(tick_marks, TARGET_NAMES, rotation=45)
    plt.yticks(tick_marks, TARGET_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    filename = f"cm_{model_name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    return filename, cm

def calculate_specificity(cm):
    tn = cm[1, 1]
    fp = cm[1, 0]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# =========================
# 6) BENCHMARK
# =========================
def evaluate_model(model_name: str, model_path: str, df_test: pd.DataFrame):
    print(f"\n⚡ Evaluando: {model_name}")

    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado: {model_path} (saltando)")
        return

    model = YOLO(model_path)

    # Preprocesa CLAHE (on-the-fly) para todo el set
    # (si el test es enorme y no cabe en RAM, lo hacemos streaming; dime y lo ajusto)
    paths = df_test["path"].tolist()
    y_true = df_test["true_label"].tolist()

    # Warmup
    warm_imgs = []
    for p in paths[:min(WARMUP_ITERS, len(paths))]:
        warm_imgs.append(load_and_preprocess(p))
    if warm_imgs:
        _ = model.predict(warm_imgs, imgsz=IMGSZ, device=DEVICE, batch=min(BATCH, len(warm_imgs)),
                          verbose=False, save=False)

    # Inference por batches (streaming: no guardamos todo en memoria)
    y_pred = []
    t0 = time.perf_counter()

    batch_imgs = []
    for i, p in enumerate(paths):
        batch_imgs.append(load_and_preprocess(p))
        if len(batch_imgs) == BATCH or i == len(paths) - 1:
            results = model.predict(batch_imgs, imgsz=IMGSZ, device=DEVICE, batch=len(batch_imgs),
                                    verbose=False, save=False)
            for r in results:
                y_pred.append(int(r.probs.top1))
            batch_imgs = []

    t1 = time.perf_counter()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    recw = recall_score(y_true, y_pred, average="weighted")
    precw = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    cm_filename, cm = plot_confusion_matrix(y_true, y_pred, model_name)
    specificity = calculate_specificity(cm)

    total_s = (t1 - t0)
    avg_ms = (total_s / len(paths)) * 1000.0
    fps = len(paths) / total_s if total_s > 0 else 0.0

    print(f"   ➤ Accuracy: {acc:.4f}")
    print(f"   ➤ F1(w):   {f1w:.4f}")
    print(f"   ➤ Avg ms/img (end-to-end): {avg_ms:.2f} ms")
    print(f"   ➤ FPS (end-to-end):        {fps:.2f}")

    with safe_start_run(run_name=f"Benchmark_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("dataset_size", len(df_test))
        mlflow.log_param("imgsz", IMGSZ)
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("batch", BATCH)
        mlflow.log_param("clahe_clip_limit", CLIP_LIMIT)
        mlflow.log_param("clahe_tile_grid", str(TILE_GRID_SIZE))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1w)
        mlflow.log_metric("recall_weighted", recw)
        mlflow.log_metric("precision_weighted", precw)
        mlflow.log_metric("specificity", specificity)

        mlflow.log_metric("avg_ms_per_image_end_to_end", avg_ms)
        mlflow.log_metric("fps_end_to_end", fps)
        mlflow.log_metric("total_seconds_end_to_end", total_s)

        eff = acc / avg_ms if avg_ms > 0 else 0.0
        mlflow.log_metric("efficiency_score", eff)

        mlflow.log_artifact(cm_filename, artifact_path="plots")
        mlflow.log_dict(
            {
                "labels": {"0": "anomaly", "1": "normal"},
                "cm": cm.tolist()
            },
            "confusion_matrix.json"
        )

    if os.path.exists(cm_filename):
        os.remove(cm_filename)


# =========================
# 7) MAIN
# =========================
if __name__ == "__main__":
    print("🚀 Iniciando Benchmark YOLO26 (test) con CLAHE on-the-fly...")
    setup_mlflow()

    df_test = load_test_dataset(TEST_DATA_PATH)

    for name, path in MODELS_TO_BENCHMARK.items():
        evaluate_model(name, path, df_test)

    print("\n✅ Benchmark terminado. Abre MLflow UI para comparar runs.")


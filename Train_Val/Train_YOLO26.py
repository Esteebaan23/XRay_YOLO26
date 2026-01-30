import os
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import mlflow

from ultralytics import YOLO, settings
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.data.dataset import ClassificationDataset


# =========================
# CONFIG
# =========================
CURRENT_DIR = Path(__file__).resolve().parent

# Dataset root: debe contener train/ y val/ (test puede existir pero NO se usa aquí)
DATA_ROOT = Path(r"/home/STUDENTS/hel0057/Downloads/XRay")  # contiene train/, val/, test/ (opcional)
# Estructura:
# DATA_ROOT/train/normal, DATA_ROOT/train/anomaly
# DATA_ROOT/val/normal,   DATA_ROOT/val/anomaly

# Pesos pretrained (ajusta esta carpeta o pon rutas directas)
PRETRAINED_DIR = CURRENT_DIR / "pretrained_models"
MODEL_VARIANTS = [
    #("YOLO26n-cls", PRETRAINED_DIR / "yolo26n-cls.pt"),
    ("YOLO26m-cls", PRETRAINED_DIR / "yolo26m-cls.pt"),
    #("YOLO26x-cls", PRETRAINED_DIR / "yolo26x-cls.pt"),
]

# Entrenamiento
EPOCHS = 1
BATCH = 32
IMGSZ = 224
PATIENCE = 5
DEVICE = 0        # GPU index
WORKERS = 2
SEED = 0

# Salidas
RUNS_DIR = CURRENT_DIR / "runs" / "YOLO26_CLAHE_MLFLOW"
EXPORT_DIR = CURRENT_DIR / "exported_models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# CLAHE params
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)

# MLflow (SQLite local)
MLFLOW_DB_PATH = CURRENT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
EXPERIMENT_NAME = "Xrays_YOLO26_CLAHE_TrainVal"
mlflow.set_experiment(EXPERIMENT_NAME)

# Ultralytics settings (runs dir)
settings.update({"runs_dir": str(RUNS_DIR)})


# =========================
# Helpers: GPU info
# =========================
def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            idx = DEVICE if isinstance(DEVICE, int) else 0
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            return {"cuda": True, "gpu_name": name, "gpu_cc": str(cap), "gpu_mem_gb": round(mem_gb, 2)}
    except Exception:
        pass
    return {"cuda": False, "gpu_name": "cpu", "gpu_cc": None, "gpu_mem_gb": None}


# =========================
# CLAHE on-the-fly
# =========================
def clahe_on_pil(pil_img: Image.Image) -> Image.Image:
    """Aplica CLAHE en RAM. Entrada PIL (RGB). Salida PIL (RGB)."""
    img = np.array(pil_img)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Si por algún motivo viene grayscale
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
        out = clahe.apply(img.astype(np.uint8))
        return Image.fromarray(out).convert("RGB")

    # RGB -> LAB (CLAHE en canal L)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)


# =========================
# Dataset con CLAHE
# =========================
class CLAHEClassificationDataset(ClassificationDataset):
    def __getitem__(self, i):
        f, cls = self.samples[i][0], self.samples[i][1]

        try:
            im = Image.open(f).convert("RGB")
            im = clahe_on_pil(im)  # CLAHE on-the-fly
        except Exception:
            im = Image.new("RGB", (IMGSZ, IMGSZ), (0, 0, 0))

        if self.torch_transforms:
            im = self.torch_transforms(im)

        # ✅ IMPORTANTE: devolver dict
        return {
            "img": im,
            "cls": cls,
            "im_file": str(f),   # útil para logs/debug
        }


# =========================
# Trainer custom (inyecta dataset)
# =========================
class CLAHEClassificationTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return CLAHEClassificationDataset(
            root=img_path,
            args=self.args,
            augment=(mode == "train"),
            prefix=f"{mode}: "
        )


# =========================
# MLflow logging helpers
# =========================
def log_common_params():
    gpu = get_gpu_info()
    mlflow.log_params({
        "data_root": str(DATA_ROOT),
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "patience": PATIENCE,
        "device": str(DEVICE),
        "workers": WORKERS,
        "seed": SEED,
        "clahe_clip_limit": CLIP_LIMIT,
        "clahe_tile_grid": str(TILE_GRID_SIZE),
        **gpu
    })


def try_log_metrics_from_trainer(trainer):
    """
    Ultralytics puede exponer métricas en trainer.metrics.
    Loggeamos lo que exista sin romper el script.
    """
    metrics = {}
    for attr in ["metrics", "fitness", "best_fitness"]:
        if hasattr(trainer, attr):
            metrics[attr] = getattr(trainer, attr)

    # trainer.metrics suele ser dict con claves tipo "metrics/accuracy_top1" (depende versión)
    if isinstance(getattr(trainer, "metrics", None), dict):
        for k, v in trainer.metrics.items():
            if isinstance(v, (int, float, np.floating)):
                mlflow.log_metric(k, float(v))

    # best_fitness / fitness si son escalares
    if isinstance(getattr(trainer, "best_fitness", None), (int, float, np.floating)):
        mlflow.log_metric("best_fitness", float(trainer.best_fitness))
    if isinstance(getattr(trainer, "fitness", None), (int, float, np.floating)):
        mlflow.log_metric("fitness", float(trainer.fitness))


def log_artifacts_from_run_dir(save_dir: Path):
    """
    Loggea artifacts típicos de Ultralytics (si existen).
    """
    candidates = [
        save_dir / "results.csv",
        save_dir / "args.yaml",
        save_dir / "results.png",
        save_dir / "confusion_matrix.png",
        save_dir / "confusion_matrix_normalized.png",
    ]
    for p in candidates:
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="ultralytics")

    # Loggear todo el directorio weights si existe (best.pt/last.pt)
    weights_dir = save_dir / "weights"
    if weights_dir.exists():
        for w in weights_dir.glob("*.pt"):
            mlflow.log_artifact(str(w), artifact_path="weights")


# =========================
# Train one variant
# =========================
def train_variant(variant_name: str, weights_path: Path):
    # ✅ Cierra cualquier run que haya quedado abierto por un crash anterior
    mlflow.end_run()

    run_name = f"{variant_name}_CLAHE_trainval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_variant", variant_name)
            mlflow.log_param("pretrained_weights", str(weights_path))
            log_common_params()

            print(f"\n🚀 Entrenando {variant_name} (GPU={DEVICE}) | MLflow run: {run_name}")

            # ✅ Si ahora dejas que Ultralytics descargue: YOLO(f"{variant_name}.pt")
            # Si weights_path es ruta local, úsalo:
            model = YOLO(str(weights_path))

            model.train(
                data=str(DATA_ROOT),
                epochs=EPOCHS,
                batch=BATCH,
                imgsz=IMGSZ,
                patience=PATIENCE,
                device=DEVICE,
                workers=WORKERS,
                seed=SEED,
                task="classify",
                name=f"{variant_name}_clahe",
                exist_ok=True,
                trainer=CLAHEClassificationTrainer,
                verbose=True
            )

            save_dir = Path(model.trainer.save_dir)
            weights_dir = save_dir / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            if best_pt.exists():
                out_best = EXPORT_DIR / f"{variant_name}_clahe_best.pt"
                shutil.copy2(best_pt, out_best)
                mlflow.log_artifact(str(out_best), artifact_path="exported_models")

            if last_pt.exists():
                out_last = EXPORT_DIR / f"{variant_name}_clahe_last.pt"
                shutil.copy2(last_pt, out_last)
                mlflow.log_artifact(str(out_last), artifact_path="exported_models")

            try_log_metrics_from_trainer(model.trainer)
            log_artifacts_from_run_dir(save_dir)

            mlflow.set_tag("ultralytics_save_dir", str(save_dir))
            mlflow.set_tag("dataset_used", "train/val only (test excluded)")

            print(f"📁 Ultralytics run dir: {save_dir}")

    finally:
        # ✅ garantiza cierre incluso si algo explota en medio
        mlflow.end_run()


def main():
    # Chequeo mínimo: train y val deben existir
    for req in ["train", "valid"]:
        p = DATA_ROOT / req
        if not p.exists():
            raise FileNotFoundError(f"Falta carpeta requerida: {p}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for variant_name, weights_path in MODEL_VARIANTS:
        train_variant(variant_name, weights_path)

    print("\n🎯 Listo. Pesos exportados en:", EXPORT_DIR)
    print("📌 Para abrir MLflow UI:")
    print(f"mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB_PATH.name}")


if __name__ == "__main__":
    main()

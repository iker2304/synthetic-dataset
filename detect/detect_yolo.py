import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _ensure_cuda(device: str | int) -> int:
    """
    Fuerza el uso de CUDA. Devuelve el índice de dispositivo (int).
    Lanza error si CUDA no está disponible.
    """
    if isinstance(device, str):
        device = device.strip()
        if device.lower() == "cpu":
            raise RuntimeError(
                "Este script está forzado a GPU. Usa --device 0 para CUDA."
            )
        try:
            device = int(device)
        except Exception:
            raise ValueError("--device debe ser entero (p. ej., 0) o 'cpu'.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Verifica drivers NVIDIA y PyTorch con CUDA."
        )
    return int(device)


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _export_result_json(r, json_dir: Path, names: Any) -> None:
    """
    Exporta predicciones de un resultado (imagen) a JSON.
    Incluye bbox y segmentos (xy y xyn si disponibles).
    """
    json_dir.mkdir(parents=True, exist_ok=True)
    image_path = getattr(r, "path", None) or getattr(r, "save_dir", "")
    im_h, im_w = r.orig_shape[:2]

    preds: List[Dict[str, Any]] = []

    boxes = r.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        cls = boxes.cls.cpu().numpy()    # [N]
        conf = boxes.conf.cpu().numpy()  # [N]
    else:
        xyxy = np.zeros((0, 4))
        cls = np.zeros((0,))
        conf = np.zeros((0,))

    segments_xy = r.masks.xy if getattr(r, "masks", None) is not None else []
    segments_xyn = r.masks.xyn if getattr(r, "masks", None) is not None else []

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        width = x2 - x1
        height = y2 - y1
        cls_id = int(cls[i])
        item: Dict[str, Any] = {
            "class_id": cls_id,
            "class_name": _class_name(names, cls_id),
            "confidence": float(conf[i]),
            "bbox": {
                "x_min": float(x1),
                "y_min": float(y1),
                "x_max": float(x2),
                "y_max": float(y2),
                "width": float(width),
                "height": float(height),
            },
        }
        if segments_xy:
            if i < len(segments_xy):
                poly_xy = [[float(x), float(y)] for x, y in segments_xy[i].tolist()]
                item["segments"] = {
                    "xy": poly_xy,
                    "xyn": (
                        [[float(x), float(y)] for x, y in segments_xyn[i].tolist()]
                        if segments_xyn else None
                    ),
                }
        preds.append(item)

    record = {
        "image_path": str(image_path),
        "image_size": {"width": int(im_w), "height": int(im_h)},
        "predictions": preds,
    }

    out_json = json_dir / (Path(image_path).stem + ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def run_detection(
    weights: Path,
    source: Path,
    save_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | int,
    max_det: int,
    half: bool,
    save_json: bool,
    save_txt: bool,
):
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics no está instalado. Instala con: pip install ultralytics"
        )

    dev_idx = _ensure_cuda(device)
    print(f"Usando GPU CUDA índice: {dev_idx}")

    # Cargar modelo
    print(f"Cargando pesos: {weights}")
    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    print(f"Clases del modelo: {names}")

    # Directorios de salida
    project = str(save_dir)
    name = "predict"
    save_dir.mkdir(parents=True, exist_ok=True)
    json_dir = save_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar predicción
    results = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=dev_idx,
        max_det=max_det,
        half=half,
        save=True,
        save_txt=save_txt,
        project=project,
        name=name,
        verbose=True,
    )

    # Exportar JSON
    if save_json:
        for r in results:
            _export_result_json(r, json_dir, names)
        print(f"JSON guardados en: {json_dir}")

    visuals_dir = Path(project) / name
    print(f"Visualizaciones guardadas en: {visuals_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Detección YOLOv8-seg: máscaras y bounding boxes")
    p.add_argument("--weights", type=str, default=str(Path("yolov8n-seg.pt").resolve()), help="Ruta a pesos .pt")
    p.add_argument("--source", type=str, default=str((Path(__file__).resolve().parents[1] / "img").resolve()), help="Imagen/video/carpeta")
    p.add_argument("--save_dir", type=str, default=str((Path(__file__).resolve().parents[1] / "detect" / "outputs").resolve()), help="Directorio salida")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen de inferencia")
    p.add_argument("--device", type=str, default="0", help="Índice GPU (p. ej., 0)")
    p.add_argument("--max_det", type=int, default=300, help="Máximo de detecciones por imagen")
    p.add_argument("--half", action="store_true", help="Usar FP16 para reducir VRAM")
    p.add_argument("--save_json", action="store_true", help="Guardar predicciones en JSON")
    p.add_argument("--save_txt", action="store_true", help="Guardar predicciones en TXT (formato YOLO)")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_detection(
        weights=weights,
        source=source,
        save_dir=save_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        half=args.half,
        save_json=args.save_json,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
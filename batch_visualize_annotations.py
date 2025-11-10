import argparse
from pathlib import Path
import json
import shutil

import cv2
import numpy as np

# Reutilizamos funciones si el módulo está disponible; si no, definimos mínimas
try:
    from visualize_annotations import load_json, create_canvas, draw_points  # type: ignore
except Exception:
    def load_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_canvas(width: int, height: int, image_path: Path | None):
        if image_path and image_path.exists():
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            else:
                h, w = img.shape[:2]
                if (w, h) != (width, height):
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                canvas = img
        else:
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        return canvas

    def draw_points(canvas: np.ndarray, data: dict, label_type: str, radius: int, labels: bool, draw_bbox: bool = False):
        objects = data.get("objects", [])
        for obj in objects:
            name = obj.get("object_name", "obj")
            bbox = obj.get("bbox_2d", {})
            x_min = float(bbox.get("x_min", 0))
            y_min = float(bbox.get("y_min", 0))
            x_max = float(bbox.get("x_max", 0))
            y_max = float(bbox.get("y_max", 0))
            cx = float(bbox.get("center_x", (x_min + x_max) / 2.0))
            cy = float(bbox.get("center_y", (y_min + y_max) / 2.0))

            if draw_bbox:
                cv2.rectangle(canvas, (int(round(x_min)), int(round(y_min))), (int(round(x_max)), int(round(y_max))), (0, 255, 0), thickness=2)
                if labels:
                    cv2.putText(canvas, str(name), (int(round(x_min))+4, int(round(y_min))-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            points = []
            if label_type == "center":
                points.append((cx, cy, (0, 0, 255), name))  # rojo
            elif label_type == "corners":
                points.extend([
                    (x_min, y_min, (255, 0, 0), name),
                    (x_min, y_max, (255, 0, 0), name),
                    (x_max, y_min, (255, 0, 0), name),
                    (x_max, y_max, (255, 0, 0), name),
                ])
            else:  # all
                points.append((cx, cy, (0, 0, 255), name))
                points.extend([
                    (x_min, y_min, (255, 0, 0), name),
                    (x_min, y_max, (255, 0, 0), name),
                    (x_max, y_min, (255, 0, 0), name),
                    (x_max, y_max, (255, 0, 0), name),
                ])

            for (x, y, color, label) in points:
                xi = int(round(x))
                yi = int(round(y))
                cv2.circle(canvas, (xi, yi), radius, color, thickness=-1)
                if labels:
                    cv2.putText(canvas, f"{label} ({int(round(cx))},{int(round(cy))})", (xi + radius + 2, yi - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Visualiza por lotes anotaciones como puntos y bbox desde JSONs")
    p.add_argument("--annotations-dir", type=str, default=str(Path("render/annotations").resolve()), help="Carpeta con archivos JSON")
    p.add_argument("--images-dir", type=str, default=str(Path("render").resolve()), help="Carpeta con imágenes base (opcional)")
    p.add_argument("--out-dir", type=str, default=str(Path("representación").resolve()), help="Carpeta de salida para las visualizaciones")
    p.add_argument("--label-type", type=str, default="all", choices=["center", "corners", "all"], help="Puntos a visualizar por objeto")
    p.add_argument("--labels", action="store_true", default=True, help="Dibujar etiquetas con nombre del objeto y centro")
    p.add_argument("--radius", type=int, default=6, help="Radio de los puntos")
    p.add_argument("--draw-bbox", action="store_true", default=True, help="Dibujar también el rectángulo del bbox (por defecto activado)")
    p.add_argument("--overwrite-image", action="store_true", help="Escribir directamente sobre el PNG de la imagen renderizada si existe")
    return p.parse_args()


def infer_image_path(json_path: Path, images_dir: Path, data: dict) -> Path | None:
    # Preferir campo image_filename
    fname = data.get("image_filename")
    if isinstance(fname, str):
        candidate = images_dir / fname
        if candidate.exists():
            return candidate
        # Intentar alternativas de extensión
        for ext in (".png", ".jpg", ".jpeg"):
            alt = images_dir / Path(fname).with_suffix(ext)
            if alt.exists():
                return alt
    # Si no, usar el nombre base del JSON
    stem = json_path.stem.replace("_annotations", "")
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    ann_dir = Path(args.annotations_dir)
    img_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(ann_dir.glob("*.json")))
    if not json_files:
        print(f"No se encontraron JSON en: {ann_dir}")
        return

    for jf in json_files:
        try:
            data = load_json(jf)
            width = int(data.get("image_resolution", {}).get("width", 1920))
            height = int(data.get("image_resolution", {}).get("height", 1080))
            img_path = infer_image_path(jf, img_dir, data)
            canvas = create_canvas(width, height, img_path)
            canvas = draw_points(canvas, data, args.label_type, args.radius, args.labels, draw_bbox=args.draw_bbox)

            stem = jf.stem.replace("_annotations", "")
            # Si hay imagen base y se solicita sobrescribir, escribir encima del PNG original
            if args.overwrite_image and img_path is not None and img_path.exists():
                out_file = img_path
            else:
                # Guardar copia anotada en la carpeta de salida
                out_file = out_dir / f"{stem}_annotated.png"
            cv2.imwrite(str(out_file), canvas)
            print(f"Guardado (anotado): {out_file}")

            # Copiar el archivo JSON de anotación junto a la imagen generada
            try:
                json_dest = out_dir / jf.name
                shutil.copy2(str(jf), str(json_dest))
                print(f"Copiado JSON: {json_dest}")
            except Exception as copy_err:
                print(f"Error copiando JSON {jf} -> {out_dir}: {copy_err}")
        except Exception as e:
            print(f"Error procesando {jf}: {e}")


if __name__ == "__main__":
    main()
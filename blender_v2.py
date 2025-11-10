import bpy
import math
import os
import json
import mathutils
from bpy_extras.object_utils import world_to_camera_view

# Intentar usar OpenCV para dibujar; si no está disponible, hacemos fallback usando API de imágenes de Blender
try:
    import cv2
except Exception:
    cv2 = None


scene = bpy.context.scene
cam = scene.camera
render = scene.render


def compute_bbox_2d(obj, cam, scene, out_w: int, out_h: int):
    """Calcula bbox 2D del objeto en coordenadas de la imagen renderizada (out_w/out_h)."""
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    projected_points = []
    for corner in bbox_corners:
        co_ndc = world_to_camera_view(scene, cam, corner)
        if co_ndc.z < 0.0:
            continue  # Se descartan los puntos que quedan detrás de la cámara (z<0)
        # Asegurar que el punto está dentro del encuadre x e y entre [0,1]
        x_ndc = max(0.0, min(1.0, co_ndc.x))
        y_ndc = max(0.0, min(1.0, co_ndc.y))
        # Conversión a coordenadas píxeles
        x_px = x_ndc * out_w # out_w --> ancho de la imagen renderizada
        y_px = (1.0 - y_ndc) * out_h  # Se invierte el eje Y. out_h --> altura de la imagen renderizada
        projected_points.append((x_px, y_px))

    if not projected_points:
        return None

    x_coords = [p[0] for p in projected_points]
    y_coords = [p[1] for p in projected_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    cx = (x_min + width) / 2.0
    cy = (y_min + height) / 2.0

    if width <= 0.0 or height <= 0.0:
        return None

    return {
        "object_name": obj.name,
        "bbox_2d": {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
            "width": float(width),
            "height": float(height),
            "center_x": float(cx),
            "center_y": float(cy),
        },
        "visible_vertices": len(projected_points),
        "total_vertices": 8,
    }


def next_filename(base_dir: str, base_name: str) -> str:
    """Genera un nombre de archivo único evitando sobrescribir (base_name, base_name_001, ...)."""
    candidate = base_name
    idx = 0
    while os.path.exists(os.path.join(base_dir, f"{candidate}.png")):
        idx += 1
        candidate = f"{base_name}_{idx:03d}"
    return candidate


def render_with_annotations():
    # Directorios de salida (relativos al .blend)
    output_dir = "//render/"
    annotations_dir = "//render/annotations/"
    output_dir_abs = bpy.path.abspath(output_dir)
    annotations_dir_abs = bpy.path.abspath(annotations_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    os.makedirs(annotations_dir_abs, exist_ok=True)

    # Elegir nombre base y renderizar
    file_base = next_filename(output_dir_abs, "render_v2")
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(output_dir_abs, file_base)
    bpy.ops.render.render(write_still=True)

    # Ruta del PNG renderizado
    image_path = os.path.join(output_dir_abs, f"{file_base}.png")
    # Leer dimensiones exactas de la imagen renderizada (por resolution_percentage)
    out_w = int(render.resolution_x * render.resolution_percentage / 100.0)
    out_h = int(render.resolution_y * render.resolution_percentage / 100.0)

    # Calcular anotaciones para todos los MESH visibles
    annotations = []
    for o in scene.objects:
        if o.type != 'MESH':
            continue
        info = compute_bbox_2d(o, cam, scene, out_w, out_h)
        if info:
            annotations.append(info)

    # Guardar JSON (basado en cálculo actual)
    ann_data = {
        "image_filename": f"{file_base}.png",
        "image_resolution": {"width": out_w, "height": out_h},
        "objects": annotations,
        "metadata": {"total_objects": len(annotations)}
    }
    print(f"Objetos visibles detectados para anotar: {len(annotations)}")
    ann_path = os.path.join(annotations_dir_abs, f"{file_base}_annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann_data, f, indent=4)
    print(f"Anotaciones guardadas: {ann_path}")

    # Releer el JSON guardado y usar esos datos para dibujar (fuente de verdad)
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_loaded = json.load(f)
        objects_to_draw = ann_loaded.get("objects", [])
        print(f"Objetos leídos desde JSON para dibujar: {len(objects_to_draw)}")
        # Opcional: verificar resolución por si difiere
        json_res = ann_loaded.get("image_resolution", {})
        out_w = int(json_res.get("width", out_w))
        out_h = int(json_res.get("height", out_h))
    except Exception as e:
        print("No se pudo leer el JSON para dibujar:", e)
        objects_to_draw = annotations

    # Dibujar overlay y guardar imagen anotada; sobrescribir la imagen original para asegurar que 'se vea encima'
    annotated_path = image_path  # Escribir encima del PNG renderizado

    if cv2 is not None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print("No se pudo leer la imagen renderizada:", image_path)
            return
        # Ajustar tamaño si fuese necesario
        h, w = img.shape[:2]
        if (w, h) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        for obj_info in objects_to_draw:
            bb = obj_info["bbox_2d"]
            x_min = int(round(bb["x_min"]))
            y_min = int(round(bb["y_min"]))
            x_max = int(round(bb["x_max"]))
            y_max = int(round(bb["y_max"]))
            name = obj_info.get("object_name", "obj")
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
            cx = int(round(bb["center_x"]))
            cy = int(round(bb["center_y"]))
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), thickness=-1)
            cv2.putText(img, str(name), (x_min + 4, max(0, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(annotated_path, img)
        print(f"Imagen anotada guardada: {annotated_path}")
        return

    # Fallback sin OpenCV: usar API de imágenes de Blender para pintar píxeles
    try:
        img_bl = bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:
        print("No se pudo cargar la imagen para overlay:", e)
        return

    if not img_bl.has_data:
        print("La imagen no tiene datos cargados")
        return

    # Garantizar dimensiones correctas
    if img_bl.size[0] != out_w or img_bl.size[1] != out_h:
        # Crear una copia a la resolución esperada
        try:
            img_bl.scale(out_w, out_h)
        except Exception:
            pass

    w = int(img_bl.size[0])
    h = int(img_bl.size[1])
    px = list(img_bl.pixels[:])  # copia mutable (float RGBA 0..1)

    def set_pixel(x, y, rgba):
        # Convertir coordenada de imagen (origen arriba-izquierda) a buffer (origen abajo-izquierda)
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        by = (h - 1) - y
        idx = ((by * w) + x) * 4
        px[idx:idx+4] = rgba

    def draw_rect(x0, y0, x1, y1, color=(0.0, 1.0, 0.0, 1.0), thickness=2):
        x0 = max(0, min(w-1, int(round(x0))))
        y0 = max(0, min(h-1, int(round(y0))))
        x1 = max(0, min(w-1, int(round(x1))))
        y1 = max(0, min(h-1, int(round(y1))))
        for t in range(thickness):
            # top and bottom
            for x in range(x0, x1+1):
                set_pixel(x, y0+t, color)
                set_pixel(x, y1-t, color)
            # left and right
            for y in range(y0, y1+1):
                set_pixel(x0+t, y, color)
                set_pixel(x1-t, y, color)

    def draw_center(cx, cy, size=5, color=(1.0, 0.0, 0.0, 1.0)):
        cx = int(round(cx))
        cy = int(round(cy))
        half = max(1, size // 2)
        for x in range(cx-half, cx+half+1):
            for y in range(cy-half, cy+half+1):
                set_pixel(x, y, color)

    for obj_info in objects_to_draw:
        bb = obj_info["bbox_2d"]
        draw_rect(bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"], color=(0.0, 1.0, 0.0, 1.0), thickness=2)
        draw_center(bb["center_x"], bb["center_y"], size=5, color=(1.0, 0.0, 0.0, 1.0))

    # Escribir píxeles de vuelta y guardar encima del archivo original
    img_bl.pixels[:] = px
    img_bl.filepath_raw = annotated_path
    img_bl.file_format = 'PNG'
    try:
        img_bl.save()
        print(f"Imagen anotada guardada: {annotated_path}")
    except Exception as e:
        print("Error al guardar imagen anotada:", e)


if __name__ == "__main__":
    render_with_annotations()
import bpy
from bpy_extras.object_utils import world_to_camera_view
import math
import mathutils
import json
import os
import sys
import argparse
import random
import inspect

camera = bpy.data.objects['Camera']
# Distancia de órbita constante basada en la posición inicial de la cámara
initial_camera_location = camera.location.copy()
orbit_distance = initial_camera_location.length if initial_camera_location.length > 1e-8 else 5.0

# Se definen las configuraciones de renderizado (para la matriz intrínseca)
scene = bpy.context.scene
scene.camera = camera
resolution_x = scene.render.resolution_x = 1920
resolution_y = scene.render.resolution_y = 1080
scene.render.resolution_percentage = 50  # Reducir resolución a 50% para acelerar
scene.render.image_settings.file_format = 'PNG'

# Configuración del motor de render para mayor realismo (Cycles + Filmic)
def configure_cycles(scene):
    # Motor Cycles con parámetros de rendimiento realistas
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 24  # Menos muestras con denoiser activo
        scene.cycles.use_adaptive_sampling = True
        try:
            scene.cycles.adaptive_threshold = 0.05
            scene.cycles.adaptive_min_samples = 16
        except Exception:
            pass
        # Denoiser
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass
        try:
            vl = bpy.context.view_layer
            vl.cycles.use_denoising = True
            try:
                vl.cycles.denoiser = 'OPTIX'
            except Exception:
                vl.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception:
            pass
        # Rebotes muy bajos para acelerar (adecuado si no hay vidrio/volúmenes complejos)
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 4
        # Suavizado de brillos para reducir ruido
        try:
            scene.cycles.filter_glossy = 0.5
        except Exception:
            pass
        # Muestreo de luces: ignora contribuciones muy débiles
        try:
            scene.cycles.light_sampling_threshold = 0.1
        except Exception:
            pass
        # Persistencia de datos para evitar recompilaciones por fotograma
        try:
            scene.cycles.use_persistent_data = True
        except Exception:
            pass
        # Intentar usar GPU si está disponible; si falla, CPU
        if not enable_gpu_devices(scene):
            try:
                scene.cycles.device = 'CPU'
                print("[Cycles] GPU no disponible, usando CPU.")
            except Exception:
                pass
    except Exception as e:
        print(f"[Cycles] Configuración básica falló: {e}")
    # Color management: Filmic
    try:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium Contrast'
        scene.view_settings.exposure = 0.0
    except Exception:
        pass

configure_cycles(scene)

def enable_gpu_devices(scene):
    """Activa dispositivos GPU para Cycles si están disponibles. Devuelve True si se activa GPU."""
    try:
        prefs = bpy.context.preferences
        cycles_addon = prefs.addons.get('cycles', None)
        if not cycles_addon:
            print("[Cycles] Addon no encontrado en preferencias.")
            return False
        cprefs = cycles_addon.preferences
        prefer_orders = ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']
        for dev_type in prefer_orders:
            try:
                cprefs.compute_device_type = dev_type
                # Inicializa listado de dispositivos del tipo
                cprefs.get_devices_for_type(dev_type)
                any_enabled = False
                for d in cprefs.devices:
                    # Habilitar todos los dispositivos disponibles de este tipo
                    d.use = True
                    any_enabled = True or any_enabled
                if any_enabled:
                    scene.cycles.device = 'GPU'
                    print(f"[Cycles] GPU activada con {dev_type}.")
                    return True
            except Exception:
                continue
        print("[Cycles] No se encontraron dispositivos GPU soportados.")
        return False
    except Exception as e:
        print(f"[Cycles] Error al configurar GPU: {e}")
        return False

# Parámetros intrínsecos
# Función para obtener la matriz intrínseca de la cámara

def get_intrinsic_matrix(camera_object, resolution_x, resolution_y):
    """
    Obtiene la matriz intrínseca de la cámara.
    
    Parámetros:
    - camera_object: Objeto de la cámara en Blender.
    - resolution_x: Ancho de resolución de la imagen renderizada.
    - resolution_y: Alto de resolución de la imagen renderizada.
    
    Retorna:
    - Matriz intrínseca 3x3 como una lista de listas.
    """
    
    # Cálculo de la distancia focal
    f_x = camera_object.data.lens * (resolution_x / camera_object.data.sensor_width)
    f_y = camera_object.data.lens * (resolution_y / camera_object.data.sensor_height)

    # Cálcula del Principle Point
    s_x = camera_object.data.shift_x
    s_y = camera_object.data.shift_y

    c_x = resolution_x / 2 + s_x * resolution_x
    c_y = resolution_y / 2 - s_y * resolution_y

    # Matriz intrínseca
    K = [
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ]

    return K

# Función Proyección

def project_3d_to_2d(point_3d, camera_matrix, intrinsic_matrix, resolution_x, resolution_y):
    """
    Proyecta un punto 3D a coordenadas de imagen usando world_to_camera_view de Blender,
    que respeta los parámetros de la cámara, sensor fit, aspect ratio y pipeline interno.

    Retorna (x_px, y_px) si el punto cae dentro del frame; de lo contrario, None.
    """
    scene = bpy.context.scene
    camera_obj = scene.camera
    if camera_obj is None:
        return None

    co_world = mathutils.Vector(point_3d)
    co_ndc = world_to_camera_view(scene, camera_obj, co_world)

    # co_ndc.x/y en [0,1] si está dentro del frame; co_ndc.z es profundidad relativa
    if not (0.0 <= co_ndc.x <= 1.0 and 0.0 <= co_ndc.y <= 1.0):
        return None

    pixel_x = float(co_ndc.x * resolution_x)
    pixel_y = float((1.0 - co_ndc.y) * resolution_y)  # convertir a origen en esquina superior izquierda

    return (pixel_x, pixel_y)

def get_object_bounding_box_2d(obj, camera_matrix, intrinsic_matrix, resolution_x, resolution_y):
    """
    Calcula el bbox 2D de un objeto en la imagen.
    Usa vértices del mesh si están disponibles; si no, hace fallback a las esquinas de bound_box.
    """

    if obj.type != 'MESH':
        return None  # Solo se puede calcular para objetos de tipo MESH

    mesh = obj.data
    world_vertices = []

    # Vértices del mesh si existen; si no, las esquinas del bound_box
    try:
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            for vertex in mesh.vertices:
                world_vertices.append(obj.matrix_world @ vertex.co)
        else:
            for corner in obj.bound_box:
                world_vertices.append(obj.matrix_world @ mathutils.Vector(corner))
    except Exception:
        # Fallback adicional por seguridad
        for corner in obj.bound_box:
            world_vertices.append(obj.matrix_world @ mathutils.Vector(corner))

    # Proyectar todos los vértices a coordenadas 2D
    projected_points = []
    for vertex in world_vertices:
        projected = project_3d_to_2d(vertex, camera_matrix, intrinsic_matrix, resolution_x, resolution_y)
        if projected:
            projected_points.append(projected)

    # Si no hay puntos proyectados visibles, el objeto no está en la vista
    if not projected_points:
        return None

    # Calcular el bbox 2D
    x_coords = [p[0] for p in projected_points]
    y_coords = [p[1] for p in projected_points]

    min_x = max(0, min(x_coords))  # Coordenada X más pequeña (más a la izquierda del objeto)
    max_x = min(resolution_x, max(x_coords))  # Coordenada X más grande (más a la derecha del objeto)
    min_y = max(0, min(y_coords))
    max_y = min(resolution_y, max(y_coords))

    # Verificar que el bbox tenga área válida
    if max_x <= min_x or max_y <= min_y:
        return None  # Bbox inválido

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    return {
        "object_name": obj.name,
        "bbox_2d": {
            "x_min": min_x,
            "y_min": min_y,
            "x_max": max_x,
            "y_max": max_y,
            "width": bbox_width,
            "height": bbox_height,
            "center_x": min_x + bbox_width / 2,
            "center_y": min_y + bbox_height / 2,
        },
        "visible_vertices": len(projected_points),
        "total_vertices": len(world_vertices),
    }

def get_all_objects_annotations(camera_matrix, intrinsic_matrix, resolution_x, resolution_y):
    """
    Obtiene las anotaciones de todos los objetos visibles en la imagen.

    Compatibilidad: si la función get_object_bounding_box_2d en el entorno actual
    tiene una firma antigua (4 parámetros), se llama con 4 argumentos. Si tiene la
    firma nueva (5 parámetros), se llama con 5.
    """
    annotations = []
    # Detectar número de parámetros de la función en tiempo de ejecución
    try:
        param_count = len(inspect.signature(get_object_bounding_box_2d).parameters)
    except Exception:
        param_count = 5  # Asumir firma nueva si introspección falla

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if param_count >= 5:
                bbox_2d = get_object_bounding_box_2d(obj, camera_matrix, intrinsic_matrix, resolution_x, resolution_y)
            else:
                bbox_2d = get_object_bounding_box_2d(obj, camera_matrix, resolution_x, resolution_y)
            if bbox_2d:
                annotations.append(bbox_2d)
    return annotations

# Parámetros extrínsecos (cambia según la vista)
def set_camera_pose(camera_object, location, rotation_euler):
    camera_object.location = location
    # Se usa cuaterniones para evitar gimbal lock
    camera_object.rotation_mode = 'QUATERNION'
    if isinstance(rotation_euler, mathutils.Quaternion):
        camera_object.rotation_quaternion = rotation_euler.normalized()
    elif hasattr(rotation_euler, 'to_quaternion'):
        camera_object.rotation_quaternion = rotation_euler.to_quaternion()
    else:
        camera_object.rotation_quaternion = mathutils.Euler(rotation_euler, 'XYZ').to_quaternion()
    # (X, Y, Z) en radianes cuando se provee Euler; si es cuaternión, ya es orientación

# Rango Z de la escena (en coordenadas de mundo)
def get_scene_z_bounds():
    z_min, z_max = None, None
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for v in obj.bound_box:
            world_v = obj.matrix_world @ mathutils.Vector(v)
            z = world_v.z
            z_min = z if z_min is None else min(z_min, z)
            z_max = z if z_max is None else max(z_max, z)
    return z_min, z_max

# Selección del objeto principal y utilidades de encuadre
def _bound_box_volume(obj):
    try:
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    except Exception:
        return 0.0, 0.0, 0.0

def find_primary_mesh_object(preferred_names=("Cube", "cube")):
    """Devuelve el objeto MESH principal: primero por nombre preferido; si no, el de mayor volumen de bound_box."""
    # Buscar por nombre preferido
    for name in preferred_names:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == 'MESH':
            return obj
    # Si no existe, escoger el mayor por volumen aproximado
    best = None
    best_vol = -1.0
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        dx, dy, dz = _bound_box_volume(obj)
        vol = max(dx, 0.0) * max(dy, 0.0) * max(dz, 0.0)
        if vol > best_vol:
            best = obj
            best_vol = vol
    return best

def is_object_in_frame(obj, camera_matrix, intrinsic_matrix, resolution_x, resolution_y):
    """Retorna True si el bbox 2D del objeto existe (alguna parte visible en el frame)."""
    bbox = get_object_bounding_box_2d(obj, camera_matrix, intrinsic_matrix, resolution_x, resolution_y)
    return bbox is not None

def is_object_fully_in_frame(obj, resolution_x, resolution_y, margin_pixels=0):
    """Retorna True si TODOS los vértices (o esquinas del bound_box) proyectan dentro del frame.
    Usa world_to_camera_view directamente para comprobar cada punto y aplica un margen opcional."""
    if obj.type != 'MESH':
        return False
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return False

    # Recoger puntos a proyectar
    points = []
    mesh = obj.data
    try:
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            for v in mesh.vertices:
                points.append(obj.matrix_world @ v.co)
        else:
            for c in obj.bound_box:
                points.append(obj.matrix_world @ mathutils.Vector(c))
    except Exception:
        for c in obj.bound_box:
            points.append(obj.matrix_world @ mathutils.Vector(c))

    # Proyectar y validar que todos estén dentro
    pixel_pts = []
    for p in points:
        ndc = world_to_camera_view(scene, cam, p)
        # Debe estar frente a la cámara y dentro del frustum en X/Y
        if not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0) or ndc.z < 0.0:
            return False
        px = float(ndc.x * resolution_x)
        py = float((1.0 - ndc.y) * resolution_y)
        pixel_pts.append((px, py))

    # Calcular bbox en píxeles y aplicar margen
    xs = [p[0] for p in pixel_pts]
    ys = [p[1] for p in pixel_pts]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if min_x < margin_pixels or min_y < margin_pixels:
        return False
    if max_x > (resolution_x - margin_pixels) or max_y > (resolution_y - margin_pixels):
        return False

    return True

# Utilidades de iluminación
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def kelvin_to_rgb(kelvin):
    """Convierte temperatura de color (K) a RGB normalizado [0,1]. Rango útil: 1000–12000K."""
    k = kelvin / 100.0
    # Rojo
    if k <= 66:
        r = 1.0
    else:
        r = 329.698727446 * ((k - 60.0) ** -0.1332047592) / 255.0
    # Verde
    if k <= 66:
        g = (99.4708025861 * math.log(max(k, 1.0)) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((k - 60.0) ** -0.0755148492) / 255.0
    # Azul
    if k >= 66:
        b = 1.0
    elif k <= 19:
        b = 0.0
    else:
        b = (138.5177312231 * math.log(max(k - 10.0, 1.0)) - 305.0447927307) / 255.0
    return (_clamp(r), _clamp(g), _clamp(b))

def ensure_light(name, light_type='AREA'):
    """Asegura que exista un objeto luz con nombre y tipo dados; lo crea si no existe."""
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'LIGHT':
        # Cambiar tipo si es necesario
        obj.data.type = light_type
        return obj
    # Crear nueva luz
    light_data = bpy.data.lights.get(name) or bpy.data.lights.new(name=name, type=light_type)
    obj = bpy.data.objects.get(name)
    if not obj:
        obj = bpy.data.objects.new(name, light_data)
        bpy.context.scene.collection.objects.link(obj)
    else:
        obj.data = light_data
    obj.data.type = light_type
    obj.data.use_shadow = True
    return obj

def randomize_world_lighting(scene, rng):
    world = scene.world or (bpy.data.worlds[0] if bpy.data.worlds else None)
    if not world:
        return
    world.use_nodes = True
    nt = world.node_tree
    bg = None
    for n in nt.nodes:
        if getattr(n, 'type', None) == 'BACKGROUND':
            bg = n
            break
    if not bg:
        bg = nt.nodes.new('ShaderNodeBackground')
        out = next((n for n in nt.nodes if getattr(n, 'type', None) == 'OUTPUT_WORLD'), None)
        if out:
            nt.links.new(bg.outputs[0], out.inputs[0])
    # Temperatura y fuerza ambiental
    kelvin = rng.randint(2700, 6500)
    rgb = kelvin_to_rgb(kelvin)
    strength = rng.uniform(0.0, 0.5)
    bg.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bg.inputs[1].default_value = strength

__HDRI_CACHE__ = {"image": None, "path": None}

def randomize_hdri_environment(scene, rng):
    """Configura el World con una textura HDRI si existe en //hdris/.
    Cachea la imagen para no recargar cada fotograma. Devuelve True si se aplicó."""
    hdris_dir = bpy.path.abspath("//hdris/")
    if not os.path.isdir(hdris_dir):
        return False
    files = [f for f in os.listdir(hdris_dir) if f.lower().endswith((".hdr", ".exr"))]
    if not files:
        return False
    world = scene.world or (bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("World"))
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    # Nodos
    bg = next((n for n in nt.nodes if getattr(n, 'type', None) == 'BACKGROUND'), None) or nt.nodes.new('ShaderNodeBackground')
    out = next((n for n in nt.nodes if getattr(n, 'type', None) == 'OUTPUT_WORLD'), None) or nt.nodes.new('ShaderNodeOutputWorld')
    env = next((n for n in nt.nodes if getattr(n, 'type', None) == 'TEX_ENVIRONMENT'), None) or nt.nodes.new('ShaderNodeTexEnvironment')
    mapping = next((n for n in nt.nodes if getattr(n, 'type', None) == 'MAPPING'), None) or nt.nodes.new('ShaderNodeMapping')
    texcoord = next((n for n in nt.nodes if getattr(n, 'type', None) == 'TEX_COORD'), None) or nt.nodes.new('ShaderNodeTexCoord')

    # Cargar imagen HDRI una vez (cache)
    if __HDRI_CACHE__["image"] is None:
        img_path = os.path.join(hdris_dir, rng.choice(files))
        try:
            img = bpy.data.images.load(img_path, check_existing=True)
            __HDRI_CACHE__["image"] = img
            __HDRI_CACHE__["path"] = img_path
        except Exception:
            return False
    env.image = __HDRI_CACHE__["image"]

    # Rotación aleatoria alrededor del eje Z
    try:
        mapping.inputs['Rotation'].default_value[2] = rng.uniform(0.0, 2 * math.pi)
    except Exception:
        pass

    # Enlazar nodos
    try:
        nt.links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])
    except Exception:
        pass
    try:
        nt.links.new(mapping.outputs['Vector'], env.inputs['Vector'])
    except Exception:
        pass
    try:
        nt.links.new(env.outputs['Color'], bg.inputs['Color'])
        nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    except Exception:
        pass

    # Fuerza ambiental moderada
    bg.inputs[1].default_value = rng.uniform(0.2, 1.5)
    return True

def randomize_lighting(scene, origin=(0.0, 0.0, 0.0), seed=None):
    """Randomiza iluminación realista: luz ambiental, key, fill y rim/sun."""
    rng = random.Random(seed) if seed is not None else random
    # Mundo: intenta HDRI, si no, fondo simple
    if not randomize_hdri_environment(scene, rng):
        randomize_world_lighting(scene, rng)

    # Key light: Área suave, cerca del objeto
    key = ensure_light('KeyLight', 'AREA')
    key.data.energy = rng.uniform(800.0, 2000.0)
    key.data.color = kelvin_to_rgb(rng.randint(3000, 5500))
    key.data.size = rng.uniform(1.0, 3.0)
    dist_k = rng.uniform(3.0, 6.0)
    az_k = rng.uniform(0.0, 2 * math.pi)
    el_k = rng.uniform(0.2, 0.8)  # elevación relativa
    key.location = (
        origin[0] + dist_k * math.cos(az_k),
        origin[1] + dist_k * math.sin(az_k),
        origin[2] + dist_k * el_k,
    )

    # Fill light: Punto más débil, opuesto al key
    fill = ensure_light('FillLight', 'POINT')
    fill.data.energy = rng.uniform(200.0, 800.0)
    fill.data.color = kelvin_to_rgb(rng.randint(3500, 6500))
    dist_f = rng.uniform(2.0, 5.0)
    az_f = az_k + rng.uniform(math.pi/2, math.pi)  # aproximadamente opuesta
    el_f = rng.uniform(0.1, 0.6)
    fill.location = (
        origin[0] + dist_f * math.cos(az_f),
        origin[1] + dist_f * math.sin(az_f),
        origin[2] + dist_f * el_f,
    )

    # Rim/Sun: luz direccional para contorno
    rim = ensure_light('RimLight', 'SUN')
    rim.data.energy = rng.uniform(1.0, 4.0)
    rim.data.color = kelvin_to_rgb(rng.randint(5000, 6500))
    rim.data.angle = math.radians(rng.uniform(0.1, 3.0))  # tamaño aparente del sol
    # Orientar el sol hacia el origen desde una dirección aleatoria
    dist_r = rng.uniform(6.0, 10.0)
    az_r = rng.uniform(0.0, 2 * math.pi)
    el_r = rng.uniform(0.3, 0.9)
    rim.location = (
        origin[0] + dist_r * math.cos(az_r),
        origin[1] + dist_r * math.sin(az_r),
        origin[2] + dist_r * el_r,
    )
    # Apuntar al origen
    dir_vec = mathutils.Vector(origin) - rim.location
    if dir_vec.length > 1e-8:
        rim.rotation_quaternion = dir_vec.to_track_quat('-Z', 'Y')

def randomize_camera_dof(camera_object, origin=(0.0, 0.0, 0.0), seed=None):
    """Randomiza profundidad de campo (DOF) con valores fotográficos plausibles."""
    rng = random.Random(seed) if seed is not None else random
    camd = camera_object.data
    try:
        camd.dof.use_dof = True
        focus_distance = (mathutils.Vector(origin) - camera_object.location).length
        camd.dof.focus_distance = max(focus_distance + rng.uniform(-0.3, 0.3), 0.01)
        if hasattr(camd.dof, 'aperture_fstop'):
            camd.dof.aperture_fstop = rng.uniform(8.0, 16.0)  # DOF suave para menos ruido
        elif hasattr(camd.dof, 'aperture_size'):
            camd.dof.aperture_size = rng.uniform(0.1, 0.5)
    except Exception:
        pass

# Loop para generar el dataset
output_dir = "//render_output/"  # Path relativo al archivo .blend
annotations_dir = "//render_output/annotations/"

# Resolver rutas absolutas y garantizar que existan los directorios
output_dir_abs = bpy.path.abspath(output_dir)
annotations_dir_abs = bpy.path.abspath(annotations_dir)
os.makedirs(output_dir_abs, exist_ok=True)
os.makedirs(annotations_dir_abs, exist_ok=True)

num_samples = 100

# Ejecutar: blender -b archivo.blend -P blender.py -- --num_samples 1000
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
parser = argparse.ArgumentParser(description="Generar del dataset desde Blender")
parser.add_argument("--num_samples", "--num_images", dest="num_images", type=int, default=None, help="Número de imágenes a generar")
args = parser.parse_args(argv)

if args.num_images is not None:
    num_samples = args.num_images
    print(f"Generando {num_samples} imágenes")

# Obtener la matriz intrínseca una vez si los parámetros no cambian
intrinsic_matrix = get_intrinsic_matrix(camera, resolution_x, resolution_y)
print("Intrinsic Matrix K:")
print(intrinsic_matrix)

# Calcular rango Z de escena para muestreo de la cámara
z_min, z_max = get_scene_z_bounds()
if z_min is None or z_max is None:
    z_min, z_max = 1.0, 4.0  

# Ampliar el rango de muestreo en Z (margen adicional)
z_expand = 1.0  # Ajusta este valor para aumentar/disminuir el rango
z_min_expanded = z_min - z_expand
z_max_expanded = z_max + z_expand

primary_obj = find_primary_mesh_object()
if primary_obj is None:
    print("[Aviso] No se encontró objeto MESH principal; se usará el origen como objetivo.")

for frame_index in range(num_samples):
    max_attempts = 100
    attempt = 0
    visible = False
    
    base_distance = orbit_distance
    while attempt < max_attempts and not visible:
        attempt += 1
        # Aumentar distancia si en intentos anteriores no cabe completo
        dist_current = min(base_distance * (1.0 + 0.02 * attempt), base_distance * 3.0)
        # Trayectoria circular simple alrededor del objetivo
        angle_z = random.uniform(0.0, 2 * math.pi)
        # Elegir Z respetando la distancia al objetivo (clamp a [-dist_current, dist_current])
        z_lower = max(z_min_expanded, -dist_current)
        z_upper = min(z_max_expanded, dist_current)
        z = random.uniform(z_lower, z_upper) if z_lower <= z_upper else 0.0
        # Ajustar radio en XY para mantener ||cam_location - target|| ~= dist_current
        r_xy_sq = max(dist_current * dist_current - z * z, 0.0)
        r_xy = math.sqrt(r_xy_sq)

        target_location = tuple(primary_obj.location) if primary_obj else (0.0, 0.0, 0.0)
        cam_location = (
            target_location[0] + r_xy * math.cos(angle_z),
            target_location[1] + r_xy * math.sin(angle_z),
            target_location[2] + z,
        )

        # Calcular la rotación para apuntar al objetivo (look-at)
        direction = mathutils.Vector(target_location) - mathutils.Vector(cam_location)
        rotation_quat = direction.to_track_quat('-Z', 'Y') if direction.length > 1e-8 else mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        set_camera_pose(camera, cam_location, rotation_quat)

        # Actualizar y calcular matriz extrínseca
        bpy.context.view_layer.update()
        world_to_camera_matrix = camera.matrix_world.inverted()

        # Comprobar visibilidad del objeto principal
        if primary_obj:
            visible = is_object_fully_in_frame(primary_obj, resolution_x, resolution_y, margin_pixels=0)
        else:
            visible = True  # Si no hay objeto, no forzar visibilidad

    if not visible:
        print(f"[Aviso] No se logró encuadrar el objeto en {max_attempts} intentos; se continúa con la última vista.")

    print(f"\n--- View {frame_index} ---")
    print(f"Camera location: {camera.location}")
    print(f"Camera rotation quaternion: {camera.rotation_quaternion}")
    print(f"World to Camera Matrix: ")
    print(world_to_camera_matrix)

    # Renderizar la imagen
    file_name = f"render_{frame_index:03d}"
    # Asegurar ruta absoluta para el render y que exista el directorio
    scene.render.filepath = os.path.join(output_dir_abs, file_name)
    # Randomizar DOF de cámara y la iluminación por frame, centrados en el objeto
    randomize_camera_dof(camera, origin=target_location, seed=frame_index)
    randomize_lighting(scene, origin=target_location, seed=frame_index)
    bpy.ops.render.render(write_still=True)

    # Guardar anotaciones
    # Obtener anotaciones de todos los objetos visibles
    object_annotations = get_all_objects_annotations(world_to_camera_matrix, intrinsic_matrix, resolution_x, resolution_y)

    annotation_data = {
            "frame": f"{frame_index}",
            "image_filename": f"{file_name}.png",
            "image_resolution": {
                "width": scene.render.resolution_x,
                "height": scene.render.resolution_y
            },
            "camera_parameters": {
                "intrinsic_matrix": [list(row) for row in intrinsic_matrix],
                "extrinsic_matrix": [list(row) for row in world_to_camera_matrix],
                "camera_location": list(camera.location),
                "camera_rotation_euler": list(camera.rotation_euler),
                "camera_rotation_quaternion": list(camera.rotation_quaternion),
                "focal_length": camera.data.lens,
                "sensor_width": camera.data.sensor_width,
                "sensor_height": camera.data.sensor_height
            },
            "objects": object_annotations,
            "metadata": {
                "total_objects": len(object_annotations),
                "render_settings": {
                    "resolution_percentage": scene.render.resolution_percentage,
                    "file_format": scene.render.image_settings.file_format
                }
            }
        }
    
    # Guardar anotaciones en archivo JSON
    annotation_filename = f"{file_name}_annotations.json"
    annotation_filepath = os.path.join(annotations_dir_abs, annotation_filename)
    os.makedirs(os.path.dirname(annotation_filepath), exist_ok=True)
    
    with open(annotation_filepath, 'w') as f:
        json.dump(annotation_data, f, indent=4)
    
    print(f"Anotaciones guardadas: {annotation_filename}")
    print(f"Objetos detectados: {len(object_annotations)}")
    for obj_info in object_annotations:
        bbox = obj_info["bbox_2d"]
        print(f"  - {obj_info['object_name']}: bbox({bbox['x_min']:.1f}, {bbox['y_min']:.1f}, {bbox['width']:.1f}, {bbox['height']:.1f})")

print("\nDataset generation complete!")
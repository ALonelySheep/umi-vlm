from omni.isaac.kit import SimulationApp
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
import logging
import sys
from flask import Flask, request, jsonify, send_file
from queue import Queue, Empty
import io
from PIL import Image
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IsaacSimFlaskServer")

# Initialize Isaac Sim (Qt UI must be on main thread)
logger.info("Initializing Isaac Sim...")
simulation_app = SimulationApp({"headless": False})

# IMPORTANT: Import Isaac Sim modules **after** initializing SimulationApp
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.physx as _physx

# Globals
world = None
objects = {}
COMMAND_QUEUE = Queue()

# Scene constants
DESK_WIDTH     = 2.0
DESK_DEPTH     = 1.0
DESK_HEIGHT    = 0.75
DESK_THICKNESS = 0.05
PORT = 5000

def setup_scene():
    """Create the scene with a desk, lights, and ground plane."""
    stage = get_current_stage()

    # Desk
    desk_path = "/World/Desk"
    desk_pos = (0, 0, DESK_HEIGHT - DESK_THICKNESS/2)
    create_prim(
        prim_path=desk_path,
        prim_type="Cube",
        position=desk_pos,
        scale=(DESK_WIDTH/2, DESK_DEPTH/2, DESK_THICKNESS/2)
    )
    prim = stage.GetPrimAtPath(desk_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(False)
    UsdGeom.Cube(prim).CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.5, 0.35)])

    # Dome light + sphere light
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(3000)
    light = UsdLux.SphereLight.Define(stage, "/World/Light_Top")
    light.AddTranslateOp().Set((0, 0, 5))
    light.CreateIntensityAttr(10000)
    light.CreateRadiusAttr(0.1)

def add_box(box_id, position, color, size):
    """Adds a box; returns (success, message)."""
    try:
        box_name = f"box_{box_id}"
        pos_np   = np.array(position, dtype=np.float32)
        scale_np = np.array([size, size, size], dtype=np.float32)
        color_np = np.array(color, dtype=np.float32)

        box_path = f"/World/Box_{box_id}"
        world.scene.add(
            DynamicCuboid(
                prim_path=box_path,
                name=box_name,
                position=pos_np,
                color=color_np,
                scale=scale_np
            )
        )
        objects[box_name] = {
            "name": box_name,
            "position": position,
            "color": color,
            "size": size
        }
        return True, f"Added box {box_name}"
    except Exception as e:
        logger.error(f"add_box error: {e}", exc_info=True)
        return False, str(e)

def move_object(obj_name, target_pos):
    """Moves a box smoothly; returns (success, message)."""
    try:
        if obj_name not in objects:
            return False, f"No such object: {obj_name}"

        obj = world.scene.get_object(obj_name)
        if obj is None:
            return False, f"Object not found in scene: {obj_name}"

        current_pos = np.array(obj.get_world_pose()[0], dtype=np.float32)
        target_np   = np.array(target_pos, dtype=np.float32)
        steps = 100
        for i in range(steps+1):
            t = i/steps
            interp = current_pos*(1-t) + target_np*t
            obj.set_world_pose(interp)
            # **Do NOT step/render here** — we'll do that in the main loop
        objects[obj_name]["position"] = target_pos
        return True, f"Moved {obj_name}"
    except Exception as e:
        logger.error(f"move_object error: {e}", exc_info=True)
        return False, str(e)

def move_object_on_top(source_name, target_name):
    """Moves source object on top of target object; returns (success, message)."""
    try:
        if source_name not in objects:
            return False, f"No such source object: {source_name}"
        if target_name not in objects:
            return False, f"No such target object: {target_name}"
            
        source_obj = world.scene.get_object(source_name)
        target_obj = world.scene.get_object(target_name)
        if source_obj is None:
            return False, f"Source object not found in scene: {source_name}"
        if target_obj is None:
            return False, f"Target object not found in scene: {target_name}"
        
        # Get target position
        target_pos = np.array(target_obj.get_world_pose()[0], dtype=np.float32)
        target_info = objects[target_name]
        
        # Calculate the height adjustment
        # Z position = target Z + target size + source size + small gap
        source_size = objects[source_name]["size"] 
        target_size = target_info["size"]
        # Calculate height to position source directly above target with enough clearance
        z_position = target_pos[2] + (target_size + source_size) + 0.01
        
        # Create new position: same X,Y as target, new Z height
        new_position = [target_pos[0], target_pos[1], z_position]
        
        # Move source object to new position
        return move_object(source_name, new_position)
        
    except Exception as e:
        logger.error(f"move_object_on_top error: {e}", exc_info=True)
        return False, str(e)

def add_initial_objects():
    """Drop in a few colored boxes to start."""
    colors = {"red": (1,0,0), "green": (0,1,0), "blue": (0,0,1)}
    positions = [
        (-0.5, 0, DESK_HEIGHT + 0.1),
        ( 0.5, 0, DESK_HEIGHT + 0.1),
        (   0, 0.4, DESK_HEIGHT + 0.1),
    ]
    for i, pos in enumerate(positions):
        col = list(colors.values())[i]
        add_box(i, pos, col, 0.1)

def capture_camera(view_name):
    """Capture an image from a specific camera viewpoint"""
    try:
        # Camera configurations for different views
        camera_configs = {
            "top": {"position": [0, 0, 7.0], "orientation": [0, 90, 0]},
            "front": {"position": [-3.0, 0, 0.75], "orientation": [0, 0, 0]},
            "side": {"position": [0, -3.0, 0.75], "orientation": [0, 0, 90]},
            "iso": {"position": [-2.5, 2.5, 4.0], "orientation": [0, 45, -45]}
        }
        
        if view_name not in camera_configs:
            return False, f"Invalid view name. Choose from: {list(camera_configs.keys())}"
        
        config = camera_configs[view_name]
        cam_path = f"/World/camera_{view_name}"
        camera = Camera(
            prim_path=cam_path,
            position=np.array(config["position"]),
            frequency=20,
            resolution=(1024, 1024),
            orientation=rot_utils.euler_angles_to_quats(np.array(config["orientation"]), degrees=True),
        )
        
        world.scene.add(camera)
        camera.initialize()
        
        # Allow camera to settle
        for _ in range(30):
            world.step(render=True)
        
        # Get the image
        rgba = camera.get_rgba()
        rgb = rgba[:, :, :3]
        
        # Convert to PIL Image and then to bytes
        pil_image = Image.fromarray((rgb * 1).astype(np.uint8))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Cleanup
        world.scene.remove_object(camera.name)
        
        return True, img_byte_arr
    except Exception as e:
        logger.error(f"Camera capture error: {e}", exc_info=True)
        return False, str(e)

from PIL import Image
import os

# def ray_cast(view_name, x, y):
#     """Perform a ray cast from specified camera view at pixel coordinates (x, y),
#        and save a PNG of the camera’s view for sanity checks."""
#     try:
#         # 1) Same camera configs as capture_camera()
#         camera_configs = {
#             "top":   {"position": [0,   0, 7.0],    "orientation": [0, 90,  0]},
#             "front": {"position": [-3.0,0, 0.75],   "orientation": [0,  0,  0]},
#             "side":  {"position": [0,  -3.0,0.75],  "orientation": [0,  0, 90]},
#             "iso":   {"position": [-2.5,2.5,4.0],   "orientation": [0, 45, -45]},
#         }
#         if view_name not in camera_configs:
#             return False, f"Invalid view name. Choose from: {list(camera_configs.keys())}"

#         config   = camera_configs[view_name]
#         cam_path = f"/World/camera_{view_name}_raycast"

#         # 2) Spawn & initialize the Camera sensor
#         camera = Camera(
#             prim_path   = cam_path,
#             position    = np.array(config["position"], dtype=np.float32),
#             orientation = rot_utils.euler_angles_to_quats(
#                               np.array(config["orientation"], dtype=np.float32),
#                               degrees=True
#                           ),
#             frequency   = 20,
#             resolution  = (1024, 1024),
#         )
#         world.scene.add(camera)
#         camera.initialize()

#         # 3) Let the sim settle
#         for _ in range(5):
#             world.step(render=True)

#         # ——— SANITY CHECK: capture & save the view ———
#         rgba = camera.get_rgba()                   # (H, W, 4) float in [0,1]
#         rgb  = (rgba[:, :, :3]).astype('uint8')
#         img  = Image.fromarray(rgb)
#         save_name = f"raycast_{view_name}.png"
#         img.save(save_name)
#         # now you’ll find e.g. “raycast_iso.png” in your working dir

#         # 4) Compute world-space origin & forward direction
#         # from pxr import UsdGeom, Gf
#         # from omni.isaac.core.utils.stage import get_current_stage
#         # stage = get_current_stage()
#         # prim  = stage.GetPrimAtPath(cam_path)
#         # xf_cache = UsdGeom.XformCache()
#         # xf = xf_cache.GetLocalToWorldTransform(prim)
#         # origin  = xf.ExtractTranslation()
#         # rot     = xf.ExtractRotationMatrix()
#         # forward = rot * Gf.Vec3d(0, 0, -1)

#         # # 5) Fire PhysX raycast_closest
#         # import carb
#         # from omni.physx import get_physx_scene_query_interface
#         # qi = get_physx_scene_query_interface()
#         # hit_info = qi.raycast_closest(
#         #     carb.Float3(origin[0], origin[1], origin[2]),
#         #     carb.Float3(forward[0], forward[1], forward[2]),
#         #     1000.0
#         # )
#         # after camera.initialize() and world.step(render=True) …
#         from omni.kit.viewport.utility import get_active_viewport
#         from pxr import Gf
#         import carb
#         from omni.physx import get_physx_scene_query_interface

#         # 1) NDC conversion
#         width, height = 1024, 1024
#         ndc_x = (2.0 * x) / width  - 1.0
#         ndc_y = 1.0 - (2.0 * y) / height

#         # 2) Unproject via viewport
#         vp  = get_active_viewport()
#         mat = vp.ndc_to_world                         # Gf.Matrix4d  [oai_citation_attribution:1‡NVIDIA Omniverse Documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.docs/latest/viewport_api.html?utm_source=chatgpt.com)

#         near_h = Gf.Vec4d(ndc_x, ndc_y, -1.0, 1.0)
#         far_h  = Gf.Vec4d(ndc_x, ndc_y,  1.0, 1.0)

#         nw = mat * near_h
#         fw = mat * far_h

#         near_pt = Gf.Vec3d(nw[0]/nw[3], nw[1]/nw[3], nw[2]/nw[3])
#         far_pt  = Gf.Vec3d(fw[0]/fw[3], fw[1]/fw[3], fw[2]/fw[3])

#         # 3) Build ray
#         origin    = carb.Float3(near_pt[0], near_pt[1], near_pt[2])
#         direction = (far_pt - near_pt).GetNormalized()

#         # 4) PhysX raycast
#         qi  = get_physx_scene_query_interface()
#         hit_info = qi.raycast_closest(origin, 
#                                 carb.Float3(direction[0], direction[1], direction[2]), 
#                          1000.0)
#         # hit_info = qi.raycast_closest(origin, forward, 1000.0)
#         print(f"Raw PhysX hit_info: {hit_info}")    

#         # 6) Cleanup
#         world.scene.remove_object(camera.name)

#         # 7) Parse and return
#         if hit_info.get("hit", False):
#             hit_path = hit_info.get("bodyPrimPath",
#                         hit_info.get("colliderPrimPath", ""))
#             info = {
#                 "sanity_image": save_name,
#                 "path":        hit_path,
#                 "position":    list(hit_info["position"]),
#                 "distance":    hit_info["distance"],
#                 "normal":      list(hit_info["normal"]),
#             }
#             for name, obj in objects.items():
#                 if hit_path.endswith(name.replace("box_", "Box_")):
#                     info["object_info"] = obj
#                     break
#             return True, info
#         else:
#             return False, {"sanity_image": save_name, "message": "No object hit"}

#     except Exception as e:
#         logger.error(f"Ray cast error: {e}", exc_info=True)
#         return False, str(e)
        
from PIL import Image, ImageDraw
import os

def ray_cast(view_name, x, y):
    """Perform a ray cast and also save a visualization of the picked pixel + ray direction."""
    try:
        # 1) Same camera configs
        camera_configs = {
            "top":   {"position": [0,   0, 7.0],    "orientation": [0, 90,  0]},
            "front": {"position": [-3.0,0, 0.75],   "orientation": [0,  0,  0]},
            "side":  {"position": [0,  -3.0,0.75],  "orientation": [0,  0, 90]},
            "iso":   {"position": [-2.5,2.5,4.0],   "orientation": [0, 45, -45]},
        }
        if view_name not in camera_configs:
            return False, f"Invalid view name. Choose from: {list(camera_configs.keys())}"
        cfg      = camera_configs[view_name]
        cam_path = f"/World/camera_{view_name}_raycast"

        # 2) Spawn & init camera
        camera = Camera(
            prim_path   = cam_path,
            position    = np.array(cfg["position"], dtype=np.float32),
            orientation = rot_utils.euler_angles_to_quats(
                              np.array(cfg["orientation"], dtype=np.float32),
                              degrees=True
                          ),
            frequency   = 20,
            resolution  = (1024, 1024),
        )
        world.scene.add(camera)
        camera.initialize()

        # 3) Let sim settle
        for _ in range(5):
            world.step(render=True)

        # — SAVE RAW VIEW — 
        raw = camera.get_rgba()  # 1D array of length W*H*4
        # reshape into (H, W, 4), using the camera.resolution tuple (width, height)
        W, H = 1024, 1024
        rgba = raw.reshape(H, W, 4)
        rgb  = (rgba[..., :3]).astype('uint8')
        img_raw = Image.fromarray(rgb)

        # 4) Compute intrinsics → back-project (with Y-flip!)
        K     = camera.get_intrinsics_matrix()
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        x_cam = (x - cx) / fx
        y_cam = (cy - y) / fy                # flip Y
        from pxr import Gf
        ray_cam = Gf.Vec3d(x_cam, y_cam, -1.0).GetNormalized()

        # 5) USD extrinsics → world-space ray
        from pxr import UsdGeom
        from omni.isaac.core.utils.stage import get_current_stage
        stage      = get_current_stage()
        prim       = stage.GetPrimAtPath(cam_path)
        xf_cache   = UsdGeom.XformCache()
        xf         = xf_cache.GetLocalToWorldTransform(prim)
        origin     = xf.ExtractTranslation()
        rot_mat    = xf.ExtractRotationMatrix()
        world_dir  = rot_mat * ray_cam

        # — DRAW VISUALIZATION — 
        viz = img_raw.copy()
        draw = ImageDraw.Draw(viz)
        # mark click
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=3)
        # draw arrow (scale arrow to 200px length)
        arrow_len = 200
        dx =  world_dir[0] / abs(world_dir[2]) * arrow_len  # project onto image plane
        dy = -world_dir[1] / abs(world_dir[2]) * arrow_len
        draw.line((x, y, x+dx, y+dy), fill="red", width=3)
        viz_name = f"raycast_{view_name}_viz.png"
        viz.save(viz_name)

        # 6) Fire PhysX pick
        import carb
        from omni.physx import get_physx_scene_query_interface
        qi   = get_physx_scene_query_interface()
        hit  = qi.raycast_closest(
            carb.Float3(origin[0],    origin[1],    origin[2]),
            carb.Float3(world_dir[0], world_dir[1], world_dir[2]),
            1000.0
        )

        # 7) Cleanup
        world.scene.remove_object(camera.name)

        # 8) Return results + image paths
        if hit.get("hit", False):
            path = hit.get("bodyPrimPath", hit.get("colliderPrimPath", ""))
            info = {
                "path":             path,
                "position":         list(hit["position"]),
                "distance":         hit["distance"],
                "normal":           list(hit["normal"]),
                "clicked_pixel":    [x, y],
                "camera_direction": [world_dir[0], world_dir[1], world_dir[2]],
            }
            for name, obj in objects.items():
                if path.endswith(name.replace("box_", "Box_")):
                    info["object_info"] = obj
                    break
            return True, info
        else:
            return False, {
                "message":       "No object hit",
                "clicked_pixel":[x, y],
                "camera_direction": [world_dir[0], world_dir[1], world_dir[2]],
            }

    except Exception as e:
        logger.error(f"Ray cast error: {e}", exc_info=True)
        return False, str(e)


# --- Flask setup ---
app = Flask("IsaacSimServer")

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "object_count": len(objects)
    })

@app.route('/objects', methods=['GET'])
def list_objects():
    return jsonify({"status":"success", "objects":objects})

@app.route('/move', methods=['POST'])
def move_endpoint():
    data = request.get_json(force=True)
    name = data.get("object")
    pos  = data.get("position")
    if not name or not pos:
        return jsonify({"status":"error","message":"Need 'object' and 'position'"}), 400

    # enqueue and wait for result
    resp_q = Queue()
    COMMAND_QUEUE.put((move_object, (name, pos), resp_q))
    success, msg = resp_q.get()
    code = 200 if success else 400
    return jsonify({"status": "success" if success else "error", "message": msg}), code

@app.route('/add', methods=['POST'])
def add_endpoint():
    data = request.get_json(force=True)
    pos   = data.get("position")
    color = data.get("color", (1,0,0))
    size  = data.get("size", 0.1)
    if not pos:
        return jsonify({"status":"error","message":"Need 'position'"}), 400

    resp_q = Queue()
    box_id = len(objects)
    COMMAND_QUEUE.put((add_box, (box_id, pos, color, size), resp_q))
    success, msg = resp_q.get()
    code = 200 if success else 400
    return jsonify({"status": "success" if success else "error", "message": msg}), code

@app.route('/capture', methods=['GET'])
def capture_endpoint():
    view_name = request.args.get('view', 'top')
    
    # Enqueue camera capture request
    resp_q = Queue()
    COMMAND_QUEUE.put((capture_camera, (view_name,), resp_q))
    success, result = resp_q.get()
    
    if not success:
        return jsonify({"status": "error", "message": result}), 400
    
    # If successful, result is a BytesIO object containing PNG image data
    img_data = result
    
    # Convert to base64 for JSON response
    img_data.seek(0)
    base64_img = base64.b64encode(img_data.read()).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "view": view_name,
        "image_base64": base64_img
    })

@app.route('/raycast', methods=['GET'])
def raycast_endpoint():
    view_name = request.args.get('view')
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    
    if not view_name or x is None or y is None:
        return jsonify({
            "status": "error", 
            "message": "Need 'view', 'x' and 'y' parameters"
        }), 400

    # Enqueue ray cast request
    resp_q = Queue()
    COMMAND_QUEUE.put((ray_cast, (view_name, x, y), resp_q))
    success, result = resp_q.get()

    if not success:
        return jsonify({
            "status": "error", 
            "message": result if isinstance(result, str) else "Ray cast failed"
        }), 400

    return jsonify({
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "view": view_name,
        "x": x,
        "y": y,
        "hit_result": result
    })

@app.route('/stack', methods=['POST'])
def stack_endpoint():
    data = request.get_json(force=True)
    source = data.get("source")
    target = data.get("target")
    
    if not source or not target:
        return jsonify({
            "status": "error", 
            "message": "Need 'source' and 'target' object names"
        }), 400
        
    # Enqueue the stack operation
    resp_q = Queue()
    COMMAND_QUEUE.put((move_object_on_top, (source, target), resp_q))
    success, msg = resp_q.get()
    
    code = 200 if success else 400
    return jsonify({
        "status": "success" if success else "error",
        "message": msg
    }), code

def flask_thread():
    """Run Flask in a background thread."""
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)

def simulation_loop():
    """Main-thread loop: drain commands, step physics, update UI."""
    logger.info("Simulation loop starting (main thread)")
    while True:
        # 1) process queued commands
        try:
            func, args, resp_q = COMMAND_QUEUE.get_nowait()
            res = func(*args)
            resp_q.put(res)
        except Empty:
            pass

        # 2) advance physics (no render) & pump UI
        world.step(render=False)
        simulation_app.update()
        time.sleep(0.02)

def main():
    global world

    # 1) Initialize world & scene
    world = World(stage_units_in_meters=1.0)
    setup_scene()
    world.scene.add_default_ground_plane()
    world.reset()
    add_initial_objects()

    # 2) Start Flask in background
    threading.Thread(target=flask_thread, daemon=True).start()
    logger.info(f"Flask server running on port {PORT}")

    # 3) Run sim+UI loop in main thread
    simulation_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        sys.exit(0)

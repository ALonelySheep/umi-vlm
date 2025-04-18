from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import json
import random
import numpy as np
from itertools import product
import omni.isaac.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
import os
from datetime import datetime

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux

"""
Purpose of this script is to create Isaac Sim environment with boxes with varying physical properties defined by the config.json file.
Input: config.json file with object property variations and pattern (created by gui_to_json.py)
Output: 
    - environment_objects.json file with object data
    - camera_views_grid.png file with composite image of camera views
"""

# ----------------------------
# Config & Constants
# ----------------------------
CONFIG_PATH = "./sim_environment/config.json"

# Create timestamped directory
timestamp = datetime.now().strftime("%m-%d-%H-%M")
output_dir = os.path.join("./sim_environment/outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(output_dir, "environment_objects.json")
OUTPUT_IMAGE_PATH = os.path.join(output_dir, "camera_views_grid.png")

print(f"✅ Created output directory: {output_dir}")

COLOR_MAP = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "black": (0.1, 0.1, 0.1)
}
VOLUME_MAP = {"small": 0.05, "medium": 0.1, "large": 0.15}
MASS_MAP = {"1kg": 1.0, "3kg": 3.0, "5kg": 5.0}

DESK_WIDTH = 2.4
DESK_DEPTH = 1.2
DESK_THICKNESS = 0.05
DESK_HEIGHT = 0.75

# ----------------------------
# Helper Functions
# ----------------------------
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def create_desk(stage):
    desk_center_z = DESK_HEIGHT - DESK_THICKNESS / 2
    desk_path = "/World/Desk"
    create_prim(
        prim_path=desk_path,
        prim_type="Cube",
        position=(0, 0, desk_center_z),
        scale=(DESK_WIDTH / 2, DESK_DEPTH / 2, DESK_THICKNESS / 2)
    )
    desk_prim = stage.GetPrimAtPath(desk_path)
    UsdPhysics.CollisionAPI.Apply(desk_prim)
    UsdPhysics.RigidBodyAPI.Apply(desk_prim).CreateRigidBodyEnabledAttr(False)
    UsdGeom.Cube(desk_prim).CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.3, 0.2)])

def setup_lighting(stage):
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(3000)
    def add_sphere_light(path, position, intensity=1000, radius=0.1):
        light = UsdLux.SphereLight.Define(stage, path)
        light.AddTranslateOp().Set(position)
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(radius)
    add_sphere_light("/World/Light_Side1", (2, 0, 2), 1500)
    add_sphere_light("/World/Light_Side2", (-2, 0, 2), 1500)
    add_sphere_light("/World/Light_Back", (0, -2, 2), 1000)

def generate_box_configs(variations, pattern, subprops):
    """
    pattern_key and pattern_value are extracted from the JSON pattern list, e.g. ["Mass", "Volume"].
    combines to defined pattern pairs, e.g. [('1kg', 'small'), ('3kg', 'medium'), ('5kg', 'large')]
    """
    pattern_key, pattern_value = pattern
    mapping_source = subprops[pattern_key]
    mapping_target = subprops[pattern_value]
    pattern_values = list(zip(mapping_source, mapping_target))

    independent_props = [p for p in variations if p not in pattern]
    configs = []

    for pk_val, pv_val in pattern_values:
        base = {pattern_key: pk_val, pattern_value: pv_val}
        if independent_props:
            combos = product(*[subprops[p] for p in independent_props])
            for combo in combos:
                cfg = base.copy()
                for i, p in enumerate(independent_props):
                    cfg[p] = combo[i]
                configs.append(cfg)
        else:
            configs.append(base)
    return configs

def spawn_boxes(world, box_configs):
    margin = 0.05
    spawned_positions = set()
    data = []
    for i, cfg in enumerate(box_configs):
        color_str = cfg.get("Color", "black")
        color_rgb = np.array(COLOR_MAP.get(color_str, (0.5, 0.5, 0.5)))
        volume_str = cfg.get("Volume", "medium")
        volume_val = VOLUME_MAP.get(volume_str, 0.1)
        mass_str = cfg.get("Mass", "1kg")
        mass_val = MASS_MAP.get(mass_str, 1.0)
        while True:
            x = random.uniform(-DESK_WIDTH/2 + margin, DESK_WIDTH/2 - margin)
            y = random.uniform(-DESK_DEPTH/2 + margin, DESK_DEPTH/2 - margin)
            key = (round(x,2), round(y,2))
            if key not in spawned_positions:
                spawned_positions.add(key)
                break
        z = DESK_HEIGHT + volume_val / 2
        pos = [round(x,3), round(y,3), round(z,3)]
        world.scene.add(DynamicCuboid(
            prim_path=f"/World/Box_{i}",
            name=f"box_{i}",
            position=np.array(pos),
            scale=np.array([volume_val]*3),
            size=1.0,
            color=color_rgb
        ))
        data.append({"name": f"box_{i}", "mass": mass_val, "color": color_str, "volume": volume_str, "position": pos})
    return data

def capture_camera_views(world, stage):
    configs = [
        ("top_down", [0, 0, 7.0], [0, 90, 0]),
        ("front_view", [-3.0, 0, 0.75], [0, 0, 0]),
        ("iso_view", [-2.5, 2.5, 4.0], [0, 45, -45]),
    ]
    images = []
    for name, pos, ori in configs:
        cam_path = f"/World/camera_{name}"
        camera = Camera(
            prim_path=cam_path,
            position=np.array(pos),
            frequency=20,
            resolution=(512, 512),
            orientation=rot_utils.euler_angles_to_quats(np.array(ori), degrees=True),
        )
        world.scene.add(camera)
        camera.initialize()
        for _ in range(60):
            world.step(render=True)
        rgb = camera.get_rgba()
        images.append((name, rgb[:, :, :3]))
        world.scene.remove_object(camera.name)
    return images

def save_composite_image(images, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, img) in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].text(0.5, -0.15, name.replace("_", " ").title(), fontsize=12, ha='center', va='top', transform=axs[i].transAxes)
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Add bottom margin for text
    plt.savefig(path, bbox_inches='tight', pad_inches=0.2)
    print(f"✅ Saved composite image to {path}")

# ----------------------------
# Main Execution
# ----------------------------
config = load_config(CONFIG_PATH)
variations = config["Object Property Variations"]
pattern = config["Object Property Pattern"]
subprops = config["Subproperties"]

world = World(stage_units_in_meters=1.0)
stage = get_current_stage()
create_desk(stage)
setup_lighting(stage)

box_configs = generate_box_configs(variations, pattern, subprops)
object_data = spawn_boxes(world, box_configs)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(object_data, f, indent=4)

world.scene.add_default_ground_plane()
world.reset()

camera_images = capture_camera_views(world, stage)
save_composite_image(camera_images, OUTPUT_IMAGE_PATH)

simulation_app.close()

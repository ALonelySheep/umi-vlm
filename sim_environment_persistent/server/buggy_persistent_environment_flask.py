from omni.isaac.kit import SimulationApp
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IsaacSimFlaskServer")

# Initialize SimulationApp (don't close it in this script)
logger.info("Initializing Isaac Sim...")
simulation_app = SimulationApp({"headless": False})
logger.info("Isaac Sim initialized")

# Import Isaac Sim modules after initializing SimulationApp
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux

"""
Purpose of this script is to create a persistent Isaac Sim environment that:
1. Creates a simple scene with a desk and some boxes
2. Starts a Flask server to receive manipulation commands
3. Processes those commands to manipulate objects in the scene
4. Keeps the simulation running until manually stopped

Input: None. Environment runs until manually terminated.
Output: Print statements for server status and received commands.
"""

# ----------------------------
# Config & Constants
# ----------------------------
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

PORT = 5000  # Flask server port

# Global variables for Flask and Isaac Sim communication
world = None
objects = {}
sim_lock = threading.Lock()

# ----------------------------
# Scene Setup Functions
# ----------------------------
def create_desk(stage):
    """Create a desk in the scene"""
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
    logger.info("Desk created")

def setup_lighting(stage):
    """Set up lighting for the scene"""
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(3000)
    def add_sphere_light(path, position, intensity=1000, radius=0.1):
        light = UsdLux.SphereLight.Define(stage, path)
        light.AddTranslateOp().Set(position)
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(radius)
    add_sphere_light("/World/Light_Side1", (2, 0, 2), 1500)
    add_sphere_light("/World/Light_Side2", (-2, 0, 2), 1500)
    add_sphere_light("/World/Light_Back", (0, -2, 2), 1000)
    logger.info("Lighting setup completed")

def create_initial_boxes(world):
    """Create some initial boxes in the scene"""
    objects = {}
    
    try:
        # Create boxes at fixed positions for demonstration
        positions = [
            # Position A - left side of table
            (-DESK_WIDTH/4, 0, DESK_HEIGHT + 0.1),
            # Position B - right side of table
            (DESK_WIDTH/4, 0, DESK_HEIGHT + 0.1),
            # Position C - center back
            (0, -DESK_DEPTH/4, DESK_HEIGHT + 0.1),
        ]
        
        colors = ["red", "blue", "green"]
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            try:
                box_name = f"box_{i}"
                # Convert color tuple to numpy array
                color_tuple = COLOR_MAP[color]
                color_rgb = np.array(color_tuple, dtype=np.float32)
                volume_val = VOLUME_MAP["medium"]
                
                # Convert tuple position to numpy array explicitly
                pos_np = np.array(pos, dtype=np.float32)
                logger.info(f"Creating box {box_name} with position {pos_np} (type: {type(pos_np)}) and color {color}")
                
                # Create the DynamicCuboid with explicit numpy arrays
                scale_np = np.array([volume_val, volume_val, volume_val], dtype=np.float32)
                
                # Debug information
                logger.info(f"Position: {pos_np}, type: {type(pos_np)}")
                logger.info(f"Scale: {scale_np}, type: {type(scale_np)}")
                logger.info(f"Color: {color_rgb}, type: {type(color_rgb)}")
                
                # First try creating box with basic parameters
                world.scene.add(DynamicCuboid(
                    prim_path=f"/World/Box_{i}",
                    name=box_name,
                    position=pos_np,
                    scale=scale_np,
                    color=color_rgb
                ))
                
                objects[box_name] = {
                    "name": box_name,
                    "color": color,
                    "position": pos
                }
                logger.info(f"Successfully added box {box_name} to scene")
                
            except Exception as e:
                logger.error(f"Error creating box {i}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Created {len(objects)} boxes in the scene")
        return objects
        
    except Exception as e:
        logger.error(f"Error in create_initial_boxes: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

# ----------------------------
# Object Manipulation Functions
# ----------------------------
def move_object(obj_name, target_pos):
    """Move an object to a specified position"""
    global world, objects
    
    with sim_lock:
        if not obj_name or obj_name not in objects:
            return {"status": "error", "message": f"Invalid object: {obj_name}"}, 400
        
        try:
            # Get the object from the scene
            obj = world.scene.get_object(obj_name)
            if not obj:
                return {"status": "error", "message": f"Object not found in scene: {obj_name}"}, 404
            
            # Calculate current position
            current_pos = obj.get_world_pose()[0]
            
            # Calculate intermediary positions for smooth movement
            steps = 100
            start_pos = current_pos
            end_pos = np.array(target_pos)
            
            # Smooth movement animation
            for i in range(steps + 1):
                t = i / steps
                interp_pos = start_pos * (1 - t) + end_pos * t
                obj.set_world_pose(interp_pos)
                world.step(render=True)
                # Small sleep to make movement visible
                time.sleep(0.01)
                
            # Update object data
            objects[obj_name]["position"] = target_pos
            
            return {"status": "success", "message": f"Moved {obj_name} to {target_pos}"}, 200
        
        except Exception as e:
            logger.error(f"Error moving object: {str(e)}")
            return {"status": "error", "message": f"Error moving object: {str(e)}"}, 500

# ----------------------------
# Simulation Loop
# ----------------------------
def run_simulation_loop():
    """Run the simulation in a loop"""
    global world
    try:
        logger.info("Starting simulation loop")
        while True:
            with sim_lock:
                # Step the simulation
                world.step(render=True)
            # Sleep to avoid maxing out CPU
            time.sleep(0.01)
    except Exception as e:
        logger.error(f"Simulation loop error: {str(e)}")

# ----------------------------
# Flask Application
# ----------------------------
app = Flask("IsaacSimPersistentEnvironment")

@app.route('/status', methods=['GET'])
def get_status():
    """Return the server status"""
    return jsonify({
        "status": "running",
        "message": "Isaac Sim environment is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/objects', methods=['GET'])
def list_objects():
    """List all objects in the scene"""
    return jsonify({
        "status": "success",
        "objects": objects
    })

@app.route('/move', methods=['POST'])
def move_object_endpoint():
    """Move an object to a new position"""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400
    
    object_name = data.get("object")
    position = data.get("position")
    
    if not object_name or not position:
        return jsonify({"status": "error", "message": "Missing required parameters: object and position"}), 400
    
    response, code = move_object(object_name, position)
    return jsonify(response), code

# ----------------------------
# Main Function
# ----------------------------
def main():
    global world, objects
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    
    # Set up the scene
    create_desk(stage)
    setup_lighting(stage)
    world.scene.add_default_ground_plane()
    
    # Create boxes
    objects = create_initial_boxes(world)
    
    # Reset the world to apply changes
    world.reset()
    
    logger.info("Environment setup complete")
    
    # Start simulation loop in a separate thread
    sim_thread = threading.Thread(target=run_simulation_loop)
    sim_thread.daemon = True
    sim_thread.start()
    
    # Start Flask server
    logger.info(f"Starting Flask server on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False) #! This will block the main thread!!!!!
    #! In this case we want the issacsim to be in main but the flask server to be in backround.

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        logger.info("Closing Isaac Sim...")
        # Don't call simulation_app.close() to avoid confusion in the demonstration
        # simulation_app.close() would normally go here in a clean shutdown
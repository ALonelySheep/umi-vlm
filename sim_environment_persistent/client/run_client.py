import requests
import json
import sys
import argparse
import os
import base64
from datetime import datetime

"""
Simple client to send commands to the persistent Isaac Sim environment using HTTP requests.
Usage examples:
  python client_flask.py status
  python client_flask.py list
  python client_flask.py move --object box_0 --position 0.5 0.3 0.85
  python client_flask.py capture --view top
  python client_flask.py pick --view top --x 512 --y 512
"""

SERVER_URL = "http://localhost:5000"

def get_status():
    """Get the server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}

def list_objects():
    """Get a list of all objects in the scene"""
    try:
        response = requests.get(f"{SERVER_URL}/objects")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}

def move_object(object_name, position):
    """Move an object to a specified position"""
    try:
        data = {
            "object": object_name,
            "position": position
        }
        response = requests.post(f"{SERVER_URL}/move", json=data)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}

def pick_object(view, x, y):
    """Get object at pixel coordinates from a specified view"""
    try:
        params = {"view": view, "x": x, "y": y}
        response = requests.get(f"{SERVER_URL}/raycast", params=params)
        if response.status_code != 200:
            return {"status": "error", "message": f"Failed to pick object: {response.text}"}
        
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}

def stack_objects(source_object, target_object):
    """Stack source object on top of target object"""
    try:
        data = {
            "source": source_object,
            "target": target_object
        }
        response = requests.post(f"{SERVER_URL}/stack", json=data)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}
        
def capture_image(view):
    """Capture an image from a specified viewpoint"""
    try:
        response = requests.get(f"{SERVER_URL}/capture", params={"view": view})
        if response.status_code != 200:
            return {"status": "error", "message": f"Failed to capture image: {response.text}"}
        
        result = response.json()
        
        # Ensure images directory exists
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create timestamped filename with view prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{view}_view_{timestamp}.png"
        filepath = os.path.join(images_dir, filename)
        
        # Decode and save image
        if "image_base64" in result:
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(result["image_base64"]))
            
            result["saved_to"] = filepath
            # Remove base64 data from response to make it cleaner to print
            del result["image_base64"]
            
        return result
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}
    except Exception as e:
        return {"status": "error", "message": f"Error saving image: {str(e)}"}

def print_response(response):
    """Print a response in a nice format"""
    print("\nResponse:")
    print(json.dumps(response, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Client for Isaac Sim persistent environment")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    subparsers.add_parser("status", help="Get server status")
    
    # List command
    subparsers.add_parser("list", help="List all objects in the scene")
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move an object to a position")
    move_parser.add_argument("--object", required=True, help="Name of the object to move")
    move_parser.add_argument("--position", required=True, nargs=3, type=float, 
                            help="Position to move to (x y z)")
    
    # Capture command
    capture_parser = subparsers.add_parser("capture", help="Capture an image from a camera viewpoint")
    capture_parser.add_argument("--view", required=True, choices=["top", "front", "side", "iso"],
                               help="Viewpoint to capture from")
    
    # Pick command
    pick_parser = subparsers.add_parser("pick", help="Get object at pixel coordinates from a view")
    pick_parser.add_argument("--view", required=True, choices=["top", "front", "side", "iso"],
                             help="Viewpoint to pick from")
    pick_parser.add_argument("--x", required=True, type=int, help="X-coordinate in pixels")
    pick_parser.add_argument("--y", required=True, type=int, help="Y-coordinate in pixels")
    
    # Stack command
    stack_parser = subparsers.add_parser("stack", help="Stack source object on top of target object")
    stack_parser.add_argument("--source", required=True, help="Name of the source object")
    stack_parser.add_argument("--target", required=True, help="Name of the target object")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print(f"Sending {args.command} command to server...")
    
    if args.command == "status":
        response = get_status()
    elif args.command == "list":
        response = list_objects()
    elif args.command == "move":
        response = move_object(args.object, args.position)
    elif args.command == "capture":
        response = capture_image(args.view)
    elif args.command == "pick":
        response = pick_object(args.view, args.x, args.y)
    elif args.command == "stack":
        response = stack_objects(args.source, args.target)
    else:
        parser.print_help()
        return
    
    print_response(response)

if __name__ == "__main__":
    main()
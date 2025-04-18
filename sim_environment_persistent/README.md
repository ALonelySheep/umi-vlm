# Persistent Isaac Sim Environment

This demo shows how to create a persistent Isaac Sim environment that can receive external commands to manipulate objects.

## Overview

The system consists of two main components:

1. **Server** (choose one implementation):
   - **Flask version** (`persistent_environment_flask.py`): Creates and maintains the Isaac Sim environment with a Flask web server for handling API requests.
   - **WebSocket version** (`persistent_environment.py`): Creates and maintains the Isaac Sim environment with a WebSocket server for real-time communication.

2. **Client** (choose the matching client for your server):
   - **Flask client** (`client_flask.py`): A simple client that sends HTTP requests to manipulate objects in the environment.
   - **WebSocket client** (`client.py`): A client that uses WebSockets to send commands to the server.

## Features

- Keeps the Isaac Sim environment running continuously
- Accepts external commands through a REST API (Flask) or WebSocket interface
- Provides smooth movement of objects within the environment
- Simple API for object manipulation

## Prerequisites

- Isaac Sim 4.5.0 or later
- Python 3.7+
- Required Python packages:
  - Flask (`pip install flask`) for the Flask implementation
  - Requests (`pip install requests`) for the Flask client
  - WebSockets (`pip install websockets`) for the WebSocket implementation

## Usage - Flask Implementation (Recommended for Simplicity)

### Step 1: Start the Server

Run the Flask server script to launch the Isaac Sim environment and start the HTTP server:

```bash
run_server_flask.bat
```

The server will:

1. Create a simple scene with a desk and three colored boxes
2. Start a Flask server on port 5000
3. Wait for HTTP requests from clients

### Step 2: Send Commands Using the Client

While the server is running, you can use the Flask client script to send commands:

```bash
# Get server status
run_client_flask.bat status

# List all objects in the environment
run_client_flask.bat list

# Move an object to a new position
run_client_flask.bat move --object box_0 --position 0.5 0.3 0.85
```

## API Endpoints (Flask Implementation)

### GET /status

Returns the current status of the server.

### GET /objects

Returns information about all objects in the environment.

### POST /move

Moves an object to a specified position.

Request body:

```json
{
  "object": "box_0",
  "position": [0.5, 0.3, 0.85]
}
```

## Usage - WebSocket Implementation (For Real-Time Communication)

### Step 1: Start the Server

Run the WebSocket server script to launch the Isaac Sim environment and start the WebSocket server:

```bash
run_server.bat
```

### Step 2: Send Commands Using the Client

While the server is running, you can use the WebSocket client script to send commands:

```bash
# List all objects in the environment
run_client.bat list

# Move an object to a new position
run_client.bat move box_0 0.5 0.3 0.85
```

## How It Works

1. The server maintains a simulation loop in a separate thread that continuously updates the Isaac Sim environment
2. A server (Flask or WebSocket) runs concurrently to listen for external commands
3. When a command is received, it's processed and executed in the simulation
4. The server sends back a response with the result of the command

## Extending the System

This demo provides a basic framework that can be extended by:

- Adding more commands (e.g., rotate, scale, change color)
- Implementing more complex object interactions
- Creating a web interface or other client applications
- Integrating with AI/ML systems for automated control

## Troubleshooting

- If the client can't connect, make sure the server is running
- Check that port 5000 (Flask) or 8765 (WebSocket) is available and not blocked by a firewall
- For any issues with Isaac Sim, consult the official NVIDIA Isaac Sim documentation

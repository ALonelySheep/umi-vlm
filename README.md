# 1. Setup Simulation
To set up the simulation, install Isaac Sim 4.2 (OLD) by following the instructions provided on the official website: [Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

# 2. Generating the Environment
Follow these steps to generate the simulation environment:

1. Move the `sim_environment` directory to the base directory of Isaac Sim that was created during installation.

2. Run the following command to create a JSON file describing the task and environment configuration:
   ``` python gui_to_json.py ```

3. Run the following commands to generate a JSON file with object descriptions and render an image of the simulation environment:

```
conda deactivate
cd isaacsim
./python.sh umi_examples/make_environment.py
```

# 3. setup VLM API



# 4. evaluting with the generated environment 

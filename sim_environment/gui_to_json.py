import tkinter as tk
import json

"""
Purpose of this script is to create config.json file which is used for Isaac Sim environment with boxes with varying physical properties.
Input: None. User specified in GUI.
Output: config.json file in the current directory.
"""

# GUI options
options = {
    "Task Action": ["Sort", "Stack", "Find"],
    "Task Description": ["Color", "Volume", "Mass", "Content"],
    "Object Property Variations": ["Color", "Volume", "Mass", "Content", "Position"],
    "Object Property Pattern": []  # dropdowns
}

variation_suboptions = {
    "Color": ["red", "blue", "green", "yellow", "black"],
    "Volume": ["small", "medium", "large"],
    "Mass": ["1kg", "3kg", "5kg"]
}

# Initialize root window
root = tk.Tk()
root.title("Box Configuration")

# Storage
selections = {
    "Task Action": tk.StringVar(value=options["Task Action"][0]),
    "Task Description": tk.StringVar(value=options["Task Description"][0]),
    "Object Property Variations": {},  # BooleanVars for toggles
    "Subproperties": {},               # BooleanVars for sub-options
    "Object Property Pattern": [tk.StringVar(value=options["Object Property Variations"][0]),
                                tk.StringVar(value=options["Object Property Variations"][1])]
}

# Frames to dynamically insert sub-options
dynamic_suboption_frames = {}

# Row 0 - Task Action
tk.Label(root, text="Task Action", width=20, anchor="w").grid(row=0, column=0, padx=5, pady=5)
for j, choice in enumerate(options["Task Action"]):
    tk.Radiobutton(root, text=choice, variable=selections["Task Action"], value=choice).grid(row=0, column=1 + j)

# Row 1 - Task Description
tk.Label(root, text="Task Description", width=20, anchor="w").grid(row=1, column=0, padx=5, pady=5)
for j, choice in enumerate(options["Task Description"]):
    tk.Radiobutton(root, text=choice, variable=selections["Task Description"], value=choice).grid(row=1, column=1 + j)

# Row 2 - Object Property Variations (Multi-select)
def update_submenus():
    # Clear previous frames
    for frame in dynamic_suboption_frames.values():
        frame.destroy()
    dynamic_suboption_frames.clear()

    row_offset = 3
    sub_row = row_offset
    for i, prop in enumerate(options["Object Property Variations"]):
        if selections["Object Property Variations"][prop].get():
            if prop in variation_suboptions:
                frame = tk.Frame(root)
                frame.grid(row=sub_row, column=1, columnspan=5, sticky="w", padx=20)
                tk.Label(frame, text=f"{prop} Options:", anchor="w").pack(side="left")
                selections["Subproperties"][prop] = {}
                for subopt in variation_suboptions[prop]:
                    var = tk.BooleanVar()
                    selections["Subproperties"][prop][subopt] = var
                    tk.Checkbutton(frame, text=subopt, variable=var).pack(side="left")
                dynamic_suboption_frames[prop] = frame
                sub_row += 1

# Row 2 GUI
tk.Label(root, text="Object Property Variations", width=20, anchor="w").grid(row=2, column=0, padx=5, pady=5)
for j, choice in enumerate(options["Object Property Variations"]):
    var = tk.BooleanVar(value=False)
    selections["Object Property Variations"][choice] = var
    cb = tk.Checkbutton(root, text=choice, variable=var, command=update_submenus)
    cb.grid(row=2, column=1 + j)

# Row 3 - Object Property Pattern (Two dropdowns)
tk.Label(root, text="Object Property Pattern", width=20, anchor="w").grid(row=100, column=0, padx=5, pady=5)
dropdown1 = tk.OptionMenu(root, selections["Object Property Pattern"][0], *options["Object Property Variations"])
dropdown2 = tk.OptionMenu(root, selections["Object Property Pattern"][1], *options["Object Property Variations"])
dropdown1.grid(row=100, column=1)
tk.Label(root, text="fixed by").grid(row=100, column=2)
dropdown2.grid(row=100, column=3)

# Save Button
def save_config():
    config = {
        "Task Action": selections["Task Action"].get(),
        "Task Description": selections["Task Description"].get(),
        "Object Property Variations": [k for k, v in selections["Object Property Variations"].items() if v.get()],
        "Object Property Pattern": [v.get() for v in selections["Object Property Pattern"]],
        "Subproperties": {}
    }
    for prop, subdict in selections["Subproperties"].items():
        config["Subproperties"][prop] = [k for k, v in subdict.items() if v.get()]
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Config saved to config.json:", config)
    root.quit()

tk.Button(root, text="Save & Exit", command=save_config).grid(row=200, column=0, columnspan=6, pady=10)

root.mainloop()

import bpy

bpy.context.scene.render.engine = "CYCLES"

bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

# Access Cycles preferences
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences

bpy.context.scene.cycles.device = "GPU"


bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    # d["use"] = 1 # Using all devices, include GPU and CPU
    print(d["name"], d["use"])

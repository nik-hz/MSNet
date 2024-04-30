import bpy



def set_render_settings(device_type="CUDA", max_samples=200):
    """
    Set rendering parameters for the entire Blender file.

    Args:
        device_type (str): Type of compute device to use ("CPU" or "GPU").
        max_samples (int): Maximum number of samples for rendering.
    """
    # Set render engine to Cycles
    bpy.context.scene.render.engine = "CYCLES"

    # Set device type for rendering
    bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device_type

    # Set maximum samples
    bpy.context.scene.cycles.samples = max_samples

    # Enable denoising
    bpy.context.scene.view_layers[0].cycles.use_denoising = True

def set_render_settings(**kwargs):
    """
    Set rendering parameters for the entire Blender file.

    Keyword Args:
        device_type (str): Type of compute device to use ("CPU" or "GPU").
        max_samples (int): Maximum number of samples for rendering.
        output_resolution (tuple): Output resolution as (width, height).
        denoising (bool): Whether to enable denoising or not.
    """
    # Set render engine to Cycles
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    cycles_preferences = bpy.context.preferences.addons["cycles"].preferences

    bpy.context.scene.cycles.device = "GPU"


    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        # d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    # Set device type for rendering
    # device_type = kwargs.get("device_type", "CUDA")
    # bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
    # bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device_type

    # cycles_preferences = bpy.context.preferences.addons["cycles"].preferences

    # bpy.context.scene.cycles.device = "GPU"
    # bpy.context.scene.cycles.feature_set = "SUPPORTED"


    # Set maximum samples
    max_samples = kwargs.get("max_samples", 200)
    bpy.context.scene.cycles.samples = max_samples

    # Set output resolution
    output_resolution = kwargs.get("output_resolution", (1920, 1080))
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = output_resolution

    # Enable denoising
    denoising = kwargs.get("denoising", True)
    bpy.context.scene.view_layers[0].cycles.use_denoising = denoising


# Function to render the image
def render_image(scene_name, camera_name: str, output_path: str, film: bool = True):
    """
    Renders an image

    Args:
        output_path (str): The file path to save the rendered image.
        camera_name (str): The name of the camera to use for rendering.
        film (bool, optional): True if the background is to be shown.
    """
    # Set the active scene
    bpy.context.window.scene = bpy.data.scenes[scene_name]

    # Set the active camera for rendering
    bpy.context.scene.camera = bpy.data.objects.get(camera_name)

    # Enable film transparency
    bpy.context.scene.render.film_transparent = film

    # Ensure the render engine is set to Cycles
    bpy.context.scene.render.engine = "CYCLES"

    # Enable transparent film in Cycles
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # Set the output path for the rendered image
    bpy.context.scene.render.filepath = output_path

    # Render the scene
    bpy.ops.render.render(write_still=True)

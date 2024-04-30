"""
Generates images of mirror surfaces in different environments

So far only flat or convex planes

"""

import os

# import GPUtil
from multiprocessing import Process
import concurrent.futures
import psutil
import time

import bpy

import utils
import generate_meshes
import normal_map
import materials
import renders

bpy.context.scene.render.engine = "CYCLES"
# bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"


def set_world_bgr(file_path: str):
    """
    Set the background of the world to an HDRI image

    Args:
        Path to the hdri will be in .hdr or .exr format.
    """
    # Get the world
    world = bpy.context.scene.world
    # Enable nodes and clear any existing ones
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    # Create a new environment texture node
    env_texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    # Set the environment texture node's image
    env_texture_node.image = bpy.data.images.load(file_path)
    # Create a new background shader node
    background_node = nodes.new(type="ShaderNodeBackground")
    # Create a new output node
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    # Link the environment texture node to the background shader node
    world.node_tree.links.new(
        env_texture_node.outputs["Color"], background_node.inputs["Color"]
    )
    # Link the background shader node to the output node
    world.node_tree.links.new(
        background_node.outputs["Background"], output_node.inputs["Surface"]
    )


def add_normal_material(name="NormalMaterial"):
    """
    Create a new material with a normal shader

    Args:
        name: str, name of the material
    """
    # Create a new material
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create geometry, vector math, and output nodes
    geometry_node = nodes.new(type="ShaderNodeNewGeometry")
    vector_math_node1 = nodes.new(type="ShaderNodeVectorMath")
    vector_math_node2 = nodes.new(type="ShaderNodeVectorMath")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Configure vector math nodes
    vector_math_node1.operation = "ADD"
    vector_math_node1.inputs[1].default_value = (1, 1, 1)  # Set to (1,1,1) for addition
    vector_math_node2.operation = "SCALE"
    vector_math_node2.inputs[3].default_value = 0.5  # Set to (0.5,0.5,0.5) for scaling

    # Link nodes
    mat.node_tree.links.new(
        geometry_node.outputs["Normal"], vector_math_node1.inputs[0]
    )
    mat.node_tree.links.new(vector_math_node1.outputs[0], vector_math_node2.inputs[0])
    mat.node_tree.links.new(vector_math_node2.outputs[0], output_node.inputs["Surface"])

    return mat


def generate_scene(scene_name, hdri_file):
    """
    Generates a scene in a new Blender file. Sets world background and creates materials

    Args:
        scene_name: str, name of the scene and Blender file.
        hdri_file: str, path to the HDRI image file.
    """
    # Set the output directory for scene files
    scenes_dir = "../scenes"
    os.makedirs(scenes_dir, exist_ok=True)

    # Create a new Blender file for the scene
    # bpy.ops.wm.read_factory_settings(use_empty=True)

    utils.delete_objects()
    set_world_bgr(hdri_file)

    mesh = generate_meshes.generate_random_mesh(use_auto_smooth=True)

    mirror = materials.add_mirror_material(name="ReflectiveMaterial")
    add_normal_material(name="NormalMaterial")

    materials.assign_material_to_object(mesh.name, mirror)

    # materials.assign_material_to_mesh(mesh, "ReflectiveMaterial")
    # materials.assign_material_to_mesh(mesh, "NormalMaterial")

    # utils.create_random_camera(mesh)  # name = RandomCamera

    # renders.set_render_settings(device_type="CUDA", max_samples=500, output_resolution=(1000, 1000), denoising=True)

    # Save the scene to the specified file path
    scene_file_path = os.path.join(scenes_dir, f"{scene_name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=scene_file_path)


def render_scene(scene_file):

    scene_file_path = os.path.join("../scenes", f"{scene_file[:-6]}.blend")
    bpy.ops.wm.open_mainfile(filepath=scene_file_path)

    reflective_output_path = os.path.join(
        output_dir, f"{scene_file[:-6]}_reflective.png"
    )  # Remove file extension
    normal_output_path = os.path.join(
        output_dir, f"{scene_file[:-6]}_normal.png"
    )  # Remove file extension

    # Render reflective image
    # materials.assign_material_to_mesh("RandomMesh", "ReflectiveMaterial")
    renders.render_image("Scene", "RandomCamera", reflective_output_path)

    # Render normal image
    # materials.assign_material_to_mesh("RandomMesh", "NormalMaterial")
    renders.render_image("Scene", "RandomCamera", normal_output_path)


hdri_dir = f"{os.getcwd()}/../hdri_downloads"
scene_dir = f"{os.getcwd()}/../scenes"
output_dir = f"{os.getcwd()}/../dataset"


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    hdri_files = [
        os.path.join(hdri_dir, file)
        for file in os.listdir(hdri_dir)
        if file.endswith(".exr")
    ]

    # for hdri_file in hdri_files:
    #     generate_scene("test", hdri_file)

    # scene_files = [
    #     os.path.join(scene_dir, file)
    #     for file in os.listdir(scene_dir)
    #     if file.endswith(".blend")
    # ]
    # scene_files = scene_files[:1]
    print(hdri_files[0])
    generate_scene("asdasd3", hdri_files[0])
    # Generate a random reflective surface
    # sphere = generate_meshes.generate_random_mesh(use_auto_smooth=True)
    # # sphere = generate_meshes.create_cube()
    # print("adding mats")

    # materials.add_mirror_material(name="ReflectiveMaterial")
    # materials.add_normal_material(name="NormalMaterial")

    # Setup the camera
    # camera_params = utils.create_random_camera(sphere)  # name = RandomCamera

    # materials.assign_material_to_mesh("RandomMesh", "ReflectiveMaterial")
    # reflective_output_path = os.path.join(output_dir, f"{file[:-4]}_reflective.png")  # Remove file extension
    # renders.render_image("Scene", "RandomCamera", './t1.png')

    # materials.assign_material_to_mesh("RandomMesh", "NormalMaterial")
    # normal_output_path = os.path.join(output_dir, f"{file[:-4]}_normal.png")  # Remove file extension
    # renders.render_image("Scene", "RandomCamera", './t2.png')

    # # Render scenes in parallel
    # max_workers = 4  # Adjust this value based on your system's resources
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_scene = {executor.submit(render_scene, scene_file): scene_file for scene_file in scene_files}

    # # Timeout after a certain duration
    #     try:
    #         concurrent.futures.wait(future_to_scene, timeout=1000)
    #     except concurrent.futures.TimeoutError:
    #         print("Timeout occurred.")
    # Handle the timeout as needed
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     # Map the render_scene function to the list of scene files
    #     future_to_scene = {executor.submit(render_scene, scene_file, output_dir): scene_file for scene_file in scene_files}

    #     # Monitor resource usage
    #     while any(not future.done() for future in future_to_scene):
    #         # Print CPU and memory usage
    #         cpu_percent = psutil.cpu_percent()
    #         memory_percent = psutil.virtual_memory().percent
    #         print(f"CPU Usage: {cpu_percent}% | Memory Usage: {memory_percent}%")

    #         # Sleep for a short duration before checking again
    #         time.sleep(1)

# if __name__ == '__main__':

#     os.makedirs(output_dir, exist_ok=True)

#     hdri_files = [os.path.join(hdri_dir, file) for file in os.listdir(hdri_dir) if file.endswith('.exr')]

#     renders.set_render_settings(device_type="GPU", max_samples=500, output_resolution=(1000, 1000), denoising=True)


#         materials.assign_material_to_mesh("RandomMesh", "ReflectiveMaterial")
#         reflective_output_path = os.path.join(output_dir, f"{file[:-4]}_reflective.png")  # Remove file extension
#         renders.render_image("Scene", "RandomCamera", reflective_output_path)

#         materials.assign_material_to_mesh("RandomMesh", "NormalMaterial")
#         normal_output_path = os.path.join(output_dir, f"{file[:-4]}_normal.png")  # Remove file extension
#         renders.render_image("Scene", "RandomCamera", normal_output_path)
# TODO parallelize this to make MOAR images

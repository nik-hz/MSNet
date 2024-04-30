"""
Utils. Mostly camera and world related

"""

import bpy
import random
import mathutils
import math

def delete_objects(scene=None, object_names=None, types=None):
    """
    Delete objects by names and types. If no names and types are provided, delete all objects of those types.

    Args:
    - object_names (list of str, optional): Names of the objects to delete. Deletes all if None.
    - types (list of str, optional): Types of objects to delete (e.g., ['MESH', 'CAMERA']). Deletes these types if None.
    """
    # If types is None, default to deleting meshes and cameras
    if types is None:
        types = ["MESH", "CAMERA"]

    # If object_names is provided, filter objects by names and types
    if object_names:
        objects_to_delete = [
            obj
            for obj in bpy.data.objects
            if obj.name in object_names and obj.type in types
        ]
    else:
        # If no names provided, select all objects of the specified types
        objects_to_delete = [obj for obj in bpy.data.objects if obj.type in types]

    # Delete the objects
    for obj in objects_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)


# Function to set up the camera
def setup_camera(location, target):
    # Create the camera
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    # Point the camera at the target
    bpy.ops.object.constraint_add(type="TRACK_TO")
    constraint = camera.constraints["Track To"]
    constraint.target = target
    constraint.up_axis = "UP_Y"
    constraint.track_axis = "TRACK_NEGATIVE_Z"

def create_random_camera(target_object):
    """
    Create a camera with random position and orientation around a target object.

    Args:
        target_object (bpy.types.Object): The object to point the camera at.

    Returns:
        dict: Dictionary containing camera parameters.
    """
    # Generate random spherical coordinates within a 10m radius
    radius = random.uniform(3, 5)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)

    # Convert spherical coordinates to Cartesian coordinates for the camera location
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    # Create the camera object
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    camera.name = "RandomCamera"

    # Set random FOV (focal length) between 10mm and 200mm
    # Assuming the default sensor size of 36mm for full-frame camera to calculate FOV
    camera.data.lens = random.uniform(25, 70)

    # Calculate direction vector towards the target object's center
    direction = target_object.location - camera.location

    # Rotate camera to point towards the target object's center
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Return camera parameters
    params = {
        "location": (x, y, z),
        "focal_length": camera.data.lens,
        "fov": 2
        * math.atan(
            (36 / camera.data.sensor_width)
            / (2 * camera.data.lens / camera.data.sensor_width)
        )
        * (180 / math.pi),  # FOV in degrees
    }

    return params


def create_viewing_camera(source_camera_name, target_object_name):
    scene = bpy.context.scene
    source_camera = scene.objects.get(source_camera_name)
    target_object = scene.objects.get(target_object_name)

    if not source_camera or not target_object:
        print("Camera or target object not found.")
        return

    # Get the dependency graph
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Compute the direction vector from the camera to the target object
    direction = target_object.location - source_camera.location
    direction.normalize()

    # Perform ray casting from the camera to find the intersection with the object
    location, normal, _, _ = target_object.ray_cast(source_camera.location, direction)

    if location is None:
        print("No intersection found.")
        return

    # Adjust the normal to world coordinates
    normal = target_object.matrix_world.to_3x3().inverted().transposed() @ normal
    normal.normalize()

    # Place a new camera at the intersection point, slightly offset by the normal
    bpy.ops.object.camera_add(location=location + normal * 0.05)
    new_camera = bpy.context.object

    # Orient the new camera to face along the normal
    # Compute the rotation matrix from the normal vector
    up = mathutils.Vector((0, 0, 1))
    angle = math.acos(normal.dot(up))
    axis = up.cross(normal)
    rotation_matrix = mathutils.Matrix.Rotation(angle, 4, axis)
    new_camera.rotation_euler = rotation_matrix.to_euler()

    # Optionally, adjust the new camera settings
    new_camera.data.lens = 50  # Set focal length to 50mm as an example


def set_world_bgr(file_path: str):
    """
    Set the background of the world to an HDRI image

    Args:
        Path to the hdri will be in .hdr or .exr format.
    """
    try:
        # Get the world
        world = bpy.context.scene.world
        if world is None:
            bpy.ops.world.new()
            world = bpy.context.scene.world

        # Enable nodes and clear any existing ones
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()

        # Create a new environment texture node
        env_texture_node = nodes.new(type="ShaderNodeTexEnvironment")

        # Load the image
        try:
            env_texture_node.image = bpy.data.images.load(file_path)
        except Exception as e:
            print(f"Error loading HDRI image: {e}")
            return

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

    except Exception as e:
        print(f"Error setting world background: {e}")

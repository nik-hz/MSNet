import random
import bpy
import bmesh

# Function to create a reflective sphere
def create_cube(size=1, name="Cube", use_auto_smooth=True):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 1))
    cube = bpy.context.active_object
    cube.name = name
    cube.data.use_auto_smooth = use_auto_smooth
    return cube

def create_sphere(radius=1, segments=64, rings=32, name="Sphere", use_auto_smooth=True):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 1), segments=segments, ring_count=rings)
    sphere = bpy.context.active_object
    sphere.name = name
    # sphere.data.use_auto_smooth = use_auto_smooth
    bpy.ops.object.shade_smooth(use_auto_smooth)

    return sphere

def create_cylinder(radius=1, depth=2, name="Cylinder", use_auto_smooth=True):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=(0, 0, 1))
    cylinder = bpy.context.active_object
    cylinder.name = name
    cylinder.data.use_auto_smooth = use_auto_smooth
    return cylinder

def create_cone(radius=1, depth=2, name="Cone", use_auto_smooth=True):
    bpy.ops.mesh.primitive_cone_add(radius1=radius, depth=depth, location=(0, 0, 1))
    cone = bpy.context.active_object
    cone.name = name
    cone.data.use_auto_smooth = use_auto_smooth
    return cone

def create_torus(major_radius=1, minor_radius=0.4, name="Torus", use_auto_smooth=True):
    bpy.ops.mesh.primitive_torus_add(major_radius=major_radius, minor_radius=minor_radius, location=(0, 0, 1))
    torus = bpy.context.active_object
    torus.name = name
    torus.data.use_auto_smooth = use_auto_smooth
    return torus

def create_ico_sphere(radius=1, name="Icosphere", use_auto_smooth=True):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, location=(0, 0, 1))
    ico_sphere = bpy.context.active_object
    ico_sphere.name = name
    ico_sphere.data.use_auto_smooth = use_auto_smooth
    return ico_sphere

def create_plane(size=1, name="Plane", use_auto_smooth=True):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 1))
    plane = bpy.context.active_object
    plane.name = name
    plane.data.use_auto_smooth = use_auto_smooth
    return plane



def apply_material_to_objects(objects, material):
    for obj in objects:
        if obj.type == 'MESH':
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material


import random

def generate_random_mesh(name="RandomMesh", use_auto_smooth=False):
    """
    Generates a random mesh of a randomly chosen type.

    Args:
        name (str, optional): Name of the mesh object.
        use_auto_smooth (bool, optional): Whether to enable auto smoothing for the mesh.

    Returns:
        bpy.types.Object: The generated mesh object.
    """
    mesh_types = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'ico_sphere', 'plane']
    random_mesh_type = random.choice(mesh_types)

    if random_mesh_type == 'cube':
        return create_cube(size=random.uniform(0.5, 2), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'sphere':
        return create_sphere(radius=random.uniform(0.5, 2), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'cylinder':
        return create_cylinder(radius=random.uniform(0.5, 2), depth=random.uniform(1, 3), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'cone':
        return create_cone(radius=random.uniform(0.5, 2), depth=random.uniform(1, 3), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'torus':
        return create_torus(major_radius=random.uniform(0.5, 2), minor_radius=random.uniform(0.2, 0.8), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'ico_sphere':
        return create_ico_sphere(radius=random.uniform(0.5, 2), name=name, use_auto_smooth=use_auto_smooth)
    elif random_mesh_type == 'plane':
        return create_plane(size=random.uniform(0.5, 2), name=name, use_auto_smooth=use_auto_smooth)

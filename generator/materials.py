import bpy


def add_mirror_material(name="ReflectiveMaterial"):
    """
    Create a new material with a glossy shader and shade auto smooth

    Args:
        name: str, name of the material
    """
    # Create a new material
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    # Add a glossy BSDF node
    glossy_node = nodes.new(type="ShaderNodeBsdfGlossy")
    glossy_node.inputs[1].default_value = 0  # Sharp reflection
    # Add a material output node and link the glossy node to it
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    links = mat.node_tree.links
    links.new(glossy_node.outputs[0], output_node.inputs["Surface"])

    return mat


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


def assign_material_to_object(obj_name, mat):
    """
    Assign a material to an object

    Args:
        obj_name: str, name of the object to assign the material to
        mat: bpy.types.Material, material to assign
    """
    obj = bpy.data.objects.get(obj_name)
    if obj:
        if obj.data.materials:
            # If the object already has materials, replace the first one with the new material
            obj.data.materials[0] = mat
        else:
            # If the object has no materials, append the new material
            obj.data.materials.append(mat)
    else:
        print(f"Object '{obj_name}' not found.")

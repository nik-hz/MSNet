import bpy

def render_normal_map(scene_name, camera_name, output_path):
    """
    Renders an image of the normal map of the specified scene using the given camera.

    Args:
    - scene_name (str): The name of the scene to render.
    - camera_name (str): The name of the camera to render from.
    - output_path (str): The file path to save the rendered image.
    """
    # Save the current scene and objects' materials
    current_scene = bpy.context.window.scene
    original_materials = {
        obj: obj.data.materials[0] if obj.data.materials else None
        for obj in bpy.data.objects
        if obj.type == "MESH"
    }

    matcap_mat = None  # Initialize matcap_mat outside the try block

    try:
        # Set the active scene
        bpy.context.window.scene = bpy.data.scenes[scene_name]

        # Create a Matcap material
        matcap_mat = bpy.data.materials.new(name="MatcapMaterial")
        matcap_mat.use_nodes = False  # Disable nodes for simplicity
        matcap_mat.diffuse_color = (1, 1, 1,1)  # Set to white for accurate normals
        matcap_mat.use_matcap = True
        # Choose a Matcap texture (e.g., "Metal")
        matcap_mat.matcap_icon = bpy.data.matcaps['Metal']

        # Apply the Matcap material to all mesh objects
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                if len(obj.data.materials) > 0:
                    obj.data.materials[0] = matcap_mat
                else:
                    obj.data.materials.append(matcap_mat)

        # Set the render camera
        bpy.context.scene.camera = bpy.data.objects[camera_name]

        # Set the output path for the render
        bpy.context.scene.render.filepath = output_path

        # Render the scene
        bpy.ops.render.render(write_still=True)

    finally:
        # Restore original materials
        for obj, mat in original_materials.items():
            if len(obj.data.materials) > 0:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

        # Clean up: remove the Matcap material if it was created
        if matcap_mat is not None:
            bpy.data.materials.remove(matcap_mat)

        # Restore the original scene
        bpy.context.window.scene = current_scene

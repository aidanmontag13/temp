#To Run & "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" `    -b `    -P setup_blender.py `    -- `    --capture-name aidan_sequence

import bpy
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Create Point Cloud For Volumetric Rendering.")

parser.add_argument(
        "--capture-name",
        type=str,
        default="aidan_sequence",
        help="Name of the capture directory inside the output folder (default is BlenderCalibrationTake)"
    )

argv = sys.argv

if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

args = parser.parse_args(argv)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.join(os.path.dirname(os.getcwd()), "outputs", args.capture_name)

filepath = os.path.join(base_dir, "mesh_sequence", "1.usd")

blender_path = os.path.join(base_dir, "mesh_sequence.blend")

bpy.ops.wm.usd_import(
    filepath=filepath
)

obj = bpy.context.selected_objects[0]

obj.scale = (1.0, 1.0, 1.0)

mod = obj.modifiers.new(
    name="MeshSequenceCache",
    type='MESH_SEQUENCE_CACHE'
)

bpy.ops.cachefile.open(filepath=filepath)

filename = os.path.basename(filepath)

mod.cache_file = bpy.data.cache_files[filename]

mod.object_path = "/mesh"

mod.cache_file.is_sequence = True

mat = obj.active_material

for node in mat.node_tree.nodes:
    if node.type == 'TEX_IMAGE':
        node.image.source = 'SEQUENCE'
        node.image_user.use_auto_refresh = True
        
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 300

bpy.ops.wm.save_as_mainfile(
    filepath=blender_path
)
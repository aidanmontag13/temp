import os
import time
import open3d as o3d
import cv2
import argparse
import numpy as np
import xatlas
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdShade, UsdGeom, Sdf, Vt

#export EGL_PLATFORM=surfaceless

def read_cameras(cameras_txt):
    """read colmap cameras txt"""
    cameras = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.strip().split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = list(map(float, elems[4:]))
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images(images_txt):
    """read colmap images txt"""
    images = {}
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_number = int(parts[8])
        image_name = parts[9]
        images[image_name] = {
            'image_id': image_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_number': camera_number
        }
        i += 2
    return images

def build_volume_parameters(pcd, scale=2.0):
    """estimate scale of voxelgrid for fine tuned carving based on pointcloud"""

    center = pcd.get_center()

    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()

    width  = extent[0] * scale
    height = extent[1] * scale
    depth  = extent[2] * scale

    return width, height, depth, center

def voxelgrid_to_pointcloud(voxel_grid):
    """ convert open3d voxelgrid to open3d pointcloud"""
    points = []

    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index  # (i, j, k)

        center = voxel_grid.get_voxel_center_coordinate(
            np.array(idx, dtype=np.int32)
        )

        points.append(center)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return pcd

def is_voxel_on_surface(index_set, idx):
    """ check if a voxel face is exposed on the surface"""
    normal = np.array([0.0, 0.0, 0.0])
    neighbors = [
        (idx[0] + 1, idx[1], idx[2]),
        (idx[0] - 1, idx[1], idx[2]),
        (idx[0], idx[1] + 1, idx[2]),
        (idx[0], idx[1] - 1, idx[2]),
        (idx[0], idx[1], idx[2] + 1),
        (idx[0], idx[1], idx[2] - 1)
    ]

    normal_lut = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    # find outward faces to calciulate surface normals and for coloring
    for n in neighbors:
        if n in index_set:
            normal = normal + normal_lut[neighbors.index(n)]

    if np.linalg.norm(normal) > 0:
        return True, normal
            
    return False, np.array([0.0, 0.0, 0.0])

def voxelgrid_to_hollow_pointcloud(voxel_grid):
    """ convert voxel grid to pointcloud, ignoring voxels that are beneth the surface"""
    points = []
    normals = []
    voxels = voxel_grid.get_voxels()

    index_set = {tuple(v.grid_index) for v in voxels}

    for voxel in voxels:
        idx = tuple(voxel.grid_index)

        is_on_surface, normal = is_voxel_on_surface(index_set, idx)
        if not is_on_surface:
            continue

        center = voxel_grid.get_voxel_center_coordinate(
            np.array(idx, dtype=np.int32)
        )

        points.append(center)
        normals.append(normal)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))

    return pcd

def create_camera_dataset(images, cameras, color_dir):
    """read images and colmap files and assemble initial datapoints"""
    cameras_parameters = []
    geometries = []
    color_images = []
    masks = []
    for image_name, data in images.items():
        color_path = os.path.join(color_dir, image_name)

        if os.path.isfile(color_path):
            color = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        
        else:
            continue

        cam_number = data['camera_number']

        rot = R.from_quat([data['qx'], data['qy'], data['qz'], data['qw']])
        t_cw = np.array([data['tx'], data['ty'], data['tz']])

        R_cw = rot.as_matrix()
        
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        image_width = cameras[cam_number]['width']
        image_height = cameras[cam_number]['height']
        fx, fy, cx, cy = cameras[cam_number]['params']

        params = o3d.camera.PinholeCameraParameters()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=image_width,
            height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        params.intrinsic = intrinsic
        params.extrinsic = T_cw

        cameras_parameters.append(params)
        
        alpha = color[:, :, 3]
        bgr = color[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ret, alpha = cv2.threshold(alpha, 32, 255, cv2.THRESH_BINARY)
        float_alpha = alpha.astype('float32') / 255.0

        mask = o3d.geometry.Image(float_alpha)

        color_images.append(rgb)

        masks.append(mask)

        cam = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=params.intrinsic,
            extrinsic=params.extrinsic,
            scale=0.3
        )
        geometries.append(cam)

    return cameras_parameters, color_images, masks, geometries

def carve_voxel_grid(camera_parameters, masks, voxel_grid, keep_voxels_outside_image=False):
    """carve voxel grid based on alpha channel of input images and camera parameters"""
    for params, mask in zip(camera_parameters, masks):
        voxel_grid.carve_silhouette(mask, params, keep_voxels_outside_image=keep_voxels_outside_image)
        num_voxels = len(voxel_grid.get_voxels())

    return voxel_grid

def extract_z_buffers(voxel_grid, camera_parameters):
    """ render depth images of voxel grid from input camera views"""
    width = camera_parameters[0].intrinsic.width
    height = camera_parameters[0].intrinsic.height

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        width,
        height
    )

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    renderer.scene.add_geometry(
        "voxels",
        voxel_grid,
        material
    )

    depth_images = []

    for param in camera_parameters:

        renderer.setup_camera(
            param.intrinsic,
            param.extrinsic
        )

        depth_image = renderer.render_to_depth_image(
            z_in_view_space=True
        )

        depth = np.asarray(depth_image)

        depth[np.isinf(depth)] = 0.0

        depth_images.append(depth)

    return depth_images

def build_gradient_map(voxel_grid, pcd, voxel_size, camera_parameters):
    """build gradient map to correlate 3D pointcloud points with 2D image points calculate weight based on distance to surface from projected point"""
    depth_maps = extract_z_buffers(voxel_grid, camera_parameters)

    points = np.asarray(pcd.points)

    gradient_map = [[] for _ in range(len(points))]
    for cam_idx, params in enumerate(camera_parameters):
        width = params.intrinsic.width
        height = params.intrinsic.height
        for pt_idx, point in enumerate(points):
            x,y,z = point

            u, v, depth = unproject_3dpoint(point, params)

            u = int(round(u))
            v = int(round(v))

            if 0 <= u < width and 0 <= v < height:
                depth_at_pixel = depth_maps[camera_parameters.index(params)][v, u]

                unscaled_weight = depth - depth_at_pixel

                if unscaled_weight < voxel_size:
                    weight = 1
                else:
                    weight = voxel_size / unscaled_weight
            else: 
                weight = 0

            gradient_map[pt_idx].append({
                'cam': cam_idx,
                'u': u,
                'v': v,
                'weight': weight
            })
    return gradient_map

def unproject_3dpoint(point, normal, camera_parameters):
    """ convert 3D point to 2D coordinate from camera projection"""
    K = camera_parameters.intrinsic.intrinsic_matrix
    extrinsic = camera_parameters.extrinsic

    point_h = np.append(point, 1.0)

    cam_point = extrinsic @ point_h

    x, y, z = cam_point[:3]

    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]

    cam_normal = extrinsic[:3, :3] @ normal

    nx, ny, nz = cam_normal

    norm_length = np.linalg.norm(cam_normal)
    if norm_length > 0:
        cam_normal = cam_normal / norm_length

    return u, v, z, nx, ny, nz

def color_pointcloud(color_maps, pcd, camera_parameters):
    """ use gradient map to color pointcloud with image colors"""
    # convert open3d pointcloud to numpy array
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # initialize color array
    colors = np.zeros((len(pcd.points), 3), dtype=np.float32)

    width = camera_parameters[0].intrinsic.width
    height = camera_parameters[0].intrinsic.height

    for i, point in enumerate(pcd.points):

        color = np.zeros(4, dtype=np.float32)

        image_colors = []

        # extract data from gradient map
        for cam, camera in enumerate(camera_parameters):
            u, v, z, nx, ny, nz = unproject_3dpoint(points[i], normals[i], camera)
        
            if 0 <= u < width and 0 <= v < height:
                if nz < 0.0:
                    image_color = color_maps[cam][int(v), int(u)] / 255.0
                    image_colors.append(image_color[:3])

                    best_weight = nz
        
                if image_colors:
                    colors[i] = np.median(image_colors, axis=0)
                else:
                    colors[i] = [0.0, 0.0, 0.0]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, colors

def add_uvs_with_xatlas(mesh):
    """use xatlas library to create UV coordinates for each vertex"""
    vertices = mesh.vertex.positions.numpy()
    faces = mesh.triangle.indices.numpy()
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    unwrapped_mesh = o3d.t.geometry.TriangleMesh()
    unwrapped_mesh.vertex.positions = o3d.core.Tensor(vertices[vmapping], dtype=o3d.core.float32)
    unwrapped_mesh.triangle.indices = o3d.core.Tensor(indices, dtype=o3d.core.int32)

    triangle_uvs = uvs[indices]
    unwrapped_mesh.triangle['texture_uvs'] = o3d.core.Tensor(triangle_uvs, dtype=o3d.core.float32)

    unwrapped_mesh.compute_vertex_normals()

    return unwrapped_mesh

def pointcloud_to_colored_mesh(pcd, camera_parameters, color_images):
    """ use open3d albedo map creation to convert pointcloud to mesh"""
    
    # topoolgize surface
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    
    #smooth voxels
    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()

    # convert to tensor mesh
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # estimate uvs
    mesh = add_uvs_with_xatlas(mesh)
    
    albedo_contrast = 1.5

    Ks = []
    Rts = []

    # build list of matrixies and transforms
    o3d_images = [
        o3d.t.geometry.Image(img)
        for img in color_images
    ]

    for params in camera_parameters:
        Ks.append(
            o3d.core.Tensor(
            params.intrinsic.intrinsic_matrix,
            dtype=o3d.core.float64
            )
        )

        Rts.append(
            o3d.core.Tensor(
            params.extrinsic,
            dtype=o3d.core.float64
            )
        )
    
    # create albedo
    albedo = mesh.project_images_to_albedo(o3d_images, Ks, Rts, 1024, True)
        
    albedo = albedo.linear_transform(scale=albedo_contrast)  # brighten albedo

    mesh.material.set_default_properties()
    mesh.material.texture_maps["albedo"] = albedo
    mesh.material.scalar_properties["emissive_strength"] = 0.0
    mesh.material.vector_properties["emissive_color"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # flip for blender
    R_np = np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0, -1.0,  0.0],
                     [ 0.0,  0.0, -1.0]], dtype=np.float32)
    
    R = o3d.core.Tensor(R_np)

    global_origin = o3d.core.Tensor([0.0, 0.0, 0.0], dtype=o3d.core.float32)
    
    mesh.rotate(R, center=global_origin)

    return mesh, albedo

def write_points3D(pcd, colmap_dir):
    """write open3d pcd as colmap points3d txt file"""
    points3D_path = os.path.join(colmap_dir, "points3D.txt")

    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    with open(points3D_path, 'w') as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (pt, color) in enumerate(zip(points, colors), 1):
            if color[0] == 0 and color[1] == 0 and color[2] == 0:
                continue
            f.write(f"{i} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]} {color[1]} {color[2]} 1.0\n")

def export_as_usd(mesh, albedo, texture_path, output_path):
    """export mesh and image as .usd file"""
    mesh = mesh.to_legacy()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)

    stage = Usd.Stage.CreateNew(output_path)

    usd_mesh = UsdGeom.Mesh.Define(stage, f"/mesh")

    usd_mesh.CreatePointsAttr(vertices)

    face_vertex_counts = np.full(
        len(triangles),
        3,
        dtype=np.int32
    )

    usd_mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray(face_vertex_counts.tolist())
    )

    usd_mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray(triangles.flatten().tolist())
    )

    st = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying
    )

    st.Set(Vt.Vec2fArray.FromNumpy(uvs))

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    usd_mesh.CreateNormalsAttr(normals)
    usd_mesh.SetNormalsInterpolation("vertex")

    # Create material
    material = UsdShade.Material.Define(stage, "/Material")

    # Create preview surface shader
    surface_shader = UsdShade.Shader.Define(stage, "/Material/PreviewSurface")
    surface_shader.CreateIdAttr("UsdPreviewSurface")

    # Create texture reader
    tex_shader = UsdShade.Shader.Define(stage, "/Material/AlbedoTexture")
    tex_shader.CreateIdAttr("UsdUVTexture")

    tex_shader.CreateInput(
        "file",
        Sdf.ValueTypeNames.Asset
    ).Set(texture_path)

    # Connect texture RGB to baseColor
    surface_shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f
    ).ConnectToSource(
        tex_shader.ConnectableAPI(),
        "rgb"
    )

    # Connect shader to material
    material.CreateSurfaceOutput().ConnectToSource(
        surface_shader.ConnectableAPI(),
        "surface"
    )

    # Bind material to mesh
    UsdShade.MaterialBindingAPI(
        usd_mesh.GetPrim()
    ).Bind(material)

    stage.GetRootLayer().Save()

    print(f"Saved USD: {output_path}")

    return output_path

def write_ply(filename, pcd, colors, gradient_map, camera_count, voxel_size=0.02):
    """export colored pointcloud as .ply file, include image uvs for differentiable rendering"""
    print(f"Writing PLY file to {filename} with {len(pcd.points)} points...")
    points = np.asarray(pcd.points)
    n = len(points)

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write("property float red\n")
        f.write("property float green\n")
        f.write("property float blue\n")
        f.write("property float alpha\n")

        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")

        f.write("property float roughness\n")
        f.write("property float metalness\n")

        f.write("property float voxel_size\n")

        for cam_idx in range(camera_count):
            f.write(f"property float cam{cam_idx}_u\n")
            f.write(f"property float cam{cam_idx}_v\n")
            f.write(f"property float cam{cam_idx}_weight\n")

        f.write("end_header\n")

        for i, (p, c) in enumerate(zip(points, colors)):
            x, y, z = p
            r, g, b = c

            cam_block = np.zeros(3 * camera_count, dtype=np.float32)

            for camera in gradient_map[i]:
                cam_idx = int(camera['cam'])
                u = camera['u']
                v = camera['v']
                w = camera['weight']

                cam_block[cam_idx*3:(cam_idx+1)*3] = [u, v, w]

            cam_str = " ".join(map(str, cam_block))

            f.write(f"{x} {y} {z} {r} {g} {b} 1 0 0 0 1 0 {voxel_size} {cam_str}\n")

def process_frame(first_frame, frame_id, base_dir, data_format="postshot", voxel_resolution=256, search_radius=5.0, export_mesh = False, visualize=False):
    """calcualte pointcloud from image set and export in colmap format"""
    if data_format == "postshot":
        color_dir = os.path.join(base_dir, "processed_color", frame_id)
        colmap_dir = color_dir


    if data_format == "instant_ngp":
        color_dir = os.path.join(base_dir, "processed_color", frame_id, "images")
        colmap_dir = os.path.join(base_dir, "processed_color", frame_id, "sparse", "0")

    # read parameters from COLMAP output
    images_txt = os.path.join(colmap_dir, "images.txt")
    cameras_txt = os.path.join(colmap_dir, "cameras.txt")

    if not os.path.isfile(images_txt) or not os.path.isfile(cameras_txt):
        print(f"[!] Missing COLMAP metadata in {colmap_dir}")
        return

    images = read_images(images_txt)

    cameras = read_cameras(cameras_txt)

    # Create dataset of camera parameters, color images, and masks
    camera_parameters, color_images, masks, geometries = create_camera_dataset(images, cameras, color_dir)

    # create low resolution voxel grid to search for bounds of object
    coarse_voxel_grid = o3d.geometry.VoxelGrid.create_dense(
        width=search_radius * 2,
        height=search_radius * 2,
        depth=search_radius * 2,
        voxel_size=(search_radius * 2) / (256 / 2),
        origin = [-search_radius, -search_radius, -search_radius],
        color=[0.0, 0.0, 0.0]
    )
    
    #print(f"Initial {num_voxels} voxels")

    #carve grid
    coarse_voxel_grid = carve_voxel_grid(camera_parameters, masks, coarse_voxel_grid, keep_voxels_outside_image=False)

    num_voxels = len(coarse_voxel_grid.get_voxels())

    if num_voxels == 0:
        pcd = pcd = o3d.geometry.PointCloud()
        write_points3D(pcd, colmap_dir)
        return

    # convert to point cloud and build volume parameters
    localizer_pcd = voxelgrid_to_hollow_pointcloud(coarse_voxel_grid)
    volume_width, volume_height, volume_depth, volume_origin = build_volume_parameters(localizer_pcd, scale = 1.5)
    voxel_size = np.linalg.norm([volume_width, volume_height, volume_depth]) / voxel_resolution

    #print(f"Volume Parameters - Width: {volume_width:.2f}, Height: {volume_height:.2f}, Depth: {volume_depth:.2f}, Origin: {volume_origin}")

    # build higher resolution voxel grid with parameters from coarse grid
    fine_voxel_grid = o3d.geometry.VoxelGrid.create_dense(
        width=volume_width,
        height=volume_height,
        depth=volume_depth,
        voxel_size=voxel_size,
        origin=volume_origin - np.array([volume_width, volume_height, volume_depth]) / 2.0,
        color=[0.5, 0.5, 0.5]
    )

    # Carve Again
    fine_voxel_grid = carve_voxel_grid(camera_parameters, masks, fine_voxel_grid, keep_voxels_outside_image=True)
    num_voxels = len(fine_voxel_grid.get_voxels())
    print(f"Carved {num_voxels} voxels")

    if num_voxels == 0:
        pcd = pcd = o3d.geometry.PointCloud()
        write_points3D(pcd, colmap_dir)
        return

    #convert to point cloud
    pcd = voxelgrid_to_hollow_pointcloud(fine_voxel_grid)
    
    if export_mesh:
        mesh, albedo = pointcloud_to_colored_mesh(pcd, camera_parameters, color_images)

        sequence_path = os.path.join(base_dir, "mesh_sequence")
        os.makedirs(sequence_path, exist_ok=True)

        frame = str(int(frame_id) - int(first_frame) + 1)

        texture_path = os.path.join(sequence_path, f"{frame}.png")
        o3d.t.io.write_image(texture_path, albedo)

        usd_path = os.path.join(sequence_path, f"{frame}.usd")  
        export_as_usd(mesh, albedo, texture_path, usd_path)

    # color point cloud based on gradient map projections and camera colors
    pcd, colors = color_pointcloud(color_images, pcd, camera_parameters)

    # write points3D and ply file for visualization and later optimization
    write_points3D(pcd, colmap_dir)
    
    geometries.append(pcd)

    if visualize == True:   
        o3d.visualization.draw_geometries(
            geometries,
            point_show_normal=False
        )
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Create Point Cloud For Volumetric Rendering.")
    parser.add_argument(
        "--capture-name",
        type=str,
        default="BlenderCalibrationTake",
        help="Name of the capture directory inside the output folder (default is BlenderCalibrationTake)"
    )

    parser.add_argument(
        "--data-format",
        type=str,
        default="postshot",
        help="Name of data format type, for example -postshot or -instant-ngp"
    )

    parser.add_argument(
        "--frame-range",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True applies frame range processing, false applies no frame range processing"
    )

    parser.add_argument(
        "--frame-start",
        type=int,
        default=100,
        help="frame to begin processing from (default is 0)"
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=100,
        help="frame to begin processing from (default is full capture length)"
    )

    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=128,
        help="resolution of the voxel grid (default is 256)"
    )

    parser.add_argument(
        "--search-radius",
        type=float,
        default=5.0,
        help="search radius for voxel carving (default is 5.0)"
    )

    parser.add_argument(
        "--export-mesh",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True exports textured mesh with pointcloud"
    )

    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.getcwd()), "outputs", args.capture_name)

    color_root = os.path.join(base_dir, "processed_color")
    frame_ids = sorted([f for f in os.listdir(color_root) if os.path.isdir(os.path.join(color_root, f))])

    first_frame = frame_ids[0]

    usd_paths = []

    for frame_id in frame_ids:
        try:
            frame_number = int(frame_id)
        except ValueError:
            continue
        if args.frame_range == True:
            if args.frame_start is not None and frame_number < args.frame_start:
                continue
            if args.frame_end is not None and frame_number > args.frame_end:
                continue
        print(f"\n=== Processing Frame: {frame_id} ===")
        process_frame(first_frame, frame_id, base_dir, args.data_format, args.voxel_resolution, args.search_radius, args.export_mesh, visualize=False)

    print("Point Cloud Creation complete.")

if __name__ == "__main__":
    main()

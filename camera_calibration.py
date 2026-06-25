import dis
import cv2
import numpy as np
import os
import glob
import argparse
import pprint
import open3d as o3d
import pupil_apriltags as apriltag
from scipy import stats
from scipy.spatial.transform import Rotation
import time
import marker_calibration
import stereo_depth
import process_images

PIXEL_ERROR_THRESHOLD = 2.0

DETECTOR = apriltag.Detector(
        families="tag36h11")

def load_marker_file(calibration_dir):
    """ load markers.txt and convert to marker data"""
    marker_data = {}
    input_file = os.path.join(calibration_dir, "markers.txt")

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 13:
                raise ValueError(f"Invalid marker line format: {line}")

            marker_id = int(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, parts[4:13])

            marker_data[marker_id] = {
                "t": np.array([[tx], [ty], [tz]], dtype=np.float64),
                "R": np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], dtype=np.float64),
                "weight": 1.0,
            }

    return marker_data

def extract_camera_data(image_path, marker_length, camera_number, stereo=False):
    """Build master dataset of marker observations from images, organized by image ID."""
    camera_id = camera_number // 10
    image_data = {}
    
    try:
        left_img = cv2.imread(image_path)
        if left_img is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # if using stereo pipeline, rectify images, if not just undistort
    if stereo == False:
        
        try:
            left_camera_matrix, left_dist_coeffs, width, height, P1 = process_images.load_camera_params_json(camera_id, 1, stereo=False)
        
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading camera parameters for camera ID {camera_number}: {e}")
            return
        
        detected_markers = marker_calibration.detect_apriltags_positions(
            left_img, None,
            left_camera_matrix, None,
            left_dist_coeffs, None,
            P1, None, None, None,
            marker_length,
            stereo=False
        )

    else:
        try:
            left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, width, height, R, T, P1, P2, R1, R2 = process_images.load_camera_params_json(camera_id, 1, stereo=True)
    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading camera parameters for camera ID {camera_number}: {e}")
            return
    
        right_img_path = process_images.find_right_img_path(image_path, camera_number)

        try:
            right_img = cv2.imread(right_img_path)
            if right_img is None:
                raise ValueError(f"Could not read image at {right_img_path}")
        except Exception as e:
            print(f"Error loading image {right_img_path}: {e}")
            return
    
        detected_markers = marker_calibration.detect_apriltags_positions(
                left_img, right_img, 
                left_camera_matrix, right_camera_matrix, 
                left_dist_coeffs, right_dist_coeffs, 
                P1, R1, P2, R2, 
                marker_length,
                stereo=True
            )

    image_data[camera_number] = {}
    
    for marker_id, pose in detected_markers.items():
        if pose["pixel_error"] >= PIXEL_ERROR_THRESHOLD:
            continue

        image_data[camera_number][marker_id] = {
            "img_center": np.asarray(pose["img_center"], dtype=np.float64).reshape(2),
            "t": np.asarray(pose["t"], dtype=np.float64).reshape(3),
            "R": np.asarray(pose["R"], dtype=np.float64),
            "weight": (PIXEL_ERROR_THRESHOLD - pose["pixel_error"]) / PIXEL_ERROR_THRESHOLD,
            }

    return image_data

def build_camera_dataset_pnp(cam_drives, capture_name, frame_number, marker_data, marker_length, stereo=False):
    """find pose for each image using ransac pnp method"""
    camera_data = {}

    for cam_drive in cam_drives:
        capture_directory = os.path.join("/mnt", cam_drive, "captures", capture_name)
        pattern = os.path.join(capture_directory, "color_a", f"*{frame_number}.tiff")
        image_files = glob.glob(pattern)

        if len(image_files) == 0:
            print("No images found in", capture_directory)
            continue
        
        for image_file in image_files:
            camera_id = int(os.path.splitext(os.path.basename(image_file))[0].split('_')[1])
            print("Solving camera pose for camera ID:", camera_id)
            camera_dataset = extract_camera_data(image_file, marker_length, camera_id, stereo=stereo)
            
            R, t, reproj_error = marker_calibration.find_camera_pose_pnp(camera_dataset, marker_data, camera_id, camera_id)
            print("reprojection error", reproj_error)
            print()

            camera_data[camera_id] = {
                "R": R,
                "t": t,
            }

    return camera_data

def refine_stereo_positions(camera_data):
    """use calcualted stereo offset and rotation to refine position (not currently in use)"""

    refined_positions = {}

    for cam_id, pose in camera_data.items():
        # Fill cam positions with current position
        refined_positions[cam_id] = {
        "R": camera_data[cam_id]["R"],
        "t": camera_data[cam_id]["t"],
        }

        # Check if camera is part of stero pair by it ending in 5
        if int(cam_id) % 10 == 5:
            # establish left id
            cam_a_id = int(cam_id) - 5
            #establish right id
            cam_b_id = int(cam_id)

            # Load stereo offsets from left to right camera
            R, T = stereo_depth.load_stereo_params(cam_a_id)

            # find left camera position given right position and stereo offset
            cam_a_r = R @ camera_data[cam_b_id]["R"]
            cam_a_t = camera_data[cam_b_id]["R"] @ T + camera_data[cam_b_id]["t"]
            
            T_a_stack = np.vstack([(camera_data[cam_a_id]["t"]), cam_a_t])
            averaged_cam_a_t = np.mean(T_a_stack, axis=0)

            R_a_stack = np.stack([camera_data[cam_a_id]["R"], cam_a_r])
            R_a_mean = np.mean(R_a_stack, axis=0)
            U_a, _, Vt_a = np.linalg.svd(R_a_mean)
            averaged_cam_a_r = U_a @ Vt_a

            cam_b_r = R.T @ averaged_cam_a_r
            cam_b_t = averaged_cam_a_r @ -T + averaged_cam_a_t
    
            # Fill cam positions with updated positions
            refined_positions[cam_a_id] = {
            "R": averaged_cam_a_r,
            "t": np.vstack(averaged_cam_a_t),
            }

            refined_positions[cam_b_id] = {
            "R": cam_b_r,
            "t": np.vstack(cam_b_t),
            }

    return refined_positions

def output_colmap_images(avg_camera_poses, output_file="images.txt"):
    """Write averaged (or shifted) camera poses to COLMAP images.txt format."""

    with open(output_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        image_id = 1
        for cam_id, pose in avg_camera_poses.items():
            try:
                cam_id_int = int(cam_id)
            except (ValueError, TypeError):
                continue  # skip non-numeric IDs
            
            try:
                Rcw = pose["R"]
            except (AttributeError, KeyError):
                continue  # skip if R is not available

            try:
                tcw = pose["t"].flatten()
            except (AttributeError, KeyError):
                continue  # skip if t is not available

            # Invert to world-to-camera
            Rwc = Rcw.T
            twc = -Rcw.T @ tcw

            # Quaternion (scipy: x,y,z,w order)
            rot = Rotation.from_matrix(Rwc)
            qx, qy, qz, qw = rot.as_quat()

            image_name = f"Camera_{cam_id_int:04d}.png"

            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{twc[0]:.6f} {twc[1]:.6f} {twc[2]:.6f} {cam_id_int} {image_name}\n")
            f.write("\n")
            image_id += 1

def visualize_cameras_and_markers(camera_poses, marker_data, scale=1.0):
    """display camera positions with open3d viewer"""
    geometries = []

    # Draw cameras
    for cam_id, pose in camera_poses.items():
        R = pose["R"]
        t = pose["t"].flatten()
        if int(cam_id) % 10 == 5:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.7)
        else:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 1.0)
        cam_frame.rotate(R, center=(0,0,0))
        cam_frame.translate(t)
        geometries.append(cam_frame)

    # Draw markers
    for marker_id, mdata in marker_data.items():
        pos = mdata["t"]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale*0.5)
        sphere.paint_uniform_color([1, 0, 0])  # red for markers
        sphere.translate(pos)
        geometries.append(sphere)
    
    # Visualize everything
    o3d.visualization.draw_geometries(geometries)

def main():
    parser = argparse.ArgumentParser(description="Calculate intrinsic camera parameters.")
    parser.add_argument(
        "--capture-name",
        type=str,
        default="16viewsAprilTags",
        help="Name of the capture directory inside the camera folders (default is BlenderCalibrationTake)"
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.25,
        help="Size of marker in meters (default is 0.18)"
    )
    parser.add_argument(
        "--stereo",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True applies stereo processing, false applies no stereo processing"
    )

    args = parser.parse_args()

    scaled_marker_length = args.marker_length

    mnt_dir = os.path.join("/mnt")

    cam_drives = sorted([
        f for f in os.listdir(mnt_dir)
        if os.path.isdir(os.path.join(mnt_dir, f))
        and f.startswith("cam_")
    ])

    calib_dir = os.path.join(os.path.dirname(os.getcwd()), 'calibration')
    
    print("Loading marker poses from markers.txt...")
    marker_data = load_marker_file(calib_dir)

    print("Building camera dataset from images...")
    print()
    frame_number = "cal" # use high resolution calibration frame for better quality
    camera_data = build_camera_dataset_pnp(cam_drives, args.capture_name, frame_number, marker_data, scaled_marker_length, args.stereo)
    print("Outputting shifted camera poses to COLMAP format...")
    output_colmap_images(camera_data, output_file=os.path.join(calib_dir, "images.txt"))

    #visualize_cameras_and_markers(camera_data, marker_data, scale=0.1)


if __name__ == "__main__":
    main()
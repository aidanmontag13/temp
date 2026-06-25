import dis
import random
import cv2
import numpy as np
import os
import glob
import argparse
import pprint
import pupil_apriltags as apriltag
import json
import copy

from scipy import stats
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import open3d as o3d

import process_images
import intrinsic_calibration

from scipy.optimize import least_squares

# filtering thresholds
PIXEL_ERROR_THRESHOLD = 1.0  # pixels - from rectification, reject y disparity
REPROJ_ERROR_THRESHOLD = 2.0 # pixels - from pnp
REPROJ_ERROR_THRESHOLD_METERS = 0.5  # meters - from main pointcloud

# initialize apriltag detector
# nb- will core dump if not global
DETECTOR  = apriltag.Detector(
        families="tag36h11")
    
def rectify_points(points, camera_matrix, distoertion_coeffs, P, R):
    """ Take a series of 2d image points and rectify given stereo parameters """

    pts = np.array(points, dtype=np.float32)
    
    pts = pts.reshape(-1, 1, 2)
    
    rect_pts = cv2.undistortPoints(
        src=pts,
        cameraMatrix=camera_matrix,
        distCoeffs=distoertion_coeffs,
        R=R,
        P=P
    )
    
    return rect_pts.reshape(-1, 2)

def detect_apriltags_positions(left_img, right_img, left_camera_matrix, right_camera_matrix, left_distortion_coeffs, right_distortion_coeffs, P1, R1, P2, R2, tag_size, stereo=True):
    """ given a set of stereo input images, camera intrinsics and stereo parameters, returns the xyz position in 3d space and xy position in image space of apriltag markers"""

    detector = DETECTOR
    # Convert to grayscale if needed
  
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    left_detections = detector.detect(left_gray)
    
    if stereo:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_detections = detector.detect(right_gray)
        right_lookup = {int(det.tag_id): det for det in right_detections}


    results = {}

    # Define marker corners in 3D (object coordinate system)
    # mystical shrinking scale constant to match absolute depth
    half_tag_size = (tag_size * 0.972) / 2.0

    obj_points = np.array([
        [-half_tag_size,  half_tag_size, 0.0],
        [ half_tag_size,  half_tag_size, 0.0],
        [ half_tag_size, -half_tag_size, 0.0],
        [-half_tag_size, -half_tag_size, 0.0]
    ], dtype=np.float64)

    # using data from the right camera to help us with the left, in this case
    # could do either/both
    for left_det in left_detections:
        tag_id = int(left_det.tag_id)

        # Get detected corner points and rectify using stereo params
        left_img_corners = left_det.corners.astype(np.float64)
        rect_left_img_corners = rectify_points(left_img_corners, left_camera_matrix, left_distortion_coeffs, P1, R1)
        
        left_img_center = left_det.center.astype(np.float64)
        rect_left_img_center = rectify_points(left_img_center, left_camera_matrix, left_distortion_coeffs, P1, R1)[0]
       
        # p1 is rectification matrix + translation from stereo
        new_camera_matrix = P1[:,:3]
        
        # solve pnp - first pass
        success, rvec, tvec = cv2.solvePnP(
            obj_points, 
            rect_left_img_corners, 
            new_camera_matrix,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue

        # Refine the solution iteratively using Levenberg-Marquardt optimization
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_points, 
            rect_left_img_corners, 
            new_camera_matrix, 
            None, 
            rvec, 
            tvec
        )
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Compute reprojection error
        projected_points, _ = cv2.projectPoints(
            obj_points, 
            rvec, 
            tvec, 
            new_camera_matrix, 
            None
        )

        # Compute reprojection error - not using the error, but computing it for debugging
        projected_points = projected_points.reshape(-1, 2)
        reproj_error = np.sqrt(np.mean(np.sum((rect_left_img_corners - projected_points)**2, axis=1)))

        if stereo:
            right_det = right_lookup.get(tag_id)

            if right_det is None:
                continue

            # Get detected corner points and rectify using stereo params
            right_img_corners = right_det.corners.astype(np.float64)
            rect_right_img_corners = rectify_points(right_img_corners, right_camera_matrix, right_distortion_coeffs, P2, R2)
       
            right_img_center = right_det.center.astype(np.float64)
            rect_right_img_center = rectify_points(right_img_center, right_camera_matrix, right_distortion_coeffs, P2, R2)[0]

            #currently we only use pnp to solve marker orientation
            ptsL = rect_left_img_corners.T 
            ptsR = rect_right_img_corners.T

            # use triangulate points to get 3d position of corners, average to find center point
            points4D = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
            points3D = points4D[:3] / points4D[3]
            points3D = np.mean(points3D, axis=1)

            # Disparity - rectification error
            xL, yL = rect_left_img_center
            xR, yR = rect_right_img_center

            rectification_error = np.clip(np.abs(yL - yR), 0, 1)
            pixel_error = rectification_error

            t = points3D

        else:
            t = tvec.reshape(3, 1)
            pixel_error = reproj_error

        # add to dataset
        results[tag_id] = {
            "img_center": rect_left_img_center,
            "R": R,
            "t": t,
            "pixel_error": pixel_error
        }

    return results

def build_image_dataset(images_dir, marker_length, stereo=False):
    """Build master dataset of marker observations from images, organized by image ID."""
    image_data = {}

    # Collect all image files
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + \
              glob.glob(os.path.join(images_dir, "*.tiff"))

    image_files = [
        f for f in image_files
        if "_cal" not in os.path.basename(f)
    ]
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))

    num_cameras = len(image_files)

    last_img_1 = None
    last_img_2 = None

    i = 0

    for img_file in image_files:
        print(f"\rFinding Apriltags in image {i+1} of {len(image_files)}", end="", flush=True)
        i += 1
        # Parse camera id from filename. Expected format: Camera_<id>_<frame>
        base = os.path.splitext(os.path.basename(img_file))[0]
        camera_number = int(base.split('_')[1])
        camera_id = camera_number // 10
        try:
            image_id = int(base.split('_')[2])
        except ValueError:
            continue

        left_img = cv2.imread(img_file)

        if left_img is None:
            print(f"Warning: Could not read image {img_file}. Skipping.")
            continue
        
        redundant = intrinsic_calibration.filter_redundant_frames(last_img_1, last_img_2, left_img)
        last_img_2 = last_img_1
        last_img_1 = left_img

        if redundant == True:
            continue

        if stereo == True:
            #print(f"Processing stereo image pair for camera {camera_number}, image ID {image_id}")
            right_img_file = process_images.find_right_img_path(img_file, camera_number)

            right_img = cv2.imread(right_img_file)
            if right_img is None:
                print()
                print(f"Warning: Could not read image {right_img_file}. Skipping.")
                continue

            # Load camera parameters, scale 1 since we are using 4k images
            try:
                left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, width, height, R, T, P1, P2, R1, R2 = process_images.load_camera_params_json(camera_id, 1, stereo=True)

            except (FileNotFoundError, ValueError) as e:
                print()
                print(f"Error loading camera parameters for camera ID {camera_number}: {e}")
                continue

            # acquire marker positions from frames
            detected_markers = detect_apriltags_positions(
                left_img, right_img, 
                left_camera_matrix, right_camera_matrix, 
                left_dist_coeffs, right_dist_coeffs, 
                P1, R1, P2, R2, 
                marker_length,
                stereo=True
            )

        else:
            # Load camera parameters, scale 1 since we are using 4k images
            try:
                left_camera_matrix, left_dist_coeffs, width, height, P1 = process_images.load_camera_params_json(camera_id, 1, stereo=False)

            except (FileNotFoundError, ValueError) as e:
                print()
                print(f"Error loading camera parameters for camera ID {camera_number}: {e}")
                continue

            detected_markers = detect_apriltags_positions(
                left_img, None,
                left_camera_matrix, None,
                left_dist_coeffs, None,
                P1, None, None, None,
                marker_length,
                stereo=False
            )

        #filter empty data
        if image_id not in image_data:
            image_data[image_id] = {}

        for marker_id, pose in detected_markers.items():
            # filter markers with too high rectification error
            if pose["pixel_error"] >= PIXEL_ERROR_THRESHOLD:
                #print("skipping marker, rect error", pose["rectification_error"] )
                continue
            # add markers to image dataset
            else:
                image_data[image_id][marker_id] = {
                    "img_center": np.asarray(pose["img_center"], dtype=np.float64).reshape(2),
                    "t": np.asarray(pose["t"], dtype=np.float64).reshape(3),
                    "R" : np.asarray(pose["R"], dtype=np.float64),
                    "weight": (PIXEL_ERROR_THRESHOLD - pose["pixel_error"]) / PIXEL_ERROR_THRESHOLD,
                }
        #print(f"Detected {len(detected_markers)} markers in image {img_file} with reprojection error below threshold.")
    
    print()    
    print(f"Processed {len(image_files)} images and detected markers in {len(image_data)} unique image IDs.")

    return image_data, num_cameras, P1[:,:3]

def convert_markerdata_to_pcd(marker_data, camera_id, shared_markers):
    """convert markerdata for specific camera image into open3D pcd data structure"""
    points = []
    normals = []

    # only add shared markers to pcd (for registration with pointcloud)
    if shared_markers is None:
        marker_range = marker_data.keys()
    else:
        marker_range = shared_markers

    for marker_id in marker_range:

        if camera_id is None:
            camera_markers = marker_data
        else:
            camera_markers = marker_data[camera_id]

        obs = camera_markers[marker_id]

        # Load normals from rotation matrix
        t = np.asarray(obs["t"], dtype=np.float64).reshape(3)
        R = np.asarray(obs["R"], dtype=np.float64)

        normal = R[:, 2]
        normal = normal / np.linalg.norm(normal) # normalize

        points.append(t)
        normals.append(normal)

    points = np.stack(points, axis=0)
    normals = np.stack(normals, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd

def find_shared_markers(image_data, marker_data, camera_id):
    """ function to create list of shared markers between image points and marker points """
    image_data = image_data[camera_id]
    
    shared_markers = []

    for marker_id in image_data.keys():

        if (marker_id) in marker_data:
            shared_markers.append(marker_id)

    return sorted(shared_markers)

def find_camera_pose_pnp(image_data, marker_data, image_id, camera_id):
    """ use ransac pnp method to estimate camera position given marker dataset """

    # load rectified camera matrix
    left_camera_matrix, left_dist_coeffs, width, height, P1 = process_images.load_camera_params_json(camera_id // 10, 1, stereo=False)
    new_camera_matrix = P1[:,:3]
    
    # find shared markers between image points and marker points
    shared_markers = find_shared_markers(image_data, marker_data, image_id)

    # ensure sufficient number of shared markers for pnp
    if len(shared_markers) < 5:
        print("Not enough shared markers for pose estimation on camera", camera_id)
        return None, None, None

    img_points = []
    obj_points = []

    # build img points and obj points data structure for pnp
    for marker in shared_markers:

        img_center = np.asarray(
            image_data[image_id][marker]["img_center"],
            dtype=np.float64
        ).reshape(2)
        
        img_points.append(img_center)

        t = marker_data[marker]["t"].reshape(3)
        obj_points.append(t)

    obj_points = np.array(obj_points, dtype=np.float32).reshape(-1,3)
    img_points = np.array(img_points, dtype=np.float32).reshape(-1,2)

    # run pnp
    # parameters are from the synthetic data, so might need tweaking for real data
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_points,
        img_points,
        new_camera_matrix,
        None,
        reprojectionError=2.0,
        iterationsCount=100,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # refine using inliers only
    if inliers is not None:
        obj_inliers = obj_points[inliers.flatten()]
        img_inliers = img_points[inliers.flatten()]
    else:
        obj_inliers = obj_points
        img_inliers = img_points

    # refine using inliers
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_inliers,
        img_inliers,
        new_camera_matrix,
        None,
        rvec,
        tvec
    )

    # find reprojection error
    projected_points, _ = cv2.projectPoints(
        obj_inliers,
        rvec,
        tvec,
        new_camera_matrix,
        None
    )

    projected_points = projected_points.reshape(-1,2)

    # use the reprojection error as we iterate to filter
    reproj_error = np.sqrt(np.mean(np.mean((img_inliers - projected_points)**2, axis=1)))

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    R = R.T
    t = -R @ t
    t = t.reshape(1,3)
    
    return R, t, reproj_error

def weighted_average_two_rotations(R1, R2, weight):
    """ average two rotation matricies based on weight """
    try:
        w = 1/(weight + 1.0)
        key_times = [0, 1]
        rots = Rotation.from_matrix([R1, R2])

        slerp = Slerp(key_times, rots)
        R_avg = slerp([w])[0].as_matrix()
    except Exception as e:
        print(f"Error occurred while averaging rotations: {e}")
        R_avg = R1  # Fallback to the first rotation

    return R_avg

def append_camera_markers(image_data, marker_data, camera_id, R, t):
    """ function to merge two point clouds based on weight and input transform"""
    camera_markers = image_data[camera_id]
    error_list = []

    for marker_id in list(camera_markers.keys()):
        obs = camera_markers[marker_id]
        
        # apply transform to image data
        camera_t = np.asarray(obs["t"], dtype=np.float64)
        shifted_camera_t = R @ camera_t + t
        shifted_camera_R = R @ obs["R"]
        shifted_camera_weight = obs["weight"]

        # average point positions based on weight
        if marker_id in marker_data:
            marker_t = marker_data[marker_id]["t"]
            weight = marker_data[marker_id]["weight"]
            average_t = ((marker_t.reshape(3) * weight) + shifted_camera_t.reshape(3) * shifted_camera_weight) / (weight + shifted_camera_weight)
            marker_R = marker_data[marker_id]["R"]

            # interpolate rotation
            average_R = weighted_average_two_rotations(marker_R, shifted_camera_R, weight)

            error_t = marker_t.reshape(3) - shifted_camera_t.reshape(3)
            error_norm = np.linalg.norm(error_t)
            error_list.append(error_norm)

            # filter points with exceptionally high variation
            if error_norm > REPROJ_ERROR_THRESHOLD_METERS:
                #print(f"Warning: High reprojection error for marker {marker_id} from camera {camera_id}: {error_norm:.6f} meters. Skipping this marker.")
                continue

        else:
            # add new markers with weight of 0
            #print(f"Adding new marker {marker_id} from camera {camera_id}")
            average_t = shifted_camera_t
            average_R = shifted_camera_R
            weight = shifted_camera_weight
    
        marker_data[marker_id] = {
            "t": average_t.astype(np.float64),
            "R": average_R.astype(np.float64),
            "weight": weight + shifted_camera_weight
        }

    # find average error across all merged markers
    average_error = np.mean(error_list) if error_list else 0.0
    
    return marker_data, average_error

def solve_marker_poses(image_data, marker_data, camera_data, camera_id):
    """ function to perform camera location estimation, marker projection and pointcloud merging for simple interation"""
    error_list = []
    reprojection_error_list = []
    for image_id in image_data.keys():
        # locate camera position (currently using pnp method)
        R, t, repojection_error = find_camera_pose_pnp(image_data, marker_data, image_id, camera_id)
        #R, t, repojection_error = find_camera_pose(image_data, marker_data, image_id)
        if t is None:
            #print(f"Could not find pose for camera {camera_id}. Skipping.")
            continue
        # filter translations with high reprojection error
        if repojection_error > REPROJ_ERROR_THRESHOLD:
            #print("reprojection error too high, ", repojection_error, " skipping this camera")
            continue
        
        # append image markers using estimated translation
        marker_data, error = append_camera_markers(image_data, marker_data, image_id, R, t)

        camera_data[image_id] = {
                    "t": np.asarray(t, dtype=np.float64).reshape(3),
                    "R" : np.asarray(R, dtype=np.float64),
                }
        
        error_list.append(error)
        reprojection_error_list.append(repojection_error)

    # find average error
    avg_error = np.mean(error_list) if error_list else 0.0
    avg_reprojection_error = np.mean(reprojection_error_list) if reprojection_error_list else 0.0

    return marker_data, camera_data, avg_error, avg_reprojection_error

def filter_bad_poses(marker_data, camera_data, image_data, camera_id):
    """ objective function to test reprojection error of optimized marker and camera poses"""
    marker_data_copy = copy.deepcopy(marker_data)
    camera_data_copy = copy.deepcopy(camera_data)

    reprojection_error_list = []

    # acquire rectified camera matrix
    left_camera_matrix, left_dist_coeffs, width, height, P1 = process_images.load_camera_params_json(camera_id // 10, 1, stereo=False)
    new_camera_matrix = P1[:,:3]

    for image_id in camera_data.keys():
        if image_id not in camera_data_copy:
            continue
        # find shared marker between image points and marker map
        shared_markers = find_shared_markers(image_data, marker_data, image_id)
    
        # extract rotation and translation of camera and convert to opencv rvec and tvec
        t = camera_data_copy[image_id]["t"]
        R = camera_data_copy[image_id]["R"]

        R_cv = R.T
        t_cv = -R_cv @ t.reshape(3, 1)

        rvec, _ = cv2.Rodrigues(R_cv)
        tvec = t_cv.reshape(3,1)

        # create object points and image points based on shared markers
        img_points = []
        obj_points = []

        for marker in shared_markers:

            img_center = np.asarray(
                image_data[image_id][marker]["img_center"],
                dtype=np.float64
            ).reshape(2)
        
            img_points.append(img_center)

            t = marker_data_copy[marker]["t"].reshape(3)
            obj_points.append(t)

        obj_points = np.array(obj_points, dtype=np.float32).reshape(-1,3)
        img_points = np.array(img_points, dtype=np.float32).reshape(-1,2)

        # project 3d object points to 2D based on camera matrix
        projected_points, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            new_camera_matrix,
            None
        )

        projected_points = projected_points.reshape(-1,2)
        
        # find reprojection error between projected 3d points and image points
        reprojection_errors = (img_points - projected_points).flatten()
        print(f"image {image_id} reprojection error (RMSE): {np.sqrt(np.mean(reprojection_errors**2)):.4f} pixels")

        reprojection_error_list.append(reprojection_errors)
    
    #concatinate into residual error vector
    residuals = np.concatenate(reprojection_error_list)

    return residuals


def output_marker_file(marker_data, output_file="markers.txt"):
    """Write averaged marker poses to markers.txt in order by marker ID"""
    with open(output_file, 'w') as f:
        f.write("# Marker list with two lines of data per marker:\n")
        f.write("# MARKER_ID,T1, T2, T3, R1, R2, R3, R4, R5, R6, R7, R8, R9\n")

        # Sort keys by numeric marker ID
        for marker_id in sorted(marker_data.keys()):
            tcw = marker_data[marker_id]["t"].flatten()
            rcw = marker_data[marker_id]["R"].flatten()

            f.write(
                f"{marker_id} "
                f"{tcw[0]:.6f} {tcw[1]:.6f} {tcw[2]:.6f} {rcw[0]:.6f} {rcw[1]:.6f} {rcw[2]:.6f} {rcw[3]:.6f} {rcw[4]:.6f} {rcw[5]:.6f} {rcw[6]:.6f} {rcw[7]:.6f} {rcw[8]:.6f}\n"
            )

def reset_marker_data_weights(marker_data, value=1.0):
    """ reset marker weights to fixed value while iterating"""
    for marker_id in marker_data:
        marker_data[marker_id]["weight"] = value

def multiply_marker_data_weights(marker_data, value=1.0):
    """ multiply marker weights by fixed value while iterating """
    for marker_id in marker_data:
        marker_data[marker_id]["weight"] = marker_data[marker_id]["weight"] * value

def convert_dataset_to_vector(marker_data, camera_data):
    """ function to convert marker data and camera dataset into 1d vector for optimization"""
    marker_t_rows = []
    camera_t_rows = []
    camera_R_rows = []
    
    for marker_id in sorted(marker_data.keys()):
        t = np.array(marker_data[marker_id]["t"]).reshape(-1)  #
        marker_t_rows.append(t)

    marker_array = np.array(marker_t_rows, dtype=float)
    marker_vector = marker_array.flatten()

    for camera_id in sorted(camera_data.keys()):
        t = np.array(camera_data[camera_id]["t"]).reshape(-1)
        camera_t_rows.append(t)

        R = np.array(camera_data[camera_id]["R"])
        rvec = cv2.Rodrigues(R)[0].flatten()  # convert rotation matrix to rotation vector
        camera_R_rows.append(rvec)

    camera_t_array = np.array(camera_t_rows, dtype=float)
    camera_t_vector = camera_t_array.flatten()

    camera_R_array = np.array(camera_R_rows, dtype=float)
    camera_R_vector = camera_R_array.flatten()

    data_vector = np.concatenate([marker_vector, camera_t_vector, camera_R_vector])

    return data_vector

def convert_vector_to_dataset(data_vector, marker_data, camera_data):
    """ function to convert 1D vector back into marker and camera dataset for objective function"""

    # Number of markers and cameras
    n_markers = len(marker_data)
    n_cameras = len(camera_data)

    # extract marker translations
    marker_vec_len = n_markers * 3
    marker_array = data_vector[:marker_vec_len].reshape(n_markers, 3)

    for idx, marker_id in enumerate(sorted(marker_data.keys())):
        marker_data[marker_id]["t"] = marker_array[idx].copy()

    # extract camera translations and rotations
    cam_t_start = marker_vec_len
    cam_t_len = n_cameras * 3
    cam_R_start = cam_t_start + cam_t_len
    cam_R_len = n_cameras * 3  # assuming 3x3 rotation matrices flattened

    camera_t_array = data_vector[cam_t_start:cam_t_start+cam_t_len].reshape(n_cameras, 3)
    camera_R_array = data_vector[cam_R_start:cam_R_start+cam_R_len].reshape(n_cameras, 3)

    for idx, camera_id in enumerate(sorted(camera_data.keys())):
        camera_data[camera_id]["t"] = camera_t_array[idx].copy()
        camera_data[camera_id]["R"] = cv2.Rodrigues(camera_R_array[idx].copy())[0] # convert rotation vector back to rotation matrix

    return marker_data, camera_data

def get_reprojection_residuals(data_vector, image_data, marker_data, camera_data, camera_id):
    """ objective function to test reprojection error of optimized marker and camera poses"""
    marker_data_copy = copy.deepcopy(marker_data)
    camera_data_copy = copy.deepcopy(camera_data)

    # convert data vector into marker and camera datasets
    marker_data_copy, camera_data_copy = convert_vector_to_dataset(data_vector, marker_data_copy, camera_data_copy)

    reprojection_error_list = []

    # acquire rectified camera matrix
    left_camera_matrix, left_dist_coeffs, width, height, P1 = process_images.load_camera_params_json(camera_id // 10, 1, stereo=False)
    new_camera_matrix = P1[:,:3]

    for image_id in image_data.keys():
        if image_id not in camera_data_copy:
            continue
        # find shared marker between image points and marker map
        shared_markers = find_shared_markers(image_data, marker_data, image_id)
    
        # extract rotation and translation of camera and convert to opencv rvec and tvec
        t = camera_data_copy[image_id]["t"]
        R = camera_data_copy[image_id]["R"]

        R_cv = R.T
        t_cv = -R_cv @ t.reshape(3, 1)

        rvec, _ = cv2.Rodrigues(R_cv)
        tvec = t_cv.reshape(3,1)

        # create object points and image points based on shared markers
        img_points = []
        obj_points = []

        for marker in shared_markers:

            img_center = np.asarray(
                image_data[image_id][marker]["img_center"],
                dtype=np.float64
            ).reshape(2)
        
            img_points.append(img_center)

            t = marker_data_copy[marker]["t"].reshape(3)
            obj_points.append(t)

        obj_points = np.array(obj_points, dtype=np.float32).reshape(-1,3)
        img_points = np.array(img_points, dtype=np.float32).reshape(-1,2)

        # project 3d object points to 2D based on camera matrix
        projected_points, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            new_camera_matrix,
            None
        )

        projected_points = projected_points.reshape(-1,2)
        
        # find reprojection error between projected 3d points and image points
        reprojection_errors = (img_points - projected_points).flatten()
        #print(f"image {image_id} reprojection error (RMSE): {np.sqrt(np.mean(reprojection_errors**2)):.4f} pixels")

        reprojection_error_list.append(reprojection_errors)
    
    #concatinate into residual error vector
    residuals = np.concatenate(reprojection_error_list)

    return residuals


def optimize_marker_data(marker_data, camera_data, image_data, camera_id):
    """ function that inputs marker and camera data, and uses least squares regression to minimize repjection error within the dataset"""

    #convert markerdata and camera data into n,1 vector
    data_vector = convert_dataset_to_vector(marker_data, camera_data)

    # run optimization
    result = least_squares(
        get_reprojection_residuals,
        data_vector,
        args=(image_data, marker_data, camera_data, camera_id),
        method='trf',        # Ensure Trust Region Reflective is utilized
        loss='huber',        # Protects against bad marker detections
        f_scale=1.5,         # Threshold where loss becomes linear (in pixels)
        x_scale='jac',       # Dynamically handles the different scales during optimization
        ftol=1e-7,
        xtol=1e-7,
        verbose=2,
    )
    marker_vector = result.x
    success = result.success
    print("Optimization success:", success)
        
    print("Final Total Cost:", result.cost) 

    # convert vector back into data structure
    marker_data, camera_data = convert_vector_to_dataset(marker_vector, marker_data, camera_data)

    return marker_data, camera_data

def rescale_marker_data(optimized_marker_data, marker_data):
    optimized_pcd = convert_markerdata_to_pcd(optimized_marker_data, None, None)
    original_pcd = convert_markerdata_to_pcd(marker_data, None, None)

    optimized_marker_datacenter = optimized_pcd.get_center()
    optimized_aabb = optimized_pcd.get_axis_aligned_bounding_box()
    optimized_extent = optimized_aabb.get_extent()

    original_marker_datacenter = original_pcd.get_center()
    original_aabb = original_pcd.get_axis_aligned_bounding_box()
    original_extent = original_aabb.get_extent()

    scale_factors = original_extent / optimized_extent
    print(f"Original extent: {original_extent}, Optimized extent: {optimized_extent}, Scale factors: {scale_factors}")
    scale_factor = np.median(scale_factors)
    print(f"Rescaling optimized marker data by factor of {scale_factor:.6f} to match original scale.")
    for marker_id in optimized_marker_data:
        optimized_marker_data[marker_id]["t"] = (optimized_marker_data[marker_id]["t"] - optimized_marker_datacenter) * scale_factor + original_marker_datacenter

    return optimized_marker_data

def main():
    parser = argparse.ArgumentParser(description="Calculate intrinsic camera parameters.")
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="ID of camera used for calibration take"
    )
    parser.add_argument(
        "--capture-name",
        type=str,
        default="flythroughAprilTag",
        help="Name of the capture directory inside the camera folders (default is BlenderCalibrationTake)"
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.254,
        help="Size of marker in meters (default is 0.18)"
    )

    parser.add_argument(
        "--stereo",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True applies stereo processing, false applies no stereo processing"
    )

    args = parser.parse_args()

    camera_number = args.camera_id * 10

    print("camera_id", args.camera_id)

    mnt_dir = os.path.join(f"/mnt/cam_{args.camera_id:02d}/captures")

    images_dir = os.path.join(mnt_dir, args.capture_name, "color_a")

    calib_dir = os.path.join(os.path.dirname(os.getcwd()), 'calibration')

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # build initial point dataset from images
    print(f"Processing images from: {images_dir}")
    image_data, num_images, new_camera_matrix = build_image_dataset(images_dir, args.marker_length, args.stereo)

    print("solving marker poses")

    start_image_id = min(image_data.keys())
    marker_data = copy.deepcopy(image_data[start_image_id])
    camera_data = {}

    # solve for marker and camera positions using pnp method
    marker_data, camera_data, error, reprojection_error = solve_marker_poses(image_data, marker_data, camera_data, camera_number)

    # test initial average reprojection error
    data_vector = convert_dataset_to_vector(marker_data, camera_data)
    reprojection_residuals = get_reprojection_residuals(data_vector, image_data, marker_data, camera_data, camera_number)
    print("number of residuals", reprojection_residuals.size)
    print("Initial Total Cost:", np.sum(np.abs(reprojection_residuals)))
    print("max residual", np.max(np.abs(reprojection_residuals)))
    print("median of residuals", np.median(np.abs(reprojection_residuals)))
    print("mean of residuals", np.mean(np.abs(reprojection_residuals)))

    # optimize camera data and marker data to minimize reprojection error
    optimized_marker_data, optimized_camera_data = optimize_marker_data(marker_data, camera_data, image_data, camera_number)

    rescaled_marker_data = rescale_marker_data(optimized_marker_data, marker_data)

    # test average reprojection error with optimized data
    data_vector = convert_dataset_to_vector(rescaled_marker_data, optimized_camera_data)
    reprojection_residuals = get_reprojection_residuals(data_vector, image_data, rescaled_marker_data, optimized_camera_data, camera_number)
    print("number of residuals 2", reprojection_residuals.size)
    print("max residual 2", np.max(np.abs(reprojection_residuals)))
    print("median of residuals 2", np.median(np.abs(reprojection_residuals)))
    print("mean of residuals 2", np.mean(np.abs(reprojection_residuals)))

    # export markers.txt file
    print("Outputting markers.txt")
    markers_file = os.path.join(calib_dir, 'markers.txt')
    output_marker_file(marker_data, markers_file)

    # display marker map
    pointcloud = convert_markerdata_to_pcd(marker_data, None, None)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pointcloud, origin], point_show_normal=True)

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
import glob
import argparse
import json

SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
SUBPIXEL_SIZE = (5, 5)

MONO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
MONO_FLAGS = ()

STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-7)
STEREO_FLAGS = (cv2.CALIB_FIX_INTRINSIC)


def filter_redundant_frames(last_image_1, last_image_2, img):
    if last_image_1 is None or last_image_2 is None or img is None:
        return True

    last_1_gray = cv2.cvtColor(last_image_1, cv2.COLOR_BGR2GRAY)
    last_2_gray = cv2.cvtColor(last_image_2, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    last_1_diff = cv2.absdiff(last_2_gray, last_1_gray)
    #print("    Last diff mean:", last_diff.mean())
    last_2_diff = cv2.absdiff(last_1_gray, gray)
    #print("    Next diff mean:", next_diff.mean())

    # if the image hasn't moved much from the last frame but moves significantly in the next frame, it's not redundant and should be kept for calibration
    if last_1_diff.mean() > 3 and last_2_diff.mean() < 3:
        return False
    else:
        return True
    
def create_checkerboard_object_points(checkerboard_width, checkerboard_length, square_size):
    object = np.zeros((checkerboard_width * checkerboard_length, 3), np.float32)
    object[:, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_length].T.reshape(-1, 2)
    object *= square_size  # apply real world scale
    
    return object


def find_checkerboard_corners(base_dir, images, camera_id, checkerboard_width, checkerboard_length, square_size, tripod_mode=True):
    """Find checkerboard corners in images for camera calibration, filtering low-quality detections for second calibration pass"""

    # Create checkerboard object points based on provided dimensions and square size
    objp = create_checkerboard_object_points(checkerboard_width, checkerboard_length, square_size)

    # initialize dictionary to hold checkerboard corners for each imageid
    checkerboard_corners = {}

    print(f"  Processing {len(images)} images for camera {camera_id}...")


    last_img_1_path = os.path.join(base_dir, images[0])
    last_img_1 = cv2.imread(last_img_1_path)

    last_img_2_path = os.path.join(base_dir, images[1])
    last_img_2 = cv2.imread(last_img_2_path)

    img_size = (last_img_1.shape[1], last_img_1.shape[0])


    # extract checkerboard corners from each image and store in dictionary with frame id as key
    for i in range(2, len(images)):
        print(f"\rFinding checkerboard corners in image {i+1} of {len(images)}", end="", flush=True)
        img_path = os.path.join(base_dir, images[i])

        base = os.path.splitext(os.path.basename(images[i]))[0]
        try:
            image_id = int(base.split('_')[2])
        
        except ValueError:
            continue
        
        img = cv2.imread(img_path)

        if img is None:
            print(f"    Warning: Could not load image {img_path}")
            continue

        if tripod_mode:
            # Skip image if moved from last frame, but did not have 1 frame to settle
            redundant = filter_redundant_frames(last_img_1, last_img_2, img)
            last_img_2 = last_img_1.copy()
            last_img_1 = img.copy()

            if redundant:
                #print(f"    Skipping redundant frame {img_path}")
                continue

        #else:
        #    print(f"    Processing frame {img_path}")
        
        # Convert image to grayscale for corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # intially test with downscaled image to speed up corner detection
        down_scaled_gray = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))

        down_scaled_ret, down_scaled_corners = cv2.findChessboardCorners(
            down_scaled_gray, (checkerboard_width,checkerboard_length),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not down_scaled_ret:
            continue
        
        # find checkerboard corners in full resolution image for better accuracy
        ret, corners = cv2.findChessboardCorners(
            gray, (checkerboard_width,checkerboard_length),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner locations to subpixel accuracy
            refined_corners = cv2.cornerSubPix(gray, corners, SUBPIXEL_SIZE, (-1, -1), SUBPIXEL_CRITERIA)

            # add refined corners to dictionary with image id as key
            if refined_corners is not None:
                checkerboard_corners[image_id] = refined_corners
            else:
                print(f"    Warning: Subpixel refinement failed for {os.path.basename(img_path)}")
    

    print(f"    Found {len(checkerboard_corners)} unqiue poses with checkerboard detections")

    return checkerboard_corners, objp, img_size

def calibrate_camera_intrinsics(checkerboard_corners, objp, img_size):
    """Calibrate camera intrinsics using detected checkerboard corners"""
    print(f"    Calibrating camera with {len(checkerboard_corners)} images...")

    # intialize object points and image points lists for calibration
    objpoints = []
    imgpoints = []

    # populate object points and image points lists from checkerboard corners dictionary
    for frame_id, corners in checkerboard_corners.items():
        objpoints.append(objp.copy())
        imgpoints.append(corners)
    
    # Initial guess for camera matrix
    initial_camera_matrix = np.array([
        [img_size[0], 0, img_size[0]/2],
        [0, img_size[0], img_size[1]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Initial distortion coefficients (assuming minimal distortion)
    initial_dist_coeffs = np.zeros((5, 1))


    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size,
        initial_camera_matrix, initial_dist_coeffs,
        #flags=MONO_FLAGS,
        criteria=MONO_CRITERIA
    )
    
    if not ret:
        raise ValueError("Camera calibration failed")
    
    filtered_checkerboard_corners = {}
    
    # Calculate reprojection error and filter outliers
    total_error = 0
    total_points = 0
    
    for i, (frame_id, corners) in enumerate(checkerboard_corners.items()):
        projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2)

        if error / len(objpoints[i]) < 0.5:

            filtered_checkerboard_corners[frame_id] = checkerboard_corners[frame_id]

        total_error += error
        total_points += len(objpoints[i])
    
    mean_error = total_error / total_points

    # return corners dictionary with outliers removed, along with camera matrix, distortion coefficients, and mean reprojection error
    return camera_matrix, dist_coeffs, mean_error, filtered_checkerboard_corners

def calibrate_stereo_cameras(left_checkerboard_corners, right_checkerboard_corners, obj, img_size, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right):
    """Calibrate stereo camera pair to find rotation and translation between them"""
    print("    Calibrating stereo camera pair...")

    # intiailize object points and image points lists for stereo calibration
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # build object points and image points lists for stereo calibration by matching frames with checkerboard detections in both cameras
    for frame_id, corners in left_checkerboard_corners.items():
        if frame_id in right_checkerboard_corners:
            left_corners = corners
            right_corners = right_checkerboard_corners[frame_id]
            
            # Detect if OpenCV flipped the board 180 degrees in one of the cameras
            # Compare the 2D image vector from the first corner to the last corner
            left_vec = left_corners[-1][0] - left_corners[0][0]
            right_vec = right_corners[-1][0] - right_corners[0][0]
            
            # If the dot product is negative, the arrays are flowing in opposite directions
            if np.dot(left_vec, right_vec) < 0:
                print(f"    Fixing 180-degree board flip in frame {frame_id}")
                right_corners = right_corners[::-1] # Reverse the array

            objpoints.append(obj.copy())
            imgpoints_left.append(left_corners)
            imgpoints_right.append(right_corners)

    print(f"    Found {len(objpoints)} valid stereo pairs for calibration")

    # Perform stereo calibration
    ret, new_camera_matrix_left, new_dist_coeffs_left, new_camera_matrix_right, new_dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        img_size,
        criteria=STEREO_CRITERIA,
        flags=STEREO_FLAGS
    )

    #filtered_left_checkerboard_corners = {}
    #filtered_right_checkerboard_corners = {}
    #stereo_idx = 0

    # for i, (frame_id, corners) in enumerate(left_checkerboard_corners.items()):
    #     if frame_id in right_checkerboard_corners:
    #         left_rvec = rvecs[stereo_idx]
    #         left_tvec = tvecs[stereo_idx]

    #         left_projected_points, _ = cv2.projectPoints(objpoints[stereo_idx], left_rvec, left_tvec, new_camera_matrix_left, new_dist_coeffs_left)
    #         left_error = cv2.norm(imgpoints_left[stereo_idx], left_projected_points, cv2.NORM_L2)
            
    #         R_left_mat, _ = cv2.Rodrigues(left_rvec)
    #         R_right_mat = R @ R_left_mat
    #         right_rvec, _ = cv2.Rodrigues(R_right_mat)
    #         right_tvec = R @ left_tvec + T

    #         right_projected_points, _ = cv2.projectPoints(objpoints[stereo_idx], right_rvec, right_tvec, new_camera_matrix_right, new_dist_coeffs_right)
    #         right_error = cv2.norm(imgpoints_right[stereo_idx], right_projected_points, cv2.NORM_L2)


    #         if right_error / len(objpoints[stereo_idx]) < 0.3 and left_error / len(objpoints[stereo_idx]) < 0.3:
    #             print(f"    Keeping frame {frame_id} with left error {left_error/len(objpoints[stereo_idx]):.4f} and right error {right_error/len(objpoints[stereo_idx]):.4f}")

    #             filtered_left_checkerboard_corners[frame_id] = left_checkerboard_corners[frame_id]
    #             filtered_right_checkerboard_corners[frame_id] = right_checkerboard_corners[frame_id]

    #         else:
    #             print(f"    Discarding frame {frame_id} with left error {left_error/len(objpoints[stereo_idx]):.4f} and right error {right_error/len(objpoints[stereo_idx]):.4f}")

    #         stereo_idx += 1


    return ret, new_camera_matrix_left, new_dist_coeffs_left, new_camera_matrix_right, new_dist_coeffs_right, R, T, E, F

def find_cam_id(image_name):
    base_name = os.path.splitext(image_name)[0]
    parts = base_name.split('_')
    cam_id = parts[1]
    
    return cam_id

def calibrate_camera(camera, capture_name, checkerboard_width, checkerboard_length, square_size, stereo, tripod_mode=True):
    """ calibrate camera intrinsics and stereo parameters for a given camera and capture, returning results in dictionaries for writing to JSON and COLMAP formats"""
    
    # initilize results dictionaries to hold intrinisic and stereo calibration results for each camera
    camera_results = {}
    stereo_results = {}

    cam_folder_stereo_a = os.path.join("/mnt", camera, "captures", capture_name, "color_a")

    # get list of image files for each camera, sorted to ensure matching frames have same index in each list
    left_frame_names = sorted([f for f in os.listdir(cam_folder_stereo_a) if f.lower().endswith(('.tiff', '.jpg', '.png'))])

    # find camera ids from first image in each folder
    left_cam_id = find_cam_id(left_frame_names[0])

    # build checkerboard corners dictionary for each camera
    left_checkerboard_corners, obj, img_size = find_checkerboard_corners(
        cam_folder_stereo_a,
        left_frame_names, 
        left_cam_id,
        checkerboard_width, 
        checkerboard_length, 
        square_size,
        tripod_mode 
    )

    # Calibrate camera intrinsics
    left_camera_matrix, left_camera_dist_coeffs, left_camera_reprojection_error, filtered_left_checkerboard_corners = calibrate_camera_intrinsics(
        left_checkerboard_corners, 
        obj,
        img_size
    )

        # Re-Calibrate using filtered corners to improve accuracy
    left_camera_matrix, left_camera_dist_coeffs, left_camera_reprojection_error, refiltered_left_checkerboard_corners = calibrate_camera_intrinsics(
        filtered_left_checkerboard_corners, 
        obj,
        img_size
    )

    new_left_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_camera_dist_coeffs, img_size, 1, img_size)

    print(f"Cam A Calibration RMS Error: {left_camera_reprojection_error:.4f} pixels")

    if not stereo:
        camera_results[left_cam_id] = {
            'camera_matrix': left_camera_matrix,
            'dist_coeffs': left_camera_dist_coeffs,
            'width': img_size[0],
            'height': img_size[1],
        }
        
        stereo_results[left_cam_id] = {
            'rotation': np.eye(3),
            'translation': np.zeros((3, 1)),
            'P1': new_left_camera_matrix,
            'R1': np.eye(3),
            'P2': np.eye(3, 4),
            'R2': np.eye(3),
        }
        return camera_results, stereo_results
    
    print("cam folder", cam_folder_stereo_a)
    cam_folder_stereo_b = os.path.join("/mnt", camera, "captures", capture_name, "color_b")

    right_frame_names = sorted([f for f in os.listdir(cam_folder_stereo_b) if f.lower().endswith(('.tiff', '.jpg', '.png'))])

    right_cam_id = find_cam_id(right_frame_names[0])
        
    right_checkerboard_corners, obj, img_size = find_checkerboard_corners(
        cam_folder_stereo_b,
        right_frame_names, 
        right_cam_id,
        checkerboard_width, 
        checkerboard_length, 
        square_size
    )
    
    # repeat for right camera
    right_camera_matrix, right_camera_dist_coeffs, right_camera_reprojection_error, filtered_right_checkerboard_corners = calibrate_camera_intrinsics(
        right_checkerboard_corners, 
        obj,
        img_size
    )

    right_camera_matrix, right_camera_dist_coeffs, right_camera_reprojection_error, refiltered_right_checkerboard_corners = calibrate_camera_intrinsics(
        filtered_right_checkerboard_corners, 
        obj,
        img_size
    )

    print(f"Cam B Calibration RMS Error: {right_camera_reprojection_error:.4f} pixels")

    # Calibrate stereo parameters using filtered corners from both cameras
    retStereo, new_camera_a_matrix, new_cam_a_dist_coeffs, new_camera_b_matrix, new_cam_b_dist_coeffs, R, T, E, F = calibrate_stereo_cameras(
        filtered_left_checkerboard_corners,
        filtered_right_checkerboard_corners,
        obj,
        img_size,
        left_camera_matrix, left_camera_dist_coeffs,
        right_camera_matrix, right_camera_dist_coeffs
    )

    print(f"Stereo Calibration RMS Error: {retStereo:.4f} pixels")
    
    # Perform stereo rectification to compute rectification transforms and projection matrices for each camera
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=new_camera_a_matrix,
        distCoeffs1=new_cam_a_dist_coeffs,
        cameraMatrix2=new_camera_b_matrix,
        distCoeffs2=new_cam_b_dist_coeffs,
        imageSize=img_size,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    camera_results[left_cam_id] = {
        'camera_matrix': new_camera_a_matrix,
        'dist_coeffs': new_cam_a_dist_coeffs,
        'width': img_size[0],
        'height': img_size[1]
    }
        
    camera_results[right_cam_id] = {
        'camera_matrix': new_camera_b_matrix,
        'dist_coeffs': new_cam_b_dist_coeffs,
        'width': img_size[0],
        'height': img_size[1]
        }
        
    stereo_results[left_cam_id] = {
        'rotation': R,
        'translation': T,
        'P1': P1,
        'R1': R1,
        'P2': P2,
        'R2': R2,
    }
        
    return camera_results, stereo_results

def write_calibration_json(camera, camera_results, stereo_results):
    """ write intrinsic and stereo calibration results to JSON file in camera's system information folder"""
    cam_dir = os.path.join("/mnt", camera, "system_information")
    os.makedirs(cam_dir, exist_ok=True)

    cal_data = {
        "intrinsic_results": {
            str(cam_id): {
                "camera_matrix": cam_data["camera_matrix"].tolist(),
                "dist_coeffs": cam_data["dist_coeffs"].tolist(),
                "width": cam_data["width"],
                "height": cam_data["height"]
            }
            for cam_id, cam_data in camera_results.items()
        },
        "stereo_results": {
            str(cam_id): {
                "rotation": stereo_data["rotation"].tolist(),
                "translation": stereo_data["translation"].tolist(),
                "P1": stereo_data["P1"].tolist(),
                "R1": stereo_data["R1"].tolist(),
                "P2": stereo_data["P2"].tolist(),
                "R2": stereo_data["R2"].tolist()
            }
            for cam_id, stereo_data in stereo_results.items()
        }
    }

    cal_path = os.path.join(cam_dir, f"intrinsic_cal_{camera}.json")

    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=4)

    print(f"Wrote calibration to {cal_path}")

def read_calibration_json(camera, camera_results, stereo_results):
    """ read intrinsic and stereo calibration results from JSON file in camera's system information folder"""
    cam_dir = os.path.join("/mnt", camera, "system_information")
    cal_path = os.path.join(cam_dir, f"intrinsic_cal_{camera}.json")

    if not os.path.isfile(cal_path):
        print(f"Calibration file not found: {cal_path}")
        return camera_results, stereo_results

    with open(cal_path, "r") as f:
        data = json.load(f)

    # ---- Intrinsics ----
    for cam_id, intr in data["intrinsic_results"].items():
        camera_results[cam_id] = {
            "camera_matrix": np.array(intr["camera_matrix"], dtype=np.float64),
            "dist_coeffs": np.array(intr["dist_coeffs"], dtype=np.float64).flatten(),
            "width": int(intr["width"]),
            "height": int(intr["height"])
        }

    # ---- Stereo ----
    for cam_id, stereo in data["stereo_results"].items():
        stereo_results[cam_id] = {
            "rotation": np.array(stereo["rotation"], dtype=np.float64),
            "translation": np.array(stereo["translation"], dtype=np.float64),
            "P1": np.array(stereo["P1"], dtype=np.float64),
            "R1": np.array(stereo["R1"], dtype=np.float64),
            "P2": np.array(stereo["P2"], dtype=np.float64),
            "R2": np.array(stereo["R2"], dtype=np.float64),
        }

    return camera_results, stereo_results

def write_colmap_cameras(camera_results, stereo_results, output_path = os.path.join(os.path.dirname(os.getcwd()), "calibration", "cameras.txt"), scale = 0.5):
    """Write COLMAP cameras.txt file from each cameras .json calibration results"""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(camera_results)))
        
        for camera_id in sorted(stereo_results.keys()):
            camera_result = camera_results[camera_id]
            width = int(round(camera_result['width'] * scale))
            height = int(round(camera_result['height'] * scale))
            stereo_result = stereo_results[camera_id]

            # extract projection matrix for camera intrinsics
            P1 = stereo_result['P1']
            
            # Extract intrinsic parameters, apply scale for downsampled images
            fx = int(round(P1[0, 0] * scale))
            fy = int(round(P1[1, 1] * scale))
            cx = int(round(P1[0, 2] * scale))
            cy = int(round(P1[1, 2] * scale))

            # Write in COLMAP PINHOLE format
            f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
    
    print(f"Written COLMAP cameras file: {output_path}")

def update_colmap_cameras():
    """ read each .json intrinsic calibration result and update COLMAP cameras.txt file with intrinisic parameters for each camera"""
    cam_drives = sorted([
        f for f in os.listdir("/mnt")
        if os.path.isdir(os.path.join("/mnt", f))
        and f.startswith("cam_")
    ])

    camera_results = {}
    stereo_results = {}

    for camera in cam_drives:
        camera_results, stereo_results = read_calibration_json(camera, camera_results, stereo_results)

    write_colmap_cameras(camera_results, stereo_results)
    
def main():
    parser = argparse.ArgumentParser(description="Calculate intrinsic camera parameters.")
    parser.add_argument(
        "--camera-id",
        type=int,
        default="0",
        help="Name of the camera id (default = 01)"
    )
    parser.add_argument(
        "--capture-name",
        type=str,
        default="Capture",
        help="Name of the capture directory inside the output folder (default is BlenderCalibrationTake)"
    )
    parser.add_argument(
        "--checkerboard-width",
        type=int,
        default=10,
        help="Number of checkers wide (default is 7)"
    )
    parser.add_argument(
        "--checkerboard-length",
        type=int,
        default=18,
        help="Number of checkers tall (default is 5)"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.040,
        help="Size of checker in meters (default is 0.040)"
    )
    parser.add_argument(
        "--stereo",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True applies stereo processing, false applies no stereo processing"
    )
    parser.add_argument(
        "--tripod-mode",
        type=lambda x: x.lower() == "true",
        default=True,
        help="True applies tripod mode, false applies no tripod mode"
    )

    args = parser.parse_args()

    print("tripod mode is", args.tripod_mode)

    camera = f"cam_{args.camera_id:02d}"

    print("camera is", camera)

    camera_results, stereo_results = calibrate_camera(
        camera, 
       args.capture_name, 
        args.checkerboard_width, 
        args.checkerboard_length, 
        args.square_size,
        args.stereo,
        args.tripod_mode
    )

    write_calibration_json(camera, camera_results, stereo_results)
    update_colmap_cameras()

if __name__ == "__main__":
    main()



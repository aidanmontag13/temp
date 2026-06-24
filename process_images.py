import os
import time
import cv2
import torch
import argparse
import numpy as np
import json
import color_calibration
from torchvision.transforms.functional import to_tensor
from PIL import Image
import stereo_depth

# === Load RVM ===
device = "cuda" if torch.cuda.is_available() else "cpu"
rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").to(device) #switch to mobilenetv3 for more speed
rvm.eval()

# Initialize RVM states
rvm_state = (None, None, None, None)

import os

def build_frame_names(cam_folder, frame_range=False, frame_start=0, frame_end=100):
    """build list of frames without range to be processed"""
    frame_names = []

    for f in os.listdir(cam_folder):

        if not f.lower().endswith(".tiff"):
            continue

        try:
            frame_num = int(os.path.splitext(f.split("_")[2])[0])

        except (IndexError, ValueError):
            continue
        
        if frame_range == True:
            if frame_start is not None and frame_num < frame_start:
                continue

            if frame_end is not None and frame_num > frame_end:
                continue

        frame_names.append(f)

    return sorted(frame_names)

def preprocess_frame(frame):
    """convert BGR cv mat to RGB tensor img for RVM"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = to_tensor(img).unsqueeze(0).to(device)
    return img

def load_camera_params_json(camera_id, scale=1.0, stereo=False):
    """load camera intrinsics from .json file"""
    # Adjust pathing logic as per your system
    cam_dir = os.path.join("/mnt", f"cam_{camera_id:02d}", "system_information")
    cal_path = os.path.join(cam_dir, f"intrinsic_cal_cam_{camera_id:02d}.json")

    if not os.path.isfile(cal_path):
        print(f"Calibration file not found: {cal_path}")
        return None

    with open(cal_path, "r") as f:
        data = json.load(f)

    # Use string keys if your JSON stores them as strings
    left_camera_number  = f"{camera_id * 10:04d}"
    right_camera_number = f"{camera_id * 10 + 5:04d}"

    try:
        # ---- Intrinsics ----
        width  = int(data["intrinsic_results"][left_camera_number]["width"])
        width = width * scale
        height = int(data["intrinsic_results"][left_camera_number]["height"])
        height = height * scale

        left_camera_matrix = np.array(data["intrinsic_results"][left_camera_number]["camera_matrix"], dtype=np.float64)
        left_camera_matrix[0:2, 0:3] *= scale

        left_dist_coeffs   = np.array(data["intrinsic_results"][left_camera_number]["dist_coeffs"], dtype=np.float64).flatten()
        
        stereo_data = data["stereo_results"][left_camera_number]

        P1 = np.array(stereo_data["P1"], dtype=np.float64)
        P1[0:2, :] *= scale

        if stereo == False: # dont return stereo rectifiaction parameters if in mono mode
            return left_camera_matrix, left_dist_coeffs, width, height, P1
        
        right_camera_matrix = np.array(data["intrinsic_results"][right_camera_number]["camera_matrix"], dtype=np.float64)
        right_camera_matrix[0:2, 0:3] *= scale

        right_dist_coeffs    = np.array(data["intrinsic_results"][right_camera_number]["dist_coeffs"], dtype=np.float64).flatten()
        
        R  = np.array(stereo_data["rotation"], dtype=np.float64)
        T  = np.array(stereo_data["translation"], dtype=np.float64)
        R1 = np.array(stereo_data["R1"], dtype=np.float64)
        P2 = np.array(stereo_data["P2"], dtype=np.float64)
        P2[0:2, :] *= scale
        R2 = np.array(stereo_data["R2"], dtype=np.float64)

        return left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, width, height, R, T, P1, P2, R1, R2

    except KeyError as e:
        print(f"Key error: Could not find {e} in calibration file.")
        return None

def mask_image(image, depth_image):
    """Use RVM to estiamte mask from image"""

    filtered_image = cv2.bilateralFilter(image, 5, 75, 75)  # Apply bilateral filter to filter noise from image
    image = filtered_image

    global rvm_state

    image_tensor = preprocess_frame(image)

    with torch.no_grad():
        fgr, pha, *rvm_state = rvm(image_tensor, *rvm_state, downsample_ratio=0.4)

    alpha_np = (pha[0, 0].cpu().numpy()).astype(np.float32)
    fg_np = (fgr[0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
    
    fg_np = fg_np * np.stack([alpha_np, alpha_np, alpha_np], axis=-1)  # Apply alpha mask to foreground
    fg_np = (fg_np * 255).astype(np.uint8)
    fg_np = cv2.cvtColor(fg_np, cv2.COLOR_RGB2BGR)

    alpha_np = (alpha_np * 255).astype(np.uint8)

    bgra = np.dstack((fg_np, alpha_np))

    if depth_image is not None:
        depth_mask = cv2.resize(alpha_np, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        masked_depth = (depth_image * (depth_mask > 200)).astype(np.uint16)
        #masked_depth = depth_image
    else:
        masked_depth = None

    return bgra, masked_depth, alpha_np

def create_output_directories(output_dir, frame_names, stereo=False, data_format="postshot"):
    """Pre build output directories for writing images based upon frame names"""

    color_dir = os.path.join(output_dir, "processed_color")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    
    if stereo == True and data_format == "postshot":
        depth_dir = os.path.join(output_dir, "processed_depth")
        os.makedirs(depth_dir, exist_ok=True)
    
    for frame_name in frame_names:
        frame = frame_name.split("_")[2].split(".")[0]

        if data_format == "postshot":
            frame_color_dir = os.path.join(color_dir, frame)
            frame_colmap_dir = frame_color_dir

        if data_format == "instant_ngp":
            frame_color_dir = os.path.join(color_dir, frame, "images")
            frame_masks_dir = os.path.join(color_dir, frame, "masks")
            frame_colmap_dir = os.path.join(color_dir, frame, "sparse", "0")
            os.makedirs(frame_colmap_dir, exist_ok=True)
            os.makedirs(frame_masks_dir, exist_ok=True)

        os.makedirs(frame_color_dir, exist_ok=True)
        
        if stereo == True:
            if data_format == "postshot":
                frame_depth_dir = os.path.join(depth_dir, frame)

            if data_format == "instant_ngp":
                frame_depth_dir = os.path.join(color_dir, frame, "depth")

            os.makedirs(frame_depth_dir, exist_ok=True)

        else: 
            frame_depth_dir = None

    return None

def populate_colemap_files(calib_dir, output_dir, data_format="postshot"):
    """Copy intrinsic and extrinsic colmap files into each frame directory"""
    color_dir = os.path.join(output_dir, "processed_color")

    cameras_txt = os.path.join(calib_dir, "cameras.txt")
    images_txt = os.path.join(calib_dir, "images.txt")

    # Check if source files exist
    if not os.path.isfile(cameras_txt):
        raise FileNotFoundError("cameras.txt not found in calibration directory.")
    
    if not os.path.isfile(images_txt):
        raise FileNotFoundError("images.txt not found in calibration directory.")

    # Loop over each subfolder in color_dir
    for folder in os.listdir(color_dir):
        if data_format == "postshot":
            folder_path = os.path.join(color_dir, folder)

        if data_format == "instant_ngp":
            folder_path = os.path.join(color_dir, folder, "sparse", "0")

        if os.path.isdir(folder_path):
            # Copy cameras.txt
            with open(cameras_txt, 'r') as src_cam, open(os.path.join(folder_path, "cameras.txt"), 'w') as dst_cam:
                dst_cam.write(src_cam.read())

            # Copy images.txt
            with open(images_txt, 'r') as src_img, open(os.path.join(folder_path, "images.txt"), 'w') as dst_img:
                dst_img.write(src_img.read())

            #print(f"Copied cameras.txt and images.txt to: {folder_path}")

def find_right_img_path(path, camera_id):
    """ locate the right image path based on left image path for stereo processing"""
    # Split path into directory and filename
    dir_path, filename = path.rsplit("/", 1)

    # Change folder (replace last directory)
    dir_parts = dir_path.split("/")
    dir_parts[-1] = "color_b"
    new_dir = "/".join(dir_parts)

    # Parse filename
    parts = filename.split("_")
    frame = parts[2]  # keeps "0010.tiff"

    # Build new filename
    new_filename = f"Camera_{(camera_id + 5):04d}_{frame}"

    return f"{new_dir}/{new_filename}"
        
def process_frame(color_dir, frame_name, output_dir, stereo, data_format="postshot"):
    """process selected frame and write to output dataset"""
    cam_number = int(frame_name.split("_")[1])
    camera_id = cam_number // 10

    left_path = os.path.join(color_dir, frame_name)

    processed_color_path = os.path.join(output_dir, "processed_color")

    # Check if files exist
    if not os.path.isfile(left_path):
        print(f"[!] Skipping frame — missing color: {left_path}")
        return
    
    left_img = cv2.imread(left_path)

    # apply color correction (Currently disabled)
    try:
        left_wb_gains, left_M = color_calibration.load_calibration_data(camera_id, cam_number)
        left_img = color_calibration.correct_img(left_img, left_wb_gains, left_M)
    except FileNotFoundError:
        pass

    # If mono, undistort without rectification
    if stereo == False:
        left_camera_matrix, left_dist_coeffs, width, height, P1 = load_camera_params_json(camera_id, 0.5, stereo=False)
        left_img = cv2.undistort(left_img, left_camera_matrix, left_dist_coeffs, None, P1)
        depth = None
        left_img, depth, left_mask = mask_image(left_img, depth)

    # If stereo, apply rectification
    else:
        left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, width, height, R, T, P1, P2, R1, R2 = load_camera_params_json(camera_id, 0.5, stereo=True)
        right_path = find_right_img_path(left_path, cam_number)
        processed_depth_path = os.path.join(output_dir, "processed_depth")
    
        if not os.path.isfile(right_path):
            print(f"[!] Skipping frame — missing color: {right_path}")
            return

        right_img = cv2.imread(right_path)

        try:
            right_wb_gains, right_M = color_calibration.load_calibration_data(camera_id, cam_number + 5)
            right_img = color_calibration.correct_img(right_img, right_wb_gains, right_M)
        except FileNotFoundError:
            pass
        
        left_img, right_img = stereo_depth.rectify_images(
            left_img, right_img, 
            left_camera_matrix, right_camera_matrix, 
            left_dist_coeffs, right_dist_coeffs, 
            P1, R1,
            P2, R2
        )

        # Estiamte depth map from rectified images
        depth = stereo_depth.create_depth_map(
            left_img,
            right_img,
            P1,
            P2, 
        )
    
        depth = cv2.resize(depth, (depth.shape[1] // 2, depth.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        left_img, depth, left_mask = mask_image(left_img, depth)
    
    if left_img is not None:
        base_name = os.path.splitext(frame_name)[0]
        parts = base_name.split('_')
        camera_name = parts[0] + '_' + parts[1] if len(parts) >= 2 else base_name
        out_name = f"{camera_name}.png"

        if data_format == "postshot":
            masked_color_out_path = os.path.join(processed_color_path, parts[2], out_name)

        if data_format == "instant_ngp":
            masked_color_out_path = os.path.join(processed_color_path, parts[2], "images", out_name)
            mask_out_path = os.path.join(processed_color_path, parts[2], "masks", out_name)
            cv2.imwrite(mask_out_path, left_mask)

        cv2.imwrite(masked_color_out_path, left_img)
        #print(f"[+] Saved processed image to {masked_color_out_path}")

        if depth is not None:
            if data_format == "postshot":
                masked_depth_out_path = os.path.join(processed_depth_path, parts[2], f"{camera_name}.tiff")

            if data_format == "instant_ngp":
                masked_depth_out_path = os.path.join(processed_color_path, parts[2], "depth", f"{camera_name}.tiff")

            cv2.imwrite(masked_depth_out_path, depth, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            #print(f"[+] Saved masked depth to {masked_depth_out_path}")
    else:
        print(f"[!] No mask generated for {left_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Images For Volumetric Rendering.")
    parser.add_argument(
        "--capture-name",
        type=str,
        default="Capture",
        help="Name of the capture directory inside the output folder (default is Capture)"
    )
    parser.add_argument(
        "--stereo",
        type=lambda x: x.lower() == "true",
        default=False,
        help="True applies stereo processing, false applies no stereo processing"
    )

    parser.add_argument(
        "--data-format",
        type=str,
        default="instant_ngp",
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
    
    args = parser.parse_args()

    mnt_dir = os.path.join("/mnt")

    # find each nfs drive int mnt folder
    cam_drives = sorted([
        f for f in os.listdir(mnt_dir)
        if os.path.isdir(os.path.join(mnt_dir, f))
        and f.startswith("cam_")
    ])

    if args.stereo == True:
        print("applying stereo processing")

    else:
        print("Not Applying Stereo Processing")

    output_dir = os.path.join(os.path.dirname(os.getcwd()), "outputs", args.capture_name)

    start_frame = None

    for camera in cam_drives: #Iterate through each camera folder
        base_dir = os.path.join(mnt_dir, camera, "captures", args.capture_name)
        calib_dir = os.path.join(os.path.dirname(os.getcwd()), "calibration")

        cam_folder = os.path.join(mnt_dir, camera, "captures", args.capture_name, "color_a")

        if os.path.exists(cam_folder):
            frame_names = build_frame_names(cam_folder, frame_range=args.frame_range, frame_start=args.frame_start, frame_end=args.frame_end)

            if start_frame == None:
                start_frame = int(os.path.splitext(frame_names[0].split("_")[2])[0])

            create_output_directories(output_dir, frame_names, stereo=args.stereo, data_format=args.data_format) # Run everytime in case a frame was missed
            populate_colemap_files(calib_dir, output_dir, data_format=args.data_format) # Run everytime in case a frame was missed
            
            print()
            print()
            print("Processing Images in", cam_folder)

            for frame_name in frame_names:
                try:
                    frame_number = int(os.path.splitext(frame_name.split("_")[2])[0])
                except (IndexError, ValueError):
                    continue
                
                print(f"\rProcessing Frame {frame_number - start_frame + 1} of {len(frame_names)}", end="", flush=True)
                process_frame(cam_folder, frame_name, output_dir, args.stereo, args.data_format)
        else:
            continue
    
    print()
    print("2D Processing complete.")
            

if __name__ == "__main__":
    main()


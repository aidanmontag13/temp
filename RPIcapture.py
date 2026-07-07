import socket, re
import cv2
import os
import threading
import time
import queue
import json
import struct
import fcntl
import logging
import subprocess
from datetime import datetime
import numpy as np
from collections import deque
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls

os.umask(0)

HOST = "0.0.0.0"
PORT = 5005
STREAM_PORT = 5006
STREAM_PORT_B = 5007
MULTICAST_GROUP = "224.1.1.1"
BUFFER_SIZE = 1024
BASE_DIR = os.path.dirname(__file__)
print(f"base dir is {BASE_DIR}")
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs"))
CAPTURE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "captures"))
os.makedirs(CAPTURE_DIR, exist_ok=True)
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
SPECS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "app", "camera-specs"))
print(f"config dir is {CONFIG_DIR}")

log_file = os.path.join(CAPTURE_DIR, "capture.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def read_camera_config():
    print("Reading camera config...")
    logging.info("Reading camera config...")
    config_path = os.path.join(OUTPUT_DIR, "system_information", "system_information.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
    except Exception as e:
        print(f"Failed to read camera config: {e}")
        logging.error(f"Failed to read camera config: {e}")
        return {}
    
    camera_id = config["id"]
    hostname = config["hostname"]
    mode = config["config"]
    camera_model = config["camera"]

    print(f"Camera ID: {camera_id}, Hostname: {hostname}, Mode: {mode}, Camera Model: {camera_model}")
    logging.info(f"Camera ID: {camera_id}, Hostname: {hostname}, Mode: {mode}, Camera Model: {camera_model}")

    return camera_id, hostname, mode, camera_model

# LED Control using external script
def led_control(command):
    """Control LED using external led-control script"""
    try:
        subprocess.run(["/home/chump/chumps/pi/config/bootstrap/led-control.sh", command], 
                      check=False, capture_output=True, timeout=5)
    except Exception:
        # Silently fail if LED control is not available
        pass


def create_session_directory(capture_name):
    session_dir = os.path.join(CAPTURE_DIR, capture_name)
    os.makedirs(session_dir, exist_ok=True)
    logging.info(f"Images will be saved in: {session_dir}")
    print(f"Images will be saved in: {session_dir}")
    return session_dir


def initialize_server():
    print(f"Starting camera server on port {PORT}...")
    logging.info(f"Starting camera server on port {PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print("Waiting for connection...")
        logging.info("Waiting for connection...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            logging.info(f"Connected by {addr}")
            data = conn.recv(1024)
            message = data.decode().strip()
            capture_name = "DefaultCapture"
            shutter_duration = 20833
            gain = 5
            color_temp = 3200
            try:
                for pair in message.split(";"):
                    key, value = pair.split("=")
                    if key == "CAPTURE_NAME":
                        capture_name = value
                    elif key == "FRAME_RATE":
                        fps = int(value)
                    elif key == "SHUTTER_DURATION":
                        shutter_duration = int(float(value))
                    elif key == "GAIN":
                        gain = float(value)
                    elif key == "COLOR_TEMP":
                        color_temp = int(value)
                    elif key == "SYNC_ROLE":
                        sync_role = value
                    elif key == "CALIB_MODE":
                        calib_mode = value
                    elif key == "FORCE_STEREO":
                        force_stereo = value
            except Exception as e:
                print(f"Failed to parse init message: {e}")
                logging.info(f"Failed to parse init message: {e}")
            conn.sendall(b"READY")
            return (
                capture_name,
                fps,
                shutter_duration,
                gain,
                color_temp,
                sync_role,
                calib_mode,
                force_stereo
            )


def initialize_color_camera(camera_id, mode, fps, shutter_duration, gain, color_temp, sync_role, cam_num, calib_mode):
    #tuning_path = _choose_tuning_path()
    print ("looking for tuning file in %s", SPECS_DIR)
    if not mode == "RGBD":
        tuning_path = os.path.join(SPECS_DIR, "imx477.json")
    else:
        tuning_path = os.path.join(SPECS_DIR, "imx462.json")
    tuning = Picamera2.load_tuning_file(tuning_path)
    print("tuning path is %s", tuning_path)
    logging.info("tuning path is %s", tuning_path)

    cam = Picamera2(camera_num=cam_num, tuning=tuning)

    if calib_mode == "1":
        cam.configure(
            cam.create_still_configuration(
                main={"size": (4056, 3040), "format": "BGR888"},
                lores={"size": (1280, 960), "format": "BGR888"}, 
                buffer_count=8
            )
        )
        fps = 8
        print("Calibration mode enabled: using full resolution and reduced frame rate")
        logging.info("Calibration mode enabled: using full resolution and reduced frame rate")
    else:
        cam.configure(
            cam.create_still_configuration(
                main={"size": (2028, 1520), "format": "BGR888"},
                lores={"size": (1280, 960), "format": "BGR888"}, 
                buffer_count=8
            )
        )

    #cam.configure(
    #    cam.create_still_configuration(raw={"size": (4056, 3040), "format": "SBGGR12"}, buffer_count=8)
    #)

    if sync_role == "MASTER":
        if cam_num == 0:
            cam.set_controls(
                {
                    "AeEnable": False,
                    "ExposureTime": shutter_duration,
                    "AwbEnable": False,
                    #"ColourGains": (1.67, 2.06),
                    "ColourTemperature": color_temp,
                    "AnalogueGain": gain,
                    "NoiseReductionMode": 1,
                    "SyncMode": controls.rpi.SyncModeEnum.Server,
                    "FrameRate": fps,
                }
            )
            time.sleep(1)

    else:
        cam.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": shutter_duration,
                "AwbEnable": False,
                #"ColourGains": (1.67, 2.06),
                "ColourTemperature": color_temp,
                "AnalogueGain": gain,
                "NoiseReductionMode": 1,
                "SyncMode": controls.rpi.SyncModeEnum.Client,
                "FrameRate": fps,
            }
        )

    cam.start()

    metadata = cam.capture_metadata()
    print(metadata)
    logging.info("%s", metadata)

    print("camera id is", camera_id)
    logging.info("camera id is %s", camera_id)

    print("fps is", fps)
    logging.info("fps is %s", fps)

    print("Exposure time (μs):", metadata.get("ExposureTime"))
    logging.info("Exposure time (μs): %s", metadata.get("ExposureTime"))

    print("Analogue gain:", metadata.get("AnalogueGain"))
    logging.info("Analogue gain: %s", metadata.get("AnalogueGain"))

    print("Digital gain:", metadata.get("DigitalGain"))
    logging.info("Digital gain: %s", metadata.get("DigitalGain"))

    print("Frame duration (μs):", metadata.get("FrameDuration"))
    logging.info("Frame duration (μs): %s", metadata.get("FrameDuration"))

    print("Brightness:", metadata.get("Brightness"))
    logging.info("Brightness: %s", metadata.get("Brightness"))

    print("Contrast:", metadata.get("Contrast"))
    logging.info("Contrast: %s", metadata.get("Contrast"))

    print("Saturation:", metadata.get("Saturation"))
    logging.info("Saturation: %s", metadata.get("Saturation"))

    print("Sharpness:", metadata.get("Sharpness"))
    logging.info("Sharpness: %s", metadata.get("Sharpness"))

    print("White Balance Gains (ColourGains):", metadata.get("ColourGains"))
    logging.info("White Balance Gains (ColourGains): %s", metadata.get("ColourGains"))

    print("White Balance Temp:", metadata.get("ColourTemperature"))
    logging.info("White Balance Temp: %s", metadata.get("ColourTemperature"))

    print("Auto White Balance enabled:", metadata.get("AwbEnable"))
    logging.info("Auto White Balance enabled: %s", metadata.get("AwbEnable"))

    print("Color Camera started")
    logging.info("Color Camera started")

    return cam


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", ifname[:15].encode()),
        )[20:24]
    )

def sync_step(mode, camera_a, camera_b):
    """Check camera sync status and update LED accordingly"""
    # Check if camera A is ready
    metadata = camera_a.capture_metadata()
    if metadata.get("SyncReady"):
        print("Camera A sync is ready.")
        logging.info("Camera A sync is ready.")
        if mode == "STEREO":
            metadata = camera_b.capture_metadata()
            if metadata.get("SyncReady"):
                print("Camera B sync is ready.")
                logging.info("Camera B sync is ready.")
                led_control("on")  # solid ON when ready
                return True
            else:
                print("Camera B Sync is not ready")
                return False
        else:
            led_control("on")  # solid ON when ready
        return True
    return False

def color_capture_worker(mode, camera_id, color_camera, write_queue, buffer_lock, viewer_buffer, recording_event):
    while True:
        start_time = time.time()
        request = color_camera.capture_request()
        frame = request.make_array("main")
        frame_lores = request.make_array("lores")

        metadata = request.get_metadata()
        synctimer = metadata.get("SyncTimer", -1)
        request.release()


        with buffer_lock:
            viewer_buffer.append((camera_id, synctimer, frame_lores,))

        if recording_event.is_set():
            print("Sync Timer:", synctimer)
            logging.info("Sync Timer: %s", synctimer)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if mode == "RGBD":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            write_queue.put((camera_id, frame, synctimer))

            end_time = time.time()
            loop_time = end_time - start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0
            print("fps: ", fps)

def writer_worker(mode, camera_a_dir, camera_b_dir, camera_a_id, camera_b_id, write_queue, fps, calib_mode):
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
        camera_id, frame, synctimer = item

        frame_number = np.round((-synctimer * 1e-6) * fps)

        if not frame_number % 4 == 1 and calib_mode == "1":
            print("Skipping frame %s for calibration mode", frame_number)
            logging.info("Skipping frame %s for calibration mode", frame_number)
            write_queue.task_done()
            continue

        if camera_id == camera_a_id:
            filepath = os.path.join(
                camera_a_dir, f"Camera_{int(camera_id):04d}_{int(frame_number):04d}.tiff"
            )

        elif camera_id == camera_b_id:
            filepath = os.path.join(
                camera_b_dir, f"Camera_{int(camera_id):04d}_{int(frame_number):04d}.tiff"
            )
        
        else:
            continue

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        print("write queue", write_queue.qsize())
        logging.info("write queue %s", write_queue.qsize())
        write_queue.task_done()


def streaming_server(mode, camera_a, camera_b, buffer_lock_a, buffer_lock_b, viewer_buffer_a, viewer_buffer_b, stream_active, port, calib_mode, recording_event):
    """Handle streaming connections for live video feed"""
    print(f"Starting streaming server on port {port}...")
    logging.info(f"Starting streaming server on port {port}...")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, port))
    server_socket.listen(5)
    server_socket.settimeout(1.0)  # Set timeout for accept()

    print(f"Streaming server listening on port {port}")
    logging.info(f"Streaming server listening on port {port}")

    while stream_active[0]:
        try:
            client_socket, addr = server_socket.accept()
            print(f"Streaming client connected from {addr}")
            logging.info(f"Streaming client connected from {addr}")

            if mode == "STEREO":
                # Handle stereo streaming client in a separate thread
                client_thread = threading.Thread(
                    target=handle_streaming_client_stereo,
                    args=(client_socket, camera_a, camera_b, buffer_lock_a, buffer_lock_b, viewer_buffer_a, viewer_buffer_b, stream_active, calib_mode, recording_event),
                    daemon=True,
                )
                client_thread.start()
                continue
            else:
                # Handle streaming client in a separate thread
                client_thread = threading.Thread(
                    target=handle_streaming_client,
                    args=(client_socket, camera_a, buffer_lock_a, viewer_buffer_a, stream_active, calib_mode, recording_event),
                    daemon=True,
                )
                client_thread.start()
        except socket.timeout:
            continue  # Check if we should still be running
        except Exception as e:
            print(f"Error accepting streaming connection: {e}")
            logging.error(f"Error accepting streaming connection: {e}")
            break

    server_socket.close()
    print("Streaming server stopped")
    logging.info("Streaming server stopped")

def handle_streaming_client_stereo(client_socket, camera_a, camera_b, buffer_lock_a, buffer_lock_b, viewer_buffer_a, viewer_buffer_b, stream_active, calib_mode, recording_event):
    """Handle individual streaming client connection"""
    try:
        # Wait for stream request
        data = client_socket.recv(1024)
        if data.decode().strip() == "STREAM_REQUEST":
            print("Streaming request received")
            logging.info("Streaming request received")

            # Send frames continuously
            while stream_active[0]:
                try:
                    # Capture frame
                    with buffer_lock_a:
                        if len(viewer_buffer_a) > 0:
                            # Grab the absolute newest frame (the right-most item)
                            camera_id_a, synctimer, frame_a = viewer_buffer_a[-1]
                    if calib_mode == "1":
                        with buffer_lock_b:
                            if len(viewer_buffer_b) > 0:
                                # Grab the absolute newest frame (the right-most item)
                                camera_id_b, synctimer, frame_b = viewer_buffer_b[-1]

                        # Process frame (same as capture processing but smaller for streaming)
                        focus_assist_a = frame_a[448:512, 608:672]
                        resized_focus_a = cv2.resize(focus_assist_a, (256, 256), cv2.INTER_NEAREST)

                        frame_a[0:256, 0:256] = resized_focus_a
                    
                        focus_assist_b = frame_b[448:512, 608:672]
                        resized_focus_b = cv2.resize(focus_assist_b, (256, 256), cv2.INTER_NEAREST)

                        frame_b[0:256, 640:896] = resized_focus_b

                        combined_frame = np.zeros((960, 1280, 3), dtype=np.uint8)

                        combined_frame[:, 0:640] = frame_a[:, 0:640]

                        combined_frame[:, 640:1280] = frame_b[:, 640:1280]
                    
                    
                        frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

                    else:
                        frame = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)

                    if recording_event.is_set():
                        cv2.putText(
                            frame,
                            "REC",          # text
                            (950, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (0, 0, 255),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )
                    else:
                        cv2.putText(
                            frame,
                            "REC",          # text
                            (950, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (64, 64, 64),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )

                    if synctimer < 0:
                        cv2.putText(
                            frame,
                            "SYNC",          # text
                            (800, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (0, 255, 0),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )
                    else:
                        cv2.putText(
                            frame,
                            "SYNC",          # text
                            (800, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (64, 64, 64),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )

                    # Resize for streaming (smaller for better performance)

                    # Encode as JPEG
                    _, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )

                    # Send frame
                    client_socket.sendall(buffer.tobytes())

                except Exception as e:
                    print(f"Error sending frame: {e}")
                    logging.error(f"Error sending frame: {e}")
                    break

    except Exception as e:
        print(f"Streaming client error: {e}")
        logging.error(f"Streaming client error: {e}")
    finally:
        client_socket.close()
        print("Streaming client disconnected")
        logging.info("Streaming client disconnected")

def handle_streaming_client(client_socket, camera_a, buffer_lock_a, viewer_buffer_a, stream_active, calib_mode, recording_event):
    """Handle individual streaming client connection"""
    try:
        # Wait for stream request
        data = client_socket.recv(1024)
        if data.decode().strip() == "STREAM_REQUEST":
            print("Streaming request received")
            logging.info("Streaming request received")

            # Send frames continuously
            while stream_active[0]:
                try:
                    # Capture frame
                    with buffer_lock_a:
                        if len(viewer_buffer_a) > 0:
                            # Grab the absolute newest frame (the right-most item)
                            camera_id_a, synctimer, frame_a = viewer_buffer_a[-1]
                    if calib_mode == "1":

                        # Process frame (same as capture processing but smaller for streaming)
                        focus_assist_a = frame_a[448:512, 608:672]
                        resized_focus_a = cv2.resize(focus_assist_a, (256, 256), cv2.INTER_NEAREST)

                        frame_a[0:256, 0:256] = resized_focus_a
                    
                        frame = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)

                    else:
                        frame = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)

                    # apply viewer data
                    if recording_event.is_set():
                        cv2.putText(
                            frame,
                            "REC",          # text
                            (950, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (0, 0, 255),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )
                    else:
                        cv2.putText(
                            frame,
                            "REC",          # text
                            (950, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (64, 64, 64),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )

                    if synctimer < 0:
                        cv2.putText(
                            frame,
                            "SYNC",          # text
                            (800, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (0, 255, 0),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )
                    else:
                        cv2.putText(
                            frame,
                            "SYNC",          # text
                            (800, 775),             # bottom-left corner of text
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,                   # font scale
                            (64, 64, 64),           # color (B, G, R)
                            2,                     # thickness
                            cv2.LINE_AA            # anti-aliased line type
                        )

                    # Resize for streaming (smaller for better performance)

                    # Encode as JPEG
                    _, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )

                    # Send frame
                    client_socket.sendall(buffer.tobytes())

                except Exception as e:
                    print(f"Error sending frame: {e}")
                    logging.error(f"Error sending frame: {e}")
                    break

    except Exception as e:
        print(f"Streaming client error: {e}")
        logging.error(f"Streaming client error: {e}")
    finally:
        client_socket.close()
        print("Streaming client disconnected")
        logging.info("Streaming client disconnected")

def start_listener(mode, write_queue, camera_a, camera_b, stream_active, recording_event):
    print(
        f"Listening for UDP pulses on port {PORT} multicast group {MULTICAST_GROUP}..."
    )
    logging.info(
        f"Listening for UDP pulses on port {PORT} multicast group {MULTICAST_GROUP}..."
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", PORT))

    interface_ip = get_ip_address("eth0")
    print(("interface ip: ", interface_ip))
    logging.info(("interface ip: %s", interface_ip))


    mreq = struct.pack(
        "4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(interface_ip)
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    sock.settimeout(0.01)  # 10ms

    # Start LED during sync process
    led_control("blink")
    synced = False

    try:
        while True:
            # Handle sync and LED
            if not synced:
                synced = sync_step(mode, camera_a, camera_b)

            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
                message = data.decode().strip()

                if message.upper() == "START":
                    print("Received START message. Starting capture.")
                    logging.info("Received START message. Starting capture.")
                    recording_event.set()

                elif message.upper() == "END":
                    print("Received END message. Exiting listener.")
                    logging.info("Received END message. Exiting listener.")
                    recording_event.clear()
                    break
                elif message.upper() == "REBOOT":
                    print("Received REBOOT message. Restarting Device.")
                    logging.info("Received REBOOT message. Rebooting Device.")
                    clean_up(stream_active, write_queue, camera_a, camera_b)
                    os.system("sudo reboot")
                    
                else:
                    try:
                        frame_number = int(message)
                    except ValueError:
                        print(f"Invalid frame number: {message}")
                        logging.error(f"Invalid frame number: {message}")
            except socket.timeout:
                # No packet → just continue (still blink)
                pass
            except Exception as e:
                print(f"Listener error: {e}")
                logging.error(f"Listener error: {e}")
    finally:
        led_control("off")  # LED OFF on shutdown
        sock.close()

def clean_up(stream_active, write_queue, camera_a, camera_b):
    print("Cleaning up...")
    logging.info("Cleaning up...")
        
    write_queue.put(None)

    # Wait for threads to finish processing current queue
    time.sleep(1)
    # Stop streaming server
    stream_active[0] = False

    camera_a.stop()
    camera_a.close()

    if camera_b is not None:
        camera_b.stop()
        camera_b.close()

    led_control("off")

    print("Shutdown complete.")
    logging.info("Shutdown complete.")

def capture_calibration_image(camera_id, cam_num, camera_dir, shutter_duration, gain, color_temp):
    print("starting camera in calibration mode...")
    tuning_path = os.path.join(SPECS_DIR, "imx477.json")
    cam = Picamera2(camera_num=cam_num, tuning=tuning_path)

    cam.configure(
        cam.create_still_configuration(main={"size": (4056, 3040)}, buffer_count=8)
    )

    cam.set_controls(
                {
                    "AeEnable": False,
                    "ExposureTime": shutter_duration,
                    "AwbEnable": False,
                    #"ColourGains": (1.67, 2.06),
                    "ColourTemperature": color_temp,
                    "AnalogueGain": gain,
                    "NoiseReductionMode": 1
                }
            )
    cam.start()
    time.sleep(1)
    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Captured calibration image for camera %s", camera_id)

    filepath = os.path.join(
                camera_dir, f"Camera_{int(camera_id):04d}_cal.tiff"
            )

    cv2.imwrite(filepath, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    cam.stop()
    cam.close()
    print("Saved calibration image for camera %s at %s", camera_id, filepath)
    
    return frame

def main():
    # LED startup - brief on then off
    led_control("on")
    time.sleep(0.5)
    led_control("off")

    # Read camera configuration
    camera_id, hostname, mode, camera_model = read_camera_config()

    write_queue = queue.Queue()
    viewer_buffer_a = deque(maxlen=2)
    viewer_buffer_b = deque(maxlen=2)
    buffer_lock_a = threading.Lock()
    buffer_lock_b = threading.Lock()

    # Shared flag for streaming server
    stream_active = [True]

    (
        capture_name,
        fps,
        shutter_duration,
        gain,
        color_temp,
        sync_role,
        calib_mode,
        force_stereo
    ) = initialize_server()

    if force_stereo == "0":
        mode = "MONO"
        print("Forcing MONO mode based on config")
        logging.info("Forcing MONO mode based on config")

    camera_a_id = int(camera_id) * 10

    if mode == "STEREO":
        camera_b_id = int(camera_id) * 10 + 5
    
    else: camera_b_id = None
    
    session_dir = create_session_directory(capture_name)

    camera_a_dir = os.path.join(session_dir, "color_a")

    if mode == "STEREO":
        camera_b_dir = os.path.join(session_dir, "color_b")
    
    else:
        camera_b_dir = None

    os.makedirs(camera_a_dir, exist_ok=True)

    capture_calibration_image(camera_a_id, 0, camera_a_dir, shutter_duration, gain, color_temp)

    recording_event = threading.Event()

    if mode == "STEREO":
        os.makedirs(camera_b_dir, exist_ok=True)
        capture_calibration_image(camera_b_id, 1, camera_b_dir, shutter_duration, gain, color_temp)
        print("Initializing stereo camera B...")
        logging.info("Initializing stereo camera B...")
        camera_b = initialize_color_camera(
            camera_b_id, mode, fps, shutter_duration, gain, color_temp, "SLAVE", 1, calib_mode
        )

        camera_b_capture_thread = threading.Thread(
            target=color_capture_worker,
            args=(mode, camera_b_id, camera_b, write_queue, buffer_lock_b, viewer_buffer_b, recording_event),
            daemon=True,
        )

        if camera_id == "1":
            time.sleep(1)  # ensure proper sync

    else:
        camera_b = None
        print("Initializing color camera A...")
        logging.info("Initializing color camera A...")

    camera_a = initialize_color_camera(
        camera_a_id, mode, fps, shutter_duration, gain, color_temp, sync_role, 0, calib_mode
    )

    camera_a_capture_thread = threading.Thread(
        target=color_capture_worker,
        args=(mode, camera_a_id, camera_a, write_queue, buffer_lock_a, viewer_buffer_a, recording_event),
        daemon=True,
    )

    writer_thread = threading.Thread(
        target=writer_worker,
        args=(
            mode,
            camera_a_dir, camera_b_dir,
            camera_a_id, camera_b_id,
            write_queue,
            fps,
            calib_mode
        ),
        daemon=True,
    )

    streaming_thread = threading.Thread(
        target=streaming_server, args=(mode, camera_a, camera_b, buffer_lock_a, buffer_lock_b, viewer_buffer_a, viewer_buffer_b, stream_active, STREAM_PORT, calib_mode, recording_event), daemon=True
    )

    camera_a_capture_thread.start()
    if mode == "STEREO":
        camera_b_capture_thread.start()

    writer_thread.start()
    streaming_thread.start()

    try:
        start_listener(
            mode,
            write_queue,
            camera_a,
            camera_b,
            stream_active,
            recording_event
        )
    finally:
        clean_up(
            stream_active,
            write_queue,
            camera_a,
            camera_b,
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"fatal error: {e}")
        logging.info(f"fatal error: {e}")

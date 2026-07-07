import socket
import struct
import threading
import time
import json
import numpy as np
import os
import subprocess
import sys

# Network constants
MULTICAST_GROUP = '224.1.1.1'
PORT = 5005
TTL = 1
INTERFACE_IP = '192.168.1.1'

class CameraNetworkController:
    def __init__(self):
        self.stop_event = threading.Event()
        self.is_capturing = False
        self.camera_connected = False
        self.is_calibrating_ext = False
        self.is_calibrating_int = False
        self.is_calibrating_color = False
        self.is_processing = False
        
    def initialization_pulse(self, ip, camera_id, capture_name, fps, shutter_angle, gain, color_temp, calib_mode, force_stereo, sync_role):
        """Send initialization pulse to a single camera"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((ip, PORT))
                shutter_duration = int((1000000 / fps) * (shutter_angle / 360))
                print("printing calib mode", calib_mode)
                message = f"CAMERA_ID={camera_id};CAPTURE_NAME={capture_name};FRAME_RATE={fps};SHUTTER_DURATION={shutter_duration};GAIN={gain};COLOR_TEMP={color_temp};SYNC_ROLE={sync_role};CALIB_MODE={int(calib_mode)};FORCE_STEREO={int(force_stereo)}"
                s.sendall(message.encode())
                response = s.recv(1024).decode().strip()
                return response == "READY"
        except Exception as e:
            print(f"Error initializing camera {ip}: {e}")
            return False

    def connect_cameras(self, camera_ips, capture_name, fps, shutter_angle, gain, color_temp, calib_mode, force_stereo):
        """Connect to all cameras with given parameters"""
        self.camera_connected = False

        for i, ip in enumerate(camera_ips, start=1):
            print(f"  Camera {i}: {ip}")

            success = self.initialization_pulse(
                ip,
                i,
                capture_name,
                fps,
                shutter_angle,
                gain,
                color_temp,
                calib_mode,
                force_stereo,
                sync_role="MASTER" if i == 1 else "SLAVE",

            )

            if success:
                self.camera_connected = True
                print(f"Camera {i} successfully initialized.")
            else:
                print(f"Failed to initialize Camera {i}.")
        return self.camera_connected

    def send_pulses(self, fps):
        """Send multicast pulses for frame synchronization"""
        frame = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
            # Set multicast TTL
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack('b', TTL))

            # Set outgoing interface
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(INTERFACE_IP))

            s.sendto(b"START", (MULTICAST_GROUP, PORT))

            while not self.stop_event.is_set():
                frame += 1
                time.sleep(1.0 / fps)

            # Send END signal once
            s.sendto(b"END", (MULTICAST_GROUP, PORT))
            print("Stopped multicast capture.")

    def start_capture(self, fps):
        """Start capture process"""
        if not self.camera_connected:
            return False
            
        self.stop_event.clear()
        self.is_capturing = True
        threading.Thread(target=self.send_pulses, args=(fps,), daemon=True).start()
        print("Capture started...")
        return True

    def stop_capture(self):
        """Stop capture process"""
        self.stop_event.set()
        self.is_capturing = False
        print("Capture stopped.")

    def send_end_signal(self, camera_ips):
        """Send END signal to all cameras to stop capture"""
        if not camera_ips:
            print("No camera IPs specified to send END signal.")
            return

        for ip in camera_ips:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.sendto(b"END", (ip, PORT))
                    print(f"Sent END signal to {ip}")
            except Exception as e:
                print(f"Failed to send END signal to {ip}: {e}")

    def reboot_cameras(self, camera_ips):
        """Reboot all cameras"""
        if not camera_ips:
            print("No camera IPs specified to send reboot signal.")
            return

        # Send to the first IP immediately
        first_ip = camera_ips[0]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(b"REBOOT", (first_ip, PORT))
                print(f"Sent REBOOT signal to {first_ip}")
        except Exception as e:
            print(f"Failed to send REBOOT signal to {first_ip}: {e}")

        # Wait before sending to the rest
        time.sleep(5)  # adjust delay as needed

        # Send to remaining IPs
        for ip in camera_ips[1:]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.sendto(b"REBOOT", (ip, PORT))
                    print(f"Sent REBOOT signal to {ip}")
            except Exception as e:
                print(f"Failed to send REBOOT signal to {ip}: {e}")

    def disconnect_cameras(self, camera_ips):
        """Disconnect from all cameras"""
        if self.is_capturing:
            self.stop_capture()
            time.sleep(0.1)
        
        self.send_end_signal(camera_ips)
        self.camera_connected = False
        print("Cameras disconnected.")

    def run_image_delete(self, camera_ips):
        """Run image deletion for all cameras"""
        try:
            subprocess.Popen([sys.executable, "delete_images.py"], cwd='.')
        except Exception as e:
            print(f"Failed to delete images")

    def run_intrinsic_calibration(self, selected_camera, capture_name, board_width, board_length, square_size, stereo, tripod_mode):
        """Run intrinsic calibration"""

        print()
        print("Starting Intrinsic Calibration")

        self.int_calibration_process = subprocess.Popen(
            [
                sys.executable, 
                "intrinsic_calibration.py", 
                "--camera-id", selected_camera, 
                "--capture-name", capture_name, 
                "--checkerboard-width", str(board_width), 
                "--checkerboard-length", str(board_length), 
                "--square-size", str(float(square_size) * 0.001), 
                "--stereo", str(stereo), 
                "--tripod-mode", str(tripod_mode)
            ], 
            cwd='.'
        )

        self.is_calibrating_int = True

        return self.int_calibration_process

    def stop_intrinsic_calibration(self):

        print()
        print("Stopping Intrinsic Calibration")

        if hasattr(self, "int_calibration_process") and self.int_calibration_process:
            self.int_calibration_process.terminate()

            try:
                self.int_calibration_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.int_calibration_process.kill() 

            self.int_calibration_process = None

            self.is_calibrating_int = False

    def run_extrinsic_calibration(self, selected_camera, capture_name, marker_size, stereo):
        """Run extrinsic calibration"""

        print()
        print("Starting Marker Calibration")

        self.ext_calibration_process = subprocess.Popen(
            [
                sys.executable, 
                "marker_calibration.py", 
                "--camera-id", selected_camera, 
                "--capture-name", capture_name, 
                "--marker-length", str(marker_size), 
                "--stereo", str(stereo)
                
            ], 
            cwd='.'
        )

        self.is_calibrating_ext = True

    def stop_extrinsic_calibration(self):
        print()
        print("Stopping Marker Calibration")
        if hasattr(self, "ext_calibration_process") and self.ext_calibration_process:
            self.ext_calibration_process.terminate()

            try:
                self.ext_calibration_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ext_calibration_process.kill() 

            self.ext_calibration_process = None

            self.is_calibrating_ext = False

    def run_color_calibration(self, capture_name, selected_camera):
        """Run image preparation"""
        subprocess.run([sys.executable, "color_calibration.py", "--capture-name", capture_name, "--cam-id", selected_camera], cwd='.')

    def run_prepare_images(self, capture_name, stereo, frame_range, start_frame, end_frame, data_format, export_mesh, blender_scene, prep_images_button):
        
        print()
        print("Starting Image Preperation")

        self.is_processing = True

        self.stop_event = threading.Event()

        self.process_thread = threading.Thread(
            target=self.run_prepare_images_worker,
            args=(capture_name, stereo, frame_range, start_frame, end_frame, data_format, export_mesh, blender_scene, prep_images_button),
            daemon=True
        )
        
        self.process_thread.start()

    def run_prepare_images_worker(self, capture_name, stereo, frame_range, start_frame, end_frame, data_format, export_mesh, blender_scene, prep_images_button):
        """Run image preparation"""
        if stereo == False:
            self.prep_images_process = subprocess.Popen([sys.executable, "camera_calibration.py", "--capture-name", capture_name, "--stereo", "False"], cwd='.')
            self.prep_images_process.wait()
            
            if self.stop_event.is_set():
                return
            
            self.prep_images_process = subprocess.Popen([sys.executable, "process_images.py", "--capture-name", capture_name, "--stereo", "False", "--frame-range", str(frame_range), "--frame-start", start_frame, "--frame-end", end_frame, "--data-format", data_format], cwd='.')
            self.prep_images_process.wait()

            if self.stop_event.is_set():
                return
            
            self.prep_images_process = subprocess.Popen([sys.executable, "create_pointcloud.py", "--capture-name", capture_name,  "--frame-range", str(frame_range), "--frame-start", start_frame, "--frame-end", end_frame, "--data-format", data_format, "--export-mesh", str(export_mesh)], cwd='.')
            self.prep_images_process.wait()

            if self.stop_event.is_set():
                return

            if blender_scene:
                blender_exe = "blender --python"

                self.prep_images_process = subprocess.Popen([
                    "blender",
                    "-b",
                    "-P", "setup_blender.py",
                    "--",
                    "--capture-name", capture_name
                ])

                return_code = self.prep_images_process.wait()

                if return_code != 0:
                    print(f"Blender failed with exit code {return_code}")
                    return
                
                if self.stop_event.is_set():
                    return
                
            if prep_images_button is not None:
                prep_images_button.configure(text="Start")
                self.is_processing = False
                
            

        else:
            self.prep_images_process = subprocess.Popen([sys.executable, "camera_calibration.py", "--capture-name", capture_name, "--stereo", "True"], cwd='.')
            self.prep_images_process.wait()

            if self.stop_event.is_set():
                    return
            
            self.prep_images_process = subprocess.Popen([sys.executable, "process_images.py", "--capture-name", capture_name, "--stereo", "True", "--frame-range", str(frame_range), "--frame-start", start_frame, "--frame-end", end_frame, "--data-format", data_format], cwd='.')
            self.prep_images_process.wait()

            if self.stop_event.is_set():
                    return

            self.prep_images_process = subprocess.Popen([sys.executable, "create_pointcloud_stereo.py", "--capture-name", capture_name, "--frame-range", str(frame_range), "--frame-start", start_frame, "--frame-end", end_frame, "--data-format", data_format], cwd='.')
            self.prep_images_process.wait()

            if self.stop_event.is_set():
                    return
            
    def stop_prepare_images(self):
        print()
        print("Stopping Image Preperation")

        if hasattr(self, "prep_images_process") and self.prep_images_process:
            self.prep_images_process.terminate()

            self.stop_event.set()

            try:
                self.prep_images_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.prep_images_process.kill() 

            self.prep_images_process = None

            self.is_processing = False
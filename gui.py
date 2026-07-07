import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import json
import os
import time
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import socket
import re

from sympy import root
from network_control import CameraNetworkController

# Default settings
DEFAULT_CAPTURE_NAME = "Capture"
DEFAULT_FPS = 24
DEFAULT_SHUTTER_ANGLE = 180
DEFAULT_GAIN = 5
DEFAULT_COLOR_TEMP = 3500
DEFAULT_BOARD_WIDTH = 18
DEFAULT_BOARD_LENGTH = 10
DEFAULT_SQUARE_SIZE = 40
DEFAULT_MARKER_SIZE = 254
DEFAULT_ENTRIES = ["camera001.local"]
DEFAULT_SELECTED_CAMERA = 1
DEFAULT_CAPTURE_FILE = "Capture"
DEFAULT_START_FRAME = 0
DEFAULT_END_FRAME = 100
DEFAULT_STEREO_MODE = False
DEFAULT_CALIB_MODE = False
DEFAULT_TRIPOD_MODE = True
DEFAULT_FRAME_RANGE_MODE = False
DEFAULT_DATA_FORMAT = "postshot"
DEFAULT_EXPORT_MESH = False
DEFAULT_BLENDER_SCENE = False
SETTINGS_FILE = "camera_settings.json"

class PiViewer:
    def __init__(self):
        self.is_streaming = False
        self.viewer_socket = None
        self.current_frame = None
        self.frame_callback = None
        
    def connect_to_pi(self, ip_address):
        """Connect to Pi for live streaming"""
        try:
            self.viewer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.viewer_socket.settimeout(5)
            # Use a different port for viewer (e.g., 5006)
            self.viewer_socket.connect((ip_address, 5006))
            self.viewer_socket.sendall(b"STREAM_REQUEST")
            return True
        except Exception as e:
            #print(f"Failed to connect to Pi viewer: {e}")
            return False
    
    def start_streaming(self, frame_callback):
        """Start receiving frames from Pi"""
        if not self.viewer_socket:
            return False
            
        self.frame_callback = frame_callback
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        return True
    
    def _stream_worker(self):
        """Worker thread to receive frames"""
        buffer = b""
        self.last_data_time = time.time()
        while self.is_streaming and self.viewer_socket:
            try:
                data = self.viewer_socket.recv(4096)
                self.last_data_time = time.time()
                if not data:
                    break
                    
                buffer += data
                
                # Look for JPEG markers
                while True:
                    start_marker = buffer.find(b'\xff\xd8')
                    end_marker = buffer.find(b'\xff\xd9')
                    
                    if start_marker != -1 and end_marker != -1 and end_marker > start_marker:
                        # Extract JPEG frame
                        jpeg_data = buffer[start_marker:end_marker + 2]
                        buffer = buffer[end_marker + 2:]
                        
                        # Decode image
                        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None and self.frame_callback:
                            self.frame_callback(frame)
                    else:
                        break

            except Exception as e:
                print(f"Streaming error: {e}")
                break

            if time.time() - self.last_data_time > 3:  # timeout after 10 seconds of no data
                    print("Viewer connection timed out.")
                    break
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        if self.viewer_socket:
            try:
                self.viewer_socket.close()
            except:
                pass
            self.viewer_socket = None

class CameraControlGUI:
    def __init__(self):
        self.network_controller = CameraNetworkController()
        self.pi_viewer = PiViewer()
        self.entries = []
        self.is_initialized = False
        self.viewer_window = None
        self.video_label = None
        
        # Load settings
        self.settings = self.load_settings()
        self.update_defaults_from_settings()
        
        # Initialize GUI
        self.setup_gui()
        self.load_ip_entries(self.settings.get("entries", DEFAULT_ENTRIES))
        self.update_viewer()

    def load_settings(self):
        """Load settings from file, return defaults if file doesn't exist"""
        default_settings = {
            "capture_name": DEFAULT_CAPTURE_NAME,
            "fps": DEFAULT_FPS,
            "shutter_angle": DEFAULT_SHUTTER_ANGLE,
            "gain": DEFAULT_GAIN,
            "color_temp": DEFAULT_COLOR_TEMP,
            "undistort": True,
            "board_width": DEFAULT_BOARD_WIDTH,
            "board_length": DEFAULT_BOARD_LENGTH,
            "square_size": DEFAULT_SQUARE_SIZE,
            "marker_size": DEFAULT_MARKER_SIZE,
            "entries": DEFAULT_ENTRIES,
            "selected_camera": DEFAULT_SELECTED_CAMERA,
            "capture_file": DEFAULT_CAPTURE_NAME,
            "frame_range": DEFAULT_FRAME_RANGE_MODE,
            "start_frame": DEFAULT_START_FRAME,
            "end_frame": DEFAULT_END_FRAME,
            "export_mesh": DEFAULT_EXPORT_MESH,
            "blender_scene": DEFAULT_BLENDER_SCENE,
            "stereo": DEFAULT_STEREO_MODE,
            "calib_mode": DEFAULT_CALIB_MODE,
            "tripod_mode": DEFAULT_TRIPOD_MODE,
        }
        
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                print(f"Loaded settings from {SETTINGS_FILE}")
                return settings
            else:
                print("Settings file not found, using defaults")
                return default_settings
        except Exception as e:
            print(f"Error loading settings: {e}, using defaults")
            return default_settings

    def update_defaults_from_settings(self):
        """Update default values from loaded settings"""
        global DEFAULT_CAPTURE_NAME, DEFAULT_FPS, DEFAULT_SHUTTER_ANGLE, DEFAULT_GAIN
        global DEFAULT_COLOR_TEMP, DEFAULT_BOARD_WIDTH, DEFAULT_BOARD_LENGTH, DEFAULT_SQUARE_SIZE, DEFAULT_SELECTED_CAMERA, DEFAULT_CAPTURE_FILE, DEFAULT_START_FRAME, DEFAULT_END_FRAME, DEFAULT_ENTRIES
        
        DEFAULT_CAPTURE_NAME = self.settings["capture_name"]
        DEFAULT_FPS = self.settings["fps"]
        DEFAULT_SHUTTER_ANGLE = self.settings["shutter_angle"]
        DEFAULT_GAIN = self.settings["gain"]
        DEFAULT_COLOR_TEMP = self.settings["color_temp"]
        DEFAULT_BOARD_WIDTH = self.settings["board_width"]
        DEFAULT_BOARD_LENGTH = self.settings["board_length"]
        DEFAULT_SQUARE_SIZE = self.settings["square_size"]
        DEFAULT_SELECTED_CAMERA = self.settings["selected_camera"]
        DEFAULT_CAPTURE_FILE = self.settings["capture_file"]
        DEFAULT_START_FRAME = self.settings["start_frame"]
        DEFAULT_END_FRAME = self.settings["end_frame"]
        DEFAULT_ENTRIES = self.settings["entries"]

    def setup_gui(self):

        self.root = ctk.CTk()
        self.root.configure(fg_color="#23272D")
        self.root.title("Main Window")
        self.root.geometry("1920x1055")
        self.root.update_idletasks()

        self.capture_name_var = tk.StringVar(value=DEFAULT_CAPTURE_NAME)
        self.fps_var = tk.StringVar(value=str(DEFAULT_FPS))
        self.shutter_var = tk.StringVar(value=str(DEFAULT_SHUTTER_ANGLE))
        self.gain_var = tk.StringVar(value=str(DEFAULT_GAIN))
        self.temp_var = tk.StringVar(value=str(DEFAULT_COLOR_TEMP))
        self.board_width_var = tk.StringVar(value=str(DEFAULT_BOARD_WIDTH))
        self.board_length_var = tk.StringVar(value=str(DEFAULT_BOARD_LENGTH))
        self.square_size_var = tk.StringVar(value=str(DEFAULT_SQUARE_SIZE))
        self.marker_size_var = tk.StringVar(value=str(DEFAULT_MARKER_SIZE))
        self.selected_camera_var = tk.StringVar(value=str(DEFAULT_SELECTED_CAMERA))
        self.capture_file_var = tk.StringVar(value=DEFAULT_CAPTURE_FILE)
        self.frame_range_var = tk.BooleanVar(value=self.settings.get("frame_range", False))
        self.start_frame_var = tk.StringVar(value=str(DEFAULT_START_FRAME))
        self.end_frame_var = tk.StringVar(value=str(DEFAULT_END_FRAME))
        self.data_format_var = tk.StringVar(value=DEFAULT_DATA_FORMAT)
        self.export_mesh_var = tk.BooleanVar(value=self.settings.get("export_mesh", False))
        self.blender_scene_var = tk.BooleanVar(value=self.settings.get("blender_scene", False))
        self.stereo_var = tk.BooleanVar(value=self.settings.get("stereo", False))
        self.calib_mode_var = tk.BooleanVar(value=self.settings.get("calib_mode", False))
        self.tripod_var = tk.BooleanVar(value=self.settings.get("tripod_mode", True))

        geometryX = 0
        geometryY = 0

        self.root.geometry("+%d+%d"%(geometryX, geometryY))

        frame = ctk.CTkFrame(master=self.root)
        frame.configure(fg_color="#EDECEC", width=142, height=110)
        frame.place(x=2095, y=607)
        frame.pack_propagate(False),frame.grid_propagate(False)

        # setup icon
        image = ctk.CTkImage(
            light_image=Image.open("gui_artifacts/icon.png"),
            dark_image=Image.open("gui_artifacts/icon.png"),
            size=(35, 31)
        )

        # create camera setup frame
        icon = ctk.CTkLabel(master=self.root, image=image, text="")
        icon.place(x=40, y=16)

        title = ctk.CTkLabel(master=self.root, text="OpenVolCap")
        title.configure(fg_color="transparent", width=100, height=47, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=24, weight="bold"))
        title.place(x=84, y=10)

        setup_frame = ctk.CTkFrame(master=self.root)
        setup_frame.configure(fg_color="#4e4949", width=275, height=835)
        setup_frame.place(x=40, y=60)
        setup_frame.pack_propagate(False),setup_frame.grid_propagate(False)

        setup_label = ctk.CTkLabel(master=setup_frame, text="Camera Setup")
        setup_label.configure(fg_color="#4e4949", width=235, height=47, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=35, weight="bold"))
        setup_label.place(x=20, y=20)

        ip_label = ctk.CTkLabel(master=setup_frame, text="Camera IPs:")
        ip_label.configure(fg_color="#4e4949", width=134, height=46, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        ip_label.place(x=20, y=79)

        add_ip_button = ctk.CTkButton(master=setup_frame, text="+")
        add_ip_button.configure(fg_color="#029cff", hover_color="#1e538d", width=40, height=30, text_color="#fff", corner_radius=5, command=self.add_ip_entry, font=ctk.CTkFont(family="Arial", size=24))
        add_ip_button.place(x=170, y=85)

        delete_ip_button = ctk.CTkButton(master=setup_frame, text="-")
        delete_ip_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=40, height=30, text_color="#fff", corner_radius=5, command=self.remove_ip_entry, font=ctk.CTkFont(family="Arial", size=24))
        delete_ip_button.place(x=215, y=85)
        
        ip_entry_frame = ctk.CTkFrame(master=setup_frame)
        ip_entry_frame.configure(fg_color="#23272d", width=238, height=616)
        ip_entry_frame.place(x=20, y=135)
        ip_entry_frame.pack_propagate(False),ip_entry_frame.grid_propagate(False)

        canvas = ctk.CTkCanvas(ip_entry_frame, width=220, height=610, bg="#232627", border=0, highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(ip_entry_frame, orientation="vertical", command=canvas.yview)
        self.ip_entries_frame = ctk.CTkFrame(canvas, fg_color="#232627")

        self.ip_entries_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.ip_entries_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Use grid instead of pack
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        delete_button = ctk.CTkButton(master=setup_frame, text="Delete Pi Data")
        delete_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=115, height=40, text_color="#fff", corner_radius=5, command=self.on_delete_pressed, font=ctk.CTkFont(family="Arial", size=18))
        delete_button.place(x=15, y=774)

        reboot_button = ctk.CTkButton(master=setup_frame, text="Reboot Pi's")
        reboot_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=115, height=40, text_color="#fff", corner_radius=5, command=self.on_reboot_pressed, font=ctk.CTkFont(family="Arial", size=18))
        reboot_button.place(x=145, y=774)

        # create camera config frame
        control_frame = ctk.CTkFrame(master=self.root)
        control_frame.configure(fg_color="#4e4949", width=420, height=500)
        control_frame.place(x=1459, y=60)
        control_frame.pack_propagate(False),control_frame.grid_propagate(False)

        control_label = ctk.CTkLabel(master=control_frame, text="Camera Control")
        control_label.configure(fg_color="transparent", width=269, height=57, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=35, weight="bold"))
        control_label.place(x=14, y=9)

        name_input_label = ctk.CTkLabel(master=control_frame, text="Capture Name:")
        name_input_label.configure(fg_color="transparent", width=166, height=57, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        name_input_label.place(x=5, y=67)

        capture_entry = ctk.CTkEntry(master=control_frame, placeholder_text="capture", textvariable=self.capture_name_var)
        capture_entry.configure(fg_color="#343739", width=228, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(family="Arial", size=18))
        capture_entry.place(x=180, y=80)

        fps_label = ctk.CTkLabel(master=control_frame, text="FPS:")
        fps_label.configure(fg_color="transparent", width=70, height=52, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        fps_label.place(x=14, y=107)

        fps_entry = ctk.CTkEntry(master=control_frame, placeholder_text="24", textvariable=self.fps_var)
        fps_entry.configure(fg_color="#343739", width=228, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(family="Arial", size=18))
        fps_entry.place(x=180, y=120)

        shutter_angle_label = ctk.CTkLabel(master=control_frame, text="Shutter Angle:")
        shutter_angle_label.configure(fg_color="transparent", width=160, height=51, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        shutter_angle_label.place(x=1, y=145)

        shutter_entry = ctk.CTkEntry(master=control_frame, placeholder_text="180", textvariable=self.shutter_var)
        shutter_entry.configure(fg_color="#343739", width=228, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(family="Arial", size=18))
        shutter_entry.place(x=180, y=160)

        gain_label = ctk.CTkLabel(master=control_frame, text="Gain:")
        gain_label.configure(fg_color="transparent", width=63, height=53, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        gain_label.place(x=16, y=187)

        gain_entry = ctk.CTkEntry(master=control_frame, placeholder_text="10.0", textvariable=self.gain_var)
        gain_entry.configure(fg_color="#343739", width=228, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(family="Arial", size=18))
        gain_entry.place(x=180, y=200)

        color_temp_label = ctk.CTkLabel(master=control_frame, text="Color Temp:")
        color_temp_label.configure(fg_color="transparent", width=141, height=56, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        color_temp_label.place(x=5, y=227)

        color_temp_entry = ctk.CTkEntry(master=control_frame, placeholder_text="3500", textvariable=self.temp_var)
        color_temp_entry.configure(fg_color="#343739", width=228, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(family="Arial", size=18))
        color_temp_entry.place(x=180, y=240)

        cal_mode_label = ctk.CTkLabel(master=control_frame, text="Calibration Mode:")
        cal_mode_label.configure(fg_color="transparent", width=190, height=54, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        cal_mode_label.place(x=0, y=272)

        calibration_checkbox = ctk.CTkCheckBox(master=control_frame, text="", variable=self.calib_mode_var)
        calibration_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=50, height=28, text_color="#fff", corner_radius=5)
        calibration_checkbox.place(x=180, y=285)

        stereo_mode_label = ctk.CTkLabel(master=control_frame, text="Stereo Mode:")
        stereo_mode_label.configure(fg_color="transparent", width=156, height=52, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=18))
        stereo_mode_label.place(x=2, y=312)

        stereo_checkbox = ctk.CTkCheckBox(master=control_frame, text="", variable=self.stereo_var)
        stereo_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=50, height=25, text_color="#fff", corner_radius=5)
        stereo_checkbox.place(x=180, y=325)

        initialize_button = ctk.CTkButton(master=control_frame, text="Initialize Cameras")
        initialize_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=195, height=45, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18), command=self.initialize_cameras)
        initialize_button.place(x=9, y=380)

        self.capture_button = ctk.CTkButton(master=control_frame, text="Start Capture")
        self.capture_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=194, height=100, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18), command=self.toggle_capture)
        self.capture_button.place(x=215, y=380)

        disconnect_cameras_button = ctk.CTkButton(master=control_frame, text="Disconnect Cameras")
        disconnect_cameras_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=195, height=45, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18), command=self.disconnect_cameras)
        disconnect_cameras_button.place(x=10, y=435)

        # create process images frame
        process_frame = ctk.CTkFrame(master=self.root)
        process_frame.configure(fg_color="#4e4949", width=420, height=315)
        process_frame.place(x=1460, y=579)
        process_frame.pack_propagate(False),process_frame.grid_propagate(False)

        process_label = ctk.CTkLabel(master=process_frame, text="Process Images")
        process_label.configure(fg_color="transparent", width=269, height=57, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=35, weight="bold"))
        process_label.place(x=14, y=9)

        capture_name_label = ctk.CTkLabel(master=process_frame, text="Capture File:")
        capture_name_label.configure(fg_color="transparent", width=125, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18))
        capture_name_label.place(x=5, y=72)

        self.capture_files = []

        self.capture_name_dropdown = ctk.CTkComboBox(master=process_frame, values=self.capture_files, variable=self.capture_file_var)
        self.capture_name_dropdown.configure(fg_color="#4e4949", width=228, height=30, text_color="#fff", corner_radius=5)
        self.capture_name_dropdown.place(x=180, y=72)

        self.load_capture_files()

        data_format_label = ctk.CTkLabel(master=process_frame, text="Data Format:")
        data_format_label.configure(fg_color="transparent", width=125, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18))
        data_format_label.place(x=5, y=115)

        datastype_dropdown = ctk.CTkComboBox(master=process_frame, values=["postshot", "instant_ngp"], variable=self.data_format_var)
        datastype_dropdown.configure(fg_color="#4e4949", width=228, height=30, text_color="#fff", corner_radius=5)
        datastype_dropdown.place(x=180, y=115)

        frame_range_checkbox = ctk.CTkCheckBox(master=process_frame, text="", variable=self.frame_range_var)
        frame_range_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=50, height=28, text_color="#fff", corner_radius=5)
        frame_range_checkbox.place(x=180, y=163)

        frame_range_label = ctk.CTkLabel(master=process_frame, text="Frame Range:")
        frame_range_label.configure(fg_color="transparent", width=120, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18))
        frame_range_label.place(x=10, y=158)

        frame_start_entry = ctk.CTkEntry(master=process_frame, placeholder_text="1", textvariable=self.start_frame_var)
        frame_start_entry.configure(fg_color="#343739", width=95, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(size=18))
        frame_start_entry.place(x=214, y=160)

        frame_end_entry = ctk.CTkEntry(master=process_frame, placeholder_text="100", textvariable=self.end_frame_var)
        frame_end_entry.configure(fg_color="#343739", width=95, height=30, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557", font=ctk.CTkFont(size=18))
        frame_end_entry.place(x=315, y=160)

        export_mesh_label = ctk.CTkLabel(master=process_frame, text="Export Mesh:")
        export_mesh_label.configure(fg_color="transparent", width=120, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18))
        export_mesh_label.place(x=8, y=201)

        export_mesh_checkbox = ctk.CTkCheckBox(master=process_frame, text="", variable=self.export_mesh_var)
        export_mesh_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=50, height=28, text_color="#fff", corner_radius=5)
        export_mesh_checkbox.place(x=180, y=201)

        blender_scene_label = ctk.CTkLabel(master=process_frame, text="Blender Scene:")
        blender_scene_label.configure(fg_color="transparent", width=80, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=18))
        blender_scene_label.place(x=212, y=201)

        blender_scene_checkbox = ctk.CTkCheckBox(master=process_frame, text="", variable=self.blender_scene_var)
        blender_scene_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=28, height=28, text_color="#fff", corner_radius=5)
        blender_scene_checkbox.place(x=383, y=201)

        self.prep_images_button = ctk.CTkButton(master=process_frame, text="Start")
        self.prep_images_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=400, height=55, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=24), command=self.run_prepare_images)
        self.prep_images_button.place(x=10, y=245)

        # Create camera selection frame
        selection_frame = ctk.CTkFrame(master=self.root)
        selection_frame.configure(fg_color="#4e4949", width=275, height=94)
        selection_frame.place(x=40, y=920)
        selection_frame.pack_propagate(False),selection_frame.grid_propagate(False)

        selected_cam_label = ctk.CTkLabel(master=selection_frame, text="Selected Camera")
        selected_cam_label.configure(fg_color="transparent", width=197, height=41, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=24, weight="bold"))
        selected_cam_label.place(x=4, y=25)

        selected_cam_entry = ctk.CTkEntry(master=selection_frame, placeholder_text="1", textvariable=self.selected_camera_var)
        selected_cam_entry.configure(fg_color="#343739", width=35, height=51, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")
        selected_cam_entry.place(x=220, y=22)

        # Create intrinsic calibration frame
        intrinsic_cal_frame = ctk.CTkFrame(master=self.root)
        intrinsic_cal_frame.configure(fg_color="#4e4949", width=690, height=96)
        intrinsic_cal_frame.place(x=346, y=920)
        intrinsic_cal_frame.pack_propagate(False),intrinsic_cal_frame.grid_propagate(False)

        intrinsic_label = ctk.CTkLabel(master=intrinsic_cal_frame, text="Intrinsic Calibration")
        intrinsic_label.configure(fg_color="transparent", width=223, height=52, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=24, weight="bold"))
        intrinsic_label.place(x=10, y=20)

        checkerboard_size_label = ctk.CTkLabel(master=intrinsic_cal_frame, text="Checkerboard Size")
        checkerboard_size_label.configure(fg_color="transparent", width=100, height=40, text_color="#fff", corner_radius=5, font=ctk.CTkFont())
        checkerboard_size_label.place(x=242, y=53)

        checkerboard_size_x_entry = ctk.CTkEntry(master=intrinsic_cal_frame, placeholder_text="10", textvariable=self.board_width_var)
        checkerboard_size_x_entry.configure(fg_color="#343739", width=42, height=43, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")
        checkerboard_size_x_entry.place(x=254, y=10)

        checker_board_size_y_entry = ctk.CTkEntry(master=intrinsic_cal_frame, placeholder_text="18", textvariable=self.board_length_var)
        checker_board_size_y_entry.configure(fg_color="#343739", width=42, height=43, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")
        checker_board_size_y_entry.place(x=304, y=10)

        square_size_label = ctk.CTkLabel(master=intrinsic_cal_frame, text="Square Size (mm)")
        square_size_label.configure(fg_color="transparent", width=105, height=40, text_color="#fff", corner_radius=5, font=ctk.CTkFont())
        square_size_label.place(x=370, y=53)

        checkerboard_square_size_entry = ctk.CTkEntry(master=intrinsic_cal_frame, placeholder_text="40.0", textvariable=self.square_size_var)
        checkerboard_square_size_entry.configure(fg_color="#343739", width=60, height=43, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")
        checkerboard_square_size_entry.place(x=394, y=10)

        tripod_checkbox = ctk.CTkCheckBox(master=intrinsic_cal_frame, text="Tripod Mode", variable=self.tripod_var)
        tripod_checkbox.configure(fg_color="#029CFF", hover_color="#1e538d", width=120, height=30, text_color="#fff", corner_radius=5)
        tripod_checkbox.place(x=490, y=34)

        self.intrinsic_cal_button = ctk.CTkButton(master=intrinsic_cal_frame, text="Start")
        self.intrinsic_cal_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=74, height=65, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=24), command=self.run_intrinsic_calibration)
        self.intrinsic_cal_button.place(x=600, y=15)

        # Create extrinsic calibration frame
        extrinsic_cal_frame = ctk.CTkFrame(master=self.root)
        extrinsic_cal_frame.configure(fg_color="#4e4949", width=470, height=95)
        extrinsic_cal_frame.place(x=1049, y=919)
        extrinsic_cal_frame.pack_propagate(False),extrinsic_cal_frame.grid_propagate(False)

        extrinsic_label = ctk.CTkLabel(master=extrinsic_cal_frame, text="Extrinsic Calibration")
        extrinsic_label.configure(fg_color="transparent", width=221, height=44, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=24, weight="bold"))
        extrinsic_label.place(x=10, y=24)

        apriltag_size_entry = ctk.CTkEntry(master=extrinsic_cal_frame, placeholder_text="254.0", textvariable=self.marker_size_var)
        apriltag_size_entry.configure(fg_color="#343739", width=60, height=40, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")
        apriltag_size_entry.place(x=270, y=10)

        apriltag_size_label = ctk.CTkLabel(master=extrinsic_cal_frame, text="Apriltag Size (mm)")
        apriltag_size_label.configure(fg_color="transparent", width=121, height=20, text_color="#fff", corner_radius=5, font=ctk.CTkFont())
        apriltag_size_label.place(x=244, y=60)

        self.extrinsic_cal_button = ctk.CTkButton(master=extrinsic_cal_frame, text="Start", font=ctk.CTkFont(size=24))
        self.extrinsic_cal_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=74, height=65, text_color="#fff", corner_radius=5, command=self.run_extrinsic_calibration)
        self.extrinsic_cal_button.place(x=384, y=14)

        # Create Color Calibration Frame
        color_cal_frame = ctk.CTkFrame(master=self.root)
        color_cal_frame.configure(fg_color="#4e4949", width=348, height=95)
        color_cal_frame.place(x=1532, y=919)
        color_cal_frame.pack_propagate(False),color_cal_frame.grid_propagate(False)

        color_calibration_label = ctk.CTkLabel(master=color_cal_frame, text="Color Calibration")
        color_calibration_label.configure(fg_color="transparent", width=189, height=35, text_color="#fff", corner_radius=5, font=ctk.CTkFont(family="Arial", size=24, weight="bold"))
        color_calibration_label.place(x=14, y=29)

        self.color_cal_button = ctk.CTkButton(master=color_cal_frame, text="Start", font=ctk.CTkFont(size=24))
        self.color_cal_button.configure(fg_color="#029CFF", hover_color="#1e538d", width=74, height=65, text_color="#fff", corner_radius=5, command=self.run_color_calibration)
        self.color_cal_button.place(x=259, y=14)

        # Create viewer frame
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TNotebook",
            background="#23272D",
            borderwidth=0,
            relief="flat"
        )

        style.configure("TNotebook.Tab", 
                        background="#353131",
                        foreground="white",   
                        darkcolor="#4e4949",
                        lightcolor="#4e4949",  
                        borderwidth=0,
                        relief="flat")
        
        style.map("TNotebook.Tab", 
                  background=[("selected", "#4e4949")],
                  foreground=[("selected", "white")])

        viewer_frame = ctk.CTkFrame(master=self.root)
        viewer_frame.configure(fg_color="#5a4040", width=1100, height=835)
        viewer_frame.place(x=340, y=60)

        viewer_frame.pack_propagate(False)

        self.notebook = ttk.Notebook(viewer_frame)
        self.notebook.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_viewer(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

        camera_ips = self.get_camera_ips()
        # Create tabs for each camera
        self.camera_tabs = {}
        for i, ip in enumerate(camera_ips):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"Camera {i+1}")
            
            # Video display label
            video_label = tk.Label(frame, text="Cameras Disconnected", bg='black', fg='white')
            video_label.pack(expand=True, fill='both')
            
            # Status label
            status_label = tk.Label(frame, text=f"Status: Disconnected from {ip}", relief='sunken', anchor='w')
            status_label.pack(fill='x', side='bottom')
            
            self.camera_tabs[ip] = {
                'frame': frame,
                'video_label': video_label,
                'status_label': status_label,
                'viewer': PiViewer(),
                'connected': False
            }

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        if self.notebook.tabs():
            self.notebook.select(0)

    def connect_current_tab(self):
        selected = self.notebook.select()

        tab_index = self.notebook.index(selected)

        ip = list(self.camera_tabs.keys())[tab_index]

        camera = self.camera_tabs[ip]

        if not camera['connected']:
            self.connect_to_camera_viewer(ip)
            camera['connected'] = True

    def show_disconnected(self):
        for ip, camera in self.camera_tabs.items():
            camera['video_label'].config(
                image="",
                text="Cameras Disconnected",
                bg="black",
                fg="white"
            )

            camera['video_label'].image = None

            camera['status_label'].config(
                text="Status: Disconnected"
            )

    def on_tab_changed(self, event):
        # Optional: disconnect all inactive viewers
        for ip, camera in self.camera_tabs.items():
            if camera['connected']:
                camera['viewer'].stop_streaming()
                camera['connected'] = False

        if self.is_initialized:
            self.connect_current_tab()

    def connect_to_camera_viewer(self, ip_address):
        """Connect to a specific camera for viewing"""
        if ip_address in self.camera_tabs:
            tab_info = self.camera_tabs[ip_address]
            viewer = tab_info['viewer']
            
            # Try to connect
            if viewer.connect_to_pi(ip_address):
                tab_info['status_label'].config(text=f"Status: Connected to {ip_address}")
                # Start streaming with callback
                viewer.start_streaming(lambda frame: self.update_video_display(ip_address, frame))
            else:
                tab_info['status_label'].config(text=f"Connecting to {ip_address}")
                tab_info['video_label'].config(text="Connecting...")

    def update_video_display(self, ip_address, frame):
        """Update video display with new frame"""
        if not self.camera_tabs[ip_address]['connected']:
            return
        
        if ip_address in self.camera_tabs:
            try:
                tab_info = self.camera_tabs[ip_address]
                
                # Resize frame to fit display
                height, width = frame.shape[:2]
                max_width, max_height = 1280, 960
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update label (must be done in main thread)
                self.root.after(0, lambda: self._update_label(ip_address, photo))
                
            except Exception as e:
                print(f"Error updating video display: {e}")

    def _update_label(self, ip_address, photo):
        """Update label in main thread"""
        if ip_address in self.camera_tabs and self:
            tab_info = self.camera_tabs[ip_address]

            if not tab_info['connected']:
                return
            
            tab_info['video_label'].config(image=photo)
            tab_info['video_label'].image = photo  # Keep a reference

    def get_camera_ips(self):
        """Get list of camera IPs from entries"""
        return [entry.get() for label, entry in self.entries if entry.get().strip() != ""]

    def get_parameters(self):
        """Get current parameter values from GUI"""
        try:
            return {
                'capture_name': self.capture_name_var.get(),
                'fps': int(self.fps_var.get()),
                'shutter_angle': int(self.shutter_var.get()),
                'gain': float(self.gain_var.get()),
                'color_temp': int(self.temp_var.get()),
                'stereo': self.stereo_var.get(),
                'tripod_mode': self.tripod_var.get(),
                'calib_mode': self.calib_mode_var.get(),
                'board_width': int(self.board_width_var.get()),
                'board_length': int(self.board_length_var.get()),
                'square_size': float(self.square_size_var.get()),
                'marker_size': float(self.marker_size_var.get()),
                'selected_camera': self.selected_camera_var.get(),
                'capture_file': self.capture_file_var.get(),
                'frame_range': self.frame_range_var.get(),
                'data_format': self.data_format_var.get(),
                'export_mesh': self.export_mesh_var.get(),
                'blender_scene': self.blender_scene_var.get(),
                'start_frame': self.start_frame_var.get(),
                'end_frame': self.end_frame_var.get()
            }
        except ValueError as e:
            raise ValueError(f"Invalid parameter values: {e}")

    def connect_after_init(self):
        print("Cameras should be ready, connecting to viewers...")

        for ip, camera in self.camera_tabs.items():
            if camera['connected']:
                camera['viewer'].stop_streaming()
                camera['connected'] = False

        if self.notebook.tabs():
            self.notebook.select(0)

        self.connect_current_tab()

    def initialize_cameras(self):
        """Toggle camera initialization state"""
        if not self.is_initialized:
            try:
                params = self.get_parameters()
                camera_ips = self.get_camera_ips()
                
                if not camera_ips:
                    messagebox.showwarning("No Cameras", "No camera IPs specified.")
                    return
                
                success = self.network_controller.connect_cameras(
                    camera_ips,
                    params['capture_name'],
                    params['fps'],
                    params['shutter_angle'],
                    params['gain'],
                    params['color_temp'],
                    params['calib_mode'],
                    params['stereo']
                )
                
                if success:
                    self.is_initialized = True
                    print("Cameras initialized.")
                    self.update_viewer()
                    self.root.after(5000, self.connect_after_init)
    
                else:
                    messagebox.showerror("Connection Failed", "Failed to initialize cameras.")
                    
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        else:
            # Disconnect cameras
            messagebox.showerror("Failed to Initialize Cameras", "Cameras Already Initialized")

        self.load_capture_files()
    
    def disconnect_cameras(self):
            camera_ips = self.get_camera_ips()

            for ip, camera in self.camera_tabs.items():
                camera['viewer'].stop_streaming()
                camera['connected'] = False
    
            self.network_controller.disconnect_cameras(camera_ips)
            self.is_initialized = False
        
            self.capture_button.configure(text="Start Capture")
            self.show_disconnected()


    def toggle_capture(self):
        """Toggle capture state"""
        if not self.network_controller.camera_connected:
            messagebox.showwarning("Not Connected", "Please connect to the camera first.")
            return

        if not self.network_controller.is_capturing:
            try:
                params = self.get_parameters()
                camera_ips = self.get_camera_ips()

                if not camera_ips:
                    messagebox.showwarning("No Cameras", "No camera IPs specified.")
                    return

                success = self.network_controller.start_capture(params['fps'])
                if success:
                    self.capture_button.configure(text="Stop Capture")
                else:
                    messagebox.showerror("Capture Failed", "Failed to start capture.")
                    
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        else:
            self.network_controller.stop_capture()
            self.capture_button.configure(text="Start Capture")
            self.increment_name()
            # Reset initialization state
            self.is_initialized = False

    def add_ip_entry(self):
        """Add new IP entry field"""
        index = len(self.entries)
        label = ctk.CTkLabel(self.ip_entries_frame, text=f"IP/Hostname")
        label.configure(fg_color="transparent", width=30, height=20, text_color="#fff", corner_radius=5, font=ctk.CTkFont(size=13))
        entry = ctk.CTkEntry(self.ip_entries_frame, placeholder_text="ip address")
        entry.configure(fg_color="#343739", width=120, height=20, text_color="#fff", corner_radius=5, border_width=2, border_color="#505557")

        label.grid(row=index, column=0, padx=5, pady=2, sticky="e")
        entry.grid(row=index, column=1, padx=5, pady=2, sticky="w")

        self.entries.append((label, entry))
        self.update_viewer()  # Update viewer to reflect new camera entry

    def remove_ip_entry(self):
        """Remove last IP entry field"""
        if self.entries:
            label, entry = self.entries.pop()
            label.destroy()
            entry.destroy()
        self.update_viewer()  # Update viewer to reflect removed camera

    def load_ip_entries(self, ip_addresses):
        """Load IP entries from saved settings"""
        # Clear existing entries
        while self.entries:
            self.remove_ip_entry()
        
        # Add entries for each saved IP
        for ip in ip_addresses:
            self.add_ip_entry()
            # Set the IP address in the newly created entry
            if self.entries:
                _, entry = self.entries[-1]
                entry.insert(0, ip)
        
        print(f"Loaded {len(ip_addresses)} camera IP addresses")

    def load_capture_files(self):

        self.capture_files = []

        mnt_dir = "/mnt"

        try:
            cam_drives = sorted([
                f for f in os.listdir(mnt_dir)
                if os.path.isdir(os.path.join(mnt_dir, f))
                and f.startswith("cam_")
            ])

            for camera in cam_drives:
                captures_path = os.path.join(mnt_dir, camera, "captures")

                if not os.path.isdir(captures_path):
                    continue

                captures = sorted([
                    f for f in os.listdir(captures_path)
                    if os.path.isdir(os.path.join(captures_path, f))
                    and f != "dolphin"
                ])


                for capture in captures:
                    if capture not in self.capture_files:
                        self.capture_files.append(capture)
                
        except:
            self.capture_files = []

        self.capture_name_dropdown.configure(
            values=self.capture_files
        )

        if len(self.capture_files) < 1:
            self.capture_name_dropdown.set("No captures found")

    def increment_name(self):
        name = self.capture_name_var.get()

        match = re.match(r"^(.*)_(\d+)$", name)

        if match:
            base = match.group(1)
            number = int(match.group(2)) + 1
            width = len(match.group(2))
            name = f"{base}_{number:0{width}d}"
        else:
            name = f"{name}_001"

        self.capture_name_var.set(name)

    def on_delete_pressed(self):
        """Handle delete button press"""
        confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete all Pi data?")
        if confirm:
            camera_ips = self.get_camera_ips()
            self.network_controller.run_image_delete(camera_ips)
    
    def on_reboot_pressed(self):
        """Handle delete button press"""
        if self.is_initialized == True:
            confirm = messagebox.askyesno("Confirm Reboot", "Are you sure you want to reboot all Pi's connected?'")
            if confirm:
                camera_ips = self.get_camera_ips()
                self.network_controller.reboot_cameras(camera_ips)
                self.is_initialized = False
        else:
            messagebox.showwarning("Not Connected", "Camera's must be initialized to reboot")

    def run_prepare_images(self):
        """Run image preparation process"""
        if not self.network_controller.is_processing:
            self.prep_images_button.configure(text="Stop")
            params = self.get_parameters()
            self.network_controller.run_prepare_images(
                params['capture_file'],
                params['stereo'],
                params['frame_range'],
                params['start_frame'],
                params['end_frame'],
                params['data_format'],
                params['export_mesh'],
                params['blender_scene'],
                self.prep_images_button
            )

        else:
            self.network_controller.stop_prepare_images()
            self.prep_images_button.configure(text="Start")

    def run_intrinsic_calibration(self):
        """Run intrinsic calibration process"""
        if not self.network_controller.is_calibrating_int:
            self.intrinsic_cal_button.configure(text="Stop")
            params = self.get_parameters()
            self.network_controller.run_intrinsic_calibration(
                params['selected_camera'],
                params['capture_file'],
                params['board_width'],
                params['board_length'],
                params['square_size'],
                params['stereo'],
                params['tripod_mode']
            )

        else:
            self.network_controller.stop_intrinsic_calibration()
            self.intrinsic_cal_button.configure(text="Start")

    def run_extrinsic_calibration(self):
        """Run extrinsic calibration process"""
        if not self.network_controller.is_calibrating_ext:
            self.extrinsic_cal_button.configure(text="Stop")
            params = self.get_parameters()
            self.network_controller.run_extrinsic_calibration(
                params['selected_camera'],
                params['capture_file'],
                params['marker_size'] * 0.001,  # Convert mm to meters
                params['stereo']
            )

        else:
            self.network_controller.stop_extrinsic_calibration()
            self.extrinsic_cal_button.configure(text="Start")
    
    def run_color_calibration(self):
        """Run extrinsic calibration process"""
        params = self.get_parameters()
        self.network_controller.run_color_calibration(
            params['capture_file'],
            params['selected_camera']
        )

    def save_settings(self):
        """Save current settings to file"""
        try:
            params = self.get_parameters()
            current_ips = self.get_camera_ips()
            
            settings = {
                "capture_name": params['capture_name'],
                "fps": params['fps'],
                "shutter_angle": params['shutter_angle'],
                "gain": params['gain'],
                "color_temp": params['color_temp'],
                "stereo": params['stereo'],
                "calib_mode": params['calib_mode'],
                "tripod_mode": params['tripod_mode'],
                "board_width": params['board_width'],
                "board_length": params['board_length'],
                "square_size": params['square_size'],
                "selected_camera": params['selected_camera'],
                "capture_file": params['capture_file'],
                "frame_range": params['frame_range'],
                "start_frame": params['start_frame'],
                "end_frame": params['end_frame'],
                "data_format": params['data_format'],
                "export_mesh": params['export_mesh'],
                "blender_scene": params['blender_scene'],
                "entries": current_ips
            }
            
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Settings saved to {SETTINGS_FILE}")
            
        except Exception as e:
            print(f"Error saving settings: {e}")

    def on_closing(self):
        """Handle program shutdown"""
        print("Saving current settings...")
        self.save_settings()
        print("Program closing - sending END signal to cameras...")
        
        # Always send END signal when closing
        camera_ips = self.get_camera_ips()
        self.network_controller.send_end_signal(camera_ips)
        
        # Give a moment for the signal to be sent
        time.sleep(0.1)
        
        # Destroy the window
        self.root.destroy()

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CameraControlGUI()
    app.run()
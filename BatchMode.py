# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame, QMessageBox, QComboBox, QSpacerItem, QSizePolicy) # Thêm QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, QSize, QTimer # Added QTimer if needed for future progress updates, although the toggle_process logic seems removed/incomplete now
from PyQt5.QtGui import QFont, QPixmap, QImage
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import threading
import traceback # Added for better error reporting

# --- Helper functions (pil_to_qpixmap, cv_image_to_qpixmap) ---
def pil_to_qpixmap(pil_image):
    try:
        if pil_image.mode == "RGB":
            img_byte_arr = pil_image.tobytes("raw", "RGB")
            bytes_per_line = pil_image.size[0] * 3
            qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], bytes_per_line, QImage.Format_RGB888)
        elif pil_image.mode == "RGBA":
            img_byte_arr = pil_image.tobytes("raw", "RGBA")
            bytes_per_line = pil_image.size[0] * 4
            qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], bytes_per_line, QImage.Format_RGBA8888)
        elif pil_image.mode == "L": # Grayscale
            # Convert grayscale to RGB for QPixmap compatibility in some cases
            rgb_image = pil_image.convert("RGB")
            img_byte_arr = rgb_image.tobytes("raw", "RGB")
            bytes_per_line = rgb_image.size[0] * 3
            qimage = QImage(img_byte_arr, rgb_image.size[0], rgb_image.size[1], bytes_per_line, QImage.Format_RGB888)
            # Alternatively, for direct Grayscale display:
            # bytes_per_line = pil_image.size[0]
            # qimage = QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], bytes_per_line, QImage.Format_Grayscale8)
        elif pil_image.mode == "P": # Palette
            rgba_image = pil_image.convert("RGBA")
            img_byte_arr = rgba_image.tobytes("raw", "RGBA")
            bytes_per_line = rgba_image.size[0] * 4
            qimage = QImage(img_byte_arr, rgba_image.size[0], rgba_image.size[1], bytes_per_line, QImage.Format_RGBA8888)
        else: # Attempt conversion for other modes
            try:
                rgb_image = pil_image.convert("RGB")
                img_byte_arr = rgb_image.tobytes("raw", "RGB")
                bytes_per_line = rgb_image.size[0] * 3
                qimage = QImage(img_byte_arr, rgb_image.size[0], rgb_image.size[1], bytes_per_line, QImage.Format_RGB888)
            except Exception as convert_err:
                print(f"[ERROR pil_to_qpixmap] Could not convert mode {pil_image.mode} to RGB: {convert_err}")
                return QPixmap()

        if qimage.isNull():
            print("[ERROR pil_to_qpixmap] Failed to create QImage.")
            return QPixmap()
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    except Exception as e:
        print(f"[ERROR pil_to_qpixmap] General error during conversion: {e}")
        traceback.print_exc()
        return QPixmap()

def cv_image_to_qpixmap(cv_img):
    try:
        if cv_img is None:
             print("[ERROR cv_image_to_qpixmap] Input OpenCV image is None.")
             return QPixmap()
        if len(cv_img.shape) == 2: # Grayscale image
            height, width = cv_img.shape
            bytes_per_line = width
            qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(cv_img.shape) == 3: # Color image
            height, width, channel = cv_img.shape
            if channel == 3: # BGR format (standard OpenCV)
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            elif channel == 4: # BGRA format
                 bytes_per_line = 4 * width
                 rgba_image = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                 qimage = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            else:
                 print(f"[ERROR cv_image_to_qpixmap] Unsupported number of channels: {channel}")
                 return QPixmap()
        else:
            print("[ERROR cv_image_to_qpixmap] Unsupported image shape.")
            return QPixmap()

        if qimage.isNull():
            print("[ERROR cv_image_to_qpixmap] Failed to create QImage from OpenCV image.")
            return QPixmap()
        return QPixmap.fromImage(qimage)
    except Exception as e:
        print(f"[ERROR cv_image_to_qpixmap] Error converting OpenCV image: {e}")
        traceback.print_exc()
        return QPixmap()


class BatchModeAppRedesigned(QMainWindow):
    def __init__(self):
        super().__init__()
        self.files = []
        self.output_directory = None
        self.yolo_model = None
        self.lp_model = None
        self.ocr_reader = None
        self.models_loaded = False
        self.ocr_allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # <<< !!! ADJUST THIS PATH IF YOUR LICENSE PLATE MODEL IS DIFFERENT !!! >>>
        self.lp_model_path = "license_plate_detector.pt"
        # ------------------------------------------------------------------------

        self.vehicle_classes_vn = {
            'car': 'car', 'motorcycle': 'motorcycle', 'bus': 'bus', 'truck': 'truck',
            # Add more mappings if your YOLO model detects other classes
        }
        # Define which detected vehicle classes should trigger LP detection
        self.target_vehicle_classes = ['car', 'motorcycle', 'bus', 'truck'] # Adjust as needed

        # Initialize UI elements that need to be accessed later BEFORE initUI
        self.status_label = QLabel("Initializing...") # Initial status
        self.file_list_widget = QListWidget()
        self.add_files_button = QPushButton("Thêm file ảnh")
        self.current_image_label = QLabel("Chọn một file để xem trước")
        self.results_scroll_area = QScrollArea()
        self.results_layout = None # Will be created in initUI
        self.output_dir_label = QLabel("Chưa chọn thư mục")
        self.select_output_dir_button = QPushButton("Chọn thư mục")
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        self.format_combo = QComboBox()
        self.process_detect_button = QPushButton("Nhận dạng Phương tiện & Biển số")
        self.resize_button = QPushButton("Xử lý Kích thước")
        self.format_button = QPushButton("Thay đổi Định dạng")

        self.initUI() # Setup the User Interface

        # Start loading models in a background thread
        self.load_models_thread = threading.Thread(target=self._load_models_background, daemon=True)
        self.load_models_thread.start()

    def _load_models_background(self):
        try:
            self.status_label.setText("Đang tải model nhận diện xe (YOLOv8)...")
            QApplication.processEvents() # Update GUI
            print("[INFO] Bắt đầu tải model nhận diện xe (YOLOv8)...")
            self.yolo_model = YOLO('yolov8n.pt') # Standard object detection model
            # Perform a dummy inference to warm up the model
            _ = self.yolo_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            print("[INFO] Model nhận diện xe đã tải xong.")
            self.status_label.setText("Đã tải model xe. Đang tải model biển số...")
            QApplication.processEvents() # Update GUI

            print(f"[INFO] Bắt đầu tải model nhận diện biển số ({os.path.basename(self.lp_model_path)})...")
            if not os.path.exists(self.lp_model_path):
                error_msg = f"Lỗi: Không tìm thấy file model biển số: {self.lp_model_path}"
                print(f"[ERROR] {error_msg}")
                self.status_label.setText(error_msg)
                self.models_loaded = False
                self.update_button_states() # Disable detection button
                return

            self.lp_model = YOLO(self.lp_model_path) # License plate detection model
            # Warm-up
            _ = self.lp_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            print("[INFO] Model nhận diện biển số đã tải xong.")
            self.status_label.setText("Đã tải model biển số. Đang khởi tạo OCR...")
            QApplication.processEvents() # Update GUI

            print("[INFO] Bắt đầu khởi tạo OCR Reader (EasyOCR)...")
            # Consider adding gpu=True if you have a compatible GPU and installed CUDA drivers
            self.ocr_reader = easyocr.Reader(['en'], gpu=False) # Use 'en' for English-based plates
            print("[INFO] OCR Reader đã sẵn sàng.")

            self.models_loaded = True
            print("[INFO] Tất cả model đã sẵn sàng.")
            self.status_label.setText("Các model đã sẵn sàng. Chọn file và thư mục lưu.")
            self.update_button_states() # Enable detection button if files/dir are selected

        except Exception as e:
            error_msg = f"Lỗi trong quá trình tải model: {e}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            self.models_loaded = False
            self.status_label.setText(f"Lỗi tải model: {e}")
            self.update_button_states() # Keep detection button disabled

    def initUI(self):
        # self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng Phương tiện & Biển số') # Title is in the header now
        self.setWindowFlag(Qt.FramelessWindowHint) # Optional: Remove window borders
        self.showFullScreen() # Make the window full screen

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Main layout for the entire window
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 10) # Margins: left, top, right, bottom
        main_layout.setSpacing(10) # Spacing between major sections

        # --- Header Layout ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5) # Add some bottom margin

        # Back Button
        back_button = QPushButton(" ← Quay lại")
        back_button.setFont(QFont('Segoe UI', 11))
        back_button.setFixedSize(120, 35)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d; color: white; border: none;
                padding: 5px 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #95a5a6; }
            QPushButton:pressed { background-color: #6c7a7d; }
        """)
        back_button.clicked.connect(self.close) # Closes this window
        header_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        # Main Title
        title_label = QLabel('Xử lý Ảnh Hàng Loạt - Nhận Dạng Phương Tiện & Biển Số')
        title_label.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title_label, stretch=1) # Allow title to expand

        # Add a spacer to balance the back button visually
        header_layout.addSpacerItem(QSpacerItem(120, 35, QSizePolicy.Fixed, QSizePolicy.Fixed))

        # --- Add Header to Main Layout ---
        main_layout.addLayout(header_layout)

        # Separator Line below Header
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(separator_line)

        # --- Main Content Area: Two Columns ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15) # Slightly more space between columns
        content_layout.setContentsMargins(0, 10, 0, 0) # Margin above the content area

        # --- Left Column (File List & Add Button) ---
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(10)

        # File List Widget
        self.file_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px;
                background-color: #ecf0f1; font-size: 11pt;
            }
            QListWidget::item:selected { background-color: #3498db; color: white; }
        """)
        left_column_layout.addWidget(self.file_list_widget, stretch=1) # List takes most vertical space

        # Add Files Button
        self.add_files_button.setFont(QFont('Segoe UI', 11))
        self.add_files_button.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; border: none;
                          padding: 10px 15px; border-radius: 5px; min-height: 30px; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.add_files_button.clicked.connect(self.add_files)
        left_column_layout.addWidget(self.add_files_button, alignment=Qt.AlignLeft) # Align button left

        # --- Right Column (Image Preview & Results) ---
        right_column_layout = QVBoxLayout()
        right_column_layout.setSpacing(10)

        # Image Preview Label
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setFrameShape(QFrame.Box)
        self.current_image_label.setFrameShadow(QFrame.Sunken)
        self.current_image_label.setMinimumSize(400, 300) # Minimum preview size
        self.current_image_label.setStyleSheet("""
            QLabel { background-color: #ffffff; border: 1px dashed #bdc3c7;
                     color: #7f8c8d; font-size: 12pt; }
        """)
        right_column_layout.addWidget(self.current_image_label, stretch=1) # Preview takes available space

        # Results Scroll Area
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.StyledPanel) # Use StyledPanel for better theme integration
        # self.results_scroll_area.setFrameShadow(QFrame.Sunken) # Shadow can make it look too deep
        self.results_scroll_area.setMinimumHeight(150) # Min height for results
        self.results_scroll_area.setMaximumHeight(250) # Max height to prevent it taking too much space
        self.results_scroll_area.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px; background-color: #ecf0f1;")

        # Container widget and layout for results inside the scroll area
        results_container_widget = QWidget()
        self.results_layout = QVBoxLayout(results_container_widget)
        self.results_layout.setAlignment(Qt.AlignTop) # Add results from the top
        self.results_layout.setSpacing(5)
        self.clear_results() # Initialize with placeholder
        self.results_scroll_area.setWidget(results_container_widget)
        right_column_layout.addWidget(self.results_scroll_area, stretch=0) # Results area has fixed height preference

        # Add columns to the content layout
        content_layout.addLayout(left_column_layout, stretch=1) # Left column takes 1 part of space
        content_layout.addLayout(right_column_layout, stretch=2) # Right column takes 2 parts

        # --- Add Content Area to Main Layout ---
        main_layout.addLayout(content_layout, stretch=1) # Content area stretches vertically

        # --- Controls Area (Below Content) ---
        controls_main_layout = QVBoxLayout()
        controls_main_layout.setSpacing(10)
        controls_main_layout.setContentsMargins(0, 10, 0, 0) # Margin above controls

        # -- Row 1: Output Directory Selection --
        output_dir_group = QGroupBox("Thư mục lưu trữ kết quả")
        output_dir_group.setFont(QFont('Segoe UI', 10))
        output_dir_group_layout = QHBoxLayout(output_dir_group)
        output_dir_group_layout.setSpacing(10)

        self.output_dir_label.setStyleSheet("font-style: italic; color: #555; padding: 5px;")
        self.output_dir_label.setWordWrap(True) # Allow text wrapping

        self.select_output_dir_button.setFont(QFont('Segoe UI', 10))
        self.select_output_dir_button.setMinimumWidth(100)
        self.select_output_dir_button.setStyleSheet("""
             QPushButton { background-color: #9b59b6; color: white; border: none;
                           padding: 8px 12px; border-radius: 5px; }
             QPushButton:hover { background-color: #8e44ad; }
        """)
        self.select_output_dir_button.clicked.connect(self.select_output_directory)

        output_dir_group_layout.addWidget(self.output_dir_label, stretch=1) # Label takes available space
        output_dir_group_layout.addWidget(self.select_output_dir_button)
        controls_main_layout.addWidget(output_dir_group) # Add the group box to the controls layout

        # -- Row 2: Options and Process Buttons --
        options_buttons_layout = QHBoxLayout()
        options_buttons_layout.setSpacing(20) # Space between groups and button section

        # Resize Options Group
        resize_group = QGroupBox("Thay đổi kích thước")
        resize_group.setFont(QFont('Segoe UI', 10))
        resize_layout = QFormLayout(resize_group)
        resize_layout.setSpacing(8)
        self.width_spin.setRange(1, 10000); self.width_spin.setValue(800); self.width_spin.setSuffix(" px"); self.width_spin.setMinimumWidth(80)
        self.height_spin.setRange(1, 10000); self.height_spin.setValue(600); self.height_spin.setSuffix(" px"); self.height_spin.setMinimumWidth(80)
        resize_layout.addRow("Chiều rộng:", self.width_spin)
        resize_layout.addRow("Chiều cao:", self.height_spin)
        options_buttons_layout.addWidget(resize_group)

        # Format Change Options Group
        format_group = QGroupBox("Thay đổi định dạng")
        format_group.setFont(QFont('Segoe UI', 10))
        format_layout = QVBoxLayout(format_group) # Use QVBoxLayout for simple vertical arrangement
        format_layout.setSpacing(5)
        format_label = QLabel("Chọn định dạng lưu mới:")
        self.format_combo.addItems(["PNG", "JPG", "BMP", "GIF", "TIFF"])
        self.format_combo.setFont(QFont('Segoe UI', 10))
        self.format_combo.setMinimumWidth(100)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch(1) # Push widgets to the top if needed
        options_buttons_layout.addWidget(format_group)

        options_buttons_layout.addStretch(1) # Push options left, buttons right

        # Processing Buttons Section (Vertical Layout)
        process_buttons_layout = QVBoxLayout()
        process_buttons_layout.setSpacing(8)

        # Detect Button
        self.process_detect_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.process_detect_button.setMinimumSize(250, 40)
        self.process_detect_button.setStyleSheet("""
            QPushButton { background-color: #16a085; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #1abc9c; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """)
        self.process_detect_button.clicked.connect(self.process_vehicle_and_lp_detection)
        self.process_detect_button.setEnabled(False) # Disabled initially
        process_buttons_layout.addWidget(self.process_detect_button)

        # Resize Button
        self.resize_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.resize_button.setMinimumSize(250, 40)
        self.resize_button.setStyleSheet("""
            QPushButton { background-color: #e67e22; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
        """)
        self.resize_button.clicked.connect(self.process_resize)
        self.resize_button.setEnabled(False) # Disabled initially
        process_buttons_layout.addWidget(self.resize_button)

        # Format Button
        self.format_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.format_button.setMinimumSize(250, 40)
        self.format_button.setStyleSheet("""
             QPushButton { background-color: #2980b9; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
             QPushButton:hover { background-color: #3498db; }
             QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
         """)
        # Original color was #9b59b6 (purple), changed to blue for differentiation
        # self.format_button.setStyleSheet("""
        #     QPushButton { background-color: #9b59b6; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
        #     QPushButton:hover { background-color: #8e44ad; }
        #     QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
        # """)
        self.format_button.clicked.connect(self.process_format_change)
        self.format_button.setEnabled(False) # Disabled initially
        process_buttons_layout.addWidget(self.format_button)

        options_buttons_layout.addLayout(process_buttons_layout) # Add button column to the row
        controls_main_layout.addLayout(options_buttons_layout) # Add the options/buttons row

        # --- Add Controls Area to Main Layout ---
        main_layout.addLayout(controls_main_layout, stretch=0) # Controls area doesn't stretch

        # --- Status Bar (Defined earlier, add it last to main layout) ---
        self.status_label.setStyleSheet("padding: 5px; color: #e67e22; border-top: 1px solid #ddd; font-size: 9pt;")
        main_layout.addWidget(self.status_label)

        # --- Connect Signals and Apply Global Styles ---
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)

        # General Stylesheet for the window and common widgets
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f8f9fa; font-family: Segoe UI; }
            QGroupBox {
                font-weight: bold; border: 1px solid #ccc; border-radius: 5px;
                margin-top: 10px; /* Space for the title */
                padding: 15px 10px 10px 10px; /* top, right, bottom, left - Increase top padding */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px; /* Padding around the title text */
                left: 10px; /* Position from the left edge */
                color: #2c3e50;
                background-color: #f8f9fa; /* Match window background */
            }
            QSpinBox, QComboBox {
                padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px;
                min-height: 25px; background-color: white; /* Ensure visible background */
            }
            QLabel { color: #2c3e50; } /* Default label color */
            QScrollArea { background-color: transparent; border: none; } /* Style scroll area if needed */
        """)

        # Initial button state update based on current conditions (no files, no dir)
        self.update_button_states()


    # --- Event Handlers and Processing Logic ---

    def add_files(self):
        image_filter = "Ảnh (*.jpg *.jpeg *.png *.bmp *.gif *.tiff);;Tất cả file (*.*)"
        # Use the directory of the last added file or the user's home directory as starting point
        start_dir = os.path.dirname(self.files[-1]) if self.files else os.path.expanduser("~")
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", start_dir, image_filter)

        if files:
            added_count = 0
            newly_added_files = []
            for file in files:
                # Check if the file (case-insensitive on Windows) is already in the list
                if not any(os.path.normcase(file) == os.path.normcase(existing_file) for existing_file in self.files):
                    self.files.append(file)
                    self.file_list_widget.addItem(os.path.basename(file))
                    newly_added_files.append(os.path.basename(file))
                    added_count += 1

            if added_count > 0:
                status_msg = f"Đã thêm {added_count} file ảnh mới."
                if not self.models_loaded: status_msg += " (Models đang tải...)"
                self.status_label.setText(status_msg)
                # Optionally select the first newly added item
                # items = self.file_list_widget.findItems(newly_added_files[0], Qt.MatchExactly)
                # if items: self.file_list_widget.setCurrentItem(items[0])
            else:
                status_msg = "Các file đã chọn đã có trong danh sách hoặc không hợp lệ."
                if not self.models_loaded: status_msg += " (Models đang tải...)"
                self.status_label.setText(status_msg)

            self.update_button_states() # Update buttons after adding files
        else:
            status_msg = "Không có file nào được chọn."
            if not self.models_loaded: status_msg += " (Models đang tải...)"
            self.status_label.setText(status_msg)

    def select_output_directory(self):
        # Start from the current output directory or user's home
        start_dir = self.output_directory if self.output_directory else os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả", start_dir)
        if directory:
            self.output_directory = directory
            # Display a shorter path if it's too long? Optional.
            display_path = directory
            if len(display_path) > 60: # Example length limit
                 display_path = "..." + display_path[-57:]
            self.output_dir_label.setText(f"{display_path}")
            self.output_dir_label.setStyleSheet("font-style: normal; color: #27ae60; padding: 5px;") # Green color for selected
            self.output_dir_label.setToolTip(directory) # Show full path on hover

            status_msg = f"Thư mục lưu kết quả: {directory}"
            if not self.models_loaded: status_msg += " (Models đang tải...)"
            self.status_label.setText(status_msg)
            self.update_button_states() # Update buttons after selecting directory
        # No need for an else clause, if the user cancels, nothing changes

    def display_selected_image(self, current_item, previous_item):
        if current_item:
            file_name = current_item.text()
            # Find the full path corresponding to the selected base name
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)

            if full_path and os.path.exists(full_path):
                try:
                    # Read image using OpenCV (handles various formats robustly)
                    img_cv = cv2.imread(full_path)
                    if img_cv is None:
                        raise ValueError(f"OpenCV không thể đọc file ảnh: {file_name}")

                    # Convert OpenCV image to QPixmap
                    pixmap = cv_image_to_qpixmap(img_cv)

                    if not pixmap.isNull():
                        # Store original pixmap if needed for other operations (optional)
                        # self.current_image_label._original_pixmap = pixmap
                        # Scale pixmap to fit the label while keeping aspect ratio
                        scaled_pixmap = pixmap.scaled(self.current_image_label.size(),
                                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.current_image_label.setPixmap(scaled_pixmap)
                        status_msg = f"Xem trước: {file_name}"
                        if not self.models_loaded: status_msg += " (Models đang tải...)"
                        self.status_label.setText(status_msg)
                    else:
                        self.current_image_label.clear()
                        self.current_image_label.setText(f"Lỗi chuyển đổi ảnh:\n{file_name}")
                        self.status_label.setText(f"Lỗi xem trước (không thể chuyển đổi): {file_name}")
                        # if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
                except Exception as e:
                    print(f"[ERROR display_selected_image] Error opening/processing image {file_name}: {e}")
                    traceback.print_exc()
                    self.current_image_label.clear()
                    self.current_image_label.setText(f"Lỗi khi mở ảnh:\n{file_name}\n({e})")
                    self.status_label.setText(f"Lỗi xem trước (không thể mở): {file_name}")
                    # if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
            else:
                self.current_image_label.clear()
                self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
                # if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
        else:
            # No item selected
            self.current_image_label.clear()
            self.current_image_label.setText("Chọn một file từ danh sách để xem trước")
            # if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
            status_msg = "Sẵn sàng." if self.models_loaded else "Models đang tải..."
            self.status_label.setText(status_msg)

    def _prepare_processing(self, operation_name):
        """Checks prerequisites and disables controls before starting a batch process."""
        if not self.files:
            QMessageBox.warning(self, "Thiếu file", "Vui lòng thêm ít nhất một file ảnh trước khi xử lý.")
            return False
        if not self.output_directory:
            QMessageBox.warning(self, "Thiếu thư mục lưu", "Vui lòng chọn thư mục để lưu kết quả trước khi xử lý.")
            return False
        if not os.path.isdir(self.output_directory):
             QMessageBox.critical(self, "Lỗi Thư Mục Lưu", f"Thư mục lưu đã chọn không hợp lệ hoặc không tồn tại:\n{self.output_directory}")
             return False

        # Specific check for detection task
        if operation_name == "nhận dạng phương tiện và biển số":
            if not self.models_loaded:
                QMessageBox.warning(self, "Models chưa sẵn sàng", "Các model cần thiết đang được tải hoặc đã xảy ra lỗi. Vui lòng chờ hoặc kiểm tra thông báo trạng thái / console.")
                return False
            if not self.yolo_model or not self.lp_model or not self.ocr_reader:
                 QMessageBox.critical(self, "Lỗi Model", "Một hoặc nhiều model cần thiết chưa được tải thành công. Không thể bắt đầu nhận dạng.")
                 return False


        self.status_label.setText(f"Bắt đầu {operation_name}...")
        self.set_controls_enabled(False) # Disable controls during processing
        self.clear_results() # Clear previous results display
        QApplication.processEvents() # Update the UI immediately
        return True

    def _finish_processing(self, operation_name, processed_count, total_count):
        """Re-enables controls and updates status after a batch process."""
        self.status_label.setText(f"Hoàn tất {operation_name} {processed_count}/{total_count} file.")
        self.set_controls_enabled(True) # Re-enable controls
        QApplication.processEvents()

    def update_button_states(self):
        """Enables/disables processing buttons based on conditions."""
        can_process_basic = bool(self.files) and self.output_directory is not None
        self.resize_button.setEnabled(can_process_basic)
        self.format_button.setEnabled(can_process_basic)
        # Detection button requires models to be loaded as well
        self.process_detect_button.setEnabled(can_process_basic and self.models_loaded)

    def set_controls_enabled(self, enabled):
        """Enable/disable all relevant controls."""
        self.add_files_button.setEnabled(enabled)
        self.select_output_dir_button.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled) # Disable list interaction during processing
        self.width_spin.setEnabled(enabled)
        self.height_spin.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        # Update process buttons based on the 'enabled' state AND other conditions
        if enabled:
            self.update_button_states()
        else: # If disabling, disable all process buttons unconditionally
            self.resize_button.setEnabled(False)
            self.format_button.setEnabled(False)
            self.process_detect_button.setEnabled(False)

    def process_resize(self):
        operation_name = "thay đổi kích thước"
        if not self._prepare_processing(operation_name): return

        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        processed_count = 0
        total_files = len(self.files)
        last_processed_pixmap = None

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            # Use the original extension for the resized file
            output_filename = f"{name_part}_resized{ext_part}"
            output_path = os.path.join(self.output_directory, output_filename)

            self.status_label.setText(f"Đang đổi kích thước: {base_name} ({i+1}/{total_files})")
            QApplication.processEvents() # Allow UI updates

            try:
                 # Use OpenCV for reading/writing - often faster and handles more cases
                img = cv2.imread(file_path)
                if img is None:
                    raise IOError(f"Không thể đọc file: {base_name}")

                # Display original image before processing
                original_pixmap = cv_image_to_qpixmap(img)
                if not original_pixmap.isNull():
                   self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                   QApplication.processEvents()

                # Perform resizing using OpenCV's INTER_AREA for shrinking, INTER_LANCZOS4 for enlarging
                interpolation = cv2.INTER_AREA if target_width < img.shape[1] or target_height < img.shape[0] else cv2.INTER_LANCZOS4
                resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)

                # Save the resized image
                success = cv2.imwrite(output_path, resized_img)
                if not success:
                     raise IOError(f"Không thể lưu file: {output_filename}")

                # Add result to the results view
                result_pixmap = cv_image_to_qpixmap(resized_img)
                if not result_pixmap.isNull():
                    self.add_result_item(result_pixmap, base_name, f"Đã đổi kích thước thành {target_width}x{target_height}", output_path)
                    last_processed_pixmap = result_pixmap # Keep track of the last successful one
                    processed_count += 1
                else:
                    self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau resize", file_path)

            except Exception as e:
                error_msg = f"LỖI resize '{base_name}': {e}"
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                self.add_result_item(None, base_name, error_msg, file_path) # Add error to results
                # Continue to the next file

        # Display the last successfully processed image after loop
        if last_processed_pixmap and not last_processed_pixmap.isNull():
             self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._finish_processing(operation_name, processed_count, total_files)

    def process_format_change(self):
        operation_name = "thay đổi định dạng"
        if not self._prepare_processing(operation_name): return

        target_format = self.format_combo.currentText().lower() # e.g., "png", "jpg"
        # Ensure correct extension mapping (e.g., jpg vs jpeg)
        if target_format == 'jpg': target_format_ext = '.jpeg'
        else: target_format_ext = '.' + target_format

        processed_count = 0
        total_files = len(self.files)
        last_processed_pixmap = None

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, _ = os.path.splitext(base_name)
            output_filename = f"{name_part}_converted{target_format_ext}"
            output_path = os.path.join(self.output_directory, output_filename)

            self.status_label.setText(f"Đang đổi định dạng: {base_name} ({i+1}/{total_files})")
            QApplication.processEvents()

            try:
                # Use PIL for robust format conversion
                with Image.open(file_path) as img:
                    # Display original
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull():
                       self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                       QApplication.processEvents()

                    # Convert to RGB if saving as JPG, as JPG doesn't support transparency well
                    save_img = img
                    if target_format == 'jpg' and img.mode in ('RGBA', 'P'):
                        save_img = img.convert('RGB')

                    # Save in the new format
                    save_img.save(output_path, format=target_format.upper()) # PIL uses upper case format names

                    # Read back the saved image for preview (optional but good for verification)
                    with Image.open(output_path) as saved_img_preview:
                         result_pixmap = pil_to_qpixmap(saved_img_preview)

                    if not result_pixmap.isNull():
                        self.add_result_item(result_pixmap, base_name, f"Đã đổi định dạng sang {target_format.upper()}", output_path)
                        last_processed_pixmap = result_pixmap
                        processed_count += 1
                    else:
                         self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau khi đổi định dạng", file_path)

            except Exception as e:
                error_msg = f"LỖI đổi định dạng '{base_name}': {e}"
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                self.add_result_item(None, base_name, error_msg, file_path)
                # Continue to the next file

        if last_processed_pixmap and not last_processed_pixmap.isNull():
             self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._finish_processing(operation_name, processed_count, total_files)


    def process_vehicle_and_lp_detection(self):
        operation_name = "nhận dạng phương tiện và biển số"
        if not self._prepare_processing(operation_name): return

        processed_count = 0
        total_files = len(self.files)
        last_processed_pixmap = None
        detection_results = {} # Store results per file {filename: [result_text]}

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            self.status_label.setText(f"Đang xử lý: {base_name} ({i+1}/{total_files})")
            QApplication.processEvents()

            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None:
                    raise IOError(f"Không thể đọc file: {base_name}")

                # Display original image
                original_pixmap = cv_image_to_qpixmap(img_cv)
                if not original_pixmap.isNull():
                   self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                   QApplication.processEvents()

                # --- 1. Vehicle Detection (YOLOv8) ---
                vehicle_results = self.yolo_model(img_cv, verbose=False)[0] # Get results for the first image
                img_with_boxes = img_cv.copy() # Draw on a copy
                found_vehicles = []
                detection_results[base_name] = [] # Initialize results list for this file

                for result in vehicle_results.boxes.data.tolist(): # Iterate through detected boxes
                    x1, y1, x2, y2, score, class_id = result
                    class_name = self.yolo_model.names[int(class_id)]

                    # Check if the detected class is one we care about
                    if class_name in self.target_vehicle_classes:
                        vehicle_label = self.vehicle_classes_vn.get(class_name, class_name) # Get Vietnamese name if available
                        label_text = f"{vehicle_label}: {score:.2f}"
                        print(f"[DEBUG] Detected {label_text} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                        found_vehicles.append({
                             'box': [int(x1), int(y1), int(x2), int(y2)],
                             'label': vehicle_label,
                             'score': score
                        })

                        # Draw bounding box for the vehicle
                        cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(img_with_boxes, label_text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # --- 2. License Plate Detection (within the vehicle bounding box) ---
                        # Crop the detected vehicle region
                        vehicle_crop = img_cv[int(y1):int(y2), int(x1):int(x2)]
                        if vehicle_crop.size == 0: continue # Skip if crop is invalid

                        lp_results = self.lp_model(vehicle_crop, verbose=False)[0]

                        for lp_result in lp_results.boxes.data.tolist():
                            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = lp_result
                            # LP coordinates are relative to the vehicle crop, convert to absolute
                            abs_lp_x1 = int(x1 + lp_x1)
                            abs_lp_y1 = int(y1 + lp_y1)
                            abs_lp_x2 = int(x1 + lp_x2)
                            abs_lp_y2 = int(y1 + lp_y2)

                            print(f"[DEBUG] Detected LP for {vehicle_label} at [{abs_lp_x1}, {abs_lp_y1}, {abs_lp_x2}, {abs_lp_y2}] score {lp_score:.2f}")
                            # Draw bounding box for the license plate
                            cv2.rectangle(img_with_boxes, (abs_lp_x1, abs_lp_y1), (abs_lp_x2, abs_lp_y2), (255, 0, 0), 2) # Blue for LP

                            # --- 3. OCR on License Plate ---
                            # Crop the license plate region from the *original* image using absolute coordinates
                            lp_crop = img_cv[abs_lp_y1:abs_lp_y2, abs_lp_x1:abs_lp_x2]
                            if lp_crop.size == 0: continue

                            # Optional Preprocessing for OCR (experiment with these)
                            # gray_lp = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                            # _, thresh_lp = cv2.threshold(gray_lp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            # processed_lp_crop = thresh_lp # Or gray_lp

                            # Use EasyOCR
                            ocr_results = self.ocr_reader.readtext(lp_crop, allowlist=self.ocr_allowed_chars, detail=0, paragraph=False) # Get only text, combine lines

                            ocr_text = "".join(ocr_results).upper().replace(" ", "") # Combine results, make upper, remove spaces
                            lp_text = f"Biển số: {ocr_text}" if ocr_text else "Biển số: [Không đọc được]"

                            print(f"[DEBUG] OCR Result for LP: {ocr_text}")
                            # Add LP text near its box
                            cv2.putText(img_with_boxes, lp_text, (abs_lp_x1, abs_lp_y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red text for OCR

                            # Store result text
                            detection_results[base_name].append(f"{vehicle_label} - {lp_text}")


                # Save the image with bounding boxes
                output_filename_boxes = f"{name_part}_detected{ext_part}"
                output_path_boxes = os.path.join(self.output_directory, output_filename_boxes)
                success = cv2.imwrite(output_path_boxes, img_with_boxes)
                if not success:
                    print(f"[WARNING] Không thể lưu ảnh với bounding box: {output_filename_boxes}")
                    output_path_to_show = file_path # Show original if save failed
                else:
                    output_path_to_show = output_path_boxes # Show the image with boxes

                # Add results to the display
                result_summary = ", ".join(detection_results[base_name]) if detection_results[base_name] else "Không phát hiện phương tiện/biển số mục tiêu."
                result_pixmap_boxes = cv_image_to_qpixmap(img_with_boxes)
                if not result_pixmap_boxes.isNull():
                    self.add_result_item(result_pixmap_boxes, base_name, result_summary, output_path_to_show)
                    last_processed_pixmap = result_pixmap_boxes
                else:
                    self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau nhận dạng", file_path)

                processed_count += 1

            except Exception as e:
                error_msg = f"LỖI xử lý '{base_name}': {e}"
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                self.add_result_item(None, base_name, error_msg, file_path)
                # Continue to the next file

        if last_processed_pixmap and not last_processed_pixmap.isNull():
             self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._finish_processing(operation_name, processed_count, total_files)


    def add_result_item(self, pixmap, original_filename, result_text, output_path):
        """Adds an entry to the results scroll area."""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 2, 5, 2) # Compact margins
        item_layout.setSpacing(10)

        # Thumbnail Label
        thumb_label = QLabel()
        thumb_size = 64 # Size for the thumbnail
        if pixmap and not pixmap.isNull():
            scaled_thumb = pixmap.scaled(QSize(thumb_size, thumb_size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb_label.setPixmap(scaled_thumb)
        else:
            thumb_label.setFixedSize(thumb_size, thumb_size)
            thumb_label.setText("Lỗi\nẢnh")
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setStyleSheet("border: 1px solid red; color: red;")
        thumb_label.setFixedSize(thumb_size, thumb_size) # Ensure consistent size

        # Text Info Layout (Vertical)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        filename_label = QLabel(f"<b>{original_filename}</b>")
        result_label = QLabel(result_text)
        result_label.setWordWrap(True) # Allow wrapping for long results
        path_label = QLabel(f"<i>Lưu tại: {os.path.basename(output_path)}</i>")
        path_label.setStyleSheet("color: #555; font-size: 8pt;")
        path_label.setToolTip(output_path) # Show full path on hover

        text_layout.addWidget(filename_label)
        text_layout.addWidget(result_label)
        text_layout.addWidget(path_label)
        text_layout.addStretch() # Push text up

        item_layout.addWidget(thumb_label, alignment=Qt.AlignTop)
        item_layout.addLayout(text_layout, stretch=1) # Text takes remaining space

        # Add the item widget to the main results layout
        self.results_layout.addWidget(item_widget)

        # Add a separator line between items (optional)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #ddd;")
        self.results_layout.addWidget(line)


    def clear_results(self):
        """Clears the results display area."""
        # Remove all widgets from the results layout
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else: # Handle layouts or spacers if any were added directly
                 layout_item = item.layout()
                 if layout_item is not None:
                      # Recursively clear layouts if needed, or just delete top-level
                      pass # Basic clearing assumes only widgets are added directly
        # Add a placeholder message
        placeholder_label = QLabel("Kết quả xử lý sẽ hiển thị ở đây.")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        self.results_layout.addWidget(placeholder_label)

    def resizeEvent(self, event):
        """Handle window resize events to rescale the displayed image."""
        super().resizeEvent(event)
        # Rescale the image currently shown in current_image_label
        # Check if the label currently holds a pixmap (not text like 'Select a file')
        if self.current_image_label.pixmap() and not self.current_image_label.pixmap().isNull():
             # We need the original pixmap to scale correctly, not the already scaled one
             # If you stored the original:
             # if hasattr(self.current_image_label, '_original_pixmap'):
             #     scaled_pixmap = self.current_image_label._original_pixmap.scaled(
             #                                     self.current_image_label.size(),
             #                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
             #     self.current_image_label.setPixmap(scaled_pixmap)
             # else:
             #     # Fallback: rescale the current pixmap (might degrade quality over multiple resizes)
                 current_pixmap = self.current_image_label.pixmap()
                 scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.current_image_label.setPixmap(scaled_pixmap)


    def keyPressEvent(self, event):
        """Handle key presses, e.g., closing with Escape."""
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Confirm before closing if models are loading or processing is happening."""
        # Add confirmation dialog if needed, e.g., if processing is active
        # For now, just accept the close event
        print("[INFO] Closing Batch Mode window.")
        event.accept()


# --- Main Execution ---
if __name__ == '__main__':
    # Set High DPI scaling for better display on modern screens
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Optional: Apply a style
    # app.setStyle('Fusion')

    # Check if required model files exist before launching
    lp_model_file = "license_plate_detector.pt" # Match the one used in the class
    yolo_model_file = "yolov8n.pt"
    if not os.path.exists(lp_model_file) or not os.path.exists(yolo_model_file):
         missing = []
         if not os.path.exists(lp_model_file): missing.append(lp_model_file)
         if not os.path.exists(yolo_model_file): missing.append(yolo_model_file)
         QMessageBox.critical(None, "Lỗi Thiếu Model",
                              f"Không tìm thấy file model cần thiết:\n"
                              f"{', '.join(missing)}\n"
                              f"Vui lòng đặt các file model vào cùng thư mục với script hoặc cung cấp đường dẫn đúng.\n"
                              "Chương trình sẽ thoát.")
         sys.exit(1) # Exit if core models are missing


    # --- This seems to be the entry point from main.py ---
    # It directly creates and shows the BatchModeAppRedesigned window
    window = BatchModeAppRedesigned()
    # window.show() # showFullScreen is called in initUI

    sys.exit(app.exec_())
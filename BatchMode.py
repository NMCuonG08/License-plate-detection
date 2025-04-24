# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame, QMessageBox, QComboBox, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, QTimer # QTimer might not be used now
from PyQt5.QtGui import QFont, QPixmap, QImage
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO # For vehicle detection
import torch # <<< Added for YOLOv5 models
import threading
import traceback

# --- Helper functions (pil_to_qpixmap, cv_image_to_qpixmap) ---
# (Keep these functions as they were in the previous corrected version)
def pil_to_qpixmap(pil_image):
    # ... (previous implementation) ...
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
            rgb_image = pil_image.convert("RGB")
            img_byte_arr = rgb_image.tobytes("raw", "RGB")
            bytes_per_line = rgb_image.size[0] * 3
            qimage = QImage(img_byte_arr, rgb_image.size[0], rgb_image.size[1], bytes_per_line, QImage.Format_RGB888)
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
    # ... (previous implementation) ...
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

# --- Main Application Class ---
class BatchModeAppRedesigned(QMainWindow):
    def __init__(self):
        super().__init__()
        self.files = []
        self.output_directory = None

        # Model Attributes
        self.yolo_vehicle_model = None # YOLOv8 for vehicle detection
        self.yolo_lp_detect_model = None # YOLOv5 for LP detection
        self.yolo_lp_ocr_model = None # YOLOv5 for LP OCR

        # Model Paths (Adjust these paths if your models are elsewhere)
        self.vehicle_model_path = "yolov8n.pt"
        self.lp_detect_model_path = "model/LP_detector.pt" 
        self.lp_ocr_model_path = "model/LP_ocr.pt"      

        self.models_loaded = False
        self.yolo_ocr_conf_threshold = 0.60 # Confidence for OCR model

        self.vehicle_classes_vn = {
            'car': 'car', 'motorcycle': 'motorcycle', 'bus': 'bus', 'truck': 'truck',
        }
        self.target_vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

        # Initialize UI elements
        self.status_label = QLabel("Initializing...")
        self.file_list_widget = QListWidget()
        self.add_files_button = QPushButton("Thêm file ảnh")
        self.current_image_label = QLabel("Chọn một file để xem trước")
        self.results_scroll_area = QScrollArea()
        self.results_layout = None
        self.output_dir_label = QLabel("Chưa chọn thư mục")
        self.select_output_dir_button = QPushButton("Chọn thư mục")
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        self.format_combo = QComboBox()
        self.process_detect_button = QPushButton("Nhận dạng Phương tiện & Biển số")
        self.resize_button = QPushButton("Xử lý Kích thước")
        self.format_button = QPushButton("Thay đổi Định dạng")

        self.initUI() # Setup the User Interface

        # Start loading models in background
        self.load_models_thread = threading.Thread(target=self._load_models_background, daemon=True)
        self.load_models_thread.start()

    def _load_models_background(self):
        """Loads the three models (Vehicle detect, LP detect, LP OCR)"""
        try:
            # 1. Load Vehicle Detection Model (YOLOv8)
            self.status_label.setText(f"Đang tải model nhận diện xe ({os.path.basename(self.vehicle_model_path)})...")
            QApplication.processEvents()
            print(f"[INFO] Bắt đầu tải model nhận diện xe ({self.vehicle_model_path})...")
            if not os.path.exists(self.vehicle_model_path):
                raise FileNotFoundError(f"Không tìm thấy file model xe: {self.vehicle_model_path}")
            self.yolo_vehicle_model = YOLO(self.vehicle_model_path)
            _ = self.yolo_vehicle_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False) # Warm-up
            print("[INFO] Model nhận diện xe đã tải xong.")
            self.status_label.setText("Đã tải model xe. Đang tải model phát hiện biển số...")
            QApplication.processEvents()

            # 2. Load License Plate Detection Model (YOLOv5)
            self.status_label.setText(f"Đang tải model phát hiện biển số ({os.path.basename(self.lp_detect_model_path)})...")
            QApplication.processEvents()
            print(f"[INFO] Bắt đầu tải model phát hiện biển số ({self.lp_detect_model_path})...")
            if not os.path.exists(self.lp_detect_model_path):
                raise FileNotFoundError(f"Không tìm thấy file model phát hiện biển số: {self.lp_detect_model_path}")
            # Ensure the yolov5 repository is accessible if source='local' is used strictly
            self.yolo_lp_detect_model = torch.hub.load('yolov5', 'custom', path=self.lp_detect_model_path, source='local', force_reload=False) # force_reload=False usually fine after first time
            print("[INFO] Model phát hiện biển số đã tải xong.")
            self.status_label.setText("Đã tải model phát hiện biển số. Đang tải model OCR biển số...")
            QApplication.processEvents()

            # 3. Load License Plate OCR Model (YOLOv5)
            self.status_label.setText(f"Đang tải model OCR biển số ({os.path.basename(self.lp_ocr_model_path)})...")
            QApplication.processEvents()
            print(f"[INFO] Bắt đầu tải model OCR biển số ({self.lp_ocr_model_path})...")
            if not os.path.exists(self.lp_ocr_model_path):
                raise FileNotFoundError(f"Không tìm thấy file model OCR biển số: {self.lp_ocr_model_path}")
            self.yolo_lp_ocr_model = torch.hub.load('yolov5', 'custom', path=self.lp_ocr_model_path, source='local', force_reload=False)
            self.yolo_lp_ocr_model.conf = self.yolo_ocr_conf_threshold # Set confidence threshold
            print(f"[INFO] Model OCR biển số đã tải xong (Conf={self.yolo_lp_ocr_model.conf}).")

            self.models_loaded = True
            print("[INFO] Tất cả model đã sẵn sàng.")
            self.status_label.setText("Các model đã sẵn sàng. Chọn file và thư mục lưu.")
            self.update_button_states()

        except Exception as e:
            error_msg = f"Lỗi trong quá trình tải model: {e}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            # Try to identify which model failed if possible
            if not hasattr(self, 'yolo_vehicle_model') or not self.yolo_vehicle_model:
                error_msg = f"Lỗi tải model Xe: {e}"
            elif not hasattr(self, 'yolo_lp_detect_model') or not self.yolo_lp_detect_model:
                error_msg = f"Lỗi tải model phát hiện LP: {e}"
            elif not hasattr(self, 'yolo_lp_ocr_model') or not self.yolo_lp_ocr_model:
                error_msg = f"Lỗi tải model OCR LP: {e}"
            else: # General error
                error_msg = f"Lỗi tải model: {e}"

            self.models_loaded = False
            self.status_label.setText(error_msg)
            self.update_button_states() # Keep detection button disabled

    def initUI(self):
        # --- This section remains largely the same as the previous corrected version ---
        # It sets up the window, header, content columns (file list, preview, results),
        # and controls (output dir, resize, format, process buttons).
        # No changes needed here unless you want to adjust the layout itself.

        # self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng Phương tiện & Biển số')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(10)

        # --- Header ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5)
        back_button = QPushButton(" ← Quay lại")
        back_button.setFont(QFont('Segoe UI', 11))
        back_button.setFixedSize(120, 35)
        back_button.setStyleSheet("""
            QPushButton { background-color: #7f8c8d; color: white; border: none; padding: 5px 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #95a5a6; } QPushButton:pressed { background-color: #6c7a7d; }
        """)
        back_button.clicked.connect(self.close)
        header_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        title_label = QLabel('Xử lý Ảnh Hàng Loạt - Nhận Dạng Phương Tiện & Biển Số')
        title_label.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title_label, stretch=1)
        header_layout.addSpacerItem(QSpacerItem(120, 35, QSizePolicy.Fixed, QSizePolicy.Fixed))
        main_layout.addLayout(header_layout)

        # --- Separator ---
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(separator_line)

        # --- Content Area ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(0, 10, 0, 0)

        # Left Column
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(10)
        self.file_list_widget.setStyleSheet("""
            QListWidget { border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px; background-color: #ecf0f1; font-size: 11pt; }
            QListWidget::item:selected { background-color: #3498db; color: white; }
        """)
        left_column_layout.addWidget(self.file_list_widget, stretch=1)
        self.add_files_button.setFont(QFont('Segoe UI', 11))
        self.add_files_button.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; border: none; padding: 10px 15px; border-radius: 5px; min-height: 30px; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.add_files_button.clicked.connect(self.add_files)
        left_column_layout.addWidget(self.add_files_button, alignment=Qt.AlignLeft)

        # Right Column
        right_column_layout = QVBoxLayout()
        right_column_layout.setSpacing(10)
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setFrameShape(QFrame.Box)
        self.current_image_label.setFrameShadow(QFrame.Sunken)
        self.current_image_label.setMinimumSize(400, 300)
        self.current_image_label.setStyleSheet("""
            QLabel { background-color: #ffffff; border: 1px dashed #bdc3c7; color: #7f8c8d; font-size: 12pt; }
        """)
        right_column_layout.addWidget(self.current_image_label, stretch=1)
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.StyledPanel)
        self.results_scroll_area.setMinimumHeight(150)
        self.results_scroll_area.setMaximumHeight(250)
        self.results_scroll_area.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px; background-color: #ecf0f1;")
        results_container_widget = QWidget()
        self.results_layout = QVBoxLayout(results_container_widget)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setSpacing(5)
        self.clear_results() # Initialize placeholder
        self.results_scroll_area.setWidget(results_container_widget)
        right_column_layout.addWidget(self.results_scroll_area, stretch=0)

        content_layout.addLayout(left_column_layout, stretch=1)
        content_layout.addLayout(right_column_layout, stretch=2)
        main_layout.addLayout(content_layout, stretch=1)

        # --- Controls Area ---
        controls_main_layout = QVBoxLayout()
        controls_main_layout.setSpacing(10)
        controls_main_layout.setContentsMargins(0, 10, 0, 0)

        # Output Directory Group
        output_dir_group = QGroupBox("Thư mục lưu trữ kết quả")
        output_dir_group.setFont(QFont('Segoe UI', 10))
        output_dir_group_layout = QHBoxLayout(output_dir_group)
        output_dir_group_layout.setSpacing(10)
        self.output_dir_label.setStyleSheet("font-style: italic; color: #555; padding: 5px;")
        self.output_dir_label.setWordWrap(True)
        self.select_output_dir_button.setFont(QFont('Segoe UI', 10))
        self.select_output_dir_button.setMinimumWidth(100)
        self.select_output_dir_button.setStyleSheet("""
             QPushButton { background-color: #9b59b6; color: white; border: none; padding: 8px 12px; border-radius: 5px; }
             QPushButton:hover { background-color: #8e44ad; }
        """)
        self.select_output_dir_button.clicked.connect(self.select_output_directory)
        output_dir_group_layout.addWidget(self.output_dir_label, stretch=1)
        output_dir_group_layout.addWidget(self.select_output_dir_button)
        controls_main_layout.addWidget(output_dir_group)

        # Options and Buttons Row
        options_buttons_layout = QHBoxLayout()
        options_buttons_layout.setSpacing(20)

        # Resize Group
        resize_group = QGroupBox("Thay đổi kích thước")
        resize_group.setFont(QFont('Segoe UI', 10))
        resize_layout = QFormLayout(resize_group)
        resize_layout.setSpacing(8)
        self.width_spin.setRange(1, 10000); self.width_spin.setValue(800); self.width_spin.setSuffix(" px"); self.width_spin.setMinimumWidth(80)
        self.height_spin.setRange(1, 10000); self.height_spin.setValue(600); self.height_spin.setSuffix(" px"); self.height_spin.setMinimumWidth(80)
        resize_layout.addRow("Chiều rộng:", self.width_spin)
        resize_layout.addRow("Chiều cao:", self.height_spin)
        options_buttons_layout.addWidget(resize_group)

        # Format Group
        format_group = QGroupBox("Thay đổi định dạng")
        format_group.setFont(QFont('Segoe UI', 10))
        format_layout = QVBoxLayout(format_group)
        format_layout.setSpacing(5)
        format_label = QLabel("Chọn định dạng lưu mới:")
        self.format_combo.addItems(["PNG", "JPG", "BMP", "GIF", "TIFF"])
        self.format_combo.setFont(QFont('Segoe UI', 10))
        self.format_combo.setMinimumWidth(100)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch(1)
        options_buttons_layout.addWidget(format_group)
        options_buttons_layout.addStretch(1)

        # Processing Buttons
        process_buttons_layout = QVBoxLayout()
        process_buttons_layout.setSpacing(8)
        self.process_detect_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.process_detect_button.setMinimumSize(250, 40)
        self.process_detect_button.setStyleSheet("""
            QPushButton { background-color: #16a085; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #1abc9c; } QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """)
        self.process_detect_button.clicked.connect(self.process_vehicle_and_lp_detection)
        self.process_detect_button.setEnabled(False)
        process_buttons_layout.addWidget(self.process_detect_button)
        self.resize_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.resize_button.setMinimumSize(250, 40)
        self.resize_button.setStyleSheet("""
            QPushButton { background-color: #e67e22; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #d35400; } QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
        """)
        self.resize_button.clicked.connect(self.process_resize)
        self.resize_button.setEnabled(False)
        process_buttons_layout.addWidget(self.resize_button)
        self.format_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.format_button.setMinimumSize(250, 40)
        self.format_button.setStyleSheet("""
             QPushButton { background-color: #2980b9; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
             QPushButton:hover { background-color: #3498db; } QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
         """)
        self.format_button.clicked.connect(self.process_format_change)
        self.format_button.setEnabled(False)
        process_buttons_layout.addWidget(self.format_button)
        options_buttons_layout.addLayout(process_buttons_layout)
        controls_main_layout.addLayout(options_buttons_layout)
        main_layout.addLayout(controls_main_layout, stretch=0)

        # --- Status Bar ---
        self.status_label.setStyleSheet("padding: 5px; color: #e67e22; border-top: 1px solid #ddd; font-size: 9pt;")
        main_layout.addWidget(self.status_label)

        # --- Connect Signals & Styles ---
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f8f9fa; font-family: Segoe UI; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; padding: 15px 10px 10px 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; left: 10px; color: #2c3e50; background-color: #f8f9fa; }
            QSpinBox, QComboBox { padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; min-height: 25px; background-color: white; }
            QLabel { color: #2c3e50; }
            QScrollArea { background-color: transparent; border: none; }
        """)
        self.update_button_states()

    # --- Event Handlers and Processing Logic ---

    # add_files, select_output_directory, display_selected_image,
    # _prepare_processing, _finish_processing, update_button_states,
    # set_controls_enabled, process_resize, process_format_change
    # --- KEEP these methods exactly as they were in the previous corrected version ---
    # --- The only method that needs significant change is process_vehicle_and_lp_detection ---

    # ... (Keep add_files method) ...
    def add_files(self):
        image_filter = "Ảnh (*.jpg *.jpeg *.png *.bmp *.gif *.tiff);;Tất cả file (*.*)"
        start_dir = os.path.dirname(self.files[-1]) if self.files else os.path.expanduser("~")
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", start_dir, image_filter)
        if files:
            added_count = 0
            newly_added_files = []
            for file in files:
                if not any(os.path.normcase(file) == os.path.normcase(existing_file) for existing_file in self.files):
                    self.files.append(file)
                    self.file_list_widget.addItem(os.path.basename(file))
                    newly_added_files.append(os.path.basename(file))
                    added_count += 1
            if added_count > 0:
                status_msg = f"Đã thêm {added_count} file ảnh mới."
                if not self.models_loaded: status_msg += " (Models đang tải...)"
                self.status_label.setText(status_msg)
            else:
                status_msg = "Các file đã chọn đã có trong danh sách hoặc không hợp lệ."
                if not self.models_loaded: status_msg += " (Models đang tải...)"
                self.status_label.setText(status_msg)
            self.update_button_states()
        else:
            status_msg = "Không có file nào được chọn."
            if not self.models_loaded: status_msg += " (Models đang tải...)"
            self.status_label.setText(status_msg)

    # ... (Keep select_output_directory method) ...
    def select_output_directory(self):
        start_dir = self.output_directory if self.output_directory else os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả", start_dir)
        if directory:
            self.output_directory = directory
            display_path = directory
            if len(display_path) > 60:
                 display_path = "..." + display_path[-57:]
            self.output_dir_label.setText(f"{display_path}")
            self.output_dir_label.setStyleSheet("font-style: normal; color: #27ae60; padding: 5px;")
            self.output_dir_label.setToolTip(directory)
            status_msg = f"Thư mục lưu kết quả: {directory}"
            if not self.models_loaded: status_msg += " (Models đang tải...)"
            self.status_label.setText(status_msg)
            self.update_button_states()

    # ... (Keep display_selected_image method) ...
    def display_selected_image(self, current_item, previous_item):
        if current_item:
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)
            if full_path and os.path.exists(full_path):
                try:
                    img_cv = cv2.imread(full_path)
                    if img_cv is None: raise ValueError(f"OpenCV không thể đọc file ảnh: {file_name}")
                    pixmap = cv_image_to_qpixmap(img_cv)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.current_image_label.setPixmap(scaled_pixmap)
                        status_msg = f"Xem trước: {file_name}"
                        if not self.models_loaded: status_msg += " (Models đang tải...)"
                        self.status_label.setText(status_msg)
                    else:
                        self.current_image_label.clear(); self.current_image_label.setText(f"Lỗi chuyển đổi ảnh:\n{file_name}")
                        self.status_label.setText(f"Lỗi xem trước (không thể chuyển đổi): {file_name}")
                except Exception as e:
                    print(f"[ERROR display_selected_image] Error opening/processing image {file_name}: {e}")
                    traceback.print_exc()
                    self.current_image_label.clear(); self.current_image_label.setText(f"Lỗi khi mở ảnh:\n{file_name}\n({e})")
                    self.status_label.setText(f"Lỗi xem trước (không thể mở): {file_name}")
            else:
                self.current_image_label.clear(); self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
        else:
            self.current_image_label.clear(); self.current_image_label.setText("Chọn một file từ danh sách để xem trước")
            status_msg = "Sẵn sàng." if self.models_loaded else "Models đang tải..."
            self.status_label.setText(status_msg)

    # ... (Keep _prepare_processing method, update model checks) ...
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
            # Check if all three models are loaded
            if not self.yolo_vehicle_model or not self.yolo_lp_detect_model or not self.yolo_lp_ocr_model:
                 QMessageBox.critical(self, "Lỗi Model", "Một hoặc nhiều model (Xe, Phát hiện LP, OCR LP) chưa được tải thành công. Không thể bắt đầu nhận dạng.")
                 return False

        self.status_label.setText(f"Bắt đầu {operation_name}...")
        self.set_controls_enabled(False)
        self.clear_results()
        QApplication.processEvents()
        return True

    # ... (Keep _finish_processing method) ...
    def _finish_processing(self, operation_name, processed_count, total_count):
        self.status_label.setText(f"Hoàn tất {operation_name} {processed_count}/{total_count} file.")
        self.set_controls_enabled(True)
        QApplication.processEvents()

    # ... (Keep update_button_states method) ...
    def update_button_states(self):
        can_process_basic = bool(self.files) and self.output_directory is not None
        self.resize_button.setEnabled(can_process_basic)
        self.format_button.setEnabled(can_process_basic)
        self.process_detect_button.setEnabled(can_process_basic and self.models_loaded)

    # ... (Keep set_controls_enabled method) ...
    def set_controls_enabled(self, enabled):
        self.add_files_button.setEnabled(enabled)
        self.select_output_dir_button.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)
        self.width_spin.setEnabled(enabled)
        self.height_spin.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        if enabled:
            self.update_button_states()
        else:
            self.resize_button.setEnabled(False)
            self.format_button.setEnabled(False)
            self.process_detect_button.setEnabled(False)

    # ... (Keep process_resize method) ...
    def process_resize(self):
        operation_name = "thay đổi kích thước"
        if not self._prepare_processing(operation_name): return
        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        processed_count = 0; total_files = len(self.files); last_processed_pixmap = None
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path); name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}_resized{ext_part}"; output_path = os.path.join(self.output_directory, output_filename)
            self.status_label.setText(f"Đang đổi kích thước: {base_name} ({i+1}/{total_files})"); QApplication.processEvents()
            try:
                img = cv2.imread(file_path); assert img is not None, f"Không thể đọc file: {base_name}"
                original_pixmap = cv_image_to_qpixmap(img)
                if not original_pixmap.isNull(): self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); QApplication.processEvents()
                interpolation = cv2.INTER_AREA if target_width < img.shape[1] or target_height < img.shape[0] else cv2.INTER_LANCZOS4
                resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
                success = cv2.imwrite(output_path, resized_img); assert success, f"Không thể lưu file: {output_filename}"
                result_pixmap = cv_image_to_qpixmap(resized_img)
                if not result_pixmap.isNull(): self.add_result_item(result_pixmap, base_name, f"Đã đổi kích thước thành {target_width}x{target_height}", output_path); last_processed_pixmap = result_pixmap; processed_count += 1
                else: self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau resize", file_path)
            except Exception as e: error_msg = f"LỖI resize '{base_name}': {e}"; print(f"[ERROR] {error_msg}"); traceback.print_exc(); self.add_result_item(None, base_name, error_msg, file_path)
        if last_processed_pixmap and not last_processed_pixmap.isNull(): self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._finish_processing(operation_name, processed_count, total_files)

    # ... (Keep process_format_change method) ...
    def process_format_change(self):
        operation_name = "thay đổi định dạng"
        if not self._prepare_processing(operation_name): return
        target_format = self.format_combo.currentText().lower(); target_format_ext = '.' + ('jpeg' if target_format == 'jpg' else target_format)
        processed_count = 0; total_files = len(self.files); last_processed_pixmap = None
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path); name_part, _ = os.path.splitext(base_name)
            output_filename = f"{name_part}_converted{target_format_ext}"; output_path = os.path.join(self.output_directory, output_filename)
            self.status_label.setText(f"Đang đổi định dạng: {base_name} ({i+1}/{total_files})"); QApplication.processEvents()
            try:
                with Image.open(file_path) as img:
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull(): self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); QApplication.processEvents()
                    save_img = img;
                    if target_format == 'jpg' and img.mode in ('RGBA', 'P'): save_img = img.convert('RGB')
                    save_img.save(output_path, format=target_format.upper())
                    with Image.open(output_path) as saved_img_preview: result_pixmap = pil_to_qpixmap(saved_img_preview)
                    if not result_pixmap.isNull(): self.add_result_item(result_pixmap, base_name, f"Đã đổi định dạng sang {target_format.upper()}", output_path); last_processed_pixmap = result_pixmap; processed_count += 1
                    else: self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau khi đổi định dạng", file_path)
            except Exception as e: error_msg = f"LỖI đổi định dạng '{base_name}': {e}"; print(f"[ERROR] {error_msg}"); traceback.print_exc(); self.add_result_item(None, base_name, error_msg, file_path)
        if last_processed_pixmap and not last_processed_pixmap.isNull(): self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._finish_processing(operation_name, processed_count, total_files)


    def process_vehicle_and_lp_detection(self):
        """Detects vehicles, then license plates, then performs OCR using YOLOv5 models."""
        operation_name = "nhận dạng phương tiện và biển số"
        if not self._prepare_processing(operation_name): return

        processed_count = 0
        total_files = len(self.files)
        last_processed_pixmap = None
        detection_results = {} # Store results per file {filename: [result_text]}

        # Define standard inference size for YOLOv5 models (adjust if needed)
        yolo_inference_size = 640

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            self.status_label.setText(f"Đang xử lý: {base_name} ({i+1}/{total_files})")
            QApplication.processEvents()

            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None: raise IOError(f"Không thể đọc file: {base_name}")

                # Display original image
                original_pixmap = cv_image_to_qpixmap(img_cv)
                if not original_pixmap.isNull():
                   self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                   QApplication.processEvents()

                img_height, img_width = img_cv.shape[:2]
                img_with_boxes = img_cv.copy() # Draw on a copy
                detection_results[base_name] = [] # Initialize results list

                # --- 1. Vehicle Detection (YOLOv8) ---
                # Use the already loaded self.yolo_vehicle_model
                vehicle_detect_results = self.yolo_vehicle_model(img_cv, verbose=False)[0] # Ultralytics YOLO results object

                for result in vehicle_detect_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    class_name = self.yolo_vehicle_model.names[int(class_id)]

                    if class_name in self.target_vehicle_classes:
                        vehicle_label = self.vehicle_classes_vn.get(class_name, class_name)
                        label_text = f"{vehicle_label}: {score:.2f}"
                        vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2) # Vehicle coordinates

                        print(f"[DEBUG] Detected {label_text} at [{vx1}, {vy1}, {vx2}, {vy2}]")
                        # Draw vehicle box
                        cv2.rectangle(img_with_boxes, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2) # Green for vehicle
                        cv2.putText(img_with_boxes, label_text, (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # --- 2. License Plate Detection (YOLOv5 LP Detector) ---
                        # Crop the vehicle region for LP detection
                        vehicle_crop = img_cv[vy1:vy2, vx1:vx2]
                        if vehicle_crop.size == 0:
                            print(f"[DEBUG] Vehicle crop for {vehicle_label} is empty, skipping LP detection.")
                            continue

                        # Run LP detection model (YOLOv5)
                        lp_detect_results = self.yolo_lp_detect_model(vehicle_crop, size=yolo_inference_size) # Specify size

                        # Parse YOLOv5 detection results (might need adjustment based on exact output format)
                        # Assuming results.xyxy[0] gives [x1, y1, x2, y2, conf, class_id] relative to crop
                        lp_detections = lp_detect_results.xyxy[0].cpu().numpy() # Detections for the first (only) image in batch

                        if len(lp_detections) > 0:
                            # Find the best LP detection (e.g., highest confidence)
                            # Assuming class 0 is the license plate
                            best_lp = max(lp_detections, key=lambda det: det[4]) # Max confidence
                            lp_x1_rel, lp_y1_rel, lp_x2_rel, lp_y2_rel, lp_conf, _ = best_lp

                            # Convert relative LP coords to absolute coords in the original image
                            abs_lp_x1 = vx1 + int(lp_x1_rel)
                            abs_lp_y1 = vy1 + int(lp_y1_rel)
                            abs_lp_x2 = vx1 + int(lp_x2_rel)
                            abs_lp_y2 = vy1 + int(lp_y2_rel)

                            # Clamp coordinates to image bounds to prevent cropping errors
                            abs_lp_x1 = max(0, abs_lp_x1)
                            abs_lp_y1 = max(0, abs_lp_y1)
                            abs_lp_x2 = min(img_width - 1, abs_lp_x2)
                            abs_lp_y2 = min(img_height - 1, abs_lp_y2)


                            print(f"[DEBUG] Detected LP for {vehicle_label} at [{abs_lp_x1}, {abs_lp_y1}, {abs_lp_x2}, {abs_lp_y2}] score {lp_conf:.2f}")
                            # Draw LP box
                            cv2.rectangle(img_with_boxes, (abs_lp_x1, abs_lp_y1), (abs_lp_x2, abs_lp_y2), (255, 0, 0), 2) # Blue for LP box

                            # --- 3. OCR on License Plate (YOLOv5 OCR Model) ---
                            # Crop the precise LP region from the *original* image
                            lp_crop = img_cv[abs_lp_y1:abs_lp_y2, abs_lp_x1:abs_lp_x2]
                            if lp_crop.size == 0:
                                print(f"[DEBUG] LP crop for {vehicle_label} is empty, skipping OCR.")
                                continue

                            # Run OCR model
                            ocr_results = self.yolo_lp_ocr_model(lp_crop, size=yolo_inference_size)
                            ocr_detections = ocr_results.xyxy[0].cpu().numpy() # Character detections

                            ocr_text = ""
                            if len(ocr_detections) > 0:
                                # Get character info (box center x, character name)
                                characters = []
                                for char_det in ocr_detections:
                                    cx1, cy1, cx2, cy2, char_conf, char_class_id = char_det
                                    char_name = self.yolo_lp_ocr_model.names[int(char_class_id)]
                                    center_x = (cx1 + cx2) / 2
                                    characters.append({'char': char_name, 'center_x': center_x, 'conf': char_conf})
                                    # Optional: Draw char boxes on the LP crop (for debugging) or main image
                                    # abs_char_x1 = abs_lp_x1 + int(cx1) ... etc.
                                    # cv2.rectangle(img_with_boxes, (abs_char_x1, ...), ..., (0,0,255), 1)

                                # Sort characters by horizontal position
                                characters.sort(key=lambda c: c['center_x'])

                                # Concatenate characters
                                ocr_text = "".join([c['char'] for c in characters])
                                print(f"[DEBUG] OCR Result for LP: {ocr_text} (from {len(characters)} chars)")

                            lp_text_display = f"License Plate: {ocr_text}" if ocr_text else "License Plate: [Không đọc được]"

                             # Add LP text near its box
                            cv2.putText(img_with_boxes, lp_text_display, (abs_lp_x1, abs_lp_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red text

                            # Store result text
                            detection_results[base_name].append(f"{vehicle_label} - {lp_text_display}")
                        else:
                             print(f"[DEBUG] No license plate detected for {vehicle_label} by LP detection model.")
                             detection_results[base_name].append(f"{vehicle_label} - [Không tìm thấy biển số]")

                # --- Saving and Displaying Results for the File ---
                # Save the image with bounding boxes
                output_filename_boxes = f"{name_part}_detected{ext_part}"
                output_path_boxes = os.path.join(self.output_directory, output_filename_boxes)
                success = cv2.imwrite(output_path_boxes, img_with_boxes)
                output_path_to_show = output_path_boxes if success else file_path
                if not success: print(f"[WARNING] Không thể lưu ảnh với bounding box: {output_filename_boxes}")

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


    # ... (Keep add_result_item method) ...
    def add_result_item(self, pixmap, original_filename, result_text, output_path):
        item_widget = QWidget(); item_layout = QHBoxLayout(item_widget); item_layout.setContentsMargins(5, 2, 5, 2); item_layout.setSpacing(10)
        thumb_label = QLabel(); thumb_size = 64
        if pixmap and not pixmap.isNull(): thumb_label.setPixmap(pixmap.scaled(QSize(thumb_size, thumb_size), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: thumb_label.setFixedSize(thumb_size, thumb_size); thumb_label.setText("Lỗi\nẢnh"); thumb_label.setAlignment(Qt.AlignCenter); thumb_label.setStyleSheet("border: 1px solid red; color: red;")
        thumb_label.setFixedSize(thumb_size, thumb_size)
        text_layout = QVBoxLayout(); text_layout.setSpacing(1)
        filename_label = QLabel(f"<b>{original_filename}</b>"); result_label = QLabel(result_text); result_label.setWordWrap(True)
        path_label = QLabel(f"<i>Lưu tại: {os.path.basename(output_path)}</i>"); path_label.setStyleSheet("color: #555; font-size: 8pt;"); path_label.setToolTip(output_path)
        text_layout.addWidget(filename_label); text_layout.addWidget(result_label); text_layout.addWidget(path_label); text_layout.addStretch()
        item_layout.addWidget(thumb_label, alignment=Qt.AlignTop); item_layout.addLayout(text_layout, stretch=1)
        self.results_layout.addWidget(item_widget)
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken); line.setStyleSheet("color: #ddd;"); self.results_layout.addWidget(line)

    # ... (Keep clear_results method) ...
    def clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0); widget = item.widget()
            if widget is not None: widget.deleteLater()
            else: layout_item = item.layout(); # Handle layouts if needed
        placeholder_label = QLabel("Kết quả xử lý sẽ hiển thị ở đây."); placeholder_label.setAlignment(Qt.AlignCenter); placeholder_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        self.results_layout.addWidget(placeholder_label)

    # ... (Keep resizeEvent method) ...
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_label.pixmap() and not self.current_image_label.pixmap().isNull():
             current_pixmap = self.current_image_label.pixmap(); scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.current_image_label.setPixmap(scaled_pixmap)

    # ... (Keep keyPressEvent method) ...
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.close()
        else: super().keyPressEvent(event)

    # ... (Keep closeEvent method) ...
    def closeEvent(self, event):
        print("[INFO] Closing Batch Mode window.")
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    # Check if required model files exist before launching
    # <<< UPDATE MODEL PATHS TO CHECK >>>
    vehicle_model_file = "yolov8n.pt"
    lp_detect_model_file = "model/LP_detector.pt"
    lp_ocr_model_file = "model/LP_ocr.pt"
    missing = []
    if not os.path.exists(vehicle_model_file): missing.append(vehicle_model_file)
    if not os.path.exists(lp_detect_model_file): missing.append(lp_detect_model_file)
    if not os.path.exists(lp_ocr_model_file): missing.append(lp_ocr_model_file)

    if missing:
         QMessageBox.critical(None, "Lỗi Thiếu Model",
                              f"Không tìm thấy file model cần thiết:\n"
                              f"{', '.join(missing)}\n"
                              f"Vui lòng đặt các file model vào đúng vị trí (vd: trong thư mục 'model/').\n"
                              "Chương trình sẽ thoát.")
         sys.exit(1)

    window = BatchModeAppRedesigned()
    # window.show() # showFullScreen is called in initUI
    sys.exit(app.exec_())
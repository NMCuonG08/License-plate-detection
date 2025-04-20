# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame, QMessageBox, QComboBox, QSpacerItem, QSizePolicy) # Thêm QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import threading

# --- Các hàm helper (pil_to_qpixmap, cv_image_to_qpixmap) giữ nguyên ---
def pil_to_qpixmap(pil_image):
    # (Code đã có)
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
        else: # Các trường hợp khác
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
        import traceback
        traceback.print_exc()
        return QPixmap()

def cv_image_to_qpixmap(cv_img):
    # (Code đã có)
    try:
        if len(cv_img.shape) == 2: # Ảnh xám
             height, width = cv_img.shape
             bytes_per_line = width
             qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(cv_img.shape) == 3: # Ảnh màu
             height, width, channel = cv_img.shape
             bytes_per_line = 3 * width
             rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
             qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
             print("[ERROR cv_image_to_qpixmap] Unsupported image shape.")
             return QPixmap()

        if qimage.isNull():
             print("[ERROR cv_image_to_qpixmap] Failed to create QImage from OpenCV image.")
             return QPixmap()
        return QPixmap.fromImage(qimage)
    except Exception as e:
        print(f"[ERROR cv_image_to_qpixmap] Error converting OpenCV image: {e}")
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

        # <<< !!! THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU FILE MODEL BIỂN SỐ CỦA BẠN KHÁC !!! >>>
        self.lp_model_path = "license_plate_detector.pt" # Ví dụ: dùng file tên best_lp_detector.pt
        # ------------------------------------------------------------------------

        self.vehicle_classes_vn = {
            'car': 'car', 'motorcycle': 'motorcycle', 'bus': 'bus', 'truck': 'truck',
        }
        self.target_vehicle_classes = ['car', 'motorcycle']
        self.initUI()
        self.load_models_thread = threading.Thread(target=self._load_models_background, daemon=True)
        self.load_models_thread.start()

    def _load_models_background(self):
        # (Code tải model giữ nguyên như trước)
        try:
            print("[INFO] Bắt đầu tải model nhận diện xe (YOLOv8)...")
            self.yolo_model = YOLO('yolov8n.pt')
            _ = self.yolo_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False) # Warm-up
            print("[INFO] Model nhận diện xe đã tải xong.")

            print(f"[INFO] Bắt đầu tải model nhận diện biển số ({os.path.basename(self.lp_model_path)})...")
            if not os.path.exists(self.lp_model_path):
                 print(f"[ERROR] Không tìm thấy file model biển số: {self.lp_model_path}")
                 self.status_label.setText(f"Lỗi: Không tìm thấy {os.path.basename(self.lp_model_path)}")
                 self.models_loaded = False
                 self.update_button_states()
                 return

            self.lp_model = YOLO(self.lp_model_path)
            _ = self.lp_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False) # Warm-up
            print("[INFO] Model nhận diện biển số đã tải xong.")

            print("[INFO] Bắt đầu khởi tạo OCR Reader (EasyOCR)...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[INFO] OCR Reader đã sẵn sàng.")

            self.models_loaded = True
            print("[INFO] Tất cả model đã sẵn sàng.")
            self.status_label.setText("Các model đã sẵn sàng.")
            self.update_button_states()

        except Exception as e:
            print(f"[ERROR] Lỗi trong quá trình tải model: {e}")
            self.models_loaded = False
            self.status_label.setText(f"Lỗi tải model: {e}")
            self.update_button_states()

    def initUI(self):
        # self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng Phương tiện & Biển số') # Sẽ đặt trong header
        self.setWindowFlag(Qt.FramelessWindowHint) # Có thể bỏ viền nếu muốn full screen thực sự
        self.showFullScreen() # <<< MỞ TOÀN MÀN HÌNH

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Layout chính của toàn cửa sổ
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 10) # Giảm margin trên/dưới một chút
        main_layout.setSpacing(10) # Giảm khoảng cách chung

        # --- Header Layout (Mới) ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5) # Thêm margin dưới cho header

        # Nút Quay lại
        back_button = QPushButton(" ← Quay lại")
        back_button.setFont(QFont('Segoe UI', 11))
        back_button.setFixedSize(120, 35) # Kích thước cố định
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
            QPushButton:pressed {
                background-color: #6c7a7d;
            }
        """)
        back_button.clicked.connect(self.close) # <<< Nhấn nút là đóng cửa sổ này
        header_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        # Tiêu đề chính
        title_label = QLabel('Xử lý Ảnh Hàng Loạt - Nhận Dạng Phương Tiện & Biển Số')
        title_label.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title_label, stretch=1) # Stretch để tiêu đề chiếm không gian còn lại

        # Thêm spacer để đẩy tiêu đề sang trái một chút nếu cần, hoặc giữ cân bằng
        # header_layout.addSpacerItem(QSpacerItem(120, 35, QSizePolicy.Fixed, QSizePolicy.Fixed)) # Giữ chỗ tương đương nút back
        header_layout.addStretch(0) # Thêm stretch để đảm bảo cân bằng


        # --- Thêm header vào layout chính ---
        main_layout.addLayout(header_layout)

        # Đường kẻ phân cách dưới header
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(separator_line)

        # --- Khu vực chính: Chia 2 cột lớn (Như cũ) ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 10, 0, 0) # Thêm margin trên cho content

        # --- Cột bên trái (Như cũ) ---
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(10)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px;
                background-color: #ecf0f1; font-size: 11pt;
            }
            QListWidget::item:selected { background-color: #3498db; color: white; }
        """)
        left_column_layout.addWidget(self.file_list_widget, stretch=1)
        self.add_files_button = QPushButton("Thêm file ảnh")
        self.add_files_button.setFont(QFont('Segoe UI', 11))
        self.add_files_button.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; border: none;
                            padding: 10px 15px; border-radius: 5px; min-height: 30px; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.add_files_button.clicked.connect(self.add_files)
        left_column_layout.addWidget(self.add_files_button, alignment=Qt.AlignLeft)

        # --- Cột bên phải (Như cũ) ---
        right_column_layout = QVBoxLayout()
        right_column_layout.setSpacing(10)
        self.current_image_label = QLabel("Khung hiển thị ảnh đang xử lý / ảnh gốc")
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setFrameShape(QFrame.Box)
        self.current_image_label.setFrameShadow(QFrame.Sunken)
        self.current_image_label.setMinimumSize(400, 300)
        self.current_image_label.setStyleSheet("""
            QLabel { background-color: #ffffff; border: 1px dashed #bdc3c7;
                     color: #7f8c8d; font-size: 12pt; }
        """)
        right_column_layout.addWidget(self.current_image_label, stretch=1)
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.Box)
        self.results_scroll_area.setFrameShadow(QFrame.Sunken)
        self.results_scroll_area.setMinimumHeight(150)
        self.results_scroll_area.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px;")
        results_container_widget = QWidget()
        self.results_layout = QVBoxLayout(results_container_widget)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setSpacing(5)
        self.clear_results()
        self.results_scroll_area.setWidget(results_container_widget)
        right_column_layout.addWidget(self.results_scroll_area, stretch=0)

        content_layout.addLayout(left_column_layout, stretch=1)
        content_layout.addLayout(right_column_layout, stretch=2)
        # --- Thêm content vào layout chính ---
        main_layout.addLayout(content_layout, stretch=1)

        # --- Khu vực điều khiển (Như cũ, đặt phía dưới) ---
        controls_main_layout = QVBoxLayout()
        controls_main_layout.setSpacing(10)

        # -- Hàng 1: Thư mục lưu (Như cũ) --
        output_dir_layout = QHBoxLayout()
        output_dir_group = QGroupBox("Thư mục lưu trữ kết quả")
        output_dir_group.setFont(QFont('Segoe UI', 10))
        output_dir_group_layout = QHBoxLayout(output_dir_group)
        self.output_dir_label = QLabel("Chưa chọn thư mục")
        self.output_dir_label.setStyleSheet("font-style: italic; color: #555;")
        self.select_output_dir_button = QPushButton("Chọn thư mục")
        self.select_output_dir_button.setFont(QFont('Segoe UI', 10))
        self.select_output_dir_button.clicked.connect(self.select_output_directory)
        output_dir_group_layout.addWidget(self.output_dir_label, stretch=1)
        output_dir_group_layout.addWidget(self.select_output_dir_button)
        output_dir_layout.addWidget(output_dir_group)
        controls_main_layout.addLayout(output_dir_layout)

        # -- Hàng 2: Options và Nút xử lý (Như cũ) --
        options_buttons_layout = QHBoxLayout()
        options_buttons_layout.setSpacing(20)
        # Nhóm thay đổi kích thước (Như cũ)
        resize_group = QGroupBox("Thay đổi kích thước")
        resize_group.setFont(QFont('Segoe UI', 10))
        resize_layout = QFormLayout(resize_group)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000); self.width_spin.setValue(800); self.width_spin.setSuffix(" px"); self.width_spin.setMinimumWidth(80)
        self.height_spin = QSpinBox()

        self.height_spin.setRange(1, 10000); self.height_spin.setValue(600); self.height_spin.setSuffix(" px"); self.height_spin.setMinimumWidth(80)
        resize_layout.addRow("Chiều rộng:", self.width_spin)
        resize_layout.addRow("Chiều cao:", self.height_spin)
        options_buttons_layout.addWidget(resize_group)
        # Nhóm thay đổi định dạng (Như cũ)
        format_group = QGroupBox("Thay đổi định dạng")
        format_group.setFont(QFont('Segoe UI', 10))
        format_layout = QVBoxLayout(format_group)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP", "GIF", "TIFF"])
        self.format_combo.setFont(QFont('Segoe UI', 10))
        format_layout.addWidget(QLabel("Chọn định dạng mới:"))
        format_layout.addWidget(self.format_combo)
        options_buttons_layout.addWidget(format_group)
        options_buttons_layout.addStretch(1)
        # Khu vực các nút xử lý (Như cũ)
        process_buttons_layout = QVBoxLayout()
        process_buttons_layout.setSpacing(8)
        self.process_detect_button = QPushButton("Nhận dạng Phương tiện & Biển số")
        self.process_detect_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.process_detect_button.setMinimumSize(250, 40)
        self.process_detect_button.setStyleSheet("""
            QPushButton { background-color: #16a085; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #1abc9c; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """)
        self.process_detect_button.clicked.connect(self.process_vehicle_and_lp_detection)
        self.process_detect_button.setEnabled(False)
        process_buttons_layout.addWidget(self.process_detect_button)
        self.resize_button = QPushButton("Xử lý Kích thước")
        self.resize_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.resize_button.setMinimumSize(250, 40)
        self.resize_button.setStyleSheet("""
            QPushButton { background-color: #e67e22; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.resize_button.clicked.connect(self.process_resize)
        self.resize_button.setEnabled(False)
        process_buttons_layout.addWidget(self.resize_button)
        self.format_button = QPushButton("Thay đổi Định dạng")
        self.format_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.format_button.setMinimumSize(250, 40)
        self.format_button.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.format_button.clicked.connect(self.process_format_change)
        self.format_button.setEnabled(False)
        process_buttons_layout.addWidget(self.format_button)
        options_buttons_layout.addLayout(process_buttons_layout)
        controls_main_layout.addLayout(options_buttons_layout)
        # --- Thêm controls vào layout chính ---
        main_layout.addLayout(controls_main_layout, stretch=0)

        # --- Status Bar (Như cũ) ---
        self.status_label = QLabel("Đang tải models...")
        self.status_label.setStyleSheet("padding: 5px; color: #e67e22; border-top: 1px solid #ddd;")

        self.height_spin.setRange(1, 5000)
        self.height_spin.setValue(600)

        settings_layout.addRow("Hành động:", self.action_combo)
        settings_layout.addRow("Định dạng đầu ra:", self.format_combo)
        settings_layout.addRow("Chiều rộng:", self.width_spin)
        settings_layout.addRow("Chiều cao:", self.height_spin)

        settings_group.setLayout(settings_layout)

        # Thư mục đầu ra
        output_group = QGroupBox("Thư mục đầu ra")
        output_layout = QHBoxLayout()

        self.output_path = QLabel("Chưa chọn")
        self.output_path.setStyleSheet("background: white; padding: 5px; border: 1px solid #ccc; border-radius: 3px;")

        output_button = QPushButton("Chọn")
        output_button.setFixedWidth(80)
        output_button.setStyleSheet("""
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 3px;
}
""")
        output_button.clicked.connect(self.select_output_directory)

        output_layout.addWidget(self.output_path, 1)
        output_layout.addWidget(output_button)
        output_group.setLayout(output_layout)

        # Nút xử lý và thanh tiến trình
        process_layout = QVBoxLayout()

        self.process_button = QPushButton("Bắt đầu xử lý")
        self.process_button.setFixedHeight(40)
        self.process_button.setStyleSheet("""
QPushButton {
    background-color: #2ecc71;
    color: white;
    border-radius: 5px;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #27ae60;
}
""")
        self.process_button.clicked.connect(self.toggle_process)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
QProgressBar {
    border: 1px solid #ccc;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2ecc71;
    width: 10px;
    margin: 0.5px;
}
""")

        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.progress_bar)

        # Thêm các phần vào panel
        left_panel.addWidget(file_group)

        right_panel.addWidget(settings_group)
        right_panel.addWidget(output_group)
        right_panel.addLayout(process_layout)

        # Thêm vào layout nội dung
        content_layout.addLayout(left_panel, 1)
        content_layout.addLayout(right_panel, 1)

        # Thêm các thành phần vào layout chính
        main_layout.addLayout(header_layout)
        main_layout.addLayout(content_layout, 1)

        # Status bar
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")

        main_layout.addWidget(self.status_label)

        # --- Kết nối tín hiệu và Style chung (Như cũ) ---
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f8f9fa; font-family: Segoe UI; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px;
                        margin-top: 10px; padding: 15px 10px 10px 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               padding: 0 5px; left: 10px; color: #2c3e50; }
            QSpinBox, QComboBox { padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; min-height: 25px;}
            QLabel { color: #2c3e50; }
            #MainWindow QFrame#separator_line { /* Thêm ID để style nếu cần */
                border-top: 1px solid #ddd;
            }
        """)

    # --- Các hàm xử lý logic (add_files, select_output_directory, display_selected_image,
    #      _prepare_processing, _finish_processing, update_button_states, set_controls_enabled,
    #      process_resize, process_format_change, process_vehicle_and_lp_detection,
    #      add_result_item, clear_results, keyPressEvent, resizeEvent, closeEvent)
    #      GIỮ NGUYÊN NHƯ CODE TRƯỚC ĐÓ ---
    def add_files(self):

        # (Code đã có)
        image_filter = "Ảnh (*.jpg *.jpeg *.png *.bmp *.gif *.tiff);;Tất cả file (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", "", image_filter)
        if files:
            added_count = 0
            for file in files:
                if file not in self.files:
                    self.files.append(file)
                    self.file_list_widget.addItem(os.path.basename(file))
                    added_count += 1
            if added_count > 0:
                if self.models_loaded: self.status_label.setText(f"Đã thêm {added_count} file ảnh.")
                else: self.status_label.setText(f"Đã thêm {added_count} file ảnh. (Models đang tải...)")
            else:
                if self.models_loaded: self.status_label.setText("Các file đã chọn đã có trong danh sách.")
                else: self.status_label.setText("Các file đã chọn đã có trong danh sách. (Models đang tải...)")
            self.update_button_states()
        else:
            if self.models_loaded: self.status_label.setText("Không có file nào được chọn.")
            else: self.status_label.setText("Không có file nào được chọn. (Models đang tải...)")

    def select_output_directory(self):
        # (Code đã có)
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if directory:
            self.output_directory = directory
            self.output_dir_label.setText(f"Lưu tại: {directory}")
            self.output_dir_label.setStyleSheet("font-style: normal; color: #27ae60;")
            if self.models_loaded: self.status_label.setText(f"Đã chọn thư mục lưu: {directory}")
            else: self.status_label.setText(f"Đã chọn thư mục lưu: {directory} (Models đang tải...)")
            self.update_button_states()

        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file", "", "Images (*.jpg *.png *.gif *.bmp);;All Files (*)")
        if files:
            for file in files:
                self.file_list.addItem(file)
            self.status_label.setText(f"Đã thêm {len(files)} file")
        else:
            self.status_label.setText("Không có file nào được chọn")

    def clear_files(self):
        self.file_list.clear()
        self.status_label.setText("Đã xóa tất cả file")

    def toggle_process(self):
        if not self.processing:
            self.processing = True
            self.process_button.setText("Đang xử lý...")
            self.progress_bar.setValue(0)

            # Giả lập tiến trình xử lý
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_progress)
            self.timer.start(100)

        else:
            if self.models_loaded: self.status_label.setText("Chưa chọn thư mục lưu.")
            else: self.status_label.setText("Chưa chọn thư mục lưu. (Models đang tải...)")
            self.output_directory = None
            self.output_dir_label.setText("Chưa chọn thư mục")
            self.output_dir_label.setStyleSheet("font-style: italic; color: #555;")
            self.update_button_states()

    def display_selected_image(self, current_item, previous_item):
        # (Code đã có)
        if current_item:
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)
            if full_path and os.path.exists(full_path):
                try:
                    img_cv = cv2.imread(full_path)
                    if img_cv is None: raise ValueError("OpenCV không thể đọc file ảnh.")
                    pixmap = cv_image_to_qpixmap(img_cv)
                    if not pixmap.isNull():
                        self.current_image_label._original_pixmap = pixmap
                        scaled_pixmap = pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.current_image_label.setPixmap(scaled_pixmap)
                        if self.models_loaded: self.status_label.setText(f"Xem trước: {file_name}")
                        else: self.status_label.setText(f"Xem trước: {file_name} (Models đang tải...)")
                    else:
                        self.current_image_label.clear(); self.current_image_label.setText(f"Lỗi chuyển đổi ảnh:\n{file_name}"); self.status_label.setText(f"Lỗi xem trước: {file_name}")
                        if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
                except Exception as e:
                    print(f"[ERROR display_selected_image] Error opening/processing image {file_name}: {e}")
                    self.current_image_label.clear(); self.current_image_label.setText(f"Lỗi khi mở ảnh:\n{file_name}\n({e})"); self.status_label.setText(f"Lỗi xem trước: {file_name}")
                    if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
            else:
                self.current_image_label.clear(); self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}"); self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
                if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
        else:
             self.current_image_label.clear(); self.current_image_label.setText("Chọn một file để xem trước")
             if hasattr(self.current_image_label, '_original_pixmap'): del self.current_image_label._original_pixmap
             if self.models_loaded: self.status_label.setText("Sẵn sàng.")
             else: self.status_label.setText("Models đang tải...")

    def _prepare_processing(self, operation_name):
        # (Code đã có)
        if not self.files: QMessageBox.warning(self, "Thiếu file", "Vui lòng thêm file ảnh trước khi xử lý."); return False
        if not self.output_directory: QMessageBox.warning(self, "Thiếu thư mục lưu", "Vui lòng chọn thư mục lưu kết quả trước khi xử lý."); return False
        if operation_name == "nhận dạng phương tiện và biển số" and not self.models_loaded:
             if not self.yolo_model or not self.lp_model or not self.ocr_reader:
                 QMessageBox.warning(self, "Models chưa sẵn sàng", "Các model cần thiết đang được tải hoặc đã xảy ra lỗi. Vui lòng chờ hoặc kiểm tra console.")
                 return False
             else: self.models_loaded = True
        self.status_label.setText(f"Bắt đầu {operation_name}..."); self.set_controls_enabled(False); self.clear_results(); QApplication.processEvents(); return True

    def _finish_processing(self, operation_name, count):
        # (Code đã có)
        self.status_label.setText(f"Hoàn tất {operation_name} {count} file."); self.set_controls_enabled(True)

    def update_button_states(self):
        # (Code đã có)
         can_process_basic = self.file_list_widget.count() > 0 and self.output_directory is not None
         self.resize_button.setEnabled(can_process_basic)
         self.format_button.setEnabled(can_process_basic)
         self.process_detect_button.setEnabled(can_process_basic and self.models_loaded)

    def set_controls_enabled(self, enabled):
        # (Code đã có)
        self.add_files_button.setEnabled(enabled); self.select_output_dir_button.setEnabled(enabled); self.file_list_widget.setEnabled(enabled)
        self.width_spin.setEnabled(enabled); self.height_spin.setEnabled(enabled); self.format_combo.setEnabled(enabled)
        can_process_basic = enabled and self.file_list_widget.count() > 0 and self.output_directory is not None
        self.resize_button.setEnabled(can_process_basic); self.format_button.setEnabled(can_process_basic)
        self.process_detect_button.setEnabled(can_process_basic and self.models_loaded)

    def process_resize(self):
        # (Code đã có)
        if not self._prepare_processing("thay đổi kích thước"): return
        target_width = self.width_spin.value(); target_height = self.height_spin.value()
        processed_count = 0; last_processed_pixmap = None
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path); name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}_resized{ext_part}"; output_path = os.path.join(self.output_directory, output_filename)
            self.status_label.setText(f"Đang đổi kích thước: {base_name} ({i+1}/{len(self.files)})"); QApplication.processEvents()
            try:
                with Image.open(file_path) as img:
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull(): self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    QApplication.processEvents()
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS); resized_img.save(output_path)
                    result_pixmap = pil_to_qpixmap(resized_img)
                    if not result_pixmap.isNull(): self.add_result_item(result_pixmap, base_name, f"Đã đổi kích thước thành {target_width}x{target_height}", output_path); last_processed_pixmap = result_pixmap; processed_count += 1
                    else: self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau resize", file_path)
            except Exception as e: print(f"Lỗi khi đổi kích thước file {base_name}: {e}"); self.add_result_item(None, base_name, f"LỖI resize: {e}", file_path)
        if last_processed_pixmap and not last_processed_pixmap.isNull(): self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._finish_processing("thay đổi kích thước", processed_count)

    def process_format_change(self):
        # (Code đã có)
        if not self._prepare_processing("thay đổi định dạng"): return
        target_format = self.format_combo.currentText().lower()
        processed_count = 0; last_processed_pixmap = None
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path); name_part, _ = os.path.splitext(base_name)
            output_filename = f"{name_part}.{target_format}"; output_path = os.path.join(self.output_directory, output_filename)
            self.status_label.setText(f"Đang đổi định dạng: {base_name} -> .{target_format} ({i+1}/{len(self.files)})"); QApplication.processEvents()
            try:
                with Image.open(file_path) as img:
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull(): self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    QApplication.processEvents()
                    save_img = img
                    if img.mode in ("RGBA", "P") and target_format.upper() in ("JPG", "JPEG"): save_img = img.convert("RGB")
                    save_img.save(output_path, format=target_format.upper())
                    with Image.open(output_path) as saved_img:
                        result_pixmap = pil_to_qpixmap(saved_img)
                        if not result_pixmap.isNull(): self.add_result_item(result_pixmap, base_name, f"Đã đổi định dạng thành .{target_format.upper()}", output_path); last_processed_pixmap = result_pixmap; processed_count += 1
                        else: self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau khi đổi định dạng", file_path)
            except Exception as e: print(f"Lỗi khi đổi định dạng file {base_name} sang {target_format}: {e}"); self.add_result_item(None, base_name, f"LỖI đổi sang .{target_format.upper()}: {e}", file_path)
        if last_processed_pixmap and not last_processed_pixmap.isNull(): self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._finish_processing("thay đổi định dạng", processed_count)

    def process_vehicle_and_lp_detection(self):
        # (Code đã có)
        operation_name = "nhận dạng phương tiện và biển số"; processed_count = 0; last_processed_pixmap = None
        if not self._prepare_processing(operation_name): return
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path); name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}_detected_recognized{ext_part}"; output_path = os.path.join(self.output_directory, output_filename)
            self.status_label.setText(f"Đang xử lý: {base_name} ({i+1}/{len(self.files)})"); QApplication.processEvents()
            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None: raise ValueError("OpenCV không thể đọc file ảnh.")
                original_pixmap = cv_image_to_qpixmap(img_cv)
                if not original_pixmap.isNull(): self.current_image_label.setPixmap(original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                QApplication.processEvents()
                vehicle_results = self.yolo_model(img_cv, verbose=False)
                detection_summary = []; img_draw = img_cv.copy()
                for result in vehicle_results:
                    for box in result.boxes:
                        x1_v, y1_v, x2_v, y2_v = map(int, box.xyxy[0]); conf_v = float(box.conf[0]); cls_v = int(box.cls[0])
                        class_name_v = self.yolo_model.names[cls_v]
                        if class_name_v in self.target_vehicle_classes and conf_v > 0.5:
                            class_name_vn = self.vehicle_classes_vn.get(class_name_v, class_name_v)
                            vehicle_label = f"{class_name_vn}: {conf_v:.2f}"; vehicle_color = (0, 255, 0)
                            cv2.rectangle(img_draw, (x1_v, y1_v), (x2_v, y2_v), vehicle_color, 2)
                            (lbl_w, lbl_h), base = cv2.getTextSize(vehicle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(img_draw, (x1_v, y1_v - lbl_h - base), (x1_v + lbl_w, y1_v), vehicle_color, -1)
                            cv2.putText(img_draw, vehicle_label, (x1_v, y1_v - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                            vehicle_roi = img_cv[y1_v:y2_v, x1_v:x2_v]
                            if vehicle_roi.size == 0: continue
                            lp_results = self.lp_model(vehicle_roi, verbose=False, conf=0.4)
                            best_lp_box = None; max_conf_lp = 0
                            for lp_result in lp_results:
                                for lp_box in lp_result.boxes:
                                     if lp_box.conf[0] > max_conf_lp: max_conf_lp = lp_box.conf[0]; best_lp_box = lp_box
                            recognized_plate = "N/A"
                            if best_lp_box is not None:
                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, best_lp_box.xyxy[0])
                                lp_image = vehicle_roi[lp_y1:lp_y2, lp_x1:lp_x2]
                                if lp_image.size > 0:
                                    gray_lp = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
                                    try:
                                        ocr_result = self.ocr_reader.readtext(gray_lp, detail=0, paragraph=False, allowlist=self.ocr_allowed_chars)
                                        if ocr_result:
                                            raw_text = "".join(ocr_result).upper().replace(" ", "")
                                            recognized_plate = raw_text # Giữ nguyên hoặc chuẩn hóa thêm nếu cần
                                            detection_summary.append(f"{class_name_vn}: {recognized_plate}")
                                            g_lp_x1 = x1_v + lp_x1; g_lp_y1 = y1_v + lp_y1; g_lp_x2 = x1_v + lp_x2; g_lp_y2 = y1_v + lp_y2
                                            lp_color = (255, 0, 0)
                                            cv2.rectangle(img_draw, (g_lp_x1, g_lp_y1), (g_lp_x2, g_lp_y2), lp_color, 2)
                                            (lbl_w_ocr, lbl_h_ocr), base_ocr = cv2.getTextSize(recognized_plate, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                            cv2.rectangle(img_draw, (g_lp_x1, g_lp_y2 + base_ocr), (g_lp_x1 + lbl_w_ocr, g_lp_y2 + lbl_h_ocr + base_ocr + 5), lp_color, -1)
                                            cv2.putText(img_draw, recognized_plate, (g_lp_x1, g_lp_y2 + lbl_h_ocr + base_ocr), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                        else: detection_summary.append(f"{class_name_vn}: (LP detected, OCR failed)")
                                    except Exception as ocr_err: print(f"Lỗi OCR cho {base_name}: {ocr_err}"); detection_summary.append(f"{class_name_vn}: (LP detected, OCR error)")
                                else: detection_summary.append(f"{class_name_vn}: (LP region empty)")
                            else: detection_summary.append(f"{class_name_vn}: (LP not found)")
                cv2.imwrite(output_path, img_draw)
                result_pixmap = cv_image_to_qpixmap(img_draw)
                if not result_pixmap.isNull():
                    info_text = "; ".join(detection_summary) if detection_summary else "Không phát hiện xe/biển số phù hợp."
                    self.add_result_item(result_pixmap, base_name, info_text, output_path)
                    last_processed_pixmap = result_pixmap; processed_count += 1
                else: self.add_result_item(None, base_name, "LỖI: Không tạo được ảnh xem trước sau xử lý", file_path)
            except Exception as e: print(f"Lỗi nghiêm trọng khi xử lý file {base_name}: {e}"); import traceback; traceback.print_exc(); self.add_result_item(None, base_name, f"LỖI xử lý: {e}", file_path)
        if last_processed_pixmap and not last_processed_pixmap.isNull(): self.current_image_label.setPixmap(last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._finish_processing(operation_name, processed_count)

    def add_result_item(self, pixmap, original_name, info_text, result_path):
        # (Code đã có)
        item_widget = QWidget(); item_layout = QVBoxLayout(item_widget); item_layout.setContentsMargins(5, 5, 5, 5); item_layout.setSpacing(3)
        img_label = QLabel(); img_label.setAlignment(Qt.AlignCenter); img_label.setMinimumSize(180, 120); img_label.setStyleSheet("background-color: #eee; border: 1px solid #ddd;")
        if pixmap and not pixmap.isNull(): img_label.setPixmap(pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); item_widget.setToolTip(f"File gốc: {original_name}\nKết quả: {os.path.basename(result_path)}\n{info_text}")
        else: img_label.setText("Lỗi xử lý"); img_label.setStyleSheet("background-color: #ffebee; border: 1px solid #e57373; color: #c62828;"); item_widget.setToolTip(f"File gốc: {original_name}\nLỗi: {info_text}")
        info_label = QLabel(f"{original_name}\n{info_text}"); info_label.setAlignment(Qt.AlignCenter); info_label.setWordWrap(True); info_label.setStyleSheet("font-size: 9pt; color: #333;")
        item_layout.addWidget(img_label); item_layout.addWidget(info_label); item_widget.setStyleSheet("background-color: white; border: 1px solid #eee; margin-bottom: 5px; border-radius: 3px;")
        if self.results_layout.count() == 1 and isinstance(self.results_layout.itemAt(0).widget(), QLabel) and "Kết quả xử lý sẽ hiển thị ở đây" in self.results_layout.itemAt(0).widget().text():
             item_to_remove = self.results_layout.takeAt(0);
             if item_to_remove.widget(): item_to_remove.widget().deleteLater()
        self.results_layout.addWidget(item_widget)

    def clear_results(self):
        # (Code đã có)
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        placeholder_result = QLabel("Kết quả xử lý sẽ hiển thị ở đây"); placeholder_result.setAlignment(Qt.AlignCenter); placeholder_result.setStyleSheet("color: #7f8c8d; font-size: 10pt; padding: 20px; background-color: transparent; border: none;")
        self.results_layout.addWidget(placeholder_result)

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu ra")
        if directory:
            self.output_path.setText(directory)
            self.status_label.setText(f"Đã chọn thư mục đầu ra: {directory}")

    def keyPressEvent(self, event):
        # (Code đã có) - Có thể thêm chức năng cho nút ESC nếu muốn
        if event.key() == Qt.Key_Escape:
            self.close() # Thoát ứng dụng BatchMode khi nhấn ESC

    def resizeEvent(self, event):
        # (Code đã có)
        super().resizeEvent(event)
        if hasattr(self, 'current_image_label') and self.current_image_label:
            current_pixmap_orig = getattr(self.current_image_label, '_original_pixmap', None)
            if current_pixmap_orig and not current_pixmap_orig.isNull():
                 scaled_pixmap = current_pixmap_orig.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 if scaled_pixmap and not scaled_pixmap.isNull(): self.current_image_label.setPixmap(scaled_pixmap)
            elif self.current_image_label.pixmap() and not self.current_image_label.pixmap().isNull():
                 current_pixmap = self.current_image_label.pixmap()
                 scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 if scaled_pixmap and not scaled_pixmap.isNull(): self.current_image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        # (Code đã có)
        super().closeEvent(event)

# --- Main execution block ---
if __name__ == '__main__':
    # (Kiểm tra thư viện giữ nguyên)
    missing_libs = []
    try: from PyQt5 import QtWidgets, QtCore, QtGui
    except ImportError: missing_libs.append("PyQt5")
    try: from PIL import Image
    except ImportError: missing_libs.append("Pillow")
    try: import cv2
    except ImportError: missing_libs.append("opencv-python")
    try: from ultralytics import YOLO
    except ImportError: missing_libs.append("ultralytics")
    try: import torch
    except ImportError: missing_libs.append("torch (pytorch)")
    try: import numpy
    except ImportError: missing_libs.append("numpy")
    try: import easyocr
    except ImportError: missing_libs.append("easyocr")

    if missing_libs:
        print("Lỗi: Thiếu các thư viện cần thiết.")
        print("Vui lòng cài đặt bằng pip:")
        if "PyQt5" in missing_libs: print("  pip install PyQt5")
        if "Pillow" in missing_libs: print("  pip install Pillow")
        if "opencv-python" in missing_libs: print("  pip install opencv-python")
        if "ultralytics" in missing_libs: print("  pip install ultralytics")
        if "torch (pytorch)" in missing_libs: print("  pip install torch torchvision torchaudio")
        if "numpy" in missing_libs: print("  pip install numpy")
        if "easyocr" in missing_libs: print("  pip install easyocr")
        sys.exit(1)


    app = QApplication(sys.argv)

    window = BatchModeAppRedesigned()
    # window.show() # Không cần gọi show() nữa vì đã có showFullScreen() trong initUI
    sys.exit(app.exec_())

    window = BatchModeApp()
    sys.exit(app.exec_())


# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame, QMessageBox, QComboBox) # Thêm QMessageBox, QComboBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage # Import QImage ở đây
from PIL import Image # Thêm thư viện Pillow
import cv2 # Thêm thư viện OpenCV
import numpy as np # Thêm thư viện numpy
from ultralytics import YOLO # Thêm thư viện YOLO

# --- Helper function to convert Pillow Image to QPixmap ---
# (Giữ nguyên hàm pil_to_qpixmap của bạn)
def pil_to_qpixmap(pil_image):
    """Converts a Pillow Image object to a QPixmap, including debug info."""
    try:
        # print(f"[DEBUG pil_to_qpixmap] Original PIL Mode: {pil_image.mode}, Size: {pil_image.size}") # Debug info

        # Xử lý các mode phổ biến
        if pil_image.mode == "RGB":
            # print("[DEBUG pil_to_qpixmap] Mode is RGB. Converting directly.") # Debug info
            img_byte_arr = pil_image.tobytes("raw", "RGB")
            bytes_per_line = pil_image.size[0] * 3
            qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], bytes_per_line, QImage.Format_RGB888)
        elif pil_image.mode == "RGBA":
            # print("[DEBUG pil_to_qpixmap] Mode is RGBA. Converting directly.") # Debug info
            img_byte_arr = pil_image.tobytes("raw", "RGBA")
            bytes_per_line = pil_image.size[0] * 4
            qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], bytes_per_line, QImage.Format_RGBA8888)
        elif pil_image.mode == "L": # Grayscale
            # print("[DEBUG pil_to_qpixmap] Mode is L (Grayscale). Converting to RGB.") # Debug info
            rgb_image = pil_image.convert("RGB")
            img_byte_arr = rgb_image.tobytes("raw", "RGB")
            bytes_per_line = rgb_image.size[0] * 3
            qimage = QImage(img_byte_arr, rgb_image.size[0], rgb_image.size[1], bytes_per_line, QImage.Format_RGB888)
        elif pil_image.mode == "P": # Palette
             # print("[DEBUG pil_to_qpixmap] Mode is P (Palette). Converting to RGBA (handling transparency).") # Debug info
             rgba_image = pil_image.convert("RGBA") # Chuyển Palette sang RGBA để đảm bảo màu và transparency
             img_byte_arr = rgba_image.tobytes("raw", "RGBA")
             bytes_per_line = rgba_image.size[0] * 4
             qimage = QImage(img_byte_arr, rgba_image.size[0], rgba_image.size[1], bytes_per_line, QImage.Format_RGBA8888)
        else: # Các trường hợp khác (CMYK, YCbCr,...) cố gắng chuyển về RGB
            # print(f"[DEBUG pil_to_qpixmap] Mode is {pil_image.mode}. Attempting conversion to RGB.") # Debug info
            try:
                 rgb_image = pil_image.convert("RGB")
                 img_byte_arr = rgb_image.tobytes("raw", "RGB")
                 bytes_per_line = rgb_image.size[0] * 3
                 qimage = QImage(img_byte_arr, rgb_image.size[0], rgb_image.size[1], bytes_per_line, QImage.Format_RGB888)
            except Exception as convert_err:
                 print(f"[ERROR pil_to_qpixmap] Could not convert mode {pil_image.mode} to RGB: {convert_err}")
                 return QPixmap() # Trả về pixmap rỗng nếu không chuyển được

        # print(f"[DEBUG pil_to_qpixmap] Created QImage - Format: {qimage.format()}, Size: {qimage.size()}, isNull: {qimage.isNull()}") # Debug info

        if qimage.isNull():
            print("[ERROR pil_to_qpixmap] Failed to create QImage.")
            return QPixmap()

        # Tạo QPixmap từ QImage
        pixmap = QPixmap.fromImage(qimage)
        # print(f"[DEBUG pil_to_qpixmap] Created QPixmap - Size: {pixmap.size()}, isNull: {pixmap.isNull()}") # Debug info
        return pixmap

    except Exception as e:
        print(f"[ERROR pil_to_qpixmap] General error during conversion: {e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi hơn
        return QPixmap()

# --- Helper function to convert OpenCV Image (NumPy array) to QPixmap ---
def cv_image_to_qpixmap(cv_img):
    """Converts an OpenCV image (NumPy array) to QPixmap."""
    try:
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        # Chuyển đổi BGR (OpenCV default) sang RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
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
        self.output_directory = None # Thêm biến lưu đường dẫn thư mục lưu
        self.yolo_model = None # Biến lưu trữ mô hình YOLO đã tải
        self.vehicle_classes_vn = { # Mapping tên lớp sang tiếng Việt
            'person': 'nguoi',
            'bicycle': 'xe dap',
            'car': 'o to',
            'motorcycle': 'xe may',
            'airplane': 'may bay',
            'bus': 'xe buyt',
            'train': 'tau hoa',
            'truck': 'xe tai',
            'boat': 'thuyen',
            # Thêm các lớp khác nếu cần
        }
        self.target_vehicle_classes = ['car', 'motorcycle', 'bus', 'truck'] # Chỉ nhận diện các loại này
        self.initUI()
        self._load_yolo_model() # Tải mô hình YOLO khi khởi tạo

    def _load_yolo_model(self):
        """Tải mô hình YOLO và xử lý lỗi."""
        try:
            # Sử dụng yolov8n.pt (nhanh, nhẹ) hoặc yolov8s.pt, yolov8m.pt,... (chính xác hơn, nặng hơn)
            self.yolo_model = YOLO('yolov8n.pt')
            # Chạy thử một dự đoán nhỏ để chắc chắn mô hình tải đúng (và tải về nếu cần)
            _ = self.yolo_model(np.zeros((64, 64, 3), dtype=np.uint8))
            print("[INFO] Đã tải thành công mô hình YOLOv8.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi tải mô hình", f"Không thể tải mô hình YOLOv8. Vui lòng kiểm tra kết nối internet và cài đặt thư viện 'ultralytics'.\nLỗi: {e}")
            self.yolo_model = None # Đặt lại thành None nếu lỗi

    def initUI(self):
        self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng phương tiện & biển số')
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Khu vực chính: Chia 2 cột lớn ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        # --- Cột bên trái ---
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

        # --- Cột bên phải ---
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
        self.clear_results() # Xóa kết quả cũ và thêm placeholder
        self.results_scroll_area.setWidget(results_container_widget)
        right_column_layout.addWidget(self.results_scroll_area, stretch=0)

        content_layout.addLayout(left_column_layout, stretch=1)
        content_layout.addLayout(right_column_layout, stretch=2)
        main_layout.addLayout(content_layout, stretch=1)

        # --- Khu vực điều khiển (phía dưới) ---
        controls_main_layout = QVBoxLayout() # Layout dọc cho toàn bộ controls
        controls_main_layout.setSpacing(10)

        # -- Hàng 1: Thư mục lưu --
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


        # -- Hàng 2: Options và Nút xử lý --
        options_buttons_layout = QHBoxLayout()
        options_buttons_layout.setSpacing(20)

        # Nhóm thay đổi kích thước
        resize_group = QGroupBox("Thay đổi kích thước")
        resize_group.setFont(QFont('Segoe UI', 10))
        resize_layout = QFormLayout(resize_group)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000); self.width_spin.setValue(800); self.width_spin.setSuffix(" px"); self.width_spin.setMinimumWidth(80)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000); self.height_spin.setValue(600); self.height_spin.setSuffix(" px"); self.height_spin.setMinimumWidth(80)
        resize_layout.addRow("Chiều rộng:", self.width_spin)
        resize_layout.addRow("Chiều cao:", self.height_spin)
        options_buttons_layout.addWidget(resize_group) # Thêm nhóm resize vào layout hàng ngang

        # Nhóm thay đổi định dạng
        format_group = QGroupBox("Thay đổi định dạng")
        format_group.setFont(QFont('Segoe UI', 10))
        format_layout = QVBoxLayout(format_group) # Dùng QVBoxLayout đơn giản
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP", "GIF", "TIFF"])
        self.format_combo.setFont(QFont('Segoe UI', 10))
        format_layout.addWidget(QLabel("Chọn định dạng mới:"))
        format_layout.addWidget(self.format_combo)
        options_buttons_layout.addWidget(format_group) # Thêm nhóm format vào layout hàng ngang

        options_buttons_layout.addStretch(1) # Đẩy các nút xử lý sang phải

        # Khu vực các nút xử lý
        process_buttons_layout = QVBoxLayout() # Layout dọc cho các nút xử lý
        process_buttons_layout.setSpacing(8)

        # --- Nút Xử lý Phương tiện --- <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW BUTTON
        self.process_vehicles_button = QPushButton("Xử lý Phương tiện")
        self.process_vehicles_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.process_vehicles_button.setMinimumSize(180, 40) # Tăng chiều rộng một chút
        self.process_vehicles_button.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.process_vehicles_button.clicked.connect(self.process_vehicle_detection)
        self.process_vehicles_button.setEnabled(False) # Mặc định vô hiệu hóa
        process_buttons_layout.addWidget(self.process_vehicles_button)
        # --- Kết thúc Nút Xử lý Phương tiện ---

        self.resize_button = QPushButton("Xử lý Kích thước")
        self.resize_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.resize_button.setMinimumSize(180, 40)
        self.resize_button.setStyleSheet("""
            QPushButton { background-color: #e67e22; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.resize_button.clicked.connect(self.process_resize)
        self.resize_button.setEnabled(False) # Mặc định vô hiệu hóa
        process_buttons_layout.addWidget(self.resize_button)

        self.format_button = QPushButton("Thay đổi Định dạng")
        self.format_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.format_button.setMinimumSize(180, 40)
        self.format_button.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; border: none; padding: 8px 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.format_button.clicked.connect(self.process_format_change)
        self.format_button.setEnabled(False) # Mặc định vô hiệu hóa
        process_buttons_layout.addWidget(self.format_button)

        options_buttons_layout.addLayout(process_buttons_layout) # Thêm layout chứa các nút xử lý

        # Thêm layout hàng 2 vào layout controls chính
        controls_main_layout.addLayout(options_buttons_layout)

        # Thêm layout controls chính vào layout cửa sổ
        main_layout.addLayout(controls_main_layout, stretch=0)

        # --- Status Bar ---
        self.status_label = QLabel("Sẵn sàng. Vui lòng thêm ảnh và chọn thư mục lưu.")
        self.status_label.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")
        main_layout.addWidget(self.status_label)

        # --- Kết nối tín hiệu chọn item trong list để xem trước ---
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)

        # --- Style chung ---
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f8f9fa; font-family: Segoe UI; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px;
                        margin-top: 10px; padding: 15px 10px 10px 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               padding: 0 5px; left: 10px; color: #2c3e50; }
            QSpinBox, QComboBox { padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; min-height: 25px;}
            QLabel { color: #2c3e50; }
        """)

    def add_files(self):
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
                self.status_label.setText(f"Đã thêm {added_count} file ảnh.")
            else:
                self.status_label.setText("Các file đã chọn đã có trong danh sách.")

            self.update_button_states() # <<<<<< Cập nhật trạng thái nút
        else:
            self.status_label.setText("Không có file nào được chọn.")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if directory:
            self.output_directory = directory
            self.output_dir_label.setText(f"Lưu tại: {directory}")
            self.output_dir_label.setStyleSheet("font-style: normal; color: #27ae60;") # Đổi style khi đã chọn
            self.status_label.setText(f"Đã chọn thư mục lưu: {directory}")
            self.update_button_states() # <<<<<< Cập nhật trạng thái nút
        else:
            self.status_label.setText("Chưa chọn thư mục lưu.")
            self.output_directory = None # Đảm bảo là None nếu không chọn
            self.output_dir_label.setText("Chưa chọn thư mục")
            self.output_dir_label.setStyleSheet("font-style: italic; color: #555;")
            self.update_button_states() # <<<<<< Cập nhật trạng thái nút


    def display_selected_image(self, current_item, previous_item):
        if current_item:
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)

            if full_path and os.path.exists(full_path):
                try:
                    # print(f"\n[DEBUG display_selected_image] Loading: {full_path}") # Debug
                    # Sử dụng OpenCV để đọc ảnh, đảm bảo định dạng nhất quán hơn
                    img_cv = cv2.imread(full_path)
                    if img_cv is None:
                        raise ValueError("OpenCV không thể đọc file ảnh.")

                    pixmap = cv_image_to_qpixmap(img_cv) # Sử dụng hàm helper mới

                    if not pixmap.isNull():
                        # print(f"[DEBUG display_selected_image] Original Pixmap size: {pixmap.size()}") # Debug
                        # print(f"[DEBUG display_selected_image] Target Label size: {self.current_image_label.size()}") # Debug

                        scaled_pixmap = pixmap.scaled(self.current_image_label.size(),
                                                      Qt.KeepAspectRatio, # Giữ tỷ lệ
                                                      Qt.SmoothTransformation) # Scale mượt

                        # print(f"[DEBUG display_selected_image] Scaled Pixmap size: {scaled_pixmap.size()}") # Debug
                        self.current_image_label.setPixmap(scaled_pixmap)
                        self.status_label.setText(f"Xem trước: {file_name}")
                    else:
                        self.current_image_label.setText(f"Lỗi chuyển đổi ảnh:\n{file_name}")
                        self.status_label.setText(f"Lỗi xem trước: {file_name}")
                except Exception as e:
                    print(f"[ERROR display_selected_image] Error opening/processing image {file_name}: {e}") # Debug
                    self.current_image_label.setText(f"Lỗi khi mở ảnh:\n{file_name}\n({e})")
                    self.status_label.setText(f"Lỗi xem trước: {file_name}")
            else:
                self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
        else:
             self.current_image_label.clear() # Xóa pixmap cũ
             self.current_image_label.setText("Chọn một file để xem trước") # Đặt lại text placeholder

    def _prepare_processing(self, operation_name):
        """Hàm trợ giúp kiểm tra điều kiện và chuẩn bị trước khi xử lý."""
        if not self.files:
            QMessageBox.warning(self, "Thiếu file", "Vui lòng thêm file ảnh trước khi xử lý.")
            return False
        if not self.output_directory:
            QMessageBox.warning(self, "Thiếu thư mục lưu", "Vui lòng chọn thư mục lưu kết quả trước khi xử lý.")
            return False
        # Kiểm tra xem mô hình YOLO đã sẵn sàng chưa nếu là xử lý phương tiện
        if operation_name == "nhận diện phương tiện" and self.yolo_model is None:
             QMessageBox.critical(self, "Lỗi mô hình", "Mô hình YOLO chưa được tải thành công. Không thể thực hiện nhận diện.")
             return False


        self.status_label.setText(f"Bắt đầu {operation_name}...")
        self.set_controls_enabled(False) # Vô hiệu hóa các controls
        self.clear_results() # Xóa kết quả cũ
        QApplication.processEvents() # Cập nhật giao diện
        return True

    def _finish_processing(self, operation_name, count):
        """Hàm trợ giúp kết thúc xử lý."""
        self.status_label.setText(f"Hoàn tất {operation_name} {count} file.")
        self.set_controls_enabled(True) # Kích hoạt lại các controls

    def update_button_states(self):
         """Cập nhật trạng thái enabled/disabled của các nút xử lý."""
         can_process = self.file_list_widget.count() > 0 and self.output_directory is not None
         self.resize_button.setEnabled(can_process)
         self.format_button.setEnabled(can_process)
         # Nút xử lý phương tiện chỉ bật khi có mô hình YOLO
         self.process_vehicles_button.setEnabled(can_process and self.yolo_model is not None)

    def set_controls_enabled(self, enabled):
        """Bật/tắt các nút và danh sách file."""
        self.add_files_button.setEnabled(enabled)
        self.select_output_dir_button.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)
        self.width_spin.setEnabled(enabled)
        self.height_spin.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)

        # Cập nhật trạng thái các nút xử lý dựa trên điều kiện chung và trạng thái 'enabled'
        can_process_now = enabled and self.file_list_widget.count() > 0 and self.output_directory is not None
        self.resize_button.setEnabled(can_process_now)
        self.format_button.setEnabled(can_process_now)
        self.process_vehicles_button.setEnabled(can_process_now and self.yolo_model is not None)


    def process_resize(self):
        """Thực hiện thay đổi kích thước ảnh."""
        if not self._prepare_processing("thay đổi kích thước"):
            return

        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        processed_count = 0
        last_processed_pixmap = None # Lưu pixmap cuối cùng để hiển thị

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}_resized{ext_part}"
            output_path = os.path.join(self.output_directory, output_filename)

            self.status_label.setText(f"Đang đổi kích thước: {base_name} ({i+1}/{len(self.files)})")
            QApplication.processEvents()

            try:
                with Image.open(file_path) as img:
                    # Hiển thị ảnh gốc trước khi resize
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull():
                         scaled_original = original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                         self.current_image_label.setPixmap(scaled_original)
                    QApplication.processEvents()

                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    resized_img.save(output_path)

                    result_pixmap = pil_to_qpixmap(resized_img)
                    if not result_pixmap.isNull():
                        self.add_result_item(result_pixmap, base_name, f"Đã đổi kích thước thành {target_width}x{target_height}", output_path)
                        last_processed_pixmap = result_pixmap # Cập nhật pixmap cuối cùng
                        processed_count += 1
                    else:
                        self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau resize", file_path)


            except Exception as e:
                print(f"Lỗi khi đổi kích thước file {base_name}: {e}")
                self.add_result_item(None, base_name, f"LỖI resize: {e}", file_path) # Thêm thông báo lỗi vào kết quả

        # Hiển thị ảnh cuối cùng đã xử lý lên khung chính
        if last_processed_pixmap and not last_processed_pixmap.isNull():
             scaled_last = last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.current_image_label.setPixmap(scaled_last)

        self._finish_processing("thay đổi kích thước", processed_count)

    def process_format_change(self):
        """Thực hiện thay đổi định dạng ảnh."""
        if not self._prepare_processing("thay đổi định dạng"):
            return

        target_format = self.format_combo.currentText().lower()
        processed_count = 0
        last_processed_pixmap = None # Lưu pixmap cuối cùng để hiển thị

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, _ = os.path.splitext(base_name)
            output_filename = f"{name_part}.{target_format}"
            output_path = os.path.join(self.output_directory, output_filename)

            self.status_label.setText(f"Đang đổi định dạng: {base_name} -> .{target_format} ({i+1}/{len(self.files)})")
            QApplication.processEvents()

            try:
                with Image.open(file_path) as img:
                    # Hiển thị ảnh gốc
                    original_pixmap = pil_to_qpixmap(img)
                    if not original_pixmap.isNull():
                         scaled_original = original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                         self.current_image_label.setPixmap(scaled_original)
                    QApplication.processEvents()

                    # Xử lý trường hợp ảnh có alpha channel (vd: PNG) khi lưu sang JPG
                    save_img = img
                    if img.mode in ("RGBA", "P") and target_format.upper() in ("JPG", "JPEG"):
                        print(f"[INFO] Converting {img.mode} to RGB before saving as JPG for {base_name}")
                        save_img = img.convert("RGB")

                    save_img.save(output_path, format=target_format.upper())

                    # Mở lại file vừa lưu để chắc chắn và hiển thị
                    with Image.open(output_path) as saved_img:
                        result_pixmap = pil_to_qpixmap(saved_img)
                        if not result_pixmap.isNull():
                             self.add_result_item(result_pixmap, base_name, f"Đã đổi định dạng thành .{target_format.upper()}", output_path)
                             last_processed_pixmap = result_pixmap # Cập nhật pixmap cuối cùng
                             processed_count += 1
                        else:
                            self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau khi đổi định dạng", file_path)


            except Exception as e:
                print(f"Lỗi khi đổi định dạng file {base_name} sang {target_format}: {e}")
                self.add_result_item(None, base_name, f"LỖI đổi sang .{target_format.upper()}: {e}", file_path)

        # Hiển thị ảnh cuối cùng đã xử lý lên khung chính
        if last_processed_pixmap and not last_processed_pixmap.isNull():
             scaled_last = last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.current_image_label.setPixmap(scaled_last)

        self._finish_processing("thay đổi định dạng", processed_count)

    # --- Hàm xử lý nhận diện phương tiện --- <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW FUNCTION
    def process_vehicle_detection(self):
        """Thực hiện nhận diện phương tiện trong ảnh."""
        if not self._prepare_processing("nhận diện phương tiện"):
            return

        processed_count = 0
        last_processed_pixmap = None # Lưu pixmap cuối cùng để hiển thị

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}_detected{ext_part}" # Đuôi file giữ nguyên
            output_path = os.path.join(self.output_directory, output_filename)

            self.status_label.setText(f"Đang nhận diện phương tiện: {base_name} ({i+1}/{len(self.files)})")
            QApplication.processEvents()

            try:
                # Đọc ảnh bằng OpenCV
                img_cv = cv2.imread(file_path)
                if img_cv is None:
                    raise ValueError("OpenCV không thể đọc file ảnh.")

                # Hiển thị ảnh gốc trước khi xử lý
                original_pixmap = cv_image_to_qpixmap(img_cv)
                if not original_pixmap.isNull():
                    scaled_original = original_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.current_image_label.setPixmap(scaled_original)
                QApplication.processEvents()

                # --- Thực hiện nhận diện ---
                results = self.yolo_model(img_cv, verbose=False) # verbose=False để không in log của YOLO ra console

                detection_info = [] # Lưu thông tin các đối tượng phát hiện được

                # Lặp qua các kết quả phát hiện
                for result in results:
                    boxes = result.boxes  # Đối tượng Boxes chứa thông tin bounding box
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Tọa độ bounding box
                        conf = float(box.conf[0]) # Độ tin cậy
                        cls = int(box.cls[0]) # ID của lớp
                        class_name = self.yolo_model.names[cls] # Lấy tên lớp từ ID

                        # Chỉ xử lý các lớp phương tiện mục tiêu và có độ tin cậy đủ cao
                        if class_name in self.target_vehicle_classes and conf > 0.4: # Ngưỡng tin cậy 0.4
                            # Lấy tên tiếng Việt
                            class_name_vn = self.vehicle_classes_vn.get(class_name, class_name) # Mặc định giữ tên gốc nếu không có trong dict
                            label = f"{class_name_vn}: {conf:.2f}"
                            detection_info.append(label) # Thêm thông tin vào list

                            # --- Vẽ bounding box và label lên ảnh ---
                            # Chọn màu ngẫu nhiên cho mỗi lớp (hoặc định nghĩa màu cố định)
                            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            color = (0, 255, 0) # Màu xanh lá cây cho dễ nhìn

                            # Vẽ bounding box
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

                            # Tính toán vị trí và kích thước của text label
                            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                            # Vẽ nền cho label
                            cv2.rectangle(img_cv, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1) # -1 để tô đầy nền

                            # Vẽ text label (màu trắng để nổi bật trên nền)
                            cv2.putText(img_cv, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


                # --- Lưu ảnh đã vẽ ---
                cv2.imwrite(output_path, img_cv)

                # --- Hiển thị kết quả ---
                result_pixmap = cv_image_to_qpixmap(img_cv)
                if not result_pixmap.isNull():
                    info = f"Đã nhận diện {len(detection_info)} phương tiện." if detection_info else "Không phát hiện phương tiện nào."
                    self.add_result_item(result_pixmap, base_name, info, output_path)
                    last_processed_pixmap = result_pixmap # Cập nhật pixmap cuối cùng
                    processed_count += 1
                else:
                    self.add_result_item(None, base_name, f"LỖI: Không thể tạo ảnh xem trước sau khi nhận diện", file_path)


            except Exception as e:
                print(f"Lỗi khi nhận diện phương tiện trong file {base_name}: {e}")
                self.add_result_item(None, base_name, f"LỖI nhận diện: {e}", file_path) # Thêm thông báo lỗi

        # Hiển thị ảnh cuối cùng đã xử lý lên khung chính
        if last_processed_pixmap and not last_processed_pixmap.isNull():
             scaled_last = last_processed_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.current_image_label.setPixmap(scaled_last)

        self._finish_processing("nhận diện phương tiện", processed_count)
    # --- Kết thúc hàm xử lý nhận diện phương tiện ---


    def add_result_item(self, pixmap, original_name, info_text, result_path):
        """Thêm một mục kết quả vào khu vực hiển thị."""
        item_widget = QWidget()
        item_layout = QVBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 5, 5, 5)
        item_layout.setSpacing(3)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumSize(180, 120) # Kích thước cố định cho ảnh kết quả
        img_label.setStyleSheet("background-color: #eee; border: 1px solid #ddd;")

        if pixmap and not pixmap.isNull():
             scaled_pixmap = pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             img_label.setPixmap(scaled_pixmap)
             item_widget.setToolTip(f"File gốc: {original_name}\nKết quả: {os.path.basename(result_path)}\n{info_text}")
        else:
             img_label.setText("Lỗi xử lý")
             img_label.setStyleSheet("background-color: #ffebee; border: 1px solid #e57373; color: #c62828;")
             item_widget.setToolTip(f"File gốc: {original_name}\nLỗi: {info_text}")


        info_label = QLabel(f"{original_name}\n{info_text}")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 9pt; color: #333;")

        item_layout.addWidget(img_label)
        item_layout.addWidget(info_label)
        item_widget.setStyleSheet("background-color: white; border: 1px solid #eee; margin-bottom: 5px; border-radius: 3px;")

        # Xóa placeholder nếu đây là kết quả đầu tiên
        if self.results_layout.count() == 1 and isinstance(self.results_layout.itemAt(0).widget(), QLabel) and "Kết quả xử lý sẽ hiển thị ở đây" in self.results_layout.itemAt(0).widget().text():
             item_to_remove = self.results_layout.takeAt(0)
             if item_to_remove.widget():
                 item_to_remove.widget().deleteLater()

        self.results_layout.addWidget(item_widget)


    def clear_results(self):
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        placeholder_result = QLabel("Kết quả xử lý sẽ hiển thị ở đây")
        placeholder_result.setAlignment(Qt.AlignCenter)
        placeholder_result.setStyleSheet("color: #7f8c8d; font-size: 10pt; padding: 20px; background-color: transparent; border: none;")
        self.results_layout.addWidget(placeholder_result)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Tự động scale lại ảnh preview khi thay đổi kích thước cửa sổ
        if hasattr(self, 'current_image_label') and self.current_image_label:
            current_pixmap_orig = getattr(self.current_image_label, '_original_pixmap', None) # Lấy pixmap gốc nếu có
            if current_pixmap_orig and not current_pixmap_orig.isNull():
                 scaled_pixmap = current_pixmap_orig.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 if scaled_pixmap and not scaled_pixmap.isNull():
                     # Chỉ set lại nếu size khác để tránh vòng lặp vô hạn (dù không chắc cần thiết)
                     # if not self.current_image_label.pixmap() or scaled_pixmap.size() != self.current_image_label.pixmap().size():
                          self.current_image_label.setPixmap(scaled_pixmap)
            elif self.current_image_label.pixmap() and not self.current_image_label.pixmap().isNull():
                 # Fallback nếu không có _original_pixmap, scale từ pixmap hiện tại
                 current_pixmap = self.current_image_label.pixmap()
                 scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 if scaled_pixmap and not scaled_pixmap.isNull():
                     # Chỉ set lại nếu size khác
                     # if scaled_pixmap.size() != current_pixmap.size():
                          self.current_image_label.setPixmap(scaled_pixmap)

    # Override display_selected_image để lưu pixmap gốc
    def display_selected_image(self, current_item, previous_item):
        if current_item:
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)

            if full_path and os.path.exists(full_path):
                try:
                    img_cv = cv2.imread(full_path)
                    if img_cv is None:
                        raise ValueError("OpenCV không thể đọc file ảnh.")

                    pixmap = cv_image_to_qpixmap(img_cv)

                    if not pixmap.isNull():
                        self.current_image_label._original_pixmap = pixmap # Lưu pixmap gốc
                        scaled_pixmap = pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.current_image_label.setPixmap(scaled_pixmap)
                        self.status_label.setText(f"Xem trước: {file_name}")
                    else:
                        self.current_image_label.clear()
                        self.current_image_label.setText(f"Lỗi chuyển đổi ảnh:\n{file_name}")
                        self.status_label.setText(f"Lỗi xem trước: {file_name}")
                        if hasattr(self.current_image_label, '_original_pixmap'):
                           del self.current_image_label._original_pixmap # Xóa pixmap gốc nếu lỗi

                except Exception as e:
                    print(f"[ERROR display_selected_image] Error opening/processing image {file_name}: {e}")
                    self.current_image_label.clear()
                    self.current_image_label.setText(f"Lỗi khi mở ảnh:\n{file_name}\n({e})")
                    self.status_label.setText(f"Lỗi xem trước: {file_name}")
                    if hasattr(self.current_image_label, '_original_pixmap'):
                       del self.current_image_label._original_pixmap # Xóa pixmap gốc nếu lỗi
            else:
                self.current_image_label.clear()
                self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
                if hasattr(self.current_image_label, '_original_pixmap'):
                   del self.current_image_label._original_pixmap # Xóa pixmap gốc nếu lỗi
        else:
             self.current_image_label.clear()
             self.current_image_label.setText("Chọn một file để xem trước")
             if hasattr(self.current_image_label, '_original_pixmap'):
                del self.current_image_label._original_pixmap # Xóa pixmap gốc


if __name__ == '__main__':
    # Kiểm tra và import các thư viện cần thiết
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


    if missing_libs:
        print("Lỗi: Thiếu các thư viện cần thiết.")
        print("Vui lòng cài đặt bằng pip:")
        if "opencv-python" in missing_libs: print("  pip install opencv-python")
        if "PyQt5" in missing_libs: print("  pip install PyQt5")
        if "Pillow" in missing_libs: print("  pip install Pillow")
        if "ultralytics" in missing_libs: print("  pip install ultralytics")
        if "torch (pytorch)" in missing_libs: print("  pip install torch torchvision torchaudio")
        if "numpy" in missing_libs: print("  pip install numpy")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = BatchModeAppRedesigned()
    window.show()
    sys.exit(app.exec_())
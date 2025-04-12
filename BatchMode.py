import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame, QMessageBox, QComboBox) # Thêm QMessageBox, QComboBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap
from PIL import Image # Thêm thư viện Pillow

# Helper function to convert Pillow Image to QPixmap
def pil_to_qpixmap(pil_image):
    """Converts a Pillow Image object to a QPixmap."""
    try:
        # Convert RGBA or P mode images with transparency to RGBA
        if pil_image.mode == "RGBA" or (pil_image.mode == "P" and 'transparency' in pil_image.info):
             img_byte_arr = pil_image.copy().convert("RGBA").tobytes("raw", "RGBA")
             qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        # Convert other modes to RGB first
        else:
             img_byte_arr = pil_image.copy().convert("RGB").tobytes("raw", "RGB")
             qimage = QImage(img_byte_arr, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)

        return QPixmap.fromImage(qimage)
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        return QPixmap() # Return empty pixmap on error


class BatchModeAppRedesigned(QMainWindow):
    def __init__(self):
        super().__init__()
        self.files = []
        self.output_directory = None # Thêm biến lưu đường dẫn thư mục lưu
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng biển số')
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Header ---
        # (Giữ nguyên hoặc bỏ qua như trước)

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
        # ... (giữ nguyên các cài đặt layout của resize_layout) ...
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
        process_buttons_layout = QVBoxLayout() # Layout dọc cho 2 nút xử lý
        process_buttons_layout.setSpacing(8)

        self.resize_button = QPushButton("Xử lý Kích thước")
        self.resize_button.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.resize_button.setMinimumSize(160, 40)
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
        self.format_button.setMinimumSize(160, 40)
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
        self.status_label = QLabel("Sẵn sàng")
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

            # Kích hoạt các nút xử lý nếu có file và đã chọn thư mục lưu
            can_process = self.file_list_widget.count() > 0 and self.output_directory is not None
            self.resize_button.setEnabled(can_process)
            self.format_button.setEnabled(can_process)
        else:
            self.status_label.setText("Không có file nào được chọn.")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if directory:
            self.output_directory = directory
            self.output_dir_label.setText(f"Lưu tại: {directory}")
            self.output_dir_label.setStyleSheet("font-style: normal; color: #27ae60;") # Đổi style khi đã chọn
            self.status_label.setText(f"Đã chọn thư mục lưu: {directory}")
             # Kích hoạt các nút xử lý nếu có file và đã chọn thư mục lưu
            can_process = self.file_list_widget.count() > 0 and self.output_directory is not None
            self.resize_button.setEnabled(can_process)
            self.format_button.setEnabled(can_process)
        else:
            self.status_label.setText("Chưa chọn thư mục lưu.")
            # Giữ nút bị vô hiệu hóa nếu chưa chọn thư mục
            self.resize_button.setEnabled(False)
            self.format_button.setEnabled(False)


    def display_selected_image(self, current_item, previous_item):
        if current_item:
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)

            if full_path and os.path.exists(full_path):
                try:
                    # Dùng Pillow để mở ảnh, an toàn hơn cho nhiều định dạng
                    with Image.open(full_path) as img:
                        pixmap = pil_to_qpixmap(img) # Chuyển PIL Image sang QPixmap
                        if not pixmap.isNull():
                             scaled_pixmap = pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                             self.current_image_label.setPixmap(scaled_pixmap)
                             self.status_label.setText(f"Xem trước: {file_name}")
                        else:
                             raise ValueError("Conversion to QPixmap failed.")
                except Exception as e:
                    print(f"Error loading image {file_name} with Pillow: {e}")
                    self.current_image_label.setText(f"Lỗi khi tải ảnh:\n{file_name}\n({e})")
                    self.status_label.setText(f"Lỗi khi xem trước: {file_name}")
            else:
                self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi không tìm thấy file: {file_name}")
        else:
             self.current_image_label.setText("Chọn một file để xem trước")

    def _prepare_processing(self, operation_name):
        """Hàm trợ giúp kiểm tra điều kiện và chuẩn bị trước khi xử lý."""
        if not self.files:
            QMessageBox.warning(self, "Thiếu file", "Vui lòng thêm file ảnh trước khi xử lý.")
            return False
        if not self.output_directory:
            QMessageBox.warning(self, "Thiếu thư mục lưu", "Vui lòng chọn thư mục lưu kết quả trước khi xử lý.")
            return False

        self.status_label.setText(f"Bắt đầu {operation_name}...")
        self.set_buttons_enabled(False) # Vô hiệu hóa các nút
        self.clear_results() # Xóa kết quả cũ
        QApplication.processEvents() # Cập nhật giao diện
        return True

    def _finish_processing(self, operation_name, count):
        """Hàm trợ giúp kết thúc xử lý."""
        self.status_label.setText(f"Hoàn tất {operation_name} {count} file.")
        self.set_buttons_enabled(True) # Kích hoạt lại các nút

    def set_buttons_enabled(self, enabled):
        """Bật/tắt các nút và danh sách file."""
        can_process = enabled and self.file_list_widget.count() > 0 and self.output_directory is not None
        self.resize_button.setEnabled(can_process)
        self.format_button.setEnabled(can_process)
        self.add_files_button.setEnabled(enabled)
        self.select_output_dir_button.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)

    def process_resize(self):
        """Thực hiện thay đổi kích thước ảnh."""
        if not self._prepare_processing("thay đổi kích thước"):
            return

        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        processed_count = 0

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, ext_part = os.path.splitext(base_name)
            # Giữ nguyên định dạng gốc khi chỉ thay đổi kích thước
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

                    # Thực hiện resize
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS) # Dùng LANCZOS cho chất lượng tốt
                    # Lưu ảnh đã resize (Pillow tự xác định định dạng từ phần mở rộng)
                    resized_img.save(output_path)

                    # Hiển thị kết quả
                    result_pixmap = pil_to_qpixmap(resized_img)
                    if not result_pixmap.isNull():
                         self.add_result_item(result_pixmap, base_name, f"Đã đổi kích thước thành {target_width}x{target_height}", output_path)
                    processed_count += 1

            except Exception as e:
                print(f"Lỗi khi đổi kích thước file {base_name}: {e}")
                self.add_result_item(None, base_name, f"LỖI: {e}", file_path) # Thêm thông báo lỗi vào kết quả

        self._finish_processing("thay đổi kích thước", processed_count)

    def process_format_change(self):
        """Thực hiện thay đổi định dạng ảnh."""
        if not self._prepare_processing("thay đổi định dạng"):
            return

        target_format = self.format_combo.currentText().lower() # Lấy định dạng chọn và chuyển thành chữ thường
        processed_count = 0

        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            name_part, _ = os.path.splitext(base_name)
            # Tạo tên file mới với định dạng đã chọn
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

                    # Lưu ảnh với định dạng mới
                    # Pillow xử lý chuyển đổi ngầm khi lưu nếu cần
                    img.save(output_path, format=target_format.upper()) # Chỉ định rõ format khi lưu

                    # Hiển thị kết quả (vẫn là ảnh gốc nhưng đã lưu với format mới)
                    # Mở lại file vừa lưu để chắc chắn
                    with Image.open(output_path) as saved_img:
                        result_pixmap = pil_to_qpixmap(saved_img)
                        if not result_pixmap.isNull():
                             self.add_result_item(result_pixmap, base_name, f"Đã đổi định dạng thành .{target_format.upper()}", output_path)
                        processed_count += 1

            except Exception as e:
                print(f"Lỗi khi đổi định dạng file {base_name} sang {target_format}: {e}")
                # Có thể file gốc không hỗ trợ lưu sang định dạng đích (ví dụ GIF nhiều frame sang JPG)
                self.add_result_item(None, base_name, f"LỖI đổi sang .{target_format.upper()}: {e}", file_path)

        self._finish_processing("thay đổi định dạng", processed_count)

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
             # Scale ảnh kết quả để vừa với QLabel
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
        if hasattr(self, 'current_image_label') and self.current_image_label:
            current_pixmap = self.current_image_label.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                 scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 if scaled_pixmap and not scaled_pixmap.isNull():
                      if scaled_pixmap.size() != self.current_image_label.pixmap().size():
                          self.current_image_label.setPixmap(scaled_pixmap)

# Cần thêm import QImage từ PyQt5.QtGui
from PyQt5.QtGui import QImage

if __name__ == '__main__':
    # Cần thêm import Image từ PIL
    try:
        from PIL import Image
    except ImportError:
        print("Thư viện Pillow chưa được cài đặt. Vui lòng cài đặt: pip install Pillow")
        sys.exit(1)

    # Cần thêm import os và sys nếu chưa có ở đầu file
    import os
    import sys

    app = QApplication(sys.argv)
    # Cần thêm import QImage và pil_to_qpixmap nếu chúng được định nghĩa trong class
    from PyQt5.QtGui import QImage
    window = BatchModeAppRedesigned()
    window.show()
    sys.exit(app.exec_())
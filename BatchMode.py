import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QFileDialog, QSpinBox, QFormLayout,
                             QGroupBox, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap # Thêm QPixmap để hiển thị ảnh

class BatchModeAppRedesigned(QMainWindow):
    def __init__(self):
        super().__init__()
        # Giữ lại danh sách file từ code gốc nếu cần
        self.files = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Xử lý ảnh hàng loạt - Nhận dạng biển số')
        # self.setWindowFlag(Qt.FramelessWindowHint) # Bỏ comment nếu muốn không có viền cửa sổ
        self.showFullScreen() # Hiển thị toàn màn hình hoặc kích thước lớn

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) # Thêm khoảng cách viền
        main_layout.setSpacing(15) # Thêm khoảng cách giữa các widget

        # --- Header (Tùy chọn - có thể thêm nút Back/Close như code gốc nếu muốn) ---
        header_layout = QHBoxLayout()
        title = QLabel('Xử Lý Ảnh Hàng Loạt')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI', 24, QFont.Bold))
        title.setStyleSheet('color: #2c3e50;')
        header_layout.addWidget(title)
        # main_layout.addLayout(header_layout) # Bỏ comment nếu dùng header

        # --- Khu vực chính: Chia 2 cột lớn ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        # --- Cột bên trái ---
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(10)

        # Danh sách file (chiếm phần lớn bên trái)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: #ecf0f1;
                font-size: 11pt;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        # Kết nối sự kiện chọn item để hiển thị ảnh (nếu cần)
        # self.file_list_widget.currentItemChanged.connect(self.display_selected_image)
        left_column_layout.addWidget(self.file_list_widget, stretch=1) # stretch=1 để list dãn ra

        # Nút thêm file ảnh (phía dưới danh sách)
        self.add_files_button = QPushButton("Thêm file ảnh")
        self.add_files_button.setFont(QFont('Segoe UI', 11))
        self.add_files_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.add_files_button.clicked.connect(self.add_files)
        left_column_layout.addWidget(self.add_files_button, alignment=Qt.AlignLeft) # Căn trái

        # --- Cột bên phải ---
        right_column_layout = QVBoxLayout()
        right_column_layout.setSpacing(10)

        # Khung hiển thị ảnh đang xử lý (chiếm phần lớn bên phải)
        self.current_image_label = QLabel("Khung hiển thị ảnh đang xử lý")
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setFrameShape(QFrame.Box) # Thêm viền để dễ thấy
        self.current_image_label.setFrameShadow(QFrame.Sunken)
        self.current_image_label.setMinimumSize(400, 300) # Kích thước tối thiểu
        self.current_image_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 1px dashed #bdc3c7;
                color: #7f8c8d;
                font-size: 12pt;
            }
        """)
        right_column_layout.addWidget(self.current_image_label, stretch=1) # stretch=1 để khung ảnh dãn ra

        # Khung hiển thị các ảnh sau khi nhận diện (phía dưới khung ảnh đang xử lý)
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.Box)
        self.results_scroll_area.setFrameShadow(QFrame.Sunken)
        self.results_scroll_area.setMinimumHeight(150) # Chiều cao tối thiểu
        self.results_scroll_area.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px;")

        results_container_widget = QWidget()
        self.results_layout = QVBoxLayout(results_container_widget) # Layout để chứa các ảnh kết quả
        self.results_layout.setAlignment(Qt.AlignTop) # Căn các ảnh kết quả lên trên
        self.results_layout.setSpacing(5)

        # Thêm placeholder cho khu vực kết quả
        placeholder_result = QLabel("Khung hiển thị các ảnh đã nhận diện")
        placeholder_result.setAlignment(Qt.AlignCenter)
        placeholder_result.setStyleSheet("color: #7f8c8d; font-size: 10pt; padding: 20px;")
        self.results_layout.addWidget(placeholder_result)

        self.results_scroll_area.setWidget(results_container_widget)
        right_column_layout.addWidget(self.results_scroll_area, stretch=0) # Không cho dãn quá nhiều theo chiều dọc

        # Thêm 2 cột vào layout nội dung chính
        content_layout.addLayout(left_column_layout, stretch=1) # Cột trái chiếm 1 phần
        content_layout.addLayout(right_column_layout, stretch=2) # Cột phải chiếm 2 phần (rộng hơn)

        # Thêm layout nội dung vào layout chính
        main_layout.addLayout(content_layout, stretch=1) # Cho phép khu vực này dãn ra

        # --- Khu vực điều khiển (phía dưới) ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        # Nhóm thay đổi kích thước
        resize_group = QGroupBox("Thay đổi kích thước ảnh (Tùy chọn)")
        resize_group.setFont(QFont('Segoe UI', 10))
        resize_layout = QFormLayout(resize_group)
        resize_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        resize_layout.setLabelAlignment(Qt.AlignLeft)
        resize_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        resize_layout.setHorizontalSpacing(10)
        resize_layout.setVerticalSpacing(8)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(800)
        self.width_spin.setSuffix(" px")
        self.width_spin.setMinimumWidth(80)


        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(600)
        self.height_spin.setSuffix(" px")
        self.height_spin.setMinimumWidth(80)

        resize_layout.addRow("Chiều rộng:", self.width_spin)
        resize_layout.addRow("Chiều cao:", self.height_spin)

        controls_layout.addWidget(resize_group)
        controls_layout.addStretch(1) # Đẩy nút xử lý sang phải

        # Nút xử lý ảnh
        self.process_button = QPushButton("Xử lý ảnh")
        self.process_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.process_button.setMinimumSize(150, 45) # Kích thước tối thiểu
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled { /* Style khi nút bị vô hiệu hóa */
                background-color: #95a5a6;
            }
        """)
        self.process_button.clicked.connect(self.start_processing) # Kết nối với hàm xử lý
        controls_layout.addWidget(self.process_button, alignment=Qt.AlignRight) # Căn phải

        # Thêm khu vực điều khiển vào layout chính
        main_layout.addLayout(controls_layout, stretch=0) # Không cho khu vực này dãn

        # Thiết lập style chung (nếu cần)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f8f9fa; /* Màu nền nhẹ nhàng */
                font-family: Segoe UI;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px; /* Khoảng cách phía trên GroupBox */
                padding: 15px 10px 10px 10px; /* Top Left Bottom Right */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* Đưa title lên góc trái */
                padding: 0 5px;
                left: 10px; /* Dịch title sang phải một chút */
                color: #2c3e50;
            }
            QSpinBox {
                padding: 3px 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QLabel {
                 color: #2c3e50; /* Màu chữ mặc định */
            }
        """)

    def add_files(self):
        # Sử dụng lại hàm add_files từ code gốc của bạn
        # Filter chỉ cho chọn file ảnh phổ biến
        image_filter = "Ảnh (*.jpg *.jpeg *.png *.bmp *.gif);;Tất cả file (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file ảnh", "", image_filter)
        if files:
            added_count = 0
            for file in files:
                if file not in self.files: # Tránh thêm trùng
                    self.files.append(file)
                    self.file_list_widget.addItem(os.path.basename(file)) # Chỉ hiển thị tên file
                    added_count += 1
            if added_count > 0:
                self.status_label.setText(f"Đã thêm {added_count} file ảnh.")
            else:
                 self.status_label.setText("Các file đã chọn đã có trong danh sách.")
            # Kích hoạt nút xử lý nếu có file
            self.process_button.setEnabled(self.file_list_widget.count() > 0)
        else:
            self.status_label.setText("Không có file nào được chọn.")

    # --- Placeholder cho các hàm xử lý ---
    def display_selected_image(self, current_item, previous_item):
        # Hàm này sẽ được gọi khi chọn một file trong danh sách
        # Bạn cần lấy đường dẫn file từ current_item và hiển thị lên self.current_image_label
        if current_item:
            # Tìm đường dẫn đầy đủ tương ứng với tên file hiển thị
            file_name = current_item.text()
            full_path = next((f for f in self.files if os.path.basename(f) == file_name), None)

            if full_path and os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                # Resize ảnh để vừa với QLabel mà vẫn giữ tỉ lệ
                scaled_pixmap = pixmap.scaled(self.current_image_label.size(),
                                              Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                self.current_image_label.setPixmap(scaled_pixmap)
                self.status_label.setText(f"Đang hiển thị: {file_name}")
            else:
                self.current_image_label.setText(f"Không tìm thấy file:\n{file_name}")
                self.status_label.setText(f"Lỗi khi hiển thị: {file_name}")
        else:
             self.current_image_label.setText("Chọn một file để xem trước")


    def start_processing(self):
        # Hàm này sẽ được gọi khi nhấn nút "Xử lý ảnh"
        if not self.files:
            self.status_label.setText("Vui lòng thêm file ảnh trước khi xử lý.")
            return

        self.status_label.setText("Bắt đầu xử lý...")
        self.process_button.setEnabled(False) # Vô hiệu hóa nút khi đang xử lý
        self.file_list_widget.setEnabled(False) # Ngăn chọn file khác khi đang xử lý

        # --- !!! LOGIC XỬ LÝ ẢNH CỦA BẠN SẼ ĐẶT Ở ĐÂY !!! ---
        # 1. Lặp qua danh sách self.files
        # 2. Với mỗi file:
        #    a. Đọc ảnh (ví dụ: dùng OpenCV - cv2.imread(file_path))
        #    b. (Tùy chọn) Hiển thị ảnh đang xử lý lên self.current_image_label
        #       pixmap = QPixmap(file_path)
        #       scaled_pixmap = pixmap.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #       self.current_image_label.setPixmap(scaled_pixmap)
        #       QApplication.processEvents() # Cập nhật giao diện ngay lập tức
        #    c. Thực hiện các bước xử lý ảnh (phát hiện biển số, nhận dạng)
        #    d. Lấy ảnh kết quả (ảnh gốc có vẽ bounding box quanh biển số)
        #    e. Hiển thị ảnh kết quả vào self.results_layout
        #       - Xóa placeholder nếu là ảnh đầu tiên
        #       - Tạo QLabel mới
        #       - Convert ảnh kết quả (numpy array từ OpenCV) sang QPixmap
        #       - Đặt QPixmap vào QLabel
        #       - Thêm QLabel vào self.results_layout
        #    f. (Tùy chọn) Cập nhật tiến trình nếu có progress bar
        # 3. Sau khi hoàn tất:
        #    - self.status_label.setText("Xử lý hoàn tất!")
        #    - self.process_button.setEnabled(True) # Kích hoạt lại nút
        #    - self.file_list_widget.setEnabled(True)

        # Ví dụ giả lập xử lý và thêm kết quả (thay thế bằng logic thật):
        self.clear_results() # Xóa kết quả cũ trước khi bắt đầu
        for i, file_path in enumerate(self.files):
            base_name = os.path.basename(file_path)
            self.status_label.setText(f"Đang xử lý: {base_name} ({i+1}/{len(self.files)})")

            # Giả lập hiển thị ảnh đang xử lý
            pixmap_current = QPixmap(file_path)
            if not pixmap_current.isNull():
                 scaled_pixmap_current = pixmap_current.scaled(self.current_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.current_image_label.setPixmap(scaled_pixmap_current)
            else:
                 self.current_image_label.setText(f"Lỗi tải ảnh:\n{base_name}")

            QApplication.processEvents() # Quan trọng để cập nhật UI

            # --- Giả lập xử lý (thay bằng code thật) ---
            # time.sleep(0.5) # Giả lập thời gian xử lý
            result_text = f"Biển số nhận dạng (giả): XXX-{i+1:03d}" # Kết quả giả

            # --- Giả lập hiển thị kết quả ---
            # Lấy ảnh gốc làm ảnh kết quả (thay bằng ảnh có bounding box)
            pixmap_result = QPixmap(file_path)
            if not pixmap_result.isNull():
                 result_label = QLabel()
                 # Resize ảnh kết quả nhỏ hơn để hiển thị nhiều ảnh
                 scaled_pixmap_result = pixmap_result.scaled(QSize(200, 150), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 result_label.setPixmap(scaled_pixmap_result)
                 result_label.setToolTip(f"{base_name}\n{result_text}") # Hiển thị tooltip khi hover

                 # Thêm cả text kết quả bên dưới ảnh (tùy chọn)
                 info_label = QLabel(f"{base_name}\n{result_text}")
                 info_label.setAlignment(Qt.AlignCenter)
                 info_label.setWordWrap(True)

                 item_layout = QVBoxLayout() # Layout cho mỗi mục kết quả
                 item_layout.addWidget(result_label, alignment=Qt.AlignCenter)
                 item_layout.addWidget(info_label)

                 item_widget = QWidget() # Widget chứa layout của mục kết quả
                 item_widget.setLayout(item_layout)
                 item_widget.setStyleSheet("border: 1px solid #eee; margin-bottom: 5px; background-color: white; border-radius: 3px;")


                 self.results_layout.addWidget(item_widget)
            # -------------------------------------

        self.status_label.setText(f"Hoàn tất xử lý {len(self.files)} file.")
        self.process_button.setEnabled(True)
        self.file_list_widget.setEnabled(True)

    def clear_results(self):
        # Xóa các widget kết quả cũ trong layout
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Thêm lại placeholder
        placeholder_result = QLabel("Khung hiển thị các ảnh đã nhận diện")
        placeholder_result.setAlignment(Qt.AlignCenter)
        placeholder_result.setStyleSheet("color: #7f8c8d; font-size: 10pt; padding: 20px;")
        self.results_layout.addWidget(placeholder_result)


    def keyPressEvent(self, event):
        # Giữ lại chức năng đóng bằng ESC từ code gốc
        if event.key() == Qt.Key_Escape:
            self.close()

    # --- Override resizeEvent để cập nhật ảnh đang hiển thị khi thay đổi kích thước cửa sổ ---
    def resizeEvent(self, event):
        # Gọi hàm resizeEvent của lớp cha trước
        super().resizeEvent(event)

        # Kiểm tra xem self.current_image_label đã được khởi tạo chưa
        if hasattr(self, 'current_image_label') and self.current_image_label:
            # Lấy pixmap hiện tại từ label
            current_pixmap = self.current_image_label.pixmap()

            # Chỉ thực hiện thay đổi kích thước nếu có pixmap hợp lệ
            if current_pixmap and not current_pixmap.isNull():
                 # Scale lại pixmap để vừa với kích thước mới của label
                 scaled_pixmap = current_pixmap.scaled(self.current_image_label.size(),
                                                      Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)
                 # Kiểm tra lại lần nữa trước khi setPixmap
                 if scaled_pixmap and not scaled_pixmap.isNull():
                      # Đặt pixmap đã scale lại vào label
                      # Quan trọng: Chỉ setPixmap nếu pixmap mới khác pixmap cũ về kích thước, tránh vòng lặp vô hạn tiềm ẩn
                      if scaled_pixmap.size() != self.current_image_label.pixmap().size():
                          self.current_image_label.setPixmap(scaled_pixmap)
        # Nếu current_image_label chưa tồn tại hoặc không có pixmap, không làm gì cả


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BatchModeAppRedesigned()
    window.show()
    sys.exit(app.exec_())
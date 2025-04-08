import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QListWidget, QComboBox, QProgressBar, QFileDialog,
                             QSpinBox, QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class BatchModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.files = []
        self.processing = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Batch Processing')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        # Widget chính
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Header với nút quay lại và nút đóng
        header_layout = QHBoxLayout()

        back_button = QPushButton("← Quay lại")
        back_button.setFont(QFont('Segoe UI', 12))
        back_button.setStyleSheet("""
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
""")
        back_button.clicked.connect(self.close)

        title = QLabel('Batch Processing')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))
        title.setStyleSheet('color: #2ecc71;')

        close_button = QPushButton("X")
        close_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        close_button.setFixedSize(40, 40)
        close_button.setStyleSheet("""
QPushButton {
    background-color: #e74c3c;
    color: white;
    border-radius: 20px;
}
QPushButton:hover {
    background-color: #c0392b;
}
""")
        close_button.clicked.connect(self.close)

        header_layout.addWidget(back_button)
        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(close_button)

        # Khu vực chính - chia 2 cột
        content_layout = QHBoxLayout()

        # Panel bên trái - Danh sách file
        left_panel = QVBoxLayout()

        file_group = QGroupBox("Danh sách file")
        file_layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
QListWidget {
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 5px;
    background: white;
    color: #333;
}
""")

        file_buttons = QHBoxLayout()

        add_button = QPushButton("Thêm file")
        add_button.setStyleSheet("""
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 3px;
    padding: 5px 10px;
}
""")
        add_button.clicked.connect(self.add_files)

        clear_button = QPushButton("Xóa tất cả")
        clear_button.setStyleSheet("""
QPushButton {
    background-color: #e74c3c;
    color: white;
    border-radius: 3px;
    padding: 5px 10px;
}
""")
        clear_button.clicked.connect(self.clear_files)

        file_buttons.addWidget(add_button)
        file_buttons.addWidget(clear_button)

        file_layout.addWidget(self.file_list)
        file_layout.addLayout(file_buttons)
        file_group.setLayout(file_layout)

        # Panel bên phải - Các thiết lập
        right_panel = QVBoxLayout()

        settings_group = QGroupBox("Thiết lập xử lý")
        settings_layout = QFormLayout()

        self.action_combo = QComboBox()
        self.action_combo.addItems(["Thay đổi kích thước", "Chuyển định dạng", "Thêm watermark", "Tối ưu hóa"])

        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG", "GIF", "BMP", "Giữ nguyên"])

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 5000)
        self.width_spin.setValue(800)

        self.height_spin = QSpinBox()
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

        # Thêm widget chính vào cửa sổ
        self.setCentralWidget(central_widget)

        # Thiết lập style chung
        self.setStyleSheet("""
QMainWindow, QWidget {
    background-color: white;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-top: 15px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}
""")

    def add_files(self):
        self.file_list.addItem("example_image1.jpg")
        self.file_list.addItem("example_image2.png")
        self.file_list.addItem("example_image3.jpg")
        self.status_label.setText("Đã thêm 3 file (chế độ giả lập)")

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
            self.processing = False
            self.process_button.setText("Bắt đầu xử lý")
            if hasattr(self, 'timer'):
                self.timer.stop()

    def update_progress(self):
        current = self.progress_bar.value()
        if current < 100:
            self.progress_bar.setValue(current + 1)
            self.status_label.setText(f"Đang xử lý: {current + 1}%")
        else:
            self.timer.stop()
            self.processing = False
            self.process_button.setText("Bắt đầu xử lý")
            self.status_label.setText("Xử lý hoàn tất!")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BatchModeApp()
    sys.exit(app.exec_())
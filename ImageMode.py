import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QFileDialog, QSlider, QComboBox, QToolBar, QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap


class ImageModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Editor')
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
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
}
QPushButton:hover {
    background-color: #2980b9;
}
""")
        back_button.clicked.connect(self.close)

        title = QLabel('Image Editor')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))
        title.setStyleSheet('color: #3498db;')

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

        # Toolbar cho các chức năng chỉnh sửa
        toolbar_layout = QHBoxLayout()

        # Nút mở file
        open_button = QPushButton("Mở ảnh")
        open_button.setStyleSheet("""
QPushButton {
    background-color: #3498db;
    color: white;
    padding: 8px 15px;
    border-radius: 4px;
}
""")
        open_button.clicked.connect(self.open_image)

        # Nút lưu
        save_button = QPushButton("Lưu ảnh")
        save_button.setStyleSheet("""
QPushButton {
    background-color: #2ecc71;
    color: white;
    padding: 8px 15px;
    border-radius: 4px;
}
""")

        # Bộ lọc
        filter_label = QLabel("Bộ lọc:")
        filter_combo = QComboBox()
        filter_combo.addItems(["Bình thường", "Đen trắng", "Mờ", "Độ sáng"])
        filter_combo.setStyleSheet("""
QComboBox {
    padding: 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
}
""")

        # Độ sáng
        brightness_label = QLabel("Độ sáng:")
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(0, 100)
        brightness_slider.setValue(50)
        brightness_slider.setFixedWidth(150)

        # Thêm vào toolbar
        toolbar_layout.addWidget(open_button)
        toolbar_layout.addWidget(save_button)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(filter_label)
        toolbar_layout.addWidget(filter_combo)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(brightness_label)
        toolbar_layout.addWidget(brightness_slider)
        toolbar_layout.addStretch()

        # Khu vực hiển thị ảnh
        self.image_label = QLabel("Chưa có ảnh nào được mở. Nhấn 'Mở ảnh' để bắt đầu.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
QLabel {
    background-color: #f5f5f5;
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    font-size: 16px;
    color: #777;
}
""")

        # Thêm các thành phần vào layout chính
        main_layout.addLayout(header_layout)
        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.image_label, 1)

        # Status bar
        status_bar = QLabel("Sẵn sàng")
        status_bar.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")
        main_layout.addWidget(status_bar)

        # Thêm widget chính vào cửa sổ
        self.setCentralWidget(central_widget)

        # Thiết lập style chung
        self.setStyleSheet("""
QMainWindow, QWidget {
    background-color: white;
}
""")

    def open_image(self):
        self.image_label.setText("Đã mở file ảnh (chế độ giả lập)")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageModeApp()
    sys.exit(app.exec_())
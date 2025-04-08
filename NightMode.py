import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, 
                           QHBoxLayout, QSlider, QCheckBox, QTimeEdit, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QFont, QColor, QPalette

class NightModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.night_mode_on = False
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Night Mode')
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
    background-color: #9b59b6;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
}
QPushButton:hover {
    background-color: #8e44ad;
}
""")
        back_button.clicked.connect(self.close)
        
        title = QLabel('Night Mode')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))
        title.setStyleSheet('color: #9b59b6;')
        
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
        
        # Nút bật/tắt chế độ đêm
        toggle_layout = QHBoxLayout()
        
        self.toggle_button = QPushButton()
        self.toggle_button.setFixedSize(100, 100)
        self.toggle_button.clicked.connect(self.toggle_night_mode)
        self.update_toggle_button()
        
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.toggle_button)
        toggle_layout.addStretch()
        
        # Các cài đặt
        settings_layout = QHBoxLayout()
        
        # Độ sáng
        brightness_group = QGroupBox("Độ sáng màn hình")
        brightness_layout = QVBoxLayout()
        
        brightness_label = QLabel("Điều chỉnh độ sáng màn hình:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(10, 100)
        self.brightness_slider.setValue(70)
        self.brightness_value
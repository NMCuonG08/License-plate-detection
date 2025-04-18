import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QFileDialog, QSlider, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage

class ImageModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None  # ảnh gốc
        self.current_image = None  # ảnh hiển thị
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Editor')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        header_layout = QHBoxLayout()
        back_button = QPushButton("← Quay lại")
        back_button.setFont(QFont('Segoe UI', 12))
        back_button.clicked.connect(self.close)

        title = QLabel('Image Editor')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))

        close_button = QPushButton("X")
        close_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        close_button.setFixedSize(40, 40)
        close_button.clicked.connect(self.close)

        header_layout.addWidget(back_button)
        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(close_button)

        toolbar_layout = QHBoxLayout()
        open_button = QPushButton("Mở ảnh")
        open_button.clicked.connect(self.open_image)

        save_button = QPushButton("Lưu ảnh")
        save_button.clicked.connect(self.save_image)

        filter_label = QLabel("Bộ lọc:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Bình thường", "Đen trắng", "Mờ", "Độ sáng"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)

        brightness_label = QLabel("Độ sáng:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)

        detect_vehicle_button = QPushButton("Phát hiện phương tiện")
        detect_vehicle_button.clicked.connect(self.detect_vehicle)

        detect_sign_button = QPushButton("Phát hiện biển báo")
        detect_sign_button.clicked.connect(self.detect_traffic_sign)

        toolbar_layout.addWidget(open_button)
        toolbar_layout.addWidget(save_button)
        toolbar_layout.addWidget(filter_label)
        toolbar_layout.addWidget(self.filter_combo)
        toolbar_layout.addWidget(brightness_label)
        toolbar_layout.addWidget(self.brightness_slider)
        toolbar_layout.addWidget(detect_vehicle_button)
        toolbar_layout.addWidget(detect_sign_button)
        toolbar_layout.addStretch()

        self.image_label = QLabel("Chưa có ảnh nào được mở.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 2px dashed #ccc; border-radius: 10px; padding: 20px;")

        main_layout.addLayout(header_layout)
        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.image_label, 1)

        status_bar = QLabel("Sẵn sàng")
        status_bar.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")
        main_layout.addWidget(status_bar)

        self.setCentralWidget(central_widget)
        self.setStyleSheet("QMainWindow, QWidget { background-color: white; }")

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.current_image = self.image.copy()
            self.show_image(self.current_image)

    def save_image(self):
        if self.current_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "Images (*.png *.jpg *.bmp)")
            if path:
                cv2.imwrite(path, self.current_image)
    
    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def apply_filter(self):
        if self.image is None:
            return
        option = self.filter_combo.currentText()
        img = self.image.copy()
        if option == "Đen trắng":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif option == "Mờ":
            img = cv2.GaussianBlur(img, (15, 15), 0)
        elif option == "Độ sáng":
            self.adjust_brightness()
            return
        self.current_image = img
        self.show_image(self.current_image)

    def adjust_brightness(self):
        if self.image is None:
            return
        alpha = self.brightness_slider.value() / 50
        bright = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
        self.current_image = bright
        self.show_image(bright)

    def detect_vehicle(self):
        if self.image is None:
            return
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        vehicles = cascade.detectMultiScale(gray, 1.1, 2)
        detected = self.image.copy()
        for (x, y, w, h) in vehicles:
            cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 255, 0), 3)
        self.current_image = detected
        self.show_image(detected)

    def detect_traffic_sign(self):
        if self.image is None:
            return
        # Giả lập detection bằng cách vẽ 1 hình tròn (có thể thay bằng YOLO / custom model).
        detected = self.image.copy()
        gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(detected, (i[0], i[1]), i[2], (0, 0, 255), 4)
                cv2.rectangle(detected, (i[0]-5, i[1]-5), (i[0]+5, i[1]+5), (0, 128, 255), -1)
        self.current_image = detected
        self.show_image(detected)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageModeApp()
    sys.exit(app.exec_())
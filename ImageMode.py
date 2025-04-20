import sys
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QFileDialog, QSlider, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ImageModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.current_image = None
        self.vehicle_model = YOLO('imgmodels/yolov8n.pt')  # Pre-trained YOLOv8n model
        self.plate_model = YOLO('imgmodels/best.pt')  # Updated to HuggingFace model
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Editor with Detection')
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
        detect_vehicle_button.clicked.connect(self.detect_vehicle_with_plate)

        toolbar_layout.addWidget(open_button)
        toolbar_layout.addWidget(save_button)
        toolbar_layout.addWidget(filter_label)
        toolbar_layout.addWidget(self.filter_combo)
        toolbar_layout.addWidget(brightness_label)
        toolbar_layout.addWidget(self.brightness_slider)
        toolbar_layout.addWidget(detect_vehicle_button)
        toolbar_layout.addStretch()

        self.image_label = QLabel("Chưa có ảnh nào được mở.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 2px dashed #ccc; border-radius: 10px; padding: 20px;")

        status_bar = QLabel("Sẵn sàng")
        status_bar.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")

        main_layout.addLayout(header_layout)
        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.image_label, 1)
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

    def detect_vehicle_with_plate(self):
        if self.image is None:
            return
        detected = self.image.copy()

        vehicle_results = self.vehicle_model.predict(self.image, conf=0.3)
        for result in vehicle_results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(detected, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detected, 'Vehicle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cropped_vehicle = self.image[y1:y2, x1:x2]
                plate_results = self.plate_model.predict(cropped_vehicle, conf=0.3)

                for p_result in plate_results:
                    for p_box in p_result.boxes.xyxy:
                        px1, py1, px2, py2 = map(int, p_box[:4])
                        real_px1 = x1 + px1
                        real_py1 = y1 + py1
                        real_px2 = x1 + px2
                        real_py2 = y1 + py2
                        cv2.rectangle(detected, (real_px1, real_py1), (real_px2, real_py2), (255, 0, 0), 2)

                        # Crop the detected plate region for OCR
                        plate_image = self.image[real_py1:real_py2, real_px1:real_px2]

                        # Convert the cropped plate to grayscale and apply OCR
                        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                        plate_text = pytesseract.image_to_string(gray_plate, config='--psm 8')  # Use psm 8 for single word

                        # Display the detected license plate number next to the box
                        cv2.putText(detected, plate_text.strip(), (real_px1, real_py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        self.current_image = detected
        self.show_image(detected)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageModeApp()
    sys.exit(app.exec_())
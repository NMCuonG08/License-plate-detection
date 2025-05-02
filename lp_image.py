import sys
import subprocess
import importlib

import cv2
import torch
import numpy as np
from PIL import Image
import os
import time

# Tiếp tục với phần còn lại của import
try:
    import function.utils_rotate as utils_rotate
    import function.helper as helper
except ImportError:
    print("Warning: Could not import helper functions.")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QFileDialog, QSlider, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage


class LicensePlateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.current_image = None

        # Load YOLOv5 models for license plate detection and recognition
        self.yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
        self.yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
        self.yolo_license_plate.conf = 0.60

        self.initUI()

    def initUI(self):
        self.setWindowTitle('License Plate Recognition')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        # Create main UI layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Header layout
        header_layout = QHBoxLayout()
        back_button = QPushButton("← Quay lại")
        back_button.setFont(QFont('Segoe UI', 12))
        back_button.clicked.connect(self.close)

        title = QLabel('License Plate Recognition')
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

        # Toolbar layout
        toolbar_layout = QHBoxLayout()
        open_button = QPushButton("Mở ảnh")
        open_button.clicked.connect(self.open_image)

        save_button = QPushButton("Lưu ảnh")
        save_button.clicked.connect(self.save_image)

        filter_label = QLabel("Bộ lọc:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Bình thường", "Đen trắng", "Mờ", "Zoom"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)

        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.adjust_zoom)

        detect_plate_button = QPushButton("Nhận dạng biển số")
        detect_plate_button.clicked.connect(self.detect_license_plates)

        toolbar_layout.addWidget(open_button)
        toolbar_layout.addWidget(save_button)
        toolbar_layout.addWidget(filter_label)
        toolbar_layout.addWidget(self.filter_combo)
        toolbar_layout.addWidget(zoom_label)
        toolbar_layout.addWidget(self.zoom_slider)
        toolbar_layout.addWidget(detect_plate_button)
        toolbar_layout.addStretch()

        # Image display area
        self.image_label = QLabel("Chưa có ảnh nào được mở.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #f5f5f5; border: 2px dashed #ccc; border-radius: 10px; padding: 20px;")

        # Status bar
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("padding: 5px; color: #666; border-top: 1px solid #ddd;")

        # Add all components to main layout
        main_layout.addLayout(header_layout)
        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(central_widget)
        self.setStyleSheet("QMainWindow, QWidget { background-color: white; }")

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.current_image = self.image.copy()
            self.show_image(self.current_image)
            self.status_label.setText(f"Đã mở ảnh: {os.path.basename(file_name)}")

    def save_image(self):
        if self.current_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "Images (*.png *.jpg *.bmp)")
            if path:
                cv2.imwrite(path, self.current_image)
                self.status_label.setText(f"Đã lưu ảnh: {os.path.basename(path)}")

    def show_image(self, img):
        if img is None:
            return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

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
        elif option == "Zoom":
            self.adjust_zoom()
            return

        self.current_image = img
        self.show_image(self.current_image)

    def adjust_zoom(self):
        if self.image is None:
            return

        zoom_factor = self.zoom_slider.value() / 100.0
        height, width = self.image.shape[:2]

        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        if zoom_factor > 1:  # Phóng to
            zoomed = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            zoomed = zoomed[start_y:start_y + height, start_x:start_x + width]
        else:  # Thu nhỏ
            zoomed = np.zeros_like(self.image)

            start_x = (width - new_width) // 2
            start_y = (height - new_height) // 2

            small = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            zoomed[start_y:start_y + new_height, start_x:start_x + new_width] = small

        self.current_image = zoomed
        self.show_image(self.current_image)

    def detect_license_plates(self):
        """Implement license plate detection using lp_image.py logic"""
        if self.image is None:
            self.status_label.setText("Hãy mở ảnh trước khi nhận dạng biển số")
            return

        start_time = time.time()
        self.status_label.setText("Đang nhận dạng biển số...")

        # Copy the original image to work on
        img = self.image.copy()
        result_img = img.copy()

        # Detect license plates using YOLOv5
        plates = self.yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()

        if len(list_plates) == 0:
            # Try reading plate from the whole image if no plate is detected
            lp = helper.read_plate(self.yolo_license_plate, img)
            if lp != "unknown":
                cv2.putText(result_img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                list_read_plates.add(lp)
                self.status_label.setText(f"Biển số: {lp}")
            else:
                self.status_label.setText("Không tìm thấy biển số")
        else:
            for plate in list_plates:
                flag = 0
                x = int(plate[0])  # xmin
                y = int(plate[1])  # ymin
                w = int(plate[2] - plate[0])  # xmax - xmin
                h = int(plate[3] - plate[1])  # ymax - ymin
                crop_img = img[y:y+h, x:x+w]

                # Draw rectangle around detected license plate
                cv2.rectangle(result_img, (x, y), (x+w, y+h), color=(0, 0, 225), thickness=2)

                # Try different rotations to read the plate
                for cc in range(0, 2):
                    for ct in range(0, 2):
                        lp = helper.read_plate(self.yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            # Display the recognized plate number above the plate
                            cv2.putText(result_img, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            flag = 1
                            break
                    if flag == 1:
                        break

            if list_read_plates:
                plates_text = ", ".join(list_read_plates)
                self.status_label.setText(f"Tìm thấy {len(list_read_plates)} biển số: {plates_text}")
            else:
                self.status_label.setText("Không đọc được biển số")

        # Update the displayed image with recognition results
        self.current_image = result_img
        self.show_image(result_img)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Thời gian xử lý: {processing_time:.2f} giây")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    # Thêm xử lý tham số dòng lệnh với giá trị mặc định
    import argparse
    
    parser = argparse.ArgumentParser(description="License Plate Recognition")
    parser.add_argument("-i", "--image", help="Path to input image", default=None)
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = LicensePlateApp()
    
    # Nếu có tham số ảnh, tự động mở ảnh đó
    if args.image and os.path.exists(args.image):
        window.image = cv2.imread(args.image)
        window.current_image = window.image.copy()
        window.show_image(window.current_image)
        window.status_label.setText(f"Đã mở ảnh: {os.path.basename(args.image)}")
    
    sys.exit(app.exec_())
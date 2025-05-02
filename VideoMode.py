import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QSlider, QStyle, QFileDialog, QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
import cv2
from datetime import timedelta
import torch
from PIL import Image
import numpy as np
import easyocr
import re
import time
from openalpr import Alpr  # Thêm import OpenALPR


class VideoModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.playing = False
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.frame_update_timer = QTimer()
        self.frame_update_timer.timeout.connect(self.update_frame)

        # For license plate detection
        self.detection_mode = False
        self.detected_plates = []
        self.was_playing = False

        # Tải model chỉ khi cần thiết để tránh lag lúc khởi động
        self.models_loaded = False
        self.plate_detector = None

        # Thêm biến để tối ưu hiệu suất
        self.skip_frames = 5  # Chỉ xử lý 1 frame sau mỗi 5 frame
        self.frame_count = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle('License Plate Detection - Video Player')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        # Main widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Header layout
        header_layout = QHBoxLayout()

        back_button = QPushButton("← Quay lại")
        back_button.setFont(QFont('Segoe UI', 12))
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        back_button.clicked.connect(self.close)

        title = QLabel('License Plate Detection - Video Player')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 24))
        title.setStyleSheet('color: #e74c3c;')

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

        # Video display area
        video_container = QFrame()
        video_container.setFrameShape(QFrame.StyledPanel)
        video_container.setStyleSheet("""
            QFrame {
                background-color: #222;
                border-radius: 10px;
                padding: 2px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(1, 1, 1, 1)

        self.video_frame = QLabel("Chưa có video nào được mở.\nNhấn 'Mở video' để bắt đầu.")
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 18px;
            }
        """)
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setMinimumHeight(500)
        video_layout.addWidget(self.video_frame)

        # Time display and slider
        time_layout = QHBoxLayout()
        self.current_time = QLabel("00:00")
        self.current_time.setFont(QFont('Segoe UI', 10))
        self.total_time = QLabel("00:00")
        self.total_time.setFont(QFont('Segoe UI', 10))

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 100)
        self.timeline.sliderMoved.connect(self.seek_position)
        self.timeline.sliderPressed.connect(self.pause_video)
        self.timeline.sliderReleased.connect(self.resume_video_if_playing)

        time_layout.addWidget(self.current_time)
        time_layout.addWidget(self.timeline, 1)
        time_layout.addWidget(self.total_time)

        # Control buttons
        controls_layout = QHBoxLayout()

        # Open button
        open_button = QPushButton("Mở video")
        open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        open_button.setFixedHeight(40)
        open_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        open_button.clicked.connect(self.open_video)

        # Play/Pause button
        self.play_button = QPushButton("▶ Phát")
        self.play_button.setFixedSize(120, 40)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)

        # Stop button
        self.stop_button = QPushButton("⏹ Dừng")
        self.stop_button.setFixedHeight(40)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        # License plate detection button
        self.detection_button = QPushButton("🔍 Nhận diện biển số")
        self.detection_button.setFixedHeight(40)
        self.detection_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.detection_button.clicked.connect(self.toggle_detection)
        self.detection_button.setEnabled(False)

        # Volume control
        volume_label = QLabel("Âm lượng:")
        volume_label.setFont(QFont('Segoe UI', 10))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)

        controls_layout.addWidget(open_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.detection_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(volume_label)
        controls_layout.addWidget(self.volume_slider)

        # Status bar
        self.status_layout = QHBoxLayout()
        self.status_label = QLabel(
            self.status_label_text if hasattr(self, 'status_label_text') else "Trạng thái: Sẵn sàng")
        self.status_label.setFont(QFont('Segoe UI', 9))
        self.status_layout.addWidget(self.status_label)

        # Recognition results
        self.results_label = QLabel("Đã nhận diện: 0 biển số")
        self.results_label.setFont(QFont('Segoe UI', 9))
        self.results_label.setAlignment(Qt.AlignRight)
        self.status_layout.addWidget(self.results_label)

        # Add components to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(video_container, 1)
        main_layout.addLayout(time_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(self.status_layout)

        # Set main widget
        self.setCentralWidget(central_widget)

        # Set global style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #333;
            }
            QWidget {
                background-color: #333;
                color: white;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #e74c3c;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)

    def load_models(self):
        if self.models_loaded:
            return True

        try:
            from ultralytics import YOLO

            # Tải mô hình YOLO để phát hiện biển số
            self.plate_detector = YOLO('my_model.pt')

            # Khởi tạo EasyOCR như dự phòng
            self.reader = easyocr.Reader(['en'])

            # Khởi tạo OpenALPR
            try:
                # Tham số: country, config_file, runtime_dir
                self.alpr = Alpr("vn", "/path/to/openalpr.conf", "/usr/share/openalpr/runtime_data")
                if not self.alpr.is_loaded():
                    print("Warning: OpenALPR failed to load, falling back to EasyOCR")
                    self.use_openalpr = False
                else:
                    # Cấu hình cho OpenALPR
                    self.alpr.set_top_n(5)  # Số lượng kết quả trả về
                    self.alpr.set_default_region("vn")  # Đặt khu vực mặc định
                    self.use_openalpr = True
                    print("OpenALPR loaded successfully")
            except Exception as alpr_ex:
                print(f"Error loading OpenALPR: {alpr_ex}")
                self.use_openalpr = False

            self.models_loaded = True
            self.status_label.setText("Trạng thái: Model đã được tải")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.status_label.setText("Trạng thái: Lỗi khi tải model")
            return False

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Mở Video", "",
                                                   "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)")

        if file_name:
            try:
                # Close previous video if open
                self.stop_video()

                # Open new video file
                self.video_path = file_name
                self.cap = cv2.VideoCapture(file_name)

                if not self.cap.isOpened():
                    QMessageBox.critical(self, "Lỗi", "Không thể mở video. Vui lòng thử lại với file khác.")
                    return

                # Get video properties
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0

                # Update UI
                self.timeline.setRange(0, self.total_frames)
                self.timeline.setValue(0)
                self.current_frame = 0

                duration_str = str(timedelta(seconds=int(duration_seconds)))
                if duration_str.startswith('0:'):
                    duration_str = duration_str[2:]  # Remove leading '0:'
                self.total_time.setText(duration_str)
                self.current_time.setText("00:00")

                # Display first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)

                # Enable controls
                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.detection_button.setEnabled(True)

                # Update status
                self.status_label.setText(f"Trạng thái: Đã mở video - {os.path.basename(file_name)}")
                self.results_label.setText("Đã nhận diện: 0 biển số")
                self.detected_plates = []

            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi mở video: {str(e)}")
                self.status_label.setText("Trạng thái: Lỗi khi mở video")

    def display_frame(self, frame):
        # Convert frame for display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # If detection mode is on, process the frame
        if self.detection_mode:
            frame = self.detect_license_plates(frame)

        # Convert to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap and display
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit but maintain aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_frame.width(), self.video_frame.height(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.video_frame.setPixmap(scaled_pixmap)
        self.video_frame.setAlignment(Qt.AlignCenter)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened() or not self.playing:
            return

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.current_frame += 1
            self.timeline.setValue(self.current_frame)

            # Update time display
            if self.fps > 0:
                current_time = self.current_frame / self.fps
                time_str = str(timedelta(seconds=int(current_time)))
                if time_str.startswith('0:'):
                    time_str = time_str[2:]  # Remove leading '0:'
                self.current_time.setText(time_str)
        else:
            # End of video
            self.stop_video()

    def toggle_play(self):
        if self.cap is None or not self.cap.isOpened():
            return

        self.playing = not self.playing

        if self.playing:
            self.play_button.setText("⏸ Tạm dừng")
            self.frame_update_timer.start(int(1000 / self.fps) if self.fps > 0 else 30)
            self.status_label.setText(f"Trạng thái: Đang phát - {os.path.basename(self.video_path)}")
        else:
            self.play_button.setText("▶ Phát")
            self.frame_update_timer.stop()
            self.status_label.setText(f"Trạng thái: Tạm dừng - {os.path.basename(self.video_path)}")

    def stop_video(self):
        self.playing = False
        self.play_button.setText("▶ Phát")

        if self.frame_update_timer.isActive():
            self.frame_update_timer.stop()

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.current_frame = 0
                self.timeline.setValue(0)
                self.current_time.setText("00:00")

        self.status_label.setText(f"Trạng thái: Dừng - {os.path.basename(self.video_path) if self.video_path else ''}")

    def seek_position(self, position):
        if self.cap is None or not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.current_frame = position

            # Update time display
            if self.fps > 0:
                current_time = position / self.fps
                time_str = str(timedelta(seconds=int(current_time)))
                if time_str.startswith('0:'):
                    time_str = time_str[2:]  # Remove leading '0:'
                self.current_time.setText(time_str)

        # Reset position back one frame to avoid skipping
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    def pause_video(self):
        if self.playing:
            self.was_playing = True
            self.playing = False
            self.frame_update_timer.stop()
        else:
            self.was_playing = False

    def resume_video_if_playing(self):
        if self.was_playing:
            self.playing = True
            self.frame_update_timer.start(int(1000 / self.fps) if self.fps > 0 else 30)

    def set_volume(self, value):
        # This is just a placeholder for volume functionality
        # In a real implementation, you would need to add audio handling
        pass

    def toggle_detection(self):
        self.detection_mode = not self.detection_mode

        if self.detection_mode:
            # Tải model khi bật chế độ nhận diện
            if not hasattr(self, 'models_loaded') or not self.models_loaded:
                success = self.load_models()
                if not success:
                    self.detection_mode = False
                    QMessageBox.critical(self, "Lỗi", "Không thể tải model nhận diện. Vui lòng kiểm tra cài đặt.")
                    return

            self.detection_button.setText("🔍 Tắt nhận diện")
            self.detection_button.setStyleSheet("""
                QPushButton {
                    background-color: #f39c12;
                    color: white;
                    border-radius: 5px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #d35400;
                }
            """)
            self.status_label.setText("Trạng thái: Đang nhận diện biển số...")
        else:
            self.detection_button.setText("🔍 Nhận diện biển số")
            self.detection_button.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: white;
                    border-radius: 5px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
            """)
            self.status_label.setText(f"Trạng thái: {os.path.basename(self.video_path)}")

        # Update current frame with/without detection
        if self.cap is not None and self.cap.isOpened():
            # Store current position
            current_pos = self.current_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                # Reset position back
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def detect_license_plates(self, frame):
        """Phát hiện và nhận dạng biển số với cơ chế ổn định kết quả"""
        # Tăng bộ đếm frame
        self.frame_count += 1

        # Chỉ xử lý một số frame để cải thiện hiệu suất
        if self.frame_count % self.skip_frames != 0 and hasattr(self, 'last_detection_result'):
            return self.last_detection_result

        # Clone frame để tránh sửa đổi bản gốc
        result_frame = frame.copy()

        if not hasattr(self, 'models_loaded') or not self.models_loaded:
            cv2.putText(result_frame, "Model không được tải", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_detection_result = result_frame
            return result_frame

        try:
            # Phát hiện biển số với YOLO
            results = self.plate_detector(frame, conf=0.45)  # Tăng ngưỡng confidence

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Lấy tọa độ biển số
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]

                    # Chỉ lấy các box có độ tin cậy cao
                    if confidence < 0.45:
                        continue

                    # Trích xuất vùng biển số
                    plate_roi = frame[y1:y2, x1:x2]

                    if plate_roi.size == 0:
                        continue

                    # Lưu trữ để debug nếu cần
                    # cv2.imwrite(f"plate_debug_{time.time()}.jpg", plate_roi)

                    # Tiền xử lý ảnh với nhiều phương pháp
                    plate_text = self.recognize_plate_text(plate_roi)

                    # Xác thực và chuẩn hóa định dạng biển số
                    plate_text = self.validate_plate_format(plate_text)

                    # Nếu biển số hợp lệ
                    if plate_text and len(plate_text) >= 5:
                        # Vẽ hình chữ nhật xung quanh biển số
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Thêm nền cho văn bản
                        text_width = len(plate_text) * 15
                        cv2.rectangle(result_frame, (x1, y1 - 30), (x1 + text_width, y1), (0, 255, 0), -1)

                        # Hiển thị văn bản đã nhận dạng
                        cv2.putText(result_frame, plate_text, (x1 + 5, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        # Lưu thông tin biển số nếu là mới
                        plate_info = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                                      'frame': self.current_frame, 'text': plate_text,
                                      'confidence': float(confidence)}

                        # Chỉ thêm nếu là biển số mới (dựa trên vị trí)
                        if not any(abs(plate['x'] - x1) < 20 and abs(plate['y'] - y1) < 20 for plate in
                                   self.detected_plates):
                            self.detected_plates.append(plate_info)
                            self.results_label.setText(f"Đã nhận diện: {len(self.detected_plates)} biển số")
                    else:
                        # Vẽ hình chữ nhật đỏ nếu văn bản không hợp lệ
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Lưu kết quả để tái sử dụng trên các frame bị bỏ qua
            self.last_detection_result = result_frame

        except Exception as e:
            print(f"Error in license plate detection: {str(e)}")
            import traceback
            traceback.print_exc()
            cv2.putText(result_frame, "Lỗi nhận diện", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_detection_result = result_frame

        return result_frame

    def recognize_plate_text(self, plate_image):
        """Nhận dạng biển số với cơ chế ổn định giữa các frame"""
        try:
            # Thử nhiều phương pháp tiền xử lý và lấy kết quả tốt nhất
            results = []

            # Method 1: Ảnh gốc (đã resize)
            height, width = plate_image.shape[:2]
            if width < 150 or height < 50:
                scale_factor = max(150 / width, 50 / height)
                resized = cv2.resize(plate_image, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv2.INTER_CUBIC)
            else:
                resized = plate_image.copy()

            result1 = self.reader.readtext(resized, detail=0)
            if result1:
                text1 = ''.join(result1)
                text1 = re.sub(r'[^A-Z0-9]', '', text1.upper())
                if len(text1) >= 4:
                    results.append((text1, 0.6))  # Độ tin cậy trung bình

            # Method 2: Ảnh xám + CLAHE enhancer
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
            enhanced = clahe.apply(gray)

            result2 = self.reader.readtext(enhanced, detail=0)
            if result2:
                text2 = ''.join(result2)
                text2 = re.sub(r'[^A-Z0-9]', '', text2.upper())
                if len(text2) >= 4:
                    results.append((text2, 0.7))  # Độ tin cậy cao hơn

            # Method 3: Binary thresholding
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            result3 = self.reader.readtext(binary, detail=0)
            if result3:
                text3 = ''.join(result3)
                text3 = re.sub(r'[^A-Z0-9]', '', text3.upper())
                if len(text3) >= 4:
                    results.append((text3, 0.8))  # Độ tin cậy cao nhất

            # Method 4: Inverse binary
            binary_inv = cv2.bitwise_not(binary)

            result4 = self.reader.readtext(binary_inv, detail=0)
            if result4:
                text4 = ''.join(result4)
                text4 = re.sub(r'[^A-Z0-9]', '', text4.upper())
                if len(text4) >= 4:
                    results.append((text4, 0.75))

            # Không có kết quả hợp lệ
            if not results:
                return ""

            # Thêm vào tracking history để ổn định kết quả
            if not hasattr(self, 'plate_history'):
                self.plate_history = {}

            # Tìm text phổ biến nhất từ kết quả hiện tại
            current_text = max(results, key=lambda x: x[1])[0]

            # Sửa các lỗi thường gặp
            current_text = current_text.replace('O', '0').replace('I', '1').replace('S', '5')
            current_text = current_text.replace('B', '8').replace('Z', '2').replace('G', '6')

            # Thêm vào lịch sử
            plate_key = f"{self.current_frame // 30}"  # Nhóm theo mỗi 30 frame
            if plate_key not in self.plate_history:
                self.plate_history[plate_key] = []

            self.plate_history[plate_key].append(current_text)

            # Giới hạn lịch sử
            if len(self.plate_history[plate_key]) > 10:
                self.plate_history[plate_key].pop(0)

            # Voting để chọn kết quả ổn định nhất
            if len(self.plate_history[plate_key]) >= 3:
                # Đếm số lần xuất hiện của mỗi text
                from collections import Counter
                text_counts = Counter(self.plate_history[plate_key])

                # Lấy text xuất hiện nhiều nhất
                most_common = text_counts.most_common(1)[0][0]

                # Nếu text phổ biến xuất hiện ít nhất 2 lần, sử dụng nó
                if text_counts[most_common] >= 2:
                    return most_common

            # Nếu không đủ dữ liệu để voting, trả về kết quả hiện tại
            return current_text

        except Exception as e:
            print(f"Error in plate recognition: {str(e)}")
            return ""

    def validate_plate_format(self, text):
        """Kiểm tra và chuẩn hóa định dạng biển số Việt Nam"""
        # Loại bỏ tất cả ký tự không phải chữ và số
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Nếu quá ngắn thì không hợp lệ
        if len(text) < 7:
            return None

        # Dạng biển xe con (thường có 2-3 ký tự đầu tiên là chữ)
        # VD: 51G-12345, 30A-12345
        if len(text) in [7, 8, 9]:
            letter_part = ''
            digit_part = ''

            # Tách phần chữ (1-3 ký tự đầu) và phần số
            for i, char in enumerate(text):
                if char.isalpha() and len(letter_part) < 3:
                    letter_part += char
                else:
                    digit_part = text[i:]
                    break

            # Nếu có 1-3 chữ cái đầu và phần còn lại là số
            if 1 <= len(letter_part) <= 3 and digit_part.isdigit() and len(digit_part) >= 5:
                # Định dạng lại biển số
                if len(letter_part) == 2:
                    return f"{letter_part[0]}{letter_part[1]}-{digit_part}"
                elif len(letter_part) == 1:
                    return f"{letter_part}-{digit_part}"
                else:
                    return f"{letter_part[0:2]}{letter_part[2]}-{digit_part}"

        # Dạng biển xe máy hoặc loại khác
        # Cố gắng tách chữ và số theo mẫu thông thường
        match = re.search(r'([A-Z]{1,3})([0-9]+)', text)
        if match:
            letters, numbers = match.groups()
            if len(letters) <= 3 and len(numbers) >= 5:
                return f"{letters}-{numbers}"

        # Trả về nguyên bản nếu không nhận dạng được mẫu cụ thể
        return text

    def preprocess_plate_advanced(self, plate_image):
        """Tiền xử lý nâng cao cho ảnh biển số"""
        # Resize về kích thước thích hợp
        height, width = plate_image.shape[:2]
        if width < 180 or height < 60:
            scale_factor = max(180 / width, 60 / height)
            plate_image = cv2.resize(plate_image, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv2.INTER_CUBIC)

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Khử noise với bilateral filter (giữ cạnh tốt hơn Gaussian)
        denoised = cv2.bilateralFilter(gray, 11, 90, 90)

        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
        enhanced = clahe.apply(denoised)

        # Sharpen ảnh để làm rõ cạnh
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Áp dụng ngưỡng thích ứng
        binary = cv2.adaptiveThreshold(sharpened, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)

        # Làm mịn kết quả
        kernel2 = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

        return morph

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:
            # Go back 5 seconds
            if self.cap is not None and self.fps > 0:
                new_frame = max(0, self.current_frame - int(5 * self.fps))
                self.timeline.setValue(new_frame)
                self.seek_position(new_frame)
        elif event.key() == Qt.Key_Right:
            # Go forward 5 seconds
            if self.cap is not None and self.fps > 0:
                new_frame = min(self.total_frames - 1, self.current_frame + int(5 * self.fps))
                self.timeline.setValue(new_frame)
                self.seek_position(new_frame)
        elif event.key() == Qt.Key_F:
            # Toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def closeEvent(self, event):
        # Giải phóng tài nguyên OpenALPR
        if hasattr(self, 'alpr') and self.alpr.is_loaded():
            self.alpr.unload()

        # Clean up resources before closing
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Accept the close event
        event.accept()


def preprocess_plate(plate_image):
    # Chuyển sang thang xám
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Cân bằng histogram để tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    enhanced = clahe.apply(gray)

    # Loại nhiễu bằng Gaussian blur nhẹ
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Áp dụng ngưỡng thích ứng
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Xử lý hình thái học để làm sạch ảnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Kiểm tra tỷ lệ đen/trắng và đảo ngược nếu cần
    if cv2.countNonZero(morph) > morph.size * 0.55:
        morph = cv2.bitwise_not(morph)

    return morph


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoModeApp()
    sys.exit(app.exec_())
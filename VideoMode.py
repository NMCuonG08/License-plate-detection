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
        self.status_label = QLabel(self.status_label_text if hasattr(self, 'status_label_text') else "Trạng thái: Sẵn sàng")
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
            import easyocr

            self.plate_detector = YOLO('license_plate_detector.pt')
            self.reader = easyocr.Reader(['en'])  # Initialize for English characters

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
        # Increment frame counter
        self.frame_count += 1

        # Only process every few frames to improve performance
        if self.frame_count % self.skip_frames != 0 and hasattr(self, 'last_detection_result'):
            return self.last_detection_result

        # Clone frame to avoid modifying original
        result_frame = frame.copy()

        if not hasattr(self, 'models_loaded') or not self.models_loaded:
            cv2.putText(result_frame, "Model không được tải", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_detection_result = result_frame
            return result_frame

        try:
            # Initialize EasyOCR if not already done
            if not hasattr(self, 'reader'):
                self.reader = easyocr.Reader(['en'])  # Initialize for English characters

            # Perform detection with YOLOv8
            results = self.plate_detector(frame, conf=0.4)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Get license plate coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract license plate region
                    plate_roi = frame[y1:y2, x1:x2]

                    if plate_roi.size == 0:
                        continue

                    # Preprocess image for better OCR
                    # Convert to grayscale
                    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                    # Enhance contrast
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray_plate)

                    # Apply bilateral filter to preserve edges while reducing noise
                    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(
                        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )

                    # Resize image to better handle text recognition
                    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # Use EasyOCR for text detection
                    ocr_results = self.reader.readtext(resized)

                    # Process OCR results
                    if ocr_results:
                        # Extract text from results
                        plate_text = ''.join([result[1] for result in ocr_results])

                        # Clean up text - remove non-alphanumeric characters
                        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

                        # If we have a reasonable-length text
                        if len(plate_text) >= 4:
                            # Draw rectangle around the license plate
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Add background for text
                            text_width = len(plate_text) * 15
                            cv2.rectangle(result_frame, (x1, y1 - 30), (x1 + text_width, y1), (0, 255, 0), -1)

                            # Display the recognized text
                            cv2.putText(result_frame, plate_text, (x1 + 5, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                            # Store plate information if it's new
                            plate_info = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                                          'frame': self.current_frame, 'text': plate_text}

                            if not any(abs(plate['x'] - x1) < 20 and abs(plate['y'] - y1) < 20 for plate in
                                       self.detected_plates):
                                self.detected_plates.append(plate_info)
                                self.results_label.setText(f"Đã nhận diện: {len(self.detected_plates)} biển số")
                        else:
                            # Draw red rectangle if text is invalid
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        # Draw red rectangle if no text was detected
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Save result for reuse on skipped frames
            self.last_detection_result = result_frame

        except Exception as e:
            print(f"Error in license plate detection: {str(e)}")
            import traceback
            traceback.print_exc()
            cv2.putText(result_frame, "Lỗi nhận diện", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_detection_result = result_frame

        return result_frame

    def process_plate(self, frame, x1, y1, x2, y2):
        """Process a detected license plate region"""
        # Extract license plate ROI
        plate_roi = frame[y1:y2, x1:x2]

        if plate_roi.size == 0:
            return

        # Preprocess for better OCR
        gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary_plate = cv2.adaptiveThreshold(
            gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Optional: Apply morphological operations to clean the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)

        # Invert if needed (white text on black background)
        if np.mean(binary_plate) > 127:
            binary_plate = cv2.bitwise_not(binary_plate)

        # Perform OCR on the plate
        try:
            plate_text = ""

            # Use appropriate OCR engine
            if hasattr(self, 'ocr_engine'):
                if self.ocr_engine == "tesseract":
                    # Using Tesseract OCR with specific config for license plates
                    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    plate_text = self.reader.image_to_string(binary_plate, config=config)
                    # Clean the text
                    plate_text = ''.join(c for c in plate_text if c.isalnum())

                elif self.ocr_engine == "easyocr":
                    # Using EasyOCR
                    results = self.reader.readtext(binary_plate)
                    plate_text = ' '.join([result[1] for result in results])
                    # Clean the text
                    plate_text = ''.join(c for c in plate_text if c.isalnum())

            # If we have valid text of reasonable length
            if plate_text and 4 <= len(plate_text) <= 12:
                # Draw rectangle around plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show the text above the rectangle
                cv2.putText(frame, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Store the detected plate if it's not already in the list
                plate_info = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                              'frame': self.current_frame, 'text': plate_text}

                # Check if this is a new detection
                if not any(abs(plate['x'] - x1) < 20 and abs(plate['y'] - y1) < 20 for plate in self.detected_plates):
                    self.detected_plates.append(plate_info)
                    self.results_label.setText(f"Đã nhận diện: {len(self.detected_plates)} biển số")
            else:
                # Draw a different color rectangle if text is invalid
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "Biển số không rõ", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        except Exception as e:
            print(f"OCR error: {str(e)}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
        # Clean up resources before closing
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Accept the close event
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoModeApp()
    sys.exit(app.exec_())
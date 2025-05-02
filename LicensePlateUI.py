import sys
import cv2
import os
import numpy as np
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                            QHBoxLayout, QSlider, QStyle, QFileDialog, QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from datetime import timedelta

# Import our license plate detection components
from src.PlateDetector import PlateDetector
from src.PlateReader import PlateReader
from src.Car import Car
from src.utils import draw_border, draw_plate, find_common_numbers

class LicensePlateUI(QMainWindow):
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
        
        # For storing car objects
        self.cars = {}

        # Lazy loading models
        self.models_loaded = False
        self.plate_detector = None
        self.plate_reader = None

        # Performance optimization
        self.skip_frames = 2  # Process every 2nd frame
        self.frame_count = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle('License Plate Recognition System')
        self.setGeometry(100, 100, 1280, 800)

        # Main widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Header layout
        header_layout = QHBoxLayout()

        title = QLabel('License Plate Recognition System')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 24))
        title.setStyleSheet('color: #2980b9;')

        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()

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

        self.video_frame = QLabel("No video loaded.\nPress 'Open Video' to start.")
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
        open_button = QPushButton("Open Video")
        open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        open_button.setFixedHeight(40)
        open_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        open_button.clicked.connect(self.open_video)

        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
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
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setFixedHeight(40)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        # License plate detection button
        self.detection_button = QPushButton("🔍 Toggle Detection")
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

        # Exit button
        self.exit_button = QPushButton("❌ Exit")
        self.exit_button.setFixedHeight(40)
        self.exit_button.setStyleSheet("""
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
        self.exit_button.clicked.connect(self.close)

        controls_layout.addWidget(open_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.detection_button)
        controls_layout.addWidget(self.exit_button)
        controls_layout.addStretch(1)

        # Status bar
        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont('Segoe UI', 9))
        self.status_layout.addWidget(self.status_label)

        # Recognition results
        self.results_label = QLabel("Detected: 0 license plates")
        self.results_label.setFont(QFont('Segoe UI', 9))
        self.results_label.setAlignment(Qt.AlignRight)
        self.status_layout.addWidget(self.results_label)

        # Detected plates list (new addition)
        plates_container = QFrame()
        plates_container.setFrameShape(QFrame.StyledPanel)
        plates_container.setMaximumHeight(100)
        plates_container.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        plates_layout = QVBoxLayout(plates_container)
        
        plates_title = QLabel("Detected License Plates:")
        plates_title.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.plates_list = QLabel("None")
        self.plates_list.setWordWrap(True)
        
        plates_layout.addWidget(plates_title)
        plates_layout.addWidget(self.plates_list)

        # Add components to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(video_container, 1)
        main_layout.addLayout(time_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(plates_container)
        main_layout.addLayout(self.status_layout)

        # Set main widget
        self.setCentralWidget(central_widget)

        # Set global style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #ddd;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)

    def load_models(self):
        if self.models_loaded:
            return
        
        try:
            self.status_label.setText("Status: Loading models...")
            QApplication.processEvents()  # Update UI
            
            self.plate_detector = PlateDetector()
            self.plate_reader = PlateReader()
            
            self.models_loaded = True
            self.status_label.setText("Status: Models loaded successfully")
        except Exception as e:
            self.status_label.setText(f"Error: Failed to load models - {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load detection models: {str(e)}")

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "",
                                                  "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)")

        if file_name:
            try:
                self.cap = cv2.VideoCapture(file_name)
                if not self.cap.isOpened():
                    QMessageBox.critical(self, "Error", "Could not open video file")
                    return

                self.video_path = file_name
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.current_frame = 0
                
                # Update timeline
                self.timeline.setRange(0, self.total_frames - 1)
                total_seconds = int(self.total_frames / self.fps)
                self.total_time.setText(str(timedelta(seconds=total_seconds)))
                
                # Get first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    
                # Enable controls
                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.detection_button.setEnabled(True)
                self.detection_mode = False
                
                # Reset car tracking
                self.cars = {}
                self.plates_list.setText("None")
                self.results_label.setText("Detected: 0 license plates")
                
                # Update status
                filename = os.path.basename(file_name)
                self.status_label.setText(f"Status: Loaded - {filename}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")

    def display_frame(self, frame):
        # Convert frame for display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

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
            self.current_frame += 1
            
            # Process with license plate detection if enabled
            if self.detection_mode:
                frame = self.process_frame_with_detection(frame)
                
            self.display_frame(frame)
            
            # Update timeline and time display
            self.timeline.setValue(self.current_frame)
            current_seconds = int(self.current_frame / self.fps)
            self.current_time.setText(str(timedelta(seconds=current_seconds)))
        else:
            # End of video
            self.playing = False
            self.play_button.setText("▶ Play")
            if self.frame_update_timer.isActive():
                self.frame_update_timer.stop()
                
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.timeline.setValue(0)
            self.current_time.setText("00:00")
            self.status_label.setText(f"Status: End of video - {os.path.basename(self.video_path)}")

    def process_frame_with_detection(self, frame):
        # Only process every few frames to improve performance
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            # Just draw existing detections
            processed_frame = frame.copy()
            for car_id in self.cars:
                car = self.cars[car_id]
                if car.plate_number != -1:
                    processed_frame = draw_border(processed_frame, 
                                               (car.x_car1, car.y_car1), 
                                               (car.x_car2, car.y_car2), 
                                               car.plate_number, 
                                               car.color)
            return processed_frame
        
        # Make sure models are loaded
        if not self.models_loaded:
            self.load_models()
        
        try:
            # Run detection
            processed_frame = frame.copy()
            results, all_cars_ids = self.plate_detector.find_vehicles(frame)
            
            # Process detected vehicles
            for result in results:
                x_car1, y_car1, x_car2, y_car2, car_id, license_plate_image = result
                if car_id in self.cars.keys():
                    self.cars[car_id].update_car(result)
                else:
                    self.cars[car_id] = Car(result)
            
            # Find cars that are currently visible
            show_ids = find_common_numbers(list(self.cars.keys()), all_cars_ids)
            
            # Draw bounding boxes for visible cars with plates
            for car_id in show_ids:
                car = self.cars[car_id]
                if car.plate_number != -1:
                    processed_frame = draw_border(processed_frame, 
                                               (car.x_car1, car.y_car1), 
                                               (car.x_car2, car.y_car2), 
                                               car.plate_number, 
                                               car.color)
            
            # Update plates list display
            plates_text = []
            valid_plates = 0
            for car_id, car in self.cars.items():
                if car.plate_number != -1:
                    plates_text.append(f"{car.plate_number}")
                    valid_plates += 1
            
            if plates_text:
                self.plates_list.setText(", ".join(plates_text))
                self.results_label.setText(f"Detected: {valid_plates} license plates")
            else:
                self.plates_list.setText("None")
                self.results_label.setText("Detected: 0 license plates")
            
            return processed_frame
            
        except Exception as e:
            self.status_label.setText(f"Error in detection: {str(e)}")
            return frame

    def toggle_play(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "No video loaded")
            return

        self.playing = not self.playing

        if self.playing:
            self.play_button.setText("⏸ Pause")
            self.frame_update_timer.start(int(1000 / self.fps))  # Update based on FPS
            self.status_label.setText(f"Status: Playing - {os.path.basename(self.video_path)}")
        else:
            self.play_button.setText("▶ Play")
            self.frame_update_timer.stop()
            self.status_label.setText(f"Status: Paused - {os.path.basename(self.video_path)}")

    def stop_video(self):
        self.playing = False
        self.play_button.setText("▶ Play")

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

        self.status_label.setText(f"Status: Stopped - {os.path.basename(self.video_path) if self.video_path else ''}")

    def seek_position(self, position):
        if self.cap is None or not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.current_frame = position
            current_seconds = int(position / self.fps)
            self.current_time.setText(str(timedelta(seconds=current_seconds)))

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
            self.frame_update_timer.start(int(1000 / self.fps))

    def toggle_detection(self):
        self.detection_mode = not self.detection_mode

        if self.detection_mode:
            # Make sure models are loaded when detection is enabled
            if not self.models_loaded:
                self.load_models()
            self.detection_button.setText("🔍 Detection ON")
            self.detection_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border-radius: 5px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
            """)
            self.status_label.setText("Status: License plate detection enabled")
        else:
            self.detection_button.setText("🔍 Detection OFF")
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
            self.status_label.setText("Status: License plate detection disabled")

        # Update current frame with/without detection
        if self.cap is not None and self.cap.isOpened():
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            ret, frame = self.cap.read()
            if ret:
                if self.detection_mode:
                    frame = self.process_frame_with_detection(frame)
                self.display_frame(frame)

    def keyPressEvent(self, event):
        # Space for play/pause
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        # Right arrow for forward 5 seconds
        elif event.key() == Qt.Key_Right:
            if self.cap is not None and self.cap.isOpened():
                skip_frames = int(5 * self.fps)
                new_pos = min(self.current_frame + skip_frames, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.current_frame = new_pos
                self.timeline.setValue(new_pos)
        # Left arrow for backward 5 seconds
        elif event.key() == Qt.Key_Left:
            if self.cap is not None and self.cap.isOpened():
                skip_frames = int(5 * self.fps)
                new_pos = max(self.current_frame - skip_frames, 0)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.current_frame = new_pos
                self.timeline.setValue(new_pos)

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateUI()
    window.showFullScreen()  # Thay thế window.show()
    sys.exit(app.exec_())
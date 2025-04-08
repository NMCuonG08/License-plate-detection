import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QSlider, QStyle, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon


class VideoModeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.playing = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Player')
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

        title = QLabel('Video Player')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))
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

        # Khu vực hiển thị video
        self.video_frame = QLabel("Chưa có video nào được mở.\nNhấn 'Mở video' để bắt đầu.")
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setStyleSheet("""
QLabel {
    background-color: #222;
    color: #aaa;
    border-radius: 10px;
    padding: 20px;
    font-size: 18px;
}
""")
        self.video_frame.setMinimumHeight(400)

        # Thời gian hiện tại / tổng
        time_layout = QHBoxLayout()
        self.current_time = QLabel("00:00")
        self.total_time = QLabel("00:00")
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 100)

        time_layout.addWidget(self.current_time)
        time_layout.addWidget(self.timeline, 1)
        time_layout.addWidget(self.total_time)

        # Nút điều khiển
        controls_layout = QHBoxLayout()

        # Nút mở file
        open_button = QPushButton("Mở video")
        open_button.setFixedHeight(40)
        open_button.setStyleSheet("""
QPushButton {
    background-color: #e74c3c;
    color: white;
    border-radius: 5px;
    padding: 5px 15px;
}
""")
        open_button.clicked.connect(self.open_video)

        # Nút điều khiển phát
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
""")
        self.play_button.clicked.connect(self.toggle_play)

        # Nút stop
        stop_button = QPushButton("⏹ Dừng")
        stop_button.setFixedHeight(40)
        stop_button.setStyleSheet("""
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 5px;
}
""")

        # Âm lượng
        volume_label = QLabel("Âm lượng:")
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setFixedWidth(100)
        volume_slider.setValue(70)

        controls_layout.addWidget(open_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(stop_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(volume_label)
        controls_layout.addWidget(volume_slider)

        # Thêm các thành phần vào layout chính
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.video_frame, 1)
        main_layout.addLayout(time_layout)
        main_layout.addLayout(controls_layout)

        # Thêm widget chính vào cửa sổ
        self.setCentralWidget(central_widget)

        # Thiết lập style chung
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

    def open_video(self):
        self.video_frame.setText("Đã mở video (chế độ giả lập)")
        self.current_time.setText("00:00")
        self.total_time.setText("05:30")

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("⏸ Tạm dừng")
        else:
            self.play_button.setText("▶ Phát")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoModeApp()
    sys.exit(app.exec_())
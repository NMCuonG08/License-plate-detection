import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGridLayout, 
                           QPushButton, QLabel, QSizePolicy, QHBoxLayout, QFrame, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QSize, QPoint, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QBrush, QColor, QPen, QPolygon, QImage
import pytesseract
import cv2
import time
from datetime import timedelta

class LoadingOverlay(QWidget):
    def __init__(self, parent=None, message="Đang tải ứng dụng..."):
        super(LoadingOverlay, self).__init__(parent)
        self.parent = parent
        self.message = message
        self.setup_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.angle = 0
        
    def setup_ui(self):
        # Set up transparent overlay
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 120);")
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add spacer to center the content
        layout.addStretch(1)
        
        # Create container for spinner and message
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px;")
        container.setFixedSize(250, 150)
        
        # Add spinner placeholder
        self.spinner = QLabel()
        self.spinner.setFixedSize(64, 64)
        self.spinner.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.spinner, alignment=Qt.AlignCenter)
        
        # Add message
        self.label = QLabel(self.message)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #333; font-size: 16px; font-weight: bold;")
        container_layout.addWidget(self.label)
        
        # Add container to main layout
        layout.addWidget(container, alignment=Qt.AlignCenter)
        
        # Add bottom spacer
        layout.addStretch(1)
        
    def update_animation(self):
        # Rotate the spinner
        self.angle = (self.angle + 10) % 360
        
        # Create spinner animation
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw spinner
        pen = QPen(QColor("#3498db"), 4)
        painter.setPen(pen)
        painter.drawArc(4, 4, 56, 56, self.angle * 16, 270 * 16)
        painter.end()
        
        self.spinner.setPixmap(pixmap)
        
    def start_animation(self):
        self.timer.start(50)  # Update every 50ms
        
    def stop_animation(self):
        self.timer.stop()
        
    def showEvent(self, event):
        # Resize to parent size
        self.resize(self.parent.size())
        self.start_animation()
        
    def paintEvent(self, event):
        # Make sure overlay is always sized to parent
        if self.parent:
            self.resize(self.parent.size())

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.active_process = None  # Lưu process đang chạy
        self.current_window = None  # Lưu cửa sổ ứng dụng hiện tại
        self.detected_plates = []  # Lưu danh sách biển số đã nhận diện
        self.cap = None  # Đối tượng video capture
        self.current_frame = 0  # Frame hiện tại
        self.fps = 0  # FPS của video
        self.video_path = None  # Đường dẫn video
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Digital Media Hub')
        self.showFullScreen()  # Hiển thị toàn màn hình
        
        # Layout chính
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Tiêu đề và nút đóng
        header_layout = QHBoxLayout()
        
        # Nút đóng X
        close_button = QPushButton("X")
        close_button.setFont(QFont('Segoe UI', 14, QFont.Bold))
        close_button.setFixedSize(50, 50)
        close_button.setStyleSheet('''
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 25px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        ''')
        close_button.clicked.connect(self.close)
        
        # Tiêu đề
        header = QLabel('Digital Media Hub')
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI Light', 36))
        header.setStyleSheet('color: #333; margin: 30px;')
        
        # Thêm vào layout header
        header_layout.addStretch(1)
        header_layout.addWidget(header)
        header_layout.addStretch(1)
        header_layout.addWidget(close_button)
        
        main_layout.addLayout(header_layout)
        
        # Thêm 4 dòng Digital Media Pub
        pub_layout = QVBoxLayout()
        pub_layout.setSpacing(10)
        
        pub_titles = [
            "Nguyen Manh Cuong - 22110015",
            "Le Cong Bao- 22110009",
            "Thai Bao Nhan - 22110057",
            "Le Truong Thinh - 22110071"
        ]
        
        for title in pub_titles:
            pub_label = QLabel(title)
            pub_label.setAlignment(Qt.AlignCenter)
            pub_label.setFont(QFont('Segoe UI', 14))
            pub_label.setStyleSheet('color: #555; margin: 5px;')
            pub_layout.addWidget(pub_label)
        
        # Thêm đường kẻ phân cách
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        separator.setStyleSheet("background-color: #ddd; margin: 20px 0;")
        
        main_layout.addLayout(pub_layout)
        main_layout.addWidget(separator)
        
        # Grid cho các nút
        grid_layout = QGridLayout()
        grid_layout.setSpacing(50)  # Tăng khoảng cách giữa các nút
        
        # Tạo 4 nút với biểu tượng và tên
        self.create_app_button(grid_layout, 0, 0, "Image Editor", "python lp_image.py ", "image_icon.png", "#3498db")
        self.create_app_button(grid_layout, 0, 1, "Video Player", "C:\\Users\\ASUS\\Python310\\python.exe D:\\XULYANH\\License-Plate-Recognition\\LicensePlateUI.py", "video_icon.png", "#e74c3c")

        
        main_layout.addLayout(grid_layout)
        main_layout.addStretch(1)
        
        # Chân trang
        footer = QLabel('© 2025 Digital Media Hub | Press ESC to exit')
        footer.setAlignment(Qt.AlignCenter)
        footer.setFont(QFont('Segoe UI', 10))
        footer.setStyleSheet('color: #666; margin: 20px;')
        main_layout.addWidget(footer)
        
        self.setLayout(main_layout)
        self.setStyleSheet('''
            QWidget {
                background-color: #f5f5f5;
            }
        ''')
    
    def create_app_button(self, layout, row, col, text, app_file, icon_file, color_code):
        # Container cho nút và nhãn
        container = QWidget()
        button_layout = QVBoxLayout(container)
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(15)  # Khoảng cách giữa nút và text
        
        # Tạo nút với biểu tượng
        button = QPushButton()
        
        # Thử tải icon từ file
        icon_path = os.path.join(os.path.dirname(__file__), "icons", icon_file)
        if not os.path.exists(icon_path):
            # Tạo thư mục icons nếu chưa tồn tại
            os.makedirs(os.path.join(os.path.dirname(__file__), "icons"), exist_ok=True)
            
            # Sử dụng biểu tượng mặc định dựa trên loại ứng dụng
            if "Image" in text:
                pixmap = QPixmap(250, 250)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(color_code)))
                painter.drawEllipse(25, 25, 200, 200)
                painter.setPen(QPen(Qt.white, 8))
                painter.drawLine(100, 125, 150, 125)
                painter.drawLine(125, 100, 125, 150)
                painter.end()
                button.setIcon(QIcon(pixmap))
            elif "Video" in text:
                pixmap = QPixmap(250, 250)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(color_code)))
                painter.drawEllipse(25, 25, 200, 200)
                painter.setBrush(QBrush(Qt.white))
                points = [QPoint(100, 80), QPoint(100, 170), QPoint(170, 125)]
                painter.drawPolygon(QPolygon(points))
                painter.end()
                button.setIcon(QIcon(pixmap))
            elif "Batch" in text:
                pixmap = QPixmap(250, 250)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(color_code)))
                painter.drawEllipse(25, 25, 200, 200)
                painter.setPen(QPen(Qt.white, 8))
                painter.drawRect(85, 85, 80, 80)
                painter.end()
                button.setIcon(QIcon(pixmap))
            else:
                pixmap = QPixmap(250, 250)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(color_code)))
                painter.drawEllipse(25, 25, 200, 200)
                painter.setPen(QPen(Qt.white, 8))
                painter.drawEllipse(100, 100, 50, 50)
                painter.end()
                button.setIcon(QIcon(pixmap))
        else:
            button.setIcon(QIcon(icon_path))
        
        # Thiết lập kích thước lớn cho nút và biểu tượng
        button.setMinimumSize(QSize(250, 250))
        button.setIconSize(QSize(200, 200))
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Style cho nút với hiệu ứng hover đổi ngay màu nền
        button.setStyleSheet(f'''
            QPushButton {{
                background-color: white;
                border-radius: 20px;
                border: 3px solid {color_code};
            }}
            QPushButton:hover {{
                background-color: {color_code};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color_code, 0.8)};
            }}
        ''')
        
        # Kết nối click nút với hàm mở file
        button.clicked.connect(lambda: self.open_application(app_file))
        
        # Nhãn cho nút
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont('Segoe UI', 14))
        label.setStyleSheet(f'color: {color_code}; font-weight: 600;')
        
        # Thêm nút và nhãn vào layout
        button_layout.addWidget(button)
        button_layout.addWidget(label)
        
        layout.addWidget(container, row, col)
    
    def lighten_color(self, color_hex, factor=0.5):
        """Tạo màu sáng hơn từ mã hex"""
        # Bỏ dấu # nếu có
        if color_hex.startswith('#'):
            color_hex = color_hex[1:]
            
        # Chuyển hex sang RGB
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        
        # Tăng các giá trị màu
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        # Chuyển lại thành hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def darken_color(self, color_hex, factor=0.8):
        """Tạo màu tối hơn từ mã hex"""
        # Bỏ dấu # nếu có
        if color_hex.startswith('#'):
            color_hex = color_hex[1:]
            
        # Chuyển hex sang RGB
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        
        # Giảm các giá trị màu
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))
        
        # Chuyển lại thành hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def open_application(self, app_file):
        try:
            # Create and show loading overlay
            self.loading_overlay = LoadingOverlay(self, f"Đang mở {app_file.split('/')[-1].split('.')[0]}...")
            self.loading_overlay.show()
            
            # Use QTimer to allow overlay to be rendered before starting process
            QTimer.singleShot(100, lambda: self._start_application_process(app_file))
            
        except Exception as e:
            print(f"Error opening application: {e}")
            if hasattr(self, 'loading_overlay'):
                self.loading_overlay.close()
            QMessageBox.critical(self, "Lỗi", f"Không thể mở {app_file}: {str(e)}")

    def _start_application_process(self, app_file):
        try:
            if app_file.startswith("C:\\Users\\ASUS\\Python310\\python.exe"):
                # Tách đường dẫn Python và đường dẫn script
                parts = app_file.split(" ", 1)
                python_path = parts[0]
                script_path = parts[1] if len(parts) > 1 else ""
                
                # Chạy bằng danh sách tham số riêng biệt, không dùng shell=True
                process = subprocess.Popen([python_path, script_path])
                self.active_process = process
                print(f"Process started with PID: {process.pid}")
            elif " " in app_file and app_file.startswith("python"):
                # Set working directory to the script's directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"Running command: {app_file} in directory {script_dir}")
                
                # Run process with the correct working directory
                process = subprocess.Popen(app_file, shell=True, cwd=script_dir)
                self.active_process = process
                
                # Show success message
                print(f"Process started with PID: {process.pid}")
            else:
                # Original handling for simple file names
                app_path = os.path.join(os.path.dirname(__file__), app_file)
                
                # If file doesn't exist, create it
                if not os.path.exists(app_path):
                    self.create_app_file(app_path, app_file)
                
                # Run as subprocess
                process = subprocess.Popen([sys.executable, app_path])
                self.active_process = process
            
            # Hide loading overlay after a delay
            QTimer.singleShot(3000, self._hide_loading_overlay)
            
        except Exception as e:
            print(f"Error in _start_application_process: {e}")
            self._hide_loading_overlay()
            QMessageBox.critical(self, "Lỗi", f"Không thể mở {app_file}: {str(e)}")

    def _hide_loading_overlay(self):
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.stop_animation()
            self.loading_overlay.close()
            self.loading_overlay = None

    def create_app_file(self, app_path, app_filename):
        # Tạo một file ứng dụng mẫu nếu chưa tồn tại
        app_name = app_filename.split('.')[0]
        app_title = ""
        
        if "Image" in app_name:
            app_title = "Image Editor"
            color_code = "#3498db"
        elif "Video" in app_name:
            app_title = "Video Player"
            color_code = "#e74c3c"
        elif "Batch" in app_name:
            app_title = "Batch Processing"
            color_code = "#2ecc71"
        else:
            app_title = "Night Mode"
            color_code = "#9b59b6"
            
        # Tạo template không có thụt lề thừa
        template = f'''import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class {app_name}App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('{app_title}')
        self.setWindowFlag(Qt.FramelessWindowHint)  # Ẩn viền cửa sổ
        self.showFullScreen()  # Mở toàn màn hình
        
        # Widget chính
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Header với nút quay lại và nút đóng
        header_layout = QHBoxLayout()
        
        # Nút quay lại
        back_button = QPushButton("← Quay lại")
        back_button.setFont(QFont('Segoe UI', 12))
        back_button.setStyleSheet("""
QPushButton {{
    background-color: {color_code};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
}}
QPushButton:hover {{
    background-color: #2980b9;
}}
""")
        back_button.clicked.connect(self.close)
        
        # Tiêu đề
        title = QLabel('{app_title}')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI Light', 28))
        title.setStyleSheet(f'color: {color_code};')
        
        # Nút đóng X
        close_button = QPushButton("X")
        close_button.setFont(QFont('Segoe UI', 12, QFont.Bold))
        close_button.setFixedSize(40, 40)
        close_button.setStyleSheet("""
QPushButton {{
    background-color: #e74c3c;
    color: white;
    border-radius: 20px;
}}
QPushButton:hover {{
    background-color: #c0392b;
}}
""")
        close_button.clicked.connect(self.close)
        
        header_layout.addWidget(back_button)
        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(close_button)
        
        # Nội dung
        content = QLabel('Đây là ứng dụng {app_title}\\nThêm các chức năng của bạn ở đây.')
        content.setAlignment(Qt.AlignCenter)
        content.setFont(QFont('Segoe UI', 18))
        content.setStyleSheet('margin: 50px; color: #555;')
        
        # Thêm các thành phần vào layout chính
        main_layout.addLayout(header_layout)
        main_layout.addWidget(content, 1)
        
        # Thêm widget chính vào cửa sổ
        self.setCentralWidget(central_widget)
        
        # Thiết lập style chung
        self.setStyleSheet("""
QMainWindow, QWidget {{
    background-color: white;
}}
""")
    
    def keyPressEvent(self, event):
        # Thoát ứng dụng khi nhấn ESC
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = {app_name}App()
    sys.exit(app.exec_())'''
    
        # Tạo thư mục cha nếu cần
        os.makedirs(os.path.dirname(app_path), exist_ok=True)
        
        # Ghi nội dung vào file
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(template)
    
    def update_detection_results(self):
        """Cập nhật và hiển thị kết quả nhận diện biển số tốt nhất"""
        if not self.detected_plates:
            return
            
        # Nhóm các biển số dựa theo vị trí (biển số giống nhau xuất hiện ở các frame khác nhau)
        plate_groups = {}
        
        for plate in self.detected_plates:
            # Tạo key dựa trên vị trí biển số
            position_key = f"{plate['x']//50}_{plate['y']//50}"  # Nhóm theo lưới 50px
            
            if position_key not in plate_groups:
                plate_groups[position_key] = []
                
            plate_groups[position_key].append(plate)
        
        # Tìm kết quả tốt nhất cho mỗi nhóm biển số
        best_plates = []
        
        for group in plate_groups.values():
            # Các tiêu chí để chọn biển số tốt nhất:
            # 1. Độ dài chuỗi nhận dạng được (ưu tiên dài hơn)
            # 2. Chuỗi có số lượng ký tự hợp lệ nhiều hơn
            
            def plate_quality(p):
                text = p['text']
                # Kiểm tra định dạng biển số hợp lệ: 2-3 chữ + 5-6 số hoặc các dạng phổ biến
                digit_count = sum(c.isdigit() for c in text)
                letter_count = sum(c.isalpha() for c in text)
                
                # Thang điểm chất lượng
                quality = len(text) * 10  # Ưu tiên biển số dài
                
                # Biển số tiêu chuẩn thường có 7-10 ký tự
                if 7 <= len(text) <= 10:
                    quality += 50
                    
                # Biển số VN thường có 1-3 chữ cái và 5-7 số
                if 1 <= letter_count <= 3 and 5 <= digit_count <= 7:
                    quality += 100
                    
                return quality
                
            best_plate = max(group, key=plate_quality)
            best_plates.append(best_plate)
        
        # Tạo widget hiển thị kết quả nếu chưa có
        if not hasattr(self, 'results_panel'):
            self.results_panel = QFrame()
            self.results_panel.setFrameShape(QFrame.StyledPanel)
            self.results_panel.setStyleSheet("""
                QFrame {
                    background-color: #2c3e50;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
            self.results_layout = QVBoxLayout(self.results_panel)
            
            # Thêm tiêu đề
            title_label = QLabel("KẾT QUẢ NHẬN DIỆN BIỂN SỐ TỐT NHẤT")
            title_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: #3498db;")
            self.results_layout.addWidget(title_label)
            
            # Container cho danh sách biển số
            self.plates_container = QWidget()
            self.plates_layout = QHBoxLayout(self.plates_container)
            self.results_layout.addWidget(self.plates_container)
            
            # Thêm panel kết quả vào layout chính
            main_layout = self.layout()
            main_layout.addWidget(self.results_panel)
        else:
            # Xóa các kết quả cũ
            for i in reversed(range(self.plates_layout.count())): 
                self.plates_layout.itemAt(i).widget().setParent(None)
        
        # Thêm các kết quả biển số tốt nhất vào panel
        for plate in best_plates:
            # Tạo widget hiển thị kết quả biển số
            plate_widget = QFrame()
            plate_widget.setFixedSize(180, 100)
            plate_widget.setFrameShape(QFrame.StyledPanel)
            plate_widget.setStyleSheet("background-color: #34495e; border-radius: 5px;")
            
            plate_layout = QVBoxLayout(plate_widget)
            
            # Hiển thị văn bản biển số
            text_label = QLabel(plate['text'])
            text_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setStyleSheet("color: #2ecc71;")
            plate_layout.addWidget(text_label)
            
            # Hiển thị thumbnail của biển số nếu có
            if self.cap:
                # Lưu vị trí hiện tại
                current_pos = self.current_frame
                
                # Tìm frame chứa biển số
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, plate['frame'])
                ret, frame = self.cap.read()
                
                if ret:
                    # Cắt vùng biển số
                    x, y, w, h = plate['x'], plate['y'], plate['w'], plate['h']
                    plate_img = frame[y:y+h, x:x+w]
                    
                    # Resize và hiển thị
                    if plate_img.size > 0:
                        plate_img = cv2.resize(plate_img, (160, 60))
                        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                        h, w, c = plate_img.shape
                        qimg = QImage(plate_img.data, w, h, w*c, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        
                        img_label = QLabel()
                        img_label.setPixmap(pixmap)
                        img_label.setAlignment(Qt.AlignCenter)
                        plate_layout.addWidget(img_label)
                
                # Khôi phục vị trí hiện tại
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            
            # Thêm thông tin frame
            frame_label = QLabel(f"Frame: {plate['frame']}")
            frame_label.setAlignment(Qt.AlignCenter)
            frame_label.setStyleSheet("color: #bdc3c7;")
            plate_layout.addWidget(frame_label)
            
            # Thêm vào layout chính
            self.plates_layout.addWidget(plate_widget)
        
        # Thêm nút xuất kết quả
        if not hasattr(self, 'export_button'):
            self.export_button = QPushButton("Xuất kết quả")
            self.export_button.setStyleSheet("""
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
            self.export_button.clicked.connect(self.export_results)
            self.results_layout.addWidget(self.export_button, alignment=Qt.AlignCenter)

    def export_results(self):
        """Xuất kết quả nhận diện ra file"""
        if not self.detected_plates:
            QMessageBox.information(self, "Thông báo", "Chưa có kết quả nhận diện nào để xuất.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Lưu kết quả", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if not filename:
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("DANH SÁCH BIỂN SỐ XE ĐÃ NHẬN DIỆN\n")
                f.write("=================================\n")
                f.write("Video: " + (self.video_path or "Không có") + "\n")
                f.write("Thời gian nhận diện: " + time.strftime("%d/%m/%Y %H:%M:%S") + "\n\n")
                
                # Nhóm các biển số
                plate_groups = {}
                for plate in self.detected_plates:
                    position_key = f"{plate['x']//50}_{plate['y']//50}"
                    if position_key not in plate_groups:
                        plate_groups[position_key] = []
                    plate_groups[position_key].append(plate)
                
                # Lấy kết quả tốt nhất từ mỗi nhóm
                for i, group in enumerate(plate_groups.values(), 1):
                    best_plate = max(group, key=lambda p: len(p['text']))
                    time_str = str(timedelta(seconds=int(best_plate['frame'] / self.fps))) if self.fps > 0 else "N/A"
                    f.write(f"{i}. Biển số: {best_plate['text']}\n")
                    f.write(f"   Xuất hiện tại: {time_str}\n")
                    f.write(f"   Frame: {best_plate['frame']}\n\n")
                    
            QMessageBox.information(self, "Thông báo", f"Đã xuất kết quả thành công!\nĐường dẫn: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất kết quả: {str(e)}")
    
    def toggle_detection(self):
        # ... code hiện tại ...
        
        if not self.detection_mode:
            # Khi tắt chế độ nhận diện, hiển thị kết quả tốt nhất
            self.update_detection_results()

    def stop_video(self):
        # ... code hiện tại ...
        
        # Hiển thị kết quả nhận diện nếu có
        if self.detection_mode or self.detected_plates:
            self.update_detection_results()



    def keyPressEvent(self, event):
        # Thoát ứng dụng khi nhấn ESC
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    # Đặt style ứng dụng
    app.setStyle('Fusion')
    
    window = MainWindow()
    app.exec_()
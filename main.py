import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGridLayout, 
                           QPushButton, QLabel, QSizePolicy, QHBoxLayout)
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QBrush, QColor, QPen, QPolygon

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.active_process = None  # Lưu process đang chạy
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
        self.create_app_button(grid_layout, 0, 0, "Image Editor", "ImageMode.py", "image_icon.png", "#3498db")
        self.create_app_button(grid_layout, 0, 1, "Video Player", "VideoMode.py", "video_icon.png", "#e74c3c")
        self.create_app_button(grid_layout, 1, 0, "Batch Processing", "BatchMode.py", "batch_icon.png", "#2ecc71")
        self.create_app_button(grid_layout, 1, 1, "Night Mode", "NightMode.py", "night_icon.png", "#9b59b6")
        
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
        # Đóng ứng dụng hiện tại nếu có
        if self.active_process is not None:
            try:
                self.active_process.terminate()
            except:
                pass
        
        app_path = os.path.join(os.path.dirname(__file__), app_file)
        
        # Nếu file chưa tồn tại, tạo file mới
        if not os.path.exists(app_path):
            self.create_app_file(app_path, app_file)
        
        try:
            # Sử dụng Python để chạy script - ẩn cửa sổ console
            startupinfo = None
            if os.name == 'nt':  # Chỉ trên Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Mở ứng dụng toàn màn hình
            self.active_process = subprocess.Popen(['python', app_path], 
                                                  startupinfo=startupinfo)
            
            # Ẩn cửa sổ chính (tùy chọn)
            # self.hide()
        except Exception as e:
            print(f"Lỗi khi mở file: {e}")
    
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
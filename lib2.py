#!/usr/bin/env python3

#############################
# Nhập thư viện
import os
import sys
import json
import cv2
import numpy as np
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QComboBox, QCheckBox, QSplitter,
    QStyleFactory, QDialog, QLineEdit, QFormLayout, QInputDialog, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QObject, QUrl, QTimer
from PyQt5.QtGui import QPixmap, QFont, QImage, QIcon, QPalette, QColor, QDesktopServices
import time
################################
try:
    from core.detector import FoodDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FoodDetector: {e}")
    DETECTOR_AVAILABLE = False
    
    class FoodDetector:
        def __init__(self, **kwargs):
            self.classes = ['cahukho', 'canhcai', 'canhchua', 'com', 'dauhusotca', 
                         'gachien', 'raumuongxao', 'thitkho', 'thitkhotrung', 'trungchien']
            self.prices = self.load_prices()
            
        def load_prices(self):
            try:  
                with open("food_price.json", "r") as f:
                    import json
                    data = json.load(f)
                    return data.get("prices", {})
            except Exception as e:
                print(f"Error loading prices: {e}")
                return {
                    "cahukho": 22000,
                    "canhcai": 9000,
                    "canhchua": 10000,
                    "com": 5000,
                    "dauhusotca": 16000,
                    "gachien": 25000,
                    "raumuongxao": 8000,
                    "thitkho": 17000,
                    "thitkhotrung": 18000,
                    "trungchien": 12000
                }
                
        def load_models(self, **kwargs):
            return False
            
        def detect_food(self, image, **kwargs):
            return {}
            
        def draw_detection(self, image, detections):
            return image.copy()
            
        def calculate_total(self, detections):
            return {"total": 0, "items": []}

#############################
# Nút có kiểu dáng
class StyledButton(QPushButton):
    def __init__(self, text, icon_path=None, color="#4CAF50"):
        super().__init__(text)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        
        # Thiết lập biểu tượng nếu có
        if icon_path and os.path.exists(icon_path):
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QSize(24, 24))
        
        # Xác định màu sắc cho các trạng thái
        hover_color = self._darken_color(color, 0.9)
        pressed_color = self._darken_color(color, 0.8)
        
        # Thiết lập stylesheet
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)
    
    def _darken_color(self, hex_color, factor=0.9):
        # Làm tối màu hex theo hệ số
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"


#############################
# Hiển thị hình ảnh
class ImageViewer(QWidget):
    def __init__(self, title="Hình ảnh", placeholder_text="Chưa có hình ảnh"):
        super().__init__()
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Tiêu đề
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #004d40;")
        self.layout.addWidget(self.title_label)
        
        # Hiển thị hình ảnh
        self.image_label = QLabel(placeholder_text)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            border-radius: 4px;
            background-color: #fafafa;
            color: #757575;
            font-size: 16px;
        """)
        self.layout.addWidget(self.image_label)
        
        self.setStyleSheet("""
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        """)
    
    def display_image(self, image_path=None, cv_image=None):
        # Hiển thị hình ảnh từ đường dẫn hoặc ảnh OpenCV
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Chỉnh kích thước ảnh phù hợp với nhãn
                pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                return True
            else:
                self.image_label.setText("Không thể hiển thị ảnh")
                return False
                
        elif cv_image is not None:
            try:
                height, width, channels = cv_image.shape
                bytes_per_line = channels * width
                cv_image_contiguous = np.ascontiguousarray(cv_image)
                qt_image = QImage(cv_image_contiguous.data, width, height, bytes_per_line, 
                                QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qt_image)
                pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                return True
            except Exception as e:
                self.image_label.setText(f"Lỗi hiển thị ảnh: {e}")
                return False
        else:
            self.image_label.setText("Không có ảnh để hiển thị")
            return False


#############################
# Bảng kết quả
class ResultTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(["Món ăn", "Độ tin cậy", "Giá (VND)", "Ảnh"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f9f9f9;
                selection-background-color: #e0f2f1;
                selection-color: #004d40;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #26a69a;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
    
    def add_detection(self, class_name, confidence, price, crop_image):
        # Thêm kết quả phát hiện vào bảng kèm hình thu nhỏ
        # Thêm hàng mới
        row = self.rowCount()
        self.insertRow(row)
        
        # Thêm thông tin
        self.setItem(row, 0, QTableWidgetItem(class_name))
        self.setItem(row, 1, QTableWidgetItem(f"{confidence:.2f}"))
        self.setItem(row, 2, QTableWidgetItem(f"{price:,}"))
        
        # Tạo hình thu nhỏ từ vùng cắt
        try:
            # Chuyển đổi không gian màu nếu cần
            if crop_image.shape[2] == 3:  # Nếu là ảnh BGR 3 kênh
                # OpenCV sử dụng BGR, QImage cần RGB, vì vậy đầu tiên ta chuyển đến định dạng phổ biến
                rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_image.shape
                bytes_per_line = channels * width
                
                # Đảm bảo mảng numpy liên tục
                crop_contiguous = np.ascontiguousarray(rgb_image)
                
                # Tạo QImage trực tiếp với RGB (không cần rgbSwapped vì đã chuyển đổi)
                qt_crop = QImage(crop_contiguous.data, width, height, bytes_per_line, 
                               QImage.Format_RGB888)
            else:
                # Xử lý các trường hợp khác nếu cần
                height, width, channels = crop_image.shape
                bytes_per_line = channels * width
                crop_contiguous = np.ascontiguousarray(crop_image)
                qt_crop = QImage(crop_contiguous.data, width, height, bytes_per_line, 
                               QImage.Format_RGB888)
                
            pixmap = QPixmap.fromImage(qt_crop)
            
            # Tạo nhãn cho hình thu nhỏ
            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            thumbnail_label.setAlignment(Qt.AlignCenter)
            
            # Thêm hình thu nhỏ vào bảng
            self.setCellWidget(row, 3, thumbnail_label)
            
            # Đặt chiều cao hàng để hiển thị tốt hơn
            self.setRowHeight(row, 120)
        except Exception as e:
            # Nếu tạo hình thu nhỏ thất bại, thêm thông báo lỗi
            self.setItem(row, 3, QTableWidgetItem(f"Lỗi thumbnail: {e}"))
    
    def clear_results(self):
        # Xóa tất cả kết quả khỏi bảng
        self.setRowCount(0)





#############################
# Tách ảnh thành lưới các ô nhỏ


#############################
# Luồng phát hiện
class DetectionThread(QThread):
    # Tín hiệu khi phát hiện hoàn tất
    # Tham số: danh sách phát hiện, ảnh đã xử lý
    finished = pyqtSignal(list, object)
    
    def __init__(self, model, image_path, class_mapping, prices, should_enhance_image=True):
        # Khởi tạo luồng phát hiện
        super().__init__()
        self.model = model  # Model YOLO
        self.image_path = image_path
        self.class_mapping = class_mapping
        self.prices = prices
        self.should_enhance_image = should_enhance_image
        self.min_confidence = 0.5
        self.detect_method = "grid"  # "auto", "manual", "container", "grid"
        self.grid_size = (2, 2)      # Phân chia ảnh thành lưới 2x2
        self.grid_overlap = 0.1      # Chồng lấp 10% giữa các ô lưới
        
        # Tạo thư mục crop nếu chưa tồn tại
        self.crop_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop")
        os.makedirs(self.crop_folder, exist_ok=True)
        
        # Tải model keras cho bước phân loại thứ hai
        self.keras_model_available = False
        self.keras_model = None
        
        # Kiểm tra xem TensorFlow có sẵn không
        if not TENSORFLOW_AVAILABLE:
            print("Không thể tải model Keras vì TensorFlow không được cài đặt")
            return
            
        # Kiểm tra cả hai loại mô hình (hihi.h5 và riellogic.keras)
        keras_model_path = None
        if os.path.exists("hihi.h5"):
            keras_model_path = "hihi.h5"
            print("Found hihi.h5 model")
        elif os.path.exists("riellogic.keras"):
            keras_model_path = "riellogic.keras"
            print("Found riellogic.keras model")
            
        if keras_model_path:
            try:
                print(f"Đang tải model Keras từ {keras_model_path}...")
                self.keras_model = tf.keras.models.load_model(keras_model_path)
                # Kiểm tra xem model có tải thành công không
                if self.keras_model is not None:
                    self.keras_model_available = True
                    print(f"Keras model loaded successfully from {keras_model_path}")
                else:
                    print(f"Failed to load Keras model from {keras_model_path}")
            except Exception as e:
                self.keras_model_available = False
                print(f"Could not load Keras model: {e}")
        else:
            print("Skipping Keras model - model file not found")
    
#############################
# Tải model YOLO
class ModelLoader(QObject):
    model_loaded = pyqtSignal(object)
    
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.model_loaded.connect(self.window.set_model)
    
    def load_model(self, model_path):
        # Tải model YOLO từ đường dẫn với nhiều định dạng hơn
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("Ultralytics YOLO không được cài đặt. Hãy cài đặt bằng lệnh: pip install ultralytics")
            
            # Kiểm tra định dạng file
            file_ext = os.path.splitext(model_path)[1].lower()
            
            # Tải model dựa trên định dạng
            if file_ext in ['.pt', '.pth', '.weights', '.onnx']:
                model = YOLO(model_path)
                print(f"Đã tải model YOLO từ {model_path}")
                
                # Kiểm tra xem model có classes và names không
                if not hasattr(model, 'names') or len(model.names) == 0:
                    print("Cảnh báo: Model không có thông tin tên lớp")
                    # Gán tên lớp mặc định nếu cần
                    if not hasattr(model, 'names'):
                        model.names = {}
                
                # Thêm hoặc thay thế tên lớp tiếng Việt
                print("Đang cập nhật tên lớp Việt Nam cho model...")
                for i, food in enumerate(CLASS_NAMES):
                    # Gán cả theo index và theo tên (nếu tên đã tồn tại)
                    model.names[i] = food
                    
                    # Nếu tên đã tồn tại trong model.names values, gán lại cho nó
                    for key, val in model.names.items():
                        if val.lower() == food.lower() or val.lower().replace(" ", "") == food.lower():
                            print(f"Tìm thấy lớp phù hợp: {val} -> {food}")
                            model.names[key] = food
                
                # In thông tin lớp của model
                print(f"Model classes: {model.names}")
                
                # Đặt ngưỡng phát hiện mặc định
                model.conf = 0.25  # Ngưỡng mặc định hợp lý
                
                self.model_loaded.emit(model)
            else:
                raise ValueError(f"Không hỗ trợ định dạng file: {file_ext}")
                
        except Exception as e:
            print(f"Error loading model: {e}")  # Lỗi tải model
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.window, "Lỗi", f"Không thể tải model: {e}")


#############################
# Cửa sổ chính ứng dụng phát hiện món ăn
class FoodDetectionWindow(QMainWindow):
    load_model_requested = pyqtSignal(str)
    
    def __init__(self, config_file='food_data.json'):
        # Khởi tạo cửa sổ chính
        super().__init__()
        
        # Thiết lập kiểu ứng dụng
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        # Tải cấu hình
        self._load_config(config_file)
        
        # Thiết lập thư mục assets và icons
        self.assets_dirs = setup_assets()
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Biến
        self.current_image_path = None
        self.batch_images = []
        self.batch_results = []
        self.export_folder = "results"
        self.current_detections = []
        self.current_image = None
        self.detection_results = {}
        
        # Khởi tạo biến camera
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        
        # Khởi tạo detector 
        if DETECTOR_AVAILABLE:
            try:
                self.detector = FoodDetector()
                self.detector.load_models()
                self.status_label.setText("Detector đã được khởi tạo")
            except Exception as e:
                print(f"Error initializing detector: {e}")
                self.detector = None
                QMessageBox.warning(self, "Error", f"Could not initialize detector: {e}")
        else:
            self.detector = None
            QMessageBox.warning(self, "Warning", "Detector module not available. Limited functionality.")
    
    def _load_config(self, config_file):
        # Define default food prices in case config file is not found
        self.FOOD_PRICES = {
            "cahukho": 22000,
            "canhcai": 9000,
            "canhchua": 10000,
            "com": 5000,
            "dauhusotca": 16000,
            "gachien": 25000,
            "raumuongxao": 8000,
            "thitkho": 17000,
            "thitkhotrung": 18000,
            "trungchien": 12000
        }
        
        # Tải cấu hình từ file JSON
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    food_data = json.load(f)
                self.class_mapping = food_data.get('classes', {})
                self.prices = food_data.get('prices', self.FOOD_PRICES)
            else:
                print(f"Config file not found: {config_file}, using defaults")  # Không tìm thấy file cấu hình, sử dụng mặc định
                self.class_mapping = {}
                self.prices = self.FOOD_PRICES
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")  # Lỗi tải cấu hình, sử dụng mặc định
            self.class_mapping = {}
            self.prices = self.FOOD_PRICES
    
    def _init_ui(self):
        # Khởi tạo giao diện người dùng
        self.setWindowTitle("Phát hiện món ăn Việt Nam")
        self.setGeometry(100, 100, 1200, 800)
        
        # Thiết lập kiểu dáng chung
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
            }
            QComboBox {
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 30px;
                background-color: white;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #bdbdbd;
            }
            QCheckBox {
                font-size: 14px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        # Widget trung tâm
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Bố cục chính
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Tiêu đề ứng dụng
        title_label = QLabel("HỆ THỐNG PHÁT HIỆN MÓN ĂN VIỆT NAM")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #004d40;
            padding: 10px;
            background-color: #e0f2f1;
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(title_label)
        
        # Bố cục nội dung (hình ảnh và kết quả)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Thành phần hiển thị hình ảnh
        self.image_viewer = ImageViewer("Hình ảnh")
        
        # Bảng điều khiển
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            background-color: #e0f2f1;
            border-radius: 4px;
            padding: 8px;
        """)
        control_layout = QVBoxLayout(control_panel)
        
        # Điều khiển ngưỡng
        threshold_layout = QHBoxLayout()
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_label = QLabel("Ngưỡng phát hiện:")
        threshold_label.setStyleSheet("font-weight: bold;")
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.5"])
        self.threshold_combo.setCurrentText("0.25")  # Đặt giá trị mặc định thành 0.25 để phù hợp với YOLO
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_combo)
        
        # Điều khiển kích thước lưới
        grid_label = QLabel("Kích thước lưới:")
        grid_label.setStyleSheet("font-weight: bold;")
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["1x1", "2x2", "3x3", "4x4"])
        self.grid_combo.setCurrentText("2x2")
        threshold_layout.addWidget(grid_label)
        threshold_layout.addWidget(self.grid_combo)
        
        # Hộp kiểm tăng cường ảnh
        self.enhance_checkbox = QCheckBox("Tăng cường ảnh (CLAHE)")
        self.enhance_checkbox.setChecked(True)
        self.enhance_checkbox.setStyleSheet("font-weight: bold;")
        threshold_layout.addWidget(self.enhance_checkbox)
        threshold_layout.addStretch()
        
        control_layout.addLayout(threshold_layout)
        
        # Nút điều khiển
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Nút duyệti
        self.browse_button = StyledButton("Chọn ảnh", color="#4CAF50")
        self.browse_button.clicked.connect(self.browse_image)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'folder-open.svg')):
            self.browse_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'folder-open.svg')))
        buttons_layout.addWidget(self.browse_button)
        
        # Nút xử lý
        self.process_button = StyledButton("Xử lý ảnh", color="#2196F3")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'play.svg')):
            self.process_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'play.svg')))
        buttons_layout.addWidget(self.process_button)
        
        # Nút camera
        self.camera_button = StyledButton("Mở camera", color="#FF5722")
        self.camera_button.clicked.connect(self.open_camera)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'camera.svg')):
            self.camera_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'camera.svg')))
        buttons_layout.addWidget(self.camera_button)
        
        # Nút chọn phương pháp phát hiện
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLO (PT)", "Keras (H5)", "Both Models"])
        self.model_combo.setCurrentText("Both Models")
        self.model_combo.setStyleSheet("""
            background-color: #f0f0f0;
            padding: 5px;
            font-size: 14px;
        """)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Phương pháp:"))
        method_layout.addWidget(self.model_combo)
        control_layout.addLayout(method_layout)
        # Đã sử dụng ComboBox để thay thế nút
        
        control_layout.addLayout(buttons_layout)
        
        # Thêm chế độ Batch
        batch_layout = QHBoxLayout()
        batch_layout.setSpacing(10)
        
        # Nút chọn thư mục
        self.batch_folder_button = StyledButton("Chọn thư mục", color="#795548")
        self.batch_folder_button.clicked.connect(self.select_batch_folder)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'folder.svg')):
            self.batch_folder_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'folder.svg')))
        batch_layout.addWidget(self.batch_folder_button)
        
        # Nút xử lý hàng loạt
        self.batch_process_button = StyledButton("Xử lý hàng loạt", color="#607D8B")
        self.batch_process_button.clicked.connect(self.process_batch)
        self.batch_process_button.setEnabled(False)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'apps.svg')):
            self.batch_process_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'apps.svg')))
        batch_layout.addWidget(self.batch_process_button)
        
        # Nút xuất kết quả
        self.export_button = StyledButton("Xuất kết quả", color="#00BCD4")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'file-export.svg')):
            self.export_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'file-export.svg')))
        batch_layout.addWidget(self.export_button)
        
        # Nút thanh toán
        self.checkout_button = StyledButton("Thanh toán", color="#FF5722")
        self.checkout_button.clicked.connect(self.show_checkout)
        self.checkout_button.setEnabled(False)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'shopping-cart.svg')):
            self.checkout_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'shopping-cart.svg')))
        batch_layout.addWidget(self.checkout_button)
        
        control_layout.addLayout(batch_layout)
        
        # Thêm bảng điều khiển vào bố cục hiển thị hình ảnh
        self.image_viewer.layout.addWidget(control_panel)
        
        # Phần kết quả
        result_container = QWidget()
        result_container.setStyleSheet("""
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        """)
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_layout.setSpacing(10)
        
        # Tiêu đề kết quả
        result_title = QLabel("KẾT QUẢ PHÁT HIỆN")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("""
            font-weight: bold; 
            font-size: 16px; 
            color: #004d40;
            padding: 5px;
            background-color: #e0f2f1;
            border-radius: 4px;
        """)
        result_layout.addWidget(result_title)
        
        # Bảng kết quả
        self.result_table = ResultTable()
        result_layout.addWidget(self.result_table)
        
        # Tổng giá
        total_container = QWidget()
        total_container.setStyleSheet("""
            background-color: #e0f2f1;
            border-radius: 4px;
            padding: 10px;
        """)
        total_layout = QHBoxLayout(total_container)
        total_layout.setContentsMargins(10, 5, 10, 5)
        
        total_label = QLabel("TỔNG CỘNG:")
        total_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        self.total_price_label = QLabel("0 VND")
        self.total_price_label.setStyleSheet("color: #d32f2f; font-weight: bold; font-size: 18px;")
        
        total_layout.addWidget(total_label)
        total_layout.addStretch()
        total_layout.addWidget(self.total_price_label)
        
        result_layout.addWidget(total_container)
        
        # Thêm nội dung vào bố cục
        content_layout.addWidget(self.image_viewer, 3)
        content_layout.addWidget(result_container, 2)
        
        main_layout.addLayout(content_layout)
        
        # Thanh trạng thái
        status_container = QWidget()
        status_container.setStyleSheet("""
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        """)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(10, 5, 10, 5)
        
        status_icon = QLabel("◉")
        status_icon.setStyleSheet("color: #4CAF50; font-size: 16px;")
        
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("color: #424242;")
        
        # Thêm thông tin batch
        self.batch_info_label = QLabel("")
        self.batch_info_label.setStyleSheet("color: #673AB7; font-style: italic;")
        
        status_layout.addWidget(status_icon)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.batch_info_label)
        
        main_layout.addWidget(status_container)
    
    def set_model(self, model):
        # Thiết lập model YOLO cho phát hiện
        self.model = model
        if model:
            self.status_label.setText(f"Model đã được tải")
            # Cập nhật ngưỡng model từ giao diện
            self.update_model_threshold()
    
    def update_model_threshold(self):
        # Cập nhật ngưỡng độ tin cậy model từ điều khiển giao diện
        if self.model:
            threshold = float(self.threshold_combo.currentText())
            self.model.conf = threshold
            print(f"Model threshold set to {threshold}")  # Ngưỡng model được đặt thành
            # Ghi log chi tiết hơn để debug
            print(f"Model: {type(self.model).__name__}, Threshold: {self.model.conf}")
    
    def check_models(self):
        """Kiểm tra và tự động sử dụng các models có sẵn trong thư mục models"""
        try:
            # Xác định thư mục models
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            # Tạo thư mục models nếu chưa tồn tại
            os.makedirs(models_dir, exist_ok=True)
            
            # Kiểm tra các model trong thư mục
            model_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
            
            # Lọc các file theo đuôi
            pt_models = [f for f in model_files if f.endswith(('.pt', '.pth', '.weights'))]
            h5_models = [f for f in model_files if f.endswith(('.h5', '.keras'))]
            tflite_models = [f for f in model_files if f.endswith('.tflite')]
            
            # Hiển thị thông báo kết quả
            msg = f"Tìm thấy {len(pt_models)} model YOLO (.pt/.pth), {len(h5_models)} model H5 (.h5/.keras), và {len(tflite_models)} model TFLite (.tflite)\n\n"
            
            if pt_models:
                msg += "Model YOLO:\n" + "\n".join([f"- {m}" for m in pt_models]) + "\n\n"
            if h5_models:
                msg += "Model H5:\n" + "\n".join([f"- {m}" for m in h5_models]) + "\n\n"
            if tflite_models:
                msg += "Model TFLite:\n" + "\n".join([f"- {m}" for m in tflite_models]) + "\n\n"
            
            # Tự động tải model YOLO nếu có
            if pt_models:
                # Lấy model YOLO đầu tiên tìm được
                yolo_path = os.path.join(models_dir, pt_models[0])
                msg += f"\nĐang tự động tải model YOLO: {pt_models[0]}"
                
                self.status_label.setText(f"Đang tải model YOLO: {yolo_path}...")
                QApplication.processEvents()
                
                # Tải model YOLO
                self.load_model_requested.emit(yolo_path)
            else:
                # Kiểm tra các đường dẫn thay thế
                alt_paths = ["best.pt", "Test1.pt", "try1.pt"]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        msg += f"\nSử dụng model YOLO thay thế: {alt_path}"
                        self.status_label.setText(f"Đang tải model YOLO thay thế: {alt_path}...")
                        QApplication.processEvents()
                        self.load_model_requested.emit(alt_path)
                        break
            
            # Hiển thị thông báo với người dùng
            QMessageBox.information(self, "Kiểm tra Models", msg)
            
            if not (pt_models or any(os.path.exists(p) for p in ["best.pt", "Test1.pt", "try1.pt"])):
                QMessageBox.warning(
                    self,
                    "Không tìm thấy Model",
                    "Không tìm thấy model YOLO (.pt/.pth) trong thư mục models hoặc thư mục gốc. "
                    "Bạn có thể thêm các file model vào thư mục 'models' và kiểm tra lại."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi kiểm tra models", 
                f"Không thể kiểm tra models: {e}"
            )
                    
    def browse_image(self):
        # Mở hộp thoại để chọn ảnh
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Ảnh (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            # Lưu đường dẫn ảnh hiện tại
            self.current_image_path = file_path
            
            # Hiển thị ảnh mới
            self.image_viewer.display_image(image_path=file_path)
            
            # Xóa kết quả phát hiện cũ
            self.result_table.clear_results()
            self.current_detections = []
            self.total_price_label.setText("0 VND")
            
            # Cập nhật trạng thái
            self.process_button.setEnabled(True)
            self.checkout_button.setEnabled(False)
            self.status_label.setText(f"Đã tải ảnh: {os.path.basename(file_path)}")
            
            # Hủy các kết quả trước đó
            QApplication.processEvents()
    
    def process_image(self):
        # Kiểm tra detector
        if not self.detector:
            QMessageBox.warning(self, "Lỗi", "Detector không khả dụng")
            return
            
        # Kiểm tra ảnh
        if not self.current_image_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ảnh trước")
            return
            
        # Tải ảnh
        image = cv2.imread(self.current_image_path)
        if image is None:
            QMessageBox.warning(self, "Lỗi", f"Không thể đọc ảnh: {self.current_image_path}")
            return
            
        # Hiển thị trạng thái xử lý
        self.status_label.setText(f"Đang xử lý ảnh: {os.path.basename(self.current_image_path)}")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Lấy cài đặt phát hiện
            threshold = float(self.threshold_combo.currentText())
            grid_text = self.grid_combo.currentText()
            grid_width, grid_height = map(int, grid_text.split('x'))
            enhance = self.enhance_checkbox.isChecked()
            
            # Áp dụng tăng cường ảnh nếu được chọn
            if enhance:
                # Áp dụng CLAHE để tăng cường tương phản
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Xác định phương pháp phát hiện dựa trên lựa chọn của người dùng
            model_option = self.model_combo.currentText()
            use_h5 = model_option in ["Keras (H5)", "Both Models"]
            use_pt = model_option in ["YOLO (PT)", "Both Models"]
            
            # Thực hiện phát hiện với các lựa chọn đã chọn
            detections = self.detector.detect_food(image, use_h5=use_h5, use_pt=use_pt)
            
            # Các tham số khác như threshold và grid_size được xử lý bên trong detector.py
            self.status_label.setText(f"Phát hiện với {model_option}...")
            QApplication.processEvents()
            
            # Vẽ kết quả lên ảnh
            result_image = self.detector.draw_detection(image, detections)
            
            # Hiển thị kết quả
            self.image_viewer.display_image(cv_image=result_image)
            
            # Xóa kết quả cũ và cập nhật mới
            self.result_table.clear_results()
            
            # Tính tổng giá
            total_price = 0
            self.current_detections = []
            
            # Thêm các món vào bảng kết quả
            # Xử lý kết quả từ model PT (YOLO)
            if "pt" in detections and detections["pt"]:
                for item in detections["pt"]:
                    class_name = item.get('class', '')
                    confidence = item.get('confidence', 0)
                    price = self.prices.get(class_name, 0)
                    
                    # Crop image có thể không có trong kết quả
                    crop_img = None
                    if "bbox" in item:
                        x1, y1, x2, y2 = item["bbox"]
                        crop_img = image[y1:y2, x1:x2] if 0 <= y1 < y2 and 0 <= x1 < x2 else None
                    
                    # Thêm vào bảng
                    self.result_table.add_detection(class_name, confidence, price, crop_img)
                    
                    # Cập nhật tổng giá
                    total_price += price
                    
                    # Lưu kết quả phát hiện
                    self.current_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'price': price,
                        'crop_image': crop_img
                    })
            
            # Xử lý kết quả từ model H5 (luôn xử lý nếu có, không quan tâm model PT có kết quả hay không)
            if "h5" in detections and detections["h5"]:
                for item in detections["h5"]:
                    class_name = item.get('class', '')
                    confidence = item.get('confidence', 0)
                    price = self.prices.get(class_name, 0)
                    
                    # Thêm vào bảng
                    self.result_table.add_detection(class_name, confidence, price, None)
                    
                    # Cập nhật tổng giá
                    total_price += price
                    
                    # Lưu kết quả phát hiện
                    self.current_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'price': price,
                        'crop_image': None
                    })
            
            # Cập nhật hiển thị tổng giá
            self.total_price_label.setText(f"{total_price:,} VND")
            
            # Kích hoạt nút thanh toán nếu có phát hiện
            self.checkout_button.setEnabled(len(self.current_detections) > 0)
            
            # Cập nhật trạng thái hiển thị số lượng món ăn chính xác
            self.status_label.setText(f"Đã phát hiện {len(self.current_detections)} món ăn")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi phát hiện", f"Có lỗi khi phát hiện món ăn: {e}")
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def update_model_threshold(self):
        # Cập nhật ngưỡng độ tin cậy model từ điều khiển giao diện
        if hasattr(self.detector, 'conf'):
            threshold = float(self.threshold_combo.currentText())
            self.detector.conf = threshold
            print(f"Detector threshold set to {threshold}")
        
    def update_camera(self):
        """Cập nhật khung hình từ camera và thực hiện phát hiện"""
        if self.video_capture is None or not self.video_capture.isOpened():
            return
            
        ret, frame = self.video_capture.read()
        if not ret:
            self.status_label.setText("Không thể đọc hình ảnh từ camera")
            return
            
        # Hiển thị khung hình gốc
        self.image_viewer.display_image(cv_image=frame)
        self.current_image = frame.copy()
        QApplication.processEvents()
        
        # Xóa kết quả trước đó
        self.result_table.clear_results()
        
        # Tính tổng giá
        total_price = 0
        
        # Thêm phát hiện vào bảng
        for detection in detections:
            class_name = detection["class"]
            confidence = detection["confidence"]
            price = detection.get("price", 0)
            crop = detection["crop_image"]
            
            # Thêm vào bảng
            self.result_table.add_detection(class_name, confidence, price, crop)
            
            # Cập nhật tổng
            total_price += price
        
        # Lưu trữ detections hiện tại cho tính năng thanh toán
        self.current_detections = detections
        
        # Cập nhật hiển thị tổng giá
        self.total_price_label.setText(f"{total_price:,} VND")
        
        # Cập nhật trạng thái
        self.status_label.setText(f"Đã phát hiện {len(detections)} món ăn")
        
        # Bật nút thanh toán nếu có món ăn được phát hiện
        self.checkout_button.setEnabled(len(detections) > 0)
    
    def show_checkout(self):
        """Hiển thị hộp thoại thanh toán"""
        if not self.current_detections:
            QMessageBox.warning(self, "Thiếu thông tin", "Không có món ăn nào để thanh toán")
            return
            
        # Tính tổng giá
        total_price = sum(detection.get("price", 0) for detection in self.current_detections)
        
        # Mở hộp thoại thanh toán
        checkout_dialog = CheckoutDialog(
            parent=self,
            items=self.current_detections,
            total_price=total_price
        )
        
        # Hiển thị hộp thoại
        checkout_dialog.exec_()
    
    def open_camera(self):
        """Mở camera để phát hiện món ăn"""
        # Kiểm tra xem model YOLO có sẵn không
        if not hasattr(self, 'model') or self.model is None:
            # Tìm kiếm file model .pt
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            if os.path.exists(models_dir):
                pt_models = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.pth', '.weights'))]
                if pt_models:
                    model_path = os.path.join(models_dir, pt_models[0])
                    try:
                        from ultralytics import YOLO
                        self.model = YOLO(model_path)
                    except ImportError:
                        QMessageBox.warning(self, "Lỗi", "Không thể tải YOLO, hãy cài đặt ultralytics")
                        return
                    except Exception as e:
                        QMessageBox.warning(self, "Lỗi", f"Không thể tải model YOLO: {e}")
                        return
                else:
                    QMessageBox.warning(self, "Lỗi", "Không tìm thấy model YOLO .pt trong thư mục models")
                    return
            else:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy thư mục models")
                return

        # Hiển thị hộp thoại chọn camera
        camera_options = ["Camera mặc định (0)", "Webcam USB (1)", "Camera khác (2)"]
        choice, ok = QInputDialog.getItem(
            self, "Chọn thiết bị camera", 
            "Chọn thiết bị camera muốn sử dụng:", 
            camera_options, 0, False
        )
        
        if not ok:
            return
        
        # Xác định ID camera
        camera_id = 0  # Mặc định
        if choice == "Webcam USB (1)":
            camera_id = 1
        elif choice == "Camera khác (2)":
            camera_id = 2
        
        # Đóng camera hiện tại nếu đang mở
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            self.video_capture.release()
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.stop()
        
        # Cập nhật trạng thái
        self.status_label.setText(f"Đang mở camera {camera_id}...")
        QApplication.processEvents()
        
        # Mở camera
        self.video_capture = cv2.VideoCapture(camera_id)
        
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Lỗi", f"Không thể mở camera {camera_id}")
            return
        
        # Tạo cửa sổ với nút tắt
        cv2.namedWindow("Camera - Phát hiện món ăn")
        
        # Lấy ngưỡng phát hiện từ giao diện
        threshold = float(self.threshold_combo.currentText())
        
        # Biến để kiểm soát cửa sổ có được đóng không
        running = True
        
        try:
            while running and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    QMessageBox.warning(self, "Lỗi", "Không đọc được khung hình từ camera")
                    break
                
                # Phát hiện trực tiếp với YOLO
                try:
                    # Tạo bản sao để hiển thị
                    display_frame = frame.copy()
                    
                    # Phát hiện với model YOLO
                    results = self.model.predict(frame, conf=threshold, verbose=False)
                    
                    # Vẽ các detection lên khung hình
                    for r in results:
                        annotated_frame = r.plot()
                        display_frame = annotated_frame
                except Exception as e:
                    print(f"Lỗi khi phát hiện với YOLO: {e}")
                
                # Hiển thị thông tin hướng dẫn
                cv2.putText(display_frame, "Nhấn 'c' để chụp ảnh, 'q' để thoát", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Hiển thị khung hình
                cv2.imshow("Camera - Phát hiện món ăn", display_frame)
                
                # Xử lý đầu vào phím
                key = cv2.waitKey(1) & 0xFF
                
                # Nếu cửa sổ đã bị đóng bằng nút X
                if cv2.getWindowProperty("Camera - Phát hiện món ăn", cv2.WND_PROP_VISIBLE) < 1:
                    running = False
                    break
                    
                if key == ord('q'):
                    running = False
                    break
                elif key == ord('c'):
                    # Hiển thị thông báo đang chụp
                    snapshot_frame = frame.copy()
                    cv2.putText(snapshot_frame, "Đang chụp và xử lý...", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Camera - Phát hiện món ăn", snapshot_frame)
                    cv2.waitKey(1)
                    
                    # Lưu ảnh và sử dụng YOLO để xử lý
                    temp_path = "temp_camera.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Xóa kết quả phát hiện cũ
                    self.result_table.clear_results()
                    self.current_detections = []
                    self.total_price_label.setText("0 VND")
                    
                    # Cập nhật đường dẫn ảnh hiện tại
                    self.current_image_path = temp_path
                    
                    # Hiển thị ảnh trong giao diện
                    self.image_viewer.display_image(image_path=temp_path)
                    
                    # Hiển thị thông báo đã chụp thành công và tắt camera
                    success_text = "Chụp thành công! Đang đóng camera..."
                    cv2.putText(snapshot_frame, success_text, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Camera - Phát hiện món ăn", snapshot_frame)
                    cv2.waitKey(500)
                    
                    # Xử lý ảnh với FoodDetector
                    self.process_image()
                    
                    # Cập nhật trạng thái
                    self.status_label.setText("Đã chụp và xử lý ảnh từ camera")
                    
                    # Dừng camera
                    running = False
                    break
        except Exception as e:
            QMessageBox.critical(self, "Lỗi camera", f"Có lỗi xảy ra khi sử dụng camera: {e}")
        finally:
            # Đảm bảo giải phóng tài nguyên
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            cv2.destroyAllWindows()

    def select_batch_folder(self):
        """Chọn thư mục chứa các ảnh cần xử lý hàng loạt"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục ảnh", ""
        )
        
        if folder_path:
            # Tìm tất cả các file ảnh trong thư mục
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                                 if f.lower().endswith(ext)])
            
            if image_files:
                self.batch_images = image_files
                self.batch_process_button.setEnabled(True)
                self.batch_info_label.setText(f"Đã chọn {len(self.batch_images)} ảnh")
                
                # Hiển thị ảnh đầu tiên
                self.current_image_path = self.batch_images[0]
                self.image_viewer.display_image(image_path=self.current_image_path)
                self.process_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy ảnh trong thư mục")
                self.batch_info_label.setText("")
                self.batch_process_button.setEnabled(False)
    
    def process_batch(self):
        """Xử lý tất cả các ảnh trong thư mục đã chọn"""
        if not self.batch_images:
            QMessageBox.warning(self, "Lỗi", "Không có ảnh để xử lý")
            return
        
        # Xóa kết quả cũ
        self.batch_results = []
        
        # Tạo thư mục kết quả nếu chưa tồn tại
        export_folder = self.export_folder
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
            
        # Số lượng ảnh tối đa xử lý cùng lúc (batch size)
        batch_size = 4  # Giữ batch size nhỏ để tránh sử dụng quá nhiều bộ nhớ
        
        # Tạo progress dialog
        progress = QMessageBox()
        progress.setIcon(QMessageBox.Information)
        progress.setWindowTitle("Đang xử lý")
        progress.setText(f"Đang xử lý {len(self.batch_images)} ảnh...\nVui lòng đợi.")
        progress.setStandardButtons(QMessageBox.NoButton)
        progress.show()
        QApplication.processEvents()
        
        start_time = time.time()
        processed_images = 0
        results_data = []
        
        # Giới hạn số lượng ảnh tối đa để tránh quá tải
        max_images = min(len(self.batch_images), 50)
        if len(self.batch_images) > max_images:
            print(f"Chỉ xử lý {max_images} ảnh đầu tiên để tránh quá tải")
        
        try:
            # Xử lý từng batch ảnh
            for i in range(0, max_images, batch_size):
                # Lấy batch hiện tại
                current_batch = self.batch_images[i:i+batch_size]
                batch_results = []
                
                for img_path in current_batch:
                    img_filename = os.path.basename(img_path)
                    result_filename = f"result_{img_filename}"
                    result_path = os.path.join(export_folder, result_filename)
                    
                    # Cập nhật progress
                    progress.setText(f"Đang xử lý {processed_images+1}/{max_images} ảnh...\n{img_filename}")
                    QApplication.processEvents()
                    
                    # Đọc và xử lý ảnh
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Không thể đọc ảnh: {img_path}")
                            continue
                            
                        # Giảm kích thước ảnh lớn để tăng tốc
                        h, w = img.shape[:2]
                        if max(h, w) > 1200:
                            scale = 1200 / max(h, w)
                            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                            
                        # Tăng cường chất lượng ảnh với phiên bản tối ưu
                        if self.enhance_checkbox.isChecked():
                            img = enhance_image_clahe(img)
                            
                        # Chạy mô hình YOLO với số lượng lớp giới hạn và chỉ lấy tối đa 5 phát hiện mỗi ảnh
                        results = self.model(img, classes=[0, 1, 2, 3, 4, 5], max_det=5)
                        
                        # Hiển thị kết quả lên ảnh với độ tin cậy cao hơn để giảm phát hiện sai
                        result_img = results[0].plot(conf_thres=0.3)
                        
                        # Lưu ảnh kết quả với chất lượng thấp hơn để giảm kích thước file
                        cv2.imwrite(result_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        
                        # Chỉ lưu thông tin các phát hiện quan trọng
                        detections = []
                        if hasattr(results[0], 'boxes') and results[0].boxes:
                            boxes = results[0].boxes.data.cpu().numpy()
                            
                            # Chỉ lấy tối đa 5 phát hiện có độ tin cậy cao nhất để giảm thời gian xử lý
                            top_indices = boxes[:, 4].argsort()[-5:][::-1]
                            for idx in top_indices:
                                det = boxes[idx]
                                x1, y1, x2, y2, score, class_id = det
                                if score > 0.3:  # Tăng ngưỡng từ 0.2 lên 0.3
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    class_id = int(class_id)
                                    
                                    # Lấy tên lớp
                                    class_name = "unknown"
                                    if hasattr(self.model, 'names') and class_id in self.model.names:
                                        class_name = self.model.names[class_id]
                                    
                                    # Lấy tên món ăn từ mapping
                                    food_name = self.class_mapping.get(str(class_id), class_name)
                                    
                                    # Lấy giá
                                    price = self.prices.get(food_name, 0)
                                    
                                    detections.append({
                                        "class_id": class_id,
                                        "class_name": food_name,
                                        "confidence": float(score),
                                        "box": [int(x1), int(y1), int(x2), int(y2)],
                                        "price": price
                                    })
                        
                        # Thêm vào kết quả batch
                        image_result = {
                            "image_path": img_path,
                            "result_path": result_path,
                            "detections": detections,
                            "total_price": sum(d["price"] for d in detections)
                        }
                        
                        batch_results.append(image_result)
                        processed_images += 1
                        
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
                
                # Thêm các kết quả batch vào kết quả tổng
                results_data.extend(batch_results)
                
                # Hiển thị tiến độ
                elapsed_time = time.time() - start_time
                avg_time_per_img = elapsed_time / processed_images if processed_images > 0 else 0
                remaining_time = avg_time_per_img * (max_images - processed_images)
                
                progress.setText(f"Đã xử lý {processed_images}/{max_images} ảnh\n"
                                f"Thời gian trung bình: {avg_time_per_img:.2f}s/ảnh\n"
                                f"Thời gian còn lại: {remaining_time:.2f}s")
                QApplication.processEvents()
            
            # Lưu JSON kết quả
            json_result_path = os.path.join(export_folder, "batch_results.json")
            with open(json_result_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            # Lưu kết quả cho ứng dụng
            self.batch_results = results_data
            
            # Hiển thị kết quả trong bảng
            self.update_batch_table(results_data)
            
            # Tính tổng thời gian
            total_time = time.time() - start_time
            avg_time = total_time / processed_images if processed_images > 0 else 0
            
            # Đóng dialog tiến trình
            progress.done(0)
            
            # Thông báo thành công
            QMessageBox.information(
                self, "Hoàn tất", 
                f"Đã xử lý {processed_images} ảnh trong {total_time:.2f}s.\n"
                f"Thời gian trung bình: {avg_time:.2f}s/ảnh\n"
                f"Kết quả đã được lưu tại: {export_folder}"
            )
            
        except Exception as e:
            progress.done(0)
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý batch: {e}")
            print(f"Lỗi chi tiết: {traceback.format_exc()}")
    
    def export_results(self):
        """Xuất kết quả phát hiện ra file"""
        if not self.batch_results:
            QMessageBox.warning(self, "Lỗi", "Không có kết quả để xuất")
            return
        
        # Mở hộp thoại lưu file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu kết quả", os.path.join(self.export_folder, "food_detection_results.csv"), 
            "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Xác định định dạng dựa vào phần mở rộng
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                # Xuất file CSV
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Viết header
                    writer.writerow(['Ảnh', 'Món ăn', 'Độ tin cậy', 'Giá', 'Tọa độ'])
                    
                    # Viết dữ liệu
                    for result in self.batch_results:
                        image_name = os.path.basename(result['image_path'])
                        for det in result['detections']:
                            writer.writerow([
                                image_name,
                                det['class_name'],
                                f"{det['confidence']:.2f}",
                                det['price'],
                                f"{det['box'][0]},{det['box'][1]},{det['box'][2]},{det['box'][3]}"
                            ])
            elif ext == '.json':
                # Xuất file JSON
                import json
                export_data = []
                
                for result in self.batch_results:
                    image_data = {
                        'image_path': result['image_path'],
                        'result_path': result['result_path'],
                        'detections': []
                    }
                    
                    for det in result['detections']:
                        image_data['detections'].append({
                            'class_name': det['class_name'],
                            'confidence': det['confidence'],
                            'price': det['price'],
                            'box': det['box']
                        })
                    
                    export_data.append(image_data)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            else:
                # Mặc định xuất text
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i, result in enumerate(self.batch_results):
                        f.write(f"Ảnh {i+1}: {result['image_path']}\n")
                        f.write(f"Kết quả: {result['result_path']}\n")
                        f.write("Các món ăn phát hiện được:\n")
                        
                        total_price = 0
                        for j, det in enumerate(result['detections']):
                            f.write(f"  {j+1}. {det['class_name']} - {det['confidence']:.2f} - {det['price']} VND\n")
                            total_price += det['price']
                        
                        f.write(f"Tổng giá: {total_price} VND\n\n")
            
            # Thông báo thành công
            QMessageBox.information(self, "Xuất kết quả", f"Đã xuất kết quả thành công tới:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất kết quả: {e}")
            import traceback
            traceback.print_exc()


#############################
# Thiết lập thư mục assets và icons
def setup_assets():
    # Tạo thư mục assets nếu chưa tồn tại
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    images_dir = os.path.join(assets_dir, "images")
    icons_dir = os.path.join(assets_dir, "icons")
    
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(icons_dir, exist_ok=True)
    
    # Tập hợp các biểu tượng cần thiết
    required_icons = {
        'browse': 'folder-open.svg',
        'process': 'play.svg',
        'camera': 'camera.svg',
        'import': 'database-import.svg',
        'batch': 'apps.svg',
        'export': 'file-export.svg',
        'checkout': 'shopping-cart.svg',
        'qrcode': 'qr-code.svg',
        'save': 'device-floppy.svg',
        'food': 'soup.svg',
        'settings': 'settings.svg',
        'refresh': 'refresh.svg'
    }
    
    # Sao chép các biểu tượng từ thư mục icons/outline
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "outline")
    
    for icon_name, icon_file in required_icons.items():
        source_path = os.path.join(source_dir, icon_file)
        target_path = os.path.join(icons_dir, icon_file)
        
        # Chỉ sao chép nếu biểu tượng tồn tại
        if os.path.exists(source_path) and not os.path.exists(target_path):
            try:
                shutil.copy2(source_path, target_path)
                print(f"Copied icon: {icon_file}")
            except Exception as e:
                print(f"Error copying icon {icon_file}: {e}")
                
    return {
        'assets_dir': assets_dir,
        'images_dir': images_dir,
        'icons_dir': icons_dir
    }


#############################
# Hộp thoại thanh toán với mã QR
class CheckoutDialog(QDialog):
    def __init__(self, parent=None, items=None, total_price=0):
        super().__init__(parent)
        self.setWindowTitle("Thanh toán")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Các món ăn và tổng tiền
        self.items = items if items else []
        self.total_price = total_price
        
        # Tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        # Bố cục chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Tiêu đề
        title_label = QLabel("THANH TOÁN HÓA ĐƠN")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #004d40;
            padding: 10px;
            background-color: #e0f2f1;
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(title_label)
        
        # Bố cục phần thân
        content_layout = QHBoxLayout()
        
        # Bảng danh sách món ăn
        items_group = QGroupBox("Danh sách món ăn")
        items_layout = QVBoxLayout(items_group)
        
        # Tạo bảng món ăn
        items_table = QTableWidget(len(self.items), 3)
        items_table.setHorizontalHeaderLabels(["Món ăn", "Số lượng", "Giá"])
        items_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        items_table.verticalHeader().setVisible(False)
        items_table.setAlternatingRowColors(True)
        
        # Thêm dữ liệu vào bảng
        for i, item in enumerate(self.items):
            name_item = QTableWidgetItem(item.get('class', ''))
            qty_item = QTableWidgetItem('1')  # Mặc định số lượng là 1
            price_item = QTableWidgetItem(f"{item.get('price', 0):,}")
            
            items_table.setItem(i, 0, name_item)
            items_table.setItem(i, 1, qty_item)
            items_table.setItem(i, 2, price_item)
        
        items_layout.addWidget(items_table)
        
        # Hiển thị tổng tiền
        total_layout = QHBoxLayout()
        total_label = QLabel("Tổng cộng:")
        total_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        self.total_price_label = QLabel(f"{self.total_price:,} VND")
        self.total_price_label.setStyleSheet("color: #d32f2f; font-weight: bold; font-size: 18px;")
        
        total_layout.addWidget(total_label)
        total_layout.addStretch()
        total_layout.addWidget(self.total_price_label)
        items_layout.addLayout(total_layout)
        
        # Thông tin khách hàng
        info_group = QGroupBox("Thông tin thanh toán")
        info_layout = QFormLayout(info_group)
        
        self.customer_name = QLineEdit()
        self.customer_phone = QLineEdit()
        self.customer_email = QLineEdit()
        
        info_layout.addRow("Họ tên:", self.customer_name)
        info_layout.addRow("Số điện thoại:", self.customer_phone)
        info_layout.addRow("Email:", self.customer_email)
        
        # Phần thanh toán QR code
        qr_group = QGroupBox("Thanh toán bằng QR Code")
        qr_layout = QVBoxLayout(qr_group)
        
        # Tạo QR code
        self.qr_image_label = QLabel()
        self.qr_image_label.setAlignment(Qt.AlignCenter)
        self.qr_image_label.setMinimumSize(200, 200)
        self.qr_image_label.setStyleSheet("""
            border: 2px dashed #bdbdbd;
            border-radius: 4px;
            background-color: white;
        """)
        qr_layout.addWidget(self.qr_image_label)
        
        # Nút tạo QR
        self.generate_qr_button = QPushButton("Tạo mã QR thanh toán")
        self.generate_qr_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 14px;
        """)
        self.generate_qr_button.clicked.connect(self.generate_qr_code)
        qr_layout.addWidget(self.generate_qr_button)
        
        # Thêm nhóm thanh toán vào layout
        content_right = QVBoxLayout()
        content_right.addWidget(info_group)
        content_right.addWidget(qr_group)
        
        # Thêm vào bố cục chính
        content_layout.addWidget(items_group, 3)
        content_layout.addLayout(content_right, 2)
        main_layout.addLayout(content_layout)
        
        # Nút đóng và in hóa đơn
        buttons_layout = QHBoxLayout()
        
        self.print_button = QPushButton("In hóa đơn")
        self.print_button.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 14px;
        """)
        self.print_button.clicked.connect(self.print_receipt)
        
        self.close_button = QPushButton("Đóng")
        self.close_button.setStyleSheet("""
            background-color: #757575;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 14px;
        """)
        self.close_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(self.print_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.close_button)
        
        main_layout.addLayout(buttons_layout)
    
    def generate_qr_code(self):
        """Hiển thị mã QR thanh toán tĩnh"""
        if not self.customer_name.text():
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tên khách hàng")
            return
            
        # Sử dụng ảnh QR code tĩnh thay vì tạo động
        qr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
        
        if os.path.exists(qr_path):
            # Hiển thị QR code từ ảnh có sẵn
            pixmap = QPixmap(qr_path)
            self.qr_image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Hiển thị thông báo (QDialog không có statusBar nên ta thêm label thông báo)
            QMessageBox.information(self, "Thành công", f"Đã tạo mã QR cho khách hàng {self.customer_name.text()}")
        else:
            # Sử dụng hình ảnh mã QR được nhúng sẵn trong mã (base64 encoded)
            qr_base64 = "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"
            
            # Decode base64 to QImage
            ba = QByteArray.fromBase64(qr_base64.encode())
            img = QImage()
            img.loadFromData(ba)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(img)
            self.qr_image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def print_receipt(self):
        """In hóa đơn hoặc lưu hóa đơn dưới dạng PDF"""
        try:
            # Tạo tệp tin hóa đơn
            receipt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "assets", "images", "receipt.txt")
            
            with open(receipt_path, 'w', encoding='utf-8') as f:
                f.write("===== HÓA ĐƠN THANH TOÁN =====\n\n")
                f.write(f"Ngày giờ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Khách hàng: {self.customer_name.text()}\n")
                if self.customer_phone.text():
                    f.write(f"Số điện thoại: {self.customer_phone.text()}\n")
                if self.customer_email.text():
                    f.write(f"Email: {self.customer_email.text()}\n")
                f.write("\n")
                f.write("== DANH SÁCH MÓN ĂN ==\n")
                
                for item in self.items:
                    f.write(f"{item.get('class', 'Unknown')}: {item.get('price', 0):,} VND\n")
                
                f.write("\n")
                f.write(f"Tổng cộng: {self.total_price:,} VND\n")
                f.write("\n===== CẢM ƠN QUÝ KHÁCH =====")
            
            # Mở file hóa đơn
            QDesktopServices.openUrl(QUrl.fromLocalFile(receipt_path))
            
            QMessageBox.information(self, "In hóa đơn", 
                                   "Hóa đơn đã được lưu và mở để xem")
                                   
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo hóa đơn: {e}")
            print(f"Error creating receipt: {e}")
            import traceback
            traceback.print_exc()



# Hàm main để khởi chạy ứng dụng


def main():
    # Thiết lập thư mục assets và icons
    setup_assets()
    
    # Kiểm tra thư mục models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    # Kiểm tra files model có tồn tại
    model_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    if not model_files:
        print("Warning: No model files found in 'models' directory. Using alternative paths.")
    else:
        print(f"Found model files in models directory: {model_files}")
    
    # Tạo ứng dụng
    app = QApplication(sys.argv)
    
    # Tạo và hiển thị cửa sổ chính
    window = FoodDetectionWindow(config_file='food_price.json')
    
    # Sử dụng detector đã được tự động tải trong file main
    
    # Hiển thị cửa sổ chính
    window.show()
    
    # Chạy ứng dụng
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

#############################
# Nhập thư viện
import os
import sys
import json
import cv2
import numpy as np
import shutil
import qrcode
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QComboBox, QCheckBox, QSplitter,
    QStyleFactory, QDialog, QLineEdit, QFormLayout, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QObject, QUrl
from PyQt5.QtGui import QPixmap, QFont, QImage, QIcon, QPalette, QColor, QDesktopServices
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Nhập thư viện TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow is available!")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow not available: {e}. Will use YOLO model only.")
    # Define a dummy tf namespace to avoid errors
    class DummyTF:
        class keras:
            class models:
                @staticmethod
                def load_model(*args, **kwargs):
                    print("TensorFlow not installed - cannot load Keras model")
                    return None
            class layers:
                pass
    tf = DummyTF()


#############################
# Dữ liệu món ăn từ file food_price.json
def load_food_data():
    try:
        with open('food_price.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('classes', []), data.get('prices', {})
    except Exception as e:
        print(f"Không thể đọc file food_price.json: {e}")
        return [], {}

# Tải dữ liệu món ăn từ file
CLASS_NAMES, FOOD_PRICES = load_food_data()

# Sử dụng giá trị mặc định nếu file trống
if not CLASS_NAMES:
    CLASS_NAMES = [
        "cahukho", "canhcai", "canhchua", "com", "dauhusotca",
        "gachien", "raumuongxao", "thitkho", "thitkhotrung", "trungchien"
    ]

if not FOOD_PRICES:
    FOOD_PRICES = {
        "cahukho": 65000,
        "canhcai": 50000,
        "canhchua": 60000,
        "com": 25000,
        "dauhusotca": 55000,
        "gachien": 38000,
        "raumuongxao": 35000,
        "thitkho": 65000,
        "thitkhotrung": 70000,
        "trungchien": 45000
    }

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
# Tăng cường chất lượng ảnh
def enhance_image_clahe(img):
    """Tăng cường chất lượng ảnh bằng CLAHE với các tham số tối ưu để cân bằng giữa tốc độ và chất lượng"""
    try:
        # Giảm kích thước ảnh để tăng tốc độ xử lý nếu ảnh quá lớn
        h, w = img.shape[:2]
        if max(h, w) > 1200:  # Chỉ resize nếu ảnh quá lớn
            scale = 1200 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Chuyển sang không gian màu LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Tách kênh L (độ sáng)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE cho kênh L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Kết hợp kênh màu
        lab = cv2.merge((l, a, b))
        
        # Chuyển lại sang BGR
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Bỏ qua các bước tối ưu thêm (unsharp mask, denoising) để tăng tốc
        return enhanced_img
    
    except Exception as e:
        print(f"Error enhancing image: {e}")  # Lỗi tăng cường ảnh
        return img


#############################
# Tách ảnh thành lưới các ô nhỏ
def split_image_into_grid(image, grid_size=(2, 2), overlap=0.2):
    """
    Tách một ảnh thành lưới các ô nhỏ với khả năng chồng lấp
    
    Parameters:
    - image: Ảnh đầu vào (numpy array)
    - grid_size: Tuple (rows, cols) chỉ định số hàng và số cột của lưới
    - overlap: Tỷ lệ chồng lấp giữa các ô liền kề (0-1)
    
    Returns:
    - List các tuple (crop_img, (x1, y1, x2, y2)) chứa ảnh đã cắt và tọa độ của nó trong ảnh gốc
    """
    height, width = image.shape[:2]
    rows, cols = grid_size
    
    # Tính kích thước mỗi ô
    cell_h = int(height / rows)
    cell_w = int(width / cols)
    
    # Tính kích thước chồng lấp
    overlap_h = int(cell_h * overlap)
    overlap_w = int(cell_w * overlap)
    
    crops = []
    
    for i in range(rows):
        for j in range(cols):
            # Tính tọa độ với chồng lấp
            x1 = max(0, j * cell_w - overlap_w)
            y1 = max(0, i * cell_h - overlap_h)
            x2 = min(width, (j + 1) * cell_w + overlap_w)
            y2 = min(height, (i + 1) * cell_h + overlap_h)
            
            # Cắt và lưu ảnh
            crop_img = image[y1:y2, x1:x2].copy()
            crops.append((crop_img, (x1, y1, x2, y2)))
    
    # Thêm một crop cho toàn bộ ảnh
    crops.append((image.copy(), (0, 0, width, height)))
    
    print(f"Đã phân chia ảnh thành {len(crops)} ô lưới")
    return crops


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
    
    def run(self):
        try:
            # Đọc ảnh (OpenCV đọc ở định dạng BGR)
            image = cv2.imread(self.image_path)
            if image is None:
                self.finished.emit([], None)
                return
            
            # Giữ bản sao BGR để hiển thị đúng màu sắc
            original_bgr = image.copy()    
            
            # Tạo bản sao BGR cho việc hiển thị
            display_img = original_bgr.copy()
            
            # Chuyển đổi sang RGB chỉ cho mục đích xử lý mô hình, không phải hiển thị
            processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = processed_img.shape
            
            # Tăng cường chất lượng ảnh nếu cần (áp dụng cho ảnh xử lý, không phải hiển thị)
            if self.should_enhance_image:
                processed_img = enhance_image_clahe(processed_img)
            
            detections = []
            
            # Phát hiện dựa trên phương pháp đã chọn
            if self.detect_method == "manual":
                # Phát hiện thủ công với vùng cố định
                regions = self._create_manual_regions(processed_img, height, width)
                detections = self._process_manual_regions(regions, processed_img)
            elif self.detect_method == "container":
                # Phát hiện các hộp đựng thức ăn, sau đó phân loại
                containers = self._detect_containers(processed_img)
                detections = self._classify_container_regions(containers, width, height)
            elif self.detect_method == "grid":
                # Phát hiện dựa trên lưới các ô nhỏ
                grid_crops = split_image_into_grid(processed_img, self.grid_size, self.grid_overlap)
                detections = self._process_grid_crops(grid_crops, processed_img)
            else:
                # GIAI ĐOẠN 1: Phát hiện và crop tất cả món ăn
                print("Giai đoạn 1: Phát hiện và crop các món ăn trên khay")
                yolo_results = self.model.predict(processed_img, conf=self.min_confidence)
                
                # Debug - in ra các lớp đã phát hiện
                try:
                    for result in yolo_results:
                        boxes = result.boxes.cpu().numpy()
                        print(f"YOLO phát hiện: {len(boxes)} đối tượng")
                        print("Thông tin phát hiện:")
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = "unknown"
                            if hasattr(self.model, 'names') and cls_id in self.model.names:
                                class_name = self.model.names[cls_id]
                            print(f"  - Đối tượng {i+1}: {class_name} (cls_id={cls_id}, conf={conf:.2f})")
                except Exception as e:
                    print(f"Lỗi khi in thông tin phát hiện: {e}")
                
                # Ánh xạ giữa các nhãn phổ biến từ YOLO và món ăn Việt Nam
                fallback_mapping = {
                    'food': 'com',              # Thực phẩm -> Cơm
                    'pizza': 'com',             # Pizza -> Cơm
                    'sandwich': 'com',          # Sandwich -> Cơm
                    'hot dog': 'thitkho',       # Hot dog -> Thịt kho
                    'carrot': 'raumuongxao',    # Cà rốt -> Rau muống xào
                    'broccoli': 'raumuongxao',  # Bông cải -> Rau muống xào
                    'orange': 'raumuongxao',    # Cam -> Rau muống xào (màu tương tự)
                    'apple': 'raumuongxao',     # Táo -> Rau muống xào
                    'banana': 'raumuongxao',    # Chuối -> Rau muống xào
                    'cake': 'trungchien',       # Bánh -> Trứng chiên
                    'donut': 'trungchien',      # Bánh donut -> Trứng chiên
                    'rice': 'com',              # Cơm -> Cơm
                    'meat': 'thitkho',          # Thịt -> Thịt kho
                    'vegetable': 'raumuongxao', # Rau -> Rau muống xào
                    'fish': 'cahukho',          # Cá -> Cá kho
                    'soup': 'canhchua',         # Súp -> Canh chua
                    'bowl': 'com',              # Tô -> Cơm (thường cơm trong tô)
                    'cup': 'dauhusotca',        # Cốc -> Đậu hũ (thường có đậu hũ trong hộp)
                    'bottle': 'canhchua',       # Chai -> Canh chua (giả định)
                    'spoon': 'canhchua',        # Thìa -> Canh chua (thường ăn với thìa)
                    'fork': 'thitkho',          # Nĩa -> Thịt kho (thường ăn với nĩa)
                    'knife': 'thitkho',         # Dao -> Thịt kho
                    'dining table': 'com',      # Bàn ăn -> Cơm (thường có cơm)
                    'sandwich': 'thitkho',      # Sandwich -> Thịt kho
                    'hot dog': 'thitkho',       # Hot dog -> Thịt kho
                    'pizza': 'com',             # Pizza -> Cơm
                    'donut': 'trungchien',      # Donut -> Trứng chiên
                    'cake': 'trungchien',       # Bánh -> Trứng chiên
                    'wine glass': 'canhchua',   # Ly rượu -> Canh chua
                    'cell phone': 'dauhusotca', # Điện thoại -> Đậu hũ (hình dạng giống nhau)
                    'book': 'com',              # Sách -> Cơm (mặc định)
                    'tie': 'raumuongxao',       # Cà vạt -> Rau muống (dạng dài)
                    'teddy bear': 'thitkho',    # Gấu bông -> Thịt kho (màu nâu)
                    'object_bottom_left': 'thitkho',     # Vị trí dưới trái -> thịt kho
                    'object_bottom_right': 'trungchien', # Vị trí dưới phải -> trứng chiên
                    'object_top_left': 'canhchua',       # Vị trí trên trái -> canh chua
                    'object_top_right': 'raumuongxao',   # Vị trí trên phải -> rau muống xào
                    'object_center': 'com'               # Vị trí giữa -> cơm
                }
                
                # Lưu danh sách crop_paths để phân loại ở giai đoạn 2
                crop_paths = []
                crop_boxes = []
                crop_classes = []
                crop_confidences = []
                
                # Xử lý kết quả YOLO để crop ảnh
                for result in yolo_results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        # Lấy tọa độ hộp giới hạn và lớp
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Khi không có TensorFlow, chấp nhận tất cả các phát hiện từ YOLO
                        # ngoại trừ những đối tượng rõ ràng không liên quan
                        ignored_classes = ['person', 'chair', 'bed', 'tv', 'laptop', 'keyboard', 'remote', 'cell phone', 
                                          'clock', 'vase', 'scissors', 'toothbrush', 'toilet', 'sink', 'car', 'bicycle']
                        
                        # Khi có TensorFlow và model keras, chúng ta có thể bỏ qua nhiều lớp hơn
                        # vì model keras sẽ phân loại chính xác hơn trong giai đoạn 2
                        if TENSORFLOW_AVAILABLE and self.keras_model_available:
                            ignored_classes += ['bowl', 'dining table', 'cup', 'bottle', 'wine glass', 'fork', 'knife', 'spoon']
                        
                        # Lấy tên lớp từ model
                        model_class_name = ""
                        if hasattr(self.model, 'names') and cls_id in self.model.names:
                            model_class_name = self.model.names[cls_id]
                        
                        # Kiểm tra xem phát hiện có phải là lớp chúng ta quan tâm không
                        if model_class_name.lower() in ignored_classes:
                            print(f"Bỏ qua đối tượng {model_class_name}")
                            continue  # Bỏ qua các loại đồ vật không phải món ăn
                        
                        # Cắt vùng chứa đối tượng từ ảnh gốc
                        try:
                            crop_img = image[y1:y2, x1:x2].copy()  # Sử dụng ảnh gốc BGR, không xử lý
                        except Exception as e:
                            print(f"Error cropping image: {e}")
                            continue
                        
                        # Lưu ảnh cắt vào thư mục crop
                        crop_filename = f"crop_{i}.jpg"
                        crop_path = os.path.join(self.crop_folder, crop_filename)
                        cv2.imwrite(crop_path, crop_img)  # Lưu trực tiếp ảnh BGR
                        
                        # Lưu lại thông tin để xử lý ở bước 2
                        crop_paths.append(crop_path)
                        crop_boxes.append([x1, y1, x2, y2])
                        crop_classes.append(cls_id)  # Lưu lại cls_id
                        crop_confidences.append(conf)  # Lưu lại conf
                        
                        # Vẽ hộp giới hạn lên ảnh hiển thị
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # GIAI ĐOẠN 2: Phân loại các ảnh đã crop với CNN
                print(f"Giai đoạn 2: Phân loại {len(crop_paths)} ảnh đã crop")
                for idx, (crop_path, box, cls_id, conf) in enumerate(zip(crop_paths, crop_boxes, crop_classes, crop_confidences)):
                    x1, y1, x2, y2 = box
                    
                    # Đọc lại ảnh đã crop từ thư mục
                    crop_img = cv2.imread(crop_path)
                    if crop_img is None:
                        print(f"Không thể đọc lại ảnh crop: {crop_path}")
                        continue
                    
                    # Mặc định đặt tên lớp ban đầu là món ăn đầu tiên trong danh sách
                    if len(CLASS_NAMES) > 0:
                        final_class_name = CLASS_NAMES[0]
                    else:
                        final_class_name = "unknown_food"
                    final_conf = 0.5  # Mặc định độ tin cậy
                    
                    # Phân loại bằng model Keras nếu có
                    if self.keras_model_available:
                        keras_class, keras_conf = self._classify_with_keras(crop_img)
                        if keras_conf > 0.3:
                            final_class_name = keras_class
                            final_conf = keras_conf
                    else:
                        # Nếu không có model Keras, sử dụng chiến lược phân loại cải tiến
                        # để ánh xạ từ phát hiện YOLO sang món ăn Việt Nam
                        
                        model_class = "unknown"
                        if hasattr(self.model, 'names') and cls_id in self.model.names:
                            model_class = self.model.names[cls_id].lower()
                        
                        # Chiến lược 1: Kiểm tra nếu là món ăn Việt Nam trực tiếp
                        found_direct_match = False
                        for food_name in CLASS_NAMES:
                            if food_name.lower() == model_class:
                                final_class_name = food_name
                                found_direct_match = True
                                print(f"Trực tiếp phát hiện món ăn Việt Nam: '{model_class}'")
                                break
                        
                        # Chiến lược 2: Áp dụng bảng fallback mapping
                        if not found_direct_match and model_class in fallback_mapping:
                            final_class_name = fallback_mapping[model_class]
                            print(f"Áp dụng ánh xạ: '{model_class}' -> '{final_class_name}'")
                        
                        # Chiến lược 3: Phân loại dựa trên vị trí trong khay thức ăn và màu sắc
                        # Nhiều khay thức ăn Việt Nam có bố cục tương đối cố định
                        if not found_direct_match and model_class not in fallback_mapping:
                            # Tính toán vị trí tương đối trong ảnh
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            rel_x = center_x / width
                            rel_y = center_y / height
                            
                            # Phân tích màu sắc - màu trung bình của vùng cắt
                            avg_color = cv2.mean(crop_img)[:3]  # BGR
                            avg_b, avg_g, avg_r = avg_color
                            
                            # Phân loại dựa trên màu sắc
                            is_red = avg_r > 120 and avg_r > (avg_g * 1.2) and avg_r > (avg_b * 1.2)
                            is_green = avg_g > 120 and avg_g > (avg_r * 1.2) and avg_g > (avg_b * 1.2)
                            is_yellow = avg_r > 150 and avg_g > 150 and avg_b < 100
                            is_white = avg_r > 180 and avg_g > 180 and avg_b > 180
                            is_dark = avg_r < 80 and avg_g < 80 and avg_b < 80
                            is_brown = avg_r > 100 and avg_g > 70 and avg_g < 150 and avg_b < 80
                            
                            # Phân loại món ăn dựa trên vị trí và màu sắc
                            # Cơm (trắng hoặc góc dưới phải)
                            if is_white or (rel_x > 0.6 and rel_y > 0.6):
                                final_class_name = "com"
                                print(f"Phân loại theo màu (trắng): '{final_class_name}'")
                            # Rau muống (xanh lá)
                            elif is_green:
                                final_class_name = "raumuongxao"
                                print(f"Phân loại theo màu (xanh lá): '{final_class_name}'")
                            # Thịt kho (nâu/đỏ sẫm)
                            elif is_brown or is_dark:
                                if rel_y > 0.5:  # Nếu nằm ở nửa dưới
                                    final_class_name = "thitkho"
                                    print(f"Phân loại theo màu (nâu/tối): '{final_class_name}'")
                                else:
                                    final_class_name = "cahukho"
                                    print(f"Phân loại theo màu (nâu/tối) và vị trí trên: '{final_class_name}'")
                            # Canh chua (đỏ, nằm ở nửa trên bên trái)
                            elif is_red and rel_x < 0.5 and rel_y < 0.5:
                                final_class_name = "canhchua"
                                print(f"Phân loại theo màu (đỏ) và vị trí trên-trái: '{final_class_name}'")
                            # Trứng chiên (vàng, thường ở góc dưới phải/trái)
                            elif is_yellow:
                                final_class_name = "trungchien"
                                print(f"Phân loại theo màu (vàng): '{final_class_name}'")
                            # Dùng vị trí cho các trường hợp còn lại
                            else:
                                # Phân loại dựa trên vị trí
                                if rel_x < 0.5 and rel_y < 0.5:  # Góc trên bên trái
                                    final_class_name = "canhchua"
                                    print(f"Phân loại theo vị trí (trên-trái): '{final_class_name}'")
                                elif rel_x >= 0.5 and rel_y < 0.5:  # Góc trên bên phải
                                    final_class_name = "raumuongxao"
                                    print(f"Phân loại theo vị trí (trên-phải): '{final_class_name}'")
                                elif rel_x < 0.5 and rel_y >= 0.5:  # Góc dưới bên trái
                                    final_class_name = "thitkho"
                                    print(f"Phân loại theo vị trí (dưới-trái): '{final_class_name}'")
                                elif rel_x >= 0.5 and rel_y >= 0.5:  # Góc dưới bên phải
                                    final_class_name = "trungchien" 
                                    print(f"Phân loại theo vị trí (dưới-phải): '{final_class_name}'")
                                    
                            # Thêm logic phát hiện size - cơm thường chiếm vùng lớn nhất
                            crop_area = (x2 - x1) * (y2 - y1)
                            image_area = width * height
                            rel_area = crop_area / image_area
                            if rel_area > 0.3:  # Nếu vùng chiếm >30% ảnh, có thể là cơm
                                final_class_name = "com"
                                print(f"Phân loại theo kích thước lớn: 'com', rel_area={rel_area:.2f}")
                        
                        # Sử dụng độ tin cậy từ YOLO
                        final_conf = conf
                    
                    # Lấy giá của món ăn
                    price = self.prices.get(final_class_name, 0)
                    
                    # Thêm vào danh sách phát hiện
                    detections.append({
                        'class': final_class_name,
                        'confidence': final_conf,
                        'box': [x1, y1, x2, y2],
                        'price': price,
                        'crop_image': crop_img
                    })
                    
                    # Vẽ nền chữ để văn bản đọc được
                    text_size = cv2.getTextSize(final_class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_img, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)
                    
                    # Thêm nhãn với màu trắng
                    label = f"{final_class_name}: {final_conf:.2f}"
                    cv2.putText(display_img, label, (x1 + 5, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Lưu ảnh kết quả (display_img đã ở dạng BGR, không cần chuyển đổi)
                result_img_path = os.path.join(self.crop_folder, "last_result.jpg")
                cv2.imwrite(result_img_path, display_img)
            
            # Gửi tín hiệu hoàn thành với kết quả phát hiện và ảnh hiển thị
            self.finished.emit(detections, display_img)
            
        except Exception as e:
            print(f"Lỗi trong quá trình phát hiện: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit([], None)
    
    def _classify_with_keras(self, crop_img):
        try:
            # Kiểm tra xem TensorFlow có sẵn không và model đã được tải chưa
            if not TENSORFLOW_AVAILABLE:
                print("TensorFlow is not available for classification")
                return None, 0.0
                
            if not self.keras_model_available or not hasattr(self, 'keras_model'):
                print("Keras model is not available")
                return None, 0.0
                
            # Tiền xử lý ảnh cho model Keras
            # Đảm bảo image có định dạng màu đúng (RGB)
            img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)  # Đảm bảo đúng không gian màu
            img = cv2.resize(img_rgb, (224, 224))  # Kích thước phù hợp với model
            
            # Chuẩn hóa
            img = img / 255.0  
            img = np.expand_dims(img, axis=0)  # Batch dimension
            
            # Dự đoán với model Keras
            predictions = self.keras_model.predict(img, verbose=0)  # Tắt output verbose
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            print(f"Keras prediction: {predictions[0]}")
            print(f"Top class: {class_idx} with confidence {confidence}")
            
            # Lấy tên lớp đúng từ CLASS_NAMES
            if 0 <= class_idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_idx]
                return class_name, confidence
            else:
                print(f"Invalid class index: {class_idx}, max index should be {len(CLASS_NAMES)-1}")
                return None, 0.0
                
        except Exception as e:
            print(f"Error in Keras classification: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    def _process_grid_crops(self, grid_crops, original_img):
        """
        Xử lý từng ô lưới để phát hiện món ăn
        
        Parameters:
        - grid_crops: List các tuple (crop_img, (x1, y1, x2, y2))
        - original_img: Ảnh gốc để tham chiếu
        
        Returns:
        - List các detections
        """
        detections = []
        height, width = original_img.shape[:2]
        
        # Ánh xạ giữa các nhãn phổ biến từ YOLO và món ăn Việt Nam
        fallback_mapping = {
            'food': 'com',              # Thực phẩm -> Cơm
            'pizza': 'com',             # Pizza -> Cơm
            'sandwich': 'com',          # Sandwich -> Cơm
            'hot dog': 'thitkho',       # Hot dog -> Thịt kho
            'carrot': 'raumuongxao',    # Cà rốt -> Rau muống xào
            'broccoli': 'raumuongxao',  # Bông cải -> Rau muống xào
            'orange': 'raumuongxao',    # Cam -> Rau muống xào (màu tương tự)
            'apple': 'raumuongxao',     # Táo -> Rau muống xào
            'banana': 'raumuongxao',    # Chuối -> Rau muống xào
            'cake': 'trungchien',       # Bánh -> Trứng chiên
            'donut': 'trungchien',      # Bánh donut -> Trứng chiên
            'rice': 'com',              # Cơm -> Cơm
            'meat': 'thitkho',          # Thịt -> Thịt kho
            'vegetable': 'raumuongxao', # Rau -> Rau muống xào
            'fish': 'cahukho',          # Cá -> Cá kho
            'soup': 'canhchua',         # Súp -> Canh chua
            'bowl': 'com',              # Tô -> Cơm (thường cơm trong tô)
            'cup': 'dauhusotca',        # Cốc -> Đậu hũ (thường có đậu hũ trong hộp)
            'bottle': 'canhchua',       # Chai -> Canh chua (giả định)
            'spoon': 'canhchua',        # Thìa -> Canh chua (thường ăn với thìa)
            'fork': 'thitkho',          # Nĩa -> Thịt kho (thường ăn với nĩa)
            'knife': 'thitkho',         # Dao -> Thịt kho
            'dining table': 'com'       # Bàn ăn -> Cơm (thường có cơm)
        }
        
        # Danh sách các loại đối tượng cần bỏ qua
        ignored_classes = ['person', 'chair', 'bed', 'tv', 'laptop', 'keyboard', 'remote', 'cell phone', 
                          'clock', 'vase', 'scissors', 'toothbrush', 'toilet', 'sink', 'car', 'bicycle']
        
        # Xử lý từng ô lưới
        for idx, (crop_img, crop_coords) in enumerate(grid_crops):
            x1_crop, y1_crop, x2_crop, y2_crop = crop_coords
            
            # Lưu ảnh cắt để debug
            crop_filename = f"grid_{idx}.jpg"
            crop_path = os.path.join(self.crop_folder, crop_filename)
            cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            
            print(f"Xử lý ô lưới {idx+1}/{len(grid_crops)}: vị trí {crop_coords}")
            
            # Chạy dự đoán trên crop
            results = self.model.predict(crop_img, conf=self.min_confidence, max_det=5)
            
            # Kiểm tra kết quả
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Lấy tọa độ trong ô lưới
                    x1_local, y1_local, x2_local, y2_local = box.xyxy[0].astype(int)
                    
                    # Chuyển đổi sang tọa độ trong ảnh gốc
                    x1 = x1_local + x1_crop
                    y1 = y1_local + y1_crop
                    x2 = x2_local + x1_crop
                    y2 = y2_local + y1_crop
                    
                    # Đảm bảo không vượt quá kích thước ảnh
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    
                    # Lấy loại và độ tin cậy
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Lấy tên lớp từ model
                    model_class_name = ""
                    if hasattr(self.model, 'names') and cls_id in self.model.names:
                        model_class_name = self.model.names[cls_id]
                    
                    # Kiểm tra xem phát hiện có phải là lớp cần bỏ qua không
                    if model_class_name.lower() in ignored_classes:
                        print(f"Bỏ qua đối tượng {model_class_name}")
                        continue
                    
                    # Cắt vùng chứa đối tượng từ ảnh gốc
                    try:
                        # Chuyển RGB sang BGR vì OpenCV sử dụng BGR
                        crop_obj = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)[y1:y2, x1:x2].copy()
                    except Exception as e:
                        print(f"Error cropping image: {e}")
                        continue
                    
                    # Lưu ảnh cắt vào thư mục crop
                    food_crop_filename = f"food_grid_{idx}_{i}.jpg"
                    food_crop_path = os.path.join(self.crop_folder, food_crop_filename)
                    cv2.imwrite(food_crop_path, crop_obj)
                    
                    # Xác định tên lớp cuối cùng
                    final_class_name = CLASS_NAMES[0] if CLASS_NAMES else "unknown_food"
                    final_conf = conf
                    
                    # Thử sử dụng Keras model nếu có
                    if self.keras_model_available:
                        keras_class, keras_conf = self._classify_with_keras(crop_obj)
                        if keras_conf > 0.3:
                            final_class_name = keras_class
                            final_conf = keras_conf
                    else:
                        # Chiến lược phân loại nếu không có Keras
                        model_class = model_class_name.lower() if model_class_name else "unknown"
                        
                        # 1. Kiểm tra trực tiếp với tên món ăn Việt Nam
                        found_direct_match = False
                        for food_name in CLASS_NAMES:
                            if food_name.lower() == model_class:
                                final_class_name = food_name
                                found_direct_match = True
                                print(f"Trực tiếp phát hiện món ăn Việt Nam: '{model_class}'")
                                break
                        
                        # 2. Sử dụng bảng ánh xạ
                        if not found_direct_match and model_class in fallback_mapping:
                            final_class_name = fallback_mapping[model_class]
                            print(f"Áp dụng ánh xạ: '{model_class}' -> '{final_class_name}'")
                        
                        # 3. Phân loại dựa trên vị trí và màu sắc
                        if not found_direct_match and model_class not in fallback_mapping:
                            # Phân tích màu sắc
                            avg_color = cv2.mean(crop_obj)[:3]  # BGR
                            avg_b, avg_g, avg_r = avg_color
                            
                            # Phân loại dựa trên màu sắc
                            is_red = avg_r > 120 and avg_r > (avg_g * 1.2) and avg_r > (avg_b * 1.2)
                            is_green = avg_g > 120 and avg_g > (avg_r * 1.2) and avg_g > (avg_b * 1.2)
                            is_yellow = avg_r > 150 and avg_g > 150 and avg_b < 100
                            is_white = avg_r > 180 and avg_g > 180 and avg_b > 180
                            is_dark = avg_r < 80 and avg_g < 80 and avg_b < 80
                            is_brown = avg_r > 100 and avg_g > 70 and avg_g < 150 and avg_b < 80
                            
                            # Vị trí tương đối trong ảnh gốc
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            rel_x = center_x / width
                            rel_y = center_y / height
                            
                            # Phân loại dựa trên màu sắc và vị trí
                            if is_white or (rel_x > 0.6 and rel_y > 0.6):
                                final_class_name = "com"
                                print(f"Phân loại theo màu (trắng): '{final_class_name}'")
                            elif is_green:
                                final_class_name = "raumuongxao"
                                print(f"Phân loại theo màu (xanh lá): '{final_class_name}'")
                            elif is_brown or is_dark:
                                if rel_y > 0.5:
                                    final_class_name = "thitkho"
                                    print(f"Phân loại theo màu (nâu/tối): '{final_class_name}'")
                                else:
                                    final_class_name = "cahukho"
                                    print(f"Phân loại theo màu (nâu/tối) và vị trí trên: '{final_class_name}'")
                            elif is_red and rel_x < 0.5 and rel_y < 0.5:
                                final_class_name = "canhchua"
                                print(f"Phân loại theo màu (đỏ) và vị trí trên-trái: '{final_class_name}'")
                            elif is_yellow:
                                final_class_name = "trungchien"
                                print(f"Phân loại theo màu (vàng): '{final_class_name}'")
                            else:
                                # Phân loại dựa trên vị trí
                                if rel_x < 0.5 and rel_y < 0.5:
                                    final_class_name = "canhchua"
                                    print(f"Phân loại theo vị trí (trên-trái): '{final_class_name}'")
                                elif rel_x >= 0.5 and rel_y < 0.5:
                                    final_class_name = "raumuongxao"
                                    print(f"Phân loại theo vị trí (trên-phải): '{final_class_name}'")
                                elif rel_x < 0.5 and rel_y >= 0.5:
                                    final_class_name = "thitkho"
                                    print(f"Phân loại theo vị trí (dưới-trái): '{final_class_name}'")
                                elif rel_x >= 0.5 and rel_y >= 0.5:
                                    final_class_name = "trungchien" 
                                    print(f"Phân loại theo vị trí (dưới-phải): '{final_class_name}'")
                    
                    # Lấy giá của món ăn
                    price = self.prices.get(final_class_name, 0)
                    
                    # Thêm vào danh sách phát hiện
                    detections.append({
                        'class': final_class_name,
                        'confidence': final_conf,
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'price': price,
                        'crop_image': crop_obj
                    })
        
        return detections


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
        self.model = None
        self.current_image_path = None
        self.detection_thread = None
        self.batch_images = []
        self.batch_results = []
        self.export_folder = "results"
        self.current_detections = []
        
        # Tạo bộ tải model
        self.model_loader = ModelLoader(self)
        self.load_model_requested.connect(self.model_loader.load_model)
    
    def _load_config(self, config_file):
        # Tải cấu hình từ file JSON
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    food_data = json.load(f)
                self.class_mapping = food_data.get('classes', {})
                self.prices = food_data.get('prices', FOOD_PRICES)
                self.vietnamese_foods = food_data.get('vietnamese_foods', list(self.prices.keys()))
            else:
                print(f"Config file not found: {config_file}, using defaults")  # Không tìm thấy file cấu hình, sử dụng mặc định
                self.class_mapping = {}
                self.prices = FOOD_PRICES
                self.vietnamese_foods = list(self.prices.keys())
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")  # Lỗi tải cấu hình, sử dụng mặc định
            self.class_mapping = {}
            self.prices = FOOD_PRICES
            self.vietnamese_foods = list(self.prices.keys())
    
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
        
        # Nút nhập model
        self.import_model_button = StyledButton("Import Model", color="#9C27B0")
        self.import_model_button.clicked.connect(self.import_model)
        # Thêm icon nếu có
        if os.path.exists(os.path.join(self.assets_dirs['icons_dir'], 'database-import.svg')):
            self.import_model_button.setIcon(QIcon(os.path.join(self.assets_dirs['icons_dir'], 'database-import.svg')))
        buttons_layout.addWidget(self.import_model_button)
        
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
    
    def import_model(self):
        # Nhập model từ file do người dùng chọn
        try:
            # Sử dụng QMessageBox để hỏi loại model
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Loại model")
            msgBox.setText("Bạn muốn import loại model nào?")
            yolo_button = msgBox.addButton("YOLO Model", QMessageBox.ActionRole)
            tflite_button = msgBox.addButton("TensorFlow Lite", QMessageBox.ActionRole)
            msgBox.setDefaultButton(yolo_button)
            msgBox.exec_()
            
            clicked_button = msgBox.clickedButton()
            
            if clicked_button == yolo_button:  # YOLO Model
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Chọn file model YOLO", "", "YOLO Model Files (*.pt *.pth *.weights);;All Files (*)"
                )
                
                if file_path and os.path.exists(file_path):
                    self.status_label.setText(f"Đang tải model: {file_path}...")
                    QApplication.processEvents()
                    
                    # Phát tín hiệu khi tải hoàn tất
                    self.load_model_requested.emit(file_path)
            elif clicked_button == tflite_button:  # TensorFlow Lite model
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Chọn file model TensorFlow Lite", "", "TFLite Model Files (*.tflite);;All Files (*)"
                )
                
                if file_path and os.path.exists(file_path):
                    # Xác nhận xem người dùng có muốn sao chép file không
                    copy_msg = QMessageBox()
                    copy_msg.setIcon(QMessageBox.Question)
                    copy_msg.setWindowTitle("Import TFLite Model")
                    copy_msg.setText("Bạn có muốn sao chép file này đến '1.tflite'?")
                    copy_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    copy_msg.setDefaultButton(QMessageBox.Yes)
                    response = copy_msg.exec_()
                    
                    if response == QMessageBox.Yes:
                        import shutil
                        try:
                            shutil.copy(file_path, "riel.tflite")
                            QMessageBox.information(
                                self, "Import Thành công", 
                                f"Đã sao chép {file_path} đến riel.tflite"
                            )
                            self.status_label.setText(f"Đã tải model TFLite: {file_path}")
                        except Exception as e:
                            QMessageBox.critical(self, "Lỗi", f"Không thể sao chép file: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải model: {e}")
    
    def browse_image(self):
        # Mở hộp thoại để chọn ảnh
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Ảnh (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_viewer.display_image(image_path=file_path)
            self.process_button.setEnabled(True)
            self.status_label.setText(f"Đã tải ảnh: {os.path.basename(file_path)}")
    
    def process_image(self):
        # Xử lý ảnh hiện tại với model đã tải
        if not self.current_image_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ảnh trước")
            return
            
        if not self.model:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tải model trước")
            return
        
        # Cập nhật ngưỡng model
        self.update_model_threshold()
        
        # Cập nhật trạng thái
        self.status_label.setText(f"Đang xử lý ảnh: {os.path.basename(self.current_image_path)}")
        QApplication.processEvents()
        
        # Lấy cài đặt tăng cường và kích thước lưới
        enhance = self.enhance_checkbox.isChecked()
        
        # Phân tích kích thước lưới từ combobox
        grid_text = self.grid_combo.currentText()
        if grid_text == "1x1":
            grid_size = (1, 1)
        elif grid_text == "2x2":
            grid_size = (2, 2)
        elif grid_text == "3x3":
            grid_size = (3, 3)
        elif grid_text == "4x4":
            grid_size = (4, 4)
        else:
            grid_size = (2, 2)  # Mặc định
        
        # Tạo và bắt đầu luồng phát hiện
        self.detection_thread = DetectionThread(
            self.model, 
            self.current_image_path,
            self.class_mapping,
            self.prices,
            enhance
        )
        
        # Cập nhật các tham số bổ sung
        self.detection_thread.grid_size = grid_size
        
        self.detection_thread.finished.connect(self.on_detection_complete)
        self.detection_thread.start()
    
    def on_detection_complete(self, detections, display_img):
        # Xử lý kết quả phát hiện
        if display_img is not None:
            # Hiển thị ảnh đã xử lý
            self.image_viewer.display_image(cv_image=display_img)
        
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
        # Mở camera để phát hiện trực tiếp
        if not self.model:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tải model trước khi sử dụng camera")
            return
            
        self.status_label.setText("Đang mở camera...")
        QApplication.processEvents()
        
        # Tạo cửa sổ camera
        cv2.namedWindow("Camera - Phát hiện món ăn", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera - Phát hiện món ăn", 800, 600)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            QMessageBox.warning(self, "Lỗi", "Không thể mở camera")
            return
        
        # Hiển thị hướng dẫn
        help_text = "Nhấn 'c' để chụp ảnh, 'q' để thoát"
        help_color = (255, 255, 255)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hiển thị văn bản trợ giúp
            cv2.putText(frame, help_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, help_color, 2, cv2.LINE_AA)
            
            # Hiển thị khung hình
            cv2.imshow("Camera - Phát hiện món ăn", frame)
            
            # Xử lý đầu vào phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Hiển thị thông báo đang xử lý
                processing_frame = frame.copy()
                cv2.putText(processing_frame, "Đang xử lý ảnh...", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Camera - Phát hiện món ăn", processing_frame)
                cv2.waitKey(1)
                
                # Lưu ảnh tạm
                temp_path = "temp_camera.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Cập nhật đường dẫn ảnh hiện tại
                self.current_image_path = temp_path
                
                # Hiển thị và xử lý ảnh
                self.image_viewer.display_image(image_path=temp_path)
                self.process_image()
                
                # Cập nhật trạng thái
                self.status_label.setText("Đã chụp và xử lý ảnh từ camera")
                break
        
        # Giải phóng tài nguyên
        cap.release()
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
        if not self.model:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tải model trước")
            return
            
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
        """Tạo mã QR với thông tin thanh toán"""
        if not self.customer_name.text():
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tên khách hàng")
            return
            
        # Tạo nội dung mã QR (ở đây tạo một chuỗi văn bản đơn giản)
        # Trong thực tế, đây có thể là một URL thanh toán hoặc dữ liệu JSON
        payment_info = {
            "customer": self.customer_name.text(),
            "phone": self.customer_phone.text(),
            "email": self.customer_email.text(),
            "amount": self.total_price,
            "currency": "VND",
            "items": [{"name": item.get('class', ''), "price": item.get('price', 0)} 
                     for item in self.items],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        qr_content = json.dumps(payment_info)
        
        # Tạo mã QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_content)
        qr.make(fit=True)
        
        # Tạo hình ảnh QR
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Chuyển đổi sang định dạng để hiển thị trong Qt
        qr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "assets", "images", "payment_qr.png")
        qr_img.save(qr_path)
        
        # Hiển thị QR code
        pixmap = QPixmap(qr_path)
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


#############################
# Hàm chính
def main():
    # Điểm vào chính cho ứng dụng
    # Kiểm tra xem YOLO có sẵn không
    if not YOLO_AVAILABLE:
        print("WARNING: Ultralytics YOLO is not installed. You need to install it using:")  # YOLO chưa được cài đặt
        print("pip install ultralytics")
        # Vẫn tiếp tục và hiển thị lỗi khi chạy
    
    # Thiết lập thư mục assets và icons
    setup_assets()
    
    # Tạo ứng dụng
    app = QApplication(sys.argv)
    
    # Tạo và hiển thị cửa sổ chính
    window = FoodDetectionWindow()
    
    # Tự động tải các model
    try:
        # Danh sách các file model YOLO (.pt) có thể
        yolo_model_candidates = [
            "best.pt",
            "rielmodel.pt",
            "yolo11n.pt",
            "Test1.pt",
            "test2.pt",
            "yolov8n.pt",
            "yolov8s.pt"
        ]
        
        # Kiểm tra từng model và sử dụng model đầu tiên tìm thấy
        yolo_model_loaded = False
        for model_file in yolo_model_candidates:
            if os.path.exists(model_file):
                print(f"Auto-loading YOLO model: {model_file}")
                window.load_model_requested.emit(model_file)
                yolo_model_loaded = True
                break
        
        if not yolo_model_loaded:
            # Tìm kiếm tất cả các file .pt trong thư mục hiện tại
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt') and os.path.isfile(f)]
            if pt_files:
                # Sắp xếp theo kích thước file (ưu tiên file nhỏ hơn vì thường là các model đã tối ưu)
                pt_files.sort(key=lambda x: os.path.getsize(x))
                model_file = pt_files[0]
                print(f"Found and auto-loading YOLO model: {model_file}")
                window.load_model_requested.emit(model_file)
                yolo_model_loaded = True
            else:
                print("No YOLO model found in the current directory. Please use the Import Model button.")
            
            # Danh sách các file H5/Keras model có thể có
            keras_model_candidates = [
                "hihi.h5",  # Ưu tiên model mới "hihi.h5"
                "riellogic.keras",
                "model.h5",
                "food_model.h5"
            ]
            
            # Tự động tìm tất cả các file .h5
            h5_files = [f for f in os.listdir('.') if f.endswith('.h5') and os.path.isfile(f)]
            for h5_file in h5_files:
                if h5_file not in keras_model_candidates:
                    keras_model_candidates.append(h5_file)
            
            # Tự động sao chép model Keras nếu cần
            keras_model_loaded = False
            for keras_file in keras_model_candidates:
                if os.path.exists(keras_file):
                    keras_target = "hihi.h5"
                    if keras_file != keras_target:
                        try:
                            print(f"Auto-copying Keras model from {keras_file} to {keras_target}")
                            shutil.copy(keras_file, keras_target)
                            keras_model_loaded = True
                            break
                        except Exception as e:
                            print(f"Error copying Keras model: {e}")
                    else:
                        print(f"Keras model already exists at {keras_target}")
                        keras_model_loaded = True
                        break
                    
            if not keras_model_loaded:
                print("No Keras model found in the current directory.")
                
            # Danh sách các file TensorFlow Lite (.tflite) có thể
            tflite_model_candidates = [
                "riel.tflite",
                "vietnamese_food.tflite",
                "food_classifier.tflite",
                "classifier.tflite"
            ]
            
            # Tự động tìm tất cả các file .tflite
            tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite') and os.path.isfile(f)]
            for tflite_file in tflite_files:
                if tflite_file not in tflite_model_candidates:
                    tflite_model_candidates.append(tflite_file)
            
            # Không cần emit signal, chỉ cần sao chép file vào vị trí mặc định
            tflite_model_loaded = False
            for tflite_file in tflite_model_candidates:
                if os.path.exists(tflite_file):
                    tflite_target = "riel.tflite"
                    if tflite_file != tflite_target:
                        import shutil
                        try:
                            print(f"Auto-copying TFLite model from {tflite_file} to {tflite_target}")
                            shutil.copy(tflite_file, tflite_target)
                            tflite_model_loaded = True
                            break
                        except Exception as e:
                            print(f"Error copying TFLite model: {e}")
                    else:
                        print(f"TFLite model already exists at {tflite_target}")
                        tflite_model_loaded = True
                        break
            
            if not tflite_model_loaded:
                print("No TensorFlow Lite model found in the current directory.")
    
    except Exception as e:
        print(f"Error during auto-loading models: {e}")
        import traceback
        traceback.print_exc()
    
    # Hiển thị cửa sổ chính
    window.show()
    
    # Chạy ứng dụng
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
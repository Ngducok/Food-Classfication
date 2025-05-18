import cv2
import numpy as np
import json
import os

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. H5 model functionality will be disabled.")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Ultralytics YOLO not available. PT model functionality will be disabled.")

class FoodDetector:
    def __init__(self, h5_model_path="models/model.h5", pt_model_path="models/best.pt"):
        # Try alternate paths if the provided ones don't exist
        if not os.path.exists(h5_model_path):
            alt_h5_paths = ["hihi.h5", "model.h5", "./hihi.h5"]
            for alt_path in alt_h5_paths:
                if os.path.exists(alt_path):
                    h5_model_path = alt_path
                    print(f"Using alternative H5 model path: {h5_model_path}")
                    break
        
        if not os.path.exists(pt_model_path):
            alt_pt_paths = ["Test1.pt", "try1.pt", "best.pt", "./Test1.pt"]
            for alt_path in alt_pt_paths:
                if os.path.exists(alt_path):
                    pt_model_path = alt_path
                    print(f"Using alternative PT model path: {pt_model_path}")
                    break
                    
        self.h5_model_path = h5_model_path
        self.pt_model_path = pt_model_path
        self.h5_model = None
        self.pt_model = None
        self.classes = ['cahukho', 'canhcai', 'canhchua', 'com', 'dauhusotca', 
                      'gachien', 'raumuongxao', 'thitkho', 'thitkhotrung', 'trungchien']
        self.prices = self.load_prices()
        self.h5_input_size = 224
        self.h5_input_type = "rgb"
        self.model_config = {}
        
    def load_prices(self):
        price_paths = ["data/food_price.json", "food_price.json", "./food_price.json"]
        
        for path in price_paths:
            try:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                        print(f"Loaded prices from {path}")
                        return data.get("prices", {})
            except Exception as e:
                print(f"Error loading prices from {path}: {e}")
                
        print("Using default prices")
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
    
    def load_models(self, use_h5=True, use_pt=True):
        model_loaded = False
        h5_loaded = False
        pt_loaded = False
        
        if use_h5 and TF_AVAILABLE and os.path.exists(self.h5_model_path):
            try:
                self.h5_model = load_model(self.h5_model_path, compile=False)
                success = self.find_h5_configuration()
                if not success:
                    success = self.use_preset_h5_configuration()
                if success:
                    h5_loaded = True
                    model_loaded = True
                    print("H5 model loaded successfully")
                else:
                    print("H5 model failed to configure properly, will rely on PT model only")
            except Exception as e:
                print(f"Error loading H5 model: {e}")
                print("Will continue with PT model only")
                
        if use_pt and YOLO_AVAILABLE and os.path.exists(self.pt_model_path):
            try:
                self.pt_model = YOLO(self.pt_model_path)
                pt_loaded = True
                model_loaded = True
                print("PT model loaded successfully")
            except Exception as e:
                print(f"Error loading PT model: {e}")
                
        return model_loaded
    
    def use_preset_h5_configuration(self):
        """Use a preset configuration for the H5 model when automatic detection fails"""
        try:
            self.h5_input_size = 180
            self.h5_input_type = "rgb"
            self.model_config = {
                "input_size": 180,
                "input_type": "rgb"
            }
            
            # Test it
            sample_image = np.zeros((180, 180, 3), dtype=np.uint8)
            processed = self.preprocess_image(sample_image, self.h5_input_size, self.h5_input_type)
            self.h5_model.predict(processed, verbose=0)
            
            print(f"Using preset configuration: size={self.h5_input_size}, type={self.h5_input_type}")
            return True
        except Exception as e:
            print(f"Preset configuration failed: {e}")
            return False
            
    def find_h5_configuration(self):
        if not TF_AVAILABLE or not self.h5_model:
            return False
            
        input_sizes = [180, 144, 128, 224, 96]
        input_types = ["rgb", "bgr", "gray", "gray_flat"]
        
        sample_image = np.zeros((180, 180, 3), dtype=np.uint8)
        
        try:
            processed = self.preprocess_image(sample_image, 180, "rgb")
            self.h5_model.predict(processed, verbose=0)
            self.h5_input_size = 180
            self.h5_input_type = "rgb"
            self.model_config = {
                "input_size": 180,
                "input_type": "rgb"
            }
            print(f"Found working configuration: size=180, type=rgb")
            return True
        except Exception as e:
            print(f"Failed likely configuration size=180, type=rgb: {str(e)[:100]}")
        
        for size in input_sizes:
            for input_type in input_types:
                if size == 180 and input_type == "rgb":
                    continue
                    
                try:
                    processed = self.preprocess_image(sample_image, size, input_type)
                    self.h5_model.predict(processed, verbose=0)
                    self.h5_input_size = size
                    self.h5_input_type = input_type
                    self.model_config = {
                        "input_size": size,
                        "input_type": input_type
                    }
                    print(f"Found working configuration: size={size}, type={input_type}")
                    return True
                except Exception as e:
                    print(f"Failed configuration size={size}, type={input_type}: {str(e)[:100]}")
                    pass
        
        return False
    
    def preprocess_image(self, image, size, input_type):
        img = cv2.resize(image, (size, size))
        
        if input_type == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return np.expand_dims(img, axis=0)
        elif input_type == "bgr":
            img = img.astype(np.float32) / 255.0
            return np.expand_dims(img, axis=0)
        elif input_type == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            return np.expand_dims(img, axis=0)
        elif input_type == "gray_flat":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = img.flatten()  # Flatten to 1D array
            return np.expand_dims(img, axis=0)
    
    def detect_food_h5(self, image):
        if not TF_AVAILABLE or self.h5_model is None:
            return []
        
        processed = self.preprocess_image(image, self.h5_input_size, self.h5_input_type)
        predictions = self.h5_model.predict(processed, verbose=0)
        
        result = []
        if len(predictions.shape) <= 2:
            preds = predictions[0]
            if len(self.classes) != len(preds):
                print(f"Warning: Number of classes ({len(self.classes)}) doesn't match prediction output ({len(preds)})")
            
            # Get top predictions
            for i, confidence in enumerate(preds):
                if i < len(self.classes) and confidence > 0.5:
                    class_name = self.classes[i]
                    price = self.prices.get(class_name, 0)
                    result.append({
                        "class": class_name,
                        "confidence": float(confidence),
                        "price": price
                    })
        
        return sorted(result, key=lambda x: x["confidence"], reverse=True)
    
    def detect_food_pt(self, image):
        if not YOLO_AVAILABLE or self.pt_model is None:
            return []
        
        results = self.pt_model.predict(image, conf=0.25)[0]
        result = []
        
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            class_id = int(class_id)
            if class_id < len(self.classes):
                class_name = self.classes[class_id]
                price = self.prices.get(class_name, 0)
                result.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "price": price,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return result
    
    def detect_food(self, image, use_h5=True, use_pt=True):
        results = {}
        
        if use_h5 and TF_AVAILABLE:
            results["h5"] = self.detect_food_h5(image)
        
        if use_pt and YOLO_AVAILABLE:
            results["pt"] = self.detect_food_pt(image)
            
        return results
    
    def draw_detection(self, image, detections):
        img_result = image.copy()
        
        if "pt" in detections:
            for item in detections["pt"]:
                if "bbox" in item:
                    x1, y1, x2, y2 = item["bbox"]
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{item['class']} ({item['confidence']:.2f}) - {item['price']:,}đ"
                    cv2.putText(img_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if "h5" in detections and detections["h5"]:
            top_detection = detections["h5"][0]
            h, w = img_result.shape[:2]
            label = f"{top_detection['class']} ({top_detection['confidence']:.2f}) - {top_detection['price']:,}đ"
            cv2.putText(img_result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        return img_result
        
    def calculate_total(self, detections):
        total = 0
        items = []
        
        if "pt" in detections:
            for item in detections["pt"]:
                total += item["price"]
                items.append({
                    "name": item["class"],
                    "price": item["price"],
                    "confidence": item["confidence"]
                })
        
        elif "h5" in detections and detections["h5"]:
            item = detections["h5"][0]
            total += item["price"]
            items.append({
                "name": item["class"],
                "price": item["price"],
                "confidence": item["confidence"]
            })
            
        return {
            "total": total,
            "items": items
        } 

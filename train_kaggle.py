#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import glob
from tqdm import tqdm
import gc  # Garbage collector
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')
except:
    print("Could not configure GPU - will use CPU")

OUTPUT_DIR = '/kaggle/working'
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        OUTPUT_DIR = './'

try:
    from google.colab import drive
    IN_COLAB = True
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted")
    print("Running in Colab...")
    
    if os.path.exists('/kaggle/input'):
        dataset_dirs = os.listdir('/kaggle/input')
        if dataset_dirs:
            DATASET_DIR = os.path.join('/kaggle/input', dataset_dirs[0])
            print(f"Found Kaggle dataset at: {DATASET_DIR}")
        else:
            DATASET_DIR = '/content/dataset'
            if not os.path.exists(DATASET_DIR):
                os.makedirs(DATASET_DIR)
                print("Please upload your dataset to:", DATASET_DIR)
            print("Using dataset directory in Colab:", DATASET_DIR)
    else:
        DATASET_DIR = '/content/dataset'
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            print("Please upload your dataset to:", DATASET_DIR)
        print("Using dataset directory in Colab:", DATASET_DIR)
except:
    IN_COLAB = False
    DATASET_DIR = 'Datasets'
    print(f"Using local dataset at: {DATASET_DIR}")

CLASS_MAPPING = {
    "0": "cahukho",
    "3": "canhcai",
    "4": "canhchua",
    "5": "canhchua",
    "7": "com",
    "9": "dauhusotca",
    "11": "gachien",
    "13": "raumuongxao",
    "16": "thitkho",
    "18": "thitkhotrung",
    "19": "thitkho",
    "20": "trungchien",
    "21": "gachien"
}

VIETNAMESE_FOODS = [
    "com",
    "canhchua",
    "gachien",
    "cahukho",
    "thitkho",
    "raumuongxao",
    "dauhusotca",
    "canhcai",
    "thitkhotrung",
    "trungchien"
]

CLASS_INDICES = {food: idx for idx, food in enumerate(VIETNAMESE_FOODS)}
REVERSE_CLASS_INDICES = {idx: food for food, idx in CLASS_INDICES.items()}

IMG_SIZE = 224  
BATCH_SIZE = 64 if IN_COLAB else 16  
EPOCHS = 30 
NUM_CLASSES = len(VIETNAMESE_FOODS)
LEARNING_RATE = 1e-4
WEIGHTS = 'imagenet'

print(f"Number of classes: {NUM_CLASSES}")
print(f"Class mapping: {CLASS_INDICES}")

def create_data_generators(train_x, train_y, val_x, val_y):
    """
    Create data generators with augmentation
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        val_x, val_y,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, val_generator

def find_dataset_directories():
    """
    Find the appropriate dataset directories
    """
    # Check standard paths first
    train_img_dir = os.path.join(DATASET_DIR, 'images/train')
    val_img_dir = os.path.join(DATASET_DIR, 'images/val')
    train_label_dir = os.path.join(DATASET_DIR, 'labels/train')
    val_label_dir = os.path.join(DATASET_DIR, 'labels/val')
    
    # Print directory structure for debugging
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Checking if train image directory exists: {os.path.exists(train_img_dir)}")
    print(f"Checking if val image directory exists: {os.path.exists(val_img_dir)}")
    
    if not os.path.exists(train_img_dir) or not os.path.exists(train_label_dir):
        print("Standard paths not found, searching for alternative paths...")
        
        image_dirs = []
        label_dirs = []
        
        for root, dirs, files in os.walk(DATASET_DIR):
            if files:
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    print(f"Found images in: {root}")
                    image_dirs.append(root)
                elif any(f.lower().endswith('.txt') for f in files):
                    print(f"Found labels in: {root}")
                    label_dirs.append(root)
        
        for img_dir in image_dirs:
            if 'train' in img_dir.lower():
                train_img_dir = img_dir
            elif 'val' in img_dir.lower() or 'test' in img_dir.lower():
                val_img_dir = img_dir
        
        for label_dir in label_dirs:
            if 'train' in label_dir.lower():
                train_label_dir = label_dir
            elif 'val' in label_dir.lower() or 'test' in label_dir.lower():
                val_label_dir = label_dir
        
        if train_img_dir == os.path.join(DATASET_DIR, 'images/train') and image_dirs:
            train_img_dir = image_dirs[0]
            if len(image_dirs) > 1:
                val_img_dir = image_dirs[1]
        
        if train_label_dir == os.path.join(DATASET_DIR, 'labels/train') and label_dirs:
            train_label_dir = label_dirs[0]
            if len(label_dirs) > 1:
                val_label_dir = label_dirs[1]
    
    print(f"Using train image directory: {train_img_dir}")
    print(f"Using validation image directory: {val_img_dir}")
    print(f"Using train label directory: {train_label_dir}")
    print(f"Using validation label directory: {val_label_dir}")
    
    return train_img_dir, val_img_dir, train_label_dir, val_label_dir

def create_dataset():
    """
    Create dataset from images and labels directories
    """
    train_img_dir, val_img_dir, train_label_dir, val_label_dir = find_dataset_directories()
    
    train_img_paths = glob.glob(os.path.join(train_img_dir, '*.jpg')) + \
                     glob.glob(os.path.join(train_img_dir, '*.jpeg')) + \
                     glob.glob(os.path.join(train_img_dir, '*.png'))
                     
    val_img_paths = glob.glob(os.path.join(val_img_dir, '*.jpg')) + \
                   glob.glob(os.path.join(val_img_dir, '*.jpeg')) + \
                   glob.glob(os.path.join(val_img_dir, '*.png'))
    
    print(f"Found {len(train_img_paths)} training images")
    print(f"Found {len(val_img_paths)} validation images")
    
    if len(val_img_paths) == 0 and len(train_img_paths) > 0:
        print("No validation images found. Splitting training images 80/20...")
        from sklearn.model_selection import train_test_split
        train_img_paths, val_img_paths = train_test_split(
            train_img_paths, test_size=0.2, random_state=42
        )
        print(f"Split into {len(train_img_paths)} training and {len(val_img_paths)} validation images")
    
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    
    # Function to process image and label
    def process_image_and_label(img_path, label_dir, is_training=True):
        base_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, base_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    parts = content.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        
                        if str(class_id) in CLASS_MAPPING:
                            food_name = CLASS_MAPPING[str(class_id)]
                            
                            if food_name not in CLASS_INDICES:
                                return None, None
                                
                            try:
                                img = cv2.imread(img_path)
                                if img is None:
                                    return None, None
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                                
                                return img, CLASS_INDICES[food_name]
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")
        return None, None
    
    print("Processing training images...")
    for img_path in tqdm(train_img_paths):
        img, class_idx = process_image_and_label(img_path, train_label_dir, is_training=True)
        if img is not None and class_idx is not None:
            train_x.append(img)
            train_y.append(class_idx)
    
    print("Processing validation images...")
    for img_path in tqdm(val_img_paths):
        img, class_idx = process_image_and_label(img_path, val_label_dir, is_training=False)
        if img is not None and class_idx is not None:
            val_x.append(img)
            val_y.append(class_idx)
    
    gc.collect()
    
    if len(train_x) > 0 and len(val_x) > 0:
        train_x = np.array(train_x, dtype=np.uint8)
        train_y = np.array(train_y)
        val_x = np.array(val_x, dtype=np.uint8)
        val_y = np.array(val_y)
        
        # One-hot encode labels
        train_y = tf.keras.utils.to_categorical(train_y, NUM_CLASSES)
        val_y = tf.keras.utils.to_categorical(val_y, NUM_CLASSES)
        
        print(f"Final dataset shapes:")
        print(f"Train X: {train_x.shape}, Train Y: {train_y.shape}")
        print(f"Val X: {val_x.shape}, Val Y: {val_y.shape}")
        
        return train_x, train_y, val_x, val_y
    else:
        print("Error: Not enough valid images found")
        
        print("Creating synthetic dataset for testing purposes...")
        train_x = np.random.randint(0, 255, size=(100, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        train_y = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=100), NUM_CLASSES)
        val_x = np.random.randint(0, 255, size=(20, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        val_y = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=20), NUM_CLASSES)
        
        print(f"Synthetic dataset shapes:")
        print(f"Train X: {train_x.shape}, Train Y: {train_y.shape}")
        print(f"Val X: {val_x.shape}, Val Y: {val_y.shape}")
        
        return train_x, train_y, val_x, val_y

def build_model():
    """
    Build EfficientNetB0 model with transfer learning, optimized for T4 GPU
    """
    base_model = EfficientNetB0(
        weights=WEIGHTS,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Only freeze the first 80 layers, make the rest trainable
    for layer in base_model.layers[:80]:
        layer.trainable = False
    for layer in base_model.layers[80:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    if hasattr(tf.keras.mixed_precision, 'LossScaleOptimizer'):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks():
    """
    Create training callbacks
    """
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # More patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    if IN_COLAB:
        try:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(OUTPUT_DIR, 'logs'),
                histogram_freq=1,
                profile_batch=0
            )
            callbacks.append(tensorboard_callback)
        except:
            print("TensorBoard callback not available")
    
    return callbacks

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    

    cm = confusion_matrix(y_true, y_pred)
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[REVERSE_CLASS_INDICES[i] for i in range(NUM_CLASSES)],
        yticklabels=[REVERSE_CLASS_INDICES[i] for i in range(NUM_CLASSES)]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(model, x_val, y_val, n_samples=5):
    """
    Plot sample predictions
    """
    if len(x_val) < n_samples:
        n_samples = len(x_val)
    
    indices = np.random.choice(len(x_val), n_samples, replace=False)
    
    samples = x_val[indices] / 255.0  # Normalize
    true_labels = np.argmax(y_val[indices], axis=1)
    predictions = model.predict(samples)
    pred_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(samples[i])
        
        true_class = REVERSE_CLASS_INDICES[true_labels[i]]
        pred_class = REVERSE_CLASS_INDICES[pred_labels[i]]
        confidence = np.max(predictions[i]) * 100
        
        title = f"True: {true_class}\nPred: {pred_class}\n({confidence:.1f}%)"
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'))
    plt.close()

def fine_tune_model(model, train_generator, val_data):
    """
    Fine-tune the model by unfreezing all layers
    """
    for layer in model.layers[0].layers:
        layer.trainable = True
    
    optimizer = Adam(learning_rate=LEARNING_RATE / 100)
    if hasattr(tf.keras.mixed_precision, 'LossScaleOptimizer'):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    

    callbacks = create_callbacks()
    
 
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_data[0]) // BATCH_SIZE + 1
    

    history = model.fit(
        train_generator,
        validation_data=val_data,
        epochs=15, 
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def evaluate_model(model, val_x, val_y):
    """
    Evaluate model on validation set
    """
    val_x_normalized = val_x.astype('float32') / 255.0
    
    loss, accuracy = model.evaluate(val_x_normalized, val_y, verbose=0)
    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation loss: {loss:.4f}")
    
    y_pred = model.predict(val_x_normalized)
    
    y_true_classes = np.argmax(val_y, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_names = [REVERSE_CLASS_INDICES[i] for i in range(NUM_CLASSES)]
    
    report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=class_names,
        digits=4
    )
    
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    plot_confusion_matrix(val_y, y_pred)
    
    plot_sample_predictions(model, val_x, val_y, n_samples=5)
    
    return accuracy, report

def save_model_and_metadata(model):
    """
    Save model and metadata
    """
    model_path = os.path.join(OUTPUT_DIR, 'vietnamese_food_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    class_mapping_path = os.path.join(OUTPUT_DIR, 'class_indices.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(REVERSE_CLASS_INDICES, f, indent=2)
    print(f"Class mapping saved to: {class_mapping_path}")
    
    with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(OUTPUT_DIR, 'vietnamese_food_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_path}")
    except Exception as e:
        print(f"Error saving TFLite model: {e}")

def main():
    """
    Main function
    """
    print("Starting Vietnamese Food Classification Training")
    print(f"TensorFlow version: {tf.__version__}")
    
    train_x, train_y, val_x, val_y = create_dataset()
    
    train_generator, val_generator = create_data_generators(train_x, train_y, val_x, val_y)
    
    gc.collect()
    
    model = build_model()
    print("Model created")
    model.summary()
    
    callbacks = create_callbacks()
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_x) // BATCH_SIZE + 1
    
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=(val_x.astype('float32') / 255.0, val_y),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history)
    
    gc.collect()
    
    print("\nFine-tuning model...")
    fine_tune_history, model = fine_tune_model(
        model, 
        train_generator, 
        (val_x.astype('float32') / 255.0, val_y)
    )
    
    plot_training_history(fine_tune_history)
    
    print("\nEvaluating model...")
    accuracy, report = evaluate_model(model, val_x, val_y)
    
    save_model_and_metadata(model)
    
    print("\nTraining completed successfully!")
    print(f"Model and results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 

"""
train_model.py — Standalone MobileNetV2 training script for ISL Recognition.

Reads images from the 'Indian/' directory and trains a model that is saved
as 'Mobilenetv2_ISL_model.keras', ready for use by script.py.

Compatible with TensorFlow >=2.16 / Keras 3.

Usage:
    pip install -r requirements.txt
    python train_model.py
"""

import os
# Suppress harmless TensorFlow and CUDA logs for Intel-only setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import tensorflow as tf
# Keras 3 imports (bundled with TF >=2.16)
from keras.utils import img_to_array
from keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                           BatchNormalization, RandomRotation,
                           RandomZoom, RandomTranslation, Rescaling)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DATA_PATH = "Indian"
MODEL_SAVE_PATH = "Mobilenetv2_ISL_model.keras"   # modern .keras format
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Early stopping will likely end training sooner
LEARNING_RATE = 0.0001

# ==================== Data Loading ====================
def load_images_and_labels(data_path, categories, img_size=IMG_SIZE, max_images=10):
    """Load and preprocess images, limited to a maximum per class."""
    images = []
    labels = []
    total = len(categories)
    for i, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            print(f"  ⚠️  Skipping missing category folder: {category}")
            continue
            
        files = os.listdir(category_path)
        
        # --- THE FIX: Limit the number of files to max_images ---
        files_to_load = files[:max_images] 
        
        print(f"  [{i+1}/{total}] Loading class '{category}': {len(files_to_load)} images (capped at {max_images})")
        for img_name in files_to_load:
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(category)
            
    return np.array(images), np.array(labels)


def main():
    print("=" * 60)
    print("  MobileNetV2 ISL Model Training")
    print("=" * 60)

    # --- Validate data directory ---
    if not os.path.isdir(DATA_PATH):
        print(f"\n❌ ERROR: Data directory '{DATA_PATH}' not found!")
        print("   Run 'python collect_imgs.py' first to collect training data.")
        sys.exit(1)

    categories = sorted(os.listdir(DATA_PATH))
    # Filter to only actual directories
    categories = [c for c in categories if os.path.isdir(os.path.join(DATA_PATH, c))]

    if len(categories) == 0:
        print(f"\n❌ ERROR: No class folders found in '{DATA_PATH}'!")
        print("   Run 'python collect_imgs.py' first to collect training data.")
        sys.exit(1)

    print(f"\nFound {len(categories)} classes: {categories}")

    # --- Load data ---
    print("\n📂 Loading images...")
    images, labels = load_images_and_labels(DATA_PATH, categories)
    print(f"\n✅ Loaded {len(images)} total images")
    print(f"   Image shape: {images[0].shape}")

    if len(images) < 100:
        print("\n⚠️  WARNING: Very few images. Model accuracy will be limited.")
        print("   Consider collecting more data with collect_imgs.py.\n")

    # --- Encode labels ---
    label_binarizer = LabelBinarizer()
    labels_encoded = label_binarizer.fit_transform(labels)
    num_classes = len(label_binarizer.classes_)
    print(f"   Number of classes: {num_classes}")

    # --- Split data: 64% train / 16% val / 20% test ---
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded.argmax(axis=1)
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train.argmax(axis=1)
    )
    print(f"   Training samples:   {len(X_train)} ({len(X_train)/len(images)*100:.0f}%)")
    print(f"   Validation samples: {len(X_val)} ({len(X_val)/len(images)*100:.0f}%)")
    print(f"   Test samples:       {len(X_test)} ({len(X_test)/len(images)*100:.0f}%)")

    # --- Normalize (will also be done inside model via Rescaling layer) ---
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # --- Build model with inline augmentation (Keras 3 way) ---
    print("\n🏗️  Building MobileNetV2 model...")

    # Data augmentation via Keras preprocessing layers
    # These layers are active only during training (no-op at inference)
    data_augmentation = Sequential([
        RandomRotation(factor=0.04),          # ±15° expressed as fraction of 2π
        RandomZoom(height_factor=0.1),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="data_augmentation")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Progressive fine-tuning: unfreeze top 30% of layers
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"   Total base layers: {len(base_model.layers)}")
    print(f"   Trainable layers:  {len(base_model.layers) - fine_tune_at}")

    model = Sequential([
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   Total parameters:     {model.count_params():,}")
    print(f"   Est. model size:      {model.count_params()*4/1024/1024:.1f} MB")

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --- Train (directly with numpy arrays — no deprecated generator) ---
    print(f"\n🚀 Starting training (max {EPOCHS} epochs, early stopping enabled)...")
    print("-" * 60)
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test Loss:     {test_loss:.4f}")

    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])
    print(f"   Best Train Acc: {best_train_acc:.4f}")
    print(f"   Best Val Acc:   {best_val_acc:.4f}")

    if test_accuracy >= 0.95:
        print("   🎯 EXCELLENT — 95%+ accuracy!")
    elif test_accuracy >= 0.85:
        print("   ✅ GOOD — consider collecting more data for better results")
    elif test_accuracy >= 0.70:
        print("   👍 DECENT — more data and epochs would help significantly")
    else:
        print("   ⚠️  LOW — collect significantly more data (500+ per class)")

    # --- Classification report ---
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print(f"\n{'=' * 60}")
    print("  Per-class Classification Report")
    print("=" * 60)
    print(classification_report(
        y_true_classes, y_pred_classes,
        target_names=categories,
        digits=4,
        zero_division=0
    ))

    # --- Save model in modern .keras format ---
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved as '{MODEL_SAVE_PATH}'")
    print(f"   File size: {os.path.getsize(MODEL_SAVE_PATH) / (1024*1024):.1f} MB")
    print(f"\n🎉 Training complete! You can now run: python script.py")


if __name__ == "__main__":
    main()

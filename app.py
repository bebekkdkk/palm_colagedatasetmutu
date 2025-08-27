import os
import io
import base64
import random
from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from collections import Counter
import shutil
from pathlib import Path

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
TRAIN_FOLDER = os.path.join('static', 'train_dataset')

# Buat folder yang diperlukan
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, TRAIN_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    
# Buat folder kelas untuk training
CLASS_LABELS = ["Abnormal", "Empty", "Overripe", "Ripe", "Rotten", "Unripe"]
for label in CLASS_LABELS:
    os.makedirs(os.path.join(TRAIN_FOLDER, label), exist_ok=True)

# Konfigurasi model paths
DET_MODEL_PATH = os.path.join('models', 'model_automl.tflite')
CLS_MODEL_PATH = os.path.join('models', 'finalmutu.tflite')


# Load models
det_interpreter = tf.lite.Interpreter(model_path=DET_MODEL_PATH)
det_interpreter.allocate_tensors()
det_input_details = det_interpreter.get_input_details()
det_output_details = det_interpreter.get_output_details()

cls_interpreter = tf.lite.Interpreter(model_path=CLS_MODEL_PATH)
cls_interpreter.allocate_tensors()
cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()
def save_crop(image, filename, label, output_dir=TRAIN_FOLDER):
    """Simpan gambar crop ke folder dataset training"""
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    save_path = os.path.join(output_dir, label, filename)
    image.save(save_path)
    return save_path

# Routes untuk endpoints web
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)


det_class_names = ['buah', 'batang']
cls_labels = ["Abnormal", "Empty", "Overripe", "Ripe", "Rotten", "Unripe"]

def preprocess_image_det(img_path, input_size):
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(input_size)
    img_array = np.array(img_resized)

    input_dtype = det_input_details[0]['dtype']
    if input_dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_with_tflite_det_stream(image, score_threshold=0.3):
    """Deteksi objek dari gambar PIL Image"""
    input_shape = det_input_details[0]['shape']
    img_resized = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img_resized)
    
    # Normalize input
    input_dtype = det_input_details[0]['dtype']
    if input_dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    det_interpreter.set_tensor(det_input_details[0]['index'], img_array)
    det_interpreter.invoke()

    output_boxes = det_interpreter.get_tensor(det_output_details[0]['index'])[0]
    output_classes = det_interpreter.get_tensor(det_output_details[1]['index'])[0]
    output_scores = det_interpreter.get_tensor(det_output_details[2]['index'])[0]
    num_detections = int(det_interpreter.get_tensor(det_output_details[3]['index'])[0])

    detections = []
    for i in range(num_detections):
        if output_scores[i] >= score_threshold:
            class_id = int(output_classes[i])
            detections.append({
                'box': output_boxes[i],
                'class_id': class_id,
                'class_name': det_class_names[class_id],
                'score': output_scores[i]
            })
    return detections, image

def crop_detections(image, detections, target_class='buah', target_size=(896, 896)):
    width, height = image.size
    cropped_images = []
    for det in detections:
        if det['class_name'] != target_class:
            continue
        ymin, xmin, ymax, xmax = det['box']
        left, top, right, bottom = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

        padding = 2
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)

        cropped_img = image.crop((left, top, right, bottom))
        cropped_img = cropped_img.resize(target_size, Image.LANCZOS)

        enhancer = ImageEnhance.Sharpness(cropped_img)
        cropped_img = enhancer.enhance(1.5)

        cropped_images.append((cropped_img, (left, top, right, bottom)))
    return cropped_images

def preprocess_image_cls(pil_img, input_size):
    img_resized = pil_img.resize(input_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_crops(cropped_images):
    results = []
    input_size = (cls_input_details[0]['shape'][1], cls_input_details[0]['shape'][2])

    for crop, bbox in cropped_images:
        input_data = preprocess_image_cls(crop, input_size)
        cls_interpreter.set_tensor(cls_input_details[0]['index'], input_data)
        cls_interpreter.invoke()

        output_data = cls_interpreter.get_tensor(cls_output_details[0]['index'])[0]
        pred_class = np.argmax(output_data)
        pred_conf = output_data[pred_class]

        results.append({
            'crop': crop,
            'bbox': bbox,
            'label': cls_labels[pred_class],
            'confidence': float(pred_conf)
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    # Count images in each class folder
    class_counts = {}
    total_images = 0
    for class_name in CLASS_LABELS:
        class_path = os.path.join(TRAIN_FOLDER, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
            total_images += count
    
    return render_template('train.html', class_counts=class_counts, total_images=total_images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read uploaded image
        image_stream = file.read()
        image = Image.open(io.BytesIO(image_stream)).convert('RGB')
        
        # Process the image
        detections, _ = predict_with_tflite_det_stream(image)
        
        # Create a copy for drawing bounding boxes
        detection_image = image.copy()
        draw = ImageDraw.Draw(detection_image)
        width, height = image.size
        
        # Draw bounding boxes
        for det in detections:
            if det['class_name'] == 'buah':
                ymin, xmin, ymax, xmax = det['box']
                left, top, right, bottom = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
                draw.rectangle([left, top, right, bottom], outline='red', width=3)
                score_text = f"{det['score']:.2f}"
                draw.text((left, top-15), score_text, fill='red')
        
        # Use original clean image for cropping and classification
        cropped_images = crop_detections(image, detections)
        results = classify_crops(cropped_images)
        
        # Save only the detection image
        detection_path = os.path.join(RESULTS_FOLDER, f"detection_{file.filename}")
        detection_image.save(detection_path)
        
        # Count detections and create summary
        detection_count = len([d for d in detections if d['class_name'] == 'buah'])
        classification_summary = Counter()
        
        # Process results without saving crops
        crops_info = []
        for result in results:
            classification_summary[result['label']] += 1
            # Convert crop to base64 for direct display
            buffer = io.BytesIO()
            result['crop'].save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            crops_info.append({
                'image_data': f'data:image/jpeg;base64,{img_str}',
                'label': result['label'],
                'confidence': result['confidence']
            })

        return jsonify({
            'detection_path': f"results/detection_{file.filename}",
            'detection_count': detection_count,
            'classification_summary': {
                label: classification_summary[label] for label in cls_labels
            },
            'crops': crops_info
        })

@app.route('/sort_train_image', methods=['POST'])
def sort_train_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    label = request.form['label']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if label not in CLASS_LABELS:
        return jsonify({'error': 'Invalid label'})
    
    # Save file to appropriate training folder
    filename = file.filename
    save_path = os.path.join(TRAIN_FOLDER, label, filename)
    file.save(save_path)
    
    return jsonify({
        'success': True,
        'message': f'File saved to {label} folder'
    })


# Global variable to store training progress
training_progress = {
    'phase': 'Not Started',
    'progress': 0,
    'message': '',
    'type': 'info',
    'complete': False,
    'metrics': None,
    'confusion_matrix': None,
    'model_path': None,
    'error': None
}

def update_progress(phase, progress, message, type='info'):
    global training_progress
    training_progress.update({
        'phase': phase,
        'progress': progress,
        'message': message,
        'type': type
    })

@app.route('/training_progress')
def get_training_progress():
    return jsonify(training_progress)

@app.route('/train', methods=['POST'])
def handle_train():
    global training_progress
    training_progress = {
        'phase': 'Starting',
        'progress': 0,
        'message': 'Initializing training process...',
        'type': 'info',
        'complete': False,
        'metrics': None,
        'confusion_matrix': None,
        'model_path': None,
        'error': None
    }

    try:
        # ===============================
        # 1. PATH DATASET & AUGMENTATION
        # ===============================
        update_progress('Augmentation', 10, 'Starting data augmentation...')
        augmented_path = Path(os.path.join(RESULTS_FOLDER, 'augmented'))
        if augmented_path.exists():
            shutil.rmtree(augmented_path)
        os.makedirs(augmented_path, exist_ok=True)

        # Process each class
        total_classes = len(CLASS_LABELS)
        for idx, cls in enumerate(CLASS_LABELS):
            class_dir = Path(os.path.join(TRAIN_FOLDER, cls))
            if not class_dir.is_dir():
                continue

            os.makedirs(augmented_path / cls, exist_ok=True)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            update_progress('Augmentation', 
                          10 + (idx * 30 // total_classes), 
                          f'Augmenting {cls} class: {len(images)} images')

            for img_name in images:
                img_path = class_dir / img_name
                img = Image.open(img_path).convert("RGB")

                # Save original
                img.save(augmented_path / cls / img_name)

                # Augmentation 1: Rotation 15 degrees
                rotated_15 = img.rotate(15, expand=True)
                rotated_15.save(augmented_path / cls / f"{Path(img_name).stem}_rot15{Path(img_name).suffix}")

                # Augmentation 2: Rotation 90 degrees
                rotated_90 = img.rotate(90, expand=True)
                rotated_90.save(augmented_path / cls / f"{Path(img_name).stem}_rot90{Path(img_name).suffix}")

                # Augmentation 3: Gaussian Blur
                blur_radius = random.uniform(0.5, 2.0)
                blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
                blurred.save(augmented_path / cls / f"{Path(img_name).stem}_blur{Path(img_name).suffix}")

        update_progress('Dataset Split', 40, 'Starting train/validation split...')

        # ====================================
        # 2. SPLIT DATASET TRAIN & VALIDATION
        # ====================================
        output_path = Path(os.path.join(RESULTS_FOLDER, 'dataset_split'))
        if output_path.exists():
            shutil.rmtree(output_path)

        train_ratio = 0.8
        random.seed(42)

        for split in ["train", "val"]:
            for cls in CLASS_LABELS:
                os.makedirs(output_path / split / cls, exist_ok=True)

        for idx, cls in enumerate(CLASS_LABELS):
            update_progress('Dataset Split', 
                          40 + (idx * 20 // total_classes), 
                          f'Splitting {cls} class...')
            
            class_dir = augmented_path / cls
            if not class_dir.is_dir():
                continue

            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(images)

            split_idx = int(len(images) * train_ratio)
            train_files = images[:split_idx]
            val_files = images[split_idx:]

            for img in train_files:
                shutil.copy(class_dir / img, output_path / "train" / cls / img)
            for img in val_files:
                shutil.copy(class_dir / img, output_path / "val" / cls / img)

        update_progress('Training', 60, 'Starting model training...')

        # ====================================
        # 3. TRAIN MODEL
        # ====================================
        import torch
        from ultralytics import YOLO

        # Check GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            update_progress('Training', 60, f'Using GPU: {gpu_name}')
        else:
            update_progress('Training', 60, 'GPU not available, using CPU')

        def on_train_epoch_end(trainer):
            epoch = trainer.epoch
            update_progress('Training', 
                          60 + (epoch * 35 // 25), 
                          f'Training epoch {epoch}/25 completed')

        model = YOLO('yolov8n-cls.pt')
        results = model.train(
            data=str(output_path),
            epochs=25,
            imgsz=896,
            batch=16,  # Increased batch size for GPU
            workers=4,  # Increased number of workers
            device=device,  # Specify device (GPU/CPU)
            callbacks={'on_train_epoch_end': on_train_epoch_end}
        )

        update_progress('Exporting', 95, 'Exporting model to TFLite format...')

        # Export to TFLite
        export_path = model.export(format='tflite', imgsz=896)

        # Update final results
        metrics = results.results_dict
        training_progress.update({
            'phase': 'Complete',
            'progress': 100,
            'message': 'Training completed successfully!',
            'type': 'success',
            'complete': True,
            'metrics': {
                'accuracy': metrics.get('metrics/accuracy', 0),
                'precision': metrics.get('metrics/precision', 0),
                'recall': metrics.get('metrics/recall', 0),
                'f1': metrics.get('metrics/f1', 0)
            },
            'confusion_matrix': metrics.get('confusion_matrix', [[0]*6]*6),
            'model_path': str(export_path)
        })

        return jsonify({'success': True})

    except Exception as e:
        training_progress.update({
            'error': str(e),
            'type': 'error',
            'complete': True
        })
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_model')
def download_model():
    model_path = request.args.get('path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    return send_from_directory(os.path.dirname(model_path), 
                             os.path.basename(model_path), 
                             as_attachment=True)


# Route for saving individual crops
@app.route('/save_crop', methods=['POST'])
def save_crop_endpoint():
    try:
        data = request.get_json()
        crop_path = data.get('crop_path')
        class_name = data.get('class_name')

        # Validate class name
        if class_name not in CLASS_LABELS:
            return jsonify({'success': False, 'error': f'Invalid class name: {class_name}'})

        # Extract the base64 image data
        if crop_path.startswith('data:image'):
            # Split the base64 string at ',' to get the actual data
            base64_data = crop_path.split(',')[1]
            # Convert base64 to image
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
        else:
            # If it's a file path, load the image directly
            image = Image.open(os.path.join('static', crop_path))

        # Generate a unique filename
        filename = f'crop_{len(os.listdir(os.path.join(TRAIN_FOLDER, class_name)))}.png'
        
        # Save the image to the appropriate class folder
        save_path = save_crop(image, filename, class_name)

        return jsonify({
            'success': True,
            'saved_path': save_path
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Route for separate training page
@app.route('/train_page')
def train_page():
    # Count images in each class folder
    class_counts = {}
    total_images = 0
    for class_name in CLASS_LABELS:
        class_path = os.path.join(TRAIN_FOLDER, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
            total_images += count
    
    return render_template('train.html', class_counts=class_counts, total_images=total_images)

if __name__ == '__main__':
    app.run(debug=True)

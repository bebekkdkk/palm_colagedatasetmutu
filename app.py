from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter
from collections import Counter
from pathlib import Path
import os
import json
import datetime
import random
import tensorflow as tf
from validation_handler import ValidationHandler

# =============================
# Inisialisasi Flask & Folder
# =============================
app = Flask(__name__)

# Folder konfigurasi
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATABASE_FILE = os.path.join(UPLOAD_FOLDER, 'database.txt')

# Buat folder utama
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi validation handler
validation_handler = ValidationHandler()

# =============================
# Routes
# =============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_valid_status', methods=['POST'])
def update_valid_status():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        image_id = data.get('image_id')
        place = data.get('place')
        
        if not image_id or place is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Convert place to integer if it's not already
        try:
            place = int(place)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid place value'}), 400

        # Use validation handler to update status
        success, message = validation_handler.update_valid_status(image_id, place)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def draw_bounding_boxes_with_numbers(image, detections):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    all_boxes = []
    for detection in detections:
        box = detection['box']
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
        all_boxes.append((left, right, top, bottom))
    for idx, detection in enumerate(detections, 1):
        box = detection['box']
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
        current_box = (left, right, top, bottom)
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        half_width = (right - left) / 2.5
        half_height = (bottom - top) / 2.5
        offset_x = half_width / 3
        offset_y = half_height / 3
        circle_radius = 3
        points = [
            (center_x, center_y),
            (center_x - offset_x, center_y),
            (center_x + offset_x, center_y),
            (center_x, center_y - offset_y),
            (center_x, center_y + offset_y)
        ]
        for point_x, point_y in points:
            point_color = "lime"
            outline_color = "darkgreen"
            for other_idx, other_box in enumerate(all_boxes):
                if other_idx != idx - 1:
                    if point_in_box(point_x, point_y, other_box):
                        point_color = "red"
                        outline_color = "darkred"
                        break
            draw.ellipse([
                (point_x - circle_radius, point_y - circle_radius),
                (point_x + circle_radius, point_y + circle_radius)
            ], fill=point_color, outline=outline_color, width=2)
        label = f"#{idx} {detection['class_name']}: {detection['score']:.2f}"
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(label) * 6, 11
        text_x = left
        text_y = top - text_height - 5
        if text_y < 0:
            text_y = bottom + 5
        draw.rectangle([
            (text_x - 2, text_y - 2),
            (text_x + text_width + 2, text_y + text_height + 2)
        ], fill="red")
        if font:
            draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            draw.text((text_x, text_y), label, fill="white")
        number_label = str(idx)
        if font:
            num_bbox = draw.textbbox((0, 0), number_label, font=font)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
        else:
            num_width, num_height = len(number_label) * 6, 11
        draw.rectangle([
            (center_x - num_width/2 - 3, center_y - num_height/2 - 2),
            (center_x + num_width/2 + 3, center_y + num_height/2 + 2)
        ], fill="blue")
        if font:
            draw.text((center_x - num_width/2, center_y - num_height/2),
                     number_label, fill="white", font=font)
        else:
            draw.text((center_x - num_width/2, center_y - num_height/2),
                     number_label, fill="white")
    return image
import os
import io
import json
import base64
import random
import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from collections import Counter
import shutil
from pathlib import Path


# =============================
# Inisialisasi Flask & Folder
# =============================
app = Flask(__name__)

# Folder konfigurasi
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATABASE_FILE = os.path.join(UPLOAD_FOLDER, 'database.txt')

# Buat folder utama
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Label kelas untuk training
CLASS_LABELS = ["Abnormal", "Empty", "Overripe", "Ripe", "Rotten", "Unripe"]

def load_database():
    """Load the database from file"""
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_database(data):
    """Save the database to file"""
    with open(DATABASE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def add_to_database(file_name, class_result, place, valid_status=False):
    """Add a new entry to the database"""
    database = load_database()
    # Ensure place is stored as string if it's "-", otherwise keep as is
    stored_place = place if place == "-" else place
    entry = {
        "file_name": file_name,
        "uid": "default_user",  # You can implement user system later
        "class_result": class_result,
        "valid_status": valid_status,
        "place": stored_place
    }
    database.append(entry)
    save_database(database)

def generate_base_filename():
    """Generate a base filename without extension and place number
    Format: YYYYMMDDHHMMSSxxxxxx (xxxxxx is random hex)"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices('0123456789abcdef', k=6))
    return f"{timestamp}{random_str}"

# Path model
DET_MODEL_PATH = os.path.join('models', 'model_automl.tflite')
CLS_MODEL_PATH = os.path.join('models', 'finalmutu.tflite')

# =============================
# Load Model Deteksi & Klasifikasi
# =============================
det_interpreter = tf.lite.Interpreter(model_path=DET_MODEL_PATH)
det_interpreter.allocate_tensors()
det_input_details = det_interpreter.get_input_details()
det_output_details = det_interpreter.get_output_details()

cls_interpreter = tf.lite.Interpreter(model_path=CLS_MODEL_PATH)
cls_interpreter.allocate_tensors()
cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()
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

def compute_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def non_max_suppression(detections, iou_threshold=0.6):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    selected = []
    while detections:
        current = detections.pop(0)
        selected.append(current)
        detections = [
            d for d in detections
            if compute_iou(current['box'], d['box']) <= iou_threshold
        ]
    return selected

def point_in_box(point_x, point_y, box_coords):
    left, right, top, bottom = box_coords
    return left <= point_x <= right and top <= point_y <= bottom

def boxes_overlap(box1, box2, width, height):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    left1, right1, top1, bottom1 = xmin1 * width, xmax1 * width, ymin1 * height, ymax1 * height
    left2, right2, top2, bottom2 = xmin2 * width, xmax2 * width, ymin2 * height, ymax2 * height
    return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)

def count_overlapping_points(detection, all_detections, width, height, current_idx):
    box = detection['box']
    ymin, xmin, ymax, xmax = box
    left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    half_width = (right - left) / 2
    half_height = (bottom - top) / 2
    offset_x = half_width / 3
    offset_y = half_height / 3
    points = [
        (center_x, center_y),
        (center_x - offset_x, center_y),
        (center_x + offset_x, center_y),
        (center_x, center_y - offset_y),
        (center_x, center_y + offset_y)
    ]
    overlap_count = 0
    for point_x, point_y in points:
        for other_idx, other_detection in enumerate(all_detections):
            if other_idx != current_idx:
                other_box = other_detection['box']
                other_ymin, other_xmin, other_ymax, other_xmax = other_box
                other_left = other_xmin * width
                other_right = other_xmax * width
                other_top = other_ymin * height
                other_bottom = other_ymax * height
                other_coords = (other_left, other_right, other_top, other_bottom)
                if point_in_box(point_x, point_y, other_coords):
                    overlap_count += 1
                    break
    return overlap_count

def get_overlapping_trunk(fruit_detection, trunk_detections, width, height):
    fruit_box = fruit_detection['box']
    for trunk in trunk_detections:
        trunk_box = trunk['box']
        if boxes_overlap(fruit_box, trunk_box, width, height):
            return trunk
    return None

def filter_overlapping_detections_with_trunk_validation(detections, width, height):
    """
    Proses filtering dengan alur:
    1. Deteksi overlapping (>4 titik overlap)
    2. Validasi trunk HANYA untuk buah yang overlapping
    3. Dari semua overlapping yang punya trunk, pilih score tertinggi secara global
    4. Buang overlapping yang tidak ada trunk
    5. Pertahankan semua buah non-overlapping (dengan/tanpa trunk)
    """
    fruit_detections = [d for d in detections if d['class_name'] == 'buah']
    trunk_detections = [d for d in detections if d['class_name'] == 'batang']

    # Step 1: Identifikasi buah yang overlapping (>4 titik)
    overlapping_fruits = []
    non_overlapping_fruits = []
    for idx, fruit in enumerate(fruit_detections):
        overlap_count = count_overlapping_points(fruit, fruit_detections, width, height, idx)
        if overlap_count > 4:
            overlapping_fruits.append({
                'index': idx,
                'fruit': fruit,
                'overlap_count': overlap_count
            })
        else:
            non_overlapping_fruits.append(fruit)

    validated_fruits = []
    overlapping_with_trunk = []
    overlapping_no_trunk = []
    for item in overlapping_fruits:
        fruit = item['fruit']
        trunk = get_overlapping_trunk(fruit, trunk_detections, width, height)
        if trunk:
            item['trunk'] = trunk
            overlapping_with_trunk.append(item)
        else:
            overlapping_no_trunk.append(item)

    # Step 3: Dari semua overlapping yang punya trunk, pilih satu dengan score tertinggi
    if overlapping_with_trunk:
        overlapping_with_trunk.sort(key=lambda x: x['fruit']['score'], reverse=True)
        selected_overlapping = overlapping_with_trunk[0]
        validated_fruits.append(selected_overlapping['fruit'])

    # Step 4: Buang semua overlapping fruits yang tidak ada trunk
    # (sudah otomatis tidak dimasukkan ke validated_fruits)

    # Step 5: Pertahankan SEMUA buah non-overlapping (tidak peduli ada trunk atau tidak)
    validated_fruits.extend(non_overlapping_fruits)

    # Hanya return buah saja, batang tidak diikutkan ke output akhir
    return validated_fruits

def predict_with_tflite_buah_only(image, score_threshold=0.35):
    input_shape = det_input_details[0]['shape']
    img_resized = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img_resized)
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
    detected_objects = []
    for i in range(num_detections):
        score = output_scores[i]
        if score > score_threshold:
            box = output_boxes[i]
            class_id = int(output_classes[i])
            class_name = det_class_names[class_id] if class_id < len(det_class_names) else f"Unknown_{class_id}"
            if class_name == 'buah':
                detected_objects.append({
                    'box': box,
                    'class_id': class_id,
                    'score': score,
                    'class_name': class_name
                })
    return detected_objects, image
def filter_overlapping_detections(detections, width, height):
    filtered_detections = detections.copy()
    removed_indices = set()
    while True:
        detection_to_remove = None
        max_overlap = 0
        for idx, detection in enumerate(filtered_detections):
            if idx in removed_indices:
                continue
            overlap_count = count_overlapping_points_excluding_removed(
                detection, filtered_detections, width, height, idx, removed_indices
            )
            if overlap_count > 4 and overlap_count > max_overlap:
                max_overlap = overlap_count
                detection_to_remove = idx
        if detection_to_remove is None:
            break
        removed_indices.add(detection_to_remove)
    for idx in sorted(removed_indices, reverse=True):
        filtered_detections.pop(idx)
    return filtered_detections

def count_overlapping_points_excluding_removed(detection, all_detections, width, height, current_idx, removed_indices):
    box = detection['box']
    ymin, xmin, ymax, xmax = box
    left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    half_width = (right - left) / 2
    half_height = (bottom - top) / 2
    offset_x = half_width / 3
    offset_y = half_height / 3
    points = [
        (center_x, center_y),
        (center_x - offset_x, center_y),
        (center_x + offset_x, center_y),
        (center_x, center_y - offset_y),
        (center_x, center_y + offset_y)
    ]
    overlap_count = 0
    for point_x, point_y in points:
        for other_idx, other_detection in enumerate(all_detections):
            if other_idx != current_idx and other_idx not in removed_indices:
                other_box = other_detection['box']
                other_ymin, other_xmin, other_ymax, other_xmax = other_box
                other_left = other_xmin * width
                other_right = other_xmax * width
                other_top = other_ymin * height
                other_bottom = other_ymax * height
                other_coords = (other_left, other_right, other_top, other_bottom)
                if point_in_box(point_x, point_y, other_coords):
                    overlap_count += 1
                    break
    return overlap_count

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
    """Redirect to Colab for training"""
    return render_template('train.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image_stream = file.read()
        image = Image.open(io.BytesIO(image_stream)).convert('RGB')
        width, height = image.size
        # 1. Deteksi buah saja
        detected_objects, _ = predict_with_tflite_buah_only(image)
        # 2. NMS
        detected_objects = non_max_suppression(detected_objects, iou_threshold=0.6)
        # 3. Filtering overlap titik (hapus jika >4 titik overlap, proses berulang)
        detected_objects = filter_overlapping_detections(detected_objects, width, height)
        # 4. Counting hasil akhir
        detection_count = len(detected_objects)
        # 5. Visualisasi bounding box dan nomor
        detection_image = image.copy()
        detection_image = draw_bounding_boxes_with_numbers(detection_image, detected_objects)
        # 6. Cropping dan klasifikasi (jika ingin, tetap gunakan crop_detections & classify_crops)
        cropped_images = crop_detections(image, detected_objects)
        results = classify_crops(cropped_images)
        # 7. Prepare results without saving
        classification_summary = Counter()
        crops_info = []
        
        # Store original image in memory
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='JPEG')
        original_img_str = base64.b64encode(original_buffer.getvalue()).decode()
        
        # Store detection image in memory
        detection_buffer = io.BytesIO()
        detection_image.save(detection_buffer, format='JPEG')
        detection_img_str = base64.b64encode(detection_buffer.getvalue()).decode()
        
        # Process all crops
        for i, result in enumerate(results, 1):
            # Update summary
            classification_summary[result['label']] += 1
            
            # Save crop to buffer for display
            buffer = io.BytesIO()
            result['crop'].save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            crops_info.append({
                'image_data': f'data:image/jpeg;base64,{img_str}',
                'label': result['label'],
                'confidence': result['confidence'],
                'bbox': result['bbox']
            })
        return jsonify({
            'original_image': f"data:image/jpeg;base64,{original_img_str}",
            'detection_image': f"data:image/jpeg;base64,{detection_img_str}",
            'detection_count': detection_count,
            'classification_summary': {
                label: classification_summary[label] for label in cls_labels
            },
            'crops': crops_info,
            'original_filename': file.filename
        })




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

        # Generate new base filename
        base_name = generate_base_filename()
        
        # Get next place number from database
        database = load_database()
        place = max([entry['place'] for entry in database], default=0) + 1
        
        # Create filename with place number
        filename = f"{base_name}_{place}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the image
        image.save(save_path)
        
        # Add to database
        add_to_database(filename, class_name, place)

        return jsonify({
            'success': True,
            'saved_path': os.path.join('uploads', filename)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Route for saving all classifications
@app.route('/save_all_classifications', methods=['POST'])
def save_all_classifications():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Generate base filename using the existing function - SATU base_name untuk semua file
        base_name = generate_base_filename()
        extension = '.jpg'
        print(f"Generated base_name: {base_name}")  # Debug log
        
        saved_files = []  # Track semua file yang disimpan
        
        # 1. Save original image with (-) place number
        original_image_data = data.get('original_image')
        if original_image_data:
            if ',' in original_image_data:
                image_data = original_image_data.split(',')[1]
            else:
                image_data = original_image_data
            image_bytes = base64.b64decode(image_data)
            
            # Save original image with (-) place number
            original_filename = f"{base_name}(-){extension}"
            original_path = os.path.join(UPLOAD_FOLDER, original_filename)
            
            with open(original_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Saved original: {original_filename}")  # Debug log
            saved_files.append(original_filename)
            add_to_database(original_filename, "original", "-")
        
        # 2. Save detection image with (0)
        detection_image_data = data.get('detection_image')
        if detection_image_data:
            if ',' in detection_image_data:
                image_data = detection_image_data.split(',')[1]
            else:
                image_data = detection_image_data
            image_bytes = base64.b64decode(image_data)
            
            # Save detection image with (0)
            detection_filename = f"{base_name}(0){extension}"
            detection_path = os.path.join(UPLOAD_FOLDER, detection_filename)
            
            with open(detection_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Saved detection: {detection_filename}")  # Debug log
            saved_files.append(detection_filename)
            add_to_database(detection_filename, "detection", 0)

        # 3. Save each crop with sequential numbers in parentheses
        crops_data = data.get('crops', [])
        print(f"Processing {len(crops_data)} crops")
        
        for i, crop in enumerate(crops_data, 1):
            if not crop.get('image_data'):
                print(f"Skipping crop {i}: No image data")
                continue
                
            # Extract base64 data - remove header if present
            image_data = crop['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                print(f"Error decoding crop {i}: {str(e)}")
                continue
            
            # Save crop with sequential number in parentheses - USING SAME base_name
            crop_filename = f"{base_name}({i}){extension}"
            crop_path = os.path.join(UPLOAD_FOLDER, crop_filename)
            
            try:
                with open(crop_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Add to database with the classification label and place number
                classification_label = crop.get('label', 'Unknown')
                add_to_database(crop_filename, classification_label, i)
                
                print(f"Saved crop {i}: {crop_filename} with label {classification_label}")
                saved_files.append(crop_filename)
            except Exception as e:
                print(f"Error saving crop {i}: {str(e)}")
                continue

        # Return success only if we saved at least original, detection, and one crop
        if len(saved_files) >= 3:
            print(f"Successfully saved {len(saved_files)} files with base_name: {base_name}")
            return jsonify({
                'success': True,
                'message': f'Saved {len(saved_files)} files successfully',
                'saved_files': saved_files,
                'base_filename': base_name
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Only saved {len(saved_files)} files, expected at least 3 (original, detection, and crops)',
                'saved_files': saved_files
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route for checking database status
@app.route('/database_status')
def database_status():
    database = load_database()
    class_counts = Counter(entry['class_result'] for entry in database if entry['place'] > 0)
    total_images = sum(class_counts.values())
    
    return jsonify({
        'class_counts': dict(class_counts),
        'total_images': total_images
    })

# History Routes - Separate from main functionality
@app.route('/history')
def history():
    """Route for history page"""
    return render_template('history.html')

@app.route('/get_history_by_date', methods=['POST'])
def get_history_by_date():
    """Get historical detections for a specific date"""
    try:
        data = request.get_json()
        selected_date = data.get('date')
        if not selected_date:
            return jsonify({'error': 'No date provided'}), 400

        database = load_database()
        original_images = []

        # Filter only original images (place="-") for the selected date
        for entry in database:
            if entry['place'] == '-':
                filename = entry['file_name']
                # Extract date from filename (assuming format: YYYYMMDDHHMMSS...)
                file_date = filename[:8]  # Get YYYYMMDD part
                formatted_date = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:]}"
                
                if formatted_date == selected_date:
                    # Get timestamp for display
                    timestamp = filename[8:14]  # Get HHMMSS part
                    formatted_time = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}"
                    
                    # Get base name without extension for later use
                    base_name = filename.split('(')[0]
                    
                    # Count total detections for this image
                    detection_count = sum(1 for d in database if d['file_name'].startswith(base_name) and d['place'] != '-' and d['place'] != 0)
                    
                    # Get classification summary
                    classifications = {}
                    for d in database:
                        if d['file_name'].startswith(base_name) and d['place'] not in ['-', 0]:
                            class_name = d['class_result']
                            if class_name not in classifications:
                                classifications[class_name] = 0
                            classifications[class_name] += 1

                    original_images.append({
                        'id': base_name,
                        'filename': filename,
                        'image_path': f"/static/uploads/{filename}",
                        'timestamp': formatted_time,
                        'detection_count': detection_count,
                        'classifications': classifications
                    })
        
        # Sort by timestamp in descending order (newest first)
        original_images.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'items': original_images})

        # Group entries by base filename
        for entry in database:
            filename = entry['file_name']
            base_name = filename.split('(')[0] if '(' in filename else filename.split('.')[0]
            
            # Extract timestamp from base_name (assuming it's in the format YYYYMMDD_HHMMSS_...)
            try:
                timestamp_str = base_name.split('_')[0] + '_' + base_name.split('_')[1]
                entry_date = datetime.strptime(timestamp_str[:8], '%Y%m%d').strftime('%Y-%m-%d')
                
                if entry_date == selected_date:
                    if base_name not in date_items:
                        date_items[base_name] = {
                            'id': base_name,
                            'timestamp': datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%H:%M:%S'),
                            'detection_count': 0,
                            'crops': [],
                            'image_path': None
                        }
                    
                    if entry['place'] == '-':
                        # Original image
                        date_items[base_name]['image_path'] = f"/static/uploads/{filename}"
                    elif entry['place'] > 0:
                        # Count detections
                        date_items[base_name]['detection_count'] += 1
                        date_items[base_name]['crops'].append({
                            'image': f"/static/uploads/{filename}",
                            'classification': entry['class_result'],
                            'confidence': 100  # Add actual confidence if available
                        })
            except (ValueError, IndexError):
                # Skip entries with invalid timestamp format
                continue

        # Convert to list and sort by timestamp
        items = list(date_items.values())
        items.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'items': [item for item in items if item['image_path']]  # Only return items with original images
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# History detail route - Separate from main detection functionality
@app.route('/historydetail')
def history_detail():
    """Render the history detail page"""
    return render_template('historydetail.html')

@app.route('/get_detection_details', methods=['POST'])
def get_detection_details():
    """Get historical detection details for viewing"""
    try:
        data = request.get_json()
        base_name = data.get('id')
        if not base_name:
            return jsonify({'error': 'No ID provided'}), 400

        database = load_database()
        details = {
            'original_image': None,
            'detection_image': None,
            'crops': [],
            'summary': {}
        }

        # Get all entries for this image set
        entries = [entry for entry in database if entry['file_name'].startswith(base_name)]
        
        # Calculate classification summary
        class_summary = {}
        for entry in entries:
            if entry['place'] not in ['-', 0]:
                class_name = entry['class_result']
                if class_name not in class_summary:
                    class_summary[class_name] = 0
                class_summary[class_name] += 1

        for entry in entries:
            filename = entry['file_name']
            image_path = f"/static/uploads/{filename}"

            if entry['place'] == '-':
                details['original_image'] = image_path
            elif entry['place'] == 0:
                details['detection_image'] = image_path
            elif isinstance(entry['place'], (int, float)) and entry['place'] > 0:
                details['crops'].append({
                    'image': image_path,
                    'classification': entry['class_result'],
                    'place': entry['place']
                })

        # Sort crops by place number
        details['crops'].sort(key=lambda x: x['place'])
        details['summary'] = class_summary
        details['total_detections'] = len(details['crops'])
        
        # Get timestamp from base_name
        timestamp = base_name[8:14]  # HHMMSS part
        details['timestamp'] = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}"

        return jsonify(details)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

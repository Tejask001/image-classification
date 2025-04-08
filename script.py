import json
import re
from collections import OrderedDict
from ultralytics import YOLO
import cv2
from fractions import Fraction
from nudenet import NudeDetector
import numpy as np
import os
import sys

os.environ['YOLO_VERBOSE'] = 'False'

# Add the places365 directory to the system path
current_dir = os.getcwd()
places365_dir = os.path.join(current_dir, 'places365')
sys.path.append(places365_dir)
from placesCNNModel import PlacesCNN

# blur detection method
def is_image_blurry(image_path, threshold=15):  # (Rest of the blur detection function remains unchanged)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable")

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust kernel size and sigma as needed

    h, w = blurred_img.shape
    h_block, w_block = h // 3, w // 3

    not_blurry = False  # Initialize to False. It becomes true only if one of the blocks is NOT blurry

    for i in range(3):
        for j in range(3):
            block = blurred_img[i * h_block:(i + 1) * h_block, j * w_block:(j + 1) * w_block]
            lap_var = cv2.Laplacian(block, cv2.CV_64F).var()
            #print(f"Block ({i}, {j}) variance: {lap_var}")
            if lap_var > threshold:
                not_blurry = True

    return not not_blurry

def get_image_format(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)  # Read the first 32 bytes (enough for most headers)

            # JPEG
            if header.startswith(b'\xFF\xD8\xFF'):
                return "JPEG"

            # PNG
            if header.startswith(b'\x89PNG\r\n\x1A\n'):
                return "PNG"

            # GIF
            if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return "GIF"

            # TIFF (Big Endian)
            if header.startswith(b'\x4D\x4D\x00\x2A'):
                return "TIFF (Big Endian)"

            # TIFF (Little Endian)
            if header.startswith(b'\x49\x49\x2A\x00'):
                return "TIFF (Little Endian)"

            # WebP
            if header[8:12] == b'WEBP':
                return "WebP"

            # BMP
            if header.startswith(b'BM'):
                return "BMP"

            # HEIC (check ftyp box)
            if header[4:8] == b'ftyp' and b'heic' in header[8:]:
                 return "HEIC"

            return "UNKNOWN"  # Format not recognized
    except FileNotFoundError:
        return "UNKNOWN"
    except Exception as e:
        print(f"Error reading file: {e}")  # print an error message
        return "UNKNOWN"

# object detection model
kiss_detection_model = YOLO("kiss_detector.pt")

# kiss detection model
object_detection_model = YOLO("object_dection_model.pt")

# sensitive content detection model
sensitive_content_detector = NudeDetector()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    image_path = sys.argv[1]

    # scene-classification
    scene_detector = PlacesCNN()
    scene_detections = scene_detector.detect(image_path)
    if scene_detections:
        location_detected = scene_detections[0][1]
        scene_score = scene_detections[0][0]
    else:
        location_detected = None
        scene_score = None

    # sensitive content detection using NudeNet model
    sensitive_content_detections = sensitive_content_detector.detect(image_path)

    sensitive_content_results = []

    for detection in sensitive_content_detections:
        if detection["class"] in ["BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
                                   "FEMALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
                                   "MALE_GENITALIA_EXPOSED"]:
            sensitive_content_results.append({"class": detection["class"], "score": detection["score"]})

    # blurriness detection using open-cv
    is_blur = is_image_blurry(image_path)

    # image format
    img_format = get_image_format(image_path)

    # image metadata using open-cv
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    aspect_ratio = Fraction(width, height)

    pixels = width * height

    img_size_in_bytes = os.path.getsize(image_path)
    img_size = f"{img_size_in_bytes / (1024 * 1024):.2f} MB"

    #kiss detection
    kiss_detection_results_list = []
    kiss_detection_results = kiss_detection_model(image_path, conf=0.5, verbose=False)

    print("Kiss Detection Results:")
    for result in kiss_detection_results:
        boxes = result.boxes
        names = result.names  # Get class names from the result
        for box in boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = names[class_id]  # Get class name
            kiss_detection_results_list.append({"class": class_name, "confidence": confidence}) # Added class name and confidence to dictionary

    # object detection using YOLOv11
    results = object_detection_model(image_path, conf=0.5, verbose=False)

    objects_detected = []

    for result in results:
        boxes = result.boxes

        img = cv2.imread(image_path)

        for box in boxes:
            # coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # confidence score
            confidence = box.conf[0].item()

            # class ID
            class_id = int(box.cls[0].item())

            # label
            names = object_detection_model.names  # Use the object detection model here
            label = names[class_id]

            objects_detected.append(label)

    image_data = OrderedDict([
        ("pixels", str(width) + "x" + str(height)),
        ("aspect_ratio", f"{aspect_ratio.numerator}/{aspect_ratio.denominator}"),
        ("img_size", img_size),
        ("img_format", img_format),
        ("objects_detected", list(set(objects_detected))),
        ("blurred", is_blur),
        ("sensitive_content", {"sensitive_content_results": sensitive_content_results,
                                 "kiss_detection_results": kiss_detection_results_list}), #Added kiss detection results here
        ("scene_classification", {"location": location_detected, "scene_detection_conf": scene_score})
    ])

    # Use a regular expression to find the JSON part
    json_string = json.dumps(image_data)
    print(json_string) # Print ONLY the JSON string
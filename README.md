# 🧠 Image Classification API

A comprehensive image classification system with multiple inbuilt capabilities, including:

- Image metadata extraction
- Blurriness detection
- Sensitive content detection (nudity, kissing scenes)
- Object detection
- Scene classification

## 🚀 Getting Started

### Run Flask API

1. First install the requirements
```bash
pip install flask requests jsonify opencv-python ultralytics "nudenet>=3.4.2"
```
2. To run the API run the app.py file
```bash
python app.py
```

The server will start on `http://localhost:3000`.

### API Endpoint

`POST /process_image`

**Sample `cURL` request:**
```bash
curl --location 'http://localhost:3000/process_image' \
--header 'Content-Type: application/json' \
--data '{"image_url": "https://www.example.com/image.jpg"}'
```

### Example JSON Response
```json
{
  "aspect_ratio": "3/2",
  "blurred": false,
  "img_format": "JPEG",
  "img_size": "0.19 MB",
  "objects_detected": [
    "person"
  ],
  "pixels": "1200x800",
  "scene_classification": {
    "location": "stadium/soccer",
    "scene_detection_conf": 0.22014829516410828
  },
  "sensitive_content": {
    "kiss_detection_results": [],
    "sensitive_content_results": []
  }
}
```

---

## 🔍 Features

### 🖼️ Image Metadata Extraction
Extracts key metadata from the image:
- **Format** (e.g., JPEG, PNG)
- **Size** in MB
- **Dimensions** (e.g., 1200x800)
- **Aspect Ratio**

### 🔍 Blurriness Detection
- Applies **Gaussian blur** to the image.
- Checks **variance of Laplacian** to detect image blurriness.
- Gaussian blur helps reduce the false negative rate significantly.

### 🔒 Sensitive Content Detection
- Detects **nudity** using the [NudeNet](https://github.com/notAI-tech/NudeNet) model.
- Detects **kissing scenes** using a custom-trained **YOLO** model.

### 🧭 Scene Classification
- Uses the [Places365](http://places2.csail.mit.edu/) model to classify the scene (e.g., beach, stadium, street).
- Includes confidence score in the result.

### 🎯 Object Detection
- Utilizes **YOLOv11** (Ultralytics) for detecting objects in the image.
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)

---

## 📎 Notes

- Ensure image URLs are publicly accessible.
- Replace the sample URL with a valid image link to get results.

---

## 📚 References

- [NudeNet](https://github.com/notAI-tech/NudeNet)
- [YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Places365 Scene Classification](http://places2.csail.mit.edu/)

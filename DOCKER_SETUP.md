# 🐳 Docker Setup Guide - FRS Face Recognition System

This guide explains how to set up, build, and deploy the Face Recognition System using Docker.

## 📋 Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Docker Compose (included with Docker Desktop)
- Model files in place:
  - `shape_predictor_68_face_landmarks (1).dat`
  - `dlib_face_recognition_resnet_model_v1.dat`
  - `face_recognizer.pkl` (trained model)

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
docker build -t frs-face-recognition .
```

### 2. Run the Container

```bash
docker run -it --rm frs-face-recognition
```

## 📁 Project Structure for Docker

```
FRS/
├── Dockerfile              # Main Docker configuration
├── docker-compose.yml      # Docker Compose orchestration
├── .dockerignore           # Files to exclude from build
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment (alternative)
│
├── testDlib.py             # Main face recognition script
├── train_script.py         # Model training script
├── create_Data.py          # Dataset creation script
│
├── DataSet/                # Face images (mounted, not copied)
├── *.dat                   # dlib model files
└── face_recognizer.pkl     # Trained classifier
```

## 🔧 Configuration Options

### Option 1: Development Mode (with volume mounts)

For development, mount your local files so changes are reflected immediately:

```bash
docker run -it --rm \
  -v $(pwd)/DataSet:/app/DataSet \
  -v $(pwd)/face_recognizer.pkl:/app/face_recognizer.pkl \
  frs-face-recognition python testDlib.py
```

### Option 2: Production Mode (all files baked in)

For production, include all files in the image:

```dockerfile
# In Dockerfile, uncomment:
COPY DataSet/ ./DataSet/
COPY *.dat ./
COPY face_recognizer.pkl ./
```

Then rebuild:
```bash
docker build -t frs-face-recognition:prod .
```

## 🖥️ Deployment Scenarios

### Scenario A: Local Development (Webcam)

**⚠️ Note**: Docker cannot easily access webcam. For webcam-based recognition, use conda locally:

```bash
conda env create -f environment.yml
conda activate frs
python testDlib.py
```

### Scenario B: Server Deployment (REST API)

For production, convert to a REST API that accepts image uploads. Add these files:

**`app.py`** - Flask API server:
```python
from flask import Flask, request, jsonify
import cv2
import dlib
import pickle
import numpy as np
import base64

app = Flask(__name__)

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
embedder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
with open("face_recognizer.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/recognize', methods=['POST'])
def recognize():
    # Get image from request
    data = request.json
    image_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(gray)
    
    results = []
    for rect in faces:
        shape = predictor(gray, rect)
        face_descriptor = embedder.compute_face_descriptor(rgb, shape)
        face_descriptor = np.array(face_descriptor).reshape(1, -1)
        
        proba = clf.predict_proba(face_descriptor)
        confidence = float(np.max(proba))
        label = clf.classes_[np.argmax(proba)]
        
        if confidence < 0.6:
            label = "Unknown"
        
        results.append({
            "label": label,
            "confidence": confidence,
            "bbox": {
                "x1": rect.left(),
                "y1": rect.top(),
                "x2": rect.right(),
                "y2": rect.bottom()
            }
        })
    
    return jsonify({"faces": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Updated `Dockerfile.api`**:
```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-python-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**Updated `docker-compose.api.yml`**:
```yaml
version: '3.8'
services:
  frs-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    volumes:
      - ./face_recognizer.pkl:/app/face_recognizer.pkl
    restart: unless-stopped
```

Run API server:
```bash
docker-compose -f docker-compose.api.yml up -d
```

Test the API:
```bash
# Using curl with base64 image
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'
```

### Scenario C: Batch Processing

Process multiple images from a folder:

```bash
docker run -it --rm \
  -v $(pwd)/input_images:/app/input \
  -v $(pwd)/output:/app/output \
  frs-face-recognition python batch_process.py
```

## 🔒 Security Best Practices

1. **Don't commit model files** - Use Docker volumes or secrets
2. **Use multi-stage builds** - Reduce image size
3. **Run as non-root user** - Add to Dockerfile:
   ```dockerfile
   RUN useradd -m appuser
   USER appuser
   ```
4. **Use specific versions** - Pin all dependency versions
5. **Scan for vulnerabilities**:
   ```bash
   docker scan frs-face-recognition
   ```

## 📊 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2 GB | 4 GB |
| CPU | 2 cores | 4 cores |
| Disk | 2 GB | 5 GB |
| GPU | Optional | NVIDIA (for faster processing) |

## 🐛 Troubleshooting

### "No module named cv2"
```bash
docker exec -it <container_id> pip install opencv-python
```

### "Cannot load model files"
Ensure model files are in the correct location and mounted properly.

### Build fails on Windows
Use WSL2 backend for Docker Desktop.

### Out of memory
Increase Docker memory limit in Docker Desktop settings.

## 📝 Useful Commands

```bash
# List running containers
docker ps

# View logs
docker logs <container_id>

# Enter container shell
docker exec -it <container_id> bash

# Stop all containers
docker-compose down

# Remove all images
docker system prune -a

# Check image size
docker images frs-face-recognition
```

## 🔄 CI/CD Integration

Example GitHub Actions workflow (`.github/workflows/docker.yml`):

```yaml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t frs-face-recognition .
      
      - name: Test
        run: docker run frs-face-recognition python -c "import dlib; import cv2; print('OK')"
```

---

## 📞 Support

For issues with Docker setup, check:
1. Docker Desktop is running
2. WSL2 is enabled (Windows)
3. Sufficient disk space
4. Model files are present


# FRS - Face Recognition System

A comprehensive face recognition system built with Python, dlib, and OpenCV. This project implements real-time face detection and recognition capabilities, designed for attendance tracking and presence management systems.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [REST API](#rest-api)
- [Database Schema](#database-schema)
- [Privacy & Security](#privacy--security)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This Face Recognition System (FRS) uses deep learning models to detect and recognize faces in real-time. The system is designed to:

- Capture and process face images from webcam
- Train recognition models on collected datasets
- Perform real-time face recognition with confidence scoring
- Integrate with a database system for attendance tracking
- Deploy as a REST API service using Docker

## Features

- **Real-time Face Detection**: Uses dlib's HOG-based face detector for fast and accurate face detection
- **Deep Learning Recognition**: Employs dlib's ResNet-based face recognition model (128-dimensional embeddings)
- **Multiple Recognition Methods**: Supports both dlib-based and OpenCV LBPH approaches
- **Confidence Scoring**: Provides confidence percentages for recognition predictions
- **Database Integration**: MySQL/MariaDB schema for attendance and presence management
- **Data Collection Tools**: Scripts for automated dataset creation
- **Privacy-Focused**: Sensitive data excluded from version control
- **Docker Support**: Containerized deployment for consistent environments
- **REST API**: Production-ready API for server deployment

## Architecture

### Recognition Model

The system uses **dlib's face recognition pipeline**:

1. **Face Detection**: `dlib.get_frontal_face_detector()` - HOG-based detector
2. **Landmark Detection**: `shape_predictor_68_face_landmarks.dat` - 68 facial landmarks
3. **Face Embedding**: `dlib_face_recognition_resnet_model_v1.dat` - ResNet-based encoder (128D vectors)
4. **Classification**: Scikit-learn classifier trained on face embeddings

### Alternative Method

The project also includes an **OpenCV LBPH (Local Binary Patterns Histograms)** implementation for comparison.

### Workflow

```
Webcam -> Face Detection -> Landmark Extraction -> Face Embedding -> Classification -> Result
```

### Deployment Architecture

```
+-------------------------------------------------------------+
|                    Deployment Options                        |
+-------------------------------------------------------------+
|                                                              |
|  Local Development          |    Production Server          |
|  -----------------          |    -----------------          |
|  - Conda environment        |    - Docker container         |
|  - Webcam access            |    - REST API endpoints       |
|  - Real-time GUI            |    - Scalable & portable      |
|                             |                               |
+-------------------------------------------------------------+
```

## Requirements

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.5.5.64 | Computer vision and image processing |
| `dlib` | Latest | Face detection and recognition library |
| `numpy` | 1.26.4 | Numerical computations |
| `scikit-learn` | >=1.0 | Machine learning classifier |
| `pandas` | >=1.4 | Data manipulation |
| `flask` | Latest | REST API (for server deployment) |

### System Requirements

- **Python**: 3.9 (recommended for dlib compatibility)
- **Operating System**: Windows, Linux, or macOS
- **Webcam**: For real-time recognition (local development)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: ~150MB for model files
- **Docker**: For containerized deployment (optional)

### Model Files Required

The following model files are needed (not included in repository for size/privacy reasons):

1. `shape_predictor_68_face_landmarks.dat` (~96MB) - Facial landmark predictor
2. `dlib_face_recognition_resnet_model_v1.dat` (~22MB) - Face recognition model
3. `face_recognizer.pkl` (~189KB) - Trained classifier (generated after training)

**Note**: Download dlib models from [dlib.net](http://dlib.net/files/) or train your own.

## Installation

### Option 1: Using Conda (Recommended for Local Development)

Best for Windows and when you need webcam access.

```bash
# Clone the repository
git clone https://github.com/nj-ysf/FRS-Face-Recognition-System.git
cd FRS-Face-Recognition-System

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate frs

# Run the model
python testDlib.py
```

### Option 2: Using pip (Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/nj-ysf/FRS-Face-Recognition-System.git
cd FRS-Face-Recognition-System

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the model
python testDlib.py
```

### Option 3: Using Docker (Recommended for Production)

See [Docker Deployment](#docker-deployment) section below.

### Download Model Files

1. Download `shape_predictor_68_face_landmarks.dat` from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Download `dlib_face_recognition_resnet_model_v1.dat` from [dlib.net](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
3. Extract and place both `.dat` files in the project root directory

### Set Up Database (Optional)

If using the attendance system:

```bash
mysql -u root -p < DataBase/presence_system.sql
```

## Docker Deployment

Docker provides a consistent, portable environment that eliminates dependency conflicts.

### Quick Start with Docker

```bash
# Build the API image
docker build -f Dockerfile.api -t frs-api .

# Run the container
docker-compose -f docker-compose.api.yml up -d

# Check if it's running
curl http://localhost:5000/health
```

### Deployment Scenarios

#### Scenario A: Local Development (Webcam)

For webcam-based real-time recognition, use **Conda** locally:

```bash
conda activate frs
python testDlib.py
```

> Note: Docker cannot easily access webcam. Use conda for local development with webcam.

#### Scenario B: Production Server (REST API)

Deploy as a REST API service:

```bash
# Build and start
docker-compose -f docker-compose.api.yml up -d

# View logs
docker-compose -f docker-compose.api.yml logs -f

# Stop
docker-compose -f docker-compose.api.yml down
```

#### Scenario C: Batch Processing

Process multiple images:

```bash
docker run -it --rm \
  -v $(pwd)/input_images:/app/input \
  -v $(pwd)/output:/app/output \
  frs-api python batch_process.py
```

### Docker Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Basic image for testing |
| `Dockerfile.api` | Production image with REST API |
| `docker-compose.yml` | Basic orchestration |
| `docker-compose.api.yml` | Production API deployment |
| `.dockerignore` | Excludes files from build |

For detailed Docker setup instructions, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## Project Structure

```
FRS-Face-Recognition-System/
|
+-- Docker
|   +-- Dockerfile              # Basic Docker image
|   +-- Dockerfile.api          # Production API image
|   +-- docker-compose.yml      # Basic orchestration
|   +-- docker-compose.api.yml  # Production deployment
|   +-- .dockerignore           # Docker build excludes
|   +-- DOCKER_SETUP.md         # Docker documentation
|
+-- Dependencies
|   +-- requirements.txt        # pip dependencies (pinned versions)
|   +-- environment.yml         # Conda environment
|
+-- Database
|   +-- DataBase/
|       +-- presence_system.sql # MySQL schema
|
+-- Dataset
|   +-- DataSet/                # Face images (excluded from git)
|       +-- Person1/
|       +-- Person2/
|       +-- ...
|
+-- Core Scripts
|   +-- testDlib.py             # Real-time recognition (dlib)
|   +-- real_time_test.py       # Real-time recognition (LBPH)
|   +-- train_script.py         # Model training
|   +-- create_Data.py          # Dataset capture
|   +-- create_labels.py        # Label generation
|   +-- app.py                  # REST API server
|
+-- Utilities
|   +-- faceDetction.py         # Face detection demo
|   +-- DataClean.py            # Dataset cleanup
|   +-- fixe_data.py            # Data preprocessing
|   +-- tracer_graph.py         # Visualization
|
+-- Documentation
    +-- README.md               # This file
    +-- DOCKER_SETUP.md         # Docker guide
```

## Usage

### 1. Collect Face Data

Capture face images for training:

```bash
python create_Data.py
```

**Note**: Modify `save_folder` in `create_Data.py` to specify the person's name/folder.

### 2. Generate Labels

Create `labels.csv` from dataset structure:

```bash
python create_labels.py
```

### 3. Train the Model

#### Option A: Train dlib-based model (Recommended)

```bash
python train_script.py
```

#### Option B: Train OpenCV LBPH model

```bash
python train_script.py --method lbph
```

### 4. Run Real-Time Recognition

#### Using dlib (Local with Webcam)

```bash
conda activate frs
python testDlib.py
```

**Controls**:
- Press `q` to quit
- Recognition confidence threshold: 60% (configurable)

#### Using REST API (Server)

```bash
docker-compose -f docker-compose.api.yml up -d
# Then use API endpoints (see REST API section)
```

## REST API

When deployed with Docker, the system exposes a REST API for face recognition.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/recognize` | Recognize faces (base64 image) |
| POST | `/recognize/file` | Recognize faces (file upload) |
| GET | `/labels` | Get all known labels |

### Examples

#### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "FRS Face Recognition API"
}
```

#### Recognize Face (File Upload)

```bash
curl -X POST http://localhost:5000/recognize/file \
  -F "image=@photo.jpg" \
  -F "threshold=0.6"
```

Response:
```json
{
  "success": true,
  "faces_detected": 1,
  "faces": [
    {
      "label": "John_Doe",
      "confidence": 87.5,
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 180}
    }
  ]
}
```

#### Recognize Face (Base64)

```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>", "threshold": 0.6}'
```

#### Get Known Labels

```bash
curl http://localhost:5000/labels
```

Response:
```json
{
  "labels": ["John_Doe", "Jane_Smith", "Bob_Wilson"]
}
```

## Database Schema

The project includes a MySQL/MariaDB schema for attendance tracking:

### Tables

- **`users`**: System users (admin, formateur)
- **`classes`**: Class information (name, niveau, filiere)
- **`students`**: Student records with photo paths
- **`formateurs`**: Instructor information
- **`seances`**: Class sessions (date, time, class, instructor)
- **`presences`**: Attendance records (student, session, status, time)

### Setup

```sql
-- Import the schema
source DataBase/presence_system.sql;

-- Or via command line
mysql -u root -p < DataBase/presence_system.sql
```

## Privacy & Security

### Data Protection

- **Dataset excluded**: The `DataSet/` folder containing face images is excluded from git
- **Labels excluded**: `labels.csv` with real names is not tracked
- **Model files excluded**: Large model files (`.dat`, `.pkl`) are excluded
- **Local storage only**: Sensitive biometric data remains on local machine
- **Docker security**: API runs as non-root user

### Best Practices

1. **Never commit** face images or personal data
2. **Use environment variables** for database credentials
3. **Encrypt sensitive data** in production
4. **Comply with privacy regulations** (GDPR, etc.) when handling biometric data
5. **Obtain consent** before collecting face data
6. **Use HTTPS** in production for API endpoints

## Troubleshooting

### dlib Installation Issues

**Windows** (Use Conda):
```bash
conda env create -f environment.yml
conda activate frs
```

**Linux**:
```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
pip install dlib
```

### NumPy Version Conflicts

If you see `module compiled against API version 0xf but this version of numpy is 0xe`:

```bash
pip install numpy==1.26.4
```

### Camera Not Opening

- Check camera permissions
- Verify camera is not used by another application
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

### Docker Issues

```bash
# Check container logs
docker-compose -f docker-compose.api.yml logs

# Rebuild image
docker-compose -f docker-compose.api.yml up --build

# Check if port is in use
netstat -an | grep 5000
```

### Model Files Missing

- Download required `.dat` files from dlib.net
- Place them in project root directory
- Ensure file names match exactly (case-sensitive)

## Model Performance

### dlib ResNet Model
- **Accuracy**: Typically 95%+ with good lighting
- **Speed**: ~30 FPS on modern hardware
- **Embedding Dimension**: 128D vectors
- **Confidence Threshold**: 60% (configurable)

### OpenCV LBPH Model
- **Accuracy**: Typically 85-90%
- **Speed**: ~40 FPS
- **Memory**: Lower than dlib
- **Best for**: Resource-constrained environments

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test your changes before submitting
- Include Docker updates if adding new dependencies

## License

This project is open source. Please ensure compliance with local privacy laws when using face recognition technology.

## Acknowledgments

- [dlib](http://dlib.net/) - Face detection and recognition library
- [OpenCV](https://opencv.org/) - Computer vision library
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
- [Flask](https://flask.palletsprojects.com/) - REST API framework
- [Docker](https://www.docker.com/) - Containerization platform

## Contact

For questions or issues, please open an issue on GitHub.

---

**Important**: This project handles biometric data. Always ensure you have proper authorization and comply with applicable privacy laws before collecting or processing face data.

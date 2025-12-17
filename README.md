# FRS - Face Recognition System

A comprehensive face recognition system built with Python, dlib, and OpenCV. This project implements real-time face detection and recognition capabilities, designed for attendance tracking and presence management systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Privacy & Security](#privacy--security)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This Face Recognition System (FRS) uses deep learning models to detect and recognize faces in real-time. The system is designed to:

- Capture and process face images from webcam
- Train recognition models on collected datasets
- Perform real-time face recognition with confidence scoring
- Integrate with a database system for attendance tracking

## ✨ Features

- **Real-time Face Detection**: Uses dlib's HOG-based face detector for fast and accurate face detection
- **Deep Learning Recognition**: Employs dlib's ResNet-based face recognition model (128-dimensional embeddings)
- **Multiple Recognition Methods**: Supports both dlib-based and OpenCV LBPH approaches
- **Confidence Scoring**: Provides confidence percentages for recognition predictions
- **Database Integration**: MySQL/MariaDB schema for attendance and presence management
- **Data Collection Tools**: Scripts for automated dataset creation
- **Privacy-Focused**: Sensitive data excluded from version control

## 🏗️ Architecture

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
Webcam → Face Detection → Landmark Extraction → Face Embedding → Classification → Result
```

## 📦 Requirements

### Python Packages

- `opencv-python >= 4.5.5` - Computer vision and image processing
- `dlib` - Face detection and recognition library
- `numpy >= 1.21` - Numerical computations
- `scikit-learn >= 1.0` - Machine learning classifier
- `pandas` - Data manipulation (for label management)

### System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, Linux, or macOS
- **Webcam**: For real-time recognition
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: ~150MB for model files

### Model Files Required

The following model files are needed (not included in repository for size/privacy reasons):

1. `shape_predictor_68_face_landmarks.dat` (~96MB) - Facial landmark predictor
2. `dlib_face_recognition_resnet_model_v1.dat` (~22MB) - Face recognition model
3. `face_recognizer.pkl` (~189KB) - Trained classifier (generated after training)

**Note**: Download dlib models from [dlib.net](http://dlib.net/files/) or train your own.

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/nj-ysf/FRS-Face-Recognition-System.git
cd FRS-Face-Recognition-System
```

### Step 2: Install Dependencies

#### Using pip (Recommended for Linux/macOS)

```bash
pip install -r requirements.txt
```

#### Using conda (Recommended for Windows)

```bash
conda install -c conda-forge dlib
pip install opencv-python numpy scikit-learn pandas
```

**Windows Note**: If `pip install dlib` fails, use conda as it includes pre-compiled binaries. Alternatively, install Visual C++ Build Tools.

### Step 3: Download Model Files

1. Download `shape_predictor_68_face_landmarks.dat` from [dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Download `dlib_face_recognition_resnet_model_v1.dat` from [dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
3. Extract and place both `.dat` files in the project root directory

### Step 4: Set Up Database (Optional)

If using the attendance system:

```bash
mysql -u root -p < DataBase/presence_system.sql
```

Or import via phpMyAdmin/MySQL Workbench.

## 📁 Project Structure

```
FRS-Face-Recognition-System/
│
├── DataSet/                    # Face images dataset (excluded from git)
│   ├── Person1/               # Individual folders per person
│   ├── Person2/
│   └── ...
│
├── DataBase/
│   └── presence_system.sql    # Database schema for attendance system
│
├── create_Data.py             # Script to capture face images from webcam
├── create_labels.py           # Generate labels.csv from dataset folders
├── train_script.py            # Train LBPH face recognizer
├── testDlib.py                # Real-time recognition using dlib (main script)
├── real_time_test.py          # Real-time recognition using OpenCV LBPH
├── faceDetction.py            # Basic face detection demo
├── DataClean.py               # Clean dataset (limit images per person)
├── fixe_data.py               # Data preprocessing utilities
├── tracer_graph.py            # Visualization utilities
├── test.py                    # Testing scripts
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules (excludes sensitive data)
└── README.md                  # This file
```

## 💻 Usage

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

This generates a CSV file mapping person names to numeric IDs.

### 3. Train the Model

#### Option A: Train dlib-based model (Recommended)

You'll need to create a training script that:
1. Loads images from `DataSet/`
2. Extracts face embeddings using dlib
3. Trains a scikit-learn classifier
4. Saves as `face_recognizer.pkl`

#### Option B: Train OpenCV LBPH model

```bash
python train_script.py
```

This generates `trainer1.yml` (LBPH model).

### 4. Run Real-Time Recognition

#### Using dlib (Recommended - Better Accuracy)

```bash
python testDlib.py
```

**Controls**:
- Press `q` to quit
- Recognition confidence threshold: 60% (configurable in code)

#### Using OpenCV LBPH

```bash
python real_time_test.py
```

### 5. Clean Dataset (Optional)

Limit number of images per person:

```bash
python DataClean.py
```

Modify `target` variable to set desired image count per person.

## 🗄️ Database Schema

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

## 🔒 Privacy & Security

### Data Protection

- **Dataset excluded**: The `DataSet/` folder containing face images is excluded from git
- **Labels excluded**: `labels.csv` with real names is not tracked
- **Model files excluded**: Large model files (`.dat`, `.pkl`) are excluded
- **Local storage only**: Sensitive biometric data remains on local machine

### Best Practices

1. **Never commit** face images or personal data
2. **Use environment variables** for database credentials
3. **Encrypt sensitive data** in production
4. **Comply with privacy regulations** (GDPR, etc.) when handling biometric data
5. **Obtain consent** before collecting face data

### .gitignore Coverage

The repository excludes:
- All image files (`.jpg`, `.png`, etc.)
- Model files (`.dat`, `.pkl`, `.yml`)
- Dataset folders
- Labels and CSV files with personal data
- IDE configuration files
- Python cache files

## 🔧 Troubleshooting

### dlib Installation Issues

**Windows**:
```bash
# Use conda instead of pip
conda install -c conda-forge dlib
```

**Linux**:
```bash
# Install system dependencies first
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
pip install dlib
```

### Camera Not Opening

- Check camera permissions
- Verify camera is not used by another application
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

### Low Recognition Accuracy

- Ensure good lighting conditions
- Collect more training images (30+ per person recommended)
- Ensure faces are clearly visible and frontal
- Adjust confidence threshold in code

### Model Files Missing

- Download required `.dat` files from dlib.net
- Place them in project root directory
- Ensure file names match exactly (case-sensitive)

## 📊 Model Performance

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

## 🤝 Contributing

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

## 📝 License

This project is open source. Please ensure compliance with local privacy laws when using face recognition technology.

## 🙏 Acknowledgments

- [dlib](http://dlib.net/) - Face detection and recognition library
- [OpenCV](https://opencv.org/) - Computer vision library
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⚠️ Important**: This project handles biometric data. Always ensure you have proper authorization and comply with applicable privacy laws before collecting or processing face data.

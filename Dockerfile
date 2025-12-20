# FRS - Face Recognition System
# Docker container with all dependencies pre-configured

FROM python:3.9-slim

# Install system dependencies for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libboost-filesystem-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    dlib \
    opencv-python==4.5.5.64 \
    scikit-learn>=1.0 \
    pandas>=1.4

# Copy project files
COPY . .

# Default command
CMD ["python", "testDlib.py"]


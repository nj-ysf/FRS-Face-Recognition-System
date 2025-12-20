"""
FRS - Face Recognition System REST API
Flask-based API server for face recognition
"""

from flask import Flask, request, jsonify
import cv2
import dlib
import pickle
import numpy as np
import base64
import os

app = Flask(__name__)

# =======================
# LOAD MODELS
# =======================
print("📦 Loading models...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
embedder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

with open("face_recognizer.pkl", "rb") as f:
    clf = pickle.load(f)

print("✅ Models loaded successfully!")


# =======================
# ROUTES
# =======================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FRS Face Recognition API"
    })


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize faces in an image
    
    Request body (JSON):
        - image: base64 encoded image string
        - threshold: confidence threshold (optional, default: 0.6)
    
    Response:
        - faces: list of recognized faces with labels and confidence
    """
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Get threshold
        threshold = data.get('threshold', 0.6)
        
        # Process image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(gray)
        
        results = []
        for rect in faces:
            # Get landmarks
            shape = predictor(gray, rect)
            
            # Get face embedding
            face_descriptor = embedder.compute_face_descriptor(rgb, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)
            
            # Predict
            proba = clf.predict_proba(face_descriptor)
            confidence = float(np.max(proba))
            label = clf.classes_[np.argmax(proba)]
            
            if confidence < threshold:
                label = "Unknown"
            
            results.append({
                "label": str(label),
                "confidence": round(confidence * 100, 2),
                "bbox": {
                    "x1": rect.left(),
                    "y1": rect.top(),
                    "x2": rect.right(),
                    "y2": rect.bottom()
                }
            })
        
        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "faces": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recognize/file', methods=['POST'])
def recognize_file():
    """
    Recognize faces from uploaded file
    
    Request: multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Read image
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Get threshold
        threshold = float(request.form.get('threshold', 0.6))
        
        # Process image
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
            
            if confidence < threshold:
                label = "Unknown"
            
            results.append({
                "label": str(label),
                "confidence": round(confidence * 100, 2),
                "bbox": {
                    "x1": rect.left(),
                    "y1": rect.top(),
                    "x2": rect.right(),
                    "y2": rect.bottom()
                }
            })
        
        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "faces": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/labels', methods=['GET'])
def get_labels():
    """Get all known labels/names"""
    return jsonify({
        "labels": list(clf.classes_)
    })


# =======================
# MAIN
# =======================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting FRS API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)


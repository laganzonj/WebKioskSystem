import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import insightface
import mysql.connector
from flask import Blueprint, render_template, Response, request, jsonify
import pickle
import base64
import re

# Flask blueprint
app1_blueprint = Blueprint('app1_blueprint', __name__)

# Database connection
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="doa"
)
db_cursor = db_connection.cursor()

# Models
spoof_model = load_model(r'C:\THESIS PROJECT 2024\revised\WebKioskSystem\mobilenetv2_real_vs_spoof.h5')
arcface_model = insightface.app.FaceAnalysis()
arcface_model.prepare(ctx_id=-1)

# Global capture status
capture_status = "Please stay on camera"
REAL_SPOOF_THRESHOLD = 0.84  # Spoof detection threshold

def preprocess_face(face_crop):
    """Preprocess the face for spoof detection."""
    try:
        resized_face = cv2.resize(face_crop, (224, 224))
        normalized_face = resized_face / 255.0
        expanded_face = np.expand_dims(normalized_face, axis=0)
        return expanded_face
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def validate_and_format_student_id(student_id):
    """Validate the student ID format (XXXX-XXXX)."""
    pattern = r'^(\d{4})-(\d{4})$'
    match = re.match(pattern, student_id)
    if match:
        return student_id  # Valid format
    return None  # Invalid format

@app1_blueprint.route('/check_duplicate', methods=['POST'])
def check_duplicate():
    """API endpoint to check for duplicate student ID or email."""
    data = request.json
    student_id = data.get('student_id')
    email = data.get('email')

    try:
        query = """
        SELECT COUNT(*) 
        FROM (
            SELECT student_id, email FROM face_register
            UNION ALL
            SELECT student_id, email FROM login1
        ) AS combined
        WHERE student_id = %s OR email = %s
        """
        db_cursor.execute(query, (student_id, email))
        result = db_cursor.fetchone()

        if result[0] > 0:
            return jsonify({'duplicate': True, 'message': 'Duplicate Student ID or Email found.'}), 400
        return jsonify({'duplicate': False, 'message': 'No duplicate found. Proceeding with registration.'}), 200
    except mysql.connector.Error as e:
        print(f"Database error during duplicate check: {e}")
        return jsonify({'error': 'Database error occurred.', 'message': 'An error occurred while checking duplicates. Please try again later.'}), 500

def check_duplicate_data(student_id, email):
    """Check if the student ID or email already exists in the face_register or login1 tables."""
    try:
        query = """
        SELECT COUNT(*) 
        FROM (
            SELECT student_id, email FROM face_register
            UNION ALL
            SELECT student_id, email FROM login1
        ) AS combined
        WHERE student_id = %s OR email = %s
        """
        db_cursor.execute(query, (student_id, email))
        result = db_cursor.fetchone()
        return result[0] > 0  # Return True if duplicate found
    except mysql.connector.Error as e:
        print(f"Database error during duplicate check: {e}")
        return True  # Assume duplicate if an error occurs


def save_to_database(student_id, username, section_year, sex, email, face_image, face_embedding):
    """Save user data to the face_register table."""
    try:
        # Check for duplicates in both face_register and login1
        if check_duplicate_data(student_id, email):
            print(f"Duplicate found for Student ID: {student_id} or Email: {email}")
            global capture_status
            capture_status = f"Duplicate Student ID or Email: {student_id} / {email}"
            return False  # Stop if duplicates exist

        query = """
        INSERT INTO face_register (student_id, name, section_year, sex, email, face, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        db_cursor.execute(query, (
            student_id, username, section_year, sex, email,
            face_image, face_embedding
        ))
        db_connection.commit()
        print(f"Data saved for {username} (ID: {student_id})")
        return True
    except mysql.connector.Error as e:
        print(f"Database error during save: {e}")
        db_connection.rollback()
        return False

def encode_embedding(embedding):
    """Serialize and encode the face embedding."""
    try:
        pickled_data = pickle.dumps(embedding)
        encoded_data = base64.b64encode(pickled_data).decode('utf-8')
        return encoded_data
    except Exception as e:
        print(f"Error encoding embedding: {e}")
        return None

def generate_frames(student_id, username, section_year, sex, email):
    """Generate frames for the registration process."""
    global capture_status
    try:
        validated_student_id = validate_and_format_student_id(student_id)
        if not validated_student_id:
            capture_status = "Invalid Student ID format (expected XXXX-XXXX)"
            return

        # Check for duplicate student ID or email in both face_register and login1
        if check_duplicate_data(validated_student_id, email):
            capture_status = f"Duplicate Student ID or Email: {validated_student_id} / {email}"
            return

        cap = cv2.VideoCapture(1)  # Use default camera
        if not cap.isOpened():
            capture_status = "Error: Unable to access the camera"
            return

        max_captures = 10
        captured_count = 0
        all_embeddings = []

        while captured_count < max_captures:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            faces = arcface_model.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                if face_crop.size > 0:
                    processed_face = preprocess_face(face_crop)
                    if processed_face is None:
                        continue

                    prediction = spoof_model.predict(processed_face)[0][0]
                    label = 'Real' if prediction < REAL_SPOOF_THRESHOLD else 'Spoof'
                    color = (0, 255, 0) if label == 'Real' else (0, 0, 255)

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    if label == 'Real':
                        _, img_encoded = cv2.imencode('.jpg', face_crop)
                        face_image = img_encoded.tobytes()
                        face_embedding_encoded = encode_embedding(face.embedding)
                        all_embeddings.append(face_embedding_encoded)
                        captured_count += 1

                        if captured_count >= max_captures:
                            if save_to_database(validated_student_id, username, section_year, sex, email, face_image, face_embedding_encoded):
                                capture_status = "Registered"
                            else:
                                capture_status = "Duplicate Found or Error Saving Data"
                            return

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during registration: {e}")
        capture_status = "Error during registration"
    finally:
        cap.release()

# Routes
@app1_blueprint.route('/')
def register():
    return render_template('register.html')

@app1_blueprint.route('/video_feed_register')
def video_feed_register():
    student_id = request.args.get('student_id', '').strip()
    username = request.args.get('username', 'user').strip()
    section_year = request.args.get('section_year', 'unknown_section').strip()
    sex = request.args.get('sex', 'unknown_sex').strip()
    email = request.args.get('email', 'unknown_email').strip()

    return Response(
        generate_frames(student_id, username, section_year, sex, email),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app1_blueprint.route('/get_status')
def get_status():
    global capture_status
    return jsonify({'status': capture_status})
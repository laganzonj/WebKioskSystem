import sys
import io
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import insightface
from numpy.linalg import norm
from flask import Flask, render_template, Response, send_from_directory, send_file, jsonify, session, request, redirect, url_for, jsonify
from datetime import datetime
from doa_unknown_user import get_response_unreg

import mysql.connector
import pickle
import base64
from app1 import app1_blueprint  # Import the Blueprint from app1.py
from chatbot import get_response, submit_feedback
import threading
import time
import webbrowser

# Ensure UTF-8 encoding is used for input/output on Windows systems
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
import secrets
print(secrets.token_hex(16))
app.secret_key = 'your_secret_key'


CHATBOT_URL = 'http://localhost:5001'
# Register the Blueprint from app1.py
app.register_blueprint(app1_blueprint, url_prefix='/register')
# Load the trained MobileNetV2 spoof detection model
spoof_model = load_model(r'C:\THESIS PROJECT 2024\revised\WebKioskSystem\mobilenetv2_real_vs_spoof.h5')
app.add_url_rule('/get_response_unreg', view_func=get_response_unreg, methods=['POST'])
# Initialize ArcFace for face detection and embedding extraction
arcface_model = insightface.app.FaceAnalysis()
arcface_model.prepare(ctx_id=-1)  # Use CPU (-1), for GPU set ctx_id=0

# Global variables to store recognition status, recognized user's name, and recognition counter
recognition_status = "recognizing"
recognized_user_name = ""
recognition_counter = 0
recognition_threshold = 3
stop_recognition = False
recognized_student_id = None

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Add your MySQL password if needed
    'database': 'doa'
}

def get_db_connection():
    """Establish a connection to the MySQL database."""
    return mysql.connector.connect(**db_config)

@app.route('/')
def landing_page():
   """Render the landing page."""
   return render_template('landing_page.html')

@app.route('/login')
def login():
   """Render the landing page."""
   return render_template('login.html')

# Load user embeddings with threading
def load_user_embeddings_from_db():
    connection = get_db_connection()
    embeddings_dict = {}
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT name, embedding FROM login1")
            rows = cursor.fetchall()
            for row in rows:
                name, embedding_blob = row
                embedding = pickle.loads(base64.b64decode(embedding_blob))
                if name not in embeddings_dict:
                    embeddings_dict[name] = []
                embeddings_dict[name].append(embedding)
            cursor.close()
        except mysql.connector.Error as error:
            print(f"Error reading from database: {error}")
        finally:
            connection.close()
    return embeddings_dict

# Global variable for user embeddings
user_embeddings = {}

# Function to refresh embeddings every 10 seconds
def refresh_embeddings():
    global user_embeddings
    while True:
        user_embeddings = load_user_embeddings_from_db()
        print("User embeddings reloaded")
        time.sleep(10)  # Reload every 10 seconds

# Start the embeddings refresh thread
refresh_thread = threading.Thread(target=refresh_embeddings, daemon=True)
refresh_thread.start()

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

# Function to recognize a user based on their embedding
def recognize_user(face_embedding, user_embeddings, threshold=0.5):
    global recognized_student_id
    best_match = None
    best_similarity = 0
    for student_id, embeddings in user_embeddings.items():
        for stored_embedding in embeddings:
            similarity = cosine_similarity(face_embedding, stored_embedding)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = student_id
                recognized_student_id = student_id  # Set recognized student's ID
    return best_match if best_similarity >= threshold else "Unknown"

# Preprocessing function for spoof detection
def preprocess_for_spoof_detection(face_crop):
    face_crop_resized = cv2.resize(face_crop, (224, 224))
    face_crop_array = face_crop_resized / 255.0
    face_crop_array = np.expand_dims(face_crop_array, axis=0)
    return face_crop_array

# Function to get user details
def get_user_details(username):
    connection = get_db_connection()
    user_details = {}
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT student_id, name, section_year, birthdate, sex, email, contact_no, face FROM login1 WHERE name = %s", (username,))
            user_details = cursor.fetchone() or {}
            if 'face' in user_details and user_details['face'] is not None:
                user_details['face'] = base64.b64encode(user_details['face']).decode('utf-8')
            cursor.close()
        except mysql.connector.Error as error:
            print(f"Error reading user details from database: {error}")
        finally:
            connection.close()
    return user_details

# Function to generate video frames for recognition
def generate_frames_for_recognition():
    global recognition_status, recognized_user_name, recognition_counter, stop_recognition
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        faces = arcface_model.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if face_crop.size > 0 and not stop_recognition:
                processed_face = preprocess_for_spoof_detection(face_crop)
                prediction = spoof_model.predict(processed_face)
                label = 'Real' if prediction < 0.92 else 'Spoof'

                display_label = label
                color = (0, 0, 255)  # Red for spoof or unknown user

                if label == 'Real':
                    face_embedding = face.embedding
                    recognized_user = recognize_user(face_embedding, user_embeddings)
                    if recognized_user != "Unknown":
                        if recognized_user == recognized_user_name:
                            recognition_counter += 1
                        else:
                            recognized_user_name = recognized_user
                            recognition_counter = 1
                        display_label = f"Real {recognized_user} ({recognition_counter}/{recognition_threshold})"
                        color = (0, 255, 0)  # Green for recognized user
                        if recognition_counter >= recognition_threshold:
                            recognition_status = "recognized"
                            stop_recognition = True
                    else:
                        display_label = "Real/Unknown User"
                        recognition_counter = 0
                        recognition_status = "unknown"
                else:
                    display_label = "Spoof"

                cv2.putText(frame, display_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/get_student_id')
def get_student_id():
    global recognized_user_name
    user_details = get_user_details(recognized_user_name)
    if user_details:
        student_id = user_details.get("student_id")
        return jsonify({"student_id": student_id})
    else:
        return jsonify({"error": "User not found"}), 404
    

@app.route('/video_feed_recognition')
def video_feed_recognition():
    return Response(generate_frames_for_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def get_recognition_status():
    global recognition_status
    return jsonify({'status': recognition_status})
        
@app.route('/announcements')
def announcements(): 
    global recognized_user_name
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title FROM announcements")
    announcements = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('announcements.html', name=recognized_user_name, announcements=announcements)

@app.route('/announcement/image/<int:id>')
def get_announcement_image(id):
    """Serve announcement images stored as LONGBLOB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT image_data FROM announcements WHERE id = %s', (id,))
    result = cursor.fetchone()
    image_data = result[0] if result else None
    
    cursor.close()
    conn.close()
    
    if image_data:
        return Response(image_data, mimetype='image/jpeg')
    else:
        return "Image not found", 404 

@app.route('/important_dates')
def important_dates():
    return render_template('important_dates.html')

@app.route('/api/events', methods=['GET'])
def get_events():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT title, start, end FROM event_management")
    events = []
    for row in cursor.fetchall():
        events.append({
            "title": row['title'],
            "start": row['start'].strftime("%Y-%m-%d"),  
            "end": row['end'].strftime("%Y-%m-%d")       
        })
    cursor.close()
    connection.close()
    return jsonify(events)

@app.route('/request_page')
def request_page():
    global recognized_user_name
    student_id = None
    name = None
    section_year = None
    face = None  # Initialize face variable

    # If a user is recognized, fetch their student ID, name, section_year, and face from the database
    if recognized_user_name:
        user_details = get_user_details(recognized_user_name)
        if user_details:
            student_id = user_details.get('student_id')       # Extract student_id
            name = user_details.get('name')                   # Extract name
            section_year = user_details.get('section_year')   # Extract section_year
            face = user_details.get('face')                   # Extract face (profile image data)

    # Pass student_id, name, section_year, and face to the template
    return render_template('request.html', student_id=student_id, name=name, section_year=section_year, face=face)

@app.route('/submit_request', methods=['POST']) 
def submit_request():
    student_id = request.form.get('student_id')
    request_type = request.form.get('requestType')
    reason_special = request.form.get('reasonSpecial')
    subject_code = request.form.get('subjectCode')
    subject_name = request.form.get('subjectName')
    reason_petition = request.form.get('reasonPetition')
    complaint_type = request.form.get('complaintType')
    explanation_complaint = request.form.get('explanationComplaint')
    subject_code_add_drop = request.form.get('subjectCodeAddDrop')
    subject_name_add_drop = request.form.get('subjectNameAddDrop')
    add_drop = request.form.get('addDrop')
    reason_add_drop = request.form.get('reasonAddDrop')

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

    try:
        cursor = connection.cursor(dictionary=True)

        # Fetch name, section_year, and face from login1 table based on student_id
        cursor.execute("SELECT name, section_year, face FROM login1 WHERE student_id = %s", (student_id,))
        user_details = cursor.fetchone()

        # Check if user details were found
        if not user_details:
            return jsonify({"status": "error", "message": "User details not found."}), 404

        name = user_details.get('name')
        section_year = user_details.get('section_year')
        face = user_details.get('face')  # Get face data (should be in binary format for BLOB)

        # Insert the new request with student_id, name, section_year, and face
        insert_query = """
            INSERT INTO student_request (student_id, name, section_year, face, request_type, reason_special, subject_code,
                                         subject_name, reason_petition, complaint_type, explanation_complaint,
                                         subject_code_add_drop, subject_name_add_drop, add_drop, reason_add_drop)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            student_id, name, section_year, face, request_type, reason_special, subject_code, subject_name, reason_petition,
            complaint_type, explanation_complaint, subject_code_add_drop, subject_name_add_drop, add_drop, reason_add_drop
        ))
        connection.commit()

        # Log the success and return a success response
        print("Request submitted successfully!")
        return jsonify({"status": "success", "message": "Request submitted successfully!"}), 200

    except mysql.connector.Error as error:
        # Log the database error and return an error response
        print(f"Error inserting data into student_request table: {error}")
        return jsonify({"status": "error", "message": f"Database error: {error}"}), 500

    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"Unexpected error: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred. Please try again."}), 500

    finally:
        # Ensure the database connection is closed properly
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/save_queue', methods=['POST'])
def save_queue():
    data = request.get_json()
    student_id = data.get('student_id')
    service_type = data.get('service_type')
    location = data.get('location')

    # Use server-side date and time
    current_datetime = datetime.now()
    date = current_datetime.date()  # YYYY-MM-DD
    time = current_datetime.time()  # HH:MM:SS

    if not student_id or not service_type or not location:
        return jsonify({"status": "error", "message": "Missing required data fields."}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

    try:
        cursor = connection.cursor()

        # Check for an existing queue for the student and service type on the same date
        cursor.execute("""
            SELECT queue_number, date, time 
            FROM queues 
            WHERE student_id = %s AND service_type = %s AND DATE(date) = CURDATE()
        """, (student_id, service_type))
        existing_record = cursor.fetchone()

        if existing_record:
            return jsonify({
                "status": "duplicate",
                "message": "You already have a queue number for this service.",
                "queue_number": existing_record[0],
                "date": str(existing_record[1]),
                "time": str(existing_record[2]),
            }), 200

        # Generate the next queue number based on the maximum queue number for the service type and current date
        cursor.execute("""
            SELECT MAX(queue_number) 
            FROM queues 
            WHERE DATE(date) = %s AND service_type = %s
        """, (date, service_type))
        last_queue_record = cursor.fetchone()

        queue_number = (last_queue_record[0] + 1) if last_queue_record and last_queue_record[0] else 1

        # Validate queue number uniqueness before saving
        cursor.execute("""
            SELECT id 
            FROM queues 
            WHERE queue_number = %s AND service_type = %s AND DATE(date) = %s
        """, (queue_number, service_type, date))
        duplicate_check = cursor.fetchone()

        if duplicate_check:
            return jsonify({
                "status": "error",
                "message": "Duplicate queue number detected. Please try again."
            }), 400

        # Get the user's name from the database
        cursor.execute("SELECT name FROM login1 WHERE student_id = %s", (student_id,))
        result = cursor.fetchone()
        user_name = result[0] if result else "Unknown"

        # Save the new queue record
        save_query = """
            INSERT INTO queues (student_id, name, service_type, location, date, time, queue_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(save_query, (student_id, user_name, service_type, location, date, time, queue_number))
        connection.commit()

        return jsonify({
            "status": "success",
            "message": "Queue saved successfully!",
            "queue_number": queue_number,
            "date": str(date),
            "time": str(time),
        }), 201

    except mysql.connector.Error as error:
        return jsonify({"status": "error", "message": f"Database error: {error}"}), 500

    finally:
        cursor.close()
        connection.close()


@app.route('/get_queue_number', methods=['GET'])
def get_queue_number():
    service_type = request.args.get('service_type')
    if not service_type:
        return jsonify({"status": "error", "message": "Service type is required."}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

    try:
        cursor = connection.cursor()

        today = datetime.now().date()
        cursor.execute("""
            SELECT MAX(queue_number) FROM queues 
            WHERE DATE(date) = %s AND service_type = %s
        """, (today, service_type))
        result = cursor.fetchone()

        next_queue_number = (result[0] + 1) if result and result[0] else 1
        return jsonify({
            "status": "success",
            "queue_number": next_queue_number,
            "message": "Queue number retrieved successfully."
        }), 200

    except mysql.connector.Error as error:
        return jsonify({"status": "error", "message": f"Failed to retrieve queue number: {error}"}), 500

    finally:
        cursor.close()
        connection.close()


@app.route('/get_existing_queue', methods=['GET'])
def get_existing_queue():
    student_id = request.args.get('student_id')
    service_type = request.args.get('service_type')

    if not student_id or not service_type:
        return jsonify({"status": "error", "message": "Missing student ID or service type."}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

    try:
        cursor = connection.cursor()

        # Fetch the queue details for the specific student and service type for today
        cursor.execute("""
            SELECT queue_number, date, time, service_type 
            FROM queues 
            WHERE student_id = %s AND service_type = %s AND DATE(date) = CURDATE()
        """, (student_id, service_type))
        result = cursor.fetchone()

        if result:
            return jsonify({
                "status": "success",
                "queue_number": result[0],
                "date": str(result[1]),
                "time": str(result[2]),
                "service_type": result[3],
                "message": "Queue record found."
            }), 200
        else:
            return jsonify({"status": "error", "message": "No queue found for the student and service type."}), 404

    except mysql.connector.Error as error:
        return jsonify({"status": "error", "message": f"Database error: {error}"}), 500

    finally:
        cursor.close()
        connection.close()

# Route to reload queues
@app.route('/reload_queues', methods=['GET'])
def reload_queues():
    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM queues ORDER BY date DESC, queue_number ASC")
        queues = cursor.fetchall()

        # Prepare a response
        return jsonify({
            "status": "success",
            "queues": [
                {
                    "student_id": row[1],
                    "name": row[2],
                    "service_type": row[3],
                    "location": row[4],
                    "date": row[5],
                    "time": row[6],
                    "queue_number": row[7]
                } for row in queues
            ]
        })

    except mysql.connector.Error as error:
        return jsonify({"status": "error", "message": f"Database error: {error}"}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/queue_number', methods=['GET', 'POST'])
def queue_number():
    alerts = []

    if request.method == 'POST':
        # Example logic to trigger alerts
        service_type = request.form.get('service_type')
        student_id = request.form.get('student_id')

        if not service_type or not student_id:
            alerts.append({"type": "danger", "message": "Service type or Student ID is missing!"})
        else:
            # Simulate queue saving logic
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                # Example query to simulate saving queue
                cursor.execute("""
                    INSERT INTO queues (student_id, service_type, date, time) 
                    VALUES (%s, %s, CURDATE(), CURTIME())
                """, (student_id, service_type))
                connection.commit()
                alerts.append({"type": "success", "message": "Queue number successfully saved!"})
            except Exception as e:
                alerts.append({"type": "danger", "message": f"Failed to save queue: {str(e)}"})
            finally:
                if cursor: cursor.close()
                if connection: connection.close()

    return render_template('queue_number.html', alerts=alerts)

@app.route('/doa')
def doa():
    """Render the DOA page."""
    return render_template('doa.html')
# Register chatbot routes from chat.py
app.add_url_rule('/get_response', view_func=get_response, methods=['POST'])
app.add_url_rule('/submit_feedback', view_func=submit_feedback, methods=['POST'])

@app.route('/register/doa_unknown_user')
def doa_unknown_user():
    """Render the DOA unknown user page."""
    return render_template('doa_unknown_user.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

def get_user_details(username):
    connection = get_db_connection()
    user_details = {}
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT student_id, name, section_year, birthdate, sex, email, contact_no, face FROM login1 WHERE name = %s", (username,))
            user_details = cursor.fetchone() or {}
            if 'face' in user_details and user_details['face'] is not None:
                # Convert binary face data to base64 string
                user_details['face'] = base64.b64encode(user_details['face']).decode('utf-8')
            cursor.close()
        except mysql.connector.Error as error:
            print(f"Error reading user details from database: {error}")
        finally:
            connection.close()
    return user_details
    
@app.route('/profile')
def profile():
    global recognized_user_name
    user_details = get_user_details(recognized_user_name)
    return render_template('profile.html', user=user_details)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    global recognized_user_name
    # Get form data
    name = request.form.get('name')
    section_year = request.form.get('section_year')
    birthdate = request.form.get('birthdate')
    sex = request.form.get('sex')
    email = request.form.get('email')
    contact_no = request.form.get('contact_no')
    profile_picture = request.files.get('profile_picture')

    # Initialize the connection
    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed."})

    try:
        cursor = connection.cursor()

        # Check if a profile picture was uploaded
        if profile_picture and profile_picture.filename:
            face_data = profile_picture.read()
            update_query = """
                UPDATE login1 
                SET name = %s, section_year = %s, birthdate = %s, sex = %s, email = %s, contact_no = %s, face = %s
                WHERE name = %s
            """
            cursor.execute(update_query, (name, section_year, birthdate, sex, email, contact_no, face_data, recognized_user_name))
        else:
            update_query = """
                UPDATE login1 
                SET name = %s, section_year = %s, birthdate = %s, sex = %s, email = %s, contact_no = %s
                WHERE name = %s
            """
            cursor.execute(update_query, (name, section_year, birthdate, sex, email, contact_no, recognized_user_name))

        # Commit the changes
        connection.commit()

        # Update the global variable if the name was changed
        recognized_user_name = name

        return jsonify({"status": "success", "message": "Profile updated successfully!"})

    except mysql.connector.Error as error:
        print(f"Error updating user details in the database: {error}")
        return jsonify({"status": "error", "message": "An error occurred while updating your profile."})

    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.route('/logout')
def logout():
    """Handle user logout and clear session."""
    # Clear all session data
    session.clear()
    
    # Reset global variables related to recognition
    global recognition_status, recognized_user_name, recognition_counter, stop_recognition, recognized_student_id
    recognition_status = "recognizing"
    recognized_user_name = ""
    recognition_counter = 0
    stop_recognition = False
    recognized_student_id = None
    
    # Redirect to the landing page
    return redirect(url_for('landing_page'))


@app.route('/exit')
def exit():
    """Handle user logout and clear session."""
    # Clear all session data
    session.clear()
    
    # Reset global variables related to recognition
    global recognition_status, recognized_user_name, recognition_counter, stop_recognition, recognized_student_id
    recognition_status = "recognizing"
    recognized_user_name = ""
    recognition_counter = 0
    stop_recognition = False
    recognized_student_id = None
    
    # Redirect to the landing page
    return redirect(url_for('landing_page'))

@app.route('/alerts')
def show_alerts():
    alerts = [
        {"type": "info", "message": "This is an informative alert!"},
        {"type": "success", "message": "Your operation was successful!"},
        {"type": "warning", "message": "Please check your input!"},
        {"type": "danger", "message": "Something went wrong!"},
    ]
    return render_template('alerts.html', alerts=alerts)

@app.route('/<path:filename>')
def serve_image(filename):
    """Serve images from the 'images' directory."""
    return send_from_directory('images', filename)

def open_browser():
    """Open the browser once."""
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  # Ensure only the reloader process opens the browser
        webbrowser.open("http://127.0.0.1:5002")
        
if __name__ == '__main__':
    threading.Timer(1, open_browser).start()  # Start the browser opener thread
    app.run(debug=True, host='0.0.0.0', port=5002)
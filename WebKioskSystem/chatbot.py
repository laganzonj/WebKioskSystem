import pandas as pd
import re
import csv
import os
from datetime import datetime, timedelta
from difflib import get_close_matches
from spellchecker import SpellChecker
from flask import Flask, render_template, jsonify
from flask import request as flask_request
import random
import dateparser
from threading import Thread
import pymysql
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
import logging
from fuzzywuzzy import fuzz

# File paths (update these paths to match your actual file locations)
DATASET_PATH = r'C:\THESIS PROJECT 2024\revised\WebKioskSystem\dataset\FAQ_CONVERSATION_Bayesian.csv'

# Flask app
chatbot = Flask(__name__)

# Load datasets
faq_data = pd.read_csv(DATASET_PATH)
faq_data.to_csv(DATASET_PATH, index=False)

# Set session lifetime to 30 minutes
chatbot.permanent_session_lifetime = timedelta(minutes=30)
def connect_to_mysql():
    # Adjust the connection string to match your MySQL database credentials
    return create_engine('mysql+pymysql://root:@localhost/doa')
def validate_schedule_data(data):
    """
    Validates the schedule data to ensure required columns exist and data is non-empty.
    """
    required_columns = ['Section', 'Course_Code', 'Instructor', 'Day', 'Time', 'Room', 'Modality']
    if data.empty:
        raise ValueError("The schedule data is empty. Please ensure the database table is populated.")
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in schedule data: {', '.join(missing_columns)}")
    logging.info("Schedule data successfully validated.")

def load_schedule_data():
    """
    Loads schedule data from the database and validates it.
    """
    try:
        engine = connect_to_mysql()
        with engine.connect() as connection:
            query = "SELECT * FROM schedule_data"
            logging.info("Executing query to load schedule data.")
            schedule_data = pd.read_sql(query, connection)

        # Validate the loaded data
        validate_schedule_data(schedule_data)

        logging.info("Schedule data successfully loaded and validated.")
        return schedule_data

    except Exception as e:
        logging.error(f"Error loading schedule data: {e}")
        raise

# Load and validate the schedule data
try:
    schedule_data = load_schedule_data()
    print("Schedule data loaded successfully.")
except Exception as e:
    print(f"Failed to load schedule data: {e}")

# Query to load all conversation logs from the database
def load_conversation_logs():
    # Connect to the database
    engine = connect_to_mysql()

    # Use a connection from the engine
    with engine.connect() as connection:
        # Perform your operations here (e.g., loading conversation logs)
        sql_query = "SELECT * FROM conversation_logs"
        conversation_log = pd.read_sql(sql_query, connection)
        # The connection will automatically close when exiting the 'with' block

    return conversation_log
# Call the function to load the logs
conversation_log = load_conversation_logs()

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)



# Function to log conversations into MySQL
def log_conversation(question, answer, feedback_rating, user_correction=None, prior=1e-10, posterior=1e-10):
    try:
        engine = connect_to_mysql()

        # Calculate feedback and timestamp
        positive_feedback = 1 if feedback_rating >= 3 else 0
        negative_feedback = 1 if feedback_rating < 3 else 0
        log_date = datetime.now().strftime('%Y-%m-%d')

        # Define the SQL query using SQLAlchemy's text() for parameterized queries
        sql_query = text("""
            INSERT INTO conversation_logs 
            (timestamp, question, answer, feedback_rating, correction, positive_feedback, negative_feedback, prior, posterior)
            VALUES (:timestamp, :question, :answer, :feedback_rating, :correction, :positive_feedback, :negative_feedback, :prior, :posterior)
        """)

        # Execute the query in a connection context
        with engine.connect() as connection:
            connection.execute(sql_query, {
                'timestamp': log_date,
                'question': question,
                'answer': answer,
                'feedback_rating': feedback_rating,
                'correction': user_correction,
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback,
                'prior': prior,
                'posterior': posterior
            })
            connection.commit()  # Commit the transaction

        print("Conversation logged successfully.")

    except Exception as e:
        print(f"Error logging conversation: {e}")

# Initialize probabilities and feedback tracking
def initialize_probabilities(faq_data):
    if 'prior' not in faq_data.columns:
        faq_data['prior'] = 0.4 # Initial uniform probability
    if 'positive_feedback' not in faq_data.columns:
        faq_data['positive_feedback'] = 0  # Track positive feedback
    if 'negative_feedback' not in faq_data.columns:
        faq_data['negative_feedback'] = 0  # Track negative feedback
    if 'posterior' not in faq_data.columns:
        faq_data['posterior'] = faq_data['prior']  # Posterior starts as prior
    return faq_data

faq_data = initialize_probabilities(faq_data)

# Ensure all column names are lowercase to avoid duplicates
def ensure_lowercase_columns(df):
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Load conversation logs directly from the MySQL database
conversation_log = load_conversation_logs()
conversation_log = ensure_lowercase_columns(conversation_log)

# Normalize the conversation log to match the format of faq_data
def initialize_log_probabilities(log_data):
    log_data.columns = log_data.columns.str.lower()
    if 'prior' not in log_data.columns:
        log_data['prior'] = 1 / len(log_data)
    if 'positive_feedback' not in log_data.columns:
        log_data['positive_feedback'] = 0
    if 'negative_feedback' not in log_data.columns:
        log_data['negative_feedback'] = 0
    if 'posterior' not in log_data.columns:
        log_data['posterior'] = log_data['prior']
    return log_data

# Initialize probabilities for log data
conversation_log = initialize_log_probabilities(conversation_log)

# Define required columns for both datasets
required_columns = ['question', 'answer', 'posterior', 'positive_feedback', 'negative_feedback']

def ensure_columns(df, required_columns):
    # Add any missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            if col == 'posterior':
                df[col] = 1 / len(df)  # Default posterior value
            elif col in ['positive_feedback', 'negative_feedback']:
                df[col] = 0  # Default feedback values
            else:
                df[col] = ''  # Default empty string for text fields like 'question' and 'answer'
    return df

# Ensure both faq_data and conversation_log have these columns
faq_data = ensure_columns(faq_data, required_columns)
conversation_log = ensure_columns(conversation_log, required_columns)

def get_combined_data(faq_data, conversation_log):
    # Ensure both datasets have the required columns
    faq_data = ensure_columns(faq_data, required_columns)
    conversation_log = ensure_columns(conversation_log, required_columns)

    # Add a 'source' column to distinguish between FAQ and conversation log entries
    faq_data['source'] = 'FAQ'
    conversation_log['source'] = 'Conversation Log'

    # Normalize the conversation_log to match faq_data columns
    log_data_relevant = conversation_log[['question', 'answer', 'posterior', 'positive_feedback', 'negative_feedback']]

    # Reset index for both dataframes to ensure unique indexing
    faq_data = faq_data.reset_index(drop=True)
    log_data_relevant = log_data_relevant.reset_index(drop=True)

    # Ensure column names are consistent before concatenation
    faq_data.columns = faq_data.columns.str.lower()
    log_data_relevant.columns = log_data_relevant.columns.str.lower()

    combined_data = pd.concat([faq_data, log_data_relevant], ignore_index=True)

    # # Debugging: Check for duplicated columns
    # print(f"Columns before dropping duplicates: {combined_data.columns}")

    # Drop duplicates if necessary
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    # print(f"Columns after dropping duplicates: {combined_data.columns}")

    return combined_data

# Normalize text for comparison, including spell-checking
def normalize_text(text):
    if isinstance(text, str):  # Ensure the input is a string
        text = text.strip().lower()  # Lowercase and strip spaces
        # Add additional replacements or spell-check handling here
        text = text.replace("dean", "daen")  # Correct known misspelling as an example
        return text
    return ""  # Return empty string if text is not a valid string (e.g., NaN)

intent_probabilities = {
    'class_schedule': 0.20,
    'faculty_meeting': 0.20,
    'annual_event': 0.20,
    'one_time_event': 0.20,
    'instructor_query': 0.20
}

# Decay process for all intents, with less decay for special intents (e.g., hot topics)
def decay_intents(intent_probabilities, decay_rate=0.01, special_intents=None):
    special_intents = special_intents or []
    
    for intent in intent_probabilities.keys():
        print(f"Type of intent_probabilities: {type(intent_probabilities)}")
        # Apply reduced decay for special intents like hot topics or high-priority events
        if intent in special_intents:
            intent_probabilities[intent] *= (1 - decay_rate / 2)  # Less decay for special cases
        else:
            intent_probabilities[intent] *= (1 - decay_rate)

# Adjust posterior with feedback (adaptive based on confidence level)
def adjust_with_feedback(posterior, feedback):
    if feedback >= 3:
        # Boost posterior more if confidence (posterior) is low, less if it's high
        return posterior + (posterior * 0.2) if posterior < 0.5 else posterior + (posterior * 0.05)
    else:
        # Decrease posterior if feedback is negative, reduce by 10%
        return posterior - (posterior * 0.1)

# Main function to detect event intent based on keywords with feedback and weight decay
def detect_event_intent(question, intent_probabilities, feedback=None, decay_rate=0.01, special_intents=None):
    event_intents = {
        'class_schedule': ['class', 'lecture', 'schedule', 'classroom', 'section', 'instructor', 'time', 'room', 'subject'],
        'faculty_query': ['dean', 'faculty', 'professor', 'staff', 'head'],
        'annual_event': ['graduation', 'orientation', 'annual', 'yearly', 'conference', 'seminar'],
        'general_info': ['information', 'office', 'location', 'help', 'hours']
    }

    # Normalize question
    question = question.lower()
    detected_intent = 'general_info'  # Default to general info if no other intent is detected
    
    # Check if the question contains keywords for any of the intents
    for intent, keywords in event_intents.items():
        if any(keyword in question for keyword in keywords):
            detected_intent = intent
            break

    # Apply decay to intents as per the logic
    decay_intents(intent_probabilities, decay_rate, special_intents)

    # Adjust probabilities based on feedback
    if feedback is not None:
        intent_probabilities[detected_intent] = adjust_with_feedback(intent_probabilities[detected_intent], feedback)

    # Normalize probabilities
    total_probability = sum(intent_probabilities.values())
    if total_probability > 0:
        intent_probabilities = {k: v / total_probability for k, v in intent_probabilities.items()}

    return detected_intent, intent_probabilities

from sqlalchemy import text

def clean_conversation_log():
    engine = connect_to_mysql()

    # Use a transaction context to manage the connection
    with engine.connect() as connection:
        # Define the delete query to remove duplicate records
        delete_query = text("""
            DELETE c1
            FROM conversation_logs c1
            INNER JOIN conversation_logs c2
            WHERE 
                c1.id > c2.id AND  -- Keep the first occurrence, delete later ones
                c1.question = c2.question AND
                c1.answer = c2.answer AND
                c1.timestamp = c2.timestamp
        """)

        # Execute the query to remove duplicates
        connection.execute(delete_query)

    print("Duplicate rows removed from conversation_logs table.")

# Call this function to remove duplicates
clean_conversation_log()

# Bayesian updating with cumulative adjustments
def bayesian_update(prior, likelihood):
    prior = max(prior, 1e-10)
    likelihood = max(likelihood, 1e-10)
    
    posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)))
    
    # Debugging
    print(f"Bayesian Update - Prior: {prior}, Likelihood: {likelihood}, Posterior: {posterior}")
    
    return max(posterior, 0.01)


def update_probabilities_and_log(faq_data, conversation_log, question, feedback_rating, user_correction=None, decay_factor=0.5):
    normalized_question = normalize_text(question)

    # Normalize questions in both datasets
    faq_data['question'] = faq_data['question'].astype(str).str.lower()
    conversation_log['question'] = conversation_log['question'].astype(str).str.lower()

    # Identify relevant entries in both datasets
    relevant_entries = faq_data[faq_data['question'] == normalized_question]
    relevant_log_entries = conversation_log[conversation_log['question'] == normalized_question]

    if relevant_entries.empty and relevant_log_entries.empty and user_correction:
        # If no relevant entry exists, add a new one with a low initial prior
        new_entry = pd.DataFrame({
            'question': [normalized_question],
            'answer': [user_correction],
            'creation_time': [datetime.now().strftime('%d/%m/%Y')],
            'prior': [0.1],  # Start with a low prior for new corrections
            'posterior': [0.5],  # Initial confidence in the new entry
            'positive_feedback': [1 if feedback_rating >= 3 else 0],
            'negative_feedback': [1 if feedback_rating < 3 else 0]
        })
        # Append new entry to the FAQ data
        faq_data = pd.concat([faq_data, new_entry], ignore_index=True)

        # Log the new entry in MySQL conversation log with a reduced prior
        log_conversation(question, user_correction, feedback_rating, user_correction, prior=0.1, posterior=0.5)
    else:
        # Update existing FAQ entries with Bayesian adjustment and apply decay factor
        for index, row in relevant_entries.iterrows():
            prior = row['prior'] * decay_factor  # Apply decay to reduce the prior
            likelihood = feedback_rating / 5.0
            posterior = bayesian_update(prior, likelihood)

            # Update prior and posterior, apply correction if needed
            faq_data.at[index, 'prior'] = prior  # Save the decayed prior
            faq_data.at[index, 'posterior'] = posterior
            if user_correction and row['answer'] != user_correction:
                faq_data.at[index, 'answer'] = user_correction
                faq_data.at[index, 'posterior'] = posterior  # Update posterior to reflect correction

            # Log the updated conversation
            log_conversation(question, row['answer'], feedback_rating, user_correction, prior, posterior)

        # Repeat for conversation log entries
        for index, row in relevant_log_entries.iterrows():
            prior = row['prior'] * decay_factor  # Apply decay
            likelihood = feedback_rating / 5.0
            posterior = bayesian_update(prior, likelihood)

            # Update the conversation log entry in MySQL
            log_conversation(question, row['answer'], feedback_rating, user_correction, prior, posterior)

    # Save updates to CSV to ensure persistence
    faq_data.to_csv(DATASET_PATH, index=False)
    return faq_data, conversation_log

# Recalculate posterior probabilities in the recalibration process
def recalibrate_probabilities(faq_data, conversation_log):
    # Combine datasets for recalibration
    combined_data = pd.concat([faq_data, conversation_log], ignore_index=True)

    # Normalize questions for comparison
    combined_data['normalized_question'] = combined_data['question'].apply(normalize_text)

    # Group by normalized question and answer to aggregate feedback and prior
    grouped_data = combined_data.groupby(['normalized_question', 'answer']).agg({
        'prior': 'sum',
        'positive_feedback': 'sum',
        'negative_feedback': 'sum'
    }).reset_index()

    # Recalculate likelihood and posterior
    total_feedback = grouped_data['positive_feedback'] + grouped_data['negative_feedback']
    grouped_data['likelihood'] = grouped_data['positive_feedback'] / total_feedback.replace(0, 1)  # Avoid division by zero
    grouped_data['posterior'] = grouped_data.apply(
        lambda row: bayesian_update(row['prior'], row['likelihood']),
        axis=1
    )

    # Update the MySQL database with recalibrated values
    engine = connect_to_mysql()
    connection = engine.connect()

    for index, row in grouped_data.iterrows():
        # Update both FAQ data and conversation logs in the MySQL database
        update_query = text("""UPDATE conversation_logs SET 
                                prior = :prior, 
                                posterior = :posterior, 
                                positive_feedback = :positive_feedback, 
                                negative_feedback = :negative_feedback
                                WHERE question = :question AND answer = :answer""")
        connection.execute(update_query, {
            'prior': row['prior'],
            'posterior': row['posterior'],
            'positive_feedback': row['positive_feedback'],
            'negative_feedback': row['negative_feedback'],
            'question': row['normalized_question'],
            'answer': row['answer']
        })

    connection.close()

    return faq_data, conversation_log

# Enhanced section extraction for various formats (e.g., "BSIT 1A", "CS4B", "WAM3A")
def extract_section_from_query(question):
    patterns = [
        r"(BSIT|BSCS|IT|CS|WAM|SMP)\s*[-]?\s*(\d)\s*([A-G])",  # Matches "BSCS 4B", "CS4B", "BSIT-1B", "WAM 3A"
        r"(BSIT|BSCS|IT|CS|WAM|SMP)\s*[-]?\s*(\d)\s*([A-G])-?WAM",  # Matches sections like "3AWAM", "BSIT-3C-WAM"
        r"(\d)\s*([A-G])\s*(SMP|WAM)?",  # Matches sections like "4B-SMP", "3AWAM"
    ]

    for pattern in patterns:
        match = re.search(pattern, question.replace(" ", ""), re.IGNORECASE)
        if match:
            if len(match.groups()) == 3:  # Full match (course, year, section)
                course, year, section = match.groups()
                return f"{course.upper()} {year}{section.upper()}"
            elif len(match.groups()) == 2:  # Only year and section
                year, section = match.groups()
                return f"BSCS {year}{section.upper()}"  # Default to BSCS if only year and section

    return None  # Return None if no match is found

# Fuzzy matching for section names
def fuzzy_match_section(input_section, available_sections, cutoff=0.8):
    # Adjusted cutoff to ensure only close matches are picked
    closest_match = get_close_matches(input_section, available_sections, n=1, cutoff=cutoff)
    
    if closest_match:
        # Log for debugging and return the match
        print(f"Closest section match: {closest_match[0]}")
        return closest_match[0]
    else:
        print("No close section match found.")
        return None  # Return None if no close matches are found

def determine_query_type(question):
    question = question.lower()
    if 'subject' in question or 'course' in question:
        return 'course'
    elif 'instructor' in question or 'who is teaching' in question:
        return 'instructor'
    elif 'room' in question or 'location' in question:
        return 'room'
    elif 'time' in question or 'when' in question:
        return 'time'
    elif 'classes' in question or 'schedule' in question:
        return 'schedule'
    elif 'modality' in question:
        return 'modality'
    elif 'type' in question:
        return 'type'
    else:
        return 'all'  # Default to returning all details

def extract_instructor_from_query(question):
    """
    Extracts instructor names from the query using fuzzy matching or pattern recognition.
    """
    # Normalize the question text
    question = question.lower().strip()
    
    # List of instructors from the dataset
    instructor_list = schedule_data['Instructor'].str.strip().unique()
    instructor_list = [instructor.lower() for instructor in instructor_list if isinstance(instructor, str)]

    # Fuzzy match the instructor name in the query
    closest_match = get_close_matches(question, instructor_list, n=1, cutoff=0.8)

    if closest_match:
        print(f"Closest instructor match: {closest_match[0]}")  # Debugging
        return closest_match[0].title()  # Return matched instructor name in title case

    return None  # Return None if no instructor match is found

def extract_room_from_query(question):
    """
    Extracts room information from the query using pattern matching or keyword extraction.
    """
    # Normalize the question text
    question = question.lower().strip()

    # List of rooms from the dataset
    room_list = schedule_data['Room'].str.strip().unique()
    room_list = [room.lower() for room in room_list if isinstance(room, str)]

    # Fuzzy match the room name in the query
    closest_match = get_close_matches(question, room_list, n=1, cutoff=0.8)

    if closest_match:
        print(f"Closest room match: {closest_match[0]}")  # Debugging
        return closest_match[0].upper()  # Return matched room name in uppercase

    # Handle common phrases or room-related keywords
    room_keywords = ['maclab', 'lab', 'tesol', 'classroom']
    for keyword in room_keywords:
        if keyword in question:
            return keyword.upper()

    return None  # Return None if no room match is found

def extract_course_code_from_query(question):
    """
    Extracts course code information from the query using pattern matching or keyword extraction.
    """
    # Normalize the question text
    question = question.lower().strip()

    # List of course codes from the dataset
    course_code_list = schedule_data['Course_Code'].str.strip().unique()
    course_code_list = [course.lower() for course in course_code_list if isinstance(course, str)]

    # Fuzzy match the course code in the query
    closest_match = get_close_matches(question, course_code_list, n=1, cutoff=0.8)

    if closest_match:
        print(f"Closest course code match: {closest_match[0]}")  # Debugging
        return closest_match[0].upper()  # Return matched course code in uppercase

    return None  # Return None if no course code match is found

def extract_modality_from_query(question):
    """
    Extracts modality information (e.g., Face-to-Face, Online) from the query.
    """
    # Normalize the question text
    question = question.lower()

    # List of modalities from the dataset
    modalities = ['face-to-face', 'online', 'hybrid']
    for modality in modalities:
        if modality in question:
            return modality.capitalize()

    return None  # Return None if no modality is found

def extract_type_from_query(question):
    """
    Extracts type information (e.g., Lecture, Laboratory) from the query.
    """
    # Normalize the question text
    question = question.lower()

    # List of types from the dataset
    types = ['lecture', 'laboratory', 'seminar']
    for type_ in types:
        if type_ in question:
            return type_.capitalize()

    return None  # Return None if no type is found

def get_schedule_info(
    section=None,
    course_code=None,
    instructor=None,
    day_of_week=None,
    time_of_day=None,
    room=None,
    modality=None,
    type_=None,
    query_type="all",
    semester=None,
    academic_year=None
):
    """
    Enhanced and comprehensive function to handle all possible schedule queries dynamically for sections, instructors, 
    rooms, courses, days, time, modality, and type.
    """
    # Define default semester and academic year
    current_semester = 'First Semester'  # Adjust as per the current semester
    current_academic_year = '2024-2025'  # Adjust as per the current academic year

    # Use defaults if not provided
    semester = semester or current_semester
    academic_year = academic_year or current_academic_year

    # Normalize and prepare the dataset
    filtered_data = schedule_data.copy()
    for column in ['Section', 'Course_Code', 'Instructor', 'Day', 'Time', 'Room', 'Modality', 'Type', 'Semester', 'Academic_Year']:
        filtered_data[column] = filtered_data[column].str.strip().str.capitalize() if column in ['Day', 'Modality', 'Type'] else filtered_data[column].str.strip()

    # Filter by semester and academic year
    filtered_data = filtered_data[
        (filtered_data['Semester'].str.contains(semester, case=False, na=False)) &
        (filtered_data['Academic_Year'].str.contains(academic_year, case=False, na=False))
    ]

    # Apply filters based on the provided parameters
    if section:
        section = section.upper().strip()
        matched_section = fuzzy_match_section(section, filtered_data['Section'].unique())
        if matched_section:
            filtered_data = filtered_data[filtered_data['Section'] == matched_section]
        else:
            return f"No matching schedule found for section '{section}' in {semester}, {academic_year}."

    if course_code:
        filtered_data = filtered_data[filtered_data['Course_Code'].str.contains(course_code, case=False, na=False)]

    if instructor:
        instructor = instructor.strip()
        matched_instructor = fuzzy_match_section(instructor, filtered_data['Instructor'].unique())
        if matched_instructor:
            filtered_data = filtered_data[filtered_data['Instructor'].str.contains(matched_instructor, case=False)]
        else:
            return f"No matching schedule found for instructor '{instructor}' in {semester}, {academic_year}."

    if day_of_week:
        day_of_week = day_of_week.capitalize().strip()
        filtered_data = filtered_data[filtered_data['Day'] == day_of_week]

    if time_of_day:
        filtered_data = filter_by_time_of_day(filtered_data, time_of_day)

    if room:
        filtered_data = filtered_data[filtered_data['Room'].str.contains(room, case=False, na=False)]

    if modality:
        filtered_data = filtered_data[filtered_data['Modality'].str.contains(modality, case=False, na=False)]

    if type_:
        filtered_data = filtered_data[filtered_data['Type'].str.contains(type_, case=False, na=False)]

    # Handle no matching results
    if filtered_data.empty:
        criteria = []
        if section:
            criteria.append(f"section '{section}'")
        if course_code:
            criteria.append(f"course '{course_code}'")
        if instructor:
            criteria.append(f"instructor '{instructor}'")
        if day_of_week:
            criteria.append(f"on {day_of_week}")
        if time_of_day:
            criteria.append(f"during {time_of_day}")
        if room:
            criteria.append(f"room '{room}'")
        if modality:
            criteria.append(f"modality '{modality}'")
        if type_:
            criteria.append(f"type '{type_}'")
        criteria.append(f"in {semester}, {academic_year}")
        return f"No matching schedule found for {', '.join(criteria)}."

    # Build responses based on query type
    response = []

    if query_type == "instructor":
        response.append(f"Schedule for Instructor {instructor} in {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} with Section {row['Section']} in {row['Room']} ({row['Modality']}, {row['Type']})"
            )

    elif query_type == "section":
        response.append(f"Schedule for Section {section} in {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} taught by {row['Instructor']} in {row['Room']} ({row['Modality']}, {row['Type']})"
            )

    elif query_type == "room":
        response.append(f"Classes in Room {room} in {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} with Section {row['Section']} taught by {row['Instructor']} ({row['Modality']}, {row['Type']})"
            )

    elif query_type == "subject":
        response.append(f"Schedule for Subject {course_code} in {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} with Section {row['Section']} taught by {row['Instructor']} in {row['Room']} ({row['Modality']}, {row['Type']})"
            )

    elif query_type == "time":
        response.append(f"Classes during {time_of_day} in {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} with Section {row['Section']} taught by {row['Instructor']} in {row['Room']} ({row['Modality']}, {row['Type']})"
            )

    else:
        response.append(f"Complete schedule details for {semester}, {academic_year}:")
        for _, row in filtered_data.iterrows():
            response.append(
                f"{row['Day']} {row['Time']}: {row['Course_Code']} with Section {row['Section']} taught by {row['Instructor']} in {row['Room']} ({row['Modality']}, {row['Type']})"
            )

    return "\n".join(response)


def filter_by_time_of_day(data, time_of_day):
    """
    Filters schedule data based on the specified time of day.
    """
    if time_of_day == "morning":
        return data[data['Time'].str.contains(r'\b(7|8|9|10|11):[0-5]\d\s*AM', case=False, regex=True)]
    elif time_of_day == "afternoon":
        return data[data['Time'].str.contains(r'\b(12|1|2|3|4):[0-5]\d\s*PM', case=False, regex=True)]
    elif time_of_day == "evening":
        return data[data['Time'].str.contains(r'\b(5|6|7|8|9):[0-5]\d\s*PM', case=False, regex=True)]
    elif time_of_day == "now":
        now = datetime.now().time()
        return data[data['Time'].apply(lambda x: is_time_within_range(now, x))]
    return data

def is_time_within_range(current_time, time_range):
    """
    Check if the current time is within a given time range.
    """
    try:
        start_time, end_time = [datetime.strptime(t.strip(), '%I:%M %p').time() for t in time_range.split('-')]
        return start_time <= current_time <= end_time
    except ValueError:
        return False

# Handle time-specific queries (e.g., "today," "tomorrow," or specific days of the week)
def handle_time_specific_queries(question):
    today = datetime.now()
    days_of_week = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    if "today" in question.lower():
        return today.strftime('%A')
    elif "tomorrow" in question.lower():
        return (today + timedelta(days=1)).strftime('%A')
    elif "yesterday" in question.lower():
        return (today - timedelta(days=1)).strftime('%A')
    else:
        for day_name in days_of_week:
            if day_name in question.lower():
                # Calculate the next or previous occurrence of that day
                if "next" in question.lower():
                    days_ahead = (days_of_week[day_name] - today.weekday()) % 7
                    return (today + timedelta(days=days_ahead)).strftime('%A')
                elif "last" in question.lower():
                    days_behind = (today.weekday() - days_of_week[day_name]) % 7
                    return (today - timedelta(days=days_behind)).strftime('%A')
                return day_name.capitalize()  # Capitalize the first letter
    return None  # Return None if no day is found in the query

def determine_time_of_day(question):
    # Check for specific keywords related to time of day
    if 'morning' in question.lower():
        return 'morning'
    elif 'afternoon' in question.lower():
        return 'afternoon'
    elif 'evening' in question.lower() or 'night' in question.lower():
        return 'evening'

    # Check for specific AM/PM format
    am_match = re.search(r'\b(\d{1,2})(:\d{2})?\s*(am|AM)\b', question)
    pm_match = re.search(r'\b(\d{1,2})(:\d{2})?\s*(pm|PM)\b', question)

    if am_match:
        hour = int(am_match.group(1))
        return 'morning' if 1 <= hour < 12 else 'early morning'
    elif pm_match:
        hour = int(pm_match.group(1))
        if 12 <= hour < 5:
            return 'afternoon'
        elif hour >= 5:
            return 'evening'

    # Default to unknown if no time of day is found
    return 'unknown'

# Extract temporal reference for specific dates, weekdays, and relative dates
def extract_temporal_reference(question):
    today = datetime.now()
    
    # Handle relative time expressions in the query
    if 'now' in question or 'current' in question:
        return 'current'
    elif 'today' in question:
        return today.strftime('%Y-%m-%d')
    elif 'tomorrow' in question:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    elif 'yesterday' in question:
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')

    # Handle weekday names (e.g., "on Monday")
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    
    for day_name, day_number in weekdays.items():
        if day_name in question.lower():
            # Calculate the next occurrence of that day of the week
            days_ahead = day_number - today.weekday()
            if days_ahead <= 0:  # If the day has already passed this week
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime('%Y-%m-%d')

    # Handle specific date formats using regex
    date_match = re.search(r'(\b\w{3,9}\s+\d{1,2}(?:,\s+\d{4})?)', question)
    if date_match:
        try:
            # Attempt to convert it to a full date if the year is included
            parsed_date = datetime.strptime(date_match.group(), '%B %d, %Y')
        except ValueError:
            try:
                # Handle cases like "Jan 1" without the year (assume current year)
                parsed_date = datetime.strptime(date_match.group(), '%B %d')
                parsed_date = parsed_date.replace(year=today.year)
            except ValueError:
                parsed_date = None

        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')  # Return date in ISO format (YYYY-MM-DD)

    # **Add the following block to handle "in X weeks" relative dates**
    elif "in" in question.lower() and "week" in question.lower():
        weeks_ahead_match = re.search(r'\bin\s*(\d+)\s*week', question.lower())
        if weeks_ahead_match:
            weeks_ahead = int(weeks_ahead_match.group(1))  # Extract the number of weeks
            target_date = today + timedelta(weeks=weeks_ahead)
            return target_date.strftime('%Y-%m-%d')

    return 'current'  # Default to current if no specific time is found

# Handle corrections with more interactive feedback options
def handle_corrections(faq_data, question, feedback_rating, user_correction=None):
    if feedback_rating < 3 and user_correction:
        print(f"Updating knowledge base for: {question}")
        update_probabilities_and_log(faq_data, question, feedback_rating, user_correction=user_correction)
        return f"Thank you! I've updated the information: {user_correction}"

    return "Thanks for the feedback!"

def get_closest_question(question, faq_data):
    normalized_question = normalize_text(question)
    available_questions = faq_data['question'].apply(normalize_text).tolist()
    
    # Find the closest matching question(s) with a cutoff similarity
    closest_matches = get_close_matches(normalized_question, available_questions, n=1, cutoff=0.6)
    
    if closest_matches:
        return closest_matches[0]  # Return the closest match
    return None  # Return None if no close matches are found

def extract_year_from_query(question):
    """
    Extracts the year from the question if provided, otherwise returns None.
    Uses regex to capture any four-digit year in the question.
    """
    year_match = re.search(r'\b(20\d{2})\b', question)
    if year_match:
        return year_match.group(0)
    return None

def parse_natural_language_dates(question):
    """
    Uses the dateparser library to parse any natural language date expressions in the question.
    Returns a standardized date (YYYY-MM-DD) if found, otherwise returns None.
    """
    parsed_date = dateparser.parse(question)
    if parsed_date:
        return parsed_date.strftime('%Y-%m-%d')
    return None

# Fuzzy matching for questions
def fuzzy_match_question(input_question, available_questions, cutoff=0.6):
    """
    Matches the input question with the closest available question using fuzzy matching.
    """
    closest_match = get_close_matches(input_question, available_questions, n=1, cutoff=cutoff)
    return closest_match[0] if closest_match else None

# Define schedule-related terms and broad terms
schedule_related_terms = ['schedule', 'teacher', 'instructor', 'time', 'room', 'subject', 'class']
broad_terms = ['CS', 'IT', 'WAM', 'SMP']

# Function to check if the query is related to schedule information
def is_schedule_related(query):
    return any(term in query.lower() for term in schedule_related_terms)

# Function to check if broad terms like "CS", "IT" are used without section
def contains_broad_terms(query):
    return any(broad_term.lower() in query.lower() for broad_term in broad_terms)

# Session state to track the context within a specific conversation
session_state = {
    'pending_query': None,  # Store the original incomplete query (e.g., asking about "CS" but without a section)
    'pending_section': None,  # Store the section if the user specifies it later
    'pending_day': None,  # Store the day extracted from the last query
    'pending_time_of_day': None  # Store the time of day extracted from the last query
}

# Handle follow-up queries and check if they resolve the pending query
def handle_follow_up(question):
    section_match = extract_section_from_query(question)

    if section_match:
        # Update the session state with the specified section
        session_state['pending_section'] = section_match
        return True  # Indicates that the follow-up has been processed

    return False  # No follow-up detected

# Clear session state after the query is answered
def clear_session_state():
    session_state['pending_query'] = None
    session_state['pending_section'] = None
    session_state['pending_day'] = None
    session_state['pending_time_of_day'] = None

# Load the list of bad words from a file
def load_bad_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        bad_words = [line.strip().lower() for line in f]
    return bad_words

# Define the path to the bad words file
BAD_WORDS_PATH = r"C:\THESIS PROJECT 2024\revised\WebKioskSystem\bad_words.txt"
bad_words = load_bad_words(BAD_WORDS_PATH)

# Function to detect bad words in the question
def contains_bad_words(question, bad_words):
    question = question.lower()  # Normalize text to lowercase
    for word in bad_words:
        # Use regex to match the bad word as a whole word only
        if re.search(rf'\b{re.escape(word)}\b', question):
            return True
    return False

# Function to load events from the event_management table in the database
def load_events():
    engine = connect_to_mysql()
    query = "SELECT title, start, end FROM event_management WHERE end >= CURDATE()"  # Only upcoming or ongoing events
    with engine.connect() as connection:
        events = pd.read_sql(query, connection)
    return events

# Function to extract keywords from the question for event title matching
def extract_keywords_from_question(question):
    # Remove common question words (like "when", "is", "the") and split into keywords
    words_to_ignore = ["when", "is", "the", "a", "an", "of", "in"]
    keywords = [word for word in re.split(r'\W+', question.lower()) if word and word not in words_to_ignore]
    return keywords

# Convert date to a readable format
def format_date(date_obj):
    # Directly format the datetime.date object
    return date_obj.strftime('%B %d, %Y')

# Main function to get the chatbot's answer (session-specific)
def get_answer(faq_data, conversation_log, question, intent_probabilities, feedback=None):
    print(f"Received question: {question}")  # Debugging line

    # Check for inappropriate language
    warning_message = "Please avoid using inappropriate language." if contains_bad_words(question, bad_words) else None

    # If a warning message is detected, return it immediately
    if warning_message:
        return warning_message, False  # False indicates no clarification needed

    # Normalize the input question for comparison
    normalized_question = normalize_text(question)
    print(f"Normalized Question: {normalized_question}")  # Debugging line

    requires_clarification = False  # Default state, assumes no clarification is needed

    # Direct handling for specific questions like "Who is the dean?" to avoid misinterpretation
    if "dean" in normalized_question:
        faculty_info = get_faculty_info("dean")
        if faculty_info:
            print("Direct response for 'who is the dean' detected.")
            return f"{warning_message} {faculty_info}" if warning_message else faculty_info, requires_clarification

    # Load the conversation logs from MySQL
    conversation_log_data = load_conversation_logs()

    # Handle follow-up queries
    print("Checking for follow-up queries...")  # Debugging line
    if handle_follow_up(normalized_question):
        print("Follow-up query detected, processing...")  # Debugging line
        # Process follow-up if a pending query exists for section, instructor, or other fields
        if session_state.get('pending_query'):
            day_of_week = session_state.get('pending_day')
            section = session_state.get('pending_section')
            instructor = session_state.get('pending_instructor')
            time_of_day = session_state.get('pending_time_of_day')
            room = session_state.get('pending_room')
            course_code = session_state.get('pending_course_code')
            modality = session_state.get('pending_modality')
            type_ = session_state.get('pending_type')

            # Determine the query type based on the question
            query_type = determine_query_type(normalized_question)

            # Retrieve schedule information
            schedule_info = get_schedule_info(
                section=section,
                instructor=instructor,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                room=room,
                course_code=course_code,
                modality=modality,
                type_=type_,
                query_type=query_type
            )
            clear_session_state()  # Clear session state after answering
            # Return the result, including a warning if necessary
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

    # Detect intent and handle schedule-related queries
    intent, updated_intent_probabilities = detect_event_intent(normalized_question, intent_probabilities, feedback=feedback)
    print(f"Detected intent: {intent}, Updated probabilities: {updated_intent_probabilities}")  # Debugging line

    if intent == 'class_schedule':
        # Extract relevant details from the question
        section_match = extract_section_from_query(normalized_question)  # Extract section
        instructor_match = extract_instructor_from_query(normalized_question)  # Extract instructor
        day_of_week = handle_time_specific_queries(normalized_question)  # Extract day
        time_of_day = determine_time_of_day(normalized_question)  # Extract time of day
        room_match = extract_room_from_query(normalized_question)  # Extract room
        course_code_match = extract_course_code_from_query(normalized_question)  # Extract course code
        modality_match = extract_modality_from_query(normalized_question)  # Extract modality
        type_match = extract_type_from_query(normalized_question)  # Extract type

        # If a section is provided, retrieve its schedule
        if section_match:
            query_type = determine_query_type(normalized_question)
            schedule_info = get_schedule_info(
                section=section_match,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                query_type=query_type,
                room=room_match,
                course_code=course_code_match,
                modality=modality_match,
                type_=type_match
            )
            print(f"Schedule info found for section '{section_match}': {schedule_info}")  # Debugging line
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

        # If an instructor is provided, retrieve their schedule
        if instructor_match:
            query_type = determine_query_type(normalized_question)
            schedule_info = get_schedule_info(
                instructor=instructor_match,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                query_type=query_type,
                room=room_match,
                course_code=course_code_match,
                modality=modality_match,
                type_=type_match
            )
            print(f"Schedule info found for instructor '{instructor_match}': {schedule_info}")  # Debugging line
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

        # If a room is provided, retrieve its schedule
        if room_match:
            query_type = determine_query_type(normalized_question)
            schedule_info = get_schedule_info(
                room=room_match,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                query_type=query_type,
                course_code=course_code_match,
                modality=modality_match,
                type_=type_match
            )
            print(f"Schedule info found for room '{room_match}': {schedule_info}")  # Debugging line
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

        # If a course code is provided, retrieve its schedule
        if course_code_match:
            query_type = determine_query_type(normalized_question)
            schedule_info = get_schedule_info(
                course_code=course_code_match,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                query_type=query_type,
                room=room_match,
                modality=modality_match,
                type_=type_match
            )
            print(f"Schedule info found for course code '{course_code_match}': {schedule_info}")  # Debugging line
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

        # Handle broad queries with no specific section, instructor, or field
        if contains_broad_terms(normalized_question) and not (section_match or instructor_match or room_match or course_code_match):
            requires_clarification = True
            clarification_prompt = "Could you please specify the section, instructor, room, course code, or any specific detail?"
            print("Broad query detected, requesting clarification.")  # Debugging line
            return clarification_prompt, requires_clarification

        # Handle modality- or type-specific queries
        if modality_match or type_match:
            query_type = determine_query_type(normalized_question)
            schedule_info = get_schedule_info(
                modality=modality_match,
                type_=type_match,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                query_type=query_type
            )
            print(f"Schedule info found for modality/type: {schedule_info}")  # Debugging line
            return f"{warning_message} {schedule_info}" if warning_message else schedule_info, requires_clarification

    # Handle events if no FAQ or log response is found
    events = load_events()  # Load events from the database
    if not events.empty:
        keywords = extract_keywords_from_question(normalized_question)  # Use NLP to extract relevant keywords
        
        # Fuzzy match event titles based on the query
        event_scores = []
        for _, event in events.iterrows():
            score = fuzz.partial_ratio(normalized_question.lower(), event['title'].lower())
            if score >= 90:  # Threshold for a "good" match
                event_scores.append((event, score))

        # Sort events by relevance score
        matched_events = sorted(event_scores, key=lambda x: x[1], reverse=True)

        # Generate response based on matched events
        if matched_events:
            response = "Here are the most relevant events matching your query:\n"
            for event, score in matched_events[:5]:  # Limit to top 5 matches for clarity
                response += f"• {event['title']} ({format_date(event['start'])} - {format_date(event['end'])})\n"
                if 'location' in event and not pd.isnull(event['location']):
                    response += f"  Location: {event['location']}\n"
                if 'organizer' in event and not pd.isnull(event['organizer']):
                    response += f"  Organizer: {event['organizer']}\n"
                if 'details' in event and not pd.isnull(event['details']):
                    response += f"  Details: {event['details']}\n"
            return response, requires_clarification

    # Query for events happening this month
    current_date = datetime.now()
    if "this month" in normalized_question:
        current_month = current_date.month
        current_year = current_date.year
        monthly_events = events[(events['start'].dt.month == current_month) & (events['start'].dt.year == current_year)]
        if not monthly_events.empty:
            response = "Here are the events happening this month:\n"
            for _, event in monthly_events.iterrows():
                response += f"• {event['title']} ({format_date(event['start'])} - {format_date(event['end'])})\n"
            return response, False
        
    # Default fallback if no match is found in any database
    combined_data = get_combined_data(faq_data, conversation_log)
    combined_data['normalized_question'] = combined_data['question'].apply(normalize_text)
    available_questions = combined_data['normalized_question'].tolist()
    closest_match = fuzzy_match_question(normalized_question, available_questions, cutoff=0.8)

    if closest_match:
        relevant_entries = combined_data[combined_data['normalized_question'] == closest_match]
        if not relevant_entries.empty:
            match_source = relevant_entries.iloc[0].get('source', 'FAQ')
            year_in_query = extract_year_from_query(normalized_question)
            if year_in_query and not relevant_entries.empty:
                relevant_entries = relevant_entries[
                    relevant_entries['answer'].str.contains(year_in_query, case=False)
                ]
            parsed_date_in_query = parse_natural_language_dates(normalized_question)
            if parsed_date_in_query and not relevant_entries.empty:
                relevant_entries = relevant_entries[
                    relevant_entries['answer'].str.contains(parsed_date_in_query, case=False)
                ]
            if not relevant_entries.empty:
                relevant_entries = relevant_entries.sort_values('posterior', ascending=False)
                best_match = relevant_entries.iloc[0]
                return (f"{warning_message} {best_match['answer']}" if warning_message else best_match['answer']), requires_clarification

    # If it's not schedule-related, or no schedule info is found, fallback to FAQ and logs
    # Use fuzzy matching to find the closest question in FAQ data
    available_questions = faq_data['question'].apply(normalize_text).tolist()
    closest_question = fuzzy_match_question(normalized_question, available_questions, cutoff=0.8)  # Adjusted cutoff for stricter matching

    if closest_question:
        # Search for matching question in FAQ and logs
        relevant_entries = faq_data[faq_data['question'] == closest_question]
        if not relevant_entries.empty:
            # If relevant entry found in FAQ
            answer = relevant_entries['answer'].values[0]
            print(f"Answer found in FAQ data: {answer}")  # Debugging line
            return f"{warning_message} {answer}" if warning_message else answer, requires_clarification
        else:
            # If no match found in FAQ, check conversation log
            relevant_log_entries = conversation_log_data[conversation_log_data['question'] == closest_question]
            if not relevant_log_entries.empty:
                # If relevant entry found in the conversation log
                answer = relevant_log_entries['answer'].values[0]
                print(f"Answer found in conversation log: {answer}")  # Debugging line
                return f"{warning_message} {answer}" if warning_message else answer, requires_clarification
            else:
                print(f"No relevant entries found for the query in FAQ or conversation log.")  # Debugging line
    else:
        print(f"No closest question match found in FAQ or logs.")  # Debugging line

    # Fallback to combined FAQ and log search if intent isn't clear or no match was found
    combined_data = get_combined_data(faq_data, conversation_log)
    print(f"Combined Data Size: {combined_data.shape}")  # Debugging line

    # Normalize and compare questions with fuzzy matching on combined data
    combined_data['normalized_question'] = combined_data['question'].apply(normalize_text)
    print(f"Sample of Normalized Questions in Combined Data: {combined_data['normalized_question'].head()}")  # Debugging line

    # Fuzzy match on normalized questions to account for typos, extra spaces, etc.
    available_questions = combined_data['normalized_question'].tolist()
    closest_match = fuzzy_match_question(normalized_question, available_questions, cutoff=0.8)  # Adjusted cutoff for stricter matching
    print(f"Closest Fuzzy Match Found: {closest_match}")  # Debugging line

    if closest_match:
        relevant_entries = combined_data[combined_data['normalized_question'] == closest_match]
        print(f"Relevant Entries Found with Fuzzy Matching: {len(relevant_entries)}")  # Debugging line

        if not relevant_entries.empty:
            match_source = relevant_entries.iloc[0].get('source', 'FAQ')
            print(f"Match Found in: {match_source}")  # Debugging line

            # If a year was mentioned in the query, filter by that year in the answer
            year_in_query = extract_year_from_query(normalized_question)
            if year_in_query and not relevant_entries.empty:
                relevant_entries = relevant_entries[relevant_entries['answer'].str.contains(year_in_query, case=False)]
                print(f"Relevant Entries After Year Filter: {len(relevant_entries)}")  # Debugging line

            if relevant_entries.empty and year_in_query:
                print("No exact year match found, searching without year filter.")  # Debugging line
                relevant_entries = combined_data[combined_data['normalized_question'].str.contains(normalized_question, case=False)]

            # If a natural language date was parsed, filter by that
            parsed_date_in_query = parse_natural_language_dates(normalized_question)
            if parsed_date_in_query and not relevant_entries.empty:
                relevant_entries = relevant_entries['answer'].str.contains(parsed_date_in_query, case=False)
                print(f"Relevant Entries After Date Parsing Filter: {len(relevant_entries)}")  # Debugging line

            # Check if any matches are found
            if not relevant_entries.empty:
                # Sort entries by posterior probability in descending order
                relevant_entries = relevant_entries.sort_values('posterior', ascending=False)

                # Select the top match after posterior update
                best_match = relevant_entries.iloc[0]
                print(f"Best Match Posterior: {best_match['posterior']}, Best Match Answer: {best_match['answer']}")  # Debugging line

                # Confidence thresholds
                high_confidence_threshold = 0.3
                uncertainty_threshold = 0.1

                # Ensure low-confidence answers are still returned
                if best_match['posterior'] < uncertainty_threshold:
                    print(f"Low Confidence Answer Chosen (Due to Posterior {best_match['posterior']}): {best_match['answer']}")  # Debugging line
                    return f"{warning_message} {best_match['answer']}" if warning_message else best_match['answer'], requires_clarification

                if best_match['posterior'] >= high_confidence_threshold:
                    chosen_answer = best_match['answer']
                    print(f"High Confidence Answer Chosen: {chosen_answer}")  # Debugging line
                    return f"{warning_message} {chosen_answer}" if warning_message else chosen_answer, requires_clarification

    print("No Matches Found, Returning Default Response")  # Debugging line
    default_response = "Sorry, I don't know the answer to that question. Can you please provide more information or rephrase?"
    return f"{warning_message} {default_response}" if warning_message else default_response, requires_clarification

# Flask route for chatbot response
@chatbot.route('/get_response', methods=['POST'])
def get_response():
    if flask_request.is_json:
        data = flask_request.get_json()
        user_message = data.get('message')
        
        # Log the received message to check
        print(f"Received message: {user_message}")
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get the chatbot answer and check if clarification is needed
        response, requires_clarification = get_answer(faq_data, conversation_log, user_message, intent_probabilities)
        print(f"Bot response: {response}")  # Log the bot's response for debugging

        # Check for inappropriate language warning
        warning_present = "Please avoid using inappropriate language" in response

        # Determine if feedback should be shown (no warning or clarification)
        feedback_needed = not requires_clarification and not warning_present

        # Determine if clarification should be asked (only for confident responses without warning)
        ask_clarification = requires_clarification and not warning_present

        return jsonify({
            'response': response, 
            'feedback_needed': feedback_needed,
            'requires_clarification': ask_clarification
        })
    else:
        return jsonify({'error': 'Invalid JSON input'}), 400

def async_recalibration(faq_data, conversation_log):
    thread = Thread(target=recalibrate_probabilities, args=(faq_data, conversation_log))
    thread.start()
    # Reload the recalibrated data after the thread finishes
    thread.join()  # Wait for recalibration to complete
    faq_data = pd.read_csv(DATASET_PATH)  # Reload updated data from CSV
    
# Flask route to handle feedback and log updates
@chatbot.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    global faq_data, conversation_log  # Declare both faq_data and conversation_log as global to persist changes

    data = flask_request.get_json()

    # Log the feedback for debugging
    print(f"Feedback received: {data}")

    # Extract feedback details with safe defaults
    question = str(data.get('question', ''))
    answer = str(data.get('answer', ''))
    feedback_rating = int(data.get('rating', 0))  # Defaults to 0 if missing
    user_correction = str(data.get('correction', ''))

    # Ensure a question is provided
    if not question:
        return jsonify({'error': 'Question is missing'}), 400

    print(f"Question: {question}, Feedback Rating: {feedback_rating}, Correction: {user_correction}")

    # Handle feedback based on rating
    if feedback_rating == 0:
        # Rating 0: Acknowledge feedback without making changes
        print("Rating is 0, no changes will be made.")
        return jsonify({'status': 'Feedback noted. No changes were made.'}), 200

    elif feedback_rating in [1, 2]:
        # Rating 1-2: Require a correction for low ratings
        if not user_correction:
            print("No correction provided for low rating.")
            return jsonify({'error': 'Please provide a correction for the wrong answer.'}), 400
        else:
            # Apply correction and update logs
            print(f"Applying correction for rating {feedback_rating}. Updating answer to: {user_correction}")
            faq_data, conversation_log = update_probabilities_and_log(faq_data, conversation_log, question, feedback_rating, user_correction=user_correction)
            return jsonify({'status': f'Thank you! The information has been updated with your correction: {user_correction}'}), 200

    elif feedback_rating in [3, 4]:
        # Rating 3-4: Log feedback without requiring correction
        print(f"Logging feedback for rating {feedback_rating}.")
        faq_data, conversation_log = update_probabilities_and_log(faq_data, conversation_log, question, feedback_rating)
        return jsonify({'status': 'Thank you for the feedback! Your input has been logged.'}), 200

    elif feedback_rating == 5:
        # Rating 5: Log feedback with high confidence
        print(f"Feedback received with highest confidence: {feedback_rating}")
        faq_data, conversation_log = update_probabilities_and_log(faq_data, conversation_log, question, feedback_rating)
        return jsonify({'status': 'Thank you! Your feedback was recorded with the highest confidence.'}), 200

    # Update probabilities in FAQ and log the conversation
    faq_data, conversation_log = update_probabilities_and_log(faq_data, conversation_log, question, feedback_rating, user_correction)

    # Recalibrate asynchronously without blocking the UI
    async_recalibration(faq_data, conversation_log)

    # Reload the updated data to ensure the chatbot uses the latest values
    faq_data.to_csv(DATASET_PATH, index=False)
    faq_data = pd.read_csv(DATASET_PATH)  # Reload dataset from the CSV

    # Default error handling
    return jsonify({'error': 'Invalid feedback rating'}), 400

# Start the Flask app
if __name__ == "__main__":
    chatbot.run(debug=True, host='0.0.0.0', port=5001)
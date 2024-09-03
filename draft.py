import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import mysql.connector
from datetime import datetime, timedelta
import json
import bcrypt
import pandas as pd
import plotly.express as px
import logging
import cv2
import numpy as np
import time
from PIL import Image
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import numpy as np
import time
import random
import smtplib
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from textblob import TextBlob
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from moviepy.editor import VideoFileClip
from speech_recognition import Recognizer, AudioFile
import streamlit as st
import cv2
import tempfile
import os
import pyaudio
import wave
import threading
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
from github import Github
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import streamlit as st
import streamlit_extras as stxa
import extra_streamlit_components as stx
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
from datetime import datetime, timedelta

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAjxGxSRwazR6jQJrctjeTgMwbiWhkFT7Y"


db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Chaithu@9515",
    database="glory"
)
cursor = db.cursor(dictionary=True)


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

class VideoProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        self.frames = []  
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        self.is_recording = True
        threading.Thread(target=self._record).start()

    def _record(self):
        while self.is_recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def reset(self):
        self.frames = []
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio = pyaudio.PyAudio() 

def initialize_session_state():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'frames' not in st.session_state:
        st.session_state.frames = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'video_interview_id' not in st.session_state:
        st.session_state.video_interview_id = None
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None


initialize_session_state()

def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        st.session_state.frames = []
        st.session_state.start_time = time.time()
        st.session_state.audio_recorder.start_recording()
    else:
        save_video()
        save_audio()

def save_video():
    if len(st.session_state.frames) > 0:
        video_dir = "video_interviews"
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"interview_{st.session_state.video_interview_id}.mp4")
        
        height, width, _ = st.session_state.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in st.session_state.frames:
            out.write(frame)
        out.release()
        
        cursor.execute("UPDATE video_interviews SET video_url = %s, status = 'completed' WHERE id = %s",
                       (video_path, st.session_state.video_interview_id))
        db.commit()
        
        st.success("Video interview completed and saved successfully!")
    else:
        st.error("No video data captured. Please try again.")

def save_audio_file(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def save_audio():
    if st.session_state.audio_recorder.frames:
        audio_dir = "audio_interviews"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"interview_{st.session_state.video_interview_id}.wav")
        save_audio_file(st.session_state.audio_recorder.frames, audio_path)
        st.session_state.audio_file = audio_path
        cursor.execute("UPDATE video_interviews SET audio_url = %s, status = 'completed' WHERE id = %s",
                        (audio_path, st.session_state.video_interview_id))
        db.commit()
        st.success("Audio saved successfully!")
    else:
        st.warning("No audio data captured. Please try recording again.")

def record_video_interview(video_interview_id):
    initialize_session_state()
    st.session_state.video_interview_id = video_interview_id

    st.subheader("Video Interview")
    
    try:
        cursor.execute("SELECT question FROM video_interviews WHERE id = %s", (video_interview_id,))
        question = cursor.fetchone()['question']
        
        st.write(f"Question: {question}")
        st.write("You have 2 minutes to record your response.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start/Stop Recording", on_click=toggle_recording):
                pass  # The actual toggling is done in the callback

        with col2:
            if st.button("Reset Interview"):
                st.session_state.recording = False
                st.session_state.frames = []
                st.session_state.start_time = None
                st.session_state.audio_file = None
                st.session_state.audio_recorder.reset()  # Reset the audio recorder
                st.rerun()

        status_placeholder = st.empty()
        video_placeholder = st.empty()

        if st.session_state.recording:
            status_placeholder.write("Recording in progress...")
            camera = cv2.VideoCapture(0)
            
            while time.time() - st.session_state.start_time < 120 and st.session_state.recording:  # 2 minutes
                ret, frame = camera.read()
                if ret:
                    st.session_state.frames.append(frame)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                remaining_time = int(120 - (time.time() - st.session_state.start_time))
                status_placeholder.write(f"Time remaining: {remaining_time} seconds")
                
               
                time.sleep(0.1)
            
            camera.release()
            
            if not st.session_state.recording: 
                save_video()
                save_audio()
            
            st.session_state.recording = False
        
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Video interview error: {str(e)}")

def create_video_interview_request(application_id, question):
    cursor.execute("""
    INSERT INTO video_interviews (application_id, question)
    VALUES (%s, %s)
    """, (application_id, question))
    db.commit()
    video_interview_id = cursor.lastrowid

   
    cursor.execute("SELECT email FROM applications WHERE id = %s", (application_id,))
    candidate_email = cursor.fetchone()['email']

 
    subject = "Video Interview Request"
    body = f"""
    You have been invited to complete a video interview for your job application.
    Please log in to the recruitment platform to record your response to the following question:

    {question}

    You will have 2 minutes to record your response.
    """
    send_notification(candidate_email, subject, body)

    return video_interview_id

def clear_unread_results():
    while cursor.nextset():
        pass

logging.basicConfig(level=logging.ERROR)

def generate_job_description(title, department):
    prompt = PromptTemplate(
        input_variables=["title", "department"],
        template="""
        Create a detailed job description for the position of {title} in the {department} department. Include:
        1. Brief overview of the role
        2. Key responsibilities
        3. Required qualifications
        4. Preferred qualifications
        5. Benefits and perks

        Format the output as a structured job description.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    job_description = chain.run(title=title, department=department)
    return job_description


def parse_resume(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_and_send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiration_time = time.time() + 300 
    sender_email = "vtu20026@veltech.edu.in"  
    sender_password = "Saikrishna@444" 

    message = MIMEMultipart("alternative")
    message["Subject"] = "Password Reset OTP"
    message["From"] = sender_email
    message["To"] = email

    text = f"Your OTP for password reset is: {otp}. This OTP is valid for 5 minutes."
    html = f"""\
    <html>
      <body>
        <p>Your OTP for password reset is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 5 minutes.</p>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    message.attach(part1)
    message.attach(part2)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        return otp, expiration_time
    except Exception as e:
        print(f"Error sending email: {e}")
        return None, None


def evaluate_resume(resume_text, job_description, required_qualifications, preferred_qualifications):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    extract_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template=(
            "Extract the following key information from the resume. If a piece of information is not present, write 'Not found'.\n"
            "1. Name\n2. Contact Information\n3. Education (list all degrees)\n4. Work Experience (list job titles and companies)\n"
            "5. Skills\n6. Certifications\n7. Projects\n8. Achievements\n\nResume:\n{resume_text}\n\n"
            "Provide the extracted information in a structured format."
        )
    )
    extract_chain = LLMChain(llm=model, prompt=extract_prompt)
    extracted_info = extract_chain.run(resume_text=resume_text)
    
    analysis_prompt = PromptTemplate(
        input_variables=["extracted_info", "job_description", "required_qualifications", "preferred_qualifications"],
        template=(
            "You are an expert resume evaluator. Analyze the following extracted resume information against the job requirements.\n\n"
            "Extracted Resume Information:\n{extracted_info}\n\n"
            "Job Description:\n{job_description}\n\n"
            "Required Qualifications:\n{required_qualifications}\n\n"
            "Preferred Qualifications:\n{preferred_qualifications}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. Match with Job Description (score out of 100 and explanation)\n"
            "2. Required Qualifications Met (list each, whether met, and explanation)\n"
            "3. Preferred Qualifications Met (list each, whether met, and explanation)\n"
            "4. Key Strengths relevant to the position\n"
            "5. Areas for Improvement or Missing Qualifications\n"
            "6. Relevant Projects or Achievements\n"
            "7. Overall Fit (score out of 100 and explanation)\n\n"
            "Ensure your analysis is thorough, impartial, and based solely on the provided information."
        )
    )
    analysis_chain = LLMChain(llm=model, prompt=analysis_prompt)
    analysis = analysis_chain.run(
        extracted_info=extracted_info,
        job_description=job_description,
        required_qualifications=required_qualifications,
        preferred_qualifications=preferred_qualifications
    )
   
    scoring_prompt = PromptTemplate(
        input_variables=["analysis"],
        template=(
            "Based on the following detailed analysis, provide a final evaluation and score for the candidate.\n\n"
            "Analysis:\n{analysis}\n\n"
            "Final Evaluation:\n"
            "1. Provide an overall score from 0 to 100.\n"
            "2. Summarize the candidate's fit for the position in 2-3 sentences.\n"
            "3. List the top 3 reasons to consider this candidate.\n"
            "4. List the top 3 concerns or areas for improvement.\n\n"
            "Format your response as follows:\n"
            "Score: [0-100]\n"
            "Summary: [Your summary]\n"
            "Top Reasons to Consider:\n1. [Reason 1]\n2. [Reason 2]\n3. [Reason 3]\n"
            "Areas of Concern:\n1. [Concern 1]\n2. [Concern 2]\n3. [Concern 3]"
        )
    )
    scoring_chain = LLMChain(llm=model, prompt=scoring_prompt)
    final_evaluation = scoring_chain.run(analysis=analysis)
    return final_evaluation

def get_score(evaluation_text):
    try:
        score_line = [line for line in evaluation_text.split('\n') if "Overall match score" in line][0]
        score = int(score_line.split(':')[1].strip().split('/')[0])
    except:
        score = 0
    return score


def add_category(category_type, category_name):
    cursor.execute("""
    INSERT INTO job_categories (category_type, category_name)
    VALUES (%s, %s)
    """, (category_type, category_name))
    db.commit()

def get_categories(category_type):
    cursor.execute("SELECT category_name FROM job_categories WHERE category_type = %s", (category_type,))
    return [row['category_name'] for row in cursor.fetchall()]

# Job Posting Management
def create_job_posting(title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, deadline):
    cursor.execute("""
    INSERT INTO job_postings (title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, deadline)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, deadline))
    db.commit()
    return cursor.lastrowid

# Application Form Management
def create_application_form(form_name, job_id, form_fields):
    cursor.execute("""
    INSERT INTO application_forms (form_name, job_id, form_fields)
    VALUES (%s, %s, %s)
    """, (form_name, job_id, json.dumps(form_fields)))
    db.commit()

def get_application_form(job_id):
    cursor.execute("SELECT * FROM application_forms WHERE job_id = %s", (job_id,))
    return cursor.fetchone()

# Application Management
def submit_application(job_id, applicant_name, email, phone, resume_file, form_responses):
    
    cursor.execute("SELECT COUNT(*) as count FROM applications WHERE job_id = %s AND email = %s", (job_id, email))
    result = cursor.fetchone()
    if result['count'] > 0:
        st.error("You have already applied for this job. You can only submit one application per job posting.")
        return

    
    resume_text = parse_resume(resume_file)
    resume_binary = resume_file.read()
    
    cursor.execute("""
    INSERT INTO applications (job_id, applicant_name, email, phone, resume_file, resume_text, form_responses, submission_date, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (job_id, applicant_name, email, phone, resume_binary, resume_text, json.dumps(form_responses), datetime.now(), "Applied"))
    db.commit()
    st.success("Application submitted successfully!")

def update_application_status(application_id, new_status):
    cursor.execute("UPDATE applications SET status = %s WHERE id = %s", (new_status, application_id))
    db.commit()

# Interview Scheduling
def schedule_interview(application_id, interview_date, interview_time, interviewers, interview_mode, interview_location=None):
    cursor.execute("""
    INSERT INTO interviews (application_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (application_id, interview_date, interview_time, json.dumps(interviewers), interview_mode, interview_location))
    db.commit()

def get_interviews(application_id):
    cursor.execute("SELECT * FROM interviews WHERE applicant_id = %s", (application_id,))
    return cursor.fetchall()

# Notes Management
def add_note(application_id, note_text):
    cursor.execute("INSERT INTO notes (application_id, note_text) VALUES (%s, %s)", (application_id, note_text))
    db.commit()

def get_notes(application_id):
    cursor.execute("SELECT * FROM notes WHERE application_id = %s ORDER BY created_at DESC", (application_id,))
    return cursor.fetchall()

# Selected Candidates Management
def add_to_selected_list(application_id):
    cursor.execute("UPDATE applications SET selected = TRUE WHERE id = %s", (application_id,))
    db.commit()

def remove_from_selected_list(application_id):
    cursor.execute("UPDATE applications SET selected = FALSE WHERE id = %s", (application_id,))
    db.commit()

def get_selected_candidates():
    cursor.execute("SELECT * FROM applications WHERE selected = TRUE")
    return cursor.fetchall()

# Notification System
def send_notification(recipient_email, subject, body):
    sender_email = "vtu20026@veltech.edu.in"
    password = "Saikrishna@444"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(message)

# User Management
def create_user(username, email, password, full_name, role):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute("""
    INSERT INTO users (username, email, password, full_name, role)
    VALUES (%s, %s, %s, %s, %s)
    """, (username, email, hashed_password, full_name, role))
    db.commit()

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return user
    return None

# Feedback Management

def add_feedback(interview_id, interviewer_id, feedback_text, rating):
    try:
        
        cursor.execute("SELECT applicant_id FROM interviews WHERE id = %s", (interview_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No interview found with id {interview_id}")
        application_id = result['applicant_id']

        
        cursor.execute("""
        INSERT INTO feedback (application_id, interviewer_id, feedback_text, rating)
        VALUES (%s, %s, %s, %s)
        """, (application_id, interviewer_id, feedback_text, rating))
        db.commit()

        
        cursor.execute("UPDATE applications SET status = 'In Review' WHERE id = %s", (application_id,))
        db.commit()

        
        cursor.execute("UPDATE interviews SET status = 'Completed' WHERE id = %s", (interview_id,))
        db.commit()

    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        st.error(f"An error occurred: {err}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred: {e}")
    
def get_feedback(application_id):
    cursor.execute("""
    SELECT f.*, u.full_name as interviewer_name
    FROM feedback f
    JOIN users u ON f.interviewer_id = u.id
    WHERE f.application_id = %s
    ORDER BY f.created_at DESC
    """, (application_id,))
    return cursor.fetchall()


def generate_recruitment_metrics():
    cursor.execute("""
    SELECT 
        COUNT(*) as total_applications,
        SUM(CASE WHEN status = 'Offered' THEN 1 ELSE 0 END) as offers_made,
        AVG(DATEDIFF(CURDATE(), submission_date)) as avg_time_to_hire
    FROM applications
    """)
    metrics = cursor.fetchone()
    
    cursor.execute("""
    SELECT department, COUNT(*) as application_count
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    GROUP BY department
    """)
    department_data = cursor.fetchall()
    
    return metrics, department_data


def ai_select_candidates(job_id, num_candidates):
    cursor.execute("""
    SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    WHERE a.job_id = %s AND a.status = 'Applied'
    """, (job_id,))
    applications = cursor.fetchall()
    
    selected_candidates = []
    for app in applications:
        evaluation = evaluate_resume(
            app['resume_text'], 
            app['description'], 
            app['required_qualifications'], 
            app['preferred_qualifications']
        )
        score = get_score(evaluation)
        selected_candidates.append((app['id'], score))
    
    selected_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = selected_candidates[:num_candidates]
    
    for candidate_id, score in top_candidates:
        update_application_status(candidate_id, "In Review")
        add_to_selected_list(candidate_id)
    
    return top_candidates

def save_job_draft(title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, created_by):
    cursor.execute("""
    INSERT INTO job_drafts (title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, created_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, created_by))
    db.commit()
    return cursor.lastrowid

def get_job_draft(draft_id):
    cursor.execute("SELECT * FROM job_drafts WHERE id = %s", (draft_id,))
    return cursor.fetchone()

def update_job_draft(draft_id, title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities):
    cursor.execute("""
    UPDATE job_drafts
    SET title = %s, description = %s, department = %s, location = %s, employment_type = %s, salary_range = %s, required_qualifications = %s, preferred_qualifications = %s, responsibilities = %s
    WHERE id = %s
    """, (title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, draft_id))
    db.commit()

def schedule_interview(application_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location=None):
    cursor.execute("""
INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
    db.commit()

def get_interviews():
    cursor.execute("""
    SELECT i.*, a.applicant_name, j.title as job_title
    FROM interviews i
    JOIN applications a ON i.applicant_id = a.id
    JOIN job_postings j ON i.job_id = j.id
    WHERE i.interview_date >= CURDATE()
    ORDER BY i.interview_date, i.interview_time
    """)
    return cursor.fetchall()

def get_interviewers():
    cursor.execute("SELECT id, full_name FROM users WHERE role IN ('Interviewer', 'HR Manager')")
    return cursor.fetchall()

def cancel_interview(interview_id):
    cursor.execute("DELETE FROM interviews WHERE id = %s", (interview_id,))
    db.commit()

def get_upcoming_interviews(interviewer_id):
    query = """
    SELECT i.*, a.applicant_name, j.title as job_title
    FROM interviews i
    JOIN applications a ON i.applicant_id = a.id
    JOIN job_postings j ON a.job_id = j.id
    WHERE JSON_CONTAINS(i.interviewers, %s)
    AND i.interview_date >= CURDATE()
    AND (i.status IS NULL OR i.status != 'Completed')
    ORDER BY i.interview_date, i.interview_time
    """
    cursor.execute(query, (json.dumps(interviewer_id),))
    return cursor.fetchall()

def clean_json_string(s):
    # Remove any leading/trailing non-JSON text
    json_pattern = r'\[.*\]'
    match = re.search(json_pattern, s, re.DOTALL)
    if match:
        return match.group()
    return s

def generate_ai_questions(job_id):
    cursor.execute("SELECT title, description, required_qualifications FROM job_postings WHERE id = %s", (job_id,))
    job_info = cursor.fetchone()
    
    prompt = PromptTemplate(
        input_variables=["title", "description", "qualifications"],
        template="""
        Generate 4 questions for a job application test based on the following job details:
        Job Title: {title}
        Job Description: {description}
        Required Qualifications: {qualifications}

        Generate:
        1. One basic coding question (provide a simple function signature and expected output)
        2. Two medium-level multiple-choice questions related to the job
        3. One open-ended HR question

        Format the output as a JSON array with each question as an object containing 'type', 'question', 'options' (for MCQs), and 'correct_answer' (except for the HR question).
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(title=job_info['title'], description=job_info['description'], qualifications=job_info['required_qualifications'])
    cleaned_result = clean_json_string(result)
    try:
        questions = json.loads(cleaned_result)
    except json.JSONDecodeError:
        st.error("Failed to generate questions in the correct format. Please try again.")
        st.text(result)  
        return
    
    for question in questions:
        cursor.execute("""
        INSERT INTO test_questions (job_id, question_type, question_text, options, correct_answer)
        VALUES (%s, %s, %s, %s, %s)
        """, (job_id, 'ai', question['question'], json.dumps(question.get('options')), question.get('correct_answer')))
    db.commit()
    st.success("AI questions generated successfully!")

def add_manual_question(job_id, question_text, question_type, options=None, correct_answer=None):
    cursor.execute("""
    INSERT INTO test_questions (job_id, question_type, question_text, options, correct_answer)
    VALUES (%s, %s, %s, %s, %s)
    """, (job_id, question_type, question_text, json.dumps(options), correct_answer))
    db.commit()

def get_test_questions(job_id):
    cursor.execute("SELECT * FROM test_questions WHERE job_id = %s", (job_id,))
    return cursor.fetchall()

def start_test_session(application_id):
    try:
        cursor.execute("""
        INSERT INTO test_sessions (application_id, start_time, status)
        VALUES (%s, %s, 'in_progress')
        """, (application_id, datetime.now()))
        db.commit()
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        db.rollback()
        return None
    finally:
        cursor.fetchall() 

def end_test_session(session_id):
    cursor.execute("""
    UPDATE test_sessions SET end_time = %s, status = 'completed'
    WHERE id = %s
    """, (datetime.now(), session_id))
    db.commit()

def save_candidate_response(application_id, question_id, answer):
    cursor.execute("""
    SELECT question_type FROM test_questions WHERE id = %s
    """, (question_id,))
    question_type = cursor.fetchone()['question_type']
    
    if question_type == 'Coding':
        
        cursor.execute("""
        INSERT INTO candidate_test_responses (application_id, question_id, candidate_answer, start_time)
        VALUES (%s, %s, %s, %s)
        """, (application_id, question_id, answer, datetime.now()))
    else:
        
        cursor.execute("""
        INSERT INTO candidate_test_responses (application_id, question_id, candidate_answer, start_time)
        VALUES (%s, %s, %s, %s)
        """, (application_id, question_id, answer, datetime.now()))
    db.commit()

def update_candidate_response(response_id, answer):
    cursor.execute("""
    UPDATE candidate_test_responses SET candidate_answer = %s, end_time = %s
    WHERE id = %s
    """, (answer, datetime.now(), response_id))
    db.commit()



def evaluate_candidate_answers(application_id):
    cursor.execute("""
    SELECT ctr.*, tq.question_text, tq.correct_answer, tq.question_type
    FROM candidate_test_responses ctr
    JOIN test_questions tq ON ctr.question_id = tq.id
    WHERE ctr.application_id = %s
    """, (application_id,))
    responses = cursor.fetchall()
    
    total_score = 0
    max_possible_score = len(responses) * 100
    explanations = []
    
    def extract_score_and_explanation(text):
        score_match = re.search(r'score"?\s*:\s*(\d+)', text, re.IGNORECASE)
        explanation_match = re.search(r'explanation"?\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        
        score = int(score_match.group(1)) if score_match else 0
        explanation = explanation_match.group(1) if explanation_match else "No explanation provided."
        
        return score, explanation

    for response in responses:
        if response['question_type'] == 'Coding':
            evaluation_prompt = PromptTemplate(
                input_variables=["question", "correct_answer", "candidate_answer"],
                template="""
                Coding Question: {question}
                Expected Output or Functionality: {correct_answer}
                Candidate's Code: {candidate_answer}
                
                Evaluate the candidate's code. Consider correctness, efficiency, and coding style.
                Provide a score between 0 and 100, where 100 is a perfect answer and 0 is completely incorrect.
                Also provide a brief explanation for the score.
                Your response should be in the format: score: <score>, explanation: "<explanation>"
                """
            )
        else:
            evaluation_prompt = PromptTemplate(
                input_variables=["question", "correct_answer", "candidate_answer"],
                template="""
                Question: {question}
                Correct Answer: {correct_answer}
                Candidate Answer: {candidate_answer}
                
                Evaluate the candidate's answer. Provide a score between 0 and 100, where 100 is a perfect answer and 0 is completely incorrect.
                Also provide a brief explanation for the score.
                Your response should be in the format: score: <score>, explanation: "<explanation>"
                """
            )
        
        evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)
        result = evaluation_chain.run(question=response['question_text'], correct_answer=response['correct_answer'], candidate_answer=response['candidate_answer'])
        
        score, explanation = extract_score_and_explanation(result)
        
        total_score += score
        explanations.append(f"Question {response['id']}: {explanation}")
        
        cursor.execute("UPDATE candidate_test_responses SET score = %s WHERE id = %s", (score, response['id']))
    
    average_score = (total_score / max_possible_score) * 100
    cursor.execute("UPDATE applications SET test_score = %s, test_status = 'evaluated' WHERE id = %s", (average_score, application_id))
    db.commit()
    
    return average_score, explanations


def ai_proctoring(frame, tab_switched):
    violations = []
    
    
    if tab_switched:
        violations.append("Tab switching detected")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        violations.append("No face detected")
    elif len(faces) > 1:
        violations.append("Multiple faces detected")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) < 2:
            violations.append("Eyes not clearly visible")
        
        if 'prev_face_center' not in ai_proctoring.cache:
            ai_proctoring.cache['prev_face_center'] = None
        
        face_center = (x + w//2, y + h//2)
        if ai_proctoring.cache['prev_face_center']:
            distance = np.sqrt((face_center[0] - ai_proctoring.cache['prev_face_center'][0])**2 + 
                               (face_center[1] - ai_proctoring.cache['prev_face_center'][1])**2)
            if distance > 50:  
                violations.append("Rapid head movement detected")
        ai_proctoring.cache['prev_face_center'] = face_center
    

    edges = cv2.Canny(frame, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000: 
            (x, y, w, h) = cv2.boundingRect(contour)
            if y < frame.shape[0] // 2: 
                violations.append("Suspicious object detected near face")
                break
    
    return len(violations) == 0, violations

ai_proctoring.cache = {}

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))

    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    ear = (A + B) / (2.0 * C)

    return ear

def analyze_video_interview(interview_id):
    try:
        cursor.execute("SELECT audio_url, question FROM video_interviews WHERE id = %s", (interview_id,))
        result = cursor.fetchone()
        if not result or not result['audio_url']:
            return 0, "Audio not found"
        
        audio_path = result['audio_url']
        question = result['question']
        
        if not os.path.exists(audio_path):
            return 0, "Audio file not found on the server"
        
        recognizer = Recognizer()
        with AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        try:
            transcript = recognizer.recognize_google(audio)
        except Exception as transcribe_error:
            logging.error(f"Error transcribing audio: {str(transcribe_error)}")
            return 0, "Failed to transcribe audio"
        
        
        analysis_result = analyze_transcript_with_llm(transcript, question)
        
        cursor.execute("UPDATE video_interviews SET score = %s, analysis_result = %s, status = 'analyzed' WHERE id = %s",
                       (analysis_result['overall_score'], analysis_result['analysis_text'], interview_id))
        db.commit()
        
        return analysis_result['overall_score'], analysis_result['analysis_text']
    
    except Exception as e:
        logging.error(f"Error in analyze_video_interview: {str(e)}")
        return 0, f"An error occurred during analysis: {str(e)}"

def analyze_transcript_with_llm(transcript, question):
    prompt = PromptTemplate(
        input_variables=["question", "transcript"],
        template="""Analyze the following interview transcript based on the given question. Provide scores (0-10) and detailed feedback for each of these aspects:

1. Sentiment Analysis: Evaluate the overall tone and emotion of the response.
2. Grammar: Assess the grammatical correctness of the response.
3. Fluency: Evaluate the smoothness and coherence of the speech.
4. Topic Relevance: Determine how well the response addresses the given question.
5. Knowledge Demonstration: Assess the depth and accuracy of knowledge shown in the response.

Question: {question}

Transcript: {transcript}

Please provide your analysis in the following format:
Sentiment Analysis:
Score: [0-10]
Feedback: [Your detailed feedback]

Grammar:
Score: [0-10]
Feedback: [Your detailed feedback]

Fluency:
Score: [0-10]
Feedback: [Your detailed feedback]

Topic Relevance:
Score: [0-10]
Feedback: [Your detailed feedback]

Knowledge Demonstration:
Score: [0-10]
Feedback: [Your detailed feedback]

Overall Score: [Average of all scores]

Summary: [A brief summary of the overall performance]"""
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(question=question, transcript=transcript)

    overall_score_match = re.search(r'Overall Score: ([\d.]+)', response)
    overall_score = float(overall_score_match.group(1)) if overall_score_match else 0

    return {
        'overall_score': overall_score,
        'analysis_text': response
    }

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_vector_db(data):
    texts = [str(item) for item in data]
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db

def query_vector_db(vector_db, query):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
    return qa_chain.run(query)

def historical_recruitment_analysis():
    st.title("Historical Recruitment Analysis Engine (Powered by Gemini Pro)")

    # Fetch historical data
    try:
        cursor.execute("""
            SELECT jp.title, jp.department, jp.location, a.applicant_name, a.status, a.test_status,
                   ts.status as test_session_status, vi.status as video_interview_status,
                   f.rating as interview_rating
            FROM job_postings jp
            JOIN applications a ON jp.id = a.job_id
            LEFT JOIN test_sessions ts ON a.id = ts.application_id
            LEFT JOIN video_interviews vi ON a.id = vi.application_id
            LEFT JOIN feedback f ON a.id = f.application_id
            ORDER BY a.submission_date DESC
            LIMIT 1000
        """)
        historical_data = cursor.fetchall()
    except mysql.connector.Error as err:
        st.error(f"Error fetching data: {err}")
        return

    if not historical_data:
        st.warning("No historical data found.")
        return

    vector_db = create_vector_db(historical_data)

    analysis_type = st.selectbox("Select Analysis Type", 
                                 ["Hiring Trends", 
                                  "Successful Candidate Profiles", 
                                  "Department Performance", 
                                  "Location-based Insights"])

    if analysis_type == "Hiring Trends":
        if st.button("Analyze Hiring Trends"):
            query = """Analyze the hiring trends in the recent applications. Identify patterns in job postings, 
            application rates, and successful hires. Consider the following aspects:
            1. Most common job titles and departments
            2. Application statuses (how many are in each stage)
            3. Test completion rates and performance
            4. Video interview completion rates
            5. Overall success rate (candidates who received high interview ratings)
            Provide insights on which roles were most frequently filled and any noticeable patterns."""
            insights = query_vector_db(vector_db, query)
            st.write(insights)

    elif analysis_type == "Successful Candidate Profiles":
        if st.button("Analyze Successful Candidates"):
            query = """Examine the profiles of candidates who were successfully hired or received high interview ratings. 
            Identify common characteristics among these candidates. Consider:
            1. Their performance in tests (test_status and test_session_status)
            2. Their performance in video interviews (video_interview_status)
            3. Their overall application status
            Provide insights on what makes a candidate successful in our hiring process."""
            insights = query_vector_db(vector_db, query)
            st.write(insights)

    elif analysis_type == "Department Performance":
        departments = list(set(item['department'] for item in historical_data if item['department']))
        if departments:
            department = st.selectbox("Select Department", departments)
            if st.button("Analyze Department Performance"):
                query = f"""Analyze the hiring performance of the {department} department. Consider:
                1. Number of job postings in this department
                2. Application rates for jobs in this department
                3. Test completion and success rates for this department's applications
                4. Video interview completion rates for this department
                5. Overall success rate (candidates who received high interview ratings) for this department
                Provide insights on how this department compares to others and any recommendations for improvement."""
                insights = query_vector_db(vector_db, query)
                st.write(insights)
        else:
            st.warning("No department data available.")

    elif analysis_type == "Location-based Insights":
        locations = list(set(item['location'] for item in historical_data if item['location']))
        if locations:
            location = st.selectbox("Select Location", locations)
            if st.button("Analyze Location-based Insights"):
                query = f"""Analyze the recruitment patterns for {location}. Consider:
                1. Types of roles most commonly filled in this location
                2. Application rates for jobs in this location
                3. Test completion and success rates for applications in this location
                4. Video interview completion rates for this location
                5. Overall success rate (candidates who received high interview ratings) for this location
                Examine any unique challenges or advantages for hiring in this location, and how it compares to other locations."""
                insights = query_vector_db(vector_db, query)
                st.write(insights)
        else:
            st.warning("No location data available.")

    if st.checkbox("Show Raw Historical Data"):
        st.write(pd.DataFrame(historical_data))


def github_profile_analysis():
    st.title("GitHub Profile Analysis")
    github_username = st.text_input("Enter GitHub Username")
    if st.button("Analyze GitHub Profile"):
        if github_username:
            try:
                user_url = f"https://api.github.com/users/{github_username}"
                user_response = requests.get(user_url)
                user_data = user_response.json()

                if user_response.status_code == 200:
                    repos_url = f"https://api.github.com/users/{github_username}/repos"
                    repos_response = requests.get(repos_url)
                    repos_data = repos_response.json()

                    if repos_response.status_code == 200:
                        profile_data = {
                            "User Info": user_data,
                            "Repositories": repos_data[:5] 
                        }
                        query = f"""Analyze the following GitHub profile and provide a detailed report:
                        1. Overview of the user's GitHub activity and profile
                        2. Detailed analysis of their top 5 projects, including:
                        - Project description
                        - Technologies used
                        - Project complexity
                        - Potential impact or usefulness
                        3. Overall skill assessment based on the projects
                        4. Areas of expertise
                        5. Suggestions for improvement
                        6. Rate the overall profile on a scale of 1-10, with justification

                        GitHub Profile Data:
                        {profile_data}
                        """

                        analysis = query_vector_db(create_vector_db([str(profile_data)]), query)

                        
                        st.subheader("GitHub Profile Analysis")
                        st.write(analysis)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a GitHub username.")

def main():
    st.set_page_config(page_title="Advanced Recruitment Aiding Software", page_icon="🎯", layout="wide")
    
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stExpander > div > div > div > button {
        background-color: #f3f4f6;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: 600;
        color: #1e3a8a;
    }
    .nav-link {
        color: #4b5563 !important;
        font-weight: 500;
    }
    .nav-link:hover {
        color: #1e3a8a !important;
    }
    .nav-link.active {
        color: #1e3a8a !important;
        font-weight: 700;
        background-color: #e5e7eb !important;
    }
    </style>
    """, unsafe_allow_html=True)
    def load_lottieurl(url: str):
        try:
            r = requests.get(url)
            r.raise_for_status() 
            return r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading Lottie URL: {e}")
        except json.JSONDecodeError:
            st.error("Error decoding JSON from Lottie URL")
        return None
    
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
    lottie_json = load_lottieurl(lottie_url)

    with st.sidebar:
        st.title("🎯 ARA Software")
        st_lottie(lottie_json, height=200, key="sidebar_animation")
        
        if 'user' in st.session_state and st.session_state.user:
            st.write(f"Welcome, {st.session_state.user['full_name']}!")
            
            if st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
                menu_items = {
                    "Job Management": ["Post Job", "Job Drafts", "Manage Categories"],
                    "Applications": ["Review Applications", "Applied Candidates", "In Review", "Selected Candidates"],
                    "Interviews": ["Interview Scheduling", "Upcoming Interviews", "Video Interviews"],
                    "Assessments": ["Evaluate Tests"],
                    "Offers": ["Offers"],
                    "Analytics": ["Analytics", "Historical Recruitment Analysis"]
                }
            elif st.session_state.user['role'] == 'Interviewer':
                menu_items = {
                    "Interviews": ["Upcoming Interviews", "Provide Feedback"],
                    "Analysis": ["GitHub Profile Analysis"]
                }
            else:  # Candidate
                menu_items = {
                    "Jobs": ["View Job Postings"],
                    "Applications": ["Submit Application", "Application Status"],
                    "Assessments": ["Take Test"],
                    "Interviews": ["Video Interviews"]
                }

            selected_menu = option_menu(
                menu_title="Navigation",
                options=list(menu_items.keys()),
                icons=["briefcase", "file-earmark-text", "camera-video", "clipboard-check", "envelope", "graph-up"],
                menu_icon="list",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#3b82f6", "font-size": "1rem"}, 
                    "nav-link": {"font-size": "0.9rem", "text-align": "left", "margin":"0px", "--hover-color": "#eef2ff"},
                    "nav-link-selected": {"background-color": "#e5e7eb"},
                }
            )

            selected_submenu = option_menu(
                menu_title=None,
                options=menu_items[selected_menu],
                icons=["chevron-right"] * len(menu_items[selected_menu]),
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#6b7280", "font-size": "0.8rem"}, 
                    "nav-link": {"font-size": "0.8rem", "text-align": "left", "margin":"0px", "--hover-color": "#eef2ff"},
                    "nav-link-selected": {"background-color": "#e5e7eb"},
                }
            )

            if st.button("Logout", key="logout_button"):
                st.session_state.user = None
                st.rerun()
        else:
            st.info("Please login to access the system.")

    # Main content area
    if 'user' not in st.session_state or st.session_state.user is None:
        st.title("Welcome to Advanced Recruitment Aiding Software")
        auth_option = stx.tab_bar(data=[
            stx.TabBarItemData(id="login", title="Login", description=""),
            stx.TabBarItemData(id="register", title="Register", description=""),
            stx.TabBarItemData(id="forgot", title="Forgot Password", description=""),
        ], default="login")

        if auth_option == "login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                if submit_button:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        elif auth_option == "register":
            with st.form("register_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                full_name = st.text_input("Full Name")
                role = st.selectbox("Role", ["Candidate", "Recruiter", "Interviewer", "HR Manager", "Admin"])
                submit_button = st.form_submit_button("Register")
                if submit_button:
                    create_user(username, email, password, full_name, role)
                    st.success("User registered successfully. Please login.")

        elif auth_option == "forgot":
            email = st.text_input("Enter your email")
            if st.button("Send OTP"):
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()
                if user:
                    otp, expiration_time = generate_and_send_otp(email)
                    if otp:
                        st.success("OTP sent to your email")
                        st.session_state.reset_email = email
                        st.session_state.reset_otp = otp
                        st.session_state.reset_otp_expiration = expiration_time
                    else:
                        st.error("Failed to send OTP. Please try again.")
                else:
                    st.error("Email not found")
                pass
    else:
        st.title(f"{selected_submenu}")

    if selected_submenu == "Post Job":
        st.subheader("Create Job Posting")
        
      
        if 'editing_draft' in st.session_state:
            draft = get_job_draft(st.session_state.editing_draft)
            del st.session_state.editing_draft
        else:
            draft = None

        title = st.text_input("Job Title", value=draft['title'] if draft else "")
        department = st.selectbox("Department", get_categories("Department"), index=get_categories("Department").index(draft['department']) if draft else 0)
        
        if st.button("Generate Job Description"):
            job_description = generate_job_description(title, department)
            st.text_area("Generated Job Description", job_description, height=300)
        
        description = st.text_area("Job Description (Edit if needed)", value=draft['description'] if draft else "", height=300)
        location = st.selectbox("Location", get_categories("Location"), index=get_categories("Location").index(draft['location']) if draft else 0)
        employment_type = st.selectbox("Employment Type", get_categories("Employment Type"), index=get_categories("Employment Type").index(draft['employment_type']) if draft else 0)
        salary_range = st.text_input("Salary Range", value=draft['salary_range'] if draft else "")
        required_qualifications = st.text_area("Required Qualifications", value=draft['required_qualifications'] if draft else "")
        preferred_qualifications = st.text_area("Preferred Qualifications", value=draft['preferred_qualifications'] if draft else "")
        responsibilities = st.text_area("Responsibilities", value=draft['responsibilities'] if draft else "")
        deadline = st.date_input("Application Deadline")
        deadline_time = st.time_input("Deadline Time")
        deadline_datetime = datetime.combine(deadline, deadline_time)
        col1, col2, col3 = st.columns(3)
        
        if col1.button("Save as Draft"):
            if draft:
                update_job_draft(draft['id'], title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities)
                st.success("Draft updated successfully!")
            else:
                save_job_draft(title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, st.session_state.user['id'])
                st.success("Draft saved successfully!")
        
        if col2.button("Post Job"):
            job_id = create_job_posting(title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, deadline_datetime)
            st.success(f"Job posted successfully! Job ID: {job_id}")
            if draft:
                cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft['id'],))
                db.commit()
        
        if draft and col3.button("Discard Changes"):
            st.rerun()

    elif selected_submenu== "Evaluate Tests":
        st.subheader("Evaluate Candidate Tests")
        
        if 'evaluated_tests' not in st.session_state:
            st.session_state.evaluated_tests = {}

        st.write("Debug: Entering Evaluate Tests section")

        cursor.execute("""
        SELECT a.id, a.applicant_name, j.title as job_title, a.test_score, a.test_status
        FROM applications a
        JOIN job_postings j ON a.job_id = j.id
        WHERE a.test_status = 'completed' AND (a.test_score IS NULL OR a.status = 'Applied')
        """)
        completed_tests = cursor.fetchall()
        
        st.write(f"Debug: Found {len(completed_tests)} tests to evaluate")

        for test in completed_tests:
            st.write(f"Candidate: {test['applicant_name']}, Job: {test['job_title']}")
            st.write(f"Current Test Score: {test['test_score']}, Status: {test['test_status']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if test['id'] not in st.session_state.evaluated_tests:
                    if st.button(f"Evaluate {test['applicant_name']}'s test", key=f"eval_{test['id']}"):
                        try:
                            with st.spinner("Evaluating test..."):
                                average_score, explanations = evaluate_candidate_answers(test['id'])
                                
                                # Store results in session state
                                st.session_state.evaluated_tests[test['id']] = {
                                    'score': average_score,
                                    'explanations': explanations
                                }
                                
                                # Store results in the database
                                cursor.execute("UPDATE applications SET test_score = %s WHERE id = %s", (average_score, test['id']))
                                db.commit()
                                
                                st.success(f"Evaluation completed. Score: {average_score:.2f}")
                                st.write(f"Debug: Evaluation completed for {test['applicant_name']}. Score: {average_score:.2f}")
                                
                        except Exception as e:
                            st.error(f"An error occurred during evaluation: {str(e)}")
                else:
                    st.write(f"Evaluation result: {st.session_state.evaluated_tests[test['id']]['score']:.2f}")
            
            with col2:
                if test['id'] in st.session_state.evaluated_tests or test['test_score'] is not None:
                    score = st.session_state.evaluated_tests[test['id']]['score'] if test['id'] in st.session_state.evaluated_tests else test['test_score']
                    cut_off = st.number_input("Set cut-off score", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key=f"cutoff_{test['id']}")
                    new_status = "Interview" if score >= cut_off else "Rejected"
                    
                    if st.button(f"Update Status", key=f"update_{test['id']}"):
                        cursor.execute("UPDATE applications SET status = %s, test_status = 'Evaluated', test_score = %s WHERE id = %s", (new_status, score, test['id']))
                        db.commit()
                        st.success(f"Status updated for {test['applicant_name']}. New status: {new_status}")
                        st.write(f"Debug: Status updated for {test['applicant_name']}. New status: {new_status}")
                        
                        # Clear the evaluation from session state
                        if test['id'] in st.session_state.evaluated_tests:
                            del st.session_state.evaluated_tests[test['id']]
                        
                        # # Force refresh
                        # st.rerun()
            
            st.write("---")

        st.write("Debug: Exiting Evaluate Tests section")

    elif selected_submenu == "Applied Candidates":
            st.subheader("Applied Candidates")
            
            # Select job
            cursor.execute("SELECT id, title FROM job_postings")
            jobs = cursor.fetchall()
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs])
            job_id = int(selected_job.split(":")[0])
            
            # Get applied candidates
            cursor.execute("""
            SELECT * FROM applications 
            WHERE job_id = %s AND status = 'Applied'
            """, (job_id,))
            candidates = cursor.fetchall()
            
            if candidates:
                st.write(f"Number of applied candidates: {len(candidates)}")
                
                # Choose between manual and AI-generated questions
                question_type = st.radio("Question Type", ["Manual", "AI-generated"])
                
                if question_type == "Manual":
                    st.subheader("Add Manual Questions")
                    question_text = st.text_area("Enter question")
                    question_type = st.selectbox("Question Type", ["Multiple Choice", "Coding", "Open-ended"])
                    options = st.text_input("Options (comma-separated, for MCQ only)")
                    correct_answer = st.text_input("Correct Answer (for MCQ and Coding)")
                    
                    if st.button("Add Question"):
                        add_manual_question(job_id, question_text, question_type, options.split(',') if options else None, correct_answer)
                        st.success("Question added successfully!")
                
                elif question_type == "AI-generated":
                    if st.button("Generate AI Questions"):
                        generate_ai_questions(job_id)
                        st.success("AI questions generated successfully!")
                
                # Set test deadline
                test_deadline_date = st.date_input("Set Test Deadline Date")
                test_deadline_time = st.time_input("Set Test Deadline Time")
                test_deadline = datetime.combine(test_deadline_date, test_deadline_time)
                if st.button("Set Deadline and Notify Candidates"):
                    for candidate in candidates:
                        # Update application status and send notification
                        cursor.execute("UPDATE applications SET test_status = 'not_started' WHERE id = %s", (candidate['id'],))
                        send_notification(candidate['email'], "Test Invitation", f"You have been invited to take a test for the {selected_job.split(':')[1].strip()} position. Please complete the test by {test_deadline}.")
                    db.commit()
                    st.success("Candidates notified and test deadline set!")
            
            else:
                st.write("No applied candidates found for this job.")


    elif selected_submenu == "Take Test":
        if st.session_state.user['role'] == 'Candidate':
            st.subheader("Candidate Test")

            cursor.execute("""
            SELECT a.id, j.title
            FROM applications a
            JOIN job_postings j ON a.job_id = j.id
            WHERE a.email = %s AND a.test_status = 'not_started'
            """, (st.session_state.user['email'],))
            available_tests = cursor.fetchall()

            if available_tests:
                selected_test = st.selectbox("Select Test to Take", [f"{test['id']}: {test['title']}" for test in available_tests])
                application_id = int(selected_test.split(":")[0])

                if st.button("Start Test"):
                    session_id = start_test_session(application_id)
                    if session_id:
                        st.session_state.current_test = session_id
                        st.session_state.test_start_time = datetime.now()
                        st.rerun()
                    else:
                        st.error("Failed to start test. Please try again later.")

                if 'current_test' in st.session_state:
                    test_id = st.session_state.current_test
                    cursor.execute("""
                    SELECT tq.id, tq.question_text, tq.options, tq.question_type
                    FROM test_questions tq
                    JOIN applications a ON tq.job_id = a.job_id
                    WHERE a.id = %s
                    """, (application_id,))
                    questions = cursor.fetchall()
                    
                    for question in questions:
                        st.write(question['question_text'])
                        
                        if question['question_type'] == 'Coding':
                            answer = st.text_area("Your Code:", key=f"question_{question['id']}_code", height=300)
                            st.write("Example solution:")
                            st.code("""
                    def sum_even_numbers(arr):
                        return sum(num for num in arr if num % 2 == 0)
                            """, language="python")
                        elif question['options']:
                            try:
                                options = json.loads(question['options'])
                                if options:
                                    answer = st.radio("Select your answer:", options, key=f"question_{question['id']}_radio")
                                else:
                                    # If no options are available, provide a text area instead
                                    answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                            except json.JSONDecodeError:
                                st.write("Error in question options format.")
                                # Provide a text area in case of error
                                answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                        else:
                            # For any other type of question, always provide a text area
                            answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                        
                        if answer is not None:
                            save_candidate_response(application_id, question['id'], answer)
                    
                    time_elapsed = datetime.now() - st.session_state.test_start_time
                    st.write(f"Time elapsed: {time_elapsed.total_seconds() // 60} minutes")
                    
                    if st.button("Submit Test") or time_elapsed > timedelta(minutes=30):
                        end_test_session(test_id)
                        cursor.execute("UPDATE applications SET test_status = 'completed' WHERE id = %s", (application_id,))
                        db.commit()
                        del st.session_state.current_test
                        del st.session_state.test_start_time
                        st.success("Test submitted successfully!")
                        st.rerun()
            else:
                st.write("No tests available at the moment.")
        else:
            st.write("You don't have permission to access this section.")

    elif selected_submenu == "Interview Scheduling":
        st.subheader("Interview Scheduling Dashboard")

        if 'interview_action_performed' in st.session_state and st.session_state.interview_action_performed:
            del st.session_state.interview_action_performed
            st.rerun()

        def has_scheduled_interview(application_id):
            cursor.execute("""
            SELECT COUNT(*) as count
            FROM interviews
            WHERE applicant_id = %s AND interview_date >= CURDATE() AND status != 'Completed'
            """, (application_id,))
            result = cursor.fetchone()
            return result['count'] > 0

        def get_interviews():
            cursor.execute("""
            SELECT i.*, a.applicant_name, j.title as job_title
            FROM interviews i
            JOIN applications a ON i.applicant_id = a.id
            JOIN job_postings j ON i.job_id = j.id
            WHERE i.interview_date >= CURDATE() AND i.status != 'Completed'
            ORDER BY i.interview_date, i.interview_time
            """)
            return cursor.fetchall()

        # Upcoming Interviews Summary
        upcoming_interviews = get_interviews()
        st.metric("Upcoming Interviews", len(upcoming_interviews))

        # Interviews List Table
        st.subheader("Scheduled Interviews")
        for interview in upcoming_interviews:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 3])
            col1.write(interview['applicant_name'])
            col2.write(interview['job_title'])
            col3.write(interview['interview_date'].strftime("%Y-%m-%d"))
            
            # Convert timedelta to string representation
            interview_time = (datetime.min + interview['interview_time']).time()
            col4.write(interview_time.strftime("%H:%M"))
            
            if col5.button("View", key=f"view_{interview['id']}"):
                st.session_state.viewing_interview = interview['id']
            if col5.button("Reschedule", key=f"reschedule_{interview['id']}"):
                st.session_state.rescheduling_interview = interview['id']
            if col5.button("Cancel", key=f"cancel_{interview['id']}"):
                cancel_interview(interview['id'])
                st.success("Interview cancelled successfully!")
                st.session_state.interview_action_performed = True
                st.rerun()

        if st.button("Schedule New Interview"):
            st.session_state.scheduling_new_interview = True

        # Interview Scheduling Screen
        if 'scheduling_new_interview' in st.session_state and st.session_state.scheduling_new_interview:
            st.subheader("Schedule New Interview")
            
            cursor.execute("""
            SELECT DISTINCT j.id, j.title
            FROM job_postings j
            JOIN applications a ON j.id = a.job_id
            WHERE a.status = 'Interview'
            """)
            jobs_with_interviews = cursor.fetchall()
            if jobs_with_interviews:
                selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs_with_interviews])
                job_id = int(selected_job.split(":")[0])

                # Fetch candidates for the selected job who are in the 'Interview' stage
                cursor.execute("""
                SELECT id, applicant_name
                FROM applications
                WHERE job_id = %s AND status = 'Interview'
                """, (job_id,))
                interview_candidates = cursor.fetchall()

                if interview_candidates:
                    selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in interview_candidates])
                    application_id = int(selected_candidate.split(":")[0])

                    interview_date = st.date_input("Interview Date")
                    interview_time = st.time_input("Interview Time")

                    # Fetch users with interviewer roles
                    cursor.execute("""
                    SELECT id, full_name 
                    FROM users 
                    WHERE role IN ('Interviewer', 'HR Manager')
                    """)
                    interviewers = cursor.fetchall()

                    selected_interviewers = st.multiselect(
                        "Select Interviewers", 
                        options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                        format_func=lambda x: x.split(": ")[1]
                    )

                    interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                    interview_location = st.text_input("Interview Location (if applicable)")

                    if st.button("Schedule Interview"):
                        if has_scheduled_interview(application_id):
                            st.error("An interview is already scheduled for this candidate.")
                        else:
                            # Extract interviewer IDs
                            interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]

                            # Schedule the interview
                            cursor.execute("""
                            INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location, status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, 'Scheduled')
                            """, (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                            db.commit()

                            st.success("Interview scheduled successfully!")
                            st.session_state.scheduling_new_interview = False
                            st.session_state.interview_action_performed = True
                            st.rerun()
                else:
                    st.write("No candidates are currently in the 'Interview' stage for this job.")
            else:
                st.write("No jobs currently have candidates in the 'Interview' stage.")

        # Interview Details Screen
        if 'viewing_interview' in st.session_state:
            interview_id = st.session_state.viewing_interview
            cursor.execute("""
            SELECT i.*, a.applicant_name, a.email, a.phone, j.title as job_title, j.department, j.location
            FROM interviews i
            JOIN applications a ON i.applicant_id = a.id
            JOIN job_postings j ON i.job_id = j.id
            WHERE i.id = %s
            """, (interview_id,))
            interview = cursor.fetchone()

            st.subheader("Interview Details")
            st.write(f"Applicant Name: {interview['applicant_name']}")
            st.write(f"Email: {interview['email']}")
            st.write(f"Phone: {interview['phone']}")
            st.write(f"Job Title: {interview['job_title']}")
            st.write(f"Department: {interview['department']}")
            st.write(f"Location: {interview['location']}")
            st.write(f"Interview Date: {interview['interview_date']}")
            
            # Convert timedelta to string representation
            interview_time = (datetime.min + interview['interview_time']).time()
            st.write(f"Interview Time: {interview_time.strftime('%H:%M')}")
            
            st.write(f"Interview Mode: {interview['interview_mode']}")
            if interview['interview_location']:
                st.write(f"Interview Location: {interview['interview_location']}")
            
            # Fetch interviewer names
            interviewer_ids = json.loads(interview['interviewers'])
            cursor.execute("SELECT full_name FROM users WHERE id IN %s", (tuple(interviewer_ids),))
            interviewer_names = [row['full_name'] for row in cursor.fetchall()]
            st.write(f"Interviewers: {', '.join(interviewer_names)}")

            if st.button("Close"):
                del st.session_state.viewing_interview

        # Reschedule Interview Screen
        if 'rescheduling_interview' in st.session_state:
            interview_id = st.session_state.rescheduling_interview
            st.subheader("Reschedule Interview")

            cursor.execute("SELECT applicant_id, job_id FROM interviews WHERE id = %s", (interview_id,))
            current_interview = cursor.fetchone()

            new_date = st.date_input("New Interview Date", min_value=datetime.now().date())
            new_time = st.time_input("New Interview Time")

            if st.button("Reschedule Interview"):
                # Check for conflicts
                cursor.execute("""
                SELECT COUNT(*) as count
                FROM interviews
                WHERE applicant_id = %s AND job_id = %s AND id != %s
                AND interview_date = %s AND interview_time = %s
                """, (current_interview['applicant_id'], current_interview['job_id'], interview_id, new_date, new_time))
                conflict = cursor.fetchone()['count'] > 0

                if conflict:
                    st.error("There is already an interview scheduled at this time for this candidate and job.")
                else:
                    cursor.execute("UPDATE interviews SET interview_date = %s, interview_time = %s WHERE id = %s",
                                (new_date, new_time, interview_id))
                    db.commit()
                    st.success("Interview rescheduled successfully!")
                    del st.session_state.rescheduling_interview
                    st.session_state.interview_action_performed = True
                    st.rerun()
                
    elif selected_submenu == "Manage Categories":
        st.subheader("Manage Job Categories")
        
        category_type = st.selectbox("Category Type", ["Department", "Location", "Employment Type"])
        category_name = st.text_input("Category Name")
        
        if st.button("Add Category"):
            add_category(category_type, category_name)
            st.success(f"Added {category_name} to {category_type} categories")

    elif selected_submenu == "Review Applications":
        st.subheader("Review Applications")
        
        cursor.execute("SELECT id, title FROM job_postings")
        jobs = cursor.fetchall()
        selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs], key="review_applications_job_select")
        job_id = int(selected_job.split(":")[0])

        cursor.execute("""
SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
FROM applications a
JOIN job_postings j ON a.job_id = j.id
WHERE a.job_id = %s AND a.status != 'Rejected'
""", (job_id,))
        applications = cursor.fetchall()

        for app in applications:
            st.write(f"Applicant: {app['applicant_name']}")
            st.write(f"Email: {app['email']}")
            st.write(f"Status: {app['status']}")
            
            if st.button(f"Analyze Resume - {app['applicant_name']}"):
                with st.spinner("Analyzing resume..."):
                    evaluation = evaluate_resume(
                        app['resume_text'],
                        app['description'],
                        app['required_qualifications'],
                        app['preferred_qualifications']
                    )
                    score = get_score(evaluation)
                    st.write(f"Match Score: {score}")
                    st.text_area("Resume Evaluation", evaluation, height=300)
            
            new_status = st.selectbox(f"Update Status for {app['applicant_name']}", 
                                    ["Applied", "In Review", "Interview", "Offered", "Rejected"],
                                    index=["Applied", "In Review", "Interview", "Offered", "Rejected"].index(app['status']))
            
            if new_status != app['status']:
                if st.button(f"Update Status for {app['applicant_name']}"):
                    update_application_status(app['id'], new_status)
                    st.success(f"Status updated for {app['applicant_name']}")
                    
                    if new_status == "Interview":
                        # Schedule interview
                        st.subheader(f"Schedule Interview for {app['applicant_name']}")
                        
                        interview_date = st.date_input("Interview Date")
                        interview_time = st.time_input("Interview Time")
                        
                        # Fetch users with interviewer roles for this specific job
                        cursor.execute("""
                        SELECT DISTINCT u.id, u.full_name 
                        FROM users u
                        WHERE u.role IN ('Interviewer', 'HR Manager')
                        """, (job_id,))
                        interviewers = cursor.fetchall()
                        
                        if interviewers:
                            selected_interviewers = st.multiselect(
                                "Select Interviewers", 
                                options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                                format_func=lambda x: x.split(": ")[1]
                            )
                            
                            interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                            interview_location = st.text_input("Interview Location (if applicable)")

                            if st.button("Schedule Interview"):
                                # Extract interviewer IDs
                                interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]
                                
                                # Schedule the interview
                                cursor.execute("""
                                INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """, (app['id'], job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                                db.commit()
                                
                                st.success("Interview scheduled successfully!")
                        else:
                            st.error("No interviewers assigned to this job. Please assign interviewers first.")


            if st.button(f"Select {app['applicant_name']}"):
                add_to_selected_list(app['id'])
                send_notification(app['email'], "Application Update", "Congratulations! You have been selected for further consideration.")
                st.success(f"{app['applicant_name']} added to the selected list.")

            st.write("---")


# Add this new elif block for the "Schedule Interview" option
    elif selected_submenu == "Schedule Interview":
        st.subheader("Schedule Interview")

        # Fetch jobs with candidates in the 'Interview' stage
        cursor.execute("""
        SELECT DISTINCT j.id, j.title
        FROM job_postings j
        JOIN applications a ON j.id = a.job_id
        WHERE a.status = 'Interview'
        """)
        jobs_with_interviews = cursor.fetchall()

        if jobs_with_interviews:
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs_with_interviews])
            job_id = int(selected_job.split(":")[0])

            # Fetch candidates for the selected job who are in the 'Interview' stage
            cursor.execute("""
            SELECT id, applicant_name
            FROM applications
            WHERE job_id = %s AND status = 'Interview'
            """, (job_id,))
            interview_candidates = cursor.fetchall()

            if interview_candidates:
                selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in interview_candidates])
                application_id = int(selected_candidate.split(":")[0])

                interview_date = st.date_input("Interview Date")
                interview_time = st.time_input("Interview Time")

                # Fetch users with interviewer roles
                cursor.execute("""
                SELECT id, full_name 
                FROM users 
                WHERE role IN ('Interviewer', 'HR Manager')
                """)
                interviewers = cursor.fetchall()

                selected_interviewers = st.multiselect(
                    "Select Interviewers", 
                    options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                    format_func=lambda x: x.split(": ")[1]
                )

                interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                interview_location = st.text_input("Interview Location (if applicable)")

                if st.button("Schedule Interview"):
                    # Extract interviewer IDs
                    interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]

                    # Schedule the interview
                    cursor.execute("""
                    INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                    db.commit()

                    st.success("Interview scheduled successfully!")
            else:
                st.write("No candidates are currently in the 'Interview' stage for this job.")
        else:
            st.write("No jobs currently have candidates in the 'Interview' stage.")

    elif selected_submenu == "Selected Candidates":
        st.subheader("Selected Candidates")
        selected_candidates = get_selected_candidates()
        for candidate in selected_candidates:
            st.write(f"Name: {candidate['applicant_name']}")
            st.write(f"Email: {candidate['email']}")
            st.write(f"Status: {candidate['status']}")
            st.write("---")

    elif selected_submenu == "Analytics":
        st.subheader("Recruitment Analytics Dashboard")
        
        # Job selection
        cursor.execute("SELECT id, title FROM job_postings")
        jobs = cursor.fetchall()
        selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs], key="analytics_job_select")
        job_id = int(selected_job.split(":")[0])

        # Fetch job-specific metrics
        cursor.execute("""
        SELECT 
            COUNT(*) as total_applications,
            SUM(CASE WHEN status = 'Offered' THEN 1 ELSE 0 END) as offers_made,
            AVG(DATEDIFF(CURDATE(), submission_date)) as avg_time_to_hire,
            SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END) as rejections,
            SUM(CASE WHEN status = 'Interview' THEN 1 ELSE 0 END) as interviews_scheduled
        FROM applications
        WHERE job_id = %s
        """, (job_id,))
        metrics = cursor.fetchone()

        # Display key metrics in a more visually appealing way
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Applications", int(metrics['total_applications']) if metrics['total_applications'] is not None else 0)
        col2.metric("Offers Made", int(metrics['offers_made']) if metrics['offers_made'] is not None else 0)
        col3.metric("Avg. Time to Hire (days)", round(float(metrics['avg_time_to_hire']), 1) if metrics['avg_time_to_hire'] is not None else 0)
        col4.metric("Rejections", int(metrics['rejections']) if metrics['rejections'] is not None else 0)
        col5.metric("Interviews Scheduled", int(metrics['interviews_scheduled']) if metrics['interviews_scheduled'] is not None else 0)

        # Application status breakdown
        st.subheader("Application Status Breakdown")
        cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM applications
        WHERE job_id = %s
        GROUP BY status
        """, (job_id,))
        status_data = cursor.fetchall()
        
        df = pd.DataFrame(status_data)
        fig = px.pie(df, values='count', names='status', title='', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Application timeline
        st.subheader("Application Timeline")
        cursor.execute("""
        SELECT DATE(submission_date) as date, COUNT(*) as count
        FROM applications
        WHERE job_id = %s
        GROUP BY DATE(submission_date)
        ORDER BY date
        """, (job_id,))
        timeline_data = cursor.fetchall()
        
        df_timeline = pd.DataFrame(timeline_data)
        fig_timeline = px.line(df_timeline, x='date', y='count', title='Daily Application Count')
        fig_timeline.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader("Candidate Selection")
        selection_method = st.radio("Selection Method", ["Manual", "AI-based"])

        cursor.execute("SELECT id, title FROM job_postings")
        jobs = cursor.fetchall()
        selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs])
        job_id = int(selected_job.split(":")[0])

        if selection_method == "Manual":
            cursor.execute("""
            SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
            FROM applications a
            JOIN job_postings j ON a.job_id = j.id
            WHERE a.job_id = %s AND a.status = 'Applied'
            """, (job_id,))
            applications = cursor.fetchall()

            for app in applications:
                st.write(f"Applicant: {app['applicant_name']}")
                st.write(f"Email: {app['email']}")
                
                selected = st.checkbox(f"Select {app['applicant_name']}", value=app['selected'])
                if selected != app['selected']:
                    if selected:
                        add_to_selected_list(app['id'])
                        update_application_status(app['id'], "In Review")
                        st.success(f"{app['applicant_name']} added to the selected list.")
                    else:
                        remove_from_selected_list(app['id'])
                        update_application_status(app['id'], "Applied")
                        st.success(f"{app['applicant_name']} removed from the selected list.")

                # Add button for resume analysis
                if st.button(f"Analyze Resume - {app['applicant_name']}"):
                    with st.spinner("Analyzing resume..."):
                        evaluation = evaluate_resume(
                            app['resume_text'],
                            app['description'],
                            app['required_qualifications'],
                            app['preferred_qualifications']
                        )
                        score = get_score(evaluation)
                        st.write(f"Match Score: {score}")
                        st.text_area("Resume Evaluation", evaluation, height=300)

                st.write("---")

        elif selection_method == "AI-based":
            num_candidates = st.number_input("Number of candidates to select", min_value=1, max_value=20, value=5)
            if st.button("Run AI Selection"):
                selected_candidates = ai_select_candidates(job_id, num_candidates)
                st.success(f"Selected {len(selected_candidates)} candidates based on AI evaluation.")
                for candidate_id, score in selected_candidates:
                    cursor.execute("SELECT applicant_name, email FROM applications WHERE id = %s", (candidate_id,))
                    candidate = cursor.fetchone()
                    st.write(f"Selected: {candidate['applicant_name']} (Score: {score})")
                    st.write(f"Email: {candidate['email']}")
                    st.write("---")

    elif selected_submenu == "Upcoming Interviews":
        st.subheader("Upcoming Interviews")
        interviewer_id = st.session_state.user['id']
        interviews = get_upcoming_interviews(interviewer_id)

        if interviews:
            for interview in interviews:
                with st.expander(f"{interview['applicant_name']} - {interview['job_title']} ({interview['interview_date']})"):
                    st.write(f"Date: {interview['interview_date']}")
                    st.write(f"Time: {interview['interview_time']}")
                    st.write(f"Mode: {interview['interview_mode']}")
                    if interview['interview_location']:
                        st.write(f"Location: {interview['interview_location']}")
                    
                    # Add feedback form
                    feedback = st.text_area("Interview Feedback", key=f"feedback_{interview['id']}")
                    rating = st.slider("Rating", 1, 5, 3, key=f"rating_{interview['id']}")
                    if st.button("Submit Feedback", key=f"submit_{interview['id']}"):
                        add_feedback(interview['id'], interviewer_id, feedback, rating)
                        st.success("Feedback submitted successfully!")
                        st.rerun()
        else:
            st.write("No upcoming interviews found.")

    elif selected_submenu == "Job Drafts":
        st.subheader("Job Drafts")
        
        cursor.execute("""
        SELECT id, title, department, created_at
        FROM job_drafts
        WHERE created_by = %s
        ORDER BY created_at DESC
        """, (st.session_state.user['id'],))
        drafts = cursor.fetchall()

        if drafts:
            for draft in drafts:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"{draft['title']} - {draft['department']}")
                col2.write(draft['created_at'].strftime("%Y-%m-%d"))
                
                if col3.button("Edit", key=f"edit_{draft['id']}"):
                    st.session_state.editing_draft = draft['id']
                    st.rerun()
                
                if col3.button("Delete", key=f"delete_{draft['id']}"):
                    cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft['id'],))
                    db.commit()
                    st.success("Draft deleted successfully!")
                    st.rerun()
        else:
            st.write("No job drafts found.")

        if 'editing_draft' in st.session_state:
            draft_id = st.session_state.editing_draft
            cursor.execute("SELECT * FROM job_drafts WHERE id = %s", (draft_id,))
            draft = cursor.fetchone()
            
            # Pre-fill the form with draft data
            title = st.text_input("Job Title", value=draft['title'])
            description = st.text_area("Job Description", value=draft['description'])
            department = st.selectbox("Department", get_categories("Department"), index=get_categories("Department").index(draft['department']))
            location = st.selectbox("Location", get_categories("Location"), index=get_categories("Location").index(draft['location']))
            employment_type = st.selectbox("Employment Type", get_categories("Employment Type"), index=get_categories("Employment Type").index(draft['employment_type']))
            salary_range = st.text_input("Salary Range", value=draft['salary_range'])
            required_qualifications = st.text_area("Required Qualifications", value=draft['required_qualifications'])
            preferred_qualifications = st.text_area("Preferred Qualifications", value=draft['preferred_qualifications'])
            responsibilities = st.text_area("Responsibilities", value=draft['responsibilities'])

            if st.button("Update Draft"):
                update_job_draft(draft_id, title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities)
                st.success("Draft updated successfully!")
                del st.session_state.editing_draft
                st.rerun()

            if st.button("Post Job"):
                deadline = st.date_input("Application Deadline")
                deadline_time = st.time_input("Deadline Time")
                deadline_datetime = datetime.combine(deadline, deadline_time)
                job_id = create_job_posting(title, description, department, location, employment_type, salary_range, required_qualifications, preferred_qualifications, responsibilities, deadline_datetime)
                st.success(f"Job posted successfully! Job ID: {job_id}")
                cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft_id,))
                db.commit()
                del st.session_state.editing_draft
                st.rerun()

    elif selected_submenu == "Offers":
        st.subheader("Offered Candidates")
        cursor.execute("""
        SELECT a.*, j.title as job_title
        FROM applications a
        JOIN job_postings j ON a.job_id = j.id
        WHERE a.status = 'Offered'
        """)
        offered_candidates = cursor.fetchall()

        for candidate in offered_candidates:
            st.write(f"Name: {candidate['applicant_name']}")
            st.write(f"Email: {candidate['email']}")
            st.write(f"Job: {candidate['job_title']}")
            
            # Check if 'status_update_date' exists, otherwise use 'submission_date'
            if 'status_update_date' in candidate:
                st.write(f"Offer Date: {candidate['status_update_date']}")
            elif 'submission_date' in candidate:
                st.write(f"Application Date: {candidate['submission_date']}")
            
            st.write("---")


    elif selected_submenu == "Provide Feedback":
        st.subheader("Provide Interview Feedback")
        interviewer_id = st.session_state.user['id']
        cursor.execute("""
SELECT i.id, a.applicant_name, j.title as job_title
FROM interviews i
JOIN applications a ON i.applicant_id = a.id
JOIN job_postings j ON a.job_id = j.id
WHERE JSON_CONTAINS(i.interviewers, %s)
AND i.interview_date < CURDATE()
ORDER BY i.interview_date DESC
""", (json.dumps(interviewer_id),))
        past_interviews = cursor.fetchall()

        if past_interviews:
            selected_interview = st.selectbox("Select Interview", [f"{i['applicant_name']} - {i['job_title']}" for i in past_interviews])
            interview_id = past_interviews[[f"{i['applicant_name']} - {i['job_title']}" for i in past_interviews].index(selected_interview)]['id']

            feedback_text = st.text_area("Feedback")
            rating = st.slider("Rating", 1, 5, 3)

            if st.button("Submit Feedback"):
                add_feedback(interview_id, interviewer_id, feedback_text, rating)
                st.success("Feedback submitted successfully!")
        else:
            st.write("No past interviews found.")

    elif selected_submenu == "View Job Postings":
        st.subheader("Available Job Postings")
        cursor.execute("SELECT * FROM job_postings WHERE deadline >= CURDATE() ORDER BY created_at DESC")
        jobs = cursor.fetchall()

        if 'viewing_job' in st.session_state:
            # Display job details
            job_id = st.session_state.viewing_job
            cursor.execute("SELECT * FROM job_postings WHERE id = %s", (job_id,))
            job = cursor.fetchone()

            if job:
                st.subheader(f"Job Details: {job['title']}")
                st.write(f"**Department:** {job['department']}")
                st.write(f"**Location:** {job['location']}")
                st.write(f"**Deadline:** {job['deadline'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Description:** {job['description']}")
                # ... add more details as needed

                if st.button("Back to Job Postings"):
                    del st.session_state.viewing_job
                    st.rerun()
            else:
                st.write("Job not found.")

        else:
            # Display job postings list
            for job in jobs:
                with st.expander(f"{job['title']} - {job['department']}"):
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Deadline:** {job['deadline'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Description:** {job['description']}")
                    if st.button(f"View Details", key=f"view_details_{job['id']}"):
                        st.session_state.viewing_job = job['id']
                        st.rerun()

    elif selected_submenu == "Submit Application":
        st.subheader("Submit Job Application")
        
        cursor.execute("SELECT id, title, department, deadline FROM job_postings WHERE deadline >= NOW()")
        jobs = cursor.fetchall()
        job_options = [f"{job['id']}: {job['title']} - {job['department']} (Deadline: {job['deadline'].strftime('%Y-%m-%d %H:%M')})" for job in jobs]
        
        selected_job = st.selectbox("Select Job", job_options)
        job_id = int(selected_job.split(":")[0])

        applicant_name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

        # Get custom application form
        application_form = get_application_form(job_id)
        form_responses = {}
        if application_form:
            st.subheader("Additional Information")
            form_fields = json.loads(application_form['form_fields'])
            for field in form_fields:
                if field['type'] == "Text":
                    form_responses[field['name']] = st.text_input(field['name'])
                elif field['type'] == "Dropdown":
                    form_responses[field['name']] = st.selectbox(field['name'], field['options'])
                elif field['type'] == "Checkbox":
                    form_responses[field['name']] = st.multiselect(field['name'], field['options'])

        if st.button("Submit Application"):
            if resume_file:
                submit_application(job_id, applicant_name, email, phone, resume_file, form_responses)
                st.success("Application submitted successfully!")
                
                # Send confirmation email
                send_notification(email, "Application Received", f"Thank you for applying to the position of {selected_job.split(':')[1].split('-')[0].strip()}. We have received your application and will review it shortly.")
            else:
                st.error("Please upload your resume.")

    elif selected_submenu == "In Review":
        if st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
            st.subheader("Applications In Review")
            
            query = """
            SELECT a.id, a.applicant_name, j.title as job_title, f.feedback_text, f.rating, u.full_name as interviewer_name
            FROM applications a
            JOIN job_postings j ON a.job_id = j.id
            JOIN feedback f ON a.id = f.application_id
            JOIN users u ON f.interviewer_id = u.id
            WHERE a.status = 'In Review'
            ORDER BY a.submission_date DESC
            """
            cursor.execute(query)
            applications = cursor.fetchall()

            for app in applications:
                with st.expander(f"{app['applicant_name']} - {app['job_title']}"):
                    st.write(f"Interviewer: {app['interviewer_name']}")
                    st.write(f"Feedback: {app['feedback_text']}")
                    st.write(f"Rating: {app['rating']}/5")

                    decision = st.selectbox("Decision", ["Select", "Offer", "Reject"], key=f"decision_{app['id']}")
                    if decision != "Select":
                        if st.button("Confirm Decision", key=f"confirm_{app['id']}"):
                            new_status = "Offered" if decision == "Offer" else "Rejected"
                            cursor.execute("UPDATE applications SET status = %s WHERE id = %s", (new_status, app['id']))
                            db.commit()
                            st.success(f"Application status updated to {new_status}")
                            st.rerun()
        else:
            st.write("You don't have permission to access this section.")

    elif selected_submenu == "Video Interviews1":
        st.subheader("Video Interviews")
        
        # Create new video interview request
        if st.checkbox("Create New Video Interview Request"):
            cursor.execute("SELECT id, applicant_name FROM applications WHERE status = 'In Review'")
            candidates = cursor.fetchall()
            selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in candidates])
            application_id = int(selected_candidate.split(":")[0])
            
            question = st.text_area("Enter the interview question")
            if st.button("Send Video Interview Request"):
                create_video_interview_request(application_id, question)
                st.success("Video interview request sent successfully!")

        # View and analyze completed video interviews
        st.subheader("Completed Video Interviews")
        cursor.execute("""
        SELECT vi.id, a.applicant_name, vi.status, vi.score
        FROM video_interviews vi
        JOIN applications a ON vi.application_id = a.id
        WHERE vi.status IN ('completed', 'analyzed')
        """)
        completed_interviews = cursor.fetchall()

        for interview in completed_interviews:
            with st.expander(f"{interview['applicant_name']} - Status: {interview['status']}"):
                if interview['status'] == 'completed':
                    if st.button(f"Analyze Interview {interview['id']}"):
                        score, analysis = analyze_video_interview(interview['id'])
                        st.success(f"Analysis complete. Overall score: {score:.2f}")
                        st.text_area("Analysis Result", analysis, height=300)
                elif interview['status'] == 'analyzed':
                    st.write(f"Score: {interview['score']:.2f}")
                    cursor.execute("SELECT analysis_result FROM video_interviews WHERE id = %s", (interview['id'],))
                    analysis = cursor.fetchone()['analysis_result']
                    st.text_area("Analysis Result", analysis, height=300)

    elif selected_submenu == "Video Interviews":
        st.subheader("Video Interviews")
        
        cursor.execute("""
        SELECT vi.id, vi.question, vi.status
        FROM video_interviews vi
        JOIN applications a ON vi.application_id = a.id
        WHERE a.email = %s AND vi.status = 'pending'
        """, (st.session_state.user['email'],))
        pending_interviews = cursor.fetchall()
        
        if pending_interviews:
            st.write("You have pending video interviews. Please complete them:")
            for interview in pending_interviews:
                st.write(f"Question: {interview['question']}")
                if st.button(f"Take Interview {interview['id']}"):
                    record_video_interview(interview['id'])
        else:
            st.write("You have no pending video interviews at this time.")

    elif selected_submenu == "Historical Recruitment Analysis":
        historical_recruitment_analysis()

    elif selected_submenu == "Application Status":
        st.subheader("Your Application Status")
        cursor.execute("""
        SELECT a.*, j.title as job_title
        FROM applications a
        JOIN job_postings j ON a.job_id = j.id
        WHERE a.email = %s
        ORDER BY a.submission_date DESC
        """, (st.session_state.user['email'],))
        applications = cursor.fetchall()

        for app in applications:
            st.write(f"Job: {app['job_title']}")
            st.write(f"Status: {app['status']}")
            st.write(f"Submitted: {app['submission_date']}")
            
            if app['status'] == 'Interview':
                cursor.execute("SELECT * FROM interviews WHERE applicant_id = %s", (app['id'],))
                interviews = cursor.fetchall()
                for interview in interviews:
                    st.write(f"Interview scheduled for: {interview['interview_date']} at {interview['interview_time']}")
                    st.write(f"Mode: {interview['interview_mode']}")
                    if interview['interview_location']:
                        st.write(f"Location: {interview['interview_location']}")
            
            st.write("---")

    elif selected_submenu == "GitHub Profile Analysis":
        github_profile_analysis()
    

if __name__ == "__main__":
    main()
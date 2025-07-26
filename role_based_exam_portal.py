# Role-Based Exam Portal Flask App

from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import jwt, datetime
from functools import wraps
import os, re, fitz, json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask_mail import Mail, Message
import logging
from datetime import datetime, timedelta, timezone
from app2 import pdf_to_images,process_and_extract 
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'raiyanmugdho6@gmail.com'
app.config['MAIL_PASSWORD'] = 'nmbmlnukshpokcau'
app.config['MAIL_DEFAULT_SENDER'] = 'raiyanmugdho6@gmail.com'
mail = Mail(app)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}}, expose_headers=["Authorization"])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['MONGO_URI'] = 'mongodb://localhost:27017/V0'
app.config['SECRET_KEY'] = 'yoursecretkey'
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
ALLOWED_EXTENSIONS = {'pdf'}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # First, check headers
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]
        # Fallback to cookie
        if not token:
            token = request.cookies.get('token')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.Autograder.find_one({'email': data['email']})
        except:
            return jsonify({'message': 'Invalid or expired token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated



@app.route('/api/student-list')
@token_required
def student_list(current_user):
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403

    students = mongo.db.Autograder.find({'role': 'student'}, {'email': 1})
    emails = [s['email'] for s in students]
    return jsonify({'students': emails})

@app.route('/api/student-data', methods=['GET'])
@token_required
def get_student_data(current_user):
    user = mongo.db.Autograder.find_one({'email': current_user['email']})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Extract data for frontend
    response_data = {
        'email': user.get('email'),
        'role': user.get('role'),
        'marks_history': user.get('marks_history', []),
        'mcq_marks_history': user.get('mcq_marks_history', [])  # ← Make sure this is included
    }

    return jsonify(response_data), 200


@app.route('/api/teacher-info')
@token_required
def teacher_info(current_user):
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403
    return jsonify({'email': current_user['email']})

# NEW ROUTES FOR UPDATE MARKS FUNCTIONALITY

@app.route('/api/all-students-marks')
@token_required
def get_all_students_marks(current_user):
    """Get all students' marks for the update marks feature"""
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403
    
    try:
        # Get all students with their marks
        students = mongo.db.Autograder.find({'role': 'student'}, {
            'email': 1, 
            'marks_history': 1, 
            'mcq_marks_history': 1,
            'updated_marks': 1  # For storing manually updated marks
        })
        
        all_marks_data = {}
        
        for student in students:
            email = student['email']
            marks_history = student.get('marks_history', [])
            mcq_marks_history = student.get('mcq_marks_history', [])
            updated_marks = student.get('updated_marks', {})
            
            # Structure the data for frontend
            student_marks = {
                'descriptive': {},
                'mcq': {}
            }
            
            # Add descriptive marks from marks_history
            for i, mark_entry in enumerate(marks_history):
                if 'marks_obtained_list' in mark_entry:
                    for j, mark in enumerate(mark_entry['marks_obtained_list']):
                        question_key = f"Question {j+1}"
                        # Use updated marks if available, otherwise use original
                        if email in updated_marks and 'descriptive' in updated_marks[email] and question_key in updated_marks[email]['descriptive']:
                            current_mark = updated_marks[email]['descriptive'][question_key]
                        else:
                            current_mark = mark
                        
                        student_marks['descriptive'][question_key] = {
                            'current': current_mark,
                            'max': 10  # Default max marks, you can adjust this
                        }
            
            # Add MCQ marks from mcq_marks_history
            for i, mcq_entry in enumerate(mcq_marks_history):
                mcq_key = f"MCQ Set {chr(65+i)}"  # A, B, C, etc.
                # Use updated marks if available, otherwise use original
                if email in updated_marks and 'mcq' in updated_marks[email] and mcq_key in updated_marks[email]['mcq']:
                    current_mark = updated_marks[email]['mcq'][mcq_key]
                else:
                    current_mark = mcq_entry.get('total_score', 0)
                
                student_marks['mcq'][mcq_key] = {
                    'current': current_mark,
                    'max': 20  # Default max marks for MCQ, you can adjust this
                }
            
            # Only include students who have marks
            if student_marks['descriptive'] or student_marks['mcq']:
                all_marks_data[email] = student_marks
        
        return jsonify(all_marks_data)
        
    except Exception as e:
        logging.error(f"Error fetching all students marks: {str(e)}")
        return jsonify({'message': f'Error fetching marks: {str(e)}'}), 500

@app.route('/api/student-marks/<student_email>')
@token_required
def get_student_marks(current_user, student_email):
    """Get marks for a specific student"""
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403
    
    try:
        student = mongo.db.Autograder.find_one({'email': student_email, 'role': 'student'}, {
            'email': 1, 
            'marks_history': 1, 
            'mcq_marks_history': 1,
            'updated_marks': 1
        })
        
        if not student:
            return jsonify({'message': 'Student not found'}), 404
        
        marks_history = student.get('marks_history', [])
        mcq_marks_history = student.get('mcq_marks_history', [])
        updated_marks = student.get('updated_marks', {})
        
        # Structure the data
        student_marks = {
            'descriptive': {},
            'mcq': {}
        }
        
        # Add descriptive marks
        for i, mark_entry in enumerate(marks_history):
            if 'marks_obtained_list' in mark_entry:
                for j, mark in enumerate(mark_entry['marks_obtained_list']):
                    question_key = f"Question {j+1}"
                    # Use updated marks if available
                    current_mark = updated_marks.get('descriptive', {}).get(question_key, mark)
                    student_marks['descriptive'][question_key] = {
                        'current': current_mark,
                        'max': 10
                    }
        
        # Add MCQ marks
        for i, mcq_entry in enumerate(mcq_marks_history):
            mcq_key = f"MCQ Set {chr(65+i)}"
            current_mark = updated_marks.get('mcq', {}).get(mcq_key, mcq_entry.get('total_score', 0))
            student_marks['mcq'][mcq_key] = {
                'current': current_mark,
                'max': 20
            }
        
        return jsonify(student_marks)
        
    except Exception as e:
        logging.error(f"Error fetching student marks: {str(e)}")
        return jsonify({'message': f'Error fetching student marks: {str(e)}'}), 500

@app.route('/api/update-student-marks', methods=['POST'])
@token_required
def update_student_marks(current_user):
    """Update marks for a specific student"""
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        student_email = data.get('studentEmail')
        updated_marks = data.get('marks')
        
        if not student_email or not updated_marks:
            return jsonify({'message': 'Missing required data'}), 400
        
        # Verify student exists
        student = mongo.db.Autograder.find_one({'email': student_email, 'role': 'student'})
        if not student:
            return jsonify({'message': 'Student not found'}), 404
        
        # Update the student's marks in the database
        update_result = mongo.db.Autograder.update_one(
            {'email': student_email, 'role': 'student'},
            {
                '$set': {
                    f'updated_marks': updated_marks,
                    f'last_updated_by': current_user['email'],
                    f'last_updated_at': datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        if update_result.modified_count == 0:
            return jsonify({'message': 'No changes made'}), 400
        
        # Log the update for audit purposes
        logging.info(f"Teacher {current_user['email']} updated marks for student {student_email}")
        
        return jsonify({
            'message': 'Marks updated successfully',
            'student_email': student_email,
            'updated_marks': updated_marks
        })
        
    except Exception as e:
        logging.error(f"Error updating student marks: {str(e)}")
        return jsonify({'message': f'Error updating marks: {str(e)}'}), 500

@app.route('/api/bulk-update-marks', methods=['POST'])
@token_required
def bulk_update_marks(current_user):
    """Update marks for multiple students at once"""
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        updates = data.get('updates', [])
        
        if not updates:
            return jsonify({'message': 'No updates provided'}), 400
        
        successful_updates = []
        failed_updates = []
        
        for update in updates:
            student_email = update.get('studentEmail')
            marks = update.get('marks')
            
            if not student_email or not marks:
                failed_updates.append({
                    'student_email': student_email,
                    'error': 'Missing required data'
                })
                continue
            
            try:
                # Verify student exists
                student = mongo.db.Autograder.find_one({'email': student_email, 'role': 'student'})
                if not student:
                    failed_updates.append({
                        'student_email': student_email,
                        'error': 'Student not found'
                    })
                    continue
                
                # Update the student's marks
                update_result = mongo.db.Autograder.update_one(
                    {'email': student_email, 'role': 'student'},
                    {
                        '$set': {
                            f'updated_marks': marks,
                            f'last_updated_by': current_user['email'],
                            f'last_updated_at': datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
                if update_result.modified_count > 0:
                    successful_updates.append(student_email)
                else:
                    failed_updates.append({
                        'student_email': student_email,
                        'error': 'No changes made'
                    })
                    
            except Exception as e:
                failed_updates.append({
                    'student_email': student_email,
                    'error': str(e)
                })
        
        return jsonify({
            'message': f'Bulk update completed. {len(successful_updates)} successful, {len(failed_updates)} failed.',
            'successful_updates': successful_updates,
            'failed_updates': failed_updates
        })
        
    except Exception as e:
        logging.error(f"Error in bulk update: {str(e)}")
        return jsonify({'message': f'Error in bulk update: {str(e)}'}), 500

# END OF NEW ROUTES FOR UPDATE MARKS FUNCTIONALITY

@app.route('/')
def select_role():
    return render_template('select_role.html')

@app.route('/verify-manual', methods=['GET', 'POST'])
def verify_manual():
    if request.method == 'POST':
        token = request.form.get('token')
        return redirect(url_for('verify_email', token=token))
    return render_template('verification.html')

@app.route('/register-student', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        data = request.json
        if mongo.db.Autograder.find_one({'email': data['email']}):
            return jsonify({'message': 'Email already exists'}), 409
        hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        verification_token = jwt.encode({
            'email': data['email'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        mongo.db.Autograder.insert_one({
            'email': data['email'],
            'password': hashed_pw,
            'verified': False,
            'role': 'student',
            'marks_history': []
        })
        msg = Message('Verify your account', recipients=[data['email']])
        msg.body = f"Hi,\n\nPlease verify your account by entering this token:\n\n{verification_token}\n\non the verification page: http://localhost:5001/verify-manual"
        mail.send(msg)
        return jsonify({'message': 'Student registered. Check your email and verify.'}), 201
    return render_template('register_student.html')

@app.route('/register-teacher', methods=['GET', 'POST'])
def register_teacher():
    if request.method == 'POST':
        data = request.json
        if mongo.db.Autograder.find_one({'email': data['email']}):
            return jsonify({'message': 'Email already exists'}), 409
        hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        verification_token = jwt.encode({
            'email': data['email'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        mongo.db.Autograder.insert_one({
            'email': data['email'],
            'password': hashed_pw,
            'verified': False,
            'role': 'teacher'
        })
        msg = Message('Verify your account', recipients=[data['email']])
        msg.body = f"Hi,\n\nPlease verify your account by entering this token:\n\n{verification_token}\n\non the verification page: http://localhost:5001/verify-manual"
        mail.send(msg)
        return jsonify({'message': 'Teacher registered. Check your email and verify.'}), 201
    return render_template('register_teacher.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        user = mongo.db.Autograder.find_one({'email': data['email']})
        if not user or not bcrypt.check_password_hash(user['password'], data['password']):
            return jsonify({'message': 'Invalid credentials'}), 401
        if not user.get('verified', False):
            return jsonify({'message': 'Please verify your email before logging in.'}), 403
        token = jwt.encode({
            'email': user['email'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=2)
          }, app.config['SECRET_KEY'], algorithm='HS256')
        resp = jsonify({'token': token, 'role': user['role']})
        resp.set_cookie('token', token)  # Save token in browser cookie
        return resp

    return render_template('login.html')

@app.route('/dashboard-student')
@token_required
def dashboard_student(current_user):
    if current_user['role'] != 'student':
        return jsonify({'message': 'Unauthorized'}), 403
    return render_template('dashboard_student.html', user=current_user)


@app.route('/dashboard-teacher')
@token_required
def teacher_dashboard(current_user):
    if current_user['role'] != 'teacher':
        return redirect(url_for('login'))
    return render_template('dashboard_teacher.html')

#mcq checking route
@app.route('/mcq-checker', methods=['GET', 'POST'])
@token_required
def mcq_checker(current_user):
    if current_user['role'] != 'teacher':
        return redirect(url_for('login'))

    students = [s['email'] for s in mongo.db.Autograder.find({'role': 'student'})]
    result_data = None
    error_msg = None

    if request.method == 'POST':
        student_email = request.form.get('student_email')
        file = request.files.get('mcq_pdf')

        if not student_email or not file:
            error_msg = "Missing PDF or student email"
        elif not file.filename.lower().endswith('.pdf'):
            error_msg = "Only PDF files are allowed"
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image_paths = pdf_to_images(file_path, app.config['UPLOAD_FOLDER'])
            all_results = [process_and_extract(img, app.config['UPLOAD_FOLDER']) for img in image_paths]
            all_results = [r for r in all_results if r is not None]

            if not all_results:
                error_msg = "Could not extract MCQ results. Please try again."
            else:
                total_score = sum([r['total_score'] for r in all_results])
                total_marks = sum([r['out_of'] for r in all_results])
                percentage = round((total_score / total_marks) * 100, 2) if total_marks > 0 else 0.0

                result_data = {
                    'total_score': total_score,
                    'total_marks': total_marks,
                    'percentage': percentage,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

                # Save to teacher
                mongo.db.Autograder.update_one(
                    {'email': current_user['email']},
                    {'$push': {'mcq_marks_history': result_data}}
                )

                # Save to student
                student_result = {
                    'total_score': total_score,
                    'percentage': percentage,
                    'timestamp': result_data['timestamp']
                }

                mongo.db.Autograder.update_one(
                    {'email': student_email, 'role': 'student'},
                    {'$push': {'mcq_marks_history': student_result}}
                )

    return render_template('mcq_checker.html', result=result_data, error=error_msg, students=students)



#mcq checking route

@app.route('/verify/<token>')
def verify_email(token):
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        user = mongo.db.Autograder.find_one({'email': data['email']})
        if not user:
            return "User not found", 404
        if user.get('verified'):
            return "Already verified!"
        mongo.db.Autograder.update_one({'email': data['email']}, {'$set': {'verified': True}})
        return redirect(url_for('login'))
    except jwt.ExpiredSignatureError:
        return "Verification link expired", 400
    except jwt.InvalidTokenError:
        return "Invalid verification token", 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_answers_from_pdf(pdf_path, is_teacher=False):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
    except Exception:
        return [], [], []
    answer_pattern = re.compile(r'Answer\s*\d+[:\s]', re.IGNORECASE)
    sections = answer_pattern.split(text)[1:]
    answers, keywords_list, marks = [], [], []
    for section in sections:
        if is_teacher:
            keywords_start = section.find("Keywords:")
            marks_start = section.find("Marks:")
            if marks_start == -1:
                continue
            answer_text = section[:min(keywords_start if keywords_start != -1 else marks_start, marks_start)].strip()
            keywords = [kw.strip() for kw in section[keywords_start+9:marks_start].split(',')] if keywords_start != -1 else []
            try:
                mark = float(section[marks_start+6:].split()[0])
            except:
                continue
            answers.append(answer_text)
            keywords_list.append(keywords)
            marks.append(mark)
        else:
            end = min([p for p in [section.find("Keywords:"), section.find("Marks:")] if p != -1] or [len(section)])
            answers.append(section[:end].strip())
    return (answers, keywords_list, marks) if is_teacher else (answers, [], [])

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return set(tokens)

def calculate_keyword_matches(teacher_keywords, student_answer):
    return len(preprocess_text(" ".join(teacher_keywords)).intersection(preprocess_text(student_answer))), len(teacher_keywords)

def calculate_combined_score(teacher_answer, teacher_keywords, student_answer, marks):
    match_count, keyword_count = calculate_keyword_matches(teacher_keywords, student_answer)
    keyword_score = match_count / keyword_count if keyword_count > 0 else 0
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, teacher_answer, student_answer).ratio()
    return round(((keyword_score + similarity) / 2) * marks, 2)

def calculate_marks(teacher_answers, student_answers, teacher_keywords, marks):
    total, scores = 0, []
    for i in range(len(teacher_answers)):
        score = calculate_combined_score(teacher_answers[i], teacher_keywords[i], student_answers[i], marks[i])
        scores.append(score)
        total += score
    percent = round((total / sum(marks)) * 100, 2) if sum(marks) > 0 else 0
    return round(total, 2), percent, scores

@app.route('/compare', methods=['POST'])
@token_required
def compare(current_user):
    from datetime import datetime
    if current_user['role'] != 'teacher':
        return jsonify({'message': 'Only teachers can perform comparisons'}), 403

    try:
        student_email = request.form.get('studentEmail')
        teacher_file = request.files.get('teachersMarksheet')
        student_file = request.files.get('studentMarksheet')

        if not all([teacher_file, student_file, student_email]):
            return jsonify({'message': 'Missing form data'}), 400

        if not allowed_file(teacher_file.filename) or not allowed_file(student_file.filename):
            return jsonify({'message': 'Only PDF files are allowed'}), 400

        student = mongo.db.Autograder.find_one({'email': student_email, 'role': 'student'})
        if not student:
            return jsonify({'message': 'Invalid or unregistered student email'}), 404

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        teacher_path = os.path.join(app.config['UPLOAD_FOLDER'], teacher_file.filename)
        student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_file.filename)
        teacher_file.save(teacher_path)
        student_file.save(student_path)

        teacher_answers, teacher_keywords, marks = extract_answers_from_pdf(teacher_path, is_teacher=True)
        student_answers, _, _ = extract_answers_from_pdf(student_path)

        if not teacher_answers or not student_answers:
            return jsonify({'message': 'Could not extract answers from one or both PDFs'}), 400

        student_marks, percentage, marks_list = calculate_marks(teacher_answers, student_answers, teacher_keywords, marks)
        result = {
            'student_marks': student_marks,
            'student_percentage': percentage,
            'marks_obtained_list': marks_list,
            'timestamp': datetime.utcnow().isoformat()
        }

        mongo.db.Autograder.update_one({'email': student_email}, {'$push': {'marks_history': result}})
        return jsonify(result)

    except Exception as e:
        return jsonify({'message': f'Server error: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5001)
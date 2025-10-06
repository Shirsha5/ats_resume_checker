print("=== APP STARTING ===")
import subprocess
import sys
# try:
#     import spacy
#     try:
#         nlp = spacy.load("en_core_web_sm")
#         print("✅ SpaCy model loaded successfully")
#     except OSError:
#         print("⚠️ SpaCy model not found, skipping for now")
#         # Don't download during startup - causes timeouts
# except ImportError:
#     print("⚠️ SpaCy not available")
print("=== LOADING FLASK ===")

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import hashlib
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from resume_parser import ResumeParser
from criteria_evaluator import CriteriaEvaluator
from models import init_db, User, get_user_by_username, create_user
import config
# Add this import after your existing imports
try:
    from ml_integration_minimal import create_ml_enhancer
    ml_enhancer = create_ml_enhancer('ml_models')
    print(f"🤖 ML Status: {ml_enhancer.get_status()}")
except ImportError:
    print("⚠️ ML enhancement not available - continuing with rule-based evaluation")
    ml_enhancer = None

import os

app = Flask(__name__)
app.config.from_object(config.Config)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)

# Initialize database
init_db()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize parsers
resume_parser = ResumeParser()
criteria_evaluator = CriteriaEvaluator()

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')

        user = get_user_by_username(username)
        if user and user.check_password(password):
            login_user(user, remember=request.form.get('remember', False))
            next_page = request.args.get('next')
            flash(f'Welcome back, {username}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_files():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files[]')
        course_type = request.form.get('course_type', '5year')
        internship_type = request.form.get('internship_type', 'long_term')

        if not uploaded_files or len(uploaded_files) == 0:
            flash('No files selected. Please choose at least one resume file.', 'error')
            return redirect(request.url)

        if len(uploaded_files) > 10:
            flash('Maximum 10 files allowed per batch. Please reduce the number of files.', 'error')
            return redirect(request.url)

        results = []
        processed_files = 0
        error_files = []

        for file in uploaded_files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                if filename.lower().endswith(('.pdf', '.docx')):
                    try:
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)

                        # Parse resume
                        parsed_resume = resume_parser.parse_resume(file_path)
                        
                        if 'error' not in parsed_resume:
                            # Evaluate criteria (RULE-BASED)
                            rule_result = criteria_evaluator.classify_candidate(
                                parsed_resume, course_type, internship_type
                            )
                            
                            # ADD ML ENHANCEMENT if available
                            if ml_enhancer and ml_enhancer.ml_enabled:
                                enhanced_result = ml_enhancer.enhance_evaluation(
                                    rule_result, 
                                    parsed_resume, 
                                    parsed_resume.get('raw_text_preview', '')
                                )
                                enhanced_result['upload_time'] = datetime.now().isoformat()
                                results.append(enhanced_result)
                            else:
                                # Use rule-based only (same as before)
                                rule_result['upload_time'] = datetime.now().isoformat()
                                results.append(rule_result)
                            
                            processed_files += 1


                        # Clean up uploaded file
                        if os.path.exists(file_path):
                            os.remove(file_path)

                    except Exception as e:
                        error_files.append(f"{filename}: Processing error - {str(e)}")
                else:
                    error_files.append(f"{filename}: Unsupported file format (only PDF and DOCX allowed)")

        # Store results in session for display
        session['results'] = results
        session['processing_summary'] = {
            'total_files': len(uploaded_files),
            'processed_files': processed_files,
            'error_files': error_files,
            'course_type': course_type,
            'internship_type': internship_type,
            'processed_time': datetime.now().isoformat()
        }

        if processed_files > 0:
            flash(f'Successfully processed {processed_files} resume(s).', 'success')
        if error_files:
            flash(f'Errors encountered with {len(error_files)} file(s). Check results for details.', 'warning')

        return redirect(url_for('show_results'))

    return render_template('upload.html')

@app.route('/results')
@login_required
def show_results():
    results = session.get('results', [])
    summary = session.get('processing_summary', {})

    if not results:
        flash('No results to display. Please upload and process resume files first.', 'info')
        return redirect(url_for('upload_files'))

    # Segregate results
    ma_team_matches = [r for r in results if r['final_category'] == 'ma_team_match']
    shortlisted = [r for r in results if r['final_category'] == 'shortlisted']
    others = [r for r in results if r['final_category'] == 'others']

    # Sort by preference score (highest first)
    ma_team_matches.sort(key=lambda x: x.get('preference_score', 0), reverse=True)
    shortlisted.sort(key=lambda x: x.get('preference_score', 0), reverse=True)

    statistics = {
        'total_candidates': len(results),
        'ma_team_count': len(ma_team_matches),
        'shortlisted_count': len(shortlisted),
        'others_count': len(others),
        'ma_team_percentage': round((len(ma_team_matches) / len(results)) * 100, 1) if results else 0,
        'shortlisted_percentage': round((len(shortlisted) / len(results)) * 100, 1) if results else 0
    }

    return render_template('results.html', 
                         ma_team_matches=ma_team_matches,
                         shortlisted=shortlisted,
                         others=others,
                         statistics=statistics,
                         summary=summary)

@app.route('/export_results')
@login_required
def export_results():
    results = session.get('results', [])
    if not results:
        flash('No results to export.', 'error')
        return redirect(url_for('dashboard'))

    # Create CSV export
    import csv
    import io
    from flask import make_response

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Filename', 'Category', 'CGPA', 'Academic Year', 'Company Law', 
        'Contract Law', 'Legal Research', 'Moot Court', 'Preference Score',
        'Long-term Eligible', 'Short-term Eligible'
    ])

    # Write data
    for result in results:
        long_term_eval = result.get('long_term_evaluation', {})
        short_term_eval = result.get('short_term_evaluation', {})
        criteria_met = long_term_eval.get('criteria_met', {})

        writer.writerow([
            result.get('filename', ''),
            result.get('final_category', ''),
            result.get('cgpa', 'N/A'),
            result.get('academic_year', 'N/A'),
            'Yes' if criteria_met.get('company_law', False) else 'No',
            'Yes' if criteria_met.get('contract_law', False) else 'No',
            'Yes' if criteria_met.get('legal_research', False) else 'No',
            'Yes' if result.get('moot_court_experience', False) else 'No',
            result.get('preference_score', 0),
            'Yes' if long_term_eval.get('eligible', False) else 'No',
            'Yes' if short_term_eval.get('eligible', False) else 'No'
        ])

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=ats_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response.headers['Content-type'] = 'text/csv'

    return response

@app.route('/clear_results')
@login_required
def clear_results():
    session.pop('results', None)
    session.pop('processing_summary', None)
    flash('Results cleared successfully.', 'info')
    return redirect(url_for('dashboard'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_code=404, error_message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
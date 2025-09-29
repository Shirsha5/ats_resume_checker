#!/usr/bin/env python3
"""
ATS Resume Checker Startup Script
Initializes the database and starts the Flask application
"""

import os
import sys
from app import app
from models import init_db

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask', 'flask_login', 'PyPDF2', 'docx', 'sqlite3'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'docx':
                import docx
            elif package == 'sqlite3':
                import sqlite3
            elif package == 'PyPDF2':
                import PyPDF2
            elif package == 'flask_login':
                import flask_login
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False

    return True

def check_optional_dependencies():
    """Check optional dependencies and provide warnings"""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy English model found - Enhanced NLP processing enabled")
        except OSError:
            print("⚠️  spaCy English model not found - Basic parsing will be used")
            print("   Install with: python -m spacy download en_core_web_sm")
    except ImportError:
        print("⚠️  spaCy not found - Basic text processing will be used")

def setup_directories():
    """Create necessary directories"""
    directories = ['uploads', 'database', 'static/css', 'static/js', 'templates']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def initialize_system():
    """Initialize the system components"""
    print("🚀 Initializing ATS Resume Checker System...")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check optional dependencies  
    check_optional_dependencies()

    # Setup directories
    setup_directories()

    # Initialize database
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    print("=" * 50)
    print("🎉 System initialization complete!")
    print()
    print("📋 Default Login Credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print()
    print("🔗 Application will be available at: http://localhost:5000")
    print("=" * 50)

if __name__ == '__main__':
    initialize_system()

    # Start the Flask application
    try:
        print("🌐 Starting Flask development server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Configuration Validator for ATS Resume Checker
Validates system configuration and dependencies
"""

import os
import sys
import importlib
from config import Config

def validate_config():
    """Validate configuration settings"""
    print("🔍 Validating Configuration...")

    issues = []

    # Check secret key
    if Config.SECRET_KEY == 'dev-secret-key-change-in-production':
        issues.append("⚠️  Using default SECRET_KEY - change for production")

    # Check database path
    db_dir = os.path.dirname(Config.DATABASE_PATH)
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            print(f"✅ Created database directory: {db_dir}")
        except Exception as e:
            issues.append(f"❌ Cannot create database directory: {e}")

    # Check upload folder
    if not os.path.exists(Config.UPLOAD_FOLDER):
        try:
            os.makedirs(Config.UPLOAD_FOLDER)
            print(f"✅ Created upload directory: {Config.UPLOAD_FOLDER}")
        except Exception as e:
            issues.append(f"❌ Cannot create upload directory: {e}")

    # Check file size limit
    max_size_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
    if max_size_mb > 50:
        issues.append(f"⚠️  Large file size limit: {max_size_mb}MB")

    print(f"✅ Configuration validation complete")

    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False

    return True

def validate_dependencies():
    """Validate required dependencies"""
    print("\n🔍 Validating Dependencies...")

    required_deps = {
        'flask': 'Flask web framework',
        'flask_login': 'User authentication',
        'PyPDF2': 'PDF text extraction', 
        'docx': 'DOCX file processing',
        'sqlite3': 'Database operations'
    }

    optional_deps = {
        'spacy': 'Enhanced NLP processing',
        'pdfplumber': 'Fallback PDF extraction'
    }

    missing_required = []
    missing_optional = []

    # Check required dependencies
    for dep, desc in required_deps.items():
        try:
            if dep == 'docx':
                importlib.import_module('docx')
            else:
                importlib.import_module(dep)
            print(f"   ✅ {dep} - {desc}")
        except ImportError:
            missing_required.append((dep, desc))
            print(f"   ❌ {dep} - {desc}")

    # Check optional dependencies
    for dep, desc in optional_deps.items():
        try:
            importlib.import_module(dep)
            print(f"   ✅ {dep} - {desc}")
        except ImportError:
            missing_optional.append((dep, desc))
            print(f"   ⚠️  {dep} - {desc} (optional)")

    if missing_required:
        print("\n❌ Missing required dependencies:")
        for dep, desc in missing_required:
            print(f"   - {dep}: {desc}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    if missing_optional:
        print("\n💡 Optional enhancements available:")
        for dep, desc in missing_optional:
            print(f"   - {dep}: {desc}")

    return True

def validate_system():
    """Run complete system validation"""
    print("🏥 ATS Resume Checker - System Health Check")
    print("=" * 50)

    config_ok = validate_config()
    deps_ok = validate_dependencies()

    print("\n" + "=" * 50)

    if config_ok and deps_ok:
        print("✅ System validation passed - Ready to run!")
        return True
    else:
        print("❌ System validation failed - Please fix issues above")
        return False

if __name__ == '__main__':
    success = validate_system()
    sys.exit(0 if success else 1)

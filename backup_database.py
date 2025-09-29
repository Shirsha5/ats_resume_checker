#!/usr/bin/env python3
"""
Database Backup Utility for ATS Resume Checker
Creates timestamped backups of the SQLite database
"""

import os
import shutil
import sqlite3
from datetime import datetime
from config import Config

def backup_database():
    """Create a timestamped backup of the database"""
    if not os.path.exists(Config.DATABASE_PATH):
        print("❌ Database file not found. Nothing to backup.")
        return False

    # Create backups directory
    backup_dir = 'backups'
    os.makedirs(backup_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"ats_backup_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)

    try:
        # Copy database file
        shutil.copy2(Config.DATABASE_PATH, backup_path)

        # Verify backup
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()

        file_size = os.path.getsize(backup_path) / 1024  # KB

        print(f"✅ Database backup created successfully!")
        print(f"   📁 Location: {backup_path}")
        print(f"   📊 Users: {user_count}")
        print(f"   💾 Size: {file_size:.1f} KB")

        return True

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def list_backups():
    """List all available backups"""
    backup_dir = 'backups'

    if not os.path.exists(backup_dir):
        print("📁 No backup directory found.")
        return

    backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]

    if not backups:
        print("📁 No backup files found.")
        return

    print("📋 Available backups:")
    for backup in sorted(backups, reverse=True):
        path = os.path.join(backup_dir, backup)
        size = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"   📄 {backup} ({size:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_backups()
    else:
        backup_database()

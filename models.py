import sqlite3
import hashlib
import os
from flask_login import UserMixin
from config import Config

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    # Create processing_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            processing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_category TEXT,
            cgpa REAL,
            academic_year INTEGER,
            preference_score INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create default admin user if not exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                      ('admin', admin_password))
        print("✅ Default admin user created (username: admin, password: admin123)")

    conn.commit()
    conn.close()

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return User(row[0], row[1], row[2])
        return None

    def check_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest() == self.password_hash

    def update_last_login(self):
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (self.id,))
        conn.commit()
        conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return User(row[0], row[1], row[2])
    return None

def create_user(username, password):
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    try:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                      (username, password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return User(user_id, username, password_hash)
    except sqlite3.IntegrityError:
        conn.close()
        return None  # Username already exists

def log_processing_result(user_id, filename, result_category, cgpa=None, academic_year=None, preference_score=0):
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO processing_logs 
        (user_id, filename, result_category, cgpa, academic_year, preference_score) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, filename, result_category, cgpa, academic_year, preference_score))
    conn.commit()
    conn.close()

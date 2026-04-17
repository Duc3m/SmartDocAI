import sqlite3
from datetime import datetime

def get_db_connection():
    """Tạo kết nối tới file SQLite."""
    conn = sqlite3.connect('smartdoc.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = sqlite3.connect('smartdoc.db')
    c = conn.cursor()
    # Bảng lưu thông tin file
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  filename TEXT, 
                  num_chunks INTEGER,
                  upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Bảng lưu tin nhắn chat (Logic mới)
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_id INTEGER,
                  role TEXT,
                  content TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (file_id) REFERENCES files (id))''')
    conn.commit()
    conn.close()

def insert_file_metadata(filename, num_chunks):
    """Lưu thông tin file vào database."""
    conn = get_db_connection()
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        'INSERT INTO files (filename, upload_date, num_chunks) VALUES (?, ?, ?)',
        (filename, upload_date, num_chunks)
    )
    inserted_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.commit()
    conn.close()
    return inserted_id

def insert_message(file_id, role, content):
    """Lưu một tin nhắn vào database"""
    conn = sqlite3.connect('smartdoc.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages (file_id, role, content) VALUES (?, ?, ?)", 
              (file_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(file_id):
    """Lấy toàn bộ lịch sử chat của một file cụ thể"""
    conn = sqlite3.connect('smartdoc.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE file_id = ? ORDER BY timestamp ASC", (file_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def delete_chat_history(file_id):
    """Xóa sạch tin nhắn của một file"""
    conn = sqlite3.connect('smartdoc.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()

def get_all_files():
    """Lấy danh sách tất cả file đã tải lên[cite: 627]."""
    conn = get_db_connection()
    files = conn.execute('SELECT * FROM files ORDER BY upload_date DESC').fetchall()
    conn.close()
    return files

def delete_file_record(file_id):
    """Xóa thông tin file trong SQLite."""
    conn = get_db_connection()
    conn.execute('DELETE FROM files WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
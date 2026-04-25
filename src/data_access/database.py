import sqlite3
import json
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
    
    # Bảng lưu tin nhắn chat
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_id INTEGER,
                  role TEXT,
                  content TEXT,
                  citations TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (file_id) REFERENCES files (id))''')

    # Kiểm tra nếu cột "citations" đã tồn tại, nếu chưa thì thêm vào
    columns = c.execute("PRAGMA table_info(messages)").fetchall()
    column_names = [column[1] for column in columns]
    if "citations" not in column_names:
        c.execute("ALTER TABLE messages ADD COLUMN citations TEXT")

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

def insert_message(file_id, role, content, citations=None):
    """Lưu một tin nhắn vào database"""
    conn = sqlite3.connect('smartdoc.db')
    c = conn.cursor()
    # Chuẩn bị dữ liệu citations để lưu vào DB (nếu có)
    serialized_citations = None
    if citations:
        serialized_citations = json.dumps(citations, ensure_ascii=False)

    c.execute(
        "INSERT INTO messages (file_id, role, content, citations) VALUES (?, ?, ?, ?)",
        (file_id, role, content, serialized_citations),
    )
    conn.commit()
    conn.close()

def get_chat_history(file_id):
    """Lấy toàn bộ lịch sử chat của một file cụ thể"""
    conn = sqlite3.connect('smartdoc.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT role, content, citations FROM messages WHERE file_id = ? ORDER BY timestamp ASC",
        (file_id,),
    )
    rows = c.fetchall()
    conn.close()
    # Chuyển đổi dữ liệu từ DB thành định dạng dễ sử dụng trong ứng dụng
    history = []
    for row in rows:
        message = {"role": row["role"], "content": row["content"]}
        raw_citations = row["citations"]
        if raw_citations:
            try:
                parsed_citations = json.loads(raw_citations)
                if isinstance(parsed_citations, list):
                    message["citations"] = parsed_citations
            except json.JSONDecodeError:
                pass
        history.append(message)

    return history

def delete_chat_history(file_id):
    """Xóa sạch tin nhắn của một file"""
    conn = sqlite3.connect('smartdoc.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()

def get_all_files():
    """Lấy danh sách tất cả file đã tải lên"""
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
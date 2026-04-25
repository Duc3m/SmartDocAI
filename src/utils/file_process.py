import os
import time

from src.utils.timer import time_it
import streamlit as st
from src.data_access.document_loader import load_and_split_document
from src.data_access.vector_store import create_vector_db, load_vector_db, save_vector_db, get_retriever
from src.data_access.database import get_chat_history, insert_file_metadata

@time_it
def process_new_uploaded_file(uploaded_file, embedding_model, chunk_size=600, chunk_overlap=100):
    """Chỉ chịu trách nhiệm xử lý logic khi có file mới tải lên"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    
    # BƯỚC A: Lưu file tạm và băm nhỏ
    original_ext = os.path.splitext(uploaded_file.name)[1].lower()    
    temp_path = os.path.join(data_dir, f"temp_{int(time.time())}{original_ext}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    chunks = load_and_split_document(
        temp_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # BƯỚC B & C: Lưu Metadata và Đổi tên file
    file_id = insert_file_metadata(uploaded_file.name, len(chunks))
    final_filename = f"{file_id}_{uploaded_file.name}"
    file_path = os.path.join(data_dir, final_filename)
    vectordb_path = os.path.join("vector_db", f"{final_filename}_index")
    os.rename(temp_path, file_path)
    
    # BƯỚC D: Tạo Vector DB
    vector_db = create_vector_db(chunks, embedding_model)
    save_vector_db(vector_db, vectordb_path)
    
    # Cập nhật Session State
    st.session_state.retriever = get_retriever(vector_db, 3)
    st.session_state.file_processed = True
    st.session_state.current_file = uploaded_file.name
    st.session_state.current_file_id = file_id

def switch_to_existing_file(target_id, target_file, embedding_model, chunk_size=600, chunk_overlap=100):
    """Xử lý việc load lại file cũ từ DB"""
    final_filename = f"{target_id}_{target_file}"
    file_path = os.path.join("data", final_filename)
    db_path = os.path.join("vector_db", f"{final_filename}_index")

    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    
    if not os.path.exists(file_path):
        return False # Trả về False nếu lỗi
        
    if os.path.exists(db_path):
        vector_db = load_vector_db(db_path, embedding_model)
    else:
        chunks = load_and_split_document(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        vector_db = create_vector_db(chunks, embedding_model)
        save_vector_db(vector_db, db_path)
    
    st.session_state.retriever = get_retriever(vector_db, 3)
    st.session_state.file_processed = True
    st.session_state.current_file = target_file
    st.session_state.current_file_id = target_id
    st.session_state.messages = get_chat_history(target_id)
    return True
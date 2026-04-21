import streamlit as st
import os
import time
from data_access.document_loader import load_and_split_pdf
from data_access.vector_store import create_vector_db, get_retriever, save_vector_db, load_vector_db
from core.rag_pipeline import answer_query
from data_access.database import get_chat_history, insert_file_metadata, insert_message
from utils.file_process import process_new_uploaded_file, switch_to_existing_file

def main_chat_view(embedding_model, llm):
    """Khu vực màn hình trả lời và công cụ"""
    
    # 1. GIAO DIỆN UPLOAD
    if not st.session_state.get("file_processed"):
        st.markdown("### 📥 Tải lên tài liệu PDF để bắt đầu")
        uploaded_file = st.file_uploader(
            "Chọn tệp PDF của bạn", 
            type=("pdf"), 
            accept_multiple_files=False, 
            label_visibility="collapsed"
        )
        if uploaded_file:
            with st.status(f"Đang xử lý '{uploaded_file.name}'...", expanded=True) as status:
                process_new_uploaded_file(uploaded_file, embedding_model)
                status.update(label="Tài liệu đã sẵn sàng!", state="complete", expanded=False)
            st.rerun()

    # 2. GIAO DIỆN CHUYỂN ĐỔI FILE
    target_id = st.session_state.get("selected_file_id_to_load")
    if target_id and target_id != st.session_state.get("current_file_id"):
        target_file = st.session_state.selected_file_to_load
        with st.status(f"Đang tải lại '{target_file}'...", expanded=True):
            success = switch_to_existing_file(target_id, target_file, embedding_model) # Gọi hàm logic 
        if success:
            st.rerun()
        else:
            if st.session_state.get("current_file_id"):
                st.error(f"Không tìm thấy file '{target_file}' trong hệ thống!")
                st.session_state.selected_file_id_to_load = None

    # 3. THÔNG TIN FILE ĐANG CHAT
    if st.session_state.get("file_processed") and st.session_state.get("current_file"):
        st.info(f"🤖 Đang làm việc với file: **{st.session_state.current_file}**")
    st.divider()

    # 4. MÀN HÌNH HIỂN THỊ CHAT
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Ô NHẬP LIỆU
    if prompt := st.chat_input("Hỏi bất cứ điều gì về tài liệu..."):
        if not st.session_state.get("file_processed"):
            st.error("Vui lòng tải lên tài liệu trước khi đặt câu hỏi!")
            return
        
        file_id = st.session_state.get("current_file_id")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Lưu tin nhắn người dùng vào DB
        insert_message(file_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                # Gọi hàm và đợi lấy toàn bộ kết quả
                full_response = answer_query(prompt, st.session_state.retriever, llm)
                # In kết quả ra màn hình một lần duy nhất
                st.markdown(full_response)
                
        # Lưu vào DB và Session State
        insert_message(file_id, "assistant", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

def format_latex_for_streamlit(text):
    """
    Chuyển đổi các ký hiệu LaTeX lạ về chuẩn $ và $$ của Streamlit
    """
    # Thay thế \[ ... \] thành $$ ... $$
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    # Thay thế \( ... \) thành $ ... $
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    return text
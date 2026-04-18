import streamlit as st
import os
import time
from data_access.document_loader import load_and_split_pdf
from data_access.vector_store import create_vector_db, get_retriever, save_vector_db, load_vector_db
from core.rag_pipeline import answer_query
from data_access.database import get_chat_history, insert_file_metadata, insert_message

def main_chat_view(embedding_model, llm):
    """Khu vực màn hình trả lời và công cụ"""
    
    # 1. QUẢN LÝ HIỂN THỊ UPLOAD
    if not st.session_state.get("file_processed"):
        st.markdown("### 📥 Tải lên tài liệu PDF để bắt đầu")
        uploaded_file = st.file_uploader("Chọn tệp PDF của bạn", type=("pdf"), label_visibility="collapsed")
        
        if uploaded_file:
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            with st.status(f"Đang xử lý '{uploaded_file.name}'...", expanded=True) as status:
                
                # BƯỚC A: Lưu file tạm để đọc và băm nhỏ
                temp_path = os.path.join(data_dir, f"temp_{int(time.time())}.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                chunks = load_and_split_pdf(temp_path)
                
                # BƯỚC B: Lưu DB để lấy ID
                file_id = insert_file_metadata(uploaded_file.name, len(chunks))
                
                # BƯỚC C: Tạo tên file chính thức (ID_TênFile.pdf)
                final_filename = f"{file_id}_{uploaded_file.name}"
                file_path = os.path.join(data_dir, final_filename)
                db_path = os.path.join("vector_db", f"{final_filename}_index")
                
                # Đổi tên file tạm thành file chính thức
                os.rename(temp_path, file_path)
                
                # BƯỚC D: Tạo và lưu Vector DB
                vector_db = create_vector_db(chunks, embedding_model)
                save_vector_db(vector_db, db_path)
                
                st.session_state.retriever = get_retriever(vector_db, 2)
                st.session_state.file_processed = True
                st.session_state.current_file = uploaded_file.name
                st.session_state.current_file_id = file_id
                
                status.update(label="Tài liệu đã sẵn sàng!", state="complete", expanded=False)
            
            st.rerun()

    # 2. CHUYỂN ĐỔI FILE (Từ Sidebar)
    # So sánh dựa trên ID thay vì Tên
    if st.session_state.get("selected_file_id_to_load") and st.session_state.selected_file_id_to_load != st.session_state.get("current_file_id"):
        target_file = st.session_state.selected_file_to_load
        target_id = st.session_state.selected_file_id_to_load

        # Tái tạo lại đường dẫn có chứa ID
        final_filename = f"{target_id}_{target_file}"
        file_path = os.path.join("data", final_filename)
        db_path = os.path.join("vector_db", f"{final_filename}_index")
        
        if os.path.exists(file_path):
            with st.status(f"Đang tải lại '{target_file}'...", expanded=True) as status:
                if os.path.exists(db_path):
                    vector_db = load_vector_db(db_path, embedding_model)
                else:
                    chunks = load_and_split_pdf(file_path)
                    vector_db = create_vector_db(chunks, embedding_model)
                    save_vector_db(vector_db, db_path)
                
                st.session_state.retriever = get_retriever(vector_db, 2)
                st.session_state.file_processed = True
                st.session_state.current_file = target_file
                st.session_state.current_file_id = target_id

                st.session_state.messages = get_chat_history(target_id)
                
            st.rerun()
        else:
            is_new_chat = not st.session_state.get("current_file_id")
            if not is_new_chat:
                st.error(f"Không tìm thấy file '{target_file}' trong hệ thống!")
                st.session_state.selected_file_id_to_load = None
                st.session_state.selected_file_to_load = None

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
        file_id = st.session_state.get("current_file_id")
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Lưu tin nhắn người dùng vào DB
        insert_message(file_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.get("file_processed"):
            st.error("Vui lòng tải lên tài liệu trước khi đặt câu hỏi!")
            return

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                # Gọi hàm và đợi lấy toàn bộ kết quả
                full_response = answer_query(prompt, st.session_state.retriever, llm)
                    
                # In kết quả ra màn hình một lần duy nhất
                st.markdown(full_response)
                
        # Lưu vào DB và Session State
        insert_message(file_id, "assistant", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        # Rerun để đồng bộ UI
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
import streamlit as st
import os
import time
from data_access.document_loader import load_and_split_document
from data_access.vector_store import create_vector_db, get_retriever, save_vector_db, load_vector_db
from core.rag_pipeline import answer_query
from core.rag_pipeline import answer_query_crag
from data_access.database import get_chat_history, insert_file_metadata, insert_message
from utils.file_process import process_new_uploaded_file, switch_to_existing_file

def main_chat_view(embedding_model, llm):
    """Khu vực màn hình trả lời và công cụ"""

    # Thiết lập trạng thái ban đầu cho chế độ xử lý (RAG Thường hoặc CRAG)
    if "processing_mode" not in st.session_state:
        st.session_state.processing_mode = "RAG Thường"

    st.markdown("""
        <style>
        /* 1. Ép Popover trôi nổi ở dưới cùng, sát trên thanh chat_input */
        .st-key-mode_selector {
            position: fixed;
            top: 0.7rem; /* Cách đỉnh màn hình 1 đoạn vừa đủ qua header mặc định */
            z-index: 999999;
        }
        
        /* 2. Làm đẹp nút bấm bên ngoài: Bo tròn viên thuốc, thêm bóng đổ (Giống Gemini) */
        .st-key-mode_selector > div[data-testid="stPopover"] > button {
            border-radius: 30px !important; /* Bo tròn hoàn toàn */
            padding: 4px 16px !important; /* Thu gọn nút lại cho tinh tế */
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            background-color: var(--background-color) !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
            transition: all 0.2s ease-in-out;
        }
        
        /* Hiệu ứng nổi lên khi rê chuột vào */
        .st-key-mode_selector > div[data-testid="stPopover"] > button:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            border-color: rgba(128, 128, 128, 0.4) !important;
            transform: translateY(-2px);
        }

        /* 3. Chỉnh form bên trong menu xổ xuống cho bo tròn và mượt mà */
        .stPopoverBody {
            padding: 10px !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    rag_mode = st.session_state.processing_mode
    mode_icon = "✨" if "CRAG" in rag_mode else "⚡"
    
    with st.popover(f"{mode_icon} **{rag_mode}**", key="mode_selector"):
        
        # Nút lựa chọn 1: RAG Thường
        if st.button("⚡ **RAG Thường**", use_container_width=True):
            st.session_state.processing_mode = "RAG Thường"
            st.rerun() # Load lại trang ngay lập tức để nhận mode mới
            
        # Nút lựa chọn 2: CRAG
        if st.button("✨ **Recursive CRAG**", use_container_width=True):
            st.session_state.processing_mode = "Recursive CRAG (LangGraph)"
            st.rerun()
    
    # 1. GIAO DIỆN UPLOAD
    if not st.session_state.get("file_processed"):
        st.markdown("### 📥 Tải lên tài liệu PDF/DOCX để bắt đầu")
        uploaded_file = st.file_uploader(
            "Chọn tệp PDF/DOCX của bạn", 
            type=("pdf", "docx"), 
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
#     rag_mode = st.radio(
#     "Chế độ xử lý:",
#     ["RAG Thường", "Recursive CRAG (LangGraph)"],
#     horizontal=True,
#     help="CRAG sẽ tự động kiểm tra tài liệu và tìm kiếm lại nếu dữ liệu không liên quan."
# )

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
            with st.spinner("Đang xử lý..." if rag_mode == "RAG Thường" else "Đang kiểm duyệt và tối ưu dữ liệu..."):
                if rag_mode == "Recursive CRAG (LangGraph)":
                    full_response = answer_query_crag(prompt)
                else:
                    full_response = answer_query(prompt, st.session_state.retriever, llm)
            
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
import streamlit as st
import os
import time
import re
import html
from data_access.document_loader import load_and_split_document
from data_access.vector_store import create_vector_db, get_retriever, save_vector_db, load_vector_db
from core.rag_pipeline import answer_query
from core.rag_pipeline import answer_query_crag
from data_access.database import get_chat_history, insert_file_metadata, insert_message
from utils.file_process import process_new_uploaded_file, switch_to_existing_file


def _highlight_context(context: str, quote: str) -> str:
    safe_context = context or ""
    safe_quote = (quote or "").strip()

    if not safe_context:
        return ""
    if not safe_quote:
        return f"<div style='white-space: pre-wrap;'>{html.escape(safe_context)}</div>"

    match = re.search(re.escape(safe_quote), safe_context, flags=re.IGNORECASE)
    if not match:
        return f"<div style='white-space: pre-wrap;'>{html.escape(safe_context)}</div>"

    start, end = match.span()
    return (
        "<div style='white-space: pre-wrap;'>"
        + html.escape(safe_context[:start])
        + f"<mark>{html.escape(safe_context[start:end])}</mark>"
        + html.escape(safe_context[end:])
        + "</div>"
    )


def _render_citations(citations):
    if not citations:
        return

    st.markdown("##### 📚 Nguồn tham chiếu")
    for index, citation in enumerate(citations, start=1):
        source_id = citation.get("source_id", f"S{index}")
        file_name = citation.get("file_name", "Unknown")
        page = citation.get("page")
        position = citation.get("position", {}) or {}
        chunk_id = citation.get("chunk_id", "N/A")
        quote = citation.get("quote", "")
        context = citation.get("context", "")

        start_pos = position.get("start")
        end_pos = position.get("end")
        page_text = f"trang {page}" if isinstance(page, int) else "trang N/A"
        if isinstance(start_pos, int) and isinstance(end_pos, int):
            pos_text = f"vị trí {start_pos}-{end_pos}"
        else:
            pos_text = "vị trí N/A"

        with st.expander(f"[{source_id}] {file_name} • {page_text} • chunk {chunk_id}"):
            st.caption(f"{page_text} • {pos_text}")
            if quote:
                st.markdown(f"**Đoạn được dùng:** _{quote}_")

            highlighted = _highlight_context(context, quote)
            if highlighted:
                st.markdown(highlighted, unsafe_allow_html=True)


def main_chat_view(embedding_model, llm):
    """Khu vực màn hình trả lời và công cụ"""

    # Thiết lập trạng thái ban đầu cho chế độ xử lý (RAG Thường hoặc CRAG)
    if "processing_mode" not in st.session_state:
        st.session_state.processing_mode = "RAG Thường"

    rag_mode = st.session_state.processing_mode
    mode_icon = ":material/auto_awesome:" if "CRAG" in rag_mode else ":material/bolt:"
    
    with st.popover(f"{mode_icon} **{rag_mode}**", key="mode_selector"):
        
        # Nút lựa chọn 1: RAG Thường
        if st.button("**RAG Thường**", 
                     icon=":material/bolt:", 
                     key="btn_mode_rag", 
                     use_container_width=True):
            st.session_state.processing_mode = "RAG Thường"
            st.rerun() # Load lại trang ngay lập tức để nhận mode mới
            
        # Nút lựa chọn 2: CRAG
        if st.button("**Recursive CRAG**", 
                     icon=":material/auto_awesome:", 
                     key="btn_mode_crag", 
                     use_container_width=True):
            st.session_state.processing_mode = "Recursive CRAG"
            st.rerun()
    
    # 1. GIAO DIỆN UPLOAD
    if not st.session_state.get("file_processed"):
        st.markdown("### :material/upload_file: Tải lên tài liệu PDF/DOCX để bắt đầu")
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
        st.info(f":material/smart_toy: Đang làm việc với file: **{st.session_state.current_file}**")
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

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_citations(message.get("citations", []))

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
                if rag_mode == "Recursive CRAG":
                    full_response = answer_query_crag(prompt, file_id)
                else:
                    full_response = answer_query(prompt,file_id, st.session_state.retriever, llm)

            if isinstance(full_response, dict):
                answer_text = full_response.get("answer", "")
                citations = full_response.get("citations", [])
            else:
                answer_text = str(full_response)
                citations = []

            st.markdown(answer_text)
            _render_citations(citations)
                
        # Lưu vào DB và Session State
        insert_message(file_id, "assistant", answer_text)
        st.session_state.messages.append({"role": "assistant", "content": answer_text, "citations": citations})
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
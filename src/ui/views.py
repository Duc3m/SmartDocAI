import streamlit as st
import os
import time
import re
import html
import unicodedata
from data_access.document_loader import load_and_split_document
from data_access.vector_store import create_vector_db, get_retriever, save_vector_db, load_vector_db
from core.rag_pipeline import answer_query
from core.rag_pipeline import answer_query_crag
from data_access.database import get_chat_history, insert_file_metadata, insert_message
from utils.file_process import process_new_uploaded_file, switch_to_existing_file


def _normalize_display_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()


def _find_exact_or_whitespace_span(context: str, quote: str):
    exact_match = re.search(re.escape(quote), context, flags=re.IGNORECASE)
    if exact_match:
        return exact_match.span()

    quote_tokens = re.findall(r"\S+", quote)
    if not quote_tokens:
        return None

    whitespace_flexible_pattern = r"\s+".join(re.escape(token) for token in quote_tokens)
    soft_match = re.search(whitespace_flexible_pattern, context, flags=re.IGNORECASE)
    if soft_match:
        return soft_match.span()
    return None


def _normalize_for_loose_match(text: str):
    normalized_chars = []
    index_map = []
    prev_is_space = False

    for original_index, char in enumerate(text):
        category = unicodedata.category(char)

        if char.isspace():
            if not prev_is_space:
                normalized_chars.append(" ")
                index_map.append(original_index)
                prev_is_space = True
            continue

        if category.startswith("P"):
            continue

        normalized_chars.append(char.lower())
        index_map.append(original_index)
        prev_is_space = False

    raw_normalized = "".join(normalized_chars)
    if not raw_normalized.strip():
        return "", []

    left_trim = len(raw_normalized) - len(raw_normalized.lstrip())
    right_boundary = len(raw_normalized.rstrip())
    normalized_text = raw_normalized[left_trim:right_boundary]
    trimmed_index_map = index_map[left_trim:right_boundary]
    return normalized_text, trimmed_index_map


def _find_loose_span(context: str, quote: str):
    normalized_context, context_map = _normalize_for_loose_match(context)
    normalized_quote, _ = _normalize_for_loose_match(quote)

    if not normalized_context or not normalized_quote:
        return None

    start_index = normalized_context.find(normalized_quote)
    if start_index == -1:
        return None

    end_index = start_index + len(normalized_quote) - 1
    if start_index >= len(context_map) or end_index >= len(context_map):
        return None

    original_start = context_map[start_index]
    original_end = context_map[end_index] + 1
    return original_start, original_end


def _find_best_span(context: str, quote: str):
    span = _find_exact_or_whitespace_span(context, quote)
    if span:
        return span
    return _find_loose_span(context, quote)


def _highlight_context(context: str, quote: str) -> str:
    safe_context = _normalize_display_text(context or "")
    safe_quote = _normalize_display_text((quote or "").strip())

    if not safe_context:
        return ""
    if not safe_quote:
        return f"<div style='white-space: pre-wrap;'>{html.escape(safe_context)}</div>"

    span = _find_best_span(safe_context, safe_quote)
    if not span:
        return f"<div style='white-space: pre-wrap;'>{html.escape(safe_context)}</div>"

    start, end = span
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
        if isinstance(file_name, str) and file_name.startswith("temp_") and st.session_state.get("current_file"):
            file_name = st.session_state.get("current_file")

        doc_type = str(citation.get("doc_type") or "").lower()
        if not doc_type and isinstance(file_name, str) and "." in file_name:
            doc_type = f".{file_name.rsplit('.', 1)[-1].lower()}"

        page = citation.get("page")
        position = citation.get("position", {}) or {}
        chunk_id = citation.get("chunk_id", "N/A")
        quote = _normalize_display_text(citation.get("quote", ""))
        context = _normalize_display_text(citation.get("context", ""))

        start_pos = position.get("start")
        end_pos = position.get("end")

        page_text = None
        if isinstance(page, int):
            page_text = f"trang {page}"
        elif doc_type == ".pdf":
            page_text = "trang N/A"

        if isinstance(start_pos, int) and isinstance(end_pos, int):
            pos_text = f"vị trí {start_pos}-{end_pos}"
        else:
            pos_text = "vị trí N/A"

        header_parts = [f"[{source_id}] {file_name}"]
        if page_text:
            header_parts.append(page_text)
        header_parts.append(f"chunk {chunk_id}")

        with st.expander(" • ".join(header_parts)):
            caption_parts = []
            if page_text:
                caption_parts.append(page_text)
            caption_parts.append(pos_text)
            st.caption(" • ".join(caption_parts))
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
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200

    rag_mode = st.session_state.processing_mode
    mode_icon = ":material/auto_awesome:" if "CRAG" in rag_mode else ":material/bolt:"
    is_new_chat_screen = not st.session_state.get("current_file_id")

    if is_new_chat_screen:
        control_col_mode, control_col_chunk = st.columns([1, 2], vertical_alignment="center")

        with control_col_mode:
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

        with control_col_chunk:
            chunk_col_1, chunk_col_2 = st.columns(2)
            with chunk_col_1:
                chunk_size_value = st.number_input(
                    "Chunk size",
                    min_value=100,
                    max_value=2000,
                    step=50,
                    value=int(st.session_state.chunk_size),
                    key="chunk_size_input",
                    help="Số ký tự tối đa cho mỗi chunk."
                )
            with chunk_col_2:
                chunk_overlap_value = st.number_input(
                    "Chunk overlap",
                    min_value=0,
                    max_value=200,
                    step=10,
                    value=int(st.session_state.chunk_overlap),
                    key="chunk_overlap_input",
                    help="Số ký tự chồng lấn giữa các chunk liên tiếp."
                )

            st.session_state.chunk_size = int(chunk_size_value)
            st.session_state.chunk_overlap = int(chunk_overlap_value)
    else:
        with st.popover(f"{mode_icon} **{rag_mode}**", key="mode_selector"):
            if st.button("**RAG Thường**", 
                        icon=":material/bolt:", 
                        key="btn_mode_rag", 
                        use_container_width=True):
                st.session_state.processing_mode = "RAG Thường"
                st.rerun()

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
                process_new_uploaded_file(
                    uploaded_file,
                    embedding_model,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
                status.update(label="Tài liệu đã sẵn sàng!", state="complete", expanded=False)
            st.rerun()

    # 2. GIAO DIỆN CHUYỂN ĐỔI FILE
    target_id = st.session_state.get("selected_file_id_to_load")
    if target_id and target_id != st.session_state.get("current_file_id"):
        target_file = st.session_state.selected_file_to_load
        with st.status(f"Đang tải lại '{target_file}'...", expanded=True):
            success = switch_to_existing_file(
                target_id,
                target_file,
                embedding_model,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            ) # Gọi hàm logic 
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
        insert_message(file_id, "assistant", answer_text, citations=citations)
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
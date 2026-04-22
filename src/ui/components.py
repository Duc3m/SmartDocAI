import streamlit as st
import os
import shutil
from data_access.database import delete_chat_history, get_all_files, delete_file_record

def perform_delete(file_id, filename, is_current):
    """Thực thi việc xóa vật lý và dọn dẹp bộ nhớ cho 1 file cụ thể."""
    # 1. Xóa trong Database
    delete_file_record(file_id)
    delete_chat_history(file_id)
    
    # 2. Xóa File PDF và Thư mục FAISS
    final_filename = f"{file_id}_{filename}"
    
    pdf_path = os.path.join("data", final_filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
    db_path = os.path.join("vector_db", f"{final_filename}_index")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    # 3. Dọn dẹp Session nếu đang mở đúng file bị xóa
    if is_current:
        st.session_state.pop("current_file", None)
        st.session_state.pop("current_file_id", None)
        st.session_state.pop("file_processed", None)
        st.session_state.pop("retriever", None)
        st.session_state.messages = []

@st.dialog("⚠️ Xác nhận xóa")
def confirm_delete_dialog(mode="single", file_id=None, filename=None, is_current=False, all_files=None):
    """
    Hộp thoại dùng chung cho cả 2 trường hợp: Xóa 1 file và Xóa tất cả.
    """
    # Hiển thị câu cảnh báo tùy theo chế độ
    if mode == "single":
        st.write(f"Bạn có chắc chắn muốn xóa tài liệu **{filename}** không?")
    else:
        st.write("Bạn có chắc chắn muốn xóa **TẤT CẢ** tài liệu và lịch sử chat không?")
        
    st.write("Hành động này không thể hoàn tác!")
    
    # Hai nút bấm xác nhận
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hủy bỏ", use_container_width=True):
            st.rerun() # Đóng popup
            
    with col2:
        if st.button("Xóa ngay", type="primary", use_container_width=True):
            if mode == "single":
                # Gọi hàm xóa 1 file
                perform_delete(file_id, filename, is_current)
            elif mode == "all" and all_files:
                # Vòng lặp gọi hàm xóa cho toàn bộ file
                for f in all_files:
                    is_curr = st.session_state.get("current_file_id") == f['id']
                    perform_delete(f['id'], f['filename'], is_curr)
            
            st.rerun() # Load lại trang sau khi xóa xong

def render_sidebar():
    """Hiển thị quản lý file và lịch sử chat bên trái"""
    with st.sidebar:
        st.markdown("# :material/rocket_launch: SmartDoc AI")
        st.subheader("Intelligent Document Q&A")
        
        # --- 1. NÚT TẠO CUỘC TRÒ CHUYỆN MỚI ---
        is_new_chat = not st.session_state.get("current_file_id")
        new_chat_btn_type = "primary" if is_new_chat else "secondary"

        if st.button("Tạo cuộc trò chuyện mới", 
                     key="btn_new_chat", 
                     use_container_width=True, 
                     type=new_chat_btn_type,
                     icon=":material/add_box:"):
            st.session_state.messages = []
            st.session_state.pop("current_file", None)
            st.session_state.pop("current_file_id", None)
            st.session_state.pop("file_processed", None)
            st.session_state.pop("retriever", None)
            st.session_state.pop("selected_file_to_load", None)
            st.session_state.pop("selected_file_id_to_load", None)
            st.filter_type = "Tất cả" # Reset filter về mặc định
            st.rerun()
            
        st.divider()

        # Lấy danh sách file trước để UI xử lý logic hiển thị
        files = get_all_files()

        # --- 2. TIÊU ĐỀ VÀ NÚT XÓA TẤT CẢ (Nằm ngang) ---
        col_title, col_btn = st.columns([4, 1], vertical_alignment="center")
        with col_title:
            st.markdown("### :material/folder_open: Lịch sử tài liệu")
            
        with col_btn:
            if files: # Chỉ hiện nút xóa tất cả nếu có ít nhất 1 file
                if st.button("\u200B", 
                             icon=":material/delete_sweep:", 
                             help="Xóa TẤT CẢ tài liệu", 
                             key="btn_del_all"):
                    confirm_delete_dialog(mode="all", all_files=files)

        # --- 3. HIỂN THỊ DANH SÁCH TỪNG FILE ---
        if not files:
            st.info("Chưa có tài liệu nào trong hệ thống.")
        else:
            # ---> BẮT ĐẦU PHẦN THÊM MỚI: Radio lọc tài liệu <---
            filter_type = st.radio(
                "Lọc tài liệu:",
                ["Tất cả", "PDF", "DOCX"],
                horizontal=True,
                label_visibility="collapsed"
            )

            # Phân loại danh sách file dựa theo lựa chọn ở Radio
            if filter_type == "PDF":
                filtered_files = [f for f in files if f['filename'].lower().endswith('.pdf')]
            elif filter_type == "DOCX":
                filtered_files = [f for f in files if f['filename'].lower().endswith(('.docx', '.doc'))]
            else:
                filtered_files = files
            # ---> KẾT THÚC PHẦN THÊM MỚI <---

            # Kiểm tra xem sau khi lọc có còn file nào không
            if not filtered_files:
                st.info(f"Không có tài liệu {filter_type} nào.")
            else:
                # ---> LƯU Ý: Vòng lặp đổi từ `files` thành `filtered_files` <---
                for file in filtered_files:
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        is_current = st.session_state.get("current_file_id") == file['id']
                        
                        display_name = file['filename']
                        ext = display_name.split('.')[-1]
                        btn_key = f"{ext}_{file['id']}"

                        if len(display_name) > 20:
                            display_name = display_name[:17] + "..."
                            
                        # BỘ NHẬN DIỆN ICON CHUẨN GOOGLE MATERIAL
                        if ext == "pdf":
                            file_icon = ":material/picture_as_pdf:" 
                        elif ext in ["docx", "doc"]:
                            file_icon = ":material/description:"
                        else:
                            file_icon = ":material/insert_drive_file:"
                            
                        btn_type = "primary" if is_current else "secondary"
                        
                        if st.button(
                            display_name,
                            icon=file_icon, 
                            key=btn_key, 
                            help=f"{file['filename']} (ID: {file['id']})",
                            use_container_width=True,
                            type=btn_type
                        ):
                            st.session_state.selected_file_to_load = file['filename']
                            st.session_state.selected_file_id_to_load = file['id']
                            st.rerun()
                            
                    with col2:
                        # Nút bấm mở Dialog Xóa 1 File
                        if st.button("\u200B", 
                                     icon=":material/close:", 
                                     key=f"del_btn_{file['id']}", 
                                     help="Xóa tài liệu này"):
                            confirm_delete_dialog(mode="single", file_id=file['id'], filename=file['filename'], is_current=is_current)
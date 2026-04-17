import streamlit as st
import os
import shutil
from data_access.database import delete_chat_history, get_all_files, delete_file_record

def render_sidebar():
    """Hiển thị quản lý file và lịch sử chat bên trái"""
    with st.sidebar:
        st.title("🚀 SmartDoc AI")
        st.subheader("Intelligent Document Q&A System")
        
        # --- 1. NÚT TẠO CUỘC TRÒ CHUYỆN MỚI ---
        # Đặt ở vị trí đầu tiên, dùng type="primary" để có màu nổi bật
        is_new_chat = not st.session_state.get("current_file_id")
        new_chat_btn_type = "primary" if is_new_chat else "secondary"

        if st.button("Tạo cuộc trò chuyện mới", use_container_width=True, type=new_chat_btn_type):
            st.session_state.messages = []
            st.session_state.pop("current_file", None)
            st.session_state.pop("current_file_id", None)
            st.session_state.pop("file_processed", None)
            st.session_state.pop("retriever", None)
            st.session_state.pop("selected_file_to_load", None)
            st.session_state.pop("selected_file_id_to_load", None)
            st.rerun()
            
        st.divider()

        # --- 2. QUẢN LÝ FILE ---
        st.title("📁 Lịch sử tài liệu")
        files = get_all_files()
        
        if not files:
            st.info("Chưa có tài liệu nào trong hệ thống.")
        else:
            for file in files:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = st.session_state.get("current_file_id") == file['id']
                    
                    # CẮT NGẮN TÊN FILE: Giới hạn 20 ký tự để không bị tràn nút bấm
                    display_name = file['filename']
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."
                        
                    btn_type = "primary" if is_current else "secondary"
                    btn_label = f"📄 {display_name}"
                    
                    # Thêm help=file['filename'] để khi rê chuột vào sẽ thấy tên đầy đủ
                    if st.button(
                        btn_label, 
                        key=f"sel_{file['id']}", 
                        help=f"{file['filename']} (ID: {file['id']})",
                        use_container_width=True,
                        type=btn_type
                    ):
                        st.session_state.selected_file_to_load = file['filename']
                        st.session_state.selected_file_id_to_load = file['id']
                        st.rerun()
                        
                with col2:
                    if st.button("❌", key=f"del_{file['id']}", help="Xóa tài liệu này"):
                        delete_file_record(file['id'])
                        delete_chat_history(file['id'])
                        
                        # LOGIC MỚI: Nối ID vào tên file để xóa đúng file vật lý
                        final_filename = f"{file['id']}_{file['filename']}"
                        
                        pdf_path = os.path.join("data", final_filename)
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                            
                        db_path = os.path.join("vector_db", f"{final_filename}_index")
                        if os.path.exists(db_path):
                            shutil.rmtree(db_path)
                        
                        if is_current:
                            st.session_state.pop("current_file", None)
                            st.session_state.pop("current_file_id", None)
                            st.session_state.pop("file_processed", None)
                            st.session_state.pop("retriever", None)
                            st.session_state.messages = []
                        st.rerun()
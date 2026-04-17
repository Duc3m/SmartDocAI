import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from models.embedding_config import get_embedding_model
from models.llm_config import get_llm
from ui.components import render_sidebar
from ui.views import main_chat_view
from data_access.database import init_db

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide"
)

# Ẩn dải màu trang trí trên cùng của Streamlit
st.markdown("""
    <style>
        [data-testid="stDecoration"] {
            display: none;
        }
        
    </style>
""", unsafe_allow_html=True)

def main():
    # Khởi tạo Database SQLite
    init_db()

    # Tải mô hình
    embedding_model = get_embedding_model()
    llm = get_llm()
    
    # Cấu trúc giao diện mới: Trái (Sidebar) và Phải (Main)
    render_sidebar()
    
    if llm is None:
        st.error("Không thể kết nối với Ollama. Vui lòng kiểm tra ứng dụng Ollama đã chạy chưa!")
        return

    # Khu vực chính
    main_chat_view(embedding_model, llm)

if __name__ == "__main__":
    main()
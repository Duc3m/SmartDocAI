import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.styles import inject_custom_css
import streamlit as st
from models.embedding_config import get_embedding_model
from models.llm_config import get_llm
from ui.components import render_sidebar
from ui.views import main_chat_view
from data_access.database import init_db

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon=":material/robot:",
    layout="wide"
)

def main():
    # Khởi tạo Database SQLite
    init_db()

    # Tải mô hình
    embedding_model = get_embedding_model()
    llm = get_llm()
    if llm is None:
        st.error("Không thể kết nối với Ollama. Vui lòng kiểm tra ứng dụng Ollama đã chạy chưa!")
        return
    
    # Render giao diện và custom CSS
    inject_custom_css()
    render_sidebar()
    main_chat_view(embedding_model, llm)

if __name__ == "__main__":
    main()
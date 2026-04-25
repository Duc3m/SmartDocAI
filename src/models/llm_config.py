import os
import streamlit as st
from langchain_ollama import ChatOllama

@st.cache_resource
def get_llm():
    """
    Khởi tạo LLM bằng Ollama
    """
    # Lấy thông tin cấu hình từ biến môi trường
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Cấu hình các tham số tối ưu cho tiếng Việt và hiệu suất
    try:
        llm = ChatOllama(
            model=model_name,      # Tên mô hình
            base_url=base_url,     # URL của Ollama server
            temperature=0.3,       # Độ sáng tạo của câu trả lời
            top_p=0.9,             # Giới hạn tập từ vựng
            num_thread=8,          # Sử dụng đa luồng để tăng tốc độ xử lý
            num_ctx=2048,          # Giới hạn kích thước context
            repeat_penalty=1.1     # Hạn chế việc mô hình bị lặp từ trong câu trả lời
        )
        return llm
    
    except Exception as e:
        print(f"Lỗi khi kết nối với Ollama: {e}")
        return None
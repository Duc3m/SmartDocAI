import os
import streamlit as st
from langchain_community.chat_models import ChatOllama

@st.cache_resource
def get_llm():
    """
    Khởi tạo và cấu hình mô hình Qwen2.5:7b thông qua Ollama[cite: 102, 222].
    """
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        llm = ChatOllama(
            model=model_name,              # Mô hình Qwen thế hệ mới tối ưu cho tiếng Việt [cite: 93, 224]
            base_url=base_url,            # URL của Ollama server [cite: 102]
            temperature=0.3,               # Độ sáng tạo của câu trả lời [cite: 373]
            top_p=0.9,                     # Nucleus sampling để lọc các từ có xác suất thấp [cite: 373]
            num_thread=8,
            num_gpu=100,
            num_ctx=2048,
            repeat_penalty=1.1             # Hạn chế việc mô hình bị lặp từ trong câu trả lời [cite: 374]
        )
        return llm
    except Exception as e:
        # Bạn có thể thêm logging ở đây nếu cần [cite: 574-575]
        print(f"Lỗi khi kết nối với Ollama: {e}")
        return None
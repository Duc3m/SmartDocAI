from langchain_community.vectorstores import FAISS
import os

def create_vector_db(chunks, embedding_model):
    """
    Tạo và trả về Vector Store từ các đoạn văn bản đã băm nhỏ [cite: 211-213].
    """
    # Tạo FAISS index từ danh sách chunks và mô hình embedding [cite: 213]
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db

def save_vector_db(vector_db, folder_path="vector_db/faiss_index"):
    """
    Lưu Vector Store xuống ổ cứng để tái sử dụng mà không cần tạo lại.
    """
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")
    vector_db.save_local(folder_path)

def load_vector_db(folder_path, embedding_model):
    """
    Tải Vector Store đã lưu từ ổ cứng.
    """
    return FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)

def get_retriever(vector_db, k=6):
    """
    Tạo retriever với MMR để tăng độ bao phủ ngữ cảnh, đặc biệt cho câu hỏi tổng quan.
    """
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )
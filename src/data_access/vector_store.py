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

def get_retriever(vector_db, k=3):
    """
    Tạo bộ truy xuất (retriever) để tìm top k đoạn văn bản liên quan nhất [cite: 214-219].
    """
    return vector_db.as_retriever(
        search_type="similarity", # Tìm kiếm dựa trên độ tương đồng vector [cite: 218]
        search_kwargs={"k": k}    # Lấy ra 3 kết quả tốt nhất mặc định [cite: 219]
    )
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import torch

@st.cache_resource
def get_embedding_model():
    """
    Khởi tạo mô hình embedding đa ngôn ngữ từ HuggingFace[cite: 189, 340].
    """
    # Tên mô hình state-of-the-art cho tìm kiếm đa ngôn ngữ [cite: 202, 349]
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chuẩn hóa vector về độ dài 1 để tính cosine similarity chính xác hơn [cite: 197, 350]
    encode_kwargs = {'normalize_embeddings': True}
    
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    return embedder
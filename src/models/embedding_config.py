import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import torch

@st.cache_resource
def get_embedding_model():
    """
    Khởi tạo model embedding
    """
    # Lấy tên mô hình từ biến môi trường
    model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tham số để chuẩn hóa embedding
    encode_kwargs = {'normalize_embeddings': True}
    
    # Cấu hình model embedding
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    return embedder
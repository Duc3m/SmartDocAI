import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.timer import time_it

@time_it
def load_and_split_document(file_path):
    """
    Đọc nội dung file và chia nhỏ thành các chunks.
    """
    # Xác định đuôi file để chọn đúng Loader
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        loader = PDFPlumberLoader(file_path)
    elif ext in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Định dạng {ext} chưa được hỗ trợ!")

    documents = loader.load()
    
    # Khởi tạo Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    # Mỗi chunk tối đa 600 ký tự
        chunk_overlap=100,  # 100 ký tự trùng lặp giữa các đoạn liên tiếp
        add_start_index=True # Lưu vị trí bắt đầu của đoạn để truy vết sau này
    )
    
    # Chia nhỏ tài liệu thành các chunks
    chunks = text_splitter.split_documents(documents)

    # Gắn metadata cho mỗi chunk để dễ dàng truy vết nguồn gốc sau này
    file_name = os.path.basename(file_path)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_id"] = index
        chunk.metadata["file_name"] = file_name
        start_index = chunk.metadata.get("start_index")
        if isinstance(start_index, int):
            chunk.metadata["char_start"] = start_index
            chunk.metadata["char_end"] = start_index + len(chunk.page_content)

    return chunks
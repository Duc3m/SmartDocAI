import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.timer import time_it

@time_it
def load_and_split_document(file_path, chunk_size=1000, chunk_overlap=200, source_name=None):
    """
    Đọc nội dung PDF và chia nhỏ thành các chunks.
    """
    # 1. Xác định đuôi file để chọn đúng Loader
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        loader = PDFPlumberLoader(file_path)
    elif ext in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Định dạng {ext} chưa được hỗ trợ!")

    documents = loader.load()
    
    # 2. Cấu hình bộ cắt văn bản [cite: 173-178]
    safe_chunk_size = max(100, int(chunk_size))
    safe_chunk_overlap = max(0, min(int(chunk_overlap), safe_chunk_size - 1))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=safe_chunk_size,
        chunk_overlap=safe_chunk_overlap,
        add_start_index=True # Lưu vị trí bắt đầu của đoạn để truy vết sau này
    )
    
    # 3. Thực hiện chia nhỏ [cite: 179-180]
    chunks = text_splitter.split_documents(documents)

    file_name = source_name or os.path.basename(file_path)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_id"] = index
        chunk.metadata["file_name"] = file_name
        chunk.metadata["source"] = file_name
        chunk.metadata["doc_type"] = ext
        start_index = chunk.metadata.get("start_index")
        if isinstance(start_index, int):
            chunk.metadata["char_start"] = start_index
            chunk.metadata["char_end"] = start_index + len(chunk.page_content)

    return chunks
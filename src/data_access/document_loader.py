from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    Đọc nội dung PDF và chia nhỏ thành các chunks [cite: 142-143, 163-164].
    """
    # 1. Load tài liệu bằng PDFPlumber để giữ layout và bảng biểu 
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    # 2. Cấu hình bộ cắt văn bản [cite: 173-178]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    # Mỗi đoạn tối đa 600 ký tự [cite: 182]
        chunk_overlap=100,  # 100 ký tự trùng lặp giữa các đoạn liên tiếp [cite: 183-184]
        add_start_index=True # Lưu vị trí bắt đầu của đoạn để truy vết sau này
    )
    
    # 3. Thực hiện chia nhỏ [cite: 179-180]
    chunks = text_splitter.split_documents(documents)
    return chunks
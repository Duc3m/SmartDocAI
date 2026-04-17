from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1. Khởi tạo lại model embedding (Đảm bảo model name giống hệt lúc bạn tạo DB)
# Lưu ý: Sửa lại tên model nếu bạn đang dùng model khác
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

# 2. Đường dẫn trỏ chính xác vào thư mục chứa index.faiss và index.pkl của bạn
db_path = "vector_db/faiss_index"

try:
    # 3. Load DB
    vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    docstore = vector_db.docstore._dict

    print(f"✅ Tải thành công! Tổng số chunks đang có trong DB: {len(docstore)}")
    print("=" * 50)


    with open('vectordb_chunks.txt', 'a', encoding='utf-8') as file:
        for i, (doc_id, doc) in enumerate(list(docstore.items())):
            file.write(f"--- Chunk thứ {i+1} ---")
            file.write(f"📝 Metadata : {doc.metadata}")
            file.write(f"📄 Nội dung :\n{doc.page_content}\n")
            file.write("-" * 50)

except Exception as e:
    print(f"❌ Có lỗi xảy ra khi load DB: {e}")
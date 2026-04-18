# 🚀 SmartDoc AI - Intelligent Document Q&A System

SmartDoc AI là hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật **RAG (Retrieval-Augmented Generation)**. Ứng dụng cho phép người dùng tải lên các tài liệu PDF và tương tác trực tiếp với nội dung văn bản thông qua giao diện trò chuyện tự nhiên, được vận hành hoàn toàn bởi các mô hình ngôn ngữ lớn (LLMs) mã nguồn mở chạy cục bộ.

## ✨ Tính năng nổi bật

* **Bảo mật tuyệt đối (Privacy-First):** Toàn bộ quá trình xử lý tài liệu và sinh văn bản được thực hiện Local (hoặc trên Google Colab riêng) thông qua Ollama. Không có bất kỳ dữ liệu nào được gửi cho bên thứ ba.
* **Hỗ trợ Đa mô hình:** Dễ dàng chuyển đổi giữa các phiên bản mô hình thông minh thông qua cấu hình biến môi trường (Ví dụ: Qwen2.5 1.5B, 3B, 7B).
* **Trích xuất Context thông minh:** Tự động đọc lướt cấu trúc tài liệu khi tải lên, kết hợp với tìm kiếm Vector (FAISS) để khắc phục điểm yếu "mất ngữ cảnh tổng quan" của RAG truyền thống.
* **Giao diện thân thiện:** Được xây dựng bằng Streamlit, hỗ trợ render hiển thị các công thức toán học (LaTeX) và lưu trữ lịch sử trò chuyện.

## 🛠️ Công nghệ sử dụng

* **Giao diện (Frontend):** Streamlit
* **Điều phối & RAG Framework:** LangChain, LangChain Community
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Engine:** Ollama (Hỗ trợ họ mô hình Qwen2.5)
* **Xử lý tài liệu:** PyPDF, RecursiveCharacterTextSplitter

## 📂 Cấu trúc thư mục

```text
SmartDocAI/
├── src/                    # Chứa mã nguồn chính của ứng dụng
│   ├── core/               # Chứa logic RAG, Pipeline và Prompts
│   ├── data_access/        # Xử lý đọc PDF và Vector Store (FAISS)
│   ├── models/             # Cấu hình kết nối Ollama và Embeddings
│   ├── ui/                 # Xử lý giao diện Streamlit (views.py)
│   └── utils/              # Các hàm hỗ trợ tiện ích
├── data/                   # Thư mục lưu trữ file PDF và Vector DB nội bộ
├── .env                    # Biến môi trường (Tên Model, Base URL, v.v.)
├── app.py                  # Điểm khởi chạy ứng dụng chính
├── requirements.txt        # Danh sách thư viện Python cần thiết
└── README.md               # Tài liệu hướng dẫn
```

## ⚙️ Hướng dẫn cài đặt và chạy Cục bộ (Local)

### Yêu cầu tiên quyết

* Python 3.9+  
* Đã cài đặt Ollama trên máy.

### Các bước thực hiện

#### Clone repository:

```Bash
git clone https://github.com/Duc3m/SmartDocAI.git
cd SmartDocAI
```

#### Cài đặt thư viện:

```bash
make setup
```

#### Tải Model từ Ollama:

```bash
make pull-model
```

#### Khởi động Ollama và chạy model (sử dụng một cửa sổ terminal riêng)

```bash
make run-model
```

#### Cấu hình môi trường:

Tạo file .env ở thư mục gốc (nếu chưa có) và thiết lập mô hình bạn muốn dùng:

```bash
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_NAME="qwen2.5:7b" #qwen2.5:1.5b nếu không có gpu mạnh
OLLAMA_BASE_URL="http://localhost:11434"
```

#### Chạy ứng dụng:

```bash
make run
```

## ☁️ Hướng dẫn triển khai trên Google Colab

Dự án này được tối ưu để có thể tận dụng sức mạnh GPU (T4 16GB VRAM) miễn phí của Google Colab.

1. Upload toàn bộ source code (hoặc dùng git clone) lên Google Colab.
2. Chọn Runtime -> Change runtime type -> T4 GPU.
3. Chạy đoạn mã khởi tạo sau trong Colab Notebook:

```python
# Cài đặt Ollama và Thư viện
!curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
!make setup
!pip install pyngrok

# Khởi động Ollama ngầm
import subprocess, time, os
with open(os.devnull, 'w') as fnull:
    subprocess.Popen(['ollama', 'serve'], stdout=fnull, stderr=fnull)
time.sleep(10)
!ollama pull qwen2.5:7b

# Tạo file .env cho Colab
!echo 'LLM_MODEL_NAME="qwen2.5:7b"' > .env
!echo 'OLLAMA_BASE_URL="http://localhost:11434"' >> .env
!echo 'NUM_GPU=100' >> .env

# Chạy ứng dụng qua Ngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN") # Thay Token của bạn vào đây
public_url = ngrok.connect(8501).public_url
print(f"Truy cập ứng dụng tại: {public_url}")

!make run
```
## 👥 Đội ngũ phát triển

* Trần Đức Em
* Hoàng Dũng
* Trầm Quang Dũng
* Đoàn Minh Đức
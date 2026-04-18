.PHONY: help setup pull-model run clean

# Định nghĩa các biến môi trường
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Lệnh mặc định khi chỉ gõ 'make'
help:
	@echo "Các lệnh hỗ trợ trong dự án SmartDoc AI:"
	@echo "  make setup       - Tạo virtual environment và cài đặt thư viện"
	@echo "  make pull-model  - Tải mô hình Qwen2.5:7b qua Ollama"
	@echo "  make run         - Khởi chạy ứng dụng Streamlit"
	@echo "  make clean       - Xóa môi trường ảo và các file cache"
	@echo "  make all         - Chạy setup và pull-model cùng lúc"

# 1. Cài đặt môi trường và thư viện
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "=> Đã cài đặt xong môi trường và thư viện!"

# 2. Tải mô hình AI (yêu cầu máy đã cài Ollama)
pull-model:
	ollama serve
	ollama pull qwen2.5:7b
	@echo "=> Đã tải xong model Qwen2.5:7b!"

run-model:
	ollama serve
	ollama run qwen2.5:7b

stop-model:
	ollama stop qwen2.5:7b

# 3. Khởi chạy ứng dụng
run:
	PYTHONPATH=src $(VENV)/bin/streamlit run app.py --server.headless true

# 4. Dọn dẹp project (xóa cache và venv)
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	@echo "=> Đã dọn dẹp project sạch sẽ!"

# Lệnh gom (chạy tuần tự setup -> pull-model)
all: setup pull-model
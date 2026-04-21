import os
import time
import pprint
from functools import wraps
from datetime import datetime

def format_for_pprint(data):
    """
    Hàm phụ trợ: Dịch các object phức tạp (như LangChain Document) 
    thành dạng dict/list cơ bản để pprint có thể format xuống dòng đẹp mắt.
    """
    if isinstance(data, list):
        return [format_for_pprint(item) for item in data]
    elif isinstance(data, dict):
        return {key: format_for_pprint(value) for key, value in data.items()}
    # Nếu là object Document của LangChain
    elif hasattr(data, 'page_content') and hasattr(data, 'metadata'):
        return {
            "metadata": data.metadata,
            "page_content": data.page_content
        }
    # Nếu là các object Pydantic chung chung
    elif hasattr(data, 'model_dump') and callable(data.model_dump):
        return data.model_dump()
    return data

def log_to_file(func):
    """
    Decorator lưu log vào thư mục 'logs/'.
    Chỉ tập trung ghi nhận kết quả đầu ra (OUTPUT) và thời gian chạy.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{func.__name__}.log")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        start_time = time.perf_counter()
        try:
            # Chạy hàm
            result = func(*args, **kwargs)
            
            # BƯỚC QUAN TRỌNG: Làm sạch dữ liệu trước khi format
            clean_result = format_for_pprint(result)
            
            # Format Outputs: Dàn ra nhiều dòng để dễ đọc
            out_formatted = pprint.pformat(clean_result, indent=2, width=100)
            out_aligned = out_formatted.replace('\n', '\n│           ')
                
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n[{current_time}]\n")
                f.write(f"┌{'─'*80}\n")
                f.write(f"│ 🔴 [OUT] Trả về:\n│           {out_aligned}\n")
                f.write(f"│ ⏱️  Thời gian: {time.perf_counter() - start_time:.2f}s\n")
                f.write(f"└{'─'*80}\n")
            
            return result
            
        except Exception as e:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n[{current_time}]\n")
                f.write(f"┌{'─'*80}\n")
                f.write(f"│ ❌ [ERR] LỖI: {e}\n")
                f.write(f"│ ⏱️  Thời gian: {time.perf_counter() - start_time:.2f}s\n")
                f.write(f"└{'─'*80}\n")
            raise e
            
    return wrapper
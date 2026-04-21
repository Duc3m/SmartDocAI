from core.prompts import get_prompt_template
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from src.utils.logger import log_to_file
from src.utils.timer import time_it

set_debug(False)

def format_docs(docs):
    """
    Kết hợp nội dung các chunks được tìm thấy thành một chuỗi context liền mạch.
    """
    return "\n\n".join(doc.page_content for doc in docs)

@log_to_file
def retrieve_relevant_docs(user_input, retriever):
    # Lấy ra các chunk liên quan nhất từ FAISS
    return retriever.invoke(user_input)

@log_to_file
def generate_final_prompt(user_input, retriever):
    docs = retrieve_relevant_docs(user_input, retriever)
    context = format_docs(docs)
    prompt_template = get_prompt_template(user_input)
    return prompt_template.format(context=context, question=user_input)

@log_to_file
def generate_llm_answer(formatted_prompt, llm):
    # Gọi LLM để sinh câu trả lời dựa trên prompt đã định dạng.
    return llm.invoke(formatted_prompt)

def answer_query(user_input: str, retriever, llm):
    """
    Hàm thực thi toàn bộ luồng RAG và trả về câu trả lời cuối cùng.
    """
    try:
        final_prompt = generate_final_prompt(user_input, retriever)
        response = generate_llm_answer(final_prompt, llm)

        # Xử lý nếu kết quả trả về là đối tượng thay vì chuỗi
        if hasattr(response, 'content'):
            return response.content
        return str(response)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "connection refused" in error_msg or "timeout" in error_msg:
            return "🔌 **Lỗi kết nối:** Hãy đảm bảo đã bật Ollama!"
        elif "model" in error_msg and "not found" in error_msg:
            return "📦 **Thiếu mô hình:** Vui lòng chạy `ollama pull qwen2.5:1.5b`."
        else:
            return f"🤖 **Lỗi:** {str(e)}"
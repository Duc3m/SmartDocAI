from core.prompts import get_prompt_template
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

set_debug(True)

def format_docs(docs):
    """
    Kết hợp nội dung các chunks được tìm thấy thành một chuỗi context liền mạch[cite: 153].
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm, user_input: str):
    """
    Khởi tạo RAG Chain kết nối Bộ truy xuất (Retriever), Prompt và Mô hình sinh (Generator) [cite: 68-74, 116].
    """
    # 1. Khởi tạo prompt động dựa trên ngôn ngữ đầu vào [cite: 155, 260]
    prompt = get_prompt_template(user_input)
    
    # 2. Xây dựng pipeline xử lý [cite: 127-128]
    # - "context": Lấy câu hỏi -> retriever tìm kiếm -> format_docs gộp văn bản
    # - "question": Chuyển tiếp (Passthrough) trực tiếp câu hỏi của người dùng
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Parse output trực tiếp ra dạng string
    )
    
    return rag_chain

def answer_query(user_input: str, retriever, llm):
    """
    Hàm thực thi toàn bộ luồng RAG và trả về câu trả lời cuối cùng [cite: 156-157].
    """
    try:
        # Tạo chain cho ngữ cảnh hiện tại
        chain = create_rag_chain(retriever, llm, user_input)
        # with open('context.txt', 'a', encoding='utf-8') as file:
        #     file.write(chain + "\n\n=*50")
        # Dùng .invoke() để lấy toàn bộ câu trả lời
        response = chain.invoke(user_input)
        
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
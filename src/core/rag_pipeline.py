from core.prompts import get_prompt_template
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
import re

set_debug(True)

STRUCTURE_KEYWORDS = (
    "bao nhiêu chương",
    "bao nhieu chuong",
    "mấy chương",
    "may chuong",
    "mục lục",
    "muc luc",
    "cấu trúc",
    "cau truc",
    "chapter"
)

CHAPTER_LINE_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?(?:chương|chuong|chapter)\s+([0-9]+|[ivxlcdm]+)(?:\b|\s*[:\.\-\)])",
    re.IGNORECASE
)

def format_docs(docs):
    """
    Kết hợp nội dung các chunks được tìm thấy thành một chuỗi context liền mạch[cite: 153].
    """
    return "\n\n".join(doc.page_content for doc in docs)


def is_structure_question(user_input: str) -> bool:
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in STRUCTURE_KEYWORDS)


def extract_chapter_evidence(docs):
    chapter_labels = []
    evidence_lines = []
    seen_labels = set()

    for doc in docs:
        lines = doc.page_content.splitlines()
        for line in lines:
            matched = CHAPTER_LINE_PATTERN.match(line.strip())
            if not matched:
                continue

            chapter_id = matched.group(1).upper()
            chapter_label = f"Chương {chapter_id}"
            if chapter_label in seen_labels:
                continue

            seen_labels.add(chapter_label)
            chapter_labels.append(chapter_label)
            evidence_lines.append(line.strip())

    return chapter_labels, evidence_lines


def retrieve_docs_for_query(retriever, user_input: str, structure_mode: bool = False):
    original_search_kwargs = None

    if structure_mode and hasattr(retriever, "search_kwargs") and isinstance(retriever.search_kwargs, dict):
        original_search_kwargs = dict(retriever.search_kwargs)
        expanded_kwargs = dict(retriever.search_kwargs)
        expanded_kwargs["k"] = max(expanded_kwargs.get("k", 6), 18)
        expanded_kwargs["fetch_k"] = max(expanded_kwargs.get("fetch_k", 24), 80)
        retriever.search_kwargs = expanded_kwargs

    try:
        return retriever.invoke(user_input)
    finally:
        if original_search_kwargs is not None:
            retriever.search_kwargs = original_search_kwargs

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


def generate_answer_with_docs(user_input: str, docs, llm):
    prompt = get_prompt_template(user_input)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": user_input})

def answer_query(user_input: str, retriever, llm):
    """
    Hàm thực thi toàn bộ luồng RAG và trả về câu trả lời cuối cùng [cite: 156-157].
    """
    try:
        if is_structure_question(user_input):
            docs = retrieve_docs_for_query(retriever, user_input, structure_mode=True)
            chapter_labels, evidence_lines = extract_chapter_evidence(docs)

            if chapter_labels:
                chapter_list = "\n".join(
                    f"{index}. {chapter}"
                    for index, chapter in enumerate(chapter_labels, start=1)
                )
                evidence_text = "\n".join(f"- {line}" for line in evidence_lines[:3])
                return (
                    f"Tài liệu này có {len(chapter_labels)} chương, cụ thể là:\n\n"
                    f"{chapter_list}\n\n"
                    f"Bằng chứng từ ngữ cảnh:\n{evidence_text}"
                )

            if not docs:
                return "Xin lỗi, tài liệu không đề cập đến thông tin này."

            response = generate_answer_with_docs(user_input, docs, llm)
            if hasattr(response, 'content'):
                return response.content
            return str(response)

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
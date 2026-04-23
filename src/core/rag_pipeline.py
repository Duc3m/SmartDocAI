from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import END, StateGraph
from core.prompts import get_prompt_template, GRADER_PROMPT, REWRITE_PROMPT
import streamlit as st
from src.utils.logger import log_to_file
from src.utils.timer import time_it
from data_access.database import get_chat_history


set_debug(False)

# 1. Định nghĩa trạng thái (State)
class GraphState(TypedDict):
    question: str
    search_query: str
    chat_history: List[dict]
    generation: str
    documents: List[str]
    loop_count: int
    max_loops: int

# 2. Định nghĩa các Nodes
def retrieve_node(state: GraphState):
    print(f"🟢 [NODE: RETRIEVE] Đang tìm kiếm (Vòng lặp: {state.get('loop_count', 0)})")
    question = state["search_query"]
    # Truy xuất từ retriever trong session_state
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents}

def grade_documents_node(state: GraphState):
    from models.llm_config import get_llm
    print("🟡 [NODE: GRADE] Đang chấm điểm tài liệu...")
    llm = get_llm()
    
    question = state["question"]
    documents = state["documents"]
    
    grader_chain = GRADER_PROMPT | llm | StrOutputParser()
    context = "\n\n".join(d.page_content for d in documents)
    
    score = grader_chain.invoke({"question": question, "context": context})
    
    if "YES" in score.upper():
        # Đánh dấu là đã tìm thấy context tốt, có thể dùng một biến cờ
        print("   -> Kết quả: ĐẠT (Chuyển sang Generate)")
        return {"documents": documents, "is_relevant": True}
    else:
        # KHÔNG XÓA documents. Giữ lại để làm fallback nếu chạm max_loops
        print("   -> Kết quả: KHÔNG ĐẠT (Cần Rewrite)")
        return {"documents": documents, "is_relevant": False}

def rewrite_node(state: GraphState):
    from models.llm_config import get_llm
    print("🟠 [NODE: REWRITE] Đang viết lại câu hỏi...")
    llm = get_llm()
    
    question = state["question"]
    count = state.get("loop_count", 0) + 1
    chat_history = state["chat_history"]
    
    rewriter_chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_query = rewriter_chain.invoke({"question": question, "chat_history": chat_history})
    
    return {"search_query": new_query, "loop_count": count}

def generate_node(state: GraphState):
    from models.llm_config import get_llm
    print("🔵 [NODE: GENERATE] Đang sinh câu trả lời cuối cùng...")
    llm = get_llm()
    
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    context = "\n\n".join(d.page_content for d in documents) if documents else ""
    
    prompt_template = get_prompt_template(question)
    gen_chain = prompt_template | llm | StrOutputParser()
    
    generation = gen_chain.invoke({"question": question, "context": context, "chat_history": chat_history})
    return {"generation": generation}

def decide_to_generate(state):
    # Nếu documents liên quan, HOẶC đã hết số lần lặp cho phép -> Trả lời
    if state.get("is_relevant", False) or state.get("loop_count", 0) >= state.get("max_loops", 3):
        return "generate"
    return "rewrite"

# 3. Xây dựng và Compile Graph
def create_crag_app():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges("grade", decide_to_generate)
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Hàm wrapper để UI gọi
def answer_query_crag(user_input: str,file_id: int, max_loops: int = 3):
    # Lấy lịch sử chat của file hiện tại
    chat_history = get_chat_history(file_id)

    history_str = ""
    if chat_history:
        for msg in chat_history[-5:]: # Chỉ lấy 5 tin nhắn gần nhất để tránh tràn context
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_str += f"{role}: {msg['content']}\n"

    app = create_crag_app()
    result = app.invoke({
        "question": user_input,
        "search_query": user_input,
        "chat_history": history_str,
        "loop_count": 0,
        "max_loops": max_loops,
        "documents": [],
        "is_relevant": False
    })
    return result["generation"]

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
def generate_final_prompt(user_input, retriever,chat_history=None):
    history_str = ""
    if chat_history:
        for msg in chat_history[-5:]: # Chỉ lấy 5 tin nhắn gần nhất để tránh tràn context
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_str += f"{role}: {msg['content']}\n"
    docs = retrieve_relevant_docs(user_input, retriever)
    context = format_docs(docs)
    prompt_template = get_prompt_template(user_input)
    return prompt_template.format(context=context, question=user_input, chat_history=history_str)

@log_to_file
def generate_llm_answer(formatted_prompt, llm):
    # Gọi LLM để sinh câu trả lời dựa trên prompt đã định dạng.
    return llm.invoke(formatted_prompt)

def answer_query(user_input: str, file_id ,retriever, llm):
    """
    Hàm thực thi toàn bộ luồng RAG và trả về câu trả lời cuối cùng.
    """
    try:
    # Lấy lịch sử chat của file hiện tại
        history = get_chat_history(file_id)
        final_prompt = generate_final_prompt(user_input, retriever,history)
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
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


set_debug(False)

# 1. Định nghĩa trạng thái (State)
class GraphState(TypedDict):
    question: str          # Câu hỏi gốc của người dùng
    search_query: str      # Câu hỏi dùng để search (có thể bị rewrite)
    generation: str        # Kết quả cuối cùng
    documents: List[str]   # Các đoạn văn bản tìm được
    loop_count: int        # Số lần đã lặp lại

# 2. Định nghĩa các Nodes
def retrieve_node(state: GraphState):
    question = state["search_query"]
    # Truy xuất từ retriever trong session_state
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents}

def grade_documents_node(state: GraphState):
    from models.llm_config import get_llm
    llm = get_llm()
    
    question = state["question"]
    documents = state["documents"]
    
    grader_chain = GRADER_PROMPT | llm | StrOutputParser()
    context = "\n\n".join(d.page_content for d in documents)
    
    score = grader_chain.invoke({"question": question, "context": context})
    
    # Trả về kết quả đánh giá thông qua một flag trong state (giả định dùng tạm documents làm flag)
    if "YES" in score.upper():
        return {"loop_count": state.get("loop_count", 0)} # Giữ nguyên
    else:
        return {"documents": []} # Xóa docs cũ để kích hoạt rewrite

def rewrite_node(state: GraphState):
    from models.llm_config import get_llm
    llm = get_llm()
    
    question = state["question"]
    count = state.get("loop_count", 0) + 1
    
    rewriter_chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_query = rewriter_chain.invoke({"question": question})
    
    return {"search_query": new_query, "loop_count": count}

def generate_node(state: GraphState):
    from models.llm_config import get_llm
    llm = get_llm()
    
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(d.page_content for d in documents) if documents else ""
    
    prompt_template = get_prompt_template(question)
    gen_chain = prompt_template | llm | StrOutputParser()
    
    generation = gen_chain.invoke({"question": question, "context": context})
    return {"generation": generation}

# 3. Xây dựng và Compile Graph
def create_crag_app():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    
    def decide_to_generate(state):
        # Nếu có tài liệu hoặc đã lặp quá 2 lần thì trả lời luôn
        if state["documents"] or state.get("loop_count", 0) >= 2:
            return "generate"
        return "rewrite"
    
    workflow.add_conditional_edges("grade", decide_to_generate)
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Hàm wrapper để UI gọi
def answer_query_crag(user_input: str):
    app = create_crag_app()
    result = app.invoke({
        "question": user_input,
        "search_query": user_input,
        "loop_count": 0,
        "documents": []
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
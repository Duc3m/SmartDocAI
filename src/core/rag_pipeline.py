from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from typing import Any, Dict, List, TypedDict
from langgraph.graph import END, StateGraph
from core.prompts import get_prompt_template, get_citation_prompt_template, GRADER_PROMPT, REWRITE_PROMPT
import streamlit as st
from src.utils.logger import log_to_file
from src.utils.timer import time_it
from data_access.database import get_chat_history
import json
import os
import re


set_debug(False)

# 1. Định nghĩa trạng thái (State)
class GraphState(TypedDict):
    question: str
    search_query: str
    chat_history: List[dict]
    generation: Dict[str, Any]
    documents: List[Any]
    loop_count: int
    max_loops: int


def _history_to_string(chat_history: List[dict], max_items: int = 5) -> str:
    history_str = ""
    if chat_history:
        for msg in chat_history[-max_items:]:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_str += f"{role}: {msg['content']}\n"
    return history_str


def _build_context_bundle(documents: List[Any]) -> tuple[str, Dict[str, Dict[str, Any]]]:
    context_blocks = []
    source_map: Dict[str, Dict[str, Any]] = {}

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source_id = f"S{index}"
        source_path = metadata.get("source") or metadata.get("file_name") or "Unknown"
        file_name = os.path.basename(source_path)

        page_value = metadata.get("page")
        if isinstance(page_value, int):
            page_number = page_value + 1
        else:
            page_number = None

        char_start = metadata.get("char_start", metadata.get("start_index"))
        char_end = metadata.get("char_end")
        if isinstance(char_start, int) and not isinstance(char_end, int):
            char_end = char_start + len(doc.page_content)

        source_map[source_id] = {
            "source_id": source_id,
            "file_name": file_name,
            "page": page_number,
            "position": {
                "start": char_start,
                "end": char_end,
            },
            "chunk_id": metadata.get("chunk_id", index),
            "context": doc.page_content,
        }

        position_text = "N/A"
        if isinstance(char_start, int) and isinstance(char_end, int):
            position_text = f"{char_start}-{char_end}"

        page_text = str(page_number) if page_number is not None else "N/A"
        context_blocks.append(
            "\n".join(
                [
                    f"[{source_id}]",
                    f"file: {file_name}",
                    f"page: {page_text}",
                    f"position: {position_text}",
                    f"chunk_id: {source_map[source_id]['chunk_id']}",
                    "content:",
                    doc.page_content,
                ]
            )
        )

    return "\n\n".join(context_blocks), source_map


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"answer": raw_text.strip(), "citations": []}


def _normalize_citation_payload(
    llm_output: str,
    source_map: Dict[str, Dict[str, Any]],
    fallback_answer: str | None = None,
) -> Dict[str, Any]:
    parsed = _safe_parse_json(llm_output)
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        answer = fallback_answer or llm_output.strip()

    normalized_citations = []
    citations = parsed.get("citations", []) if isinstance(parsed, dict) else []
    if not isinstance(citations, list):
        citations = []

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source_id = str(citation.get("source_id", "")).strip()
        quote = str(citation.get("quote", "")).strip()
        if source_id not in source_map:
            continue

        source_payload = source_map[source_id]
        normalized_citations.append(
            {
                "source_id": source_id,
                "file_name": source_payload["file_name"],
                "page": source_payload["page"],
                "position": source_payload["position"],
                "chunk_id": source_payload["chunk_id"],
                "quote": quote,
                "context": source_payload["context"],
            }
        )

    return {
        "answer": answer.strip(),
        "citations": normalized_citations,
    }


def _generate_cited_answer(question: str, documents: List[Any], chat_history: str, llm) -> Dict[str, Any]:
    context, source_map = _build_context_bundle(documents)
    prompt_template = get_citation_prompt_template(question)
    gen_chain = prompt_template | llm | StrOutputParser()
    raw_generation = gen_chain.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history,
        }
    )
    return _normalize_citation_payload(raw_generation, source_map)

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

    generation = _generate_cited_answer(question, documents, chat_history, llm)
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

    history_str = _history_to_string(chat_history)

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
    history_str = _history_to_string(chat_history)
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
        docs = retrieve_relevant_docs(user_input, retriever)
        history_str = _history_to_string(history)
        return _generate_cited_answer(user_input, docs, history_str, llm)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "connection refused" in error_msg or "timeout" in error_msg:
            return "🔌 **Lỗi kết nối:** Hãy đảm bảo đã bật Ollama!"
        elif "model" in error_msg and "not found" in error_msg:
            return "📦 **Thiếu mô hình:** Vui lòng chạy `ollama pull qwen2.5:1.5b`."
        else:
            return f"🤖 **Lỗi:** {str(e)}"
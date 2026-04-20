from langchain_core.prompts import PromptTemplate

def is_vietnamese(user_input: str) -> bool:
    """
    Kiểm tra xem câu hỏi có chứa ký tự tiếng Việt có dấu hay không.
    """
    vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    return any(char in user_input.lower() for char in vietnamese_chars)

# --- PROMPT CHẤM ĐIỂM (CRAG EVALUATOR) ---
GRADER_PROMPT = PromptTemplate.from_template(
    """Bạn là một chuyên gia đánh giá mức độ liên quan của tài liệu.
    Câu hỏi: {question}
    Tài liệu: {context}
    
    Nhiệm vụ: Hãy xác định xem tài liệu trên có chứa thông tin để trả lời câu hỏi hay không.
    - Nếu có: Trả lời duy nhất từ 'YES'.
    - Nếu không: Trả lời duy nhất từ 'NO'.
    
    Kết quả:"""
)

# --- PROMPT VIẾT LẠI CÂU HỎI (REWRITE QUERY) ---
REWRITE_PROMPT = PromptTemplate.from_template(
    """Bạn là một chuyên gia tối ưu hóa truy vấn tìm kiếm.
    Câu hỏi gốc: {question}
    
    Dữ liệu hiện tại không đủ để trả lời. Hãy viết lại câu hỏi này bằng tiếng Việt sao cho rõ ràng hơn.
    LƯU Ý QUAN TRỌNG: Nếu người dùng yêu cầu "tóm tắt", "nội dung chính" của một phần/chương, hãy đổi câu hỏi thành yêu cầu trích xuất các chủ đề, từ khóa, hoặc ý chính của phần/chương đó thay vì chỉ dùng từ "tóm tắt".
    
    Chỉ trả về nội dung câu hỏi mới, không giải thích gì thêm.
    Câu hỏi mới:"""
)

def get_prompt_template(user_input: str) -> PromptTemplate:
    """
    Trả về PromptTemplate tối ưu dựa trên ngôn ngữ được phát hiện từ câu hỏi.
    """
    if is_vietnamese(user_input):
        # Prompt cho tiếng Việt (Giữ nguyên format cũ của bạn)
        prompt_text = r"""
            Bạn là một chuyên gia phân tích tài liệu chuyên nghiệp.
            Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA VÀO DUY NHẤT ngữ cảnh (Context) được cung cấp dưới đây.

            Context:
            {context}

            Question:
            {question}

            🚨 CÁC QUY TẮC BẮT BUỘC PHẢI TUÂN THỦ (NẾU VI PHẠM SẼ BỊ PHẠT):
            1. NGÔN NGỮ: BẠN PHẢI LUÔN LUÔN TRẢ LỜI BẰNG TIẾNG VIỆT (VIETNAMESE). TUYỆT ĐỐI KHÔNG ĐƯỢC SỬ DỤNG TIẾNG TRUNG HOẶC BẤT KỲ NGÔN NGỮ NÀO KHÁC.
            2. SỰ THẬT: Chỉ sử dụng thông tin xuất hiện trong phần Context. Nếu phần Context không chứa câu trả lời cho câu hỏi, bạn PHẢI trả lời chính xác là: "Xin lỗi, tài liệu không đề cập đến thông tin này."
            3. KHÔNG BỊA ĐẶT: Tuyệt đối không được tự suy diễn, không được dùng kiến thức bên ngoài Context để trả lời.

            🚨 QUY TẮC ĐỊNH DẠNG TOÁN HỌC:
            - Nếu có công thức toán học, bạn PHẢI sử dụng LaTeX.
            - Sử dụng cặp dấu $$ cho công thức hiển thị riêng biệt (ví dụ: $$E=mc^2$$).
            - Sử dụng cặp dấu $ cho công thức nằm trong dòng văn bản (ví dụ: $x=2$).
            - TUYỆT ĐỐI KHÔNG dùng các ký hiệu như \[ \] hoặc \( \).
        """
    else:
        # Prompt cho tiếng Anh (Giữ nguyên format cũ của bạn)
        prompt_text = r"""
            You are a professional document analysis expert.
            Your task is to answer the user's question based ONLY on the provided Context.

            Context:
            {context}

            Question:
            {question}

            🚨 MANDATORY RULES:
            1. LANGUAGE: Respond in the language of the question.
            2. TRUTH: Use only information from the Context. If the answer is not in the Context, respond: "I'm sorry, the document does not mention this information."
            3. NO HALLUCINATION: Do not use external knowledge.

            🚨 MATH FORMATTING RULES:
            - Use LaTeX for formulas.
            - Use $$ for block formulas and $ for inline formulas.
        """
    
    return PromptTemplate.from_template(prompt_text)
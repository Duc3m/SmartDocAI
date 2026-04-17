from langchain_core.prompts import PromptTemplate

def is_vietnamese(user_input: str) -> bool:
    """
    Kiểm tra xem câu hỏi có chứa ký tự tiếng Việt có dấu hay không [cite: 231-233].
    """
    vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    return any(char in user_input.lower() for char in vietnamese_chars)

def get_prompt_template(user_input: str) -> PromptTemplate:
    """
    Trả về PromptTemplate tối ưu dựa trên ngôn ngữ được phát hiện từ câu hỏi [cite: 230-264].
    """
    if is_vietnamese(user_input):
        # Prompt cho tiếng Việt [cite: 240-247]
        prompt_text = """
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
        # Prompt cho tiếng Anh và ngôn ngữ khác [cite: 250-258]
        prompt_text = """Use the following context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep answer concise (3-4 sentences).
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
    return PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )
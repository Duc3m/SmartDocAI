import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        .stAppDeployButton {
            display: none !important;
        }

        /* 1. VỊ TRÍ POPOVER */
        .st-key-mode_selector {
            position: fixed;
            top: 0.7rem; 
            z-index: 999999;
        }
        
        /* 2. NÚT TRIGGER Ở NGOÀI CÙNG */
        .st-key-mode_selector div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
            border: none !important;
            
            /* THAY ĐỔI Ở ĐÂY: Dùng màu nền của app thay vì trong suốt */
            background: var(--background-color) !important; 
            
            border-radius: 8px !important; /* Giữ bo góc cho mượt */
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            padding: 4px 10px !important;
            transition: all 0.2s;
            z-index: 999999 !important; /* Đảm bảo nó luôn nằm trên cùng */
        }
        
        .st-key-mode_selector div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:focus:not(:active),
        .st-key-mode_selector div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:hover,
        .st-key-mode_selector div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:active {
            border: none !important;
            background-color: rgba(128, 128, 128, 0.15) !important;
            color: inherit !important;
            /* Vẫn giữ bóng đổ khi rê chuột vào */
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important; 
        }

        /* ========================================================
           3. CÁCH LY MENU XỔ XUỐNG: Chỉ nhắm vào popover của CRAG
           ======================================================== */
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) {
            border: 1px solid rgba(128, 128, 128, 0.1) !important;
            border-radius: 10px !important;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.15) !important;
            padding: 4px !important; /* Bóp padding của khung */
            min-width: max-content !important; 
            width: max-content !important;
        }

        /* Nút con bên trong (Thu nhỏ) */
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button {
            border: none !important; 
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 6px 10px !important; /* <-- NÚT CON NHỎ LẠI */
            border-radius: 6px !important;
            width: 100% !important;
        }
        
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button:focus:not(:active) {
            border-color: transparent !important;
            box-shadow: none !important;
            color: inherit !important;
        }
        
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button:hover {
            background-color: rgba(128, 128, 128, 0.15) !important;
        }
        
        /* Ép căn trái cho các nút con */
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button > div,
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button > div > span,
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button div[data-testid="stMarkdownContainer"] {
            display: flex !important;
            width: 100% !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }

        /* Chữ của nút con (Thu nhỏ font) */
        div[data-testid="stPopoverBody"]:has(.st-key-btn_mode_rag) button p {
            text-align: left !important;
            margin: 0 !important;
            width: 100% !important;
            font-size: 0.9rem !important; /* <-- CHỮ NÚT CON NHỎ LẠI */
            font-weight: 500 !important;
        }
                
        /* ========================================================
           6. LÀM PHẲNG VÀ TỐI ƯU SIDEBAR (CHUẨN CHATGPT)
           ======================================================== */
        
        /* 1. KHÓA CHẾT ĐỘ RỘNG (Vô hiệu hóa tính năng kéo dãn) */
        /* Chỉ khóa khi sidebar đang mở để không làm hỏng nút đóng/mở (Collapse) */
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 310px !important;
            max-width: 310px !important;
            width: 310px !important;
        }
        
        /* Ẩn con trỏ chuột mũi tên 2 chiều và xóa điểm neo kéo chuột */
        [data-testid="stSidebarResizer"] {
            display: none !important;
            width: 0px !important;
            pointer-events: none !important;
        }

        /* 2. SAN PHẲNG BORDER CỦA TẤT CẢ NÚT TRONG SIDEBAR */
        [data-testid="stSidebar"] button {
            border: 0px solid transparent !important;
            box-shadow: none !important;
            border-radius: 8px !important; /* Bo góc chuẩn ChatGPT */
            transition: all 0.2s ease;
        }

        /* Làm cho nút Secondary (Danh sách file) tàng hình nền khi bình thường */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: transparent !important;
        }
        
        /* Khi rê chuột: Nổi nền xám nhẹ (Hover) */
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: rgba(128, 128, 128, 0.15) !important;
        }
        
        /* Xóa viền màu khi Click (Focus) */
        [data-testid="stSidebar"] button:focus:not(:active) {
            border-color: transparent !important;
            box-shadow: none !important;
            color: inherit !important;
        }

        /* 3. CĂN CHỈNH NÚT TRONG CỘT (st.columns) */
        
        /* Cột 1 (Tên file): Ép dạt trái tuyệt đối */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button {
            justify-content: flex-start !important;
            padding-left: 12px !important; /* Tạo một chút khoảng lề trái cho icon không bị sát vách */
        }
        
        /* Bẻ khóa TẤT CẢ các lớp div, span tàng hình bên trong nút */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button > div,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button > div > span,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button div[data-testid="stMarkdownContainer"] {
            display: flex !important;
            width: 100% !important;
            justify-content: flex-start !important; /* Ép toàn bộ dạt trái */
            align-items: center !important;
        }

        /* Lớp lõi cùng chứa chữ */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button p {
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
        }

        /* 4. GIẢM KHOẢNG CÁCH (PADDING & GAP) ĐỂ GỌN GÀNG HƠN */
        /* Giảm Gap ngang: Ép nút Xóa (Cột 2) sát lại gần Tên file (Cột 1) */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            gap: 0.2rem !important; 
        }
        
        /* Giảm Gap dọc: Ép các dòng file nằm khít lại với nhau */
        [data-testid="stSidebar"] .stVerticalBlock {
            gap: 0.4rem !important; 
        }
        
        /* ========================================================
           7. ÉP NỘI DUNG SIDEBAR LÊN SÁT MÉP TRÊN (TOP)
           ======================================================== */
        /* Thu hồi vùng đệm của Sidebar Content */
        [data-testid="stSidebarUserContent"] {
            padding-top: 0rem !important; 
        }
        
        /* Bóp nghẹt khoảng trống của Sidebar Header (nơi chứa nút đóng/mở sidebar) */
        [data-testid="stSidebarHeader"] {
            padding-top: 0.5rem !important; 
            padding-bottom: 0 !important;
            min-height: 0 !important;
            height: auto !important;
        }

        /* Xóa margin mặc định của thẻ Title (h1) "SmartDoc AI" đầu tiên */
        [data-testid="stSidebarUserContent"] h1:first-child {
            margin-top: 0 !important;
            padding-top: 0.5rem !important;
        }
        /* ========================================================
           5. CĂN GIỮA DỌC TIÊU ĐỀ VÀ NÚT THÙNG RÁC (FIX CHUẨN)
           ======================================================== */
        
        /* 1. Ép hai cột trong hàng ngang phải căn giữa với nhau */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            align-items: center !important; 
        }

        /* 2. Cạo sạch margin/padding của thẻ h3 thủ phạm */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) h3 {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            line-height: 1 !important; /* Ép chiều cao dòng sát lại */
        }

        /* 3. Bắt các lớp div bọc ngoài cái h3 phải ôm sát và dồn nó vào giữa */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1),
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) [data-testid="stMarkdownContainer"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            height: 100% !important;
            margin: 0 !important;
        }
                /* ========================================================
           8. CĂN TRÁI NÚT "TẠO CUỘC TRÒ CHUYỆN MỚI"
           ======================================================== */
        .st-key-btn_new_chat {
            margin-top: 1.5rem !important; /* Bạn có thể tăng giảm số này (1rem, 2rem...) cho vừa mắt */
        }
        .st-key-btn_new_chat button {
            justify-content: flex-start !important;
            padding-left: 12px !important;
        }
        
        /* Phá vỡ các lớp div, span tàng hình bọc quanh chữ */
        .st-key-btn_new_chat button > div,
        .st-key-btn_new_chat button > div > span,
        .st-key-btn_new_chat button div[data-testid="stMarkdownContainer"] {
            display: flex !important;
            width: 100% !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }

        .st-key-btn_new_chat button p {
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
            font-weight: 600 !important; /* Làm chữ đậm lên một chút cho đẹp */
        }
                /* ========================================================
           9. THU GỌN KHOẢNG TRỐNG CỦA ĐƯỜNG KẺ (DIVIDER)
           ======================================================== */
        [data-testid="stSidebar"] hr {
            margin-top: 0.5rem !important;  /* Bóp lề trên chỉ còn một chút xíu */
            margin-bottom: 0.5rem !important; /* Bóp lề dưới */
            padding: 0 !important;
        }
        
        /* Đảm bảo container bọc bên ngoài đường kẻ không bị phình to */
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]:has(hr) {
            min-height: 0 !important;
            margin-bottom: 0 !important;
        }
        /* ========================================================
           11. ĐỔI MÀU ICON THEO LOẠI FILE (FIX CHUẨN DOM)
           ======================================================== */
        
        /* Tô màu Đỏ cho icon PDF */
        div[class*="st-key-pdf_"] span[data-testid="stIconMaterial"] {
            color: #FF4B4B !important;
        }

        /* Tô màu Xanh cho icon DOCX và DOC */
        div[class*="st-key-doc"] span[data-testid="stIconMaterial"] {
            color: #0078D4 !important;
        }
        
        /* Hiệu ứng: Khi hover vào nút, icon sáng lên một chút */
        div[class*="st-key-pdf_"]:hover span[data-testid="stIconMaterial"],
        div[class*="st-key-doc"]:hover span[data-testid="stIconMaterial"] {
            filter: brightness(1.2) drop-shadow(0px 0px 4px rgba(255,255,255,0.2));
        }
        </style>
    """, unsafe_allow_html=True)
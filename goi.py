import fitz  # PyMuPDF
import zipfile
import io
import re
import html
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient

# ==========================================
# HÀM PHỤ TRỢ: CHUYỂN HTML TABLE SANG LATEX
# ==========================================
def convert_html_table_to_latex(html_str):
    # Trích xuất toàn bộ các hàng <tr>
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html_str, re.IGNORECASE | re.DOTALL)
    if not rows:
        return html_str  # Trả về nguyên gốc nếu không tìm thấy thẻ tr
        
    latex_rows = []
    max_cols = 0
    
    for row in rows:
        # Lấy toàn bộ các ô <td> hoặc <th> trong hàng
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
        
        clean_cells = []
        for cell in cells:
            # Loại bỏ các thẻ HTML rác bên trong nếu có
            cell_text = re.sub(r'<.*?>', '', cell).strip()
            # Giải mã các ký tự HTML (ví dụ: &amp; thành &, &lt; thành <)
            cell_text = html.unescape(cell_text)
            clean_cells.append(cell_text)
        
        # Cập nhật số cột lớn nhất để vẽ khung bảng
        max_cols = max(max_cols, len(clean_cells))
        
        # Nối các ô bằng dấu '&' và kết thúc hàng bằng '\\ \hline'
        latex_rows.append(" & ".join(clean_cells) + r" \\ \hline")
        
    if max_cols == 0:
        return html_str
        
    # Tạo chuỗi định dạng cột, ví dụ có 3 cột sẽ tạo ra |l|l|l| (căn trái)
    col_format = "|" + "|".join(["l"] * max_cols) + "|"
    
    # Xây dựng bảng LaTeX hoàn chỉnh
    # Dùng p{...} nếu nội dung dài cần tự xuống dòng, ở đây dùng 'l' mặc định
    latex_tb = "\\begin{table}[htbp]\n\\centering\n"
    latex_tb += "\\begin{tabular}{" + col_format + "}\n\\hline\n"
    latex_tb += "\n".join(latex_rows) + "\n"
    latex_tb += "\\end{tabular}\n\\end{table}"
    
    return latex_tb

def process_pdf_to_latex_zip(pdf_path, zip_output_name):
    # ==========================================
    # 1. KHỞI TẠO MÔ HÌNH MINERU
    # ==========================================
    print("Đang tải mô hình MinerU...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "opendatalab/MinerU2.5-Pro-2604-1.2B", dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(
        "opendatalab/MinerU2.5-Pro-2604-1.2B", use_fast=True
    )
    client = MinerUClient(
        backend="transformers", model=model, processor=processor,
        image_analysis=False
    )

    doc = fitz.open(pdf_path)
    all_blocks = []
    
    print(f"Bắt đầu xử lý file: {pdf_path}")
    
    # ==========================================
    # 2. XỬ LÝ PDF VÀ GHI TRỰC TIẾP VÀO ZIP
    # ==========================================
    with zipfile.ZipFile(zip_output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        for page_index in range(len(doc)):
            print(f"Đang phân tích trang {page_index + 1}/{len(doc)}...")
            page = doc[page_index]
            
            # Render trang PDF thành ảnh
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_width, img_height = img.size
            
            # Gọi mô hình trích xuất
            page_output = client.two_step_extract(img)
            
            for item in page_output:
                item['page'] = page_index 
                
                if item.get('type') in ['image', 'figure']:
                    bbox = item.get('bbox')
                    raw_content = item.get('content', '')
                    
                    filename = raw_content if raw_content else f"page_{page_index+1}_img_{len(all_blocks)}.png"
                    if not filename.endswith(('.png', '.jpg', '.jpeg')):
                        filename += ".png"
                    
                    if bbox[2] <= 1.0 and bbox[3] <= 1.0:
                        x0, y0 = bbox[0] * img_width, bbox[1] * img_height
                        x1, y1 = bbox[2] * img_width, bbox[3] * img_height
                    else:
                        x0, y0, x1, y1 = bbox
                        
                    try:
                        cropped_img = img.crop((x0, y0, x1, y1))
                        img_byte_arr = io.BytesIO()
                        cropped_img.save(img_byte_arr, format='PNG')
                        
                        image_path_in_zip = f"images/{filename}"
                        zipf.writestr(image_path_in_zip, img_byte_arr.getvalue())
                        
                        item['content'] = image_path_in_zip
                    except Exception as e:
                        print(f"Lỗi khi cắt ảnh tại trang {page_index + 1}: {e}")
                        
                all_blocks.append(item)

        # ==========================================
        # 3. SẮP XẾP VÀ TẠO LATEX
        # ==========================================
        print("Đang tổng hợp và tạo file LaTeX...")
        
        all_blocks.sort(key=lambda b: (b.get('page', 0), b['bbox'][1], b['bbox'][0]))

        # Chuẩn bị khung LaTeX cơ bản (Bổ sung thêm thư viện amsmath, array nếu bảng có chứa công thức)
        latex_content = (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[vietnamese]{babel}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{array}\n"
            "\\begin{document}\n\n"
        )
        
        for block in all_blocks:
            content = block.get('content', '')
            b_type = block.get('type')
            
            if b_type in ['image', 'figure']:
                # Ghi chú: Sử dụng \\textwidth thay vì \textwidth để Python không hiểu nhầm là escape character
                latex_content += f"\\includegraphics[width=0.8\\textwidth]{{{content}}}\n\n"
            
            elif b_type == 'table_caption':
                # Chuyển tiêu đề bảng thành dạng in đậm và căn giữa
                latex_content += f"\\begin{{center}}\n\\textbf{{{content}}}\n\\end{{center}}\n\n"
            
            elif b_type == 'table':
                # Gọi hàm để chuyển HTML string sang cú pháp bảng LaTeX
                latex_table = convert_html_table_to_latex(content)
                latex_content += f"{latex_table}\n\n"
                
            # XỬ LÝ RIÊNG BIỆT CHO HEADER -> IN ĐẬM
            elif b_type == 'header':
                # Bọc nội dung content vào lệnh \textbf{} của LaTeX
                latex_content += f"\\section*{{{content}}}\n\n"
                
            elif b_type == 'title':
                # Bọc nội dung content vào lệnh \textbf{} của LaTeX
                latex_content += f"\\textbf{{{content}}}\n\n"

            else:
                # Các khối văn bản bình thường khác (text, paragraph...)
                latex_content += f"{content}\n\n"
                
        latex_content += "\\end{document}"
        
        with open("output.tex", "w", encoding="utf-8") as f:
            f.write(latex_content)
            
        zipf.writestr("tailieu_chinh.tex", latex_content.encode('utf-8'))

    print(f"Hoàn tất! Toàn bộ file LaTeX và hình ảnh đã được đóng gói trực tiếp vào: {zip_output_name}")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    file_pdf_dau_vao = r"data\De thi THCS Cau giay 2024-2025.pdf" 
    file_zip_dau_ra = "ketqua_tonghop.zip"
    
    process_pdf_to_latex_zip(file_pdf_dau_vao, file_zip_dau_ra)
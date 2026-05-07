import fitz 
import zipfile
import io
import os
import re
import html
from PIL import Image
from paddleocr import PaddleOCRVL

# ==========================================
# 1. HÀM CHUYỂN HTML SANG BẢNG LATEX 
# ==========================================
def convert_html_table_to_latex(html_str):
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html_str, re.IGNORECASE | re.DOTALL)
    if not rows:
        return html_str
        
    latex_rows = []
    max_cols = 0
    for row in rows:
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
        clean_cells = [html.unescape(re.sub(r'<.*?>', '', cell).strip()) for cell in cells]
        max_cols = max(max_cols, len(clean_cells))
        latex_rows.append(" & ".join(clean_cells) + r" \\ \hline")
        
    if max_cols == 0: return html_str
    
    
    if max_cols == 1:
        col_format = "|X|"
    else:
        col_format = "|c|" + "|".join(["X"] * (max_cols - 1)) + "|"
        
    # Sử dụng tabularx và set chiều rộng bảng bằng \textwidth
    latex_tb = "\\begin{table}[htbp]\n\\centering\n"
    latex_tb += "\\begin{tabularx}{\\textwidth}{" + col_format + "}\n\\hline\n"
    latex_tb += "\n".join(latex_rows) + "\n"
    latex_tb += "\\end{tabularx}\n\\end{table}"
    
    return latex_tb

# ==========================================
# 2. HÀM TRÍCH XUẤT THUỘC TÍNH AN TOÀN TỪ PADDLE
# ==========================================
def extract_block_data(block):
    """Hàm này đảm bảo lấy được dữ liệu bất kể block là Dict hay Object"""
    if isinstance(block, dict):
        label = block.get('block_label', block.get('label', ''))
        content = block.get('block_content', block.get('content', ''))
        bbox = block.get('block_bbox', block.get('bbox', [0, 0, 0, 0]))
        order = block.get('block_order')
    else:
        label = getattr(block, 'block_label', getattr(block, 'label', ''))
        content = getattr(block, 'block_content', getattr(block, 'content', ''))
        bbox = getattr(block, 'block_bbox', getattr(block, 'bbox', [0, 0, 0, 0]))
        order = getattr(block, 'block_order', None)
        
    if order is None:
        order = 9999
    return label, content, bbox, order

# ==========================================
# 3. HÀM CHÍNH: XỬ LÝ & TẠO FILE
# ==========================================
def process_paddle_to_latex_zip(input_file, zip_output_name):
    folder = "output/"
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("Đã xoá toàn bộ file trong thư mục output/")

    print("Đang khởi tạo mô hình PaddleOCRVL...")
    pipeline = PaddleOCRVL()
    predict_output = pipeline.predict(input_file)

    if not isinstance(predict_output, list):
        predict_output = [predict_output]

    is_pdf = input_file.lower().endswith('.pdf')
    if is_pdf: doc = fitz.open(input_file)
    else: doc = Image.open(input_file)

    # KHAI BÁO THÊM THƯ VIỆN TABULARX VÀO PREAMBLE ĐỂ CHỐNG TRÀN BẢNG
    latex_content = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[vietnamese]{babel}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{array}\n"
        "\\usepackage{tabularx}\n" 
        "\\begin{document}\n\n"
    )

    image_counter = 0
    ignore_labels = ["header", "footer", "number", "footnote", "header_image", "footer_image", "aside_text"]

    with zipfile.ZipFile(zip_output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        for page_index, page_data in enumerate(predict_output):
            parsing_list = []
            if isinstance(page_data, dict):
                parsing_list = page_data.get("parsing_res_list", [])
            elif hasattr(page_data, "parsing_res_list"):
                parsing_list = getattr(page_data, "parsing_res_list", [])
            elif isinstance(page_data, list):
                parsing_list = page_data
            
            print(f"Trang {page_index}: Tìm thấy {len(parsing_list)} khối văn bản (blocks).")

            parsing_list.sort(key=lambda b: (extract_block_data(b)[3], extract_block_data(b)[2][1]))

            for block in parsing_list:
                b_label, b_content, b_bbox, b_order = extract_block_data(block)

                if not b_label or b_label in ignore_labels:
                    continue
                
                if b_label == 'doc_title':
                    latex_content += f"\\begin{{center}}\n\\Large\\textbf{{{b_content}}}\n\\end{{center}}\n\n"
                
                elif b_label in ['paragraph_title', 'figure_title']:
                    latex_content += f"\\textbf{{{b_content}}}\n\n"
                
                elif b_label == 'table':
                    latex_table = convert_html_table_to_latex(b_content)
                    latex_content += f"{latex_table}\n\n"
                
                elif b_label == 'image':
                    try:
                        img_byte_arr = io.BytesIO()
                        if is_pdf:
                            page = doc[page_index]
                            mat = fitz.Matrix(2.0, 2.0) 
                            pix = page.get_pixmap(matrix=mat)
                            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            x0, y0, x1, y1 = [v * 2.0 for v in b_bbox] 
                            cropped_img = pil_img.crop((x0, y0, x1, y1))
                        else:
                            cropped_img = doc.crop(b_bbox)
                        
                        cropped_img.save(img_byte_arr, format='PNG')
                        img_name = f"images/page_{page_index}_{image_counter}.png"
                        zipf.writestr(img_name, img_byte_arr.getvalue())
                        
                        latex_content += f"\\begin{{center}}\n\\includegraphics[width=0.8\\textwidth]{{{img_name}}}\n\\end{{center}}\n\n"
                        image_counter += 1
                    except Exception as e:
                        print(f"Lỗi lưu ảnh: {e}")
                
                else:
                    latex_content += f"{b_content}\n\n"

        latex_content += "\\end{document}"
        
        zipf.writestr("tailieu_chinh.tex", latex_content.encode('utf-8'))
        with open("output/output.tex", "w", encoding="utf-8") as f:
            f.write(latex_content)

    print(f"Hoàn tất! Cấu trúc đã được lưu vào: {zip_output_name}")

if __name__ == "__main__":
    file_dau_vao = r"data\Doc1.pdf" 
    file_zip_dau_ra = "ketqua_paddleocr.zip"
    process_paddle_to_latex_zip(file_dau_vao, file_zip_dau_ra)

# PyMuPDF
pip install pymupdf

# Pillow
pip install pillow

# PaddleOCR + PaddleOCR-VL
pip install paddleocr

# PaddlePaddle (GPU - CUDA 12.6)
# The following command installs the PaddlePaddle version for CUDA 12.6. For other CUDA versions and the CPU version, please refer to https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/



# Nếu dùng CPU thì thay bằng:
# pip install paddlepaddle

# Thư viện hỗ trợ thường dùng
pip install numpy opencv-python tqdm

# Nếu chạy trên Colab/Kaggle nên cài thêm:
pip install shapely pyclipper scikit-image lmdb

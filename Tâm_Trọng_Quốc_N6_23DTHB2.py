# kính đọc sách - bản cải tiến sửa lỗi OCR & hậu xử lý (Tiếng Việt)
import cv2                            # Xử lý ảnh (OpenCV)
import pytesseract                    # Thư viện OCR nhận dạng ký tự trong ảnh
import nltk                           # Xử lý ngôn ngữ tự nhiên (Natural Language Toolkit)
from gtts import gTTS                 # Chuyển văn bản thành giọng nói (Google Text-to-Speech)
import os                             # Thao tác với hệ thống tệp
import numpy as np                    # Xử lý mảng số liệu, tính toán khoa học
import re                             # Xử lý biểu thức chính quy (regular expression)
from pyvi import ViTokenizer          # Tách từ tiếng Việt
import unicodedata                    # Chuẩn hóa ký tự Unicode
from math import atan2, degrees       # Hàm toán học: tính góc và đổi sang độ

# nếu cần, tải punkt (dùng cho fallback nếu muốn)
nltk.download('punkt', quiet=True)

# đường dẫn tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------
# Tiền xử lý ảnh: resize, denoise, tăng tương phản, làm nét, deskew (tùy chọn)
# ------------------------------
def preprocess_image_for_ocr(img, target_width=1600):
    # resize giữ tỉ lệ
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # loại nhiễu nhưng giữ cạnh: bilateralFilter tốt cho văn bản
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # tăng tương phản nhẹ (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # adaptive threshold để tách nền chữ
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 9)

    # morphology: đóng để nối nét bị gãy, rồi mở nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    # (tùy chọn) deskew: ước lượng góc nghiêng và hiệu chỉnh
    coords = np.column_stack(np.where(processed > 0))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:  # chỉ quay nếu nghiêng đáng kể
            (h2, w2) = processed.shape[:2]
            center = (w2 // 2, h2 // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed = cv2.warpAffine(processed, M, (w2, h2),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # sharpen (khi cần)
    kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    processed = cv2.filter2D(processed, -1, kernel_sharp)

    return processed

# ------------------------------
# Mở camera và chụp ảnh (giữ gần với code gốc)
# ------------------------------
def capture_text_images(max_pages=2):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("❌ Không mở được camera.")
        return []

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    captured_files = []
    print(f"📸 Nhấn 'Space' để chụp (tối đa {max_pages} trang), 'Esc' để thoát")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Không nhận được khung hình")
            break

        preview = frame.copy()
        cv2.putText(preview, "Press SPACE to capture, ESC to exit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Camera - Kính đọc sách", preview)
        k = cv2.waitKey(1)

        if k % 256 == 27:  # ESC
            print("👋 Thoát camera!")
            break
        elif k % 256 == 32:  # SPACE
            img_name = f"captured_page_{len(captured_files)+1}.png"
            cv2.imwrite(img_name, frame)
            print(f"✅ Đã lưu ảnh {img_name}")
            captured_files.append(img_name)

            if len(captured_files) >= max_pages:
                print("📚 Đã chụp đủ số trang yêu cầu.")
                break

    cam.release()
    cv2.destroyAllWindows()
    return captured_files

# ------------------------------
# OCR với tiền xử lý và cấu hình Tesseract
# ------------------------------
def ocr_image_to_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không đọc được ảnh:", image_path)
        return ""

    processed = preprocess_image_for_ocr(img, target_width=1600)
    # dùng psm 3 hoặc 6; nếu tài liệu là trang văn bản thì 3 hoặc 6 đều OK
    custom_config = r'--oem 3 --psm 3'
    try:
        text = pytesseract.image_to_string(processed, lang="vie", config=custom_config)
    except Exception as e:
        print("⚠️ Tesseract (lang 'vie') lỗi hoặc chưa cài: ", e)
        text = pytesseract.image_to_string(processed, config=custom_config)

    # hậu xử lý: chuẩn hóa unicode, loại control chars, giữ lại chữ/số/dấu câu thông dụng
    text = unicodedata.normalize('NFC', text)

    # lọc ký tự: chỉ giữ chữ (Unicode category L), số (N) và một số dấu câu phổ biến
    allowed_punct = set(". , : ; ! ? ( ) - — \" ' % / \n".split())
    out_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('L') or cat.startswith('N') or ch.isspace() or ch in allowed_punct:
            out_chars.append(ch)
        # else: loại bỏ ký tự lạ (biểu tượng, control, v.v.)
    cleaned = ''.join(out_chars)

    # thu gọn khoảng trắng + chuẩn hóa xuống dòng: giữ một xuống dòng giữa đoạn
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

# ------------------------------
# Chuẩn hóa & token hóa tiếng Việt (pyvi)
# - ViTokenizer trả về từ nối bằng dấu gạch dưới: "Trí_tuệ"
# - Trước khi đọc bằng gTTS ta sẽ biến _ -> ' ' để giọng đọc tự nhiên
# ------------------------------
def correct_vietnamese_text(text):
    if not text.strip():
        return ""
    try:
        tokenized = ViTokenizer.tokenize(text)  # trả về chuỗi có dấu gạch dưới
    except Exception as e:
        print("⚠️ Lỗi ViTokenizer:", e)
        tokenized = text
    # giữ tokenized cho xử lý NLP nhưng trả về cả phiên bản de-tokenized để đọc
    detokenized = tokenized.replace('_', ' ')
    return tokenized, detokenized

# ------------------------------
# Tách câu cho tiếng Việt (regex đơn giản theo dấu câu)
# ------------------------------
def process_text_with_nlp_for_sentences(text):
    # tách câu theo .!? (giữ dấu) — đơn giản nhưng hiệu quả với văn bản OCR
    pieces = re.split(r'(?<=[\.\?\!])\s+', text)
    # loại các câu quá ngắn
    sentences = [p.strip() for p in pieces if len(p.strip()) > 0]
    return sentences

# ------------------------------
# Thuật toán tham lam: chọn câu theo SỐ TỪ (words) để giới hạn kích thước
# ------------------------------
def greedy_sentence_selection(sentences, max_words=100):
    # loại câu quá ngắn (ví dụ <=2 từ)
    clean_sentences = [s for s in sentences if len(s.split()) > 2]
    # sắp theo độ dài (số từ) giảm dần — ưu tiên câu giàu nội dung
    sorted_sentences = sorted(clean_sentences, key=lambda s: len(s.split()), reverse=True)

    result = []
    total_words = 0
    for sent in sorted_sentences:
        wcount = len(sent.split())
        if total_words + wcount <= max_words:
            result.append(sent)
            total_words += wcount
    # trả về theo thứ tự xuất hiện trong văn bản (đẹp hơn khi đọc)
    result_in_order = [s for s in sentences if s in result]
    return result_in_order

# ------------------------------
# Text to Speech (gTTS) - dùng detokenized text
# ------------------------------
def text_to_speech_vie(text, filename="output.mp3"):
    if not text.strip():
        print("⚠️ Không có văn bản để đọc.")
        return
    try:
        tts = gTTS(text=text, lang="vi")
        tts.save(filename)
        print("🔊 Đã tạo file giọng đọc:", filename)
        # mở file (Windows)
        if os.name == 'nt':
            os.system(f'start "" "{filename}"')
        else:
            os.system(f'xdg-open "{filename}"')
    except Exception as e:
        print("⚠️ Lỗi khi tạo/khởi chạy gTTS:", e)

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    img_paths = capture_text_images(max_pages=2)
    if img_paths:
        full_text = ""
        for img_path in img_paths:
            text = ocr_image_to_text(img_path)
            print(f"\n📖 Văn bản raw sau OCR ({img_path}):\n{text[:1000]}...\n")  # show 1k chars để kiểm tra
            full_text += text + "\n"

        # token hóa và de-token hóa
        tokenized_all, detokenized_all = correct_vietnamese_text(full_text)
        # tách câu trên bản detokenized (để dấu câu tự nhiên hơn)
        sentences = process_text_with_nlp_for_sentences(detokenized_all)

        selected_sentences = greedy_sentence_selection(sentences, max_words=120)
        final_text = " ".join(selected_sentences)

        print("🤖 Văn bản sẽ được đọc (preview):")
        print(final_text)
        text_to_speech_vie(final_text, filename="book_read.mp3")
    else:
        print("Không có ảnh để xử lý.")

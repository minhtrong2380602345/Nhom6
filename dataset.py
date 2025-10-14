#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_pipeline_all_in_one.py
Một script "all-in-one" để:
 - sinh ảnh văn bản tiếng Việt (synthetic)
 - tiền xử lý (deskew, binarize)
 - tăng cường (augmentation cơ bản)
 - lưu nhãn (one .txt per image)
 - (tuỳ chọn) chạy Tesseract OCR trên ảnh và dùng greedy để ghép đoạn
 - (tuỳ chọn) đọc kết quả bằng TTS offline (pyttsx3)

Cấu trúc thư mục (relative to script):
ocr_dataset/
  fonts/           -> chứa .ttf hỗ trợ tiếng Việt
  raw_texts/       -> chứa nhiều .txt mỗi file 1+ dòng văn bản tiếng Việt
  synthetic_images/
  synthetic_labels/
  preprocessed_images/
  aug_images/
  ocr_results/

Usage:
  python ocr_pipeline_all_in_one.py --generate 200
  python ocr_pipeline_all_in_one.py --preprocess
  python ocr_pipeline_all_in_one.py --augment
  python ocr_pipeline_all_in_one.py --ocr --read_tts
  python ocr_pipeline_all_in_one.py --all --n 500

Dependencies:
  pillow, opencv-python, numpy, pytesseract (optional), pyttsx3 (optional)
"""

import os
import sys
import random
import argparse
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2

# Optional imports
try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False

try:
    import pyttsx3
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False

# ---------------------------
# Config / Paths
# ---------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_dataset")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
TEXTS_DIR = os.path.join(BASE_DIR, "raw_texts")
SYN_IMG_DIR = os.path.join(BASE_DIR, "synthetic_images")
SYN_LABEL_DIR = os.path.join(BASE_DIR, "synthetic_labels")
PREP_DIR = os.path.join(BASE_DIR, "preprocessed_images")
AUG_DIR = os.path.join(BASE_DIR, "aug_images")
OCR_DIR = os.path.join(BASE_DIR, "ocr_results")

for d in [BASE_DIR, FONTS_DIR, TEXTS_DIR, SYN_IMG_DIR, SYN_LABEL_DIR, PREP_DIR, AUG_DIR, OCR_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def load_texts(path):
    texts = []
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith(".txt"):
            p = os.path.join(path, fn)
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts

def list_fonts(path):
    fonts = []
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith(".ttf") or fn.lower().endswith(".otf"):
            fonts.append(os.path.join(path, fn))
    return fonts

# ---------------------------
# Synthetic generator
# ---------------------------
def random_background(w, h):
    # create light textured background
    base = Image.new("RGB", (w,h), (255,255,255))
    # add faint noise rectangles
    draw = ImageDraw.Draw(base)
    for i in range(30):
        x1 = random.randint(0,w)
        y1 = random.randint(0,h)
        x2 = min(w, x1 + random.randint(10, w//4))
        y2 = min(h, y1 + random.randint(10, h//6))
        gray = random.randint(245,255)
        draw.rectangle([x1,y1,x2,y2], fill=(gray,gray,gray))
    return base

def render_text_image(text, font_path, out_img_path, out_label_path, img_idx, W=1024, H=512):
    img = random_background(W,H)
    draw = ImageDraw.Draw(img)
    # choose font size based on length
    base_size = 28
    font_size = base_size if len(text) < 100 else max(18, int(base_size*80/len(text)))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    margin = 24
    max_text_width = W - 2*margin
    # break into lines using textwrap but measure width for better wrapping
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        size = draw.textbbox((0,0), test, font=font)
        if size[2] <= max_text_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = margin
    line_gap = int(font_size * 0.28) + 4
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(0,0,0))
        y += font_size + line_gap
        if y > H - margin:
            break
    # small random rotation (simulate capture tilt)
    angle = random.uniform(-2.0, 2.0)
    img = img.rotate(angle, expand=False, fillcolor=(255,255,255))
    # blur sometimes
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0,1.0)))
    # save
    fn = f"synthetic_{img_idx:06d}.png"
    out_path = os.path.join(out_img_path, fn)
    img.save(out_path)
    # write label (GT) as joined lines
    label_fn = fn.replace(".png", ".txt")
    with open(os.path.join(out_label_path, label_fn), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return fn

def generate_synthetic(n_images=500, w=1024, h=512):
    texts = load_texts(TEXTS_DIR)
    fonts = list_fonts(FONTS_DIR)
    if not texts:
        print("Không có file .txt trong raw_texts/. Vui lòng thêm các đoạn văn tiếng Việt.")
        return
    if not fonts:
        print("Không có font .ttf/.otf trong fonts/. Vui lòng thêm font hỗ trợ tiếng Việt.")
        return
    idx = 0
    for i in range(n_images):
        txt = random.choice(texts)
        font = random.choice(fonts)
        fn = render_text_image(txt, font, SYN_IMG_DIR, SYN_LABEL_DIR, idx, W=w, H=h)
        idx += 1
        if idx % 100 == 0:
            print(f"Generated {idx}")
    print("Hoàn tất tạo ảnh synthetic:", idx)

# ---------------------------
# Preprocessing (OpenCV)
# ---------------------------
def deskew_cv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_images(in_dir=SYN_IMG_DIR, out_dir=PREP_DIR):
    n = 0
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        p = os.path.join(in_dir, fn)
        img = cv2.imread(p)
        if img is None: continue
        img = deskew_cv(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # adaptive threshold
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
        # noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        outp = os.path.join(out_dir, fn)
        cv2.imwrite(outp, clean)
        n += 1
        if n % 200 == 0:
            print("Preprocessed", n)
    print("Preprocessing done:", n)

# ---------------------------
# Augmentation (Pillow simple)
# ---------------------------
def augment_image_pil(image: Image.Image):
    img = image.copy()
    # brightness
    if random.random() < 0.6:
        enh = ImageEnhance.Brightness(img)
        img = enh.enhance(random.uniform(0.7,1.3))
    # contrast
    if random.random() < 0.5:
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(random.uniform(0.8,1.4))
    # rotate tiny
    if random.random() < 0.4:
        ang = random.uniform(-1.8,1.8)
        img = img.rotate(ang, expand=False, fillcolor=(255,255,255))
    # slight crop & pad (simulate camera framing)
    if random.random() < 0.3:
        w,h = img.size
        cx = random.randint(0, int(w*0.02))
        cy = random.randint(0, int(h*0.02))
        img = img.crop((cx, cy, w-cx, h-cy)).resize((w,h), Image.LANCZOS)
    return img

def augment_all(in_dir=PREP_DIR, out_dir=AUG_DIR, per_image=2):
    n = 0
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith((".png",".jpg")): continue
        p = os.path.join(in_dir, fn)
        img = Image.open(p).convert("RGB")
        base_name = os.path.splitext(fn)[0]
        for k in range(per_image):
            out_img = augment_image_pil(img)
            out_fn = f"{base_name}_aug{k:02d}.png"
            out_img.save(os.path.join(out_dir, out_fn))
            n += 1
        # also copy original to aug dir (optional)
        img.save(os.path.join(out_dir, f"{base_name}_orig.png"))
    print("Augmentation done:", n)

# ---------------------------
# Greedy postprocess for OCR
# ---------------------------
def greedy_group_boxes(boxes, y_tol=12, x_gap_tol=40):
    """
    boxes: list of tuples (x1,y1,x2,y2,text,score)
    returns: list of joined lines (top->bottom)
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])  # by y1
    used = [False]*len(boxes)
    lines = []
    for i, b in enumerate(boxes):
        if used[i]: continue
        cur_group = [b]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, bb in enumerate(boxes):
                if used[j]: continue
                # vertical overlap / alignment criteria
                y_diff = abs(bb[1] - cur_group[-1][1])
                if y_diff < y_tol or abs(bb[3] - cur_group[-1][3]) < y_tol:
                    max_x = max([g[2] for g in cur_group])
                    if bb[0] - max_x < x_gap_tol:
                        cur_group.append(bb)
                        used[j] = True
                        changed = True
        cur_group = sorted(cur_group, key=lambda g: g[0])
        joined = " ".join([g[4] for g in cur_group])
        lines.append((cur_group[0][1], joined))
    lines = sorted(lines, key=lambda x: x[0])
    return [t for _, t in lines]

# ---------------------------
# OCR using Tesseract (optional)
# ---------------------------
def run_tesseract_on_dir(in_dir=PREP_DIR, out_dir=OCR_DIR, lang="vie"):
    if not HAVE_TESSERACT:
        print("pytesseract không khả dụng — cài pytesseract và tesseract để sử dụng OCR.")
        return
    results = {}
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith((".png",".jpg")): continue
        p = os.path.join(in_dir, fn)
        # Tesseract output as hOCR or box? We'll use simple image_to_data to obtain words + boxes
        try:
            data = pytesseract.image_to_data(p, lang=lang, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print("Tesseract error:", e)
            data = pytesseract.image_to_data(p, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['text'])
        boxes = []
        for i in range(n_boxes):
            txt = data['text'][i].strip()
            if txt == "" or int(data['conf'][i]) < 20:  # drop low-confidence
                continue
            x = int(data['left'][i]); y = int(data['top'][i])
            w = int(data['width'][i]); h = int(data['height'][i])
            conf = float(data['conf'][i])
            boxes.append((x, y, x+w, y+h, txt, conf))
        # greedy group into lines
        lines = greedy_group_boxes(boxes)
        out_txt = os.path.join(out_dir, fn.replace(".png", ".txt"))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        results[fn] = lines
    print("Tesseract OCR done on folder:", in_dir)
    return results

# ---------------------------
# TTS reading (pyttsx3)
# ---------------------------
def read_lines_with_tts(lines, rate=160):
    if not HAVE_TTS:
        print("pyttsx3 không khả dụng. Cài pyttsx3 để dùng TTS offline.")
        return
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    # optionally set voice (system-dependent)
    for line in lines:
        print("[TTS] ", line)
        engine.say(line)
    engine.runAndWait()

# ---------------------------
# CLI / main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OCR dataset pipeline all-in-one (Vietnamese support)")
    p.add_argument("--generate", "-g", type=int, nargs='?', const=100, help="Sinh ảnh synthetic (num)")
    p.add_argument("--preprocess", "-p", action="store_true", help="Tiền xử lý ảnh trong synthetic_images -> preprocessed_images")
    p.add_argument("--augment", "-a", action="store_true", help="Tăng cường ảnh từ preprocessed_images -> aug_images")
    p.add_argument("--ocr", action="store_true", help="Chạy Tesseract OCR trên preprocessed_images (yêu cầu pytesseract)")
    p.add_argument("--read_tts", action="store_true", help="Đọc kết quả OCR bằng TTS (pyttsx3)")
    p.add_argument("--all", action="store_true", help="Chạy tất cả: generate->preprocess->augment->ocr")
    p.add_argument("--n", type=int, default=500, help="số ảnh (khi dùng --generate hoặc --all)")
    p.add_argument("--per_image_aug", type=int, default=2, help="số augment per image")
    return p.parse_args()

def main():
    args = parse_args()
    if args.generate:
        n = args.generate if isinstance(args.generate, int) else args.n
        print("Generating", n, "synthetic images ...")
        generate_synthetic(n_images=n)
    if args.preprocess:
        print("Preprocessing images ...")
        preprocess_images()
    if args.augment:
        print("Augmenting images ...")
        augment_all(per_image=args.per_image_aug)
    if args.ocr:
        print("Running OCR (Tesseract) ...")
        results = run_tesseract_on_dir()
        if args.read_tts and results:
            # concatenate all lines from all images (simple demo)
            all_lines = []
            for k in sorted(results.keys()):
                all_lines += results[k]
            read_lines_with_tts(all_lines)
    if args.all:
        print("Running full pipeline ...")
        generate_synthetic(n_images=args.n)
        preprocess_images()
        augment_all(per_image=args.per_image_aug)
        run_tesseract_on_dir()
    if not any([args.generate, args.preprocess, args.augment, args.ocr, args.all]):
        print("Không có hành động nào được chỉ định. Dùng --help để xem tùy chọn.")
        print("Ví dụ: python ocr_pipeline_all_in_one.py --generate 300 --preprocess --augment --ocr --read_tts")

if __name__ == "__main__":
    main()

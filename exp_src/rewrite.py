import os
import argparse
from tqdm import tqdm

import preprocess.ocr._tesseract as tessocrapp
import preprocess._ours_wo_ocr as ourswoocrapp
import preprocess._ours_wo_mllm as ourswomllmapp
import preprocess._ours_wo_rewrite as oursworewriteapp
import preprocess._ours as oursapp

IMG_DIR = "data/esun_dataset/reference/finance_source_img"
OCR_TEXT_DIR = "result/Tess"

def count_total_images(img_dir):
    total_images = 0
    for pdf_folder in os.listdir(img_dir):
        pdf_folder_path = os.path.join(img_dir, pdf_folder)
        if os.path.isdir(pdf_folder_path):
            total_images += len([f for f in os.listdir(pdf_folder_path) if f.endswith(".png")])
    return total_images

def load_ocr_texts(ocr_text_dir):
    """Load OCR text from files and store them in a dictionary."""
    ocr_texts = {}
    for txt_file in os.listdir(ocr_text_dir):
        if txt_file.endswith(".txt"):
            pdf_folder = os.path.splitext(txt_file)[0]
            with open(os.path.join(ocr_text_dir, txt_file), "r", encoding="utf-8") as file:
                ocr_texts[pdf_folder] = file.read()
    return ocr_texts

def process_images(img_dir, ocr_text_dir, output_dir, task):
    os.makedirs(output_dir, exist_ok=True)

    total_images = count_total_images(img_dir)
    progress_bar = tqdm(total=total_images, desc="Processing Images", unit="page")

    ocr_texts = load_ocr_texts(ocr_text_dir)

    for pdf_folder in os.listdir(img_dir):
        try:
            pdf_folder_path = os.path.join(img_dir, pdf_folder)
            if not os.path.isdir(pdf_folder_path):
                continue

            txt_file_path = os.path.join(output_dir, f"{pdf_folder}.txt")
            all_text = []

            ocr_text = ocr_texts.get(pdf_folder, "")

            image_files = sorted(os.listdir(pdf_folder_path), key=lambda x: int(os.path.splitext(x)[0]))
            for image_file in image_files:
                if not image_file.endswith(".png"):
                    continue

                img_path = os.path.join(pdf_folder_path, image_file)
                with open(img_path, "rb") as img_file:
                    image_bytes = img_file.read()

                    if task == "Tess":
                        response_text = tessocrapp.tessocr(image_bytes)
                    elif task == "Ourswoocr":
                        response_text = ourswoocrapp.ours(image_bytes)
                    elif task == "Ourswomllm":
                        response_text = ourswomllmapp.ours(ocr_text)
                    elif task == "Oursworewrite":
                        response_text = oursworewriteapp.ours(image_bytes, ocr_text)
                    elif task == "Ours":
                        response_text = oursapp.ours(image_bytes, ocr_text)
                    else:
                        raise ValueError("Invalid task.")

                    all_text.append(response_text)

                progress_bar.update(1)

            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write("\n\n".join(all_text))

        except Exception as e:
            print(f'for "{str(pdf_folder)}" Unexpected error: {str(e)}')

    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save as TXT")
    parser.add_argument("--task", choices=["Tess", "Ourswoocr", "Ourswomllm", "Oursworewrite", "Ours"], required=True)
    args = parser.parse_args()

    output_dir = f"result/{args.task}"

    process_images(IMG_DIR, OCR_TEXT_DIR, output_dir, args.task)
    print(f"Results saved in folder: {output_dir}")

import os
from tqdm import tqdm
from pdf2image import convert_from_path

PDF_DIR = "data/esun_dataset/reference/finance_source"
IMG_DIR = "data/esun_dataset/reference/finance_source_img"

def pdf_to_images(pdf_dir, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    for filename in tqdm(pdf_files, desc="Converting PDFs to Images"):
        pdf_path = os.path.join(pdf_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        pdf_output_dir = os.path.join(img_dir, pdf_name)
        
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images, start=1):
            img_path = os.path.join(pdf_output_dir, f"{i}.png")
            image.save(img_path, "PNG")

if __name__ == "__main__":
    pdf_to_images(PDF_DIR, IMG_DIR)
    print(f"Images saved in: {IMG_DIR}")

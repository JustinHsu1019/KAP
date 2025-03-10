import pytesseract
from PIL import Image
from io import BytesIO

def tessocr(image_bytes):
    if image_bytes is None:
        return "Error: No image data provided."

    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang="chi_tra")
        return text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import time
    start_time = time.time()

    image_path = "data/test/image.png"

    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
    except FileNotFoundError:
        image_bytes = None
    
    print(tessocr(image_bytes))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間: {execution_time:.6f} 秒")

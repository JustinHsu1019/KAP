import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import utils.ai.claude_tem as call_ai


def ours(image_bytes, ocr_text):
  PROMPT = f"""請擔任專業的繁體中文知識改寫專家，基於 OCR 轉換後的文本進行修正。

你的任務為以下這點：

1. OCR 錯誤校正
- 修正 OCR 轉換中的錯誤（例如：錯字、漏字、語句順序倒置等），使文本更加流暢且符合語法規則。

---

請基於以下 OCR 後的雜亂文本進行優化改寫：  
{ocr_text}

---

你可以參考附件的圖片協助你理解這段文本在原 PDF 檔案上是以什麼格式 (e.g. 表格、敘述句) 呈現，各個文字、數字又分別代表什麼、呈現在原文哪些位置。

輸出格式：
1. 請輸出「完整」文本，確保文本上的所有內容皆有輸出
2. 請不要輸出任何與文本無關的其他字元
3. 請用繁體中文"""

  response = call_ai.template(PROMPT, image_bytes)
  return response


if __name__ == "__main__":
    import time
    start_time = time.time()

    image_path = "data/test/image.png"
    ocr_text_path = "data/test/ocr_text.txt"

    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
    except FileNotFoundError:
        image_bytes = None

    ocr_text = ""

    try:
        with open(ocr_text_path, "r", encoding="utf-8") as txt_file:
            ocr_text = txt_file.read()
    except FileNotFoundError:
        ocr_text = ""
        print("FileNotFoundError")
    
    print(ours(image_bytes, ocr_text))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間: {execution_time:.6f} 秒")

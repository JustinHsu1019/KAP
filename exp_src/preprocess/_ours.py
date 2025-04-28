import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import utils.ai.claude_tem as call_ai


def ours(image_bytes, ocr_text):
  PROMPT = f"""請擔任專業的繁體中文知識改寫專家，基於 OCR 轉換後的文本進行改寫，使其滿足混合檢索（BM25 + Dense Retrieval）的需求。

你的任務包括以下幾點：

1. OCR 錯誤校正
- 修正 OCR 轉換中的錯誤（例如：錯字、漏字、語句順序倒置等），使文本更加流暢且符合語法規則。

2. 檢索友善改寫（適合 Dense Retrieval）
- 針對「表格型 / 非純敘述型」內容，轉換為易於語意檢索的敘述句。  
  - 例如，若原文本包含發票、財務報表等表格資訊，請改寫為連貫的描述句，確保數據與背景資訊完整：
    - 原始表格內容：
      ```
      日期：2022/03/03  
      公司：XX 公司  
      購買項目：XXX  
      金額：YY 元  
      ```
    - 改寫後：
      「2022 年 3 月 3 日，XX 公司購買了 XXX，總共花費了 YY 元。」
  - 若文本內容雜亂，請優化段落結構，使其更適合語意理解與檢索。

3. 檢索友善改寫（適合 BM25）
- 在「格式優化」後，針對同義詞做檢索友善改寫，在不改變原文本語意的前提下，自然融入同義詞與近義詞。  
  - 例如：
    - 原文： 
      > 該系統可分析數據，提升企業決策能力。  
    - 改寫後（確保原關鍵詞保留，並擴展一般民眾提問時常用的同義詞）： 
      > 該系統能夠分析數據與相關資訊，幫助企業或公司更準確地做出決策與判斷，提升整體經營策略。
  - 避免過度使用同義詞，確保語意不變，且不影響向量搜索的效果。

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

import json
import re
import time
from tqdm import tqdm  # 導入 tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.ai.claude_tem as ai_call

# 讀取原始 JSON 文件
input_file = "data/esun_dataset/dataset/preliminary/question_finance.json"
output_file = "data/esun_dataset/dataset/preliminary/question_finance_augmented.json"

# 讀取原始問題集
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 生成增強問題
augmented_questions = []

# 使用 tqdm 顯示進度條
for item in tqdm(data["questions"], desc="Processing Questions", unit="question"):
    original_query = item["query"]
    
    max_retries = 5
    retries = 0
    success = False
    
    while retries < max_retries:
        # 準備 prompt
        question_augment_prompt = f"""你是一位問題改寫專家，請幫我改寫下方的「問題」，分為以下幾種改寫法：
        1. 將問題中的關鍵字皆換為同義詞（e.g. 變更 -> 改變）
        2. 將問題中的一半關鍵字換為同義詞，另一半維持原關鍵字
        3. 提取出問題的關鍵字：用空格分開
        4. 提取出問題的關鍵字，並將關鍵字皆換為同義詞，用空格分開
        5. 提取出問題的關鍵字，將問題中的一半關鍵字換為同義詞，另一半維持原關鍵字，用空格分開
        6. 改變句型：在不改變語意的前提下，調換主詞和動詞位置
        7. 濃縮語意表達：使用最精簡的文字
        8. 濃縮後的句型變化：用濃縮後的句子，在不影響語意的情況下，改變句型，例如把主詞動詞的位置做調換
        9. 改變問題風格：將問題改為更非正式、休閒的問法

        問題："{original_query}"

        輸出格式（列點輸出）：
        1. 
        2. 
        3. 
        ...
        9.
        """

        response = ai_call.template(question_augment_prompt)
        augmented_queries = response.strip().split("\n")
        
        # 去除前綴數字標號（例如 "1. "、"2. "）
        cleaned_queries = [re.sub(r"^\d+\.\s*", "", query).strip() for query in augmented_queries]
        
        # 確保格式正確
        if len(cleaned_queries) == 9:
            success = True
            break
        
        retries += 1
        wait_time = 2 ** retries  # 指數回退 (2^retries)
        tqdm.write(f"⚠️ 第 {retries} 次重試 ({wait_time}s)：問題 '{original_query}' 產生的變體數量不正確。")
        time.sleep(wait_time)
    
    if not success:
        tqdm.write(f"❌ 跳過問題 '{original_query}'，因為無法成功獲取 9 個變體。")
        continue
    
    # 建立擴增後的問題資料
    augmented_questions.append({
        "qid": item["qid"],
        "source": item["source"],
        "query": original_query,
        "category": item["category"]
    })
    
    for idx, new_query in enumerate(cleaned_queries, start=1):
        augmented_questions.append({
            "qid": f"{item['qid']}_{idx}",  # 為擴增問題加上唯一編號
            "source": item["source"],
            "query": new_query,
            "category": item["category"]
        })

# 輸出新的 JSON 文件
output_data = {"questions": augmented_questions}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"✅ 擴增問題集已儲存至 {output_file}，總問題數：{len(augmented_questions)}")

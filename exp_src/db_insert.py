# import json
import weaviate
import warnings
import time
import uuid
import sys
import os
import argparse
from tqdm import tqdm
import jieba

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.config_log as config_log
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 讀取設定檔與初始化日誌
config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

# 從設定檔讀取 Weaviate 相關參數
wea_url = config.get('Weaviate', 'weaviate_url')
openai_api_key = config.get('OpenAI', 'api_key')


# 忽略所有的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


class WeaviateManager:
    """Weaviate 資料插入管理器"""

    def __init__(self, classnm):
        """初始化 Weaviate 連線，並檢查或建立 class"""
        self.url = wea_url
        self.client = weaviate.Client(url=wea_url, additional_headers={'X-OpenAI-Api-Key': openai_api_key})
        self.classnm = classnm
        self.check_class_exist()

    def check_class_exist(self):
        """檢查指定的 class 是否存在；若不存在則建立之"""
        if self.client.schema.exists(self.classnm):
            print(f'{self.classnm} is ready')
            return True

        schema = {
            'class': self.classnm,
            'properties': [
                {'name': 'uuid', 'dataType': ['text']},
                {'name': 'pid', 'dataType': ['text']},
                {'name': 'content', 'dataType': ['text']},
            ],
            'vectorizer': 'text2vec-openai',
            'moduleConfig': {
                'text2vec-openai': {
                    'model': 'text-embedding-3-large',
                    'dimensions': 3072,
                    'type': 'text'
                }
            },
        }
        print(f'Creating class: {self.classnm} ...')
        self.client.schema.create_class(schema)
        print(f'{self.classnm} is ready')
        return True

    def delete_class(self):
        self.client.schema.delete_class(self.classnm)

    def insert_data(self, pid, content):
        """
        將資料插入 Weaviate。
        加入錯誤處理，包括：
          - 當內容超長時回傳 'TOO_LONG'
          - 429 (速率限制) 與 500 (內部錯誤) 時重試
          - 其他錯誤則停止重試並回傳 False
        """
        data_object = {
            'uuid': str(uuid.uuid4()),
            'pid': pid,
            'content': content
        }
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.client.data_object.create(data_object, self.classnm)
                return True  # 成功插入
            except weaviate.exceptions.UnexpectedStatusCodeException as e:
                error_msg = str(e)
                if 'maximum context length' in error_msg:
                    print(f'Content too long for pid: {pid}. Splitting content.')
                    return 'TOO_LONG'
                elif '429' in error_msg:
                    print(f'Rate limit exceeded, retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})')
                    time.sleep(5)
                elif '500' in error_msg:
                    print(f'Weaviate Internal Server Error (500), retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})')
                    time.sleep(10)
                else:
                    print(f'Unexpected error for pid: {pid} - {error_msg}')
                    return False
            except Exception as e:
                print(f'Error inserting data for pid: {pid}, class: {self.classnm} - {str(e)}')
                return False
        print(f'Failed to insert data for pid: {pid} after {max_retries} attempts.')
        return False


if __name__ == "__main__1":
    """ Delete Classes """
    delete_list = ["Tess", "Ourswoocr", "Ourswomllm", "Oursworewrite", "Ours"]
    for task in delete_list:
        manager = WeaviateManager(task)
        managerkey = WeaviateManager(f"{task}_key")
        print(manager.delete_class())
        print(managerkey.delete_class())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB Insert")
    parser.add_argument("--task", choices=["Tess", "Ourswoocr", "Ourswomllm", "Oursworewrite", "Ours"], required=True)
    args = parser.parse_args()
    manager = WeaviateManager(args.task)
    managerkey = WeaviateManager(f"{args.task}_key")

    directory = f'result/{args.task}'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)

    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    all_chunks = []

    for filename in tqdm(txt_files, desc="Preprocessing Files"):
        file_path = os.path.join(directory, filename)
        pid = os.path.splitext(filename)[0]

        with open(file_path, encoding='utf-8') as file:
            content = file.read()

        chunks = text_splitter.split_text(content)
        all_chunks.extend([(pid, chunk_text) for chunk_text in chunks])

    jieba.set_dictionary('data/dict.txt.big')

    # 統一插入 Weaviate
    for pid, chunk_text in tqdm(all_chunks, desc="Inserting Data"):
        manager.insert_data(pid, chunk_text)
        words = jieba.cut(chunk_text, cut_all=False)
        cont_keyword = ""
        for w in words:
            cont_keyword = cont_keyword + " " + w
        managerkey.insert_data(pid, cont_keyword)
        print(f"Keyword: {cont_keyword}")
        print("========================================")
        print("========================================")

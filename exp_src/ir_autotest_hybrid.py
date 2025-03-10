import json
import os
import weaviate
import concurrent.futures
import utils.config_log as config_log
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
import jieba
import argparse


# 設定 alpha
parser = argparse.ArgumentParser(description="Alpha Setup")
parser.add_argument("--alpha", required=True)
args = parser.parse_args()

alpha = float(args.alpha)


# 設定 jieba 字典
jieba.set_dictionary('data/dict.txt.big')

# 載入 JSON 檔案的函數
def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 設定 Weaviate 和 OpenAI API Key
config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

wea_url = config.get('Weaviate', 'weaviate_url')
os.environ['OPENAI_API_KEY'] = config.get('OpenAI', 'api_key')

# Weaviate 搜尋類別
class WeaviateHybridSearch:
    def __init__(self, vector_class, bm25_class):
        self.url = wea_url
        self.embeddings = OpenAIEmbeddings(chunk_size=1, model='text-embedding-3-large')
        self.client = weaviate.Client(url=wea_url)
        self.vector_class = vector_class
        self.bm25_class = bm25_class

    def vector_search(self, query, source):
        """執行 Vector Search，回傳全部找到的結果"""
        try:
            # 產生 query 向量
            query_vector = self.embeddings.embed_query(query)
            vector_str = ','.join(map(str, query_vector))

            # 建立 GraphQL 條件
            where_conditions = ' '.join([f'{{path: ["pid"], operator: Equal, valueText: "{pid}"}}' for pid in source])

            # 防止特殊字元影響 GraphQL
            query_safe = json.dumps(query, ensure_ascii=False)

            gql_query = f"""
            {{
                Get {{
                    {self.vector_class}(where: {{
                        operator: Or,
                        operands: [{where_conditions}]
                    }}, hybrid: {{
                        query: {query_safe},
                        vector: [{vector_str}],
                        alpha: 1
                    }}) {{
                        pid
                        content
                        _additional {{
                            distance
                            score
                        }}
                    }}
                }}
            }}
            """
            search_results = self.client.query.raw(gql_query)

            if 'errors' in search_results:
                raise Exception(search_results['errors'][0]['message'])

            return search_results['data']['Get'][self.vector_class]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def bm25_search(self, query, source):
        """執行 BM25 Search，回傳全部找到的結果"""
        try:
            # 使用 jieba 斷詞
            words = jieba.cut(query, cut_all=False)
            cont_keyword = " ".join(words)

            # 建立 GraphQL 條件
            where_conditions = ' '.join([f'{{path: ["pid"], operator: Equal, valueText: "{pid}"}}' for pid in source])

            # 防止特殊字元影響 GraphQL
            query_safe = json.dumps(cont_keyword, ensure_ascii=False)

            gql_query = f"""
            {{
                Get {{
                    {self.bm25_class}(where: {{
                        operator: Or,
                        operands: [{where_conditions}]
                    }}, hybrid: {{
                        query: {query_safe},
                        alpha: 0
                    }}) {{
                        pid
                        content
                        _additional {{
                            score
                        }}
                    }}
                }}
            }}
            """
            search_results = self.client.query.raw(gql_query)

            if 'errors' in search_results:
                raise Exception(search_results['errors'][0]['message'])

            return search_results['data']['Get'][self.bm25_class]
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    def hybrid_search(self, query, source, alpha):
        """執行自訂 alpha 的 Hybrid Search，回傳 Top-3 結果"""
        vector_results = self.vector_search(query, source)
        bm25_results = self.bm25_search(query, source)

        # 計算 Aggregative Count
        num = len(vector_results) + len(bm25_results)
        if num == 0:
            return []

        # 建立結果字典，合併 vector 與 bm25 結果
        result_dict = {}

        for res in vector_results:
            pid = res['pid']
            vector_score = res['_additional']['score']
            result_dict[pid] = {'content': res['content'], 'vector_score': vector_score, 'bm25_score': 0}

        for res in bm25_results:
            pid = res['pid']
            bm25_score = res['_additional']['score']
            if pid in result_dict:
                result_dict[pid]['bm25_score'] = bm25_score
            else:
                result_dict[pid] = {'content': res['content'], 'vector_score': 0, 'bm25_score': bm25_score}

        # 計算最終分數
        for pid in result_dict:
            result_dict[pid]['vector_score'] = float(result_dict[pid]['vector_score'])
            result_dict[pid]['bm25_score'] = float(result_dict[pid]['bm25_score'])
            result_dict[pid]['final_score'] = alpha * result_dict[pid]['vector_score'] + (1 - alpha) * result_dict[pid]['bm25_score']

        # 排序並回傳 Top-3 結果
        sorted_results = sorted(result_dict.items(), key=lambda x: x[1]['final_score'], reverse=True)
        return [{'pid': pid, 'content': data['content'], 'score': data['final_score']} for pid, data in sorted_results[:3]]

# 載入資料集
question_file = "data/esun_dataset/dataset/preliminary/question_finance_augmented.json"
ground_truth_file = "data/esun_dataset/dataset/preliminary/ground_truths_finance.json"
questions = load_json(question_file)['questions']
ground_truths = {item['qid']: item['retrieve'] for item in load_json(ground_truth_file)}

tasks = ["Tess", "Ourswoocr", "Ourswomllm", "Oursworewrite", "Ours"]
results_metrics = {}

def evaluate_task(task, alpha):
    """
    計算特定 task 的三項指標：
    1. AP@1 (top-1 accuracy)
    2. MRR (第一個正確答案的倒數排名)
    """
    total = 0
    ap1_correct = 0
    mrr_total = 0
    searcher = WeaviateHybridSearch(vector_class=task, bm25_class=f"{task}_key")

    for q in tqdm(questions, desc=f"Processing {task}"):
        qid = q['qid']
        query = q['query']
        source = q['source']
        expected = ground_truths.get(qid)

        top_results = searcher.hybrid_search(query, source, alpha)
        total += 1

        if top_results:
            # AP@1: 只看第一筆結果
            if str(top_results[0]['pid']) == str(expected):
                ap1_correct += 1

            # MRR: 找出第一個出現正確答案的排名
            mrr_value = 0
            for rank, res in enumerate(top_results, start=1):
                if str(res['pid']) == str(expected):
                    mrr_value = 1.0 / rank
                    break
            mrr_total += mrr_value

    # 計算平均指標
    ap1 = ap1_correct / total if total > 0 else 0
    mrr = mrr_total / total if total > 0 else 0

    metrics = {'AP@1': ap1, 'MRR': mrr}
    return task, metrics

# 多線程執行各個 task 的評估
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_task = {executor.submit(evaluate_task, task, alpha): task for task in tasks}
    for future in concurrent.futures.as_completed(future_to_task):
        task, metrics = future.result()
        results_metrics[task] = metrics

# 分別整理三個指標的結果
ap1_results = {task: metrics['AP@1'] for task, metrics in results_metrics.items()}
mrr_results = {task: metrics['MRR'] for task, metrics in results_metrics.items()}

today = datetime.today().strftime("%m%d")

# 畫出 AP@1 圖表
plt.figure(figsize=(10, 6))
plt.bar(ap1_results.keys(), ap1_results.values(), alpha=0.7)
plt.xlabel('Tasks')
plt.ylabel('AP@1 (Top-1 Accuracy)')
plt.title(f'AP@1 Comparison (alpha={alpha})')
plt.ylim(0, 1)
for i, v in enumerate(ap1_results.values()):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=12)
plt.savefig(f"result/ap1_comparison_{today}_hybrid_alpha{alpha}.png", dpi=300, bbox_inches='tight')
plt.show()

# 畫出 MRR 圖表
plt.figure(figsize=(10, 6))
plt.bar(mrr_results.keys(), mrr_results.values(), alpha=0.7)
plt.xlabel('Tasks')
plt.ylabel('MRR')
plt.title(f'MRR Comparison (alpha={alpha})')
plt.ylim(0, 1)
for i, v in enumerate(mrr_results.values()):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=12)
plt.savefig(f"result/mrr_comparison_{today}_hybrid_alpha{alpha}.png", dpi=300, bbox_inches='tight')
plt.show()

# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/3/11 18:52
:last_date:
    2026/3/11 18:52
:description:

"""
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from utils.common_utils import read_json


# ================= 核心功能函数 =================

def init_model_and_db(model_name_or_path, db_path=r'W:\project\python_project\auto_video\utils\taobao\product_db', collection_name="my_products", device="cuda"):
    """
    初始化模型与数据库，返回实例供后续函数调用（只需在程序启动时执行一次）。
    提示：如果 model_name_or_path 传入的是本地有效的文件夹路径，将自动以离线模式加载模型。
    """
    print(f"正在加载模型与数据库连接: {model_name_or_path} ...")
    model = SentenceTransformer(model_name_or_path, device=device)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return model, collection


def add_products_from_dict(goods_dict, model, collection, batch_size=5000):
    """
    将包含商品信息的字典向量化并分批入库（专为大数据量优化）
    """
    if not goods_dict or not isinstance(goods_dict, dict):
        return {"status": "error", "msg": "商品数据为空或格式不正确"}

    existing_ids = set(collection.get(include=[])['ids'])
    ids_to_add, documents_to_add, metadatas_to_add = [], [], []

    print("正在解析并过滤已存在的商品...")
    for item_id, item_data in goods_dict.items():
        product_id = str(item_id).strip()

        # 【新增这行防御代码】如果 ID 是空的，直接无视这条脏数据
        if not product_id:
            # 你也可以加个 print 看看是哪条数据坏了: print(f"发现空ID脏数据，已跳过: {item_data.get('item_name')}")
            continue

        if product_id in existing_ids:
            continue

        ids_to_add.append(product_id)
        doc_str = f"{item_data.get('item_name', '')} {item_data.get('category_name', '')} {item_data.get('short_title', '')}"
        documents_to_add.append(doc_str)

        metadata = {}
        for k, v in item_data.items():
            if isinstance(v, (int, float, bool)):
                metadata[k] = v
            elif v is None:
                metadata[k] = ""
            else:
                metadata[k] = str(v)
        metadatas_to_add.append(metadata)

    total_new = len(ids_to_add)
    if total_new == 0:
        return {"status": "success", "msg": "没有新商品，无需更新", "added_count": 0}

    print(f"发现 {total_new} 个新商品，准备开始向量化与入库...")

    # 【核心优化】分批处理，防止 ChromaDB 撑爆内存或超出单次插入限制
    for i in range(0, total_new, batch_size):
        end_idx = min(i + batch_size, total_new)
        batch_ids = ids_to_add[i:end_idx]
        batch_docs = documents_to_add[i:end_idx]
        batch_metas = metadatas_to_add[i:end_idx]

        print(f"👉 正在处理第 {i + 1} 到 {end_idx} 条数据 (共 {total_new} 条)...")

        # show_progress_bar=True 会在控制台显示编码进度条，让你心里有数
        embeddings = model.encode(
            batch_docs,
            batch_size=64,  # 编码的内部批次，如果显存/内存大可以调到 128 或 256
            normalize_embeddings=True,
            show_progress_bar=True
        )

        collection.add(
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

    print(f"🎉 全部 {total_new} 个新商品入库完成！")
    return {"status": "success", "msg": "入库完成", "added_count": total_new}

def search_goods(keywords, model, collection, top_n=5):
    """
    核心搜索逻辑：
    输入关键词列表，返回去重后（保留最高相似度）且按相似度降序排列的商品列表。
    （完全独立的函数，只要外部提供 model 和 collection 即可直接调用）
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    all_results = []
    # 1. 遍历收集所有关键词的搜索结果
    for kw in keywords:
        query_embedding = model.encode(kw, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )

        if results and results['ids'][0]:
            for i, item_id in enumerate(results['ids'][0]):
                all_results.append({
                    "id": item_id,
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1 - results['distances'][0][i]  # 余弦距离转相似度
                })

    # 2. 按 item_id 智能去重：如果不同词搜出同一个商品，保留相似度得分最高的那次
    unique_products = {}
    for item in all_results:
        metadata = item.get('metadata', {})
        product_id = metadata.get('item_id')

        if product_id:
            # 如果是新商品，或者当前相似度比记录在案的更高，则更新覆盖
            if product_id not in unique_products or item['similarity'] > unique_products[product_id].get(
                    'search_similarity', 0):
                # 将相似度得分也塞进 metadata，方便前端展示或后续排序
                metadata['search_similarity'] = item['similarity']
                # 新增搜索的词
                metadata['matched_keyword'] = kw
                unique_products[product_id] = metadata

    # 3. 按相似度从高到低排序后返回
    sorted_results = sorted(unique_products.values(), key=lambda x: x.get('search_similarity', 0), reverse=True)
    return sorted_results


# ================= 调用示例 =================


def update_and_search(keyword_list, top_n=5):
    MODEL_PATH = r"C:\Users\zxh\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    BASE_DIR = r'W:\project\python_project\auto_video\utils\temp\goods'
    all_goods_info_file = os.path.join(BASE_DIR, "all_goods_info_with_score.json")

    all_goods_info = read_json(all_goods_info_file)
    global_model, global_collection = init_model_and_db(MODEL_PATH)

    # 2. 数据更新（传入 dict 替代原来的 CSV）
    if all_goods_info:
        add_status = add_products_from_dict(all_goods_info, global_model, global_collection)
        print("入库状态:", add_status)
    else:
        print("未读取到商品数据或文件为空。")

    # 3. 业务代码中独立、高频调用的搜索逻辑
    search_results = search_goods(keyword_list, global_model, global_collection, top_n=top_n)

    return search_results



if __name__ == "__main__":

    search_results = update_and_search(keyword_list=["英雄联盟联名", "电竞周边", "游戏挂件"], top_n=5)
    for res in search_results:
        # 字段变更为 item_name
        print(f"商品: {res.get('item_name', '未知')}, 相似度: {res.get('search_similarity'):.4f}")
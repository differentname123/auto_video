import asyncio
import os
import time
import hashlib
import json
import logging
from pathlib import Path
from urllib import parse, request

BASE_DIR = r'W:\project\python_project\auto_video\utils\temp\goods'
KEYWORD_HISTORY_FILE = os.path.join(BASE_DIR, 'keyword_history.json')

from utils.common_utils import read_json, download_cover_minimal, save_json

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_sign(secret, params):
    """生成淘宝 API 签名 (MD5 算法)"""
    sorted_params = sorted(params.items())
    query_str = secret
    for k, v in sorted_params:
        query_str += f"{k}{v}"
    query_str += secret
    m = hashlib.md5()
    m.update(query_str.encode('utf-8'))
    return m.hexdigest().upper()


def get_tbk_material(app_key, app_secret, adzone_id, q=None, material_id=80309, desire_count=50, **kwargs):
    """获取淘宝客商品物料，脱离官方 SDK 手动实现"""
    url = "http://gw.api.taobao.com/router/rest"
    all_map_data = []
    page_no = 1

    kwargs.pop('page_no', None)
    kwargs.pop('page_size', None)

    while len(all_map_data) < desire_count:
        remaining_count = desire_count - len(all_map_data)
        current_page_size = min(100, remaining_count)

        params = {
            "method": "taobao.tbk.dg.material.optional.upgrade",
            "app_key": app_key,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "format": "json",
            "v": "2.0",
            "sign_method": "md5",
            "adzone_id": str(adzone_id),
            "material_id": str(material_id),
            "page_no": str(page_no),
            "page_size": str(current_page_size)
        }

        if q:
            params["q"] = q

        for k, v in kwargs.items():
            if isinstance(v, bool):
                params[k] = "true" if v else "false"
            else:
                params[k] = str(v)

        params["sign"] = generate_sign(app_secret, params)
        res_data = ""  # 提前初始化，防止异常时 UnboundLocalError

        try:
            data = parse.urlencode(params).encode('utf-8')
            req = request.Request(url, data=data)
            with request.urlopen(req) as response:
                res_data = response.read().decode('utf-8')
                res_json = json.loads(res_data)

                if "error_response" in res_json:
                    err = res_json["error_response"]
                    logger.error(
                        f"[API业务错误-物料搜索] 停止拉取 {q} | 错误码: {err.get('code')} | 描述: {err.get('msg')} | 子错误码: {err.get('sub_code')} | 子描述: {err.get('sub_msg')}")
                    break

                resp_node = res_json.get("tbk_dg_material_optional_upgrade_response", {})
                result_list = resp_node.get("result_list") or {}

                if isinstance(result_list, dict):
                    map_data = result_list.get("map_data", [])
                else:
                    map_data = []

                if not map_data:
                    break

                for item in map_data:
                    price_info = item.get("price_promotion_info", {})
                    basic_info = item.get("item_basic_info", {})
                    publish_info = item.get("publish_info", {})
                    income_info = publish_info.get("income_info", {})
                    item_id = item.get("item_id", "")

                    extracted_item = {
                        "item_id": item_id,
                        "item_name": basic_info.get("title", ""),
                        "brand": basic_info.get("brand_name", ""),
                        "pict_url": basic_info.get("pict_url", ""),
                        "category_name": basic_info.get("category_name", ""),
                        "level_one_category_name": basic_info.get("level_one_category_name", ""),
                        "shop_title": basic_info.get("shop_title", ""),
                        "short_title": basic_info.get("short_title", ""),
                        "original_price": price_info.get("reserve_price", ""),
                        "final_price": price_info.get("final_promotion_price", price_info.get("zk_final_price", "")),
                        "commission_rate": income_info.get("commission_rate", ""),
                        "commission_amount": income_info.get("commission_amount", ""),
                        "click_url": publish_info.get("coupon_share_url", publish_info.get("click_url", "")),
                        "raw_data": item
                    }
                    all_map_data.append(extracted_item)

                page_no += 1

        except json.JSONDecodeError as e:
            logger.error(f"[JSON解析异常-物料搜索] 停止拉取 | 详情: {e} | 原始返回: {res_data}")
            break
        except Exception as e:
            logger.error(f"[网络或系统异常-物料搜索] 停止拉取 | 详情: {e}")
            break

    return all_map_data[:desire_count]


def create_tbk_tpwd(app_key, app_secret, target_url):
    """根据目标链接生成淘口令"""
    url = "http://gw.api.taobao.com/router/rest"

    if target_url and target_url.startswith("//"):
        target_url = "https:" + target_url

    params = {
        "method": "taobao.tbk.tpwd.create",
        "app_key": app_key,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "format": "json",
        "v": "2.0",
        "sign_method": "md5",
        "url": target_url
    }

    params["sign"] = generate_sign(app_secret, params)

    try:
        data = parse.urlencode(params).encode('utf-8')
        req = request.Request(url, data=data)
        with request.urlopen(req) as response:
            res_data = response.read().decode('utf-8')
            res_json = json.loads(res_data)

            if "error_response" in res_json:
                err = res_json["error_response"]
                logger.error(
                    f"[API业务错误-淘口令生成] 失败 | 错误码: {err.get('code')} | 描述: {err.get('msg')} | 子错误码: {err.get('sub_code')} | 子描述: {err.get('sub_msg')}")
                return None

            resp_node = res_json.get("tbk_tpwd_create_response", {})
            return resp_node.get("data", {})

    except Exception as e:
        logger.error(f"[网络或系统异常-淘口令生成] 详情: {e}")
        return None


def batch_append_tpwd(app_key, app_secret, item_list):
    """批量为商品列表追加淘口令信息"""
    for item in item_list:
        link = item.get("click_url")
        if link:
            tpwd_result = create_tbk_tpwd(app_key, app_secret, link)
            if tpwd_result:
                item["tpwd_simple"] = tpwd_result.get("password_simple", "")
                item["tpwd_model"] = tpwd_result.get("model", "")
            else:
                item["tpwd_simple"] = "生成失败"
                item["tpwd_model"] = "生成失败"
        else:
            item["tpwd_simple"] = "无推广链接"
            item["tpwd_model"] = "无推广链接"

    return item_list


def get_all_keywords(data):
    """安全、简洁地从复杂字典中提取关键词"""
    keyword_set = set()
    for video_data in data.values():
        try:
            recommendations = video_data.get("final_goods", {}).get("product_recommendations", [])
            for product in recommendations:
                keyword_set.update(product.get("keywords", []))
        except Exception as e:
            # logger.warning(f"⚠️ 处理视频数据时发生异常，已跳过该条数据 | 详情: {e}")
            continue
    return list(keyword_set)


def update_all_goods():
    """根据已有的关键词进行商品信息的拉取（增加 24h 缓存跳过功能）"""
    all_goods_info_file = os.path.join(BASE_DIR, "all_goods_info.json")

    # 初始化文件内容，防止原先没有文件时读取报错
    all_goods_info = read_json(all_goods_info_file) if os.path.exists(all_goods_info_file) else {}
    keyword_history = read_json(KEYWORD_HISTORY_FILE) if os.path.exists(KEYWORD_HISTORY_FILE) else {}

    json_file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.json') and 'replay_goods_inf' in file:
                json_file_list.append(os.path.join(root, file))

    keyword_set = set()
    for json_file in json_file_list:
        file_data = read_json(json_file)
        if file_data:
            keyword_set.update(get_all_keywords(file_data))

    keyword_list = list(keyword_set)
    logger.info(f"扫描到 {len(json_file_list)} 个商品账号信息文件。提取到了 {len(keyword_list)} 个去重关键词")

    current_time = time.time()

    for keyword in keyword_list:
        # 核心逻辑：检查是否在 24 小时内拉取过
        history_record = keyword_history.get(keyword)
        if history_record:
            last_pull_time = history_record.get("last_time", 0)
            if current_time - last_pull_time < 24 * 3600:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_pull_time))
                logger.info(
                    f"⏭️ [跳过] 关键词 '{keyword}' 最近拉取时间为 {time_str} (获取 {history_record.get('count', 0)} 个)，未满24小时。")
                continue

        start_time = time.time()
        goods_info_list = get_goods_info(keyword)

        for item in goods_info_list:
            item_id = item.get("item_id")
            all_goods_info[item_id] = item

        # 更新全局缓存和记录
        save_json(all_goods_info_file, all_goods_info)

        keyword_history[keyword] = {
            "last_time": time.time(),
            "count": len(goods_info_list)
        }
        save_json(KEYWORD_HISTORY_FILE, keyword_history)

        logger.info(
            f"✅ [更新成功] 关键词 '{keyword}' 获取商品数量: {len(goods_info_list)} | 当前总商品数: {len(all_goods_info)} | 耗时: {time.time() - start_time:.2f} 秒")


def get_goods_info(key_word, desire_count=100):
    """根据关键词拉取商品信息"""
    APP_KEY = "35288611"
    APP_SECRET = "3f861aab50cd92d29870becf34d6fc64"
    ADZONE_ID = 116238500456

    logger.info(f"🔄 开始处理关键词 '{key_word}' ...")

    # 第一步：先搜索商品
    result_list = get_tbk_material(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        adzone_id=ADZONE_ID,
        q=key_word,
        desire_count=desire_count
    )

    if not result_list:
        logger.warning(f"⚠️ 关键词 '{key_word}' 未获取到任何商品数据。")
        return []

    # 第二步：为获取到的商品追加生成淘口令并处理图片
    result_list = batch_append_tpwd(APP_KEY, APP_SECRET, result_list)
    result_list = update_image(result_list)

    current_timestamp = int(time.time())
    for item in result_list:
        item["update_timestamp"] = current_timestamp

    logger.info(f"🎯 关键词 '{key_word}' 处理完毕，有效返回 {len(result_list)} 个商品。")
    return result_list


# 提取并发执行的辅助方法，避免混淆原有的导入结构
async def _gather_downloads(tasks):
    """并发执行多个下载任务"""
    return await asyncio.gather(*tasks)


def update_image(result_list):
    """批量更新/下载图片 (优化为异步并发)"""
    save_dir = Path(BASE_DIR) / 'images'
    save_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    task_indices = []

    # 1. 第一遍筛选：过滤出本地不存在，真正需要下载的图片
    for i, item in enumerate(result_list):
        image_url = item.get("pict_url", "")
        if not image_url:
            item["local_image_path"] = None
            continue

        name_seed = f"{image_url}".encode('utf-8')
        file_hash = hashlib.md5(name_seed).hexdigest()
        save_path = save_dir / f"{file_hash}.jpg"

        if save_path.exists():
            item["local_image_path"] = str(save_path.resolve())
        else:
            item["local_image_path"] = None  # 预先置空
            download_tasks.append(download_cover_minimal(image_url, save_path))
            task_indices.append((i, save_path))

    # 2. 并发下载缺失的图片
    if download_tasks:
        results = asyncio.run(_gather_downloads(download_tasks))

        # 3. 将成功下载的结果填回字典
        for (original_index, save_path), is_success in zip(task_indices, results):
            if is_success:
                result_list[original_index]["local_image_path"] = str(save_path.resolve())

    return result_list


# ================= 使用示例 =================
if __name__ == "__main__":
    update_all_goods()
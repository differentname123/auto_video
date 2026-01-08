#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import re
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import requests
import time
import logging
import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import json
import threading
from queue import Queue, Empty


from application.video_common_config import ALL_BILIBILI_EMOTE_PATH, TaskStatus
from utils.bilibili.bili_utils import update_bili_user_sign
from utils.bilibili.comment import BilibiliCommenter
from utils.bilibili.watch_video import watch_video
from utils.common_utils import get_config, read_json, init_config
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

bvid_file_path = '../../LLM/TikTokDownloader/back_up/bvid_file.json'
all_bvid_file_path = '../../LLM/TikTokDownloader/back_up/all_bvid_file.json'

interaction_data_file = '../../LLM/TikTokDownloader/back_up/interaction_data.json'

# --- 1. 全局常量 ---
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# --- 2. 全局配置 ---
total_cookie = get_config("nana_bilibili_total_cookie")
csrf_token = get_config("nana_bilibili_csrf_token")

CONFIG = {
    "STRATEGIES": {
        "popular": False,  # 热门视频通常不是目标用户，可以关闭
        "following": True,  # 已经关注的UP主不需要再处理
        "search": False,
        "ranking": False,  # <<< NEW: 新增分区排行榜策略开关
    },
    "COOKIE": total_cookie,
    "CSRF_TOKEN": csrf_token,
    "TARGET_UIDS": [  # 监控动态时使用，当前已关闭
        "1223805908",
        "1639172564",
        "3546909677455941",
        "3546717871934392",
    ],
    # <<< NEW: START - 新增分区排行榜相关配置 >>>
    "RANKING_TIDS": {  # 目标分区ID (rid) 和名称的映射
        0: "全站",
        1: "动画",
        168: "国创",
        3: "音乐",
        129: "舞蹈",
        4: "游戏",
        36: "知识",
        188: "科技",
        234: "运动",
        223: "汽车",
        160: "生活",
        211: "美食",
        217: "动物圈",
        119: "鬼畜",
        155: "时尚",
        5: "娱乐",
        181: "影视",
    },
    # <<< NEW: END - 新增分区排行榜相关配置 >>>
    "TARGET_KEYWORDS": [
        "互关", "互粉", "互赞", "互助", "新人UP主", "回关", "回粉", "互暖",
        "互评", "互捞", "三连", "求三连", "互三连", "互币", "新人报道", "新人up",
        "小UP主", "萌新UP", "底层UP主", "小透明", "涨粉", "求关注", "求抱团",
        "抱团取暖", "一起加油", "挑战100粉", "冲击千粉", "有粉必回", "有赞必回",
        "在线秒回", "已关求回"
    ],
    "FOLLOW_KEYWORDS": [
        "互关", "互粉", "回关", "互赞", "互助", "回粉", "必回", "必回关",
        "有粉必回", "有访必回", "诚信互关", "诚信互粉", "永不取关", "不取关",
        "赞评必回", "互赞互评", "互三连", "互币", "关我必回", "私信秒回",
        "你关我就关"
    ],
    "MAX_VIDEOS_PER_SOURCE": 20,  # 每次搜索/每个分区排行可以多拉取一些
    "PROCESSED_VIDEOS_FILE": "comment_processed_bvideos.json",
    "GEN_PROCESSED_VIDEOS_FILE": "gen_comment_processed_bvideos.json",
    "COMMENTED_PROCESSED_VIDEOS_FILE": "commented_processed_bvideos.json",

    "PROCESSED_FIDS_FILE": "processed_fids.json",  # 新增：记录已处理的用户ID
    "REQUEST_TIMEOUT": 10,
    "REQUEST_DELAY": 1,
}

# --- 3. 日志与会话配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 创建一个全局会话对象，用于保持登录状态
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Cookie': CONFIG['COOKIE']
})

import difflib
from typing import List, Optional


def most_similar_text(text_list: List[str], target_text: str) -> Optional[str]:
    """
    返回 text_list 中与 target_text 最为相似的字符串。
    """
    if not text_list:
        return None

    best_match = '[吃瓜]'
    best_score = -1.0
    for text in text_list:
        score = difflib.SequenceMatcher(None, text, target_text).ratio()
        if score > best_score:
            best_score = score
            best_match = text

    return best_match


def replace_bracketed(text: str, text_list: List[str]) -> str:
    """
    找到 text 中所有被 [ 和 ] 包围的子串。
    对于前5个子串，提取其中内容 item，用 most_similar_text(text_list, item) 的返回值去替换整个 [item]。
    对于第6个及以后的 [ 和 ] 子串，直接删除。

    :param text: 包含若干 […] 片段的原始字符串
    :param text_list: 用于匹配的候选字符串列表
    :return: 处理后的新字符串
    """

    # 在外部函数作用域定义一个计数器
    match_count = 0

    # 回调函数：为每一个匹配项计算替换结果
    def _replacer(match: re.Match) -> str:
        nonlocal match_count
        match_count += 1

        # 如果是前5个匹配项，执行替换逻辑
        if match_count <= 5:
            inner = match.group(1)
            best = most_similar_text(text_list, inner)
            # 如果没找到任何匹配，保留原括号内容
            return best if best is not None else match.group(0)
        # 如果是第6个及以后的匹配项，返回空字符串，即删除该匹配
        else:
            return ""

    # 使用正则替换所有 [内容]
    # re.sub 会对每一个匹配项调用一次 _replacer 函数
    return re.sub(r'\[([^\]]+)\]', _replacer, text)


# --- 4. API请求核心函数 ---
def send_get_request(url, params=None):
    """通用GET请求函数"""
    try:
        # 每次API请求前，随机暂停
        time.sleep(random.uniform(1.5, 3.5))
        response = session.get(url, params=params, timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        data = response.json()
        if data.get('code', 0) != 0:
            logging.warning(f"API返回错误: code={data.get('code')}, message={data.get('message')}, url={response.url}")
            return None
        return data.get('data')
    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求失败: {e}")
    except json.JSONDecodeError:
        logging.error("无法解析服务器返回的JSON数据。")
    return None


comment_danmu = [
    "本来已经划走了，结果看到一个评论，还是没忍住回来点赞。",
    "退出去又被评论区炸回来了，你们是魔鬼吗？",
    "谢谢评论区，差点就错过这个视频的精髓了。",

    # “认知颠覆”视角 (暗示评论区有惊天发现或不同解读)
    "看完视频一脸问号，看完评论区一句卧槽。",
    "我以为我懂了，直到我打开了评论区。",
    "这个视频需要搭配评论区“食用”，风味更佳。",

    # “强烈推荐”视角 (用个人感受为评论区的精彩程度背书)
    "评论区第一条直接给我干沉默了。",
    "你们去看评论区那个热评，我笑到打嗝。",
    "听说评论区比视频还精彩，特来围观。",

    "视频还没把我怎么样，评论区差点把我笑走。",
    "我宣布，这里是第一现场，评论区是第二现场！",
    "这个视频的弹幕一半，评论区一半，UP主只负责上传。",
]


def modify_relation(fid, action_type, csrf_token):
    """
    修改用户关系 (关注或取消关注)。
    fid: 目标用户的UID
    action_type: 1 为关注, 2 为取消关注
    csrf_token: 从Cookie中获取的bili_jct值
    """
    action_text = "关注" if action_type == 1 else "取消关注"
    payload = {
        "fid": fid,
        "act": action_type,
        "re_src": 11,  # 关系来源，通常用 11
        "csrf": csrf_token
    }
    try:
        response = session.post(URL_MODIFY_RELATION, data=payload, timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        result = response.json()
        if result.get('code') == 0:
            logging.info(f"  {'✅' if action_type == 1 else '🗑️'} 成功{action_text} UID: {fid}")
            return True
        # 常见错误码处理
        elif result.get('code') == 22014:  # 对方将你拉黑
            logging.warning(f"  ⚠️ {action_text} UID: {fid} 失败: {result['message']} (可能已被对方拉黑)")
            return True  # 返回True，避免重试
        elif result.get('code') == 22007:  # 已经关注了
            logging.info(f"  ℹ️ {action_text} UID: {fid}: 已经是关注状态。")
            return True  # 返回True，避免重试
        else:
            logging.error(
                f"  ❌ {action_text} UID: {fid} 失败: {result.get('message', '未知错误')} (Code: {result.get('code')})")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"  ❌ 请求{action_text} UID: {fid} 失败: {e}")
        return False
    except ValueError:  # 对应 json.JSONDecodeError
        logging.error(f"  ❌ {action_text} UID: {fid} 响应内容不是有效的 JSON。")
        return False


# --- 5. 视频获取策略实现 ---
def fetch_from_popular(max_count=100):
    """
    循环获取B站热门榜单的视频，直到没有更多数据为止。
    """
    logging.info("开始执行 [策略一：获取热门视频]...")

    # 将 video_list 初始化在循环外部，用于累加所有页的数据
    all_videos = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    page_number = 1  # 从第一页开始

    while True:
        logging.info(f"  > 正在尝试获取热门榜单第 {page_number} 页...")
        params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': page_number}

        data = send_get_request(url, params)

        # 检查API响应是否成功，并且 'list' 键存在且不为空
        if data and 'list' in data and data['list']:
            page_videos = data['list']
            for item in page_videos:
                if 'bvid' in item:
                    item['_source_strategy'] = 'popular'
                    all_videos.append(item)

            logging.info(f"  > 成功从第 {page_number} 页获取 {len(page_videos)} 个视频。")
            if len(all_videos) >= max_count:
                logging.info(f"  > 已达到最大获取数量 {max_count}，停止获取更多数据。")
                break
            # 准备获取下一页
            page_number += 1

            # 增加延时，避免请求过快被API限制。可根据需要调整时间。
            time.sleep(1)
        else:
            # 如果 'list' 不存在、为空，或者API请求失败，则认为没有更多数据了
            logging.info("  > 热门榜单数据已全部获取完毕，或API未返回有效数据，停止获取。")
            break  # 退出循环

    if all_videos:
        logging.info(f"  > [策略一：获取热门视频] 执行完毕。总共获取 {len(all_videos)} 个视频。")
    else:
        logging.warning("  > [策略一：获取热门视频] 执行完毕，但未能获取到任何视频。")

    return all_videos


def fetch_from_following():
    logging.info("开始执行 [策略二：监控关注的UP主]...")
    if not CONFIG['TARGET_UIDS']:
        logging.warning("  > 未配置目标UID，跳过此策略。")
        return []
    video_list = []
    url_template = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
    for uid in CONFIG['TARGET_UIDS']:
        logging.info(f"  > 正在获取UP主(UID: {uid})的最新动态...")
        params = {'host_mid': uid}
        data = send_get_request(url_template, params=params)
        if data and 'items' in data:
            found_count = 0
            for item in data['items']:
                if item.get('type') == 'DYNAMIC_TYPE_AV':
                    major = item.get('modules', {}).get('module_dynamic', {}).get('major')
                    if major and major.get('type') == 'MAJOR_TYPE_ARCHIVE':
                        video_data = major.get('archive')
                        if video_data and 'bvid' in video_data:
                            author_info = item.get('modules', {}).get('module_author', {})
                            video_data['owner'] = {
                                'mid': author_info.get('mid'),
                                'name': author_info.get('name'),
                                'face': author_info.get('face'),
                            }
                            # 补全mid字段，与搜索结果对齐
                            if 'mid' not in video_data:
                                video_data['mid'] = author_info.get('mid')
                            video_data['_source_strategy'] = 'following'
                            video_list.append(video_data)
                            found_count += 1
                            if found_count >= CONFIG['MAX_VIDEOS_PER_SOURCE']: break
            logging.info(f"    - 从UID {uid} 处获取 {found_count} 个新视频。")
    return video_list


def fetch_from_search():
    logging.info("开始执行 [策略三：关键词搜索]...")
    if not CONFIG['TARGET_KEYWORDS']:
        logging.warning("  > 未配置目标关键词，跳过此策略。")
        return []

    video_list = []
    url = "https://api.bilibili.com/x/web-interface/search/type"

    # 定义每页获取的数据量
    PAGE_SIZE = 20

    for keyword in CONFIG['TARGET_KEYWORDS']:
        logging.info(f"  > 正在搜索关键词 '{keyword}'...")

        current_page = 1
        videos_fetched_for_keyword = 0  # 记录当前关键词已获取的视频数量

        while videos_fetched_for_keyword < CONFIG['MAX_VIDEOS_PER_SOURCE']:
            params = {
                'search_type': 'video',
                'keyword': keyword,
                'order': 'pubdate',  # 按最新发布排序
                'page': current_page,
                'ps': PAGE_SIZE  # 固定每页20个
            }

            logging.info(f"    - 请求第 {current_page} 页，目标获取 {PAGE_SIZE} 个视频...")
            data = send_get_request(url, params=params)

            if not data or 'result' not in data:
                logging.warning(
                    f"      - 未能获取到关键词 '{keyword}' 第 {current_page} 页的数据，或数据格式不正确。停止此关键词的搜索。")
                break  # 无法获取数据，停止当前关键词的搜索

            search_results = data.get('result', [])
            # 兼容老版本和新版本API的返回格式
            if not isinstance(search_results, list):
                search_results = data.get('result', {}).get('video', [])

            if not search_results:
                logging.info(f"      - 关键词 '{keyword}' 第 {current_page} 页没有更多视频了。")
                break  # 当前页没有数据，说明已经到头了

            page_videos_added = 0  # 记录当前页实际添加的视频数量
            for item in search_results:
                if item.get('type') == 'video' and 'bvid' in item:
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    videos_fetched_for_keyword += 1
                    page_videos_added += 1

                    # 如果已经达到或超过了目标数量，就停止
                    if videos_fetched_for_keyword >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                        break  # 跳出 inner loop (for item in search_results)

            logging.info(
                f"      - 从关键词 '{keyword}' 第 {current_page} 页获取 {page_videos_added} 个视频，当前关键词累计 {videos_fetched_for_keyword} 个。")

            # 如果当前页获取的视频数量少于PAGE_SIZE，说明已经是最后一页了，或者没有更多符合条件的视频了
            if page_videos_added < PAGE_SIZE:
                logging.info(f"      - 关键词 '{keyword}' 已获取完所有可用视频（不足 {PAGE_SIZE} 个）。")
                break  # 跳出 outer loop (while videos_fetched_for_keyword < CONFIG.MAX_VIDEOS_PER_SOURCE)

            current_page += 1

            # 添加延迟，避免请求过快被封禁
            time.sleep(1)  # 建议延迟1秒，可根据需要调整

        logging.info(
            f"  > 关键词 '{keyword}' 搜索完成，总共获取 {videos_fetched_for_keyword} 个视频 (目标 {CONFIG['MAX_VIDEOS_PER_SOURCE']})。")
        logging.info("-" * 50)  # 分隔线
    CONFIG['MAX_VIDEOS_PER_SOURCE'] = 20  # 重置为每页20个，避免影响后续搜索，因为不会更新这么快速
    return video_list


# <<< NEW: START - 新增分区排行榜获取函数 >>>
def fetch_from_ranking():
    """
    从指定分区的排行榜获取视频。
    """
    logging.info("开始执行 [策略四：获取分区排行榜视频]...")
    if not CONFIG['RANKING_TIDS']:
        logging.warning("  > 未配置目标分区ID (RANKING_TIDS)，跳过此策略。")
        return []

    video_list = []
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"

    for tid, name in CONFIG['RANKING_TIDS'].items():
        logging.info(f"  > 正在获取分区 '{name}' (TID: {tid}) 的排行榜...")
        params = {
            'rid': tid,
            'type': 'all',  # 获取全部分类，可根据需求改为 'rookie' 或 'origin'
        }

        data = send_get_request(url, params=params)

        if data and 'list' in data and data['list']:
            # API返回最多100个视频，我们根据配置取前N个
            ranking_videos = data['list']
            for item in ranking_videos:
                if 'bvid' in item:
                    item['_source_strategy'] = 'ranking'
                    video_list.append(item)
            logging.info(f"    - 成功从分区 '{name}' 获取 {len(ranking_videos)} 个视频。")
        else:
            logging.warning(f"    - 未能从分区 '{name}' 获取到视频数据，或数据为空。")

    if video_list:
        logging.info(f"  > [策略四：获取分区排行榜视频] 执行完毕。总共获取 {len(video_list)} 个视频。")
    else:
        logging.warning("  > [策略四：获取分区排行榜视频] 执行完毕，但未能获取到任何视频。")

    return video_list


# <<< NEW: END - 新增分区排行榜获取函数 >>>


# --- 6. 已处理记录管理 (视频BVID和用户FID) ---
def load_processed_set(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()


def load_processed_dict(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_processed_set(data_set, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 将集合转换为列表以便JSON序列化
            json.dump(list(data_set), f, indent=4)
    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")


def save_processed_dict(data_dict, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 关键改动：添加 ensure_ascii=False
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"数据已成功保存到 {filepath}")
    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")


# --- 7. 视频拉取主逻辑 ---
def fetch_videos():
    logging.info("==================== 开始获取待处理视频 ====================")
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    # processed_bvideos = set()
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    all_found_videos = []
    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    if CONFIG['STRATEGIES']['following']:
        all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())
    # <<< MODIFIED: START - 集成新的获取策略 >>>
    if CONFIG['STRATEGIES']['ranking']:
        all_found_videos.extend(fetch_from_ranking())
    # <<< MODIFIED: END - 集成新的获取策略 >>>

    unique_videos_map = {video['bvid']: video for video in reversed(all_found_videos) if 'bvid' in video}
    logging.info(f"所有策略共找到 {len(all_found_videos)} 个视频，去重后剩 {len(unique_videos_map)} 个。")

    videos_to_process = [video for bvid, video in unique_videos_map.items() if bvid not in processed_bvideos]
    logging.info(f"过滤掉已处理的视频后，最终得到 {len(videos_to_process)} 个新视频待处理。")

    newly_processed_bvid_set = {video['bvid'] for video in videos_to_process}
    updated_processed_set = processed_bvideos.union(newly_processed_bvid_set)
    save_processed_set(updated_processed_set, CONFIG['PROCESSED_VIDEOS_FILE'])
    logging.info(f"已处理视频记录已更新，总数: {len(updated_processed_set)}。")

    logging.info("==================== 获取任务完成 ====================")
    return videos_to_process


# --- 8. 并发执行逻辑 ---
videos_queue = Queue()
comment_videos_queue = Queue()


def video_fetcher_worker():
    """视频拉取线程：定期拉取新视频并放入队列。"""
    while True:
        new_videos = fetch_videos()
        if new_videos:
            # 随机打乱顺序，避免行为模式过于固定
            random.shuffle(new_videos)
            for video in new_videos:
                videos_queue.put(video)
        else:
            logging.info("本次未获取到新视频。")
        logging.info(f'本次获取到 {len(new_videos)} 个新视频。队列当前长度：{videos_queue.qsize()}')
        # 每次拉取大循环，随机暂停20到30分钟
        sleep_time = random.uniform(1200, 1800)
        logging.info(f"视频拉取线程休眠 {int(sleep_time / 60)} 分钟...")
        time.sleep(sleep_time)


def get_comment_user(bvid):
    result_id_list = []
    try:
        comments = get_bilibili_comments(bvid)
        for i, reply in enumerate(comments):
            UID = reply['member']['mid']
            message = reply['content']['message']
            should_follow = any(keyword.lower() in message for keyword in CONFIG['FOLLOW_KEYWORDS'])
            if should_follow:
                result_id_list.append(UID)
    except Exception as e:
        logging.error(f"获取评论失败: {e}")
        return result_id_list
    return result_id_list


# (新功能)
def gen_comment():
    """关注线程：从队列获取视频，判断是否需要关注作者。"""
    detail_video_info_map = load_processed_dict(CONFIG['GEN_PROCESSED_VIDEOS_FILE'])
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    # 只保留processed_bvideos中gen_comment不为空的视频
    detail_video_info_map = {bvid: info for bvid, info in detail_video_info_map.items() if info.get('gen_comment')}

    for bvid in processed_bvideos:
        if bvid not in detail_video_info_map:
            temp_dict = {}
            temp_dict['bvid'] = bvid
            videos_queue.put(temp_dict)

    logging.info(f"已加载 {len(detail_video_info_map)} 个已生成的记录。")

    while True:
        try:
            video = videos_queue.get(timeout=30)  # 等待30秒，如果没有新视频则继续循环
            logging.info(f"获取到新视频 BVID: {video.get('bvid', '未知')}，开始处理...")
        except Empty:
            continue

        bvid = video.get('bvid')
        if bvid in detail_video_info_map.keys():
            logging.info(f"视频 BVID {bvid} 已经处理过，跳过。")
            continue
        else:
            video_info = gen_proper_comment(bvid)
            if video_info:
                detail_video_info_map[bvid] = video_info
                save_processed_dict(detail_video_info_map, CONFIG['GEN_PROCESSED_VIDEOS_FILE'])
                logging.info(f"视频 BVID {bvid} 处理完成，已保存生成信息。")
                comment_videos_queue.put(video_info)


def _deep_update(orig: dict, new: dict):
    """
    将 new 合并到 orig：
    - 如果某个 key 在 orig 和 new 中对应的 value 都是 dict，则递归合并；
    - 否则直接用 new[key] 覆盖 orig[key]（或新增）。
    """
    for k, v in new.items():
        if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
            _deep_update(orig[k], v)
        else:
            orig[k] = v


def save_json(path: str, data: dict):
    """
    1. 先创建目录
    2. 读已有内容（不存在或解析失败就当空 dict）
    3. 深度合并 data 到已有内容
    4. 写回文件（带缩进、美化）
    """
    # 1. 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 2. 尝试加载已有文件
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    # 3. 深度合并
    _deep_update(existing, data)

    # 4. 写回
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def find_video_by_bvid(bvid_to_find: str, data_dict: dict):
    """
    在给定的字典中根据 bvid 查找对应的视频信息 value。

    这个函数会遍历字典中的每一个 value，并安全地检查其内部是否包含
    'upload_info' -> 'upload_result' -> 'bvid' 这个路径，
    且其值与要查找的 bvid 相匹配。

    Args:
        bvid_to_find (str): 需要查找的 Bilibili 视频 ID (bvid)。
        data_dict (dict): 包含多个视频信息的数据字典。

    Returns:
        dict or None: 如果找到，返回包含该 bvid 的整个 value（即视频信息字典）；
                      如果遍历完整个字典都找不到，则返回 None。
    """
    # 遍历字典中的每一个键值对
    for key, video_info in data_dict.items():
        upload_info = video_info.get('upload_info', {})
        upload_result = upload_info.get('upload_result', {})
        found_bvid = upload_result.get('bvid')
        if found_bvid and found_bvid == bvid_to_find:
            return video_info
    return None


def find_video_by_title(title_to_find: str, data_dict: dict):
    for key, video_info in data_dict.items():
        upload_info = video_info.get('upload_info', {})
        upload_result = upload_info.get('upload_params', {})
        found_bvid = upload_result.get('title')
        if found_bvid and found_bvid == title_to_find:
            return video_info
    return None


def parse_and_group_danmaku(data: dict) -> list:
    """
    解析输入的字典，将弹幕按时间戳进行分组。

    Args:
        data: 包含弹幕信息的源字典。

    Returns:
        一个按时间戳排序的列表。每个元素是一个字典，
        包含 "建议时间戳" 和一个该时间戳下所有 "推荐弹幕内容" 的列表。
    """
    # 1. 使用 defaultdict(list) 来自动处理分组
    grouped_danmaku = defaultdict(list)

    # 2. 遍历 "开场弹幕" 并添加到分组字典中
    opening_danmaku = data.get("开场弹幕")
    if opening_danmaku:
        timestamp = opening_danmaku.get("建议时间戳")
        contents = opening_danmaku.get("推荐弹幕内容", [])
        if timestamp and contents:
            # 使用 extend 将列表中的所有元素都添加进去
            grouped_danmaku[timestamp].extend(contents)

    # 3. 遍历 "推荐弹幕列表" 并添加到分组字典中
    recommendation_list = data.get("推荐弹幕列表", [])
    recommendation_list_back = data.get("精选弹幕再创作列表", [])
    recommendation_list.extend(recommendation_list_back)
    for item in recommendation_list:
        timestamp = item.get("建议时间戳")
        contents = item.get("推荐弹幕内容", [])
        if timestamp and contents:
            grouped_danmaku[timestamp].extend(contents)

    # 4. 将分组后的字典转换为目标格式的列表
    final_list = []
    for timestamp, contents in grouped_danmaku.items():
        final_list.append({
            "建议时间戳": timestamp,
            "推荐弹幕内容": contents
        })

    # 5. 按时间戳对最终列表进行排序
    final_list.sort(key=lambda x: x["建议时间戳"])

    return final_list


danmu_praises_general_quality = [
    # --- 1. 极度通用型 (几乎适用于所有非劣质视频) ---
    "UP主用心了",
    "这个视频做得真好",
    "质量不错，支持一下",
    "观感很舒服",
    "好评！",
    "制作不易，给你点赞了",
    "感觉很流畅",
    "看得出来是认真做的",
    "这个质量可以的",
    "不错不错",

    # --- 2. 夸赞剪辑与节奏 ---
    "这剪辑，有点东西",
    "节奏很棒，不知不觉就看完了",
    "转场好自然",
    "BGM和画面配合得真好",
    "这个剪辑节奏爱了",
    "信息密度刚刚好，不拖沓",
    "神仙剪辑！",

    # --- 3. 夸赞画面与视听体验 (非特指高清) ---
    "画面很干净",
    "看着很清爽",
    "镜头很稳，好评",
    "这个构图学到了",
    "字幕好评，看得舒服多了",
    "收音很清晰，没有杂音",
    "字体和排版好评",
    "bgm好听，求bgm！",  # 侧面夸赞品味

    # --- 4. 夸赞整体质感与氛围 ---
    "质感拉满了",
    "有电影感了",  # 泛指，不一定是真的电影机
    "这视频有种高级感",
    "赏心悦目",
    "完成度好高啊",
    "是个宝藏UP主",

    # --- 5. 互动与鼓励型 ---
    "这质量，值得一个三连！",
    "果断三连了",
    "已关注，期待更多好作品",
    "好活，当赏！",  # 偏二次元/B站风格
    "你更新，我三连，就这么定了",
    "这不得狠狠点个赞",
    "码住，回头再看一遍",  # 表达对视频质量的认可
]


def filter_danmu(danmu_list, duration):
    """
    过滤和调整弹幕列表。
    1. 确保所有弹幕的时间戳在视频时长范围内，无效时间戳会随机分配。
    2. 如果最终弹幕数量不足25条，则从通用弹幕池中随机抽取补足。

    Args:
        danmu_list: 弹幕列表，每个元素是包含 '建议时间戳' 和 '推荐弹幕内容' 的字典。
        duration: 视频总时长，格式为 "HH:MM:SS" 或 "MM:SS" 或秒数。

    Returns:
        调整后的弹幕列表，至少有25条弹幕（除非视频时长无效）。
    """
    common_danmu_list = [
        "屏幕那头的陌生人，不管你在哪里，祝你天天开心。",
        "祝刷到这条视频的你，烦恼全消，未来可期。",
        "愿刷到这里的你，凛冬散尽，星河长明。",
        "希望这条弹幕能吸收你今天所有的不开心。",
        "这条弹幕不为什么，就是想祝你万事胜意。",

        "外面在下雨，屋里看视频，感觉很安心。",
        "这里是弹幕许愿池，许个愿吧，万一实现了呢？",
        "感觉累了，大家能在这里留下一句加油吗？给我也给你自己。",
        "我的电量比进度条还多，优势在我！",
        "前方高能！",
        "白嫖失败，投币了投币了",
        "给屏幕对面那个或许有些疲惫的你，一个看不见的拥抱。",
        "今天也要好好吃饭，好好生活呀！",
        "很高兴在此刻，与屏幕前的各位“网友”共度这一分一秒。",
        "把不开心的事，都留在当下吧！",
        "让这条弹幕带走你今天的疲惫。",
    ]

    danmaku_zouxin_sanlian_gongmian = [
        "就冲结尾这句，放心把三连交了",
        "三连送上，这结尾太值得",
        "最后一段值得三连收藏",
        "这句祝福让我毫不犹豫三连",
        "把这段当成今日小确幸，三连已交付",
        "这结尾值得多按几下（已按）",
        "已三连，愿这份祝福常在",
        "悄悄三连，最后一句反复回放中",
        "被最后这句治愈了，三连必须的",
        "最后这句值得三连也值得收藏",
        "三连已给，感恩这份温柔",
        "手滑三连了（是真的走心）",
        "这祝福像暖阳，照进烦心处",
        "一句走心话，整天都舒服了"
    ]
    try:
        total_seconds = time_to_ms(duration) / 1000
        total_seconds = int(total_seconds)
    except Exception as e:
        total_seconds = None
    if total_seconds is None or total_seconds <= 0:
        return danmu_list

    # === 第一步：处理并规范化传入的弹幕列表 ===
    processed_danmu = []
    for item in danmu_list:
        try:
            # 为了不修改原始列表，创建一个副本进行操作
            new_item = item.copy()
            ts = new_item.get('建议时间戳')
            seconds = time_to_ms(ts) / 1000
            seconds = int(seconds) if seconds is not None else None

            # 如果时间戳无法解析或超出范围，则随机分配
            if seconds is None or seconds < 0 or seconds > total_seconds:
                seconds = random.randint(2, total_seconds - 10)

            new_item['建议时间戳'] = seconds
            processed_danmu.append(new_item)
        except Exception as e:
            logging.error(f"处理弹幕时出错: {e}")
            continue

    processed_danmu_count = 0
    for item in processed_danmu:
        if isinstance(item.get('推荐弹幕内容'), list):
            processed_danmu_count += len(item['推荐弹幕内容'])

    target_num = 25
    # === 第二步（新增逻辑）：检查弹幕数量并补足到25条 ===
    num_to_add = target_num - processed_danmu_count
    num_to_add = min(num_to_add, len(common_danmu_list))  # 避免超出通用池范围
    if num_to_add > 0:
        print(f"弹幕数量为 {processed_danmu_count}，不足{target_num}条，需要补充 {num_to_add} 条。")
        # 从通用弹幕池中随机选择 num_to_add 条
        random_choices = random.sample(common_danmu_list, k=num_to_add)

        for content in random_choices:
            # 2. 在视频时长范围内随机分配一个时间戳（秒）
            timestamp = random.randint(2, total_seconds - 10)

            # 3. 创建新的弹幕字典并添加到列表中
            new_danmu = {
                '建议时间戳': timestamp,
                '推荐弹幕内容': [content]
            }
            processed_danmu.append(new_danmu)

    # 增加固定的三连弹幕
    random_choices = random.sample(danmaku_zouxin_sanlian_gongmian, k=2)
    time_diff = 6
    for content in random_choices:
        new_danmu = {
            '建议时间戳': total_seconds - time_diff,
            '推荐弹幕内容': [content]
        }
        time_diff += 4
        processed_danmu.append(new_danmu)
    return processed_danmu


def extract_guides(data):
    """
    从给定的数据字典中提取“互动引导”和“补充信息”列表。

    参数:
        data: dict
            数据结构中每个顶层 key 对应一个方案，方案内可能包含“简介”字典，
            其下包含“互动引导”和“补充信息”字段。

    返回:
        Tuple[List[str], List[str]]
            第一个元素是所有方案的“互动引导”列表，第二个元素是所有方案的“补充信息”列表。
            如果没有对应字段，则返回空列表。
    """
    interaction_prompts = []

    supplementary_notes = []

    for scheme_name, scheme_content in data.items():
        # 获取“简介”部分
        brief = scheme_content.get("简介", {})
        # 提取互动引导
        prompt = brief.get("互动引导")
        if isinstance(prompt, str) and prompt.strip():
            interaction_prompts.append(prompt.strip())
        # 提取补充信息
        note = brief.get("补充信息")
        if isinstance(note, str) and note.strip():
            supplementary_notes.append(note.strip())

    return interaction_prompts, supplementary_notes


def format_bilibili_emote(comment_list, all_emote_list):
    """
    进行b站的emote转换，避免没有正常输出表情
    """
    for comment in comment_list:
        # 将第一个元素调用 replace_bracketed
        comment[0] = replace_bracketed(comment[0], all_emote_list)


def generate_danmaku_plan(total_duration: int, text_list: list, target_num: int = 4) -> list:
    """
    在 total_duration 范围内随机生成 target_num 个弹幕计划（时间戳为整数秒）

    参数:
        total_duration (int): 视频总时长（秒）
        text_list (list[str]): 可供选择的弹幕内容
        target_num (int): 需要生成的弹幕数量，默认为4

    返回:
        list[dict]: 每个元素包含 '建议时间戳' 和 '推荐弹幕内容'
    """
    if not text_list:
        raise ValueError("text_list不能为空")
    if target_num > len(text_list):
        target_num = len(text_list)  # 避免超出可选范围

    # 随机选取 target_num 个弹幕内容
    chosen_texts = random.sample(text_list, target_num)

    # 随机生成不重复的时间戳（整数秒）
    chosen_timestamps = sorted(random.sample(range(total_duration + 1), target_num))

    # 拼接结果
    result = []
    for ts, text in zip(chosen_timestamps, chosen_texts):
        result.append({
            "建议时间戳": ts,
            "推荐弹幕内容": [text]
        })

    return result


def gen_hudong_info(bvid, interaction_data, metadata_cache_with_uploads, all_emote_list):
    """
    为 bvid 生成相应的推荐弹幕与评论，增强了对 None 和缺失字段的容错能力。
    """
    try:
        target_value = find_video_by_bvid(bvid, metadata_cache_with_uploads) or {}
    except Exception as e:
        # 发生异常时记录或打印 e（可选），并使用空 dict 继续
        # logger.warning(f"find_video_by_bvid error for {bvid}: {e}")
        target_value = {}
    if target_value.get('hudong', {}) == {}:
        return {}
    hudong_info = {}
    # 1. 如果已有缓存，直接返回
    existing = interaction_data.get(bvid, {})
    if existing and 'hudong' in existing:
        hud = existing['hudong']
        hudong_info = existing['hudong']
        if target_value.get('hudong', {}).get('comment_list', []) == []:
            if hud:
                return hud

    # 2. 安全调用 find_video_by_bvid

    duration = target_value.get('metadata', [{}])[0].get('duration', '00:02')
    try:
        total_seconds = time_to_ms(duration) / 1000
        comment_list = target_value.get('hudong', {}).get('comment_list', []) or []
        # 如果评论列表为空，使用 gen_proper_comment 生成
        if not comment_list:
            gen_info = gen_proper_comment(bvid) or {}
            if duration == '00:02':
                duration = gen_info.get('总时长', '00:02')

            raw_comments = gen_info.get('gen_comment', [])
            # 将字符串列表转为 (comment, weight, extra) 结构
            comment_list = [[c, 1, "None"] for c in raw_comments]
    except Exception as e:
        comment_list = []
        print(f"处理评论列表时出错: {e}")

    try:
        danmu_info = target_value.get('hudong', {}).get('danmu_info', {})
        if danmu_info:
            danmu_list = parse_and_group_danmaku(danmu_info)
        else:
            # fallback: 一条通用弹幕
            danmu_list = [{'建议时间戳': '00:01', '推荐弹幕内容': danmu_praises_general_quality}]
    except Exception as e:
        # logger.error(f"处理弹幕列表时出错: {e}")
        danmu_list = [{'建议时间戳': '00:01', '推荐弹幕内容': danmu_praises_general_quality}]
    danmu_list = filter_danmu(danmu_list, duration)
    total_seconds = int(total_seconds)

    title_schemes = target_value.get('title_schemes', {})
    interaction_prompts, supplementary_notes = extract_guides(title_schemes)  # 提取互动引导和补充信息（如果有）
    if len(interaction_prompts) == 0:
        interaction_prompts = ["刷到这个视频的你，希望今天能有个好心情呀~",
                               "叮！你收到一份来自UP主的好运，请注意查收哦！",
                               "不管此刻你在做什么，都要记得好好照顾自己。",
                               "嘿，朋友，为你正在付出的一切点赞，你超棒的！",
                               "很高兴遇见你，愿所有美好都向你奔赴而来。"]
    if len(supplementary_notes) == 0:
        supplementary_notes = ["感谢你愿意花时间看到最后，愿这份好运能一直陪着你。",
                               "如果觉得视频还不错，不妨点个赞，把这份快乐和祝福一起带走吧！",
                               "视频虽已结束，但我的祝福不会。祝你，不止今天，天天开心！",
                               "感谢我们的这次相遇，我们下期再见，在那之前，要一切顺利哦！",
                               "那么，就到这里啦。晚安，祝你好梦，忘掉所有烦恼。"]
    interaction_danmu_list = [{'建议时间戳': 1, '推荐弹幕内容': interaction_prompts}]
    supplementary_notes_list = [{'建议时间戳': total_seconds - 8, '推荐弹幕内容': supplementary_notes}]
    owner_danmu_list = []  # 用于存储UP主的弹幕
    owner_danmu_list.extend(interaction_danmu_list)  # 将互动引导弹幕添加到UP主弹幕列表中
    owner_danmu_list.extend(supplementary_notes_list)  # 将补充信息弹幕添加到UP主弹幕列表中
    format_bilibili_emote(comment_list, all_emote_list)
    # 5. 组装结果，写回缓存，并返回
    hudong_info["duration"] = total_seconds
    hudong_info['comment_list'] = comment_list
    hudong_info['danmu_list'] = danmu_list
    hudong_info['owner_danmu'] = owner_danmu_list
    # 写回 interaction_data 时包裹在 'hudong' 字段里，以保持与入口逻辑一致
    interaction_data[bvid] = {'hudong': hudong_info}

    return hudong_info


def path_exists(path) -> bool:
    """
    判断输入的路径字符串是否存在。

    参数:
        path: 路径字符串或 None。

    返回:
        如果 path 是非空字符串且对应路径存在，则返回 True；
        否则返回 False。
    """
    # 排除 None 和非字符串
    if not isinstance(path, str):
        return False

    # 去除首尾空白后为空则认为不存在
    stripped = path.strip()
    if not stripped:
        return False

    # 最终判断文件或目录是否存在
    return os.path.exists(stripped)


def post_comments_once(commenter_list,
                       comment_list,
                       bvid,
                       max_success_comment_count,
                       comment_used_list,
                       path_exists,
                       max_workers=5,
                       jitter=(0.4, 1.0)):
    """
    最终修订版V3：
    1. 使用 futures.wait() 实现可靠的全局超时。
    2. 修复了 comment_used_list 的同步BUG，只记录真正成功的评论。
    3. 将 jitter 延迟放回 worker 线程以实现并发延迟。
    4. 确保返回的 success_count 是在超时前确定的值。
    """
    # --- 1. 准备工作：分配评论任务 (逻辑保持不变) ---
    random.shuffle(commenter_list)
    selected = commenter_list[:max_success_comment_count]

    # 锁和共享状态
    used_lock = threading.Lock()
    successful_texts_lock = threading.Lock()
    used_texts = set(comment_used_list)
    successful_texts = [] # 只存储本次调用中成功发送的评论文本

    assignments = []
    for c in selected:
        assigned = None
        # 从 comment_list 中找到一条未被使用的评论
        for detail in comment_list:
            text = detail[0] if detail and len(detail) > 0 else None
            if not text or len(text) <= 1:
                continue

            with used_lock:
                if text in used_texts:
                    continue
                # 预先锁定，防止被其他任务分配
                used_texts.add(text)
                assigned = detail
                break # 找到一条就跳出内层循环

        if assigned:
            assignments.append((c, assigned))
        else:
            break # 如果找不到可用的评论了，就停止分配

    if not assignments:
        print("没有可分配的评论或 commenter，退出。")
        return 0

    # --- 2. Worker 函数定义 (修订版) ---
    def worker(pair):
        # 4. jitter放回worker，实现并发延迟
        time.sleep(random.uniform(*jitter))

        commenter, detail = pair
        text = detail[0]
        image_path = detail[2] if len(detail) > 2 else None

        try:
            # 执行评论操作
            if image_path and path_exists(image_path):
                rpid = commenter.post_comment(bvid, text, 1, like_video=True, image_path=image_path, forward_to_dynamic=False)
            else:
                rpid = commenter.post_comment(bvid, text, 1, like_video=True, forward_to_dynamic=False)

            if rpid:
                # 3. 只有成功时，才将文本记录到 successful_texts
                with successful_texts_lock:
                    successful_texts.append(text)
                name = commenter.all_params.get('name', 'unknown')
                print(f"[评论成功] by {name} rpid:{rpid}: {text}")
                return True # 返回成功状态
            else:
                print(f"[评论失败] by {getattr(commenter, 'name', 'unknown')} (接口返回): {text}")
                return False # 返回失败状态

        except Exception as e:
            print(f"[评论异常] by {getattr(commenter, 'name', 'unknown')}: {text} -> {e}")
            return False # 异常也视为失败
        finally:
            # 无论成功、失败还是异常，都要从“预锁定”集合中释放
            # 因为只有 successful_texts 里的才算真正“已使用”
            with used_lock:
                if text in used_texts:
                    used_texts.remove(text)

    # ==========================================================
    # ==================== 核心执行区域 =======================
    # ==========================================================

    TOTAL_TIMEOUT = 300  # 整个评论环节最多执行5分钟
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(assignments)))

    # 将任务和原始信息关联起来
    future_to_info = {executor.submit(worker, a): a[1][0] for a in assignments}

    # 局部变量，用于统计在超时前确认的成功数
    confirmed_success_count = 0

    try:
        done, not_done = wait(future_to_info.keys(), timeout=TOTAL_TIMEOUT, return_when=ALL_COMPLETED)

        # 处理已完成的任务
        for future in done:
            try:
                # 获取worker的返回结果 (True/False)
                if future.result():
                    confirmed_success_count += 1
            except Exception:
                # worker内部的异常已经被捕获并返回False，这里只是为了代码健壮性
                pass

        # 处理未完成/超时的任务
        if not_done:
            print(f"[评论总超时] {len(not_done)} 个任务在 {TOTAL_TIMEOUT} 秒后仍未完成，将被放弃。")
            for future in not_done:
                text = future_to_info[future]
                print(f"  - 超时任务的评论: '{text[:30]}...'")
                # 【重要】超时任务也需要从“预锁定”集合中释放，worker 的 finally 无法执行
                with used_lock:
                    if text in used_texts:
                        used_texts.remove(text)

    finally:
        print(f"在超时前确认成功的评论数: {confirmed_success_count}")
        # 将本次调用中所有确认成功的评论文本，同步回原始的list中
        # 过滤掉可能重复的项
        new_successes = [text for text in successful_texts if text not in comment_used_list]
        comment_used_list.extend(new_successes)

        # 立即关闭线程池，不等待僵尸线程
        executor.shutdown(wait=False)
        print("线程池已发出关闭信号，主流程继续。")

    return confirmed_success_count

def send_danmaku_thread_function(owner_commenter, owner_danmu_list, max_success_owner_danmu_count, bvid,
                                 owner_danmu_used_list):
    """
    这个函数包含了发送弹幕的完整逻辑，它将在一个独立的线程中被执行。
    """
    success_owner_danmu_count = 0  # 计数器在线程内部初始化和使用
    if owner_commenter:
        for detail_owner_danmu in owner_danmu_list:
            if success_owner_danmu_count >= max_success_owner_danmu_count:
                print(f"线程 {threading.current_thread().name}: 已达到最大成功UP主弹幕数，停止处理。")
                break

            danmaku_time_ms = detail_owner_danmu['建议时间戳'] * 1000  # 转换为毫秒
            danmu_text_list = detail_owner_danmu['推荐弹幕内容']

            for danmu_text in danmu_text_list:
                if danmu_text in owner_danmu_used_list or len(danmu_text) == 0:
                    continue

                # 再次检查是否达到最大数量，避免在内层循环中超出
                if success_owner_danmu_count >= max_success_owner_danmu_count:
                    break

                danmaku_sent = owner_commenter.send_danmaku(
                    bvid=bvid,
                    msg=danmu_text,
                    progress=danmaku_time_ms,
                    is_up=True
                )

                if danmaku_sent:
                    owner_danmu_used_list.append(danmu_text)
                    success_owner_danmu_count += 1
                    print(
                        f" [主人弹幕发送流程成功个数 {success_owner_danmu_count}] {danmu_text} BVID: {bvid} name {owner_commenter.all_params['name']}")
                    time.sleep(random.uniform(5, 10))
                else:
                    print(
                        f"{success_owner_danmu_count} 主人弹幕发送流程失败！{danmu_text} BVID: {bvid} name {owner_commenter.all_params['name']} danmaku_time_ms: {danmaku_time_ms}")
                    time.sleep(random.uniform(10, 15))

            # 在处理完一个弹幕包后稍作等待
            time.sleep(random.uniform(10, 15))
    print(f"线程 {threading.current_thread().name} 完成。成功发送 UP 主弹幕数: {success_owner_danmu_count}")


def _send_danmu_worker(danmu_list, other_commenters, bvid, max_success_other_danmu_count, stop_event, result):
    try:
        random.shuffle(other_commenters)
        senders = deque(other_commenters)
        success_count = 0
        sent_texts = []

        for detail in danmu_list:
            if stop_event.is_set():
                print("send worker: 收到停止信号，退出。")
                break

            if success_count >= max_success_other_danmu_count:
                break

            danmaku_time_ms = int(detail.get('建议时间戳', 0) * 1000)
            danmu_text_list = detail.get('推荐弹幕内容', []) or []

            for text in danmu_text_list:
                if stop_event.is_set() or success_count >= max_success_other_danmu_count:
                    break
                if not text:
                    continue

                sender = senders.popleft()
                try:
                    danmaku_sent = sender.send_danmaku(
                        bvid=bvid,
                        msg=text,
                        progress=danmaku_time_ms,
                        is_up=False
                    )
                except Exception as e:
                    print("发送异常:", e)
                    danmaku_sent = False

                # 轮转发送者
                senders.append(sender)

                if danmaku_sent:
                    success_count += 1
                    sent_texts.append(text)
                    # 仅打印，不修改外部列表
                    print(f"[成功弹幕个数 {success_count}] {text} 发送者: {sender.all_params.get('name')}")
                else:
                    print(f"[失败] {text} 发送者: {sender.all_params.get('name')}，稍后继续或跳过。")
                    time.sleep(random.uniform(5, 10))

                # 控制速率
                time.sleep(random.uniform(1, 2))

        result.success_count = success_count
        result.sent_texts = sent_texts
        print("send worker 完成。成功发送:", success_count)
    except Exception as e:
        print("worker 未捕获异常:", e)
        result.exception = e

def start_send_danmu_background(danmu_list, other_commenters, bvid, max_success_other_danmu_count, daemon=True):
    """
    启动后台线程发送弹幕（极简版）。
    返回 (thread, stop_event, result)：
      - thread: threading.Thread 对象
      - stop_event: threading.Event，可以通过 stop_event.set() 停止线程
      - result: SimpleNamespace，线程结束后包含 .success_count, .sent_texts, 以及可选的 .exception
    说明：该实现不会修改外部的 danmu_used_list 或 hudong_info，需你在主线程中自行处理。
    """
    stop_event = threading.Event()
    result = SimpleNamespace(success_count=0, sent_texts=[])
    t = threading.Thread(
        target=_send_danmu_worker,
        args=(danmu_list, other_commenters, bvid, max_success_other_danmu_count, stop_event, result),
        daemon=daemon
    )
    t.start()
    return t, stop_event, result


def pick_commenters(commenter_map, usage_path, n=3):
    """
    从 commenter_map 中尽量均匀选 n 个账号，读取/更新 usage_path。
    特殊 uid 使用一次记 2 次，其他记 1 次。
    返回选中的 commenter 对象列表。
    """
    usage_map = {'196823511':6,'3546972143225467':4,'3546717871934392':5,'3632304865937878':2, '3546970887031023':3, '3546979686681114':3, '3546970725550911':3, '3632307990694238':3}

    usage = read_json(usage_path) or {}
    # ensure keys are strings
    usage = {str(k): int(v) for k,v in usage.items()}
    for uid in list(commenter_map.keys()):
        usage.setdefault(str(uid), 0)

    # 随机打乱后按使用次数升序选择，打破并列的确定性
    uids = list(map(str, commenter_map.keys()))
    random.shuffle(uids)
    uids.sort(key=lambda x: usage.get(x, 0))

    selected = uids[:min(n, len(uids))]
    for uid in selected:
        usage[uid] = usage.get(uid, 0) + 8 - usage_map.get(uid, 2)

    save_json(usage_path, usage)
    selected_commenter = [commenter_map[uid] for uid in selected if uid in commenter_map]
    return selected_commenter


def process_single_video(bvid, hudong_info, uid, commenter_map, today=None):
    # --- 新增：为线程等待定义统一的超时时间 (单位：秒) ---
    THREAD_JOIN_TIMEOUT = 900  # 15分钟

    print(f"[{bvid}] --- process_single_video 开始 ---")

    if not today:
        today = datetime.date.today().isoformat()
    if hudong_info.get('last_processed_date'):
        print(f"[{bvid}] 跳过：该视频已有处理日期。")
        return hudong_info, True
    if hudong_info.get('last_processed_date') == today:
        hudong_info['last_processed_date_count'] = hudong_info.get('last_processed_date_count', 0)
        if hudong_info['last_processed_date_count'] >= 1:
            print(f"[{bvid}] 跳过：今天已处理过 {hudong_info['last_processed_date_count']} 次。")
            return hudong_info, True

    print(f"[{bvid}] [步骤 1/8] 调用 gen_proper_comment 获取已有互动信息...")
    result = gen_proper_comment(bvid, dont_need_comment=True)
    print(f"[{bvid}] [步骤 1/8] gen_proper_comment 调用完成。")

    exist_comment = result.get('已有评论', [])
    exist_comment_text = [comment[0] for comment in exist_comment]
    exist_danmu = result.get('已有弹幕', [])
    exist_danmu_text = [danmu[0] for danmu in exist_danmu]
    max_success_comment_count = 2
    max_success_owner_danmu_count = 5
    max_success_other_danmu_count = 5

    print(f"获得到已有评论：{len(exist_comment_text)} 条，已有弹幕：{len(exist_danmu_text)} 条。| BVID: {bvid}")
    owner_commenter = commenter_map.get(uid, None)
    other_commenters = [c for k, c in commenter_map.items() if k != uid]
    share_video = hudong_info.get("share_video", False)
    triple_like_video = hudong_info.get("triple_like_video", False)

    # 初始化 watch_thread 变量，防止后续引用报错
    watch_thread = None

    print(f"[{bvid}] [步骤 2/8] 检查是否需要分享和三连...")
    if not share_video or not triple_like_video:
        print(f"[{bvid}] [步骤 2a/8] 需要执行分享/三连。正在启动 watch_video 后台线程...")

        # --- 修改点：将 watch_video 放入后台线程启动 ---
        try:
            watch_thread = threading.Thread(
                target=watch_video,
                args=([bvid],)
            )
            watch_thread.start()
            print(f"[{bvid}] [步骤 2a/8] watch_video 后台线程已启动，主程序继续执行分享操作。")
        except Exception as e:
            print(f"[{bvid}] 启动 watch_video 线程失败: {e}")
        # -------------------------------------------

        for commenter in commenter_map.values():
            name = commenter.all_params.get('name', 'unknown')
            print(f"[{bvid}] [步骤 2b/8] 用户 '{name}' 正在执行 share_video...")
            share_success = commenter.share_video(bvid=bvid)
            if share_success:
                share_video = True
            else:
                print(f"[{bvid}] 用户 '{name}' 分享操作流程失败。")
            print(f"[{bvid}] [步骤 2b/8] 用户 '{name}' share_video 调用完成。")

            print(f"[{bvid}] [步骤 2c/8] 用户 '{name}' 正在执行 triple_like_video...")
            triple_like_success = commenter.triple_like_video(bvid=bvid)
            if triple_like_success:
                triple_like_video = True
            else:
                print(f"[{bvid}] 用户 '{name}' 一键三连操作流程失败。")
            print(f"[{bvid}] [步骤 2c/8] 用户 '{name}' triple_like_video 调用完成。")

        max_success_comment_count = 20
        max_success_owner_danmu_count = 20
        max_success_other_danmu_count = 30
    print(f"[{bvid}] [步骤 2/8] 分享和三连操作检查完成（观看任务可能仍在后台进行）。")

    hudong_info['share_video'] = share_video
    hudong_info['triple_like_video'] = triple_like_video
    owner_danmu_list = hudong_info.get('owner_danmu', [])
    owner_danmu_used_list = hudong_info.get('owner_danmu_used', [])
    owner_danmu_used_list.extend(exist_danmu_text)
    danmaku_thread = None

    print(f"[{bvid}] [步骤 3/8] 准备启动主人弹幕线程...")
    if owner_commenter:
        danmaku_thread = threading.Thread(
            target=send_danmaku_thread_function,
            args=(
                owner_commenter,
                owner_danmu_list,
                max_success_owner_danmu_count,
                bvid,
                owner_danmu_used_list
            )
        )
        danmaku_thread.start()
        print(f"[{bvid}] [步骤 3/8] 主人弹幕线程已启动。")
    else:
        print(f"[{bvid}] [步骤 3/8] 无主人评论者，跳过启动主人弹幕线程。")

    danmu_list = hudong_info.get('danmu_list', [])
    danmu_used_list = hudong_info.get('danmu_used', [])
    danmu_used_list.extend(exist_danmu_text)

    print(f"[{bvid}] [步骤 4/8] 准备启动其他用户弹幕线程...")
    t, stop_event, result = start_send_danmu_background(danmu_list, other_commenters, bvid,
                                                        max_success_other_danmu_count)
    print(f"[{bvid}] [步骤 4/8] 其他用户弹幕线程已启动。")

    max_success_comment_count = 5
    if uid in ['3632307990694238', '3632313749473288', '3632309148322699']:
        max_success_comment_count = 10
    comment_list = hudong_info.get('comment_list', [])
    comment_used_list = hudong_info.get('comment_used', [])
    comment_used_list.extend(exist_comment_text)

    print(f"[{bvid}] [步骤 5/8] 调用 pick_commenters 选择评论者...")
    comment_commenters = pick_commenters(commenter_map, '../../LLM/TikTokDownloader/back_up/commenter_usage.json',
                                         n=max_success_comment_count)
    print(f"[{bvid}] [步骤 5/8] pick_commenters 调用完成，选择了 {len(comment_commenters)} 个评论者。")

    print(f"[{bvid}] [步骤 6/8] 准备调用 post_comments_once 发送评论...")
    post_comments_once(
        commenter_list=comment_commenters,
        comment_list=comment_list,
        bvid=bvid,
        max_success_comment_count=max_success_comment_count,
        comment_used_list=comment_used_list,
        path_exists=path_exists,
        max_workers=5,
        jitter=(0.4, 1.0)
    )
    print(f"[{bvid}] [步骤 6/8] post_comments_once 调用完成。")

    hudong_info['comment_used'] = comment_used_list
    if hudong_info.get('last_processed_date') == today:
        last_count = int(hudong_info.get('last_processed_date_count', 0) or 0)
        hudong_info['last_processed_date_count'] = last_count + 1
    else:
        hudong_info['last_processed_date_count'] = 1
    hudong_info['last_processed_date'] = today

    print(f"[{bvid}] [步骤 7/8] 准备等待主人弹幕线程...")
    if danmaku_thread and danmaku_thread.is_alive():
        danmaku_thread.join(timeout=THREAD_JOIN_TIMEOUT)
        if danmaku_thread.is_alive():
            print(f"[{bvid}] 警告：主人弹幕线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
        else:
            print(f"[{bvid}] 主人弹幕线程已成功执行完毕。")
    else:
        print(f"[{bvid}] 主人弹幕任务未启动或已执行完毕。")
    print(f"[{bvid}] [步骤 7/8] 主人弹幕线程等待完成。")

    hudong_info['owner_danmu_used'] = owner_danmu_used_list

    print(f"[{bvid}] [步骤 8/8] 准备等待其他用户弹幕线程...")
    t.join(timeout=THREAD_JOIN_TIMEOUT)
    if t.is_alive():
        print(f"[{bvid}] 警告：其他用户弹幕线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
        stop_event.set()
    else:
        print(f"[{bvid}] 其他用户弹幕线程已成功执行完毕。")
    print(f"[{bvid}] [步骤 8/8] 其他用户弹幕线程等待完成。")

    hudong_info['danmu_used'] = result.sent_texts

    # --- 新增：最后等待 watch_video 线程结束 ---
    if watch_thread:
        print(f"[{bvid}] 准备等待 watch_video 后台线程...")
        if watch_thread.is_alive():
            watch_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if watch_thread.is_alive():
                print(f"[{bvid}] 警告：watch_video 线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
            else:
                print(f"[{bvid}] watch_video 线程已成功执行完毕。")
        else:
            print(f"[{bvid}] watch_video 线程此前已自动完成。")
    # ---------------------------------------

    print(f"[{bvid}] --- process_single_video 结束 ---")
    return hudong_info, False


def fix_metadata_cache_with_uploads(all_found_videos, metadata_cache_with_uploads):
    for video in all_found_videos:
        title = video.get('title', '')

        bvid = video.get('bvid', '')
        video_info = find_video_by_title(title, metadata_cache_with_uploads)
        if video_info:
            if video_info['upload_info']['upload_result']['bvid'] != bvid:
                print(f"修正视频标题 {title} 的 BVID: {video_info['upload_info']['upload_result']['bvid']} -> {bvid}")
                video_info['upload_info']['upload_result']['bvid'] = bvid
                save_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json', metadata_cache_with_uploads)


stop_event = threading.Event()

NEED_UPDATE_SIGN = True
signatures = [
    "谢谢你这么好看还来看看我，愿你每天都被温柔对待。",
    "能遇见你真好，祝你笑口常开。",
    "你看我一眼，我就把好运给你留着。",
    "看到你真暖，愿你的每一天都晴朗。",
    "谢谢你停留，愿快乐找上门。",
    "因为有你，我的世界更亮。",
    "你这么棒，别忘了对自己好一点。",
    "感谢你的关注，愿你心想事成。",
    "谢谢你来看我，愿你夜夜好梦。",
    "你好可爱，谢谢你来，愿你事事顺心。",
    "你的出现，让我的心情变好了。",
    "你来过，我就足够幸福了。",
    "看见你就想笑，愿你永远被喜欢。",
    "谢谢你温柔以待，愿你被生活温柔相待。",
    "有你在，平凡也变有趣。",
    "遇见你是最好的巧合，祝你安好。",
    "你的微笑很暖，谢谢你停留。",
    "谢谢你把时间借给我，愿你被世界温柔以待。",
    "你在的地方就有光，愿你前路无忧。",
    "感谢今天的相遇，愿你一直好运连连。",
    "谢谢你来看看，愿所有小确幸都向你靠近。",
    "谢谢你为我点亮一眼，愿你每天被幸运宠爱。",
    "你的好看值得被世界赞美，祝你被爱包围。",
    "因为你，平淡也变成仪式感。",
    "你的出现，让我相信美好还在。",
    "谢谢你这么温柔地看我，愿你永远被温柔相待。",
    "你把好心情带来，我把祝福送给你。",
    "有你点赞真开心，愿你此刻快乐。",
    "谢谢你路过我的世界，愿你永远心平气和。"
]


def filter_recent_data(data_dict, days=10):
    """
    根据 last_processed_date 保留最近 N 天的数据。
    兼容 import datetime 的导入方式。
    """
    # 1. 计算时间阈值
    now = datetime.datetime.now()
    cutoff_date = now - datetime.timedelta(days=days)

    # 2. 遍历并过滤
    filtered_data = {}
    for key, val in data_dict.items():
        # 尝试获取日期字符串
        date_str = val.get('hudong', {}).get('last_processed_date')
        if date_str:
            try:
                # 将字符串解析为时间对象
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                # 保留 大于等于 截止时间的数据
                if dt >= cutoff_date:
                    filtered_data[key] = val
            except ValueError:
                # 如果日期格式不对，默认跳过
                continue

    # 3. 打印高信息密度日志
    total_cnt = len(data_dict)
    kept_cnt = len(filtered_data)
    dropped_cnt = total_cnt - kept_cnt
    drop_rate = (dropped_cnt / total_cnt * 100) if total_cnt > 0 else 0

    # 日志包含：当前时间、截止日期、输入->输出变化、移除数量及占比
    print(f"[FilterLog] {now.strftime('%Y-%m-%d %H:%M:%S')} | "
          f"Cutoff: {cutoff_date.strftime('%Y-%m-%d')} (Past {days}d) | "
          f"Items: {total_cnt} -> {kept_cnt} (Dropped {dropped_cnt}, {drop_rate:.1f}%)")

    return filtered_data

def fun(manager):
    global NEED_UPDATE_SIGN
    try:
        now = datetime.now()
        pre_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # 往前减去两天
        pre_midnight = pre_midnight - timedelta(days=2)

        # 查询今日已投稿的任务
        recent_uploaded_tasks = manager.find_tasks_after_time_with_status(pre_midnight, [TaskStatus.UPLOADED])

        processed_count = 0
        print("开始执行 fun 函数...当前时间:", datetime.datetime.now().isoformat())
        stop_event.clear()  # 清除停止事件
        today = datetime.date.today().isoformat()
        # 加载all_emote.json
        all_emote_list = load_processed_dict(ALL_BILIBILI_EMOTE_PATH)
        config_map = init_config()
        commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
        commenter_map = {}
        for key, detail_config in config_map.items():
            name = detail_config.get('name', key)
            # if name in ['mama']:
            #     continue
            cookie = detail_config.get('total_cookie', '')
            all_params = detail_config.get('all_params', {})
            commenter_map[key] = BilibiliCommenter(
                total_cookie=cookie,
                csrf_token=detail_config.get('BILI_JCT', ''),
                all_params=all_params,
            )
            print(f"已创建评论者 {name} (UID: {key})")
        print(f"共创建 {len(commenter_map)} 个评论者实例。")

        bvid_file_data = load_processed_dict(bvid_file_path)
        all_bvid_file_data = load_processed_dict(all_bvid_file_path)

        bvid_uid_map = {}
        all_found_videos = []
        for uid in config_map.keys():
            name = config_map[uid].get('name', uid)
            # if uid in ['3546965562362625']:
            #     continue
            # if name in ['hao', 'shuijun1', 'shuijun2', 'shuijun3', 'xiaodan', 'xiaoxiaosu', 'ruruxiao']:
            #     continue

            if NEED_UPDATE_SIGN:
                detail_config = config_map[uid]
                signature = random.choice(signatures)
                cookie = detail_config.get('total_cookie', '')
                result = update_bili_user_sign(cookie,signature)
                print(f"更新用户签名结果: {result}")

            logging.info(f"  > 正在获取UP主(UID: {uid} {name})的最新动态...")
            temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=25)
            bvid_uid_map.update({video.get('bvid'): uid for video in temp_found_videos if 'bvid' in video})
            all_found_videos.extend(temp_found_videos)
            bvid_file_data[name] = temp_found_videos
            for video in temp_found_videos:
                all_bvid_file_data[video.get('bvid')] = video

            save_json(all_bvid_file_path, all_bvid_file_data)
            save_json(bvid_file_path, bvid_file_data)
        NEED_UPDATE_SIGN = False
        all_found_videos.sort(key=lambda x: x.get('created', 0), reverse=True)
        # 只保留最近1小时的视频
        one_hour_ago = time.time() - 3600 * 3
        all_found_videos = [video for video in all_found_videos if video.get('created', 0) >= one_hour_ago]

        all_found_videos = all_found_videos
        print(f"共找到 {len(all_found_videos)} 个视频。")
        count = 0
        for video in all_found_videos:
            print(f"正在处理视频 BVID: {video.get('bvid', '未知')}...")
            count += 1
            start_time = time.time()
            bvid = video.get('bvid')
            uid = bvid_uid_map.get(bvid, '未知UID')
            hudong_info = gen_hudong_info(bvid, interaction_data, metadata_cache_with_uploads, all_emote_list)
            if hudong_info == {}:
                print(f"无互动信息跳过{bvid}")
                continue

            hudong_info, is_skip = process_single_video(bvid, hudong_info, uid, commenter_map, today)
            if not is_skip:
                processed_count += 1
            interaction_data[bvid] = {'hudong': hudong_info}
            save_json(interaction_data_file, interaction_data)
            print(
                f"视频 {bvid} 的互动信息已生成并保存。耗时: {time.time() - start_time:.2f} 秒 进度: {count}/{len(all_found_videos)} {datetime.datetime.now().isoformat()}")
            if stop_event.is_set():
                print("检测到停止请求，退出当前任务...")
                return  # 停止当前执行，退出
        print(
            f"所有视频处理所有完成所有，正在保存数据..当前时间: {datetime.datetime.now().isoformat()} 共处理 {processed_count} 个视频。共找到 {len(all_found_videos)} 个视频")
    except Exception as e:
        traceback.print_exc()
    finally:
        stop_event.set()  # 标记任务结束


def run_periodically(manager):
    while True:
        loop_start = time.time()  # 记录本轮 fun 开始时间

        stop_event.set()
        fun_thread = threading.Thread(target=fun, args=(manager,))
        fun_thread.start()
        fun_thread.join()

        elapsed = time.time() - loop_start
        remaining = max(0, 30 * 60 - elapsed)  # 剩余等待时间
        print(f"fun 执行耗时 {elapsed:.2f} 秒，等待 {remaining:.2f} 秒后再执行下一轮...")
        if remaining > 0:
            time.sleep(remaining)


if __name__ == '__main__':
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    # 启动定时任务线程
    threading.Thread(target=run_periodically, args=(manager,), daemon=True).start()

    # 主线程可用于其他任务，或者继续保持程序运行
    while True:
        time.sleep(10)

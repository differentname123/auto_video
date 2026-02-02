# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/5 12:28
:last_date:
    2025/12/5 12:28
:description:

"""
import ast
import difflib
import json
import os
import random
import re
import shutil
import string
import time
import traceback
from collections import defaultdict
from datetime import timedelta, datetime
from functools import wraps
from pathlib import Path
from typing import Union, List, Optional

import aiofiles
import aiohttp
from filelock import FileLock, Timeout


def read_json(json_path):
    """
    读取 JSON 文件并返回内容。

    Args:
        json_path (str): JSON 文件的路径。

    Returns:
        dict: 解析后的 JSON 内容。
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 文件 '{json_path}': {e}")


def save_json(json_path, data):
    dir_path = os.path.dirname(json_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 锁文件通常以 .lock 结尾
    lock_path = json_path + ".lock"

    # FileLock 会在文件系统层面创建锁，支持多进程和多线程安全
    with FileLock(lock_path):
        # 原子写入
        tmp_path = json_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=str)
        os.replace(tmp_path, json_path)


# 2. 基础辅助功能
async def download_cover_minimal(url: str, save_path) -> bool:
    """
    一个最小化的异步函数，用于下载单个图片 URL 并保存到本地。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    connector = aiohttp.TCPConnector(ssl=False)

    try:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(await response.read())
                return True
    except Exception as e:
        print(f"[ERROR] 下载图片失败 {url}: {e}")
        return False


def get_config(key):
    """
    从 config.json 文件中获取指定字段的值
    :param key: 配置字段名
    :return: 配置字段值
    """
    # 获取当前脚本所在目录
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent
    # 拼接 config.json 文件的绝对路径
    config_file = os.path.join(base_dir, 'config/config.json')

    # 检查 config.json 文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在，请检查文件路径。")

    # 读取配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件 '{config_file}' 格式错误: {e}")

    # 获取指定字段的值
    if key not in config_data:
        raise KeyError(f"配置文件中缺少字段: {key}")

    return config_data[key]





def read_file_to_str(filepath: Union[str, Path],
                     encoding: str = "utf-8",
                     errors: str = "strict") -> str:
    """
    读取文件并返回整个内容的字符串。

    参数:
        filepath: 文件路径（str 或 pathlib.Path）。
        encoding: 文本编码（默认 'utf-8'）。
        errors: 解码错误处理策略（'strict'|'replace'|'ignore' 等，默认 'strict'）。
                'strict' 会在遇到无法解码的字节时抛出 UnicodeDecodeError，
                'replace' 会用替代字符替换无法解码的字节，'ignore' 则忽略它们。

    返回:
        文件内容（str）。

    抛出:
        FileNotFoundError 如果文件不存在。
        UnicodeDecodeError 如果 decoding 失败且 errors='strict'。
    """
    p = Path(filepath)
    with p.open("r", encoding=encoding, errors=errors) as f:
        return f.read()



def is_valid_target_file_simple(path, min_size_bytes: int = 1) -> bool:
    try:
        if not path:
            return False
        p = Path(path)
        return p.is_file() and p.stat().st_size >= int(min_size_bytes)
    except Exception:
        return False



def string_to_object(input_str: str):
    """
    从字符串中提取并解析出 Python 列表或字典对象，设计得更加健壮。

    该函数增强了对不规范格式的容忍度，特别适合处理来自 LLM 的输出。

    核心功能：
    1.  **智能提取**: 自动在整个字符串中定位 JSON/Python 对象的边界（从第一个 '{' 或 '[' 到最后一个 '}' 或 ']），
        忽略前导和尾随的无关文本（例如 "当然，这是您要的JSON："）。
    2.  **兼容 Markdown**: 能够处理被 ```json ... ``` 代码块包裹的内容。
    3.  **错误修正**:
        - 自动移除常见的行内 (//) 和块级 (/* */) 注释。
        - 自动移除导致 JSON 解析失败的尾随逗号 (trailing commas)。
    4.  **双引擎解析**:
        - 首先尝试使用 `json.loads`，因为它更符合标准，速度更快。
        - 如果失败，则回退到 `ast.literal_eval`，以支持 Python 特有的字面量
          （如 `None`, `True`, `False` 以及单引号字符串）。

    如果无法找到或解析出有效的对象，则抛出 ValueError 异常。

    :param input_str: 包含列表或字典的输入字符串。
    :return: 解析后的 Python 列表或字典。
    :raises ValueError: 如果无法从字符串中找到或解析出有效的对象。
    :raises TypeError: 如果输入不是字符串。
    """
    # 0. 输入校验：处理 None 或非字符串输入
    if not isinstance(input_str, str):
        # 抛出 TypeError 更符合 Python 语义，但根据您的要求统一为 ValueError 也可以
        raise TypeError(f"输入必须是字符串，但收到了 {type(input_str).__name__}。")

    # 创建一个统一的错误信息生成器
    def _create_error_message(reason: str) -> str:
        # 预览原始输入的前50个字符
        preview = (input_str[:50] + '...') if len(input_str) > 50 else input_str
        return f"{reason} | 输入内容预览: '{preview}'"

    # 1. 智能提取：在字符串中寻找对象边界 (重构后，逻辑更清晰)
    first_bracket = input_str.find('[')
    first_curly = input_str.find('{')

    # 确定第一个开括号的位置
    if first_bracket == -1 and first_curly == -1:
        raise ValueError(_create_error_message("输入字符串中未找到疑似列表或字典的起始符号 '[' 或 '{'"))

    if first_bracket == -1:
        start_pos = first_curly
    elif first_curly == -1:
        start_pos = first_bracket
    else:
        start_pos = min(first_bracket, first_curly)

    # 确定最后一个闭括号的位置
    end_pos = max(input_str.rfind(']'), input_str.rfind('}'))

    if end_pos <= start_pos:
        raise ValueError(_create_error_message("未找到与起始括号匹配的结束括号 ']' 或 '}'"))

    # 提取出最可能包含对象的子字符串
    potential_obj_str = input_str[start_pos: end_pos + 1]

    # 2. 错误修正：清理提取出的字符串
    # 移除 JavaScript/JSONC 风格的注释
    potential_obj_str = re.sub(r"//.*", "", potential_obj_str)
    potential_obj_str = re.sub(r"/\*[\s\S]*?\*/", "", potential_obj_str, flags=re.MULTILINE)
    # 移除尾随逗号 (例如, [1, 2,])
    potential_obj_str = re.sub(r",\s*([}\]])", r"\1", potential_obj_str)
    cleaned_str = potential_obj_str.strip()

    # 3. 双引擎解析
    try:
        # 首先尝试使用 json.loads (更标准，通常更快)
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        # 如果 json.loads 失败，回退到 ast.literal_eval (更宽容，支持 Python 语法)
        try:
            return ast.literal_eval(cleaned_str)
        except (ValueError, SyntaxError, MemoryError) as e:
            # 如果两种方法都失败，则抛出最终的异常，并提供丰富的上下文信息
            cleaned_preview = (cleaned_str[:150] + '...') if len(cleaned_str) > 150 else cleaned_str
            error_reason = f"无法将提取的内容解析为列表或字典，解析器错误: {e}"
            # 最终的错误信息包含：原因，原始输入预览，以及尝试解析的内容预览
            raise ValueError(f"{_create_error_message(error_reason)}\n"
                             f"尝试解析的内容 (清理后): '''{cleaned_preview}'''")



def time_to_ms(time_input: str | float | int) -> int:
    """
    将多种时间格式统一转换为毫秒。
    该函数非常稳健，可以处理以下格式：
    - 数字 (int/float): 12.345 (代表秒)
    - 纯秒数字符串: "12.345"
    - 标准SRT时间码: "00:01:02,345" 或 "00:01:02.345"
    - 省略小时的时间码: "01:02.345"
    - 只有分秒的时间码: "03.482"

    Args:
        time_input: 多种格式的时间输入。

    Returns:
        总毫秒数 (int)。
    """
    if isinstance(time_input, (int, float)):
        return int(time_input * 1000)

    time_str = str(time_input).strip()

    try:
        return int(float(time_str.replace(',', '.')) * 1000)
    except ValueError:
        pass

    time_str = time_str.replace(',', '.')

    parts = time_str.split(':')
    h, m, s = 0, 0, 0.0

    try:
        if len(parts) == 3:  # HH:MM:SS.ms
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        elif len(parts) == 2:  # MM:SS.ms
            m = int(parts[0])
            s = float(parts[1])
        elif len(parts) == 1:  # SS.ms
            s = float(parts[0])
        else:
            raise ValueError("时间码中的冒号过多")

        return int((h * 3600 + m * 60 + s) * 1000)

    except (ValueError, IndexError):
        raise ValueError(f"无法解析的时间格式: '{time_input}'")


def ms_to_time(ms: int) -> str:
    """将毫秒转换为'HH:MM:SS.ms'格式的时间字符串。"""
    ms = int(ms)  # 确保输入是整数
    if ms < 0: ms = 0
    s, ms_rem = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    # --- 修改点：将逗号改为点 ---
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def merge_intervals(intervals):
    """
    合并相邻或重叠的时间段

    Args:
        intervals: 时间段列表，每个元素为 (start, end) 的元组

    Returns:
        合并后的时间段列表
    """
    # 处理空列表的情况
    if not intervals:
        return []

    # 按开始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # 初始化结果列表，第一个时间段作为起点
    merged = [sorted_intervals[0]]

    # 遍历剩余的时间段
    for current in sorted_intervals[1:]:
        # 获取当前合并结果中的最后一个时间段
        last = merged[-1]

        # 如果当前时间段与最后一个时间段相邻或重叠
        # 相邻的条件是：current[0] <= last[1]
        # （因为如果 current[0] == last[1]，它们是连续的，应该合并）
        if current[0] <= last[1]:
            # 合并时间段，结束时间取两者中的最大值
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # 不相邻，直接添加到结果中
            merged.append(current)

    return merged

def first_greater(target_num, num_list):
    """
    返回 num_list 中第一个严格大于 target_num 的元素；找不到时返回 None。
    """
    for x in num_list:
        if x > target_num:
            return x
    return None


def remove_last_punctuation(sentence: str) -> str:
  """
  如果句子的最后一个字符是标点符号，则移除它。

  参数:
    sentence: 输入的字符串。

  返回:
    移除末尾标点符号的字符串。
  """
  chinese_punctuations = "。，！？；：（）《》【】“”‘’、"

  # 将英文和中文标点合并
  all_punctuations = string.punctuation + chinese_punctuations

  if sentence and sentence[-1] in all_punctuations:
      return sentence[:-1]
  return sentence

def check_timestamp(all_timestamps, duration):
    """
    检查时间是否都在正常的范围内
    :param all_timestamps:
    :param duration:
    :return:
    """
    error_info = ""
    duration_ms = time_to_ms(duration)
    for ts in all_timestamps:
        ts_ms = time_to_ms(ts)
        if ts_ms < 0 or ts_ms > duration_ms:
            error_info += f"时间戳 {ts} 超出视频范围 (0 - {duration})。\n"
            return error_info
    return ""


def get_remaining_segments(duration_ms, remove_segments, min_duration=1000):
    """
    计算删除指定时间段后的视频保留区间（左闭右开区间 [start, end)）。

    约定：
      - 视频范围为 [0, duration_ms)
      - remove_segments 是一组 (start, end)，可能无序、重叠或超出边界
      - 返回值为按时间升序的非重叠保留区间列表 [(start, end), ...]

    参数：
      duration_ms    - 视频总时长（毫秒或秒），应为非负数
      remove_segments - 要删除的时间段列表，例如 [(10, 30), (60, 90)]
      min_duration    - [新增] 最小保留片段时长，默认为 0.1。
                        用于过滤因浮点数精度导致的极微小片段（防止 FFmpeg 报错 -ss >= -to）
    """
    # 边界与空输入处理
    if duration_ms <= 0:
        return []
    if not remove_segments:
        # 如果总时长也小于最小阈值，理论上不应保留，但在空删除列表情况下通常保留原视频
        return [(0, duration_ms)]

    # 1) 规范化并裁剪删除区间到视频范围内，丢弃无效区间
    cleaned = []
    for s, e in remove_segments:
        # 裁剪到 [0, duration_ms]
        if s < 0:
            s = 0
        if e > duration_ms:
            e = duration_ms
        # 只保留有效区间（开始 < 结束）
        if s < e:
            cleaned.append((s, e))

    if not cleaned:
        return [(0, duration_ms)]

    # 2) 按起点排序
    cleaned.sort(key=lambda seg: seg[0])

    # 3) 合并重叠或相邻的删除区间
    merged = []
    cur_start, cur_end = cleaned[0]
    for s, e in cleaned[1:]:
        if s <= cur_end:        # 重叠或相邻（s == cur_end 也当作合并处理）
            # 扩展当前区间右端点
            if e > cur_end:
                cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    # 4) 计算补集：保留区间
    remaining = []
    prev_end = 0
    for s, e in merged:
        # 【修改点】只有当片段长度大于阈值时才保留，避免浮点数精度产生 0 秒片段
        if s - prev_end >= min_duration:
            remaining.append((prev_end, s))
        prev_end = e

    # 视频末尾还有剩余
    # 【修改点】同上，检查尾部片段长度
    if duration_ms - prev_end >= min_duration:
        remaining.append((prev_end, duration_ms))

    return remaining

def distribute_by_counts(accounts, allocation, default=None):
    """
    按顺序分配：
    - accounts: 可迭代（例如 list(accounts.keys())）
    - allocation: [(count, proxy), ...], count 可为 None 表示“剩下的全部”
    - default: allocation 没覆盖到时的填充值
    返回：与 accounts 等长的 proxy 列表
    """
    n = len(accounts)
    proxies = []
    assigned = 0
    for count, proxy in allocation:
        if count is None:
            proxies += [proxy] * (n - assigned)
            assigned = n
            break
        if count <= 0:
            continue
        take = min(count, n - assigned)
        proxies += [proxy] * take
        assigned += take
        if assigned >= n:
            break
    if assigned < n:
        proxies += [default] * (n - assigned)
    return proxies



def init_config():
    config_map = {}

    # 账号配置：key 是 config_map 中的 UID，value 是账号的前缀（name）
    accounts = {
        # '3546973573482600': 'shuijun3',
        '3690972783315441': 'mama',
        # '3546717871934392': 'nana',
        # '3546759607355668': 'ruru',
        # '3546973825141556': 'tao',
        # '437687603': 'taoxiao',

        '3546977480477153': 'hong',
        # '3546977184778261': 'yan',
        # '3546947566700892': 'su',
        '3546764430805071': 'ningtao',

        '3546977048463920': 'jie',
        # '3546977369328324': 'qiqi',
        '3690990592329746': 'xue',
        # '3546977600014812': 'cai',
        # '3632311899786168':'xiaosu',
        '3494364332427809':'xiaoxiaosu',
        '3546978046708266':'jun',
        # '3632306801609223': 'shuijun2',
        # '3632301781026991': 'junxiao',
        '3632319661344845': 'junda',
        '3632315397834763':'lin',
        # '3632317008447555': 'jj',
        # '3546913316014394':'xiaohao',
        # '196823511': 'hao',
        '3546965935655696': 'danzhu',
        '3632300566776371': 'dan',
        # '3546982253595619': 'ning',
        '3546970725550911': 'yiyi',
        '3546981674781282': 'qiqixiao',
        # '3690979349498713': 'mu',
        # '3690989401147694': 'yang',
        # '3690996967672056': 'ruruxiao',
        # '3632306870814900': 'xiaodan',
        # '3632309148322699': 'xiaoxue',

        '3690973307603884': 'dahao',
        # '3632313749473288': 'shun',
        '3632314758203558': 'xiaocai',
        '1516147639': 'qizhu',
        # '3632318595991783': 'xiaomu',
        # "3546971140786786": 'ping',
        # "3690972028340306": 'xiu',
        "3690971298531782": 'zhong',
        "3546594393721601": 'qiuru',
        "3546764430805848": 'huazhu'

    # '3546909677455941': 'base'  # 如果需要恢复 base 账号，取消注释即可
    }

    # 三段代理：
    # - 前5个账户使用 proxy_A
    # - 中间5个账户为 None
    # - 剩下的账户使用 last_proxy
    proxy_A = {"http": "http://115.190.54.74:8888", "https": "http://115.190.54.74:8888"}
    no_proxy = {"http": None, "https": None}
    proxy_B = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

    account_items = list(accounts.items())
    keys = [k for k, _ in account_items]

    # 一个常见策略：前5个 no_proxy，接2个 proxy_B，剩下用 proxy_A
    proxies_values = distribute_by_counts(
        keys,
        allocation=[
            (10, proxy_A),
            (5, proxy_B),
            (None, no_proxy),  # None 表示“其余全部”
        ],
        default=None
    )

    for idx in range(len(account_items)):
        uid, name = account_items[idx]
        sessdata = get_config(f"{name}_bilibili_sessdata_cookie")
        bili_jct = get_config(f"{name}_bilibili_csrf_token")
        total_cookie = get_config(f"{name}_bilibili_total_cookie")
        proxies = proxies_values[idx] if idx < len(proxies_values) else no_proxy

        all_params = {
            "uid": uid,
            "name": name,
            "SESSDATA": sessdata,
            "BILI_JCT": bili_jct,
            "total_cookie": total_cookie,
            "proxies": proxies
        }

        config_map[uid] = {
            "name": name,
            "SESSDATA": sessdata,
            "BILI_JCT": bili_jct,
            "total_cookie": total_cookie,
            "all_params": all_params
        }

    return config_map


def get_top_comments(video_info_dict, target_limit=20, min_guarantee=2, need_image=False):
    """
    从视频字典中筛选评论：
    1. 每个视频保底选择 min_guarantee 条（不足则全选）。
    2. 剩余名额从所有视频剩下的评论中，按点赞量由高到低补齐。
    """
    selected_comments = []
    candidate_pool = []

    for info in video_info_dict.values():
        # 提取评论数据: [(内容, 点赞), ...]
        comments = info.get('comment_list', [])
        if not need_image:
            comments = [(c[0], c[1]) for c in info.get('comment_list', [])]

        # 当前视频评论按点赞倒序排列
        comments.sort(key=lambda x: x[1], reverse=True)

        # 核心逻辑：切分保底区和候选区
        selected_comments.extend(comments[:min_guarantee])
        candidate_pool.extend(comments[min_guarantee:])

    # 计算剩余需要填充的名额
    slots_needed = target_limit - len(selected_comments)

    if slots_needed > 0:
        # 候选池按点赞倒序，取前 N 个补齐
        candidate_pool.sort(key=lambda x: x[1], reverse=True)
        selected_comments.extend(candidate_pool[:slots_needed])

    # 最终结果按点赞数再次整理（可选，为了列表有序）
    selected_comments.sort(key=lambda x: x[1], reverse=True)

    return selected_comments



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

def format_bilibili_emote(comment_list, all_emote_list):
    """
    进行b站的emote转换，避免没有正常输出表情
    """
    for comment in comment_list:
        # 将第一个元素调用 replace_bracketed
        comment[0] = replace_bracketed(comment[0], all_emote_list)



def extract_guides(data_list):
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

    for upload_info in data_list:
        # 获取“简介”部分
        brief = upload_info.get("introduction", {})
        # 提取互动引导
        prompt = brief.get("interaction_guide")
        if isinstance(prompt, str) and prompt.strip():
            interaction_prompts.append(prompt.strip())
        # 提取补充信息
        note = brief.get("supplement_info")
        if isinstance(note, str) and note.strip():
            supplementary_notes.append(note.strip())

    return interaction_prompts, supplementary_notes



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



def filter_danmu(danmu_list, total_seconds):
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
            traceback.print_exc()
            print(f"处理弹幕时出错，跳过该条弹幕。错误信息: {e}")
            continue

    processed_danmu_count = 0
    for item in processed_danmu:
        if isinstance(item.get('推荐弹幕内容'), list):
            processed_danmu_count += len(item['推荐弹幕内容'])

    target_num = 20
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


def delete_files_in_dir_except_target(keep_file_path):
    """
    获取指定文件的目录，并删除该目录下除了该文件以外的所有内容。

    Args:
        keep_file_path (str): 需要保留的文件的路径
    """
    try:
        # 1. 获取绝对路径，确保路径比较准确
        abs_keep_path = os.path.abspath(keep_file_path)

        # 2. 获取目录路径
        directory = os.path.dirname(abs_keep_path)

        # 如果目录不存在，直接返回，不报错
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return

        # 3. 遍历目录下的所有内容
        for filename in os.listdir(directory):
            # 拼接当前遍历到的文件的完整路径
            file_path = os.path.join(directory, filename)

            # 获取当前文件的绝对路径用于比较
            abs_file_path = os.path.abspath(file_path)

            # 4. 核心逻辑：如果是我们要保留的文件，直接跳过
            if abs_file_path == abs_keep_path:
                continue

            # 5. 删除操作（由于是遍历，建议加一个内部try，防止因为某个文件被占用导致整个流程中断）
            try:
                if os.path.isfile(abs_file_path) or os.path.islink(abs_file_path):
                    os.remove(abs_file_path)  # 删除文件或软链接
                    print(f"已删除文件: {filename}")
                elif os.path.isdir(abs_file_path):
                    shutil.rmtree(abs_file_path)  # 删除子文件夹
                    print(f"已删除目录: {filename}")
            except Exception as e:
                # 打印错误但不抛出，确保继续处理下一个文件
                print(f"删除 {filename} 失败: {e}")

    except Exception as e:
        # 6. 外层捕获所有未知异常，确保绝不影响调用者
        print(f"清理目录函数发生错误: {e}")
        # 这里可以选择打印堆栈信息用于调试，但在生产环境中可以注释掉
        # traceback.print_exc()



def safe_process_limit(limit, name, lock_dir="./process_locks"):
    """
    基于文件锁的稳定并发控制装饰器，带等待时间统计。

    :param limit: 允许的并发数量
    :param name: 业务名称（不同函数用不同名称）
    :param lock_dir: 锁文件存放位置
    """
    # 确保锁目录存在
    if not os.path.exists(lock_dir):
        try:
            os.makedirs(lock_dir, exist_ok=True)
        except OSError:
            # 多进程并发创建目录时可能会报错，忽略即可
            pass

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 记录开始排队的时间
            start_wait_time = time.time()

            acquired_lock = None
            slot_index = -1

            # 2. 进入抢锁循环
            while True:
                # 尝试遍历所有的“槽位” (0 到 limit-1)
                for i in range(limit):
                    lock_path = os.path.join(lock_dir, f"{name}_{i}.lock")
                    lock = FileLock(lock_path)

                    try:
                        # timeout=0 表示“非阻塞”，抢不到立刻报错，不傻等
                        lock.acquire(timeout=0)

                        # 抢到了！
                        acquired_lock = lock
                        slot_index = i
                        break
                    except Timeout:
                        # 这个槽位被人占了，继续试下一个
                        continue

                if acquired_lock:
                    # 成功获取锁，跳出死循环
                    break

                # 如果所有槽位都满了，稍微睡一会再试
                # 随机睡眠 0.05 ~ 0.15 秒，避免“惊群效应”（所有进程同时醒来抢）
                time.sleep(random.uniform(1, 5))

            # 3. 计算等待时间
            end_wait_time = time.time()
            wait_duration = end_wait_time - start_wait_time

            # 打印等待信息（如果等待时间 > 0.1秒，说明发生排队了，可以重点关注）
            if wait_duration > 0.01:
                print(f"函数 [{name}] 槽位[{slot_index}] 获取成功。排队耗时: {wait_duration:.4f} 秒")
            else:
                print(f"函数 [{name}] 槽位[{slot_index}] 秒抢成功。无需等待。")

            # 4. 执行真正的业务逻辑
            try:
                return func(*args, **kwargs)
            finally:
                # 5. 无论代码是否报错，必须释放锁
                # 操作系统保证：即使进程被 Kill，这个锁也会被内核释放
                if acquired_lock:
                    acquired_lock.release()

        return wrapper

    return decorator


def get_simple_play_distribution(data_list, start_timestamp, interval_minutes=30, max_elapsed_minutes=60*48):
    clean_data = []
    for item in data_list:
        time_str = next(iter(item))
        clean_data.append((datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S"), item[time_str]['play_count']))
    clean_data.sort(key=lambda x: x[0])

    result = {}
    elapsed_minutes = interval_minutes
    # start_timestamps是一个时间戳， 将start_timestamp转换为datetime对象
    start_time = datetime.fromtimestamp(start_timestamp)

    current_time = start_time

    while True:
        # Step 1: 找到最近的 Next (列表里第一个时间 > 当前时间的点)
        # 使用 enumerate 是为了方便拿到索引，从而知道 pre 是谁
        next_index = -1
        for i, (t, c) in enumerate(clean_data):
            if t > current_time:
                next_index = i
                break

        # 如果找不到 Next (说明当前时间已经超过了所有数据的时间)，直接结束
        if next_index == -1:
            break

        # Step 2: 确定 Prev
        next_time, next_val = clean_data[next_index]

        if next_index > 0:
            # 正常情况：Next的前一个就是 Prev
            prev_time, prev_val = clean_data[next_index - 1]
        else:
            # 特殊情况：Next是第一个数据，说明当前时间比所有数据都早
            # 按照你的逻辑：Prev默认为0，时间取起始时间(或当前时间)
            prev_time = start_time
            prev_val = 0

        # Step 3: 计算增量和占比 (直接套公式)
        total_seconds = (next_time - prev_time).total_seconds()
        target_seconds = (current_time - prev_time).total_seconds()

        # 防止除以0
        ratio = target_seconds / total_seconds if total_seconds > 0 else 0
        increase = next_val - prev_val

        # 公式：(增数 * 占比) + 上个时间点数据
        current_val = (increase * ratio) + prev_val

        # 记录结果
        result[elapsed_minutes] = int(current_val)

        # Step 4: 时间推移
        current_time += timedelta(minutes=interval_minutes)
        elapsed_minutes += interval_minutes
        if elapsed_minutes > max_elapsed_minutes:
            break

    return result


def calculate_averages(data_list, min_count=20):
    """
    计算列表中字典 key 的平均值。

    参数:
    data_list: 包含字典的列表
    min_count: 阈值，只有 key 出现的次数大于该值时才计算平均值

    返回:
    一个字典，包含满足条件的 key 及其平均值
    """

    # 用于存储每个 key 的累加值
    key_sums = defaultdict(float)
    # 用于存储每个 key 出现的次数
    key_counts = defaultdict(int)

    # 1. 遍历列表中的每个字典，收集数据
    for item in data_list:
        for key, value in item.items():
            key_sums[key] += value
            key_counts[key] += 1

    # 2. 计算平均值，并过滤掉次数不足的 key
    averages = {}
    for key, count in key_counts.items():
        # 注意题目要求是“大于” min_count
        if count > min_count:
            averages[key] = key_sums[key] / count

    return averages


def gen_true_type_and_tags(upload_info_list):
    """
    生成准确的视频类型以及实体标签
    :return:
    """
    try:
        all_tags_info = {}
        for upload_info in upload_info_list:
            tags = upload_info.get("tags", [])
            for tag in tags:
                all_tags_info[tag] = all_tags_info.get(tag, 0) + 1

        category_id_list = [upload_info["category_id"] for upload_info in upload_info_list if "category_id" in upload_info]
        category_data_info = read_json(r'W:\project\python_project\auto_video\config\bili_category_data.json')
        category_name_list = []
        for category_id in category_id_list:
            category_name = category_data_info.get(str(category_id), {}).get("name", "")
            if category_name:
                category_name_list.append(category_name)
        category_name_list_str = str(category_name_list)
        video_type = "fun"
        if category_name_list_str:
            if "游戏" in category_name_list_str:
                video_type = "game"
            elif "运动" in category_name_list_str or "体育" in category_name_list_str:
                video_type = "sport"
            elif "搞笑" in category_name_list_str or "趣味" in category_name_list_str or "娱乐" in category_name_list_str or "新闻" in category_name_list_str or "影视" in category_name_list_str or "情感" in category_name_list_str or "知识" in category_name_list_str:
                video_type = "fun"

        return video_type, all_tags_info
    except Exception as e:
        traceback.print_exc()
        return None, None


def get_user_type(user_name):
    user_type = "fun"
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type_info = user_config.get('user_type_info')
    for user_type, user_list in user_type_info.items():
        if user_name in user_list:
            return user_type
    return user_type


def has_continuous_common_substring(list1, list2, threshold):
    """
    判断两个列表中是否存在一对字符串，其连续公共子串长度 > threshold
    即：是否存在长度为 threshold + 1 的连续公共子串
    """
    # 我们要寻找的连续长度至少是 threshold + 1
    target_len = threshold + 1
    found_any = False

    # 为了方便查找，我们先记录 list1 中所有子串及其来源
    # 格式：{子串: [来源字符串1, 来源字符串2...]}
    sub_map1 = {}
    for s1 in list1:
        if len(s1) >= target_len:
            for i in range(len(s1) - target_len + 1):
                sub = s1[i: i + target_len]
                if sub not in sub_map1:
                    sub_map1[sub] = set()
                sub_map1[sub].add(s1)

    # 存储找到的结果，防止重复打印
    results = []

    # 遍历 list2 进行比对
    for s2 in list2:
        if len(s2) >= target_len:
            for i in range(len(s2) - target_len + 1):
                sub = s2[i: i + target_len]
                if sub in sub_map1:
                    found_any = True
                    # 获取该子串在 list1 中的所有来源字符串
                    for s1_origin in sub_map1[sub]:
                        match_info = {
                            "substring": sub,
                            "from_list1": s1_origin,
                            "from_list2": s2
                        }
                        if match_info not in results:
                            results.append(match_info)

    return found_any, results


def has_long_common_substring(str1, str2, threshold=2):
    """
    判断两个字符串是否有长度 *大于* 阈值的连续公共子串，并返回匹配到的内容。

    :param str1: 第一个字符串
    :param str2: 第二个字符串
    :param threshold: 阈值 (整数)
    :return: (Boolean, String or None) - (是否匹配, 匹配到的子串)
    """
    # 目标长度必须大于阈值，所以至少是 threshold + 1
    target_len = threshold + 1

    # 这一步其实主要是为了防止 target_len 超过字符串本身长度导致逻辑错误
    # 但实际主要的判断在下面的 len(short_str) < target_len
    target_len = min(target_len, len(str1), len(str2))

    # 优化：为了减少循环次数，我们遍历较短的那个字符串
    if len(str1) < len(str2):
        short_str, long_str = str1, str2
    else:
        short_str, long_str = str2, str1

    # 如果短字符串本身的长度都不够目标长度，直接返回 False, None
    if len(short_str) < target_len:
        return False, None

    # 滑动窗口遍历
    for i in range(len(short_str) - target_len + 1):
        # 截取长度为 target_len 的子串
        sub = short_str[i: i + target_len]

        # 检查该子串是否存在于长字符串中
        if sub in long_str:
            # 找到匹配，返回 True 和具体的子串
            return True, sub

    # 未找到匹配
    return False, None
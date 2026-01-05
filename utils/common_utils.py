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
import json
import os
import re
import string
from pathlib import Path
from typing import Union

import aiofiles
import aiohttp
from filelock import FileLock


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
            json.dump(data, f, ensure_ascii=False, indent=4)
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
    """
    极简判断：文件存在且大小 >= min_size_bytes。
    默认 min_size_bytes=1 （即大小必须大于 0 字节）。
    """
    try:
        p = Path(path)
        return p.exists() and p.stat().st_size >= int(min_size_bytes)
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


def get_remaining_segments(duration_ms, remove_segments):
    """
    计算删除指定时间段后的视频保留区间（左闭右开区间 [start, end)）。

    约定：
      - 视频范围为 [0, duration_ms)
      - remove_segments 是一组 (start, end)，可能无序、重叠或超出边界
      - 返回值为按时间升序的非重叠保留区间列表 [(start, end), ...]

    参数：
      duration_ms    - 视频总时长（毫秒），应为非负整数
      remove_segments - 要删除的时间段列表，例如 [(1000, 30000), (60000, 90000)]
    """
    # 边界与空输入处理
    if duration_ms <= 0:
        return []
    if not remove_segments:
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
        if prev_end < s:
            remaining.append((prev_end, s))
        prev_end = e

    # 视频末尾还有剩余
    if prev_end < duration_ms:
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
        # '3546979686681114': 'ruru',
        # '3546973825141556': 'tao',
        '437687603': 'taoxiao',

        # '3546977480477153': 'hong',
        # '3546977184778261': 'yan',
        # '3546947566700892': 'su',


        '3546977048463920': 'jie',
        # '3546977369328324': 'qiqi',
        # '3632304865937878': 'xue',
        # '3546977600014812': 'cai',
        '3632311899786168':'xiaosu',
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
        '3546982253595619': 'ning',
        '3546970725550911': 'yiyi',
        '3546981674781282': 'qiqixiao',
        # '3690979349498713': 'mu',
        # '3690989401147694': 'yang',
        # '3690982356814585': 'ruruxiao',
        # '3632306870814900': 'xiaodan',
        '3632309148322699': 'xiaoxue',

        # '3690973307603884': 'dahao',
        '3632313749473288': 'shun',
        # '3632314758203558': 'xiaocai',
        '1516147639': 'qizhu',
        '3632318595991783': 'xiaomu',
        "3546971140786786": 'ping',
        # "3690972028340306": 'xiu',
        "3690971298531782": 'zhong'

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


def get_top_comments(video_info_dict, target_limit=20, min_guarantee=2):
    """
    从视频字典中筛选评论：
    1. 每个视频保底选择 min_guarantee 条（不足则全选）。
    2. 剩余名额从所有视频剩下的评论中，按点赞量由高到低补齐。
    """
    selected_comments = []
    candidate_pool = []

    for info in video_info_dict.values():
        # 提取评论数据: [(内容, 点赞), ...]
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
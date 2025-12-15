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
from pathlib import Path
from typing import Union

import aiofiles
import aiohttp
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
    """
    将数据保存为 JSON 文件。如果路径不存在则自动创建。
    """
    dir_path = os.path.dirname(json_path)
    if dir_path:  # 只有在有实际目录时才创建
        os.makedirs(dir_path, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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
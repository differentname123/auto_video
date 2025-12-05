# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/5 12:28
:last_date:
    2025/12/5 12:28
:description:
    
"""
import json
import os
from pathlib import Path

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
# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/1/5 18:07
:last_date:
    2026/1/5 18:07
:description:
    获取有奖话题列表
"""
import os
import time
import json

from utils.bilibili.bilibili_uploader import get_bili_category_id, fetch_bili_topics
from utils.common_utils import read_json, init_config, save_json


def find_paid_topics(topics):
    """
    分析活动数据，筛选出可能有金钱/贝壳/现金奖励的话题。

    Args:
        json_data (dict): 包含活动信息的字典

    Returns:
        list: 包含原始topic字段并增加matched_reason字段的列表
    """

    # 定义暗示有“钱”的关键词
    money_keywords = [
        "钱", "奖金", "现金", "激励金", "激励计划",
        "瓜分", "贝壳", "万元", "赏金", "大奖", "红包"
    ]

    paid_topics = []

    if not topics:
        return []

    for topic in topics:
        show_activity_icon = topic.get("show_activity_icon")

        # 保持原逻辑：必须有 icon 才继续
        if not show_activity_icon:
            continue

        # 1. 提取字段用于拼接检查（仅用于判断逻辑，不改变输出结构）
        name = topic.get("topic_name") or ""
        desc = topic.get("topic_description") or ""
        act_text = topic.get("activity_text") or ""
        act_desc = topic.get("activity_description") or ""

        # 拼接临时字符串用于搜索
        search_text = f"{name} {desc} {act_text} {act_desc}".strip()

        # 2. 检查是否包含关键词
        is_paid = False
        matched_reason = ""

        for keyword in money_keywords:
            if keyword in search_text:
                is_paid = True
                matched_reason = f"匹配到关键词: {keyword}"
                break

        # 3. 构造返回结果：保留原始所有字段，仅增加 matched_reason
        if is_paid:
            # 使用 copy() 避免直接修改原始数据源中的对象（是个好习惯）
            result_item = topic.copy()
            result_item["matched_reason"] = matched_reason

            paid_topics.append(result_item)

    return paid_topics


def get_all_paid_topics(max_age_days=1):
    topic_file = r'W:\project\python_project\auto_video\config\paid_topics.json'
    category_data_file = r'W:\project\python_project\auto_video\config\bili_category_data.json'

    need_update = True
    if os.path.exists(topic_file):
        mtime = os.path.getmtime(topic_file)
        current_time = time.time()
        if current_time - mtime < 86400*max_age_days:  # 1天 = 60*60*24 = 86400秒
            need_update = False
    # 如果不需要更新，直接读取返回
    if not need_update:
        print("本地文件未超过1天，直接读取已有的文件...")
        return read_json(topic_file), read_json(category_data_file)

    print("本地文件不存在或已超过1天，开始拉取最新数据...")
    paid_topic_all = read_json(topic_file)
    category_data_all = read_json(category_data_file)

    user_config = init_config()
    total_cookie = ""  # 初始化变量，防止下面循环没找到报错

    for uid, user_info in user_config.items():
        total_cookie = user_info.get("total_cookie", "")

        if total_cookie:
            break

    if not total_cookie:
        print("未找到有效的 Cookie，无法拉取数据")
        return paid_topic_all

    category_data = get_bili_category_id(total_cookie)
    category_data_list = category_data.get("data", {}).get('type_list', [])

    for category_info in category_data_list:
        category_name = category_info.get("name", "Unknown")  # 防止取不到报错
        try:
            category_id = category_info.get("id")
            category_data_all[category_name] = category_info
            category_data_all[category_id] = category_info
            topic_json = fetch_bili_topics(total_cookie, type_pid=category_id)
            topics = topic_json.get("data", {}).get("topics", [])
            paid_topics = find_paid_topics(topics)
            # 为所有的paid_topics增加category_id和category_name字段
            for topic in paid_topics:
                topic["category_id"] = category_id
                topic["category_name"] = category_name

            paid_topic_all[category_name] = paid_topics
            print(f"分类 {category_name} 下找到 {len(paid_topics)} 个有奖话题")
        except Exception as e:
            print(f"处理分类 {category_name} 时出错: {e}")
            continue

    save_json(topic_file, paid_topic_all)
    save_json(category_data_file, category_data_all)

    # 拉取完后返回最新数据
    return paid_topic_all, category_data_all

if __name__ == "__main__":
    get_all_paid_topics(0)
# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/14 18:39
:last_date:
    2025/12/14 18:39
:description:
    进行视频的制作以及投稿
"""
import concurrent.futures
import os
import random
import time
import traceback
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

import cv2
from rich import box
from rich.console import Console
from rich.table import Table

from application.process_video import process_single_task, query_need_process_tasks
from application.video_common_config import TaskStatus, ERROR_STATUS, check_failure_details, build_task_video_paths, \
    SINGLE_DAY_UPLOAD_COUNT, SINGLE_UPLOAD_COUNT, USER_STATISTIC_INFO_PATH
from utils.bilibili.bilibili_uploader import upload_to_bilibili
from utils.common_utils import read_json, is_valid_target_file_simple, init_config, save_json
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from utils.video_utils import get_frame_at_time_safe, create_enhanced_cover
config_map = {}
error_user_map = {}

account_executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = defaultdict(
    lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1)
)

def gen_user_upload_info(uploaded_tasks_today):
    """
    通过今日投稿的任务生成用户投稿的信息
    """
    # 定义默认值结构
    user_upload_info = defaultdict(lambda: {'today_upload_count': 0, 'platform_upload_count': 0, 'latest_upload_time': datetime.min})

    for task in uploaded_tasks_today:
        user_name = task['userName']
        upload_time = task['uploaded_time']
        play_count = task.get('play_count', None)

        # 更新数据
        info = user_upload_info[user_name]
        info['today_upload_count'] += 1
        if play_count:
            info['platform_upload_count'] += play_count
        # 比较并保留较大的时间
        if upload_time > info['latest_upload_time']:
            info['latest_upload_time'] = upload_time

    return dict(user_upload_info)  # 转回普通字典返回


def sort_tasks(existing_video_tasks, not_existing_video_tasks, user_info_map):
    """
    对任务列表进行排序并合并
    规则:
    1. 优先展示 existing_video_tasks，然后是 not_existing_video_tasks
    2. 内部排序规则: count(asc) -> schedule_date字符串(asc) -> update_time(asc)
    """

    def get_sort_key(task):
        # 1. 获取 userName
        user_name = task.get('userName')

        # 2. 获取 count (从 user_info_map 中查找)
        count = user_info_map.get(user_name, 0)

        # 3. 获取 schedule_date (字符串直接使用)
        # 结构: task['creation_guidance_info']['schedule_date']
        guidance_info = task.get('creation_guidance_info', {})

        # 直接获取字符串 '2026-01-05'
        # 给个默认值 '' (空字符串)，以防数据缺失导致排序报错
        schedule_date_str = guidance_info.get('schedule_date', '')

        # 4. 获取 update_time
        update_time = task.get('update_time', '')

        # 5. 返回元组
        # 字符串比较: '2026-01-05' < '2026-01-06'，符合预期
        return (count, schedule_date_str, update_time)

    # 分别对两个列表执行排序
    # Python的sort是原地排序(in-place)，无需重新赋值
    existing_video_tasks.sort(key=get_sort_key)
    not_existing_video_tasks.sort(key=get_sort_key)

    # 合并列表，existing 在前
    return existing_video_tasks + not_existing_video_tasks, existing_video_tasks, not_existing_video_tasks


def check_type(task_info, user_config):
    """
    检查用户类型与视频题材是否匹配。
    题材映射：
      - 包含 '游戏' -> 'game'
      - 包含 '运动' 或 '体育' -> 'sport'
      - 包含 '搞笑'/'趣味'/'娱乐'/'新闻' -> 'fun'
    """
    user_name = task_info.get("userName", "other")
    upload_info_list = task_info.get("upload_info")
    # 获取category_id
    category_id_list = [upload_info["category_id"] for upload_info in upload_info_list if "category_id" in upload_info]
    category_data_info = read_json(r'W:\project\python_project\auto_video\config\bili_category_data.json')
    category_name_list = []
    for category_id in category_id_list:
        category_name = category_data_info.get(str(category_id), {}).get("name", "")
        if category_name:
            category_name_list.append(category_name)
    category_name_list_str = str(category_name_list)
    video_type = "no"
    if category_name_list_str:
        if "游戏" in category_name_list_str:
            video_type = "game"
        elif "运动" in category_name_list_str or "体育" in category_name_list_str:
            video_type = "sport"
        elif "搞笑" in category_name_list_str or "趣味" in category_name_list_str or "娱乐" in category_name_list_str or "新闻" in category_name_list_str:
            video_type = "fun"
    user_type = "other"
    user_type_info = user_config.get('user_type_info')
    for user_type , user_list in user_type_info.items():
        if user_name in user_list:
            break

    if user_type != video_type:
        error_info = f"⚠️ 用户 {user_name} 的类型 {user_type} 与视频题材 {category_name_list_str} 的类型 {video_type} 不匹配，跳过上传。"
        return error_info
    return ""


def get_wait_minutes():
    """
    根据当前时间的小时数，返回一个非线性的等待分钟数。
    - 凌晨和清晨等待时间最长。
    - 白天和傍晚逐渐减少。
    - 深夜等待时间最短。
    - 等待时间以5分钟为单位变化。

    Returns:
        int: 建议的等待分钟数。
    """
    # 1. 获取当前时间的小时数 (0-23)
    current_hour = datetime.now().hour

    # 2. 根据不同的时间段，返回不同的等待时间
    # 规则：越早时间越长，越晚时间越短
    if current_hour <= 8:  # 清晨 06:00 - 08:59，开始苏醒，等待时间减少
        return 40

    elif current_hour <= 11:  # 上午 09:00 - 11:59，工作时间，等待时间减少
        return 30

    elif current_hour <= 17:  # 中午及下午 12:00 - 17:59，活跃时间
        return 20

    elif current_hour <= 21:  # 傍晚 18:00 - 21:59，晚上休息前
        return 10

    else:  # 深夜 22:00 - 23:59，准备休息，等待时间最短
        return 0

def check_need_upload(task_info, user_upload_info, current_time, already_upload_users, user_config, config_map, max_count=SINGLE_DAY_UPLOAD_COUNT):
    """
    总的来说就是检查该任务是否应该投稿
    :param task_info:
    :param user_upload_info:
    :return:
    """
    creation_guidance_info = task_info.get('creation_guidance_info', {})
    log_pre = f"{task_info.get('video_id_list', [])} {creation_guidance_info} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    global error_user_map

    schedule_date = creation_guidance_info.get('schedule_date', '2026-01-05')
    is_future = datetime.strptime(schedule_date, '%Y-%m-%d').date() > datetime.now().date()
    if is_future:
        print(f"还没到计划的投稿时间，跳过 {log_pre}")
        return False


    user_name = task_info.get('userName')
    if user_name not in config_map.keys():
        print(f"⚠️ 跳过 {user_name} 用户上传 请检查配置数据 {log_pre}")
        return False


    if user_name in error_user_map.keys():
        error_info = error_user_map[user_name]
        print(f'{user_name} 最近报错为 {error_info} 跳过 {log_pre}')
        return False

    if user_name in already_upload_users:
        print(f"{user_name} 本轮已投稿，跳过 {log_pre}")
        return False



    self_user_list = user_config.get('self_user_list', [])
    if user_name in self_user_list:
        error_info = check_type(task_info, user_config)
        if error_info:
            print(f"{user_name} 检查题材报错 {error_info}，跳过 {log_pre}")
            return False
    if len(already_upload_users) >= SINGLE_UPLOAD_COUNT:
        print(f"本轮已投稿用户过多，跳过 {log_pre}")
        return False

    right_now_user_list = user_config.get('right_now_user_list', [])
    if user_name not in right_now_user_list:
        if not (5 <= datetime.now().hour < 24):
            cooldown_reason = "当前时间不在允许的上传时间段（5点-24点）内。"
            print(f"{user_name} 因为 {cooldown_reason} 跳过 {log_pre}")

        need_waite_minutes = get_wait_minutes()
        latest_upload_time = user_upload_info.get(user_name, {}).get('latest_upload_time', datetime.min)
        # 计算和上次投稿的差值分数数
        time_diff = (current_time - latest_upload_time).total_seconds() / 60

        if time_diff < need_waite_minutes:
            print(f"{user_name} 距离上次投稿仅 {time_diff:.2f} 分钟，一共需等待 {need_waite_minutes} 分钟，跳过 {log_pre}")
            return False

    platform_upload_count = user_upload_info.get(user_name, {}).get('platform_upload_count', 0)
    today_upload_count = user_upload_info.get(user_name, {}).get('today_upload_count', 0)
    if platform_upload_count >= max_count or today_upload_count > 25:
        print(f"{user_name}  今天投稿 {today_upload_count} 实际数量{platform_upload_count} 今日投稿次数已达上限 {max_count} 次，跳过 {log_pre}")
        return False


    return True


def gen_video(task_info, config_map, user_config, manager):
    failure_details = {}
    try:
        failure_details, video_info_dict, chosen_script = process_single_task(task_info, manager, gen_video=True)
        user_name = task_info.get('userName')
        all_task_video_path_info = build_task_video_paths(task_info)
        final_output_path = all_task_video_path_info['final_output_path']
        account_config = config_map.get(user_name)
        upload_params = build_bilibili_params(final_output_path, chosen_script, user_config, user_name, video_info_dict, account_config)

        return failure_details, video_info_dict, chosen_script, upload_params
    except Exception as e:
        traceback.print_exc()
        error_info = f"❌  严重错误: 处理任务 {task_info.get('_id', 'N/A')} 时发生未知异常: {str(e)}"
        print(error_info)
        failure_details[str(task_info.get('_id', 'N/A'))] = {
            "error_info": error_info,
            "error_level": ERROR_STATUS.CRITICAL
        }
        return failure_details, {}, {}, {}
    finally:
        if check_failure_details(failure_details):
            failed_count = task_info.get('failed_count', 0)
            task_info['failed_count'] = failed_count + 1
            task_info['status'] = TaskStatus.FAILED
        else:
            # task_info['status'] = TaskStatus.COMPLETED
            pass

        task_info['failure_details'] = str(failure_details)
        manager.upsert_tasks([task_info])

def gen_cover_path(final_output_path, video_info_dict, cover_text):
    """
    生成最终的封面路径
    :return:
    """
    available_cover_path_list = []
    for video_id, video_info in video_info_dict.items():
        meta_data = video_info.get('metadata')[0]
        is_duplicate = video_info.get('is_duplicate', False)
        if is_duplicate:
            continue
        abs_cover_path = meta_data.get('abs_cover_path', '')
        if is_valid_target_file_simple(abs_cover_path):
            available_cover_path_list.append(abs_cover_path)

    if not available_cover_path_list:
        output_dir = os.path.dirname(final_output_path)
        target_frame = get_frame_at_time_safe(final_output_path, "00:00")
        if target_frame is not None:
            image_filename = f"first_frame.jpg"
            image_save_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_save_path, target_frame)
            available_cover_path_list.append(image_save_path)

    # 随机选择一个封面
    base_cover_path = random.choice(available_cover_path_list)
    output_image_path = base_cover_path.replace(".jpg", "_enhanced.jpg")
    if is_valid_target_file_simple(output_image_path):
        return output_image_path
    create_enhanced_cover(
        input_image_path=base_cover_path,
        output_image_path=output_image_path,
        text_lines=[cover_text],
    )
    return output_image_path


def build_bilibili_params(video_path, best_script, user_config, userName, video_info_dict, config):
    """
    生成投稿需要的参数
    :return:
    """
    upload_info = best_script.get('upload_info', {})


    title = best_script.get("title", "欢迎来看我的视频！")
    if len(title) > 80:
        title = title[:70]
        print(f"⚠️ 标题过长，已截断为：{title}")

    description_json = upload_info.get("introduction", {})
    target_keys = ["core_highlight", "value_promise", "interaction_guide", "supplement_info"]
    description = "\n".join(str(description_json[k]) for k in target_keys if k in description_json)



    tags = upload_info.get('tags', [])
    video_recommend_user_list = user_config.get('video_recommend_user_list', [])
    fun_user_list = user_config.get('fun_user_list', [])
    if userName in video_recommend_user_list:
        tags.insert(0, "B站好片有奖种草")
    if userName in fun_user_list:
        tags.insert(0, "娱乐盘点")
    tags = list(set(tags))
    tags = [tag for tag in tags if len(tag) <= 18]
    tags = tags[:12]
    tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)


    dynamic = upload_info.get("introduction", {}).get("interaction_guide", "希望大家喜欢")

    cover_text = best_script.get("cover_text", "")
    cover_path = gen_cover_path(video_path, video_info_dict, cover_text)

    human_type2 = upload_info.get("category_id", 1002)

    topic_id = upload_info.get("topic_id", 1105274)
    topic_detail = {
        "from_topic_id": topic_id,
        "from_source": "arc.web.recommend",
        "topic_name": "骑行去追夏天的风",
    }



    upload_params = {
        "title": title,
        "description": description,
        "tags": tags_str,
        "dynamic": dynamic,
        "cover_path": cover_path,
        "video_path": video_path,
        "sessdata": config[0],
        "bili_jct": config[1],
        "human_type2": human_type2,
        "topic_detail": topic_detail,
        "topic_id": topic_id,
    }
    return upload_params


def build_user_config():
    base_config_map = init_config()

    for uid, detail_info in base_config_map.items():
        name = detail_info.get("name", f"user_{uid}")
        sessdata = detail_info.get("SESSDATA", f"SESSDATA")
        bili_jct = detail_info.get("BILI_JCT", f"user_{uid}")
        total_cookie = detail_info.get("total_cookie", f"user_{uid}")
        # 判断total_cookie是否和之前的不一样，如果不一样则更新
        before_total_cookie = config_map.get(name, (None, None, None))[2]
        if before_total_cookie != total_cookie:
            print(f"🔄 检测到用户 {name} 的 total_cookie 发生变化，已更新。")
            # 如果name在error_user_map中，删除对应的错误记录
            if name in error_user_map:
                del error_user_map[name]

        config_map[name] = (sessdata, bili_jct, total_cookie)
    return config_map

def statistic_tasks_with_video(tasks_to_upload_list):
    """
    统计已有的视频的任务，并且排序
    :param tasks_to_upload:
    :return:
    """
    existing_video_tasks = []
    tobe_upload_video_info = {}
    not_existing_video_tasks = []
    for task_info in tasks_to_upload_list:
        task_path_info = build_task_video_paths(task_info)
        final_output_path = task_path_info['final_output_path']
        if is_valid_target_file_simple(final_output_path):
            existing_video_tasks.append(task_info)
            user_name = task_info.get('userName')
            if user_name not in tobe_upload_video_info:
                tobe_upload_video_info[user_name] = 0
            tobe_upload_video_info[user_name] += 1
        else:
            not_existing_video_tasks.append(task_info)

    # 将tobe_upload_video_info变成字符串，也就是 username: count 然后拼接一个长的字符串
    tobe_upload_video_info_str = ", ".join([f"{k}: {v}" for k, v in tobe_upload_video_info.items()])

    print(f"总共 {len(tasks_to_upload_list)} 个待投稿任务，其中已有视频 {len(existing_video_tasks)} 个，未生成视频 {len(not_existing_video_tasks)}  已有视频的分布情况：{tobe_upload_video_info_str}")
    return existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info


def upload_worker(
        upload_params: Dict[str, Any],
        task_info,
        files_to_cleanup: List[Optional[str]],
        userName: str,
        manager
) -> None:
    """
    后台上传任务（在各自账号的单线程 executor 中运行，保证同账号串行）；
    完整地执行上传重试、结果处理、metadata 更新、临时文件清理与日志持久化。
    """
    global error_user_map
    video_id_list = task_info.get("video_id_list", [])
    max_retries = 3
    result: Optional[Dict[str, Any]] = None
    t_upload = time.time()

    # 上传重试
    for attempt in range(1, max_retries + 1):
        try:
            result = upload_to_bilibili(**upload_params)
            break
        except Exception as e:
            print(
                f"❌ 上传接口异常 (第 {attempt} 次重试) user={userName} video_id_list={video_id_list}：{e} {upload_params}"
            )
            if attempt < max_retries:
                time.sleep(60)
            else:
                print("已达最大重试次数，放弃本次上传（后台）。")

    # 上传成功
    if result and isinstance(result, dict) and result.get("aid") and result.get("bvid"):
        try:
            print(
                f"🎉 后台投稿成功！AID={result['aid']}  BVID={result['bvid']} video_id_list={video_id_list} "
                f"user={userName} 上传耗时 {time.time() - t_upload:.2f} 秒。"
            )
            # 删除临时文件（上传成功后清理）
            for p in files_to_cleanup or []:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f"⚠️ 清理文件 {p} 失败：{e}")

        except Exception as e:
            print(f"⚠️ 后台上传后处理异常：{e}")

        task_info["upload_params"] = upload_params
        task_info["upload_result"] = result
        task_info["uploaded_time"] = datetime.now()
        task_info["status"] = TaskStatus.UPLOADED
        manager.upsert_tasks([task_info])

    else:
        # 上传失败：记录 error_user_map，并把错误信息写到 upload_log
        try:
            err = result.get("message", str(result)) if isinstance(result, dict) else str(result)
        except Exception:
            err = str(result)
        error_user_map[userName] = err or "未知错误"
        print(f"❌ 后台投稿失败 user={userName} video_id_list={video_id_list}：{err}")


def print_simple_stats(statistic_data):
    if not statistic_data:
        print("暂无统计数据")
        return
    header = (
        "用户名            "  # 6个空格
        "  今日已投 "  # 2个空格
        "  平台存量 "  # 2个空格
        "  准备就绪 "  # 2个空格
        "  今日待传   "  # 2个空格
        "  明日待传 "  # 2个空格
        "         最近上传时间"  # 9个空格
    )

    separator = "-" * 83

    print(separator)
    print(header)
    print(separator)

    sorted_users = sorted(
        statistic_data.keys(),
        key=lambda u: statistic_data[u].get('today_upload_count', 0),
        reverse=True
    )

    for user in sorted_users:
        info = statistic_data[user]
        time_str = str(info.get('latest_upload_time') or '-')

        # 这里保持全是 ASCII 字符（英文/数字），所以 Python 的宽度计算是准确的
        row = (
            f"{user:<12}"  # 对应 "用户名      "
            f"{info.get('today_upload_count', 0):>10}"  # 对应 "  今日已投"
            f"{info.get('platform_upload_count', 0):>10}"  # 对应 "  平台存量"
            f"{info.get('tobe_upload_count', 0):>10}"  # 对应 "  准备就绪"
            f"{info.get('today_process', 0):>10}"  # 对应 "  今日待传"
            f"{info.get('tomorrow_process', 0):>10}"  # 对应 "  明日待传"
            f"{time_str:>21}"  # 对应 "         最近上传时间"
        )
        print(row)

    print(separator)

def gen_all_statistic_info(already_upload_users, user_upload_info, need_process_tasks_list, tobe_upload_video_info):
    """
    没一轮投稿后进行的统计，理论上要统计每个账号的信息 包括 今日投稿数量today_upload_count 平台实际数量platform_upload_count 已准备好的数据tobe_upload_count 今日待上传数量today_process 明天待上传数量tomorrow_process 最近上传时间latest_upload_time
    :return:
    """
    user_statistic_info = user_upload_info
    for this_turn_user_name in already_upload_users:
        user_info = user_statistic_info.get(this_turn_user_name, {})
        if "today_upload_count" not in user_info:
            user_info['today_upload_count'] = 0
        user_info['today_upload_count'] += 1

    for user_name, tobe_count in tobe_upload_video_info.items():
        if 'tobe_upload_count' not in user_statistic_info.get(user_name, {}):
            if user_name not in user_statistic_info:
                user_statistic_info[user_name] = {}
            user_statistic_info[user_name]['tobe_upload_count'] = 0
        user_statistic_info[user_name]['tobe_upload_count'] += tobe_count

    for task_info in need_process_tasks_list:
        user_name = task_info.get('userName')
        creation_guidance_info = task_info.get('creation_guidance_info', {})
        schedule_date = creation_guidance_info.get('schedule_date', '2026-01-05')
        is_future = datetime.strptime(schedule_date, '%Y-%m-%d').date() > datetime.now().date()
        if is_future:
            if 'tomorrow_process' not in user_statistic_info.get(user_name, {}):
                if user_name not in user_statistic_info:
                    user_statistic_info[user_name] = {}
                user_statistic_info[user_name]['tomorrow_process'] = 0
            user_statistic_info[user_name]['tomorrow_process'] += 1
        else:
            if 'today_process' not in user_statistic_info.get(user_name, {}):
                if user_name not in user_statistic_info:
                    user_statistic_info[user_name] = {}
                user_statistic_info[user_name]['today_process'] = 0
            user_statistic_info[user_name]['today_process'] += 1

    # 将user_statistic_info中的latest_upload_time转换成字符串格式
    for user_name, info in user_statistic_info.items():
        latest_time = info.get('latest_upload_time')
        if isinstance(latest_time, datetime):
            info['latest_upload_time'] = latest_time.strftime('%Y-%m-%d %H:%M:%S')

    save_json(USER_STATISTIC_INFO_PATH, user_statistic_info)
    # 规范的打印出来这个统计信息

    print_simple_stats(user_statistic_info)
    return user_statistic_info


def process_idle_tasks(
        tasks: list,
        tobe_upload_video_info: dict,
        futures: List[concurrent.futures.Future],
        config_map: dict,
        user_config: dict,
        manager
):
    """
    利用上传等待的空闲时间，处理未生成视频的任务
    """
    total_candidates = len(tasks)
    start_time = time.time()
    print(
        f"开始处理 {total_candidates} 个未生成视频的任务 利用空闲时间...当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    count = 0
    for task_info in tasks:
        count += 1
        user_name = task_info.get('userName')
        # 注意：修正了原代码f-string中引号嵌套的潜在兼容性问题
        print(f"处理用户 {user_name} 的任务{task_info.get('video_id_list', [])}...已有的数量{tobe_upload_video_info.get(user_name)} 进度 {count}/{total_candidates} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ")

        # 生成视频核心逻辑
        gen_video(task_info, config_map, user_config, manager)

        # 计算时间与状态
        processing_duration = time.time() - start_time
        pending_uploads_count = sum(1 for f in futures if not f.done())
        is_uploading = pending_uploads_count > 0

        # 更新统计信息 (引用传递，会同步修改外部字典)
        if user_name not in tobe_upload_video_info:
            tobe_upload_video_info[user_name] = 0
        tobe_upload_video_info[user_name] += 1

        print(
            f"处理完成，处理用户 {user_name} 的任务{task_info.get('video_id_list', [])} 耗时 {processing_duration:.2f} 秒，当前待上传任务数 {pending_uploads_count}，是否有上传任务正在进行: {is_uploading} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ")

        # 核心退出逻辑：如果处理时间超过200秒且没有后台上传在进行，则结束“压榨算力”
        if processing_duration > 200 and not is_uploading:
            print(
                f"🎉 【有效处理完成】 任务 '{task_info.get('video_id_list', [])}' 耗时 {processing_duration:.2f} 秒. 且无后台投稿。 进度 {count}/{total_candidates} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ")
            print("   - 目标达成，备用处理流程结束。")
            break

        if processing_duration > 200:
            print(
                f"   ⚡ [算力压榨] 耗时已超 {processing_duration:.2f}s，利用 {pending_uploads_count} 个后台上传间隙继续处理...")
        else:
            print(f"   ⚡ 未进行实际的处理 处理太快了 {processing_duration:.2f}s，继续处理...")


def auto_upload(manager):
    """
    进行单次循环的投稿
    :return:
    """
    already_upload_users = []
    current_time = datetime.now()
    config_map = build_user_config()
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    start_time = time.time()

    # 1. 获取并统计任务
    tasks_to_upload = manager.find_tasks_by_status([TaskStatus.PLAN_GENERATED])
    print(f"找到 {len(tasks_to_upload)} 个待投稿任务，开始处理...耗时 {time.time() - start_time:.2f} 秒")
    existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info = statistic_tasks_with_video(tasks_to_upload)

    futures: List[concurrent.futures.Future] = []

    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # 查询今日已投稿的任务
    uploaded_tasks_today = manager.find_tasks_after_time_with_status(today_midnight, [TaskStatus.UPLOADED])
    user_upload_info = gen_user_upload_info(uploaded_tasks_today)

    sort_tasks_to_upload, sort_existing_video_tasks, sort_not_existing_video_tasks = sort_tasks(existing_video_tasks,
                                                                                                not_existing_video_tasks,
                                                                                                tobe_upload_video_info)

    # 2. 循环提交上传任务
    for task_info in sort_tasks_to_upload:
        check_result = check_need_upload(task_info, user_upload_info, current_time, already_upload_users, user_config,
                                         config_map)
        user_name = task_info.get('userName')

        if not check_result:
            continue

        failure_details, video_info_dict, chosen_script, upload_params = gen_video(task_info, config_map, user_config,
                                                                                   manager)
        print(upload_params)
        if not chosen_script:
            print(f"❌ 生成视频失败，跳过上传 {task_info.get('video_id_list', [])} 用户 {user_name} ")
            continue

        all_files_to_cleanup = []
        account_executor = account_executors[user_name]
        future = account_executor.submit(
            upload_worker,
            upload_params,
            task_info,
            all_files_to_cleanup,
            user_name,
            manager,
        )
        futures.append(future)
        already_upload_users.append(user_name)

    # 3. 【重构点】利用上传间隙，处理未生成视频的任务
    process_idle_tasks(
        tasks=sort_not_existing_video_tasks,
        tobe_upload_video_info=tobe_upload_video_info,
        futures=futures,
        config_map=config_map,
        user_config=user_config,
        manager=manager
    )

    # 4. 收尾与统计
    print(
        f"等待所有等待后台上传完成... 本轮投稿数量 {len(already_upload_users)}  用户{already_upload_users}  当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')} {error_user_map}")
    need_process_tasks = query_need_process_tasks()

    # 注意：tobe_upload_video_info 在 process_idle_tasks 中可能被修改，这里使用修改后的值，逻辑正确
    gen_all_statistic_info(already_upload_users, user_upload_info, need_process_tasks, tobe_upload_video_info)
    concurrent.futures.wait(futures, timeout=None)


if __name__ == "__main__":
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    while True:
        auto_upload(manager)
        print(f"本轮投稿处理完成，等待下一轮...当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(60)  # 每分钟运行一次
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading  # 需要确保头部导入了这个
import cv2
from rich import box
from rich.console import Console
from rich.table import Table

from application.process_video import process_single_task, query_need_process_tasks
from application.video_common_config import TaskStatus, ERROR_STATUS, check_failure_details, build_task_video_paths, \
    SINGLE_DAY_UPLOAD_COUNT, SINGLE_UPLOAD_COUNT, USER_STATISTIC_INFO_PATH, build_video_paths, ALL_BILIBILI_EMOTE_PATH, \
    USER_BVID_FILE
from utils.bilibili.bilibili_uploader import upload_to_bilibili
from utils.common_utils import read_json, is_valid_target_file_simple, init_config, save_json, get_top_comments, \
    extract_guides, format_bilibili_emote, parse_and_group_danmaku, filter_danmu
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from utils.video_utils import get_frame_at_time_safe, create_enhanced_cover, probe_duration

config_map = {}
error_user_map = {}

account_executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = defaultdict(
    lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1)
)

def gen_user_upload_info(uploaded_tasks_today):
    """
    通过今日投稿的任务生成用户投稿的信息
    """
    remote_upload_dict = {}
    today_start = datetime.combine(datetime.today(), datetime.min.time()).timestamp()
    # 检查用户今日上传数量（本地 + 平台）
    try:
        bvid_file_data = read_json(USER_BVID_FILE)
    except Exception as e:
        print(f"❌ 读取 {USER_BVID_FILE} 失败：{e}")
        bvid_file_data = {}
    for userName, user_videos in bvid_file_data.items():
        recent_videos = [v for v in user_videos if v.get("created") and v["created"] >= today_start]
        remote_upload_count = len(recent_videos)
        remote_upload_dict[userName] = remote_upload_count

    # 定义默认值结构
    user_upload_info = defaultdict(lambda: {'today_upload_count': 0, 'platform_upload_count': 0, 'latest_upload_time': datetime.min})

    for task in uploaded_tasks_today:
        user_name = task['userName']
        upload_time = task['uploaded_time']
        play_count = task.get('play_comment_info_list', None)

        # 更新数据
        info = user_upload_info[user_name]
        info['today_upload_count'] += 1
        if play_count:
            info['platform_upload_count'] += 1
        # 比较并保留较大的时间
        if upload_time > info['latest_upload_time']:
            info['latest_upload_time'] = upload_time
    for user_name, remote_count in remote_upload_dict.items():
        info = user_upload_info[user_name]
        info['platform_upload_count_local'] = remote_count

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

        count = user_info_map.get(user_name, 0)
        guidance_info = task.get('creation_guidance_info', {})

        schedule_date_str = guidance_info.get('schedule_date', '')

        # 默认是小的时间
        create_time = task.get('create_time', datetime.min)

        return (count, schedule_date_str, create_time)

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
        elif "音乐" in category_name_list_str or "动物" in category_name_list_str or "搞笑" in category_name_list_str or "小剧场" in category_name_list_str or "资讯" in category_name_list_str or "旅游出行" in category_name_list_str or "趣味" in category_name_list_str or "娱乐" in category_name_list_str or "新闻" in category_name_list_str or "影视" in category_name_list_str or "情感" in category_name_list_str or "知识" in category_name_list_str:
            video_type = "fun"
        task_info['video_type'] = video_type
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
    log_pre = f"{task_info.get('video_id_list', [])} {task_info.get('_id', '')} {creation_guidance_info} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    global error_user_map
    bvid = task_info.get('bvid', '')
    if bvid:
        print(f"❌❌❌ 任务已有 bvid {bvid}，跳过 {log_pre}")
        return False

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
    if len(already_upload_users) >= 5:
        print(f"本轮已投稿用户过多，跳过 {log_pre}")
        return False

    right_now_user_list = user_config.get('right_now_user_list', [])
    if user_name not in right_now_user_list:
        if not (5 <= datetime.now().hour < 24):
            cooldown_reason = "当前时间不在允许的上传时间段（5点-24点）内。"
            print(f"{user_name} 因为 {cooldown_reason} 跳过 {log_pre}")
            return False

        need_waite_minutes = get_wait_minutes()
        latest_upload_time = user_upload_info.get(user_name, {}).get('latest_upload_time', datetime.min)
        # 计算和上次投稿的差值分数数
        time_diff = (current_time - latest_upload_time).total_seconds() / 60

        if time_diff < need_waite_minutes:
            print(f"{user_name} 距离上次投稿仅 {time_diff:.2f} 分钟，一共需等待 {need_waite_minutes} 分钟，跳过 {log_pre}")
            return False

    platform_upload_count = user_upload_info.get(user_name, {}).get('platform_upload_count', 0)
    today_upload_count = user_upload_info.get(user_name, {}).get('today_upload_count', 0)
    platform_upload_count_local = user_upload_info.get(user_name, {}).get('platform_upload_count_local', 0)
    if platform_upload_count >= max_count or today_upload_count > 25 or platform_upload_count_local >= max_count:
        print(f"{user_name}  今天投稿 {today_upload_count} 实际数量{platform_upload_count} 今日投稿次数已达上限 {max_count} 次，跳过 {log_pre}")
        return False


    return True


def gen_video(task_info, config_map, user_config, manager):
    failure_details = {}
    try:
        failure_details, video_info_dict, chosen_script = process_single_task(task_info, manager, gen_video=True)
        user_name = task_info.get('userName')
        id = task_info.get('_id', '__')
        all_task_video_path_info = build_task_video_paths(task_info)
        final_output_path = all_task_video_path_info['final_output_path']
        account_config = config_map.get(user_name)
        upload_params = build_bilibili_params(final_output_path, chosen_script, user_config, user_name, video_info_dict, account_config, id)

        return failure_details, video_info_dict, chosen_script, upload_params
    except Exception as e:
        traceback.print_exc()
        error_info = f"❌  严重错误: 处理任务 {task_info.get('_id', 'N/A')} 制作视频时发生未知异常: {str(e)}"
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

def gen_cover_path(final_output_path, video_info_dict, cover_text, id):
    """
    生成最终的封面路径
    :return:
    """
    available_cover_path_list = []
    try:
        for video_id, video_info in video_info_dict.items():
            if not video_info.get('metadata'):
                continue
            meta_data = video_info.get('metadata')[0]
            is_duplicate = video_info.get('is_duplicate', False)
            if is_duplicate:
                continue
            abs_cover_path = meta_data.get('abs_cover_path', '')
            if is_valid_target_file_simple(abs_cover_path):
                available_cover_path_list.append(abs_cover_path)
    except Exception as e:
        traceback.print_exc()
        print(f"⚠️ 生成封面时发生错误：{e} {id}")

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
    output_image_path = base_cover_path.replace(".jpg", f"_{id}_enhanced.jpg")
    if is_valid_target_file_simple(output_image_path):
        return output_image_path
    create_enhanced_cover(
        input_image_path=base_cover_path,
        output_image_path=output_image_path,
        text_lines=[cover_text],
    )
    return output_image_path


def build_bilibili_params(video_path, best_script, user_config, userName, video_info_dict, config, id):
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

    tags = list(upload_info.get('tags', []))
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
    cover_path = gen_cover_path(video_path, video_info_dict, cover_text, id)

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

def statistic_tasks_with_video(tasks_to_upload_list, allowed_user_name_list):
    """
    统计已有的视频的任务，并且排序
    :param tasks_to_upload:
    :return:
    """
    existing_video_tasks = []
    tobe_upload_video_info = {}
    not_existing_video_tasks = []
    not_existing_video_info = {}
    all_video_info = {}
    for task_info in tasks_to_upload_list:
        task_path_info = build_task_video_paths(task_info)
        final_output_path = task_path_info['final_output_path']
        user_name = task_info.get('userName')
        if user_name not in allowed_user_name_list:
            continue

        if is_valid_target_file_simple(final_output_path):
            existing_video_tasks.append(task_info)
            if user_name not in tobe_upload_video_info:
                tobe_upload_video_info[user_name] = 0
            tobe_upload_video_info[user_name] += 1
        else:
            not_existing_video_tasks.append(task_info)
            if user_name not in not_existing_video_info:
                not_existing_video_info[user_name] = 0
            not_existing_video_info[user_name] += 1
        if user_name not in all_video_info:
            all_video_info[user_name] = 0
        all_video_info[user_name] += 1

    # 将tobe_upload_video_info变成字符串，也就是 username: count 然后拼接一个长的字符串
    tobe_upload_video_info_str = ", ".join([f"{k}: {v}" for k, v in tobe_upload_video_info.items()])
    not_existing_video_info_str = ", ".join([f"{k}: {v}" for k, v in not_existing_video_info.items()])
    all_video_info_str = ", ".join([f"{k}: {v}" for k, v in all_video_info.items()])

    print(f"总共 {len(tasks_to_upload_list)} 个待投稿任务，其中已有视频 {len(existing_video_tasks)} 个，未生成视频 {len(not_existing_video_tasks)}  已有视频的分布情况：{tobe_upload_video_info_str} 未生成视频的分布情况：{not_existing_video_info_str} 全部任务的分布情况：{all_video_info_str} ")
    return existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info


def upload_worker(
        upload_params: Dict[str, Any],
        task_info,
        files_to_cleanup: List[Optional[str]],
        userName: str,
        manager,
        video_info_dict
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
    print(f"🚀 准备为{userName} 投稿 video_id_list={video_id_list} 上传参数：{upload_params}")
    # 上传重试
    for attempt in range(1, max_retries + 1):
        try:
            result = upload_to_bilibili(**upload_params)
            break
        except Exception as e:
            traceback.print_exc()

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
                f"user={userName} 上传耗时 {time.time() - t_upload:.2f} 秒。 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} 上传参数：{upload_params}")
            # 删除临时文件（上传成功后清理）
            for p in files_to_cleanup or []:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f"⚠️ 清理文件 {p} 失败：{e}")

        except Exception as e:
            traceback.print_exc()

            print(f"⚠️ 后台上传后处理异常：{e}")

        try:
            video_path = upload_params.get("video_path", "")
            video_duration = probe_duration(video_path)
        except Exception as e:
            traceback.print_exc()
            video_duration = 120

        hudong_info = build_hudong_info(task_info, video_info_dict, video_duration)

        task_info["upload_params"] = upload_params
        task_info["upload_result"] = result
        task_info["bvid"] = result["bvid"]
        task_info["video_duration"] = video_duration
        task_info["hudong_info"] = hudong_info
        task_info["uploaded_time"] = datetime.now()
        task_info["status"] = TaskStatus.UPLOADED
        manager.upsert_tasks([task_info])

    else:
        # 上传失败：记录 error_user_map，并把错误信息写到 upload_log
        try:
            err = result.get("message", str(result)) if isinstance(result, dict) else str(result)
        except Exception:
            traceback.print_exc()

            err = str(result)
        error_user_map[userName] = err or "未知错误"
        print(f"❌ 后台投稿失败 user={userName} video_id_list={video_id_list}：{err}")


def print_simple_stats(statistic_data):
    if not statistic_data:
        print("暂无统计数据")
        return
    header = (
        "用户名            "  # 12
        "  今日已投 "  # 10
        "  平台存量 "  # 10
        "  准备就绪 "  # 10
        "  今日待传   "  # 10
        "  明日待传 "  # 10
        "  上次距今分钟"  # 14 (新增列)
        "         最近上传时间"  # 21
    )

    # 原长度83 + 新增列宽14 = 97
    separator = "-" * 97

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
        raw_time = info.get('latest_upload_time')
        time_str = str(raw_time or '-')

        # --- 新增计算逻辑 ---
        minutes_str = '-'
        if raw_time and isinstance(raw_time, str) and raw_time != '-':
            try:
                # 解析时间字符串 '2026-01-16 18:18:14'
                dt_obj = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
                # 计算时间差
                delta = datetime.now() - dt_obj
                # 转换为分钟 (总秒数 // 60)
                minutes_str = str(int(delta.total_seconds() // 60))
            except ValueError:
                pass  # 解析失败则保持为 '-'
        # ------------------

        # 这里保持全是 ASCII 字符（英文/数字），所以 Python 的宽度计算是准确的
        row = (
            f"{user:<12}"  # 对应 "用户名      "
            f"{info.get('today_upload_count', 0):>10}"  # 对应 "  今日已投"
            f"{info.get('platform_upload_count_local', 0):>10}"  # 对应 "  平台存量"
            f"{info.get('tobe_upload_count', 0):>10}"  # 对应 "  准备就绪"
            f"{info.get('today_process', 0):>10}"  # 对应 "  今日待传"
            f"{info.get('tomorrow_process', 0):>10}"  # 对应 "  明日待传"
            f"{minutes_str:>14}"  # 对应 "  上次距今分钟" (新增)
            f"{time_str:>21}"  # 对应 "         最近上传时间"
        )
        print(row)

    print(separator)

def gen_all_statistic_info(already_upload_users, user_upload_info, need_process_tasks_list, tobe_upload_video_info, allowed_user_name_list):
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

    # 只保留allowed_user_name_list中的用户
    user_statistic_info = {k: v for k, v in user_statistic_info.items() if k in allowed_user_name_list}

    print_simple_stats(user_statistic_info)
    return user_statistic_info


def process_idle_tasks(
        tasks: list,
        tobe_upload_video_info: dict,
        futures: List[concurrent.futures.Future],
        config_map: dict,
        user_config: dict,
        manager,
        processed_task_ids
):
    """
    利用上传等待的空闲时间，并行处理未生成视频的任务 (2个并发)，同时保持日志和功能与原版高度一致。
    """
    # 1. 预先筛选任务，并补上缺失的“跳过”日志
    candidate_tasks = []
    original_total = len(tasks)
    print(
        f"开始处理 {original_total} 个未生成视频的任务 利用空闲时间...当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    used_video_id_list = []
    for task_info in tasks:
        bvid = task_info.get('bvid', '')
        video_id_list = task_info.get('video_id_list', [])
        # 判断video_id_list是否有在used_video_id_list中
        if any(vid in used_video_id_list for vid in video_id_list):
            print(
                f" 任务的 video_id_list {video_id_list} 中有已使用的视频ID，跳过...当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ")
            continue

        status = task_info.get('status')
        if bvid or status == TaskStatus.UPLOADED or str(task_info.get('_id')) in processed_task_ids:
            print(
                f"任务已有 bvid {bvid} 或状态为已上传，跳过 {task_info.get('video_id_list', [])}...当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ")
            continue
        used_video_id_list.extend(video_id_list)
        candidate_tasks.append(task_info)

    total_candidates = len(candidate_tasks)
    if total_candidates == 0:
        print("没有需要处理的空闲任务。")
        return

    # 【新增】定义一个停止事件，用于控制 wrapper 里的“1秒后检查”
    stop_event = threading.Event()

    # 使用原子计数器（线程安全）来记录已处理的数量，用于日志打印
    completed_counter = threading.atomic_count = 0
    lock = threading.Lock()

    # 包装 gen_video 以在线程内部打印“开始处理”日志
    def gen_video_wrapper(task_info, index, total, start_time_ref, *args):
        """
        :param index: 当前任务在候选列表中的索引（1-based）
        :param total: 总任务数
        :param start_time_ref: 这一批次任务开始的时间戳
        """
        # 【修改】核心改动：先等待1秒，给主线程留出cancel的时间窗口
        time.sleep(1)

        # 【修改】睡醒后，检查是否收到了停止信号
        if stop_event.is_set():
            return None  # 如果被叫停，直接返回None，不执行后续逻辑

        user_name = task_info.get('userName')

        # 1. 打印“开始处理”日志（已优化：加入进度信息）
        print(
            f"🚀 [并行] 开始处理用户 {user_name} 的任务 {task_info.get('video_id_list', [])} "
            f"(进度: {index}/{total}) 已生成数量: {tobe_upload_video_info.get(user_name, 0)} "
            f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        )

        # 调用真正的视频生成函数
        gen_video(task_info, *args)

        # 2. 打印“处理完成”日志 (按你的要求移入内部，确保必打)
        # 获取最新的状态信息
        current_duration = time.time() - start_time_ref
        pending_uploads = sum(1 for f in futures if not f.done())
        is_uploading_now = pending_uploads > 0

        if not is_uploading_now:
            upload_status_str = "且无后台投稿"
        else:
            upload_status_str = f"后台有 {pending_uploads} 个投稿进行中"

        print(
            f"🎉 【有效处理完成】 任务 '{task_info.get('video_id_list', [])}' "
            f"耗时 {current_duration:.2f} 秒. {upload_status_str}。 "
            f"进度 {index}/{total} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
        )

        # 返回结果，以便主线程可以识别任务
        return task_info

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_task = {}

        # 提交所有候选任务
        # 使用 enumerate 获取序号，方便传入 wrapper
        for idx, task_info in enumerate(candidate_tasks, 1):
            # 这是一个功能变更，但对于并发是好的实践。提前标记，防止重复。
            processed_task_ids.add(str(task_info.get('_id')))

            # 注意：这里把 idx, total_candidates, start_time 传进去了
            future = executor.submit(gen_video_wrapper, task_info, idx, total_candidates, start_time, config_map, user_config, manager)
            future_to_task[future] = task_info

        completed_count = 0
        try:
            for future in as_completed(future_to_task):
                completed_count += 1
                try:
                    # 获取执行结果。如果 gen_video 内部报错，这里会抛出
                    completed_task_info = future.result()

                    # 【修改】如果因为stop_event返回了None，说明该任务被跳过了，直接处理下一个
                    if completed_task_info is None:
                        continue

                    user_name = completed_task_info.get('userName')
                except Exception as e:
                    task_info_on_error = future_to_task[future]
                    print(f"❌ 处理任务 {task_info_on_error.get('video_id_list', [])} 发生异常: {e}")
                    traceback.print_exc()
                    continue  # 继续处理下一个已完成的任务

                # 更新统计信息 (在主线程中更新，线程安全)
                if user_name not in tobe_upload_video_info:
                    tobe_upload_video_info[user_name] = 0
                tobe_upload_video_info[user_name] += 1

                # 计算时间与状态
                processing_duration = time.time() - start_time
                pending_uploads_count = sum(1 for f in futures if not f.done())
                is_uploading = pending_uploads_count > 0

                # 核心退出逻辑
                if processing_duration > 200 and not is_uploading:
                    # 补上与原版一致的、包含任务信息的退出日志
                    print("   - 目标达成，备用处理流程结束。将取消未开始的任务。")

                    # 【修改】设置停止标志，通知那些正在 sleep(1) 的线程不要继续了
                    stop_event.set()

                    # 主动取消队列中还未开始的任务
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
                    break  # 跳出 as_completed 循环

                # 补上与原版一致的“算力压榨”日志逻辑
                if processing_duration > 200:
                    print(
                        f"   ⚡ [算力压榨] 耗时已超 {processing_duration:.2f}s，利用 {pending_uploads_count} 个后台上传间隙继续处理...")
                else:
                    # 补上缺失的 "处理太快" 日志
                    print(f"   ⚡ 未进行实际的处理 处理太快了 {processing_duration:.2f}s，继续处理...")
        finally:
            # 确保即使发生意外，也能尝试关闭执行器
            # with 语句会自动处理 shutdown，但这里的消息可以提供更好的日志
            print("所有任务处理循环结束，等待正在运行的线程完成...")


def gen_all_files_to_cleanup(task_info):
    """
    梳理出投稿后需要删除的文件
    扫描 file_path_list 目录及其子目录，收集不在 exclude_file_list 中的所有文件路径
    :param task_info: 任务信息字典
    :return: 需要清理的文件路径列表 clean_files
    """
    # 1. 初始化排除列表和目录列表
    exclude_file_list = ['merged_timestamps.json', 'subtitle_box.json']
    file_path_list = []

    tasks_to_process = query_need_process_tasks()
    # 统计所有的 video_id
    video_ids_in_process = set()
    for task in tasks_to_process:
        video_ids_in_process.update(task.get('video_id_list', []))
    skip_id_list = []
    # 2. 获取视频相关的路径和排除文件
    video_id_list = task_info.get('video_id_list', [])
    for video_id in video_id_list:
        if video_id in video_ids_in_process:
            print(f"跳过正在处理的 video_id {video_id} 的文件清理")
            skip_id_list.append(video_id)
            continue
        # 假设 build_video_paths 是外部定义的函数
        video_path_info = build_video_paths(video_id)
        origin_video_path = video_path_info.get('origin_video_path')

        if origin_video_path:
            video_file_name = os.path.basename(origin_video_path)
            base_dir = os.path.dirname(origin_video_path)
            file_path_list.append(base_dir)
            exclude_file_list.append(video_file_name)
    task_path_info = build_task_video_paths(task_info)
    final_output_path = task_path_info.get('final_output_path')

    if final_output_path:
        final_output_file_name = os.path.basename(final_output_path)
        exclude_file_list.append(final_output_file_name)
        base_dir = os.path.dirname(final_output_path)
        file_path_list.append(base_dir)

    clean_files = []
    keep_files = []

    # 转换为集合以提高查找性能 (O(1) 复杂度)
    exclude_files_set = set(exclude_file_list)
    # 目录去重，避免重复扫描同一个文件夹
    unique_dirs = set(file_path_list)

    for dir_path in unique_dirs:
        # 检查目录是否存在，防止报错
        if not os.path.exists(dir_path):
            continue

        # os.walk 会递归遍历 dir_path 下的所有子目录
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if file_name not in exclude_files_set:
                    full_path = os.path.join(root, file_name)
                    clean_files.append(str(full_path))
                else:
                    keep_files.append(file_name)
    print(f"🧹 文件清理 清理所有清理 所有文件所有 {task_info.get('userName', 'N/A')} {task_info.get('video_id_list', 'N/A')}。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 清理文件列表：{len(clean_files)}，保留文件列表：{len(keep_files)} 已跳过的id {skip_id_list}")
    return clean_files, keep_files



def build_hudong_info(task_info, video_info_dict, video_duration):
    """
    生成该视频的互动信息
    :param task_info:
    :param video_info_dict:
    :param all_emote_list:
    :return:
    """
    try:
        all_emote_list = read_json(ALL_BILIBILI_EMOTE_PATH)

        hudong_info = {}
        upload_info = task_info.get('upload_info', {})
        total_seconds = int(video_duration)
        interaction_prompts, supplementary_notes = extract_guides(upload_info)
        if len(interaction_prompts) == 0:
            interaction_prompts = ["刷到这个视频的你，希望今天能有个好心情呀~",
                                   "叮！你收到一份来自UP主的好运，请注意查收哦！"]
        if len(supplementary_notes) == 0:
            supplementary_notes = ["感谢你愿意花时间看到最后，愿这份好运能一直陪着你。",
                                   "如果觉得视频还不错，不妨点个赞，把这份快乐和祝福一起带走吧！"]
        interaction_danmu_list = [{'建议时间戳': 1, '推荐弹幕内容': interaction_prompts}]
        supplementary_notes_list = [{'建议时间戳': total_seconds - 8, '推荐弹幕内容': supplementary_notes}]

        owner_danmu_list = []  # 用于存储UP主的弹幕
        owner_danmu_list.extend(interaction_danmu_list)  # 将互动引导弹幕添加到UP主弹幕列表中
        owner_danmu_list.extend(supplementary_notes_list)  # 将补充信息弹幕添加到UP主弹幕列表中

        comment_list = get_top_comments(video_info_dict, need_image=True)
        format_bilibili_emote(comment_list, all_emote_list)

        all_danmu_list = []
        for video_id, video_info in video_info_dict.items():
            danmu_info = video_info.get('hudong_info', {})
            danmu_list = parse_and_group_danmaku(danmu_info)
            all_danmu_list.extend(danmu_list)

        danmu_list = filter_danmu(all_danmu_list, total_seconds)
        hudong_info['comment_list'] = comment_list
        hudong_info['owner_danmu'] = owner_danmu_list
        hudong_info["duration"] = total_seconds
        hudong_info['danmu_list'] = danmu_list
    except Exception as e:
        traceback.print_exc()
        print(f"⚠️ 生成互动信息失败：{e}")
        hudong_info = {}
    return hudong_info


def fix_user_name(tasks, user_config, manager):
    for task_info in tasks:
        check_type(task_info, user_config)
        video_type = task_info.get('video_type', '未知')
        if video_type == 'fun':
            task_info['userName'] = random.choice(["shun", "ping", "ping"])
        if video_type == 'game':
            task_info['userName'] = random.choice(["lin", "zhong", "qizhu", "mama", "hong"])
        if video_type == 'sport':
            task_info['userName'] = random.choice(["nana"])
    manager.upsert_tasks(tasks)



global_config_lock = threading.Lock()

def get_safe_configs():
    """
    线程安全地获取最新配置，防止读写冲突
    """
    with global_config_lock:
        config_m = build_user_config()
        user_conf = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
        return config_m, user_conf


def auto_upload(manager):
    """
    【投稿流】只负责把现成的视频投出去。
    单轮内通过 already_upload_users 防重，并在最后死等所有投稿完成。
    """
    # 1. 线程安全地获取配置
    config_map, user_config = get_safe_configs()

    already_upload_users = []
    current_time = datetime.now()
    allowed_user_name_list = list(config_map.keys())

    stop_flag = user_config.get('stop_flag', False)
    if stop_flag:
        print(f"检测到停止投稿开关已开启，暂停本轮投稿。 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    processed_task_ids = set()
    start_time = time.time()
    all_task = []
    exist_id_list = []
    filter_task_list = []
    tasks_to_process = query_need_process_tasks()

    # 获取并统计任务
    tasks_to_upload = manager.find_tasks_by_status([TaskStatus.PLAN_GENERATED, TaskStatus.TO_UPLOADED])
    all_task.extend(tasks_to_upload)
    all_task.extend(tasks_to_process)

    for task in all_task:
        id_str = str(task.get('_id'))
        if id_str in exist_id_list:
            continue
        user_name = task.get('userName')
        if user_name not in config_map:
            print(f"用户 {user_name} 未在配置中找到，跳过任务 {task.get('video_id_list', [])} ")
            continue
        exist_id_list.append(id_str)
        filter_task_list.append(task)

    existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info = statistic_tasks_with_video(tasks_to_upload, allowed_user_name_list)

    futures: List[concurrent.futures.Future] = []

    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    uploaded_tasks_today = manager.find_tasks_after_time_with_status(today_midnight, [TaskStatus.UPLOADED])
    user_upload_info = gen_user_upload_info(uploaded_tasks_today)

    sort_tasks_to_upload, sort_existing_video_tasks, sort_not_existing_video_tasks = sort_tasks(
        existing_video_tasks,
        not_existing_video_tasks,
        tobe_upload_video_info
    )

    # 2. 核心：遍历已有视频的任务，触发上传
    for task_info in sort_existing_video_tasks:
        # 使用 already_upload_users 完美限制单轮单用户的重复投稿
        check_result = check_need_upload(task_info, user_upload_info, current_time, already_upload_users, user_config, config_map)
        # 检测 状态 是否为 TaskStatus.TO_UPLOADED
        if task_info.get('status') != TaskStatus.TO_UPLOADED:
            continue
        if not check_result:
            continue

        user_name = task_info.get('userName')

        failure_details, video_info_dict, chosen_script, upload_params = gen_video(task_info, config_map, user_config, manager)
        if check_failure_details(failure_details):
            print(f"❌ 获取视频信息失败，跳过上传 {task_info.get('video_id_list', [])} 用户 {user_name} ")
            continue

        clean_files, keep_files = gen_all_files_to_cleanup(task_info)
        account_executor = account_executors[user_name]

        future = account_executor.submit(
            upload_worker,
            upload_params,
            task_info,
            clean_files,
            user_name,
            manager,
            video_info_dict
        )
        futures.append(future)
        processed_task_ids.add(str(task_info.get('_id')))

        # 记录本轮已投用户
        already_upload_users.append(user_name)

    # 收尾与统计
    gen_all_statistic_info(already_upload_users, user_upload_info, filter_task_list, tobe_upload_video_info, allowed_user_name_list)

    # 3. 【恢复原样】：死等所有上传任务完成，锁住大循环节奏
    if futures:
        print(f"⏳ 等待 {len(futures)} 个后台投稿任务完成...")
        concurrent.futures.wait(futures, timeout=None)
        print("✅ 本轮后台投稿任务全部完成！")

    return len(already_upload_users)


def auto_produce(manager, max_produce_count=2):
    """
    【制作流】一个没有感情的渲染机器，恢复并发能力，做完一批接一批
    """
    # 1. 线程安全地获取配置
    config_map, user_config = get_safe_configs()
    allowed_user_name_list = list(config_map.keys())

    stop_flag = user_config.get('stop_flag', False)
    if stop_flag:
        print(f"检测到停止制作开关已开启，暂停本轮制作。 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    tasks_to_upload = manager.find_tasks_by_status([TaskStatus.PLAN_GENERATED, TaskStatus.TO_UPLOADED])
    existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info = statistic_tasks_with_video(tasks_to_upload, allowed_user_name_list)

    _, _, sort_not_existing_video_tasks = sort_tasks(existing_video_tasks, not_existing_video_tasks, tobe_upload_video_info)

    candidate_tasks = []
    used_video_id_list = []

    # 筛选任务
    for task_info in sort_not_existing_video_tasks:
        user_name = task_info.get('userName')
        if user_name not in config_map:
            continue

        bvid = task_info.get('bvid', '')
        status = task_info.get('status')
        video_id_list = task_info.get('video_id_list', [])

        if bvid or status == TaskStatus.UPLOADED:
            continue

        if any(vid in used_video_id_list for vid in video_id_list):
            continue

        used_video_id_list.extend(video_id_list)
        candidate_tasks.append(task_info)

        if len(candidate_tasks) >= max_produce_count:
            break

    if not candidate_tasks:
        return 0

    # 定义并行的 Worker
    def produce_worker(task):
        u_name = task.get('userName')
        v_list = task.get('video_id_list', [])

        print(f"\n🎬 [并行制作流] 开始为 {u_name} 独立渲染制作视频 {v_list} 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

        failure_details, _, _, _ = gen_video(task, config_map, user_config, manager)

        if check_failure_details(failure_details):
            print(f"❌ [并行制作流] 制作失败 {v_list} 用户 {u_name}")
        else:
            print(f"✅ [并行制作流] 制作成功，随时可被投稿流接管 {v_list} 用户 {u_name}")

    # 并行渲染
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_produce_count) as executor:
        futures = [executor.submit(produce_worker, t) for t in candidate_tasks]
        # 等待这批视频渲染完毕，再开启下一轮循环去取新任务
        concurrent.futures.wait(futures, timeout=None)

    return len(candidate_tasks)


def upload_loop(manager):
    """主线程循环：永不停歇的调度中心"""
    while True:
        exist_count = auto_upload(manager)
        if exist_count == 0:
            time.sleep(60)
        else:
            time.sleep(5)


def produce_loop(manager):
    """后台线程循环：永不停歇的生产车间"""
    while True:
        produced_count = auto_produce(manager, max_produce_count=2)
        if produced_count == 0:
            time.sleep(60)
        else:
            time.sleep(1)


if __name__ == "__main__":
    from utils.mongo_base import gen_db_object
    from utils.mongo_manager import MongoManager

    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)

    print("🚀 启动独立双流模式：[投稿流] 与 [并行制作流] 已建立...")

    # 1. 启动制作流的独立后台线程 (daemon=True 保证脚本结束时自动清理)
    produce_thread = threading.Thread(target=produce_loop, args=(manager,), daemon=True, name="ProduceThread")
    produce_thread.start()

    # 2. 将投稿流挂载在主线程上运行
    upload_loop(manager)
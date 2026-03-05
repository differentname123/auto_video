# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/14 18:39
:last_date:
    2025/12/14 18:39
:description:
    进行视频剪辑的主代码
    整体逻辑：
        1.查询需要处理的任务
"""
import copy
import multiprocessing
import os
import shutil
import time
import traceback
from datetime import datetime, timezone, timedelta

from bson import ObjectId

from application.llm_generator import gen_logical_scene_llm, gen_overlays_text_llm, gen_owner_asr_by_llm, \
    gen_hudong_by_llm, gen_video_script_llm, align_single_timestamp, gen_upload_info_llm
from application.video_process import gen_video_by_script
from utils.bilibili.bili_utils import check_duplicate_video
from utils.video_utils import remove_static_background_video, reduce_and_replace_video, probe_duration, get_scene, \
    clip_and_merge_segments, has_audio, probe_video, dynamic_video_area_blur
from video_common_config import VIDEO_MAX_RETRY_TIMES, VIDEO_MATERIAL_BASE_PATH, VIDEO_ERROR, \
    _configure_third_party_paths, TaskStatus, NEED_REFRESH_COMMENT, ERROR_STATUS, build_video_paths, \
    check_failure_details, fix_split_time_points

_configure_third_party_paths()

from third_party.TikTokDownloader.douyin_downloader import download_douyin_video_sync, get_comment
from utils.common_utils import is_valid_target_file_simple, time_to_ms, merge_intervals, get_remaining_segments, \
    safe_process_limit, read_json, get_simple_play_distribution
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager


def query_need_process_tasks():
    """
    查询需要处理的任务。
    1. 查找状态不是 '已完成' 的任务。
    2. 过滤掉失败次数 (failed_count) 超过最大重试次数的任务。
    """
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    max_retry_time = VIDEO_MAX_RETRY_TIMES

    # 1. 查询publish_tasks中status不为'已完成'的记录
    unfinished_tasks = manager.find_unfinished_tasks()

    # 2. 在内存中过滤掉失败次数超过上限的任务
    tasks_to_process = []
    for task in unfinished_tasks:
        # 使用 .get(key, default_value) 方法安全地获取 failed_count，如果字段不存在则默认为 0
        failed_count = task.get('failed_count', 0)

        # 如果失败次数小于或等于最大重试次数，则该任务需要处理
        if failed_count <= max_retry_time:
            tasks_to_process.append(task)

    return tasks_to_process


@safe_process_limit(limit=2, name="cutoff_target_segment")
def cutoff_target_segment(video_path, remove_time_segments, output_path):
    """
    按照期望的时间段，剔除指定时间段的视频
    :param video_path:
    :param remove_time_segments:
    :param output_path:
    :return:
    """
    start_time = time.time()
    print(
        f"开始剔除视频 {video_path} 的时间段: {remove_time_segments}，输出路径: {output_path} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    all_timestamp_list = []
    for remove_time_segment in remove_time_segments:
        # 简单校验格式，防止 crash
        if '-' in remove_time_segment:
            start_str, end_str = remove_time_segment.split('-')
            # 确保转换为整数或浮点数
            start_ms = time_to_ms(start_str)
            end_ms = time_to_ms(end_str)
            all_timestamp_list.append(start_ms)
            all_timestamp_list.append(end_ms)
    # 对all_timestamp_list进行去重和排序
    all_timestamp_list = sorted(set(all_timestamp_list))
    if len(all_timestamp_list) > 0:
        print(f"准备剔除视频 {video_path} 的时间段: {remove_time_segments}，对应的时间戳列表: {all_timestamp_list}")
    else:
        # 复制一份video_path到output_path
        shutil.copy2(video_path, output_path)
        print(
            f"完成剔除视频 {video_path} 的时间段，输出路径: {output_path} 耗时 {time.time() - start_time:.2f}s  当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

        return []
    time_map = {}
    merged_timestamps = get_scene(video_path)

    for target_ts in all_timestamp_list:
        new_ts, strategy, info = align_single_timestamp(target_ts, merged_timestamps, video_path)
        # 3. 打印日志
        if strategy == 'visual':
            print(f"[Scene: {target_ts} -> {new_ts} "
                  f"(🖼️ 视觉修正: count={info['count']}, diff={info['diff']}ms, score={info['score']:.2f})")

        elif strategy == 'subtitle':
            print(f": {target_ts} -> {new_ts} "
                  f"(🛠️ 字幕修正: {info['reason']})")

        elif strategy == 'failed':
            print(f" {target_ts} (保持不变, 字幕对齐失败, 原因: {info['reason']})")
        time_map[target_ts] = new_ts

    # 根据时间映射生成新的剔除时间段

    fixed_remove_time_segments = []
    for remove_time_segment in remove_time_segments:
        # 简单校验格式，防止 crash
        if '-' in remove_time_segment:
            start_str, end_str = remove_time_segment.split('-')
            # 确保转换为整数或浮点数
            start_ms = time_to_ms(start_str)
            end_ms = time_to_ms(end_str)
            fixed_start_ms = time_map.get(start_ms, start_ms)
            fixed_end_ms = time_map.get(end_ms, end_ms)
            fixed_remove_time_segments.append((fixed_start_ms, fixed_end_ms))

    merged_fixed_remove_time_segments = merge_intervals(fixed_remove_time_segments)
    duration_ms = probe_duration(video_path) * 1000

    remaining_segments = get_remaining_segments(duration_ms, merged_fixed_remove_time_segments)

    # 使用ffmpeg命令行工具进行视频剪辑
    clip_and_merge_segments(video_path, remaining_segments, output_path)
    print(
        f"完成剔除视频 {video_path} 的时间段，输出路径: {output_path} 耗时 {time.time() - start_time:.2f}s  当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return fixed_remove_time_segments


def generate_blur_segments(raw_data: list, video_w: int, video_h: int, duration) -> list:
    """
    将包含归一化坐标的原始数据转换为 dynamic_video_area_blur 可用的像素级参数列表。

    :param raw_data: 包含 start_time, end_time, boxes 的列表
    :param video_w: 视频总宽度 (px)
    :param video_h: 视频总高度 (px)
    :return: 格式化后的 blur_segments 列表
    """
    formatted_segments = []

    for item in raw_data:
        # 获取当前时间段
        start = item.get("start_time", "00:00")
        end = item.get("end_time", "00:00")
        start = time_to_ms(start) / 1000
        end = time_to_ms(end) / 1000
        if end == 0 or end > duration:
            end = duration
        boxes = item.get("boxes", [])

        for box in boxes:
            # 1. 提取归一化坐标 (假设输入为 x, y, w, h 且为 0.0-1.0 的比例)
            # 注意：这里假设 x, y 是左上角坐标。如果是中心点坐标，计算方式需调整。
            norm_x = box.get("x", 0)
            norm_y = box.get("y", 0)
            norm_w = box.get("w", 0)
            norm_h = box.get("h", 0)

            # 2. 转换为绝对像素坐标
            # 使用 int() 向下取整，确保坐标是整数
            pixel_x = int(norm_x * video_w)
            pixel_y = int(norm_y * video_h)
            pixel_w = int(norm_w * video_w)
            pixel_h = int(norm_h * video_h)

            # 3. 计算右下角坐标 (x2, y2)
            # x2 = x1 + w, y2 = y1 + h
            x1 = pixel_x
            y1 = pixel_y
            x2 = pixel_x + pixel_w
            y2 = pixel_y + pixel_h

            # 4. 构建符合 dynamic_video_area_blur 要求的字典
            segment = {
                "start": start,
                "end": end,
                "bbox": (x1, y1, x2, y2)
            }
            formatted_segments.append(segment)

    return formatted_segments

@safe_process_limit(limit=1, name="gen_blur_video_path")
def gen_blur_video_path(video_path, output_path, watermark_list):
    """
    生成模糊指定区域的视频
    :return:
    """
    start_time = time.time()
    if not watermark_list:
        shutil.copy2(video_path, output_path)
        return
    video_info = probe_video(video_path)
    width = video_info['width']
    height = video_info['height']
    duration = video_info['duration']
    blur_configs = generate_blur_segments(watermark_list, width, height, duration)
    dynamic_video_area_blur(video_path, output_path, blur_configs)
    video_size = os.path.getsize(video_path)
    print(
        f"模糊视频生成完成: {output_path}，耗时 {time.time() - start_time:.2f}s 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not is_valid_target_file_simple(output_path, video_size * 0.1):
        raise Exception(f"模糊视频生成失败: {output_path}")


@safe_process_limit(limit=2, name="process_origin_video")
def process_origin_video(video_id, video_info):
    """
    处理原始视频生成后续需要处理的视频
    :param video_id:
    :return:
    """
    print(f"开始处理视频 {video_id} 的原始视频 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    video_path_info = build_video_paths(video_id)
    origin_video_path = video_path_info['origin_video_path']
    origin_video_path_blur = video_path_info['origin_video_path_blur']
    origin_video_delete_part_path = video_path_info['origin_video_delete_part_path']
    low_origin_video_path = video_path_info['low_origin_video_path']
    static_cut_video_path = video_path_info['static_cut_video_path']
    low_resolution_video_path = video_path_info['low_resolution_video_path']

    if not is_valid_target_file_simple(origin_video_path):
        raise FileNotFoundError(f"原始视频文件不存在: {origin_video_path}")

    # 使用一个变量标识文件状态是否发生变更，利用连锁反应触发后续更新
    file_changed = False

    # 进行指定区域的模糊处理
    if not is_valid_target_file_simple(origin_video_path_blur):
        watermark_list = video_info.get('extra_info', {}).get('watermark_list', [])
        gen_blur_video_path(origin_video_path, origin_video_path_blur, watermark_list)

    # 1. 剪切片段处理
    # 如果文件不存在，或者强制要求重剪，则执行
    if not is_valid_target_file_simple(origin_video_delete_part_path) or video_info.get('need_recut', True):
        remove_time_segments = video_info.get('extra_info', {}).get('remove_time_segments', [])
        fixed_remove_time_segments = cutoff_target_segment(origin_video_path_blur, remove_time_segments,
                                                           origin_video_delete_part_path)
        video_info['extra_info']['fixed_remove_time_segments'] = fixed_remove_time_segments
        video_info['need_recut'] = False
        split_time_points = video_info.get('extra_info', {}).get('split_time_points', [])
        fixed_split_time_points = fix_split_time_points(fixed_remove_time_segments, split_time_points)
        video_info['extra_info']['fixed_split_time_points'] = fixed_split_time_points
        # 标记变动，后续步骤将强制执行
        file_changed = True

    # 2. 生成 low_origin_video
    # 如果文件不存在，或者上一步发生了变动(file_changed为True)，则执行
    if not is_valid_target_file_simple(low_origin_video_path) or file_changed:
        shutil.copy2(origin_video_delete_part_path, low_origin_video_path)
        file_changed = True

    # 3. 生成 static_cut_video
    # 如果文件不存在，或者上一步发生了变动，则执行
    if not is_valid_target_file_simple(static_cut_video_path) or file_changed:
        # 第一步先进行降低分辨率和帧率(初步)
        params = {
            'crf': 23,
            'target_width': 2560,
            'target_fps': 30
        }
        reduce_and_replace_video(low_origin_video_path, **params)
        retention_area_boxes = (
                                       (video_info.get('extra_info') or {})
                                       .get('retention_area') or {}
                               ).get('boxes') or []
        bbox = None
        if retention_area_boxes:
            bbox = retention_area_boxes[0]

        # 第二步进行静态背景去除
        crop_result, crop_path = remove_static_background_video(low_origin_video_path, bbox=bbox)
        shutil.copy2(crop_path, static_cut_video_path)
        file_changed = True

    # 4. 生成 low_resolution_video
    # 如果文件不存在，或者上一步发生了变动，则执行
    if not is_valid_target_file_simple(low_resolution_video_path) or file_changed:
        # 第三步进行降低分辨率和帧率（超级压缩）
        shutil.copy2(static_cut_video_path, low_resolution_video_path)
        reduce_and_replace_video(low_resolution_video_path)
        file_changed = True

    print(f"视频 {video_id} 的原始视频处理完成。")


def gen_extra_info(video_info_dict, manager, gen_video):
    """
    为每个视频生成额外信息 逻辑场景划分 覆盖文字识别 作者语音识别
    :param video_info_dict:
    :return:
    """
    failure_details = {}
    cost_time_info = {}
    all_start_time = time.time()
    for video_id, video_info in video_info_dict.items():
        start_time = time.time()
        cost_time_info[video_id] = {}
        all_path_info = build_video_paths(video_id)

        # 生成逻辑性的场景划分
        logical_scene_info = video_info.get('logical_scene_info')
        video_path = all_path_info['low_resolution_video_path']
        logical_cost_time_info = {}
        if not logical_scene_info:
            error_info, logical_scene_info, logical_cost_time_info = gen_logical_scene_llm(video_path, video_info,
                                                                                           all_path_info)
            if not error_info:
                video_info['logical_scene_info'] = logical_scene_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
            update_video_info(video_info_dict, manager, failure_details, error_key='logical_error')
        # 记录耗时
        logical_cost_time_info['total_time'] = time.time() - start_time
        cost_time_info[video_id]['logical_scene'] = logical_cost_time_info
        if check_failure_details(failure_details):
            return failure_details, cost_time_info
        print(
            f"视频 {video_id} logical_scene_info生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时 {logical_cost_time_info['total_time']:.2f}s")

        # ---------------- 阶段2: 情绪性花字 ----------------
        t_start = time.time()

        # 生气情绪性花字
        video_overlays_text_info = video_info.get('video_overlays_text_info', {})
        if not video_overlays_text_info and not gen_video:
            error_info, video_overlays_text_info = gen_overlays_text_llm(video_path, video_info)
            if not error_info:
                video_info['video_overlays_text_info'] = video_overlays_text_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.WARNING
                }
                video_info["overlays_text_error"] = error_info
            update_video_info(video_info_dict, manager, failure_details, error_key='overlays_text_error')

        # 记录耗时
        cost_time_info[video_id]['overlays_text'] = time.time() - t_start
        if check_failure_details(failure_details):
            return failure_details, cost_time_info
        failure_details = {}
        print(
            f"视频 {video_id} overlays_text_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时{cost_time_info[video_id]['overlays_text']}")

        # ---------------- 阶段3: ASR识别 ----------------
        t_start = time.time()

        # 生成asr识别结果
        owner_asr_info = video_info.get('owner_asr_info', None)
        is_contains_author_voice = video_info.get('extra_info', {}).get('is_contains_author_voice', True)
        if is_contains_author_voice and owner_asr_info is None:
            error_info, owner_asr_info = gen_owner_asr_by_llm(video_path, video_info)
            if not error_info:
                video_info['owner_asr_info'] = owner_asr_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
            update_video_info(video_info_dict, manager, failure_details, error_key='owner_asr_error')

        # 记录耗时
        cost_time_info[video_id]['owner_asr'] = time.time() - t_start

        if check_failure_details(failure_details):
            return failure_details, cost_time_info
        print(
            f"视频 {video_id} owner_asr_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时{cost_time_info[video_id]['owner_asr']:.2f}s")

        # ---------------- 阶段4: 互动信息 ----------------
        t_start = time.time()

        # 生成互动信息
        hudong_info = video_info.get('hudong_info', {})
        if not hudong_info:
            error_info, hudong_info = gen_hudong_by_llm(video_path, video_info)
            if not error_info:
                video_info['hudong_info'] = hudong_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
            update_video_info(video_info_dict, manager, failure_details, error_key='hudong_error')

        # 记录耗时
        cost_time_info[video_id]['hudong_info'] = time.time() - t_start
        if check_failure_details(failure_details):
            return failure_details, cost_time_info
        print(
            f"视频 {video_id} hudong_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时{cost_time_info[video_id]['hudong_info']:.2f}s")

        # ---------------- 最后: 打印各阶段耗时 ----------------
        print(
            f"📊 视频 {video_id} 总耗时: {time.time() - start_time:.2f}s 各阶段处理耗时统计: [{cost_time_info[video_id]}] ")
    print(
        f"🎉 {video_info_dict.keys()} 所有视频额外信息生成完成。总耗时: {time.time() - all_start_time:.2f}s {cost_time_info}")
    final_cost_time_info = {}
    final_cost_time_info['extra_info'] = cost_time_info
    return failure_details, final_cost_time_info


def gen_video_info_dict(task_info, manager):
    """
    生成相应的单视频信息字典，key为video_id，值为物料表的视频信息
    :param task_info:
    :param manager:
    :return:
    """
    start_time = time.time()
    cost_time_info = {}
    failure_details = {}  # 使用字典记录每个失败视频的详细原因

    video_id_list = task_info.get('video_id_list', [])
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志

    if not video_id_list:
        error_info = f"任务 {task_id} 的 video_id_list 为空，直接标记为失败。"
        print(error_info)
        failure_details[task_id] = {
            "error_info": error_info,
            "error_level": ERROR_STATUS.CRITICAL
        }
        return failure_details, {}

    # 1. 批量获取所有需要的物料信息
    video_info_list = manager.find_materials_by_ids(video_id_list)
    video_info_dict = {video_info['video_id']: video_info for video_info in video_info_list}
    for video_id in video_id_list:
        video_info = video_info_dict.get(video_id)
        if not video_info:
            print(f"任务 {task_id} 严重错误: 视频 {video_id} 在物料库中不存在，已跳过。")
            failure_details[video_id] = {
                "error_info": "Video info not found in database",
                "error_level": ERROR_STATUS.CRITICAL
            }

    cost_time = time.time() - start_time
    cost_time_info['准备视频数据'] = cost_time
    return failure_details, video_info_dict, cost_time_info


def prepare_basic_video_info(video_info_dict):
    """
    准备基础视频信息，比如评论，原始视频，等
    :param video_info_dict:
    :return:
    """
    cost_time_info = {}
    log_pre = f"1️⃣ 准备基础视频信息  当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    start_time = time.time()
    failure_details = {}
    for video_id, video_info in video_info_dict.items():
        try:
            # 准备路径和URL
            video_path_info = build_video_paths(video_id)
            origin_video_path = video_path_info['origin_video_path']
            video_url = f"https://www.douyin.com/video/{video_id}"

            # 步骤A: 保证视频文件存在，且 metadata 也必须存在
            # 修改点: 增加了 `or video_info.get('metadata') is None` 校验
            if not is_valid_target_file_simple(origin_video_path) or video_info.get('metadata') is None:
                print(f"{log_pre} 视频 {video_id} 的原始文件不存在或元数据(metadata)缺失，准备下载/获取...")
                result = download_douyin_video_sync(video_url)

                if not result:
                    error_info = f"{log_pre}错误: 视频 {video_id} 下载失败。"
                    print(error_info)
                    failure_details[video_id] = {
                        "error_info": error_info,
                        "error_level": ERROR_STATUS.ERROR
                    }
                    continue

                # 下载成功
                original_file_path, metadata = result
                os.makedirs(os.path.dirname(origin_video_path), exist_ok=True)
                os.replace(original_file_path, origin_video_path)
                print(f"{log_pre} 视频 {video_id} 下载并移动成功。")
                video_info['metadata'] = metadata

            check_result = has_audio(origin_video_path)
            if not check_result:
                print(f"{log_pre} 警告: 视频 {video_id} 不包含音频轨道，可能影响后续处理。")
                raise ValueError(f"视频 {video_id} 不包含音频轨道")
            # 步骤B: 保证评论信息完整
            comment_list = video_info.get('comment_list', [])
            if not comment_list or NEED_REFRESH_COMMENT:
                print(f"{log_pre} 视频 {video_id} 的评论需要获取或刷新...")
                fetched_comments = get_comment(video_id, comment_limit=100)
                video_info['comment_list'] = fetched_comments
            print(f"{log_pre} 视频 {video_id} 的基础信息准备完成。 耗时 {time.time() - start_time:.2f}s")

            # 判断is_duplicate是否已经存在，避免重复计算
            is_duplicate = video_info.get('is_duplicate')
            if is_duplicate is None:
                try:
                    is_duplicate = check_duplicate_video(video_info.get('metadata')[0])
                except Exception as e:
                    traceback.print_exc()
                    print(f"{log_pre} 警告: 视频 {video_id} 检测重复时发生异常: {str(e)}，默认设置为非重复。")
                    is_duplicate = False
                video_info['is_duplicate'] = is_duplicate

        except Exception as e:
            traceback.print_exc()
            error_info = f"{log_pre} ⚠️ 严重错误: 处理视频 {video_id} 时发生未知异常: {str(e)}"
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
    cost_time_info['准备基础视频信息'] = time.time() - start_time
    return failure_details, video_info_dict, cost_time_info


def update_video_info(video_info_dict, manager, failure_details, error_key='last_error'):
    """
    更新物料视频信息
    :param video_info_dict:
    :param manager:
    :param failure_details:
    :param error_key:
    :return:
    """
    for video_id, detail in failure_details.items():
        video_info = video_info_dict.get(video_id)
        # 注释掉下面的一行代码就能够保存历史错误，而不至于覆盖
        video_info[error_key] = ""
        if video_info:
            video_info[error_key] = detail.get('error_info', 'Unknown error')
    manager.upsert_materials(video_info_dict.values())


def gen_derive_videos(video_info_dict):
    """
    生成后续需要处理的派生视频，主要是静态去除以及降低分辨率后的视频
    :param video_info_dict:
    :return:
    """
    cost_time_info = {}
    start_time = time.time()
    failure_details = {}
    for video_id, video_info in video_info_dict.items():
        try:
            process_origin_video(video_id, video_info)
        except Exception as e:
            traceback.print_exc()
            error_info = f"严重错误: 处理视频 {video_id} 的原始视频时发生异常: {str(e)}"
            print(error_info)
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
    cost_time_info['生成派生视频'] = time.time() - start_time
    return failure_details, cost_time_info


def gen_video_script(task_info, video_info_dict, manager):
    """
    生成多素材的方案
    :param task_info:
    :param video_info_dict:
    :param manager:
    :return:
    """
    start_time = time.time()
    cost_time_info = {}
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志
    failure_details = {}
    video_script_info = task_info.get('video_script_info', {})
    if not video_script_info:
        error_info, video_script_info, final_scene_info = gen_video_script_llm(task_info, video_info_dict)
        if not error_info:
            task_info['video_script_info'] = video_script_info
            task_info['final_scene_info'] = final_scene_info
        else:
            failure_details[task_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            task_info["script_error"] = error_info
        manager.upsert_tasks([task_info])
    cost_time_info['生成视频脚本'] = time.time() - start_time
    return failure_details, cost_time_info


def gen_upload_info(task_info, video_info_dict, manager):
    """
    生成投稿需要的信息
    :param task_info:
    :param video_info_dict:
    :return:
    """
    cost_time_info = {}
    start_time = time.time()
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志
    failure_details = {}
    upload_info = task_info.get('upload_info', {})
    if not upload_info:
        error_info, upload_info = gen_upload_info_llm(task_info, video_info_dict)
        if not error_info:
            task_info['upload_info'] = upload_info
            task_info['status'] = TaskStatus.PLAN_GENERATED
            # task_info = gen_true_type_and_tags(task_info, upload_info)
        else:
            failure_details[task_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            task_info["upload_info_error"] = error_info
        manager.upsert_tasks([task_info])
    cost_time_info['生成投稿信息'] = time.time() - start_time
    return failure_details, cost_time_info


def process_single_task(task_info, manager, gen_video=False):
    """
    处理单个任务的逻辑，此函数经过了全面的健壮性和效率优化。

    - manager: 外部传入的 MongoManager 实例，用于数据库操作。
    """
    print(
        f"🚀 视频开始视频处理任务{task_info.get('userName', 'N/A')} {task_info.get('_id', 'N/A')} {task_info.get('video_id_list', 'N/A')}。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # [新增] 初始化计时变量
    all_cost_time_info = {}
    start_time = time.time()
    chosen_script = None
    # 准备好相应的视频数据
    failure_details, video_info_dict, cost_time_info = gen_video_info_dict(task_info, manager)
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 确保基础数据存在，比如视频文件，评论等
    failure_details, video_info_dict, cost_time_info = prepare_basic_video_info(video_info_dict)
    update_video_info(video_info_dict, manager, failure_details, error_key='prepare_basic_video_error')
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 生成后续需要处理的派生视频，删除指定片段主要是静态去除以及降低分辨率后的视频
    failure_details, cost_time_info = gen_derive_videos(video_info_dict)
    update_video_info(video_info_dict, manager, failure_details, error_key='gen_derive_error')
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 为每一个视频生成需要的大模型信息 场景切分 asr识别， 图片文字等
    failure_details, cost_time_info = gen_extra_info(video_info_dict, manager, gen_video)
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    print(
        f"2️⃣ 任务 {video_info_dict.keys()} 单视频信息生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时 {cost_time_info}")

    # 生成新的视频脚本方案
    failure_details, cost_time_info = gen_video_script(task_info, video_info_dict, manager)
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    print(
        f"3️⃣ 任务 {video_info_dict.keys()} 脚本生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时 {cost_time_info}")

    # 生成投稿所需的信息
    failure_details, cost_time_info = gen_upload_info(task_info, video_info_dict, manager)
    all_cost_time_info.update(cost_time_info)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    task_info['status'] = TaskStatus.PLAN_GENERATED
    manager.upsert_tasks([task_info])
    print(
        f"4️⃣任务 {video_info_dict.keys()} 投稿信息生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 耗时 {cost_time_info}")

    if gen_video:
        # 根据方案生成最终视频
        failure_details, chosen_script, cost_time_info = gen_video_by_script(task_info, video_info_dict)
        all_cost_time_info.update(cost_time_info)
        if check_failure_details(failure_details):
            return failure_details, video_info_dict, chosen_script
        task_info['status'] = TaskStatus.TO_UPLOADED
        manager.upsert_tasks([task_info])
        update_video_info(video_info_dict, manager, failure_details, error_key='gen_video_error')
        print(
            f"任务 {video_info_dict.keys()} 最终视频生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}  耗时 {cost_time_info}")

    print(
        f"✅完成视频完成 成功视频成功 完成所有完成处理耗时统计 (Task Keys: {list(video_info_dict.keys())}) 任务总耗时: {time.time() - start_time:.2f}s {all_cost_time_info}")

    return failure_details, video_info_dict, chosen_script


def _task_process_worker(task_queue, running_task_ids):
    """
    抽取出的进程工作函数：消费者模式
    修改说明：
    1. 增加了 running_task_ids 参数
    2. 在任务结束（成功或彻底失败）时，从 running_task_ids 移除对应的 video_id
    """
    # 在进程内部初始化数据库连接
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)

    # print(f"[消费者-{os.getpid()}] 启动...")

    while True:
        try:
            start_time = time.time()
            task_info = task_queue.get()
            if task_info is None:
                break

            # 获取当前任务关联的视频ID列表，用于后续解锁
            current_video_ids = task_info.get('video_id_list', [])

            failure_details = {}
            try:
                # 执行具体任务逻辑
                failure_details, video_info_dict, chosen_script = process_single_task(task_info, manager)
            except Exception as e:
                traceback.print_exc()
                error_info = f"严重错误: 处理任务 {task_info.get('_id', 'N/A')} 时发生未知异常: {str(e)}"
                print(error_info)
                failure_details[str(task_info.get('_id', 'N/A'))] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.CRITICAL
                }
            finally:
                # --- 状态判断与重试逻辑 ---
                is_failed = check_failure_details(failure_details)

                if is_failed:
                    current_failed_count = task_info.get('failed_count', 0) + 1
                    task_info['failed_count'] = current_failed_count

                    if current_failed_count < 3:
                        print(
                            f"任务 {task_info.get('userName')}{task_info.get('video_id_list')} {task_info.get('_id')} 失败 {current_failed_count} 次，准备重试...当前队列大小: {task_queue.qsize()} 耗时 {time.time() - start_time:.2f}s 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

                        task_info['failure_details'] = str(failure_details)
                        manager.upsert_tasks([task_info])

                        # 稍微延迟后放回队列
                        time.sleep(2)
                        task_queue.put(task_info)
                        continue
                    else:
                        print(
                            f"任务 {task_info.get('userName')}{task_info.get('video_id_list')} {task_info.get('_id')} 失败次数已达 {current_failed_count} 次，标记为失败。当前队列大小: {task_queue.qsize()} 耗时 {time.time() - start_time:.2f}s 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        task_info['status'] = TaskStatus.FAILED
                else:
                    print(
                        f"任务成功 {task_info.get('userName')}{task_info.get('video_id_list')} 成功完成。当前队列大小: {task_queue.qsize()} 耗时 {time.time() - start_time:.2f}s 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    pass
                task_info['failure_details'] = str(failure_details)
                manager.upsert_tasks([task_info])
                if current_video_ids:
                    for v_id in current_video_ids:
                        running_task_ids.pop(v_id, None)

        except Exception as outer_e:
            traceback.print_exc()
            print(f"Worker 进程发生未捕获异常: {outer_e}")
            time.sleep(1)


def check_task_queue(running_task_ids, task_info, check_time=True):
    """

    :param running_task_ids:
    :param task:
    :return:
    """
    update_time = task_info.get('update_time')
    # 如果在10分钟以内，那就false
    if check_time:
        if update_time and (time.time() - update_time.timestamp()) < 600:
            print(
                f"⚠️ [生产者] 任务 {task_info.get('userName')}{task_info.get('video_id_list')} 更新时间过近，跳过入队。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return False
    video_id_list = task_info.get('video_id_list', [])
    # 只要有一个视频id在运行中，就返回false
    for video_id in video_id_list:
        if video_id in running_task_ids:
            print(
                f"⚠️ [生产者] 任务 {task_info.get('userName')}{task_info.get('video_id_list')} 中的视频 {video_id} 正在处理中，跳过入队。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return False
    return True


def maintain_task_queue_once(task_queue, running_task_ids):
    """
    执行一次任务队列维护逻辑：清理僵尸锁 -> 查询任务 -> 排序 -> 入队。
    方便在循环外单独调用。
    """
    # 定义超时时间，例如 2 小时 (7200秒)
    LOCK_TIMEOUT = 7200 * 4

    now = time.time()
    stale_keys = []

    # --- 1. 清理僵尸锁 ---
    try:
        # 转换为普通字典做检查，避免长时间占用 Manager 锁
        snapshot = dict(running_task_ids)
        for v_id, timestamp in snapshot.items():
            if now - timestamp > LOCK_TIMEOUT:
                stale_keys.append(v_id)

        for k in stale_keys:
            # 注意：这里引用 snapshot[k] 必须保证 key 存在，逻辑与原代码一致
            print(f"[生产者] 发现僵尸锁 {k} (超时 {(now - snapshot[k]) / 60:.1f} 分钟)，强制移除。")
            running_task_ids.pop(k, None)
    except Exception as e:
        print(f"清理僵尸锁时出错: {e}")

    queue_size = task_queue.qsize()

    # --- 2. 查询任务 ---
    tasks_to_process = query_need_process_tasks()
    # 过滤掉 方案已生成 状态的任务
    tasks_to_process = [task for task in tasks_to_process if task.get('status') != TaskStatus.PLAN_GENERATED]

    # 读取配置用于排序
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    last_user_set = user_config.get('self_user_list', [])

    sorted_tasks = sorted(
        tasks_to_process,
        key=lambda task: task['userName'] in last_user_set
    )

    print(f"找到 {len(tasks_to_process)} 个需要处理的任务。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    count = 0
    skip_count = 0
    check_time = True
    if queue_size <= 1:
        check_time = False

    for task in sorted_tasks:
        # 检查队列是否过满，防止生产者阻塞太久
        if task_queue.qsize() > 1000:
            print("队列过满，暂停生产")
            break

        # 检查逻辑
        if not check_task_queue(running_task_ids, task, check_time=check_time):
            skip_count += 1
            continue

        # 加锁
        v_ids = task.get('video_id_list', [])
        for v_id in v_ids:
            running_task_ids[v_id] = time.time()  # 确保写入当前时间

        task_queue.put(task)
        count += 1

    print(
        f" 完成恢复完成 完成入库完成 入队 {count} 个任务。 跳过{skip_count} 个任务 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 队列大小: {task_queue.qsize()} 运行中任务数: {len(running_task_ids)}")
    return count + skip_count


def _task_producer_worker(task_queue, running_task_ids):
    """
    生产者工作进程，循环调用维护函数，并实现动态休眠。
    """
    IDLE_SLEEP_TIME = 1800  # 长时间休眠（30分钟）
    BUSY_SLEEP_TIME = 300  # 短时间休眠（5分钟）

    while True:
        try:
            # 调用抽取出的单次维护函数，并获取入队数量
            enqueued_count = maintain_task_queue_once(task_queue, running_task_ids)

            # 如果没有找到新任务入队，并且队列也快空了，就长休眠
            if enqueued_count == 0 and task_queue.qsize() < 5:
                print(f"未发现新任务，生产者进入长休眠 {IDLE_SLEEP_TIME} 秒...")
                time.sleep(IDLE_SLEEP_TIME)
            else:
                # 如果有任务入队，或者队列中还有存货，就短休眠，保持系统活跃
                print(f"任务生产循环完成，生产者进入短休眠 {BUSY_SLEEP_TIME} 秒...")
                time.sleep(BUSY_SLEEP_TIME)

        except Exception as e:
            print(f"生产者异常: {e}")
            # 发生异常时，也进行长休眠，避免因持续异常而耗尽资源
            time.sleep(IDLE_SLEEP_TIME)


def update_narration_key(data_list):
    """
    接收一个列表，将内部结构中的 'new_narration_script_list' 键名修改为 'new_new_narration_script_list'。
    如果处理过程中发生任何错误，返回原始列表。
    """
    try:
        # 使用 deepcopy 创建数据的副本进行操作
        # 这样做是为了确保如果中间出错，原始数据 data_list 不会被部分修改
        result_list = copy.deepcopy(data_list)

        for item in result_list:
            # 定位到 "场景顺序与新文案" 列表
            scenes = item.get("场景顺序与新文案")

            if isinstance(scenes, list):
                for scene in scenes:
                    # 检查是否存在目标 key
                    if "new_narration_script_list" in scene:
                        # 使用 pop 方法取出旧 key 的值并赋值给新 key，同时删除旧 key
                        scene["new_narration_script_list"] = scene.pop("new_narration_script_list")

        return result_list

    except Exception as e:
        # 打印错误日志（可选）
        print(f"处理数据时发生错误: {e}")
        # 发生异常，直接返回传入的原始列表
        return data_list

def recover_task():
    query_2 = {
        "create_time": {
            # 当前时间减去 12 小时
            "$gt": datetime.now() - timedelta(hours=24)
        },
        "failed_count": {
            "$gt": 5
        },
        "failure_details": {
            "$not": {
                "$regex": "禁止|Content blocked|ASR 文本中未检测到任何"
            }
        }
    }

    """
    {
  create_time: {
    $gt: ISODate("2023-01-18T20:44:03.015Z")
  },
  failed_count: {
    $gt: 5
  },
  failure_details: {
    $not: {
      $regex: /禁止|Content blocked/
    }
  }
}

    
    """

    all_task = manager.find_by_custom_query(manager.tasks_collection, query_2)
    print(f"找到 {len(all_task)} 个需要恢复的任务。")
    for task_info in all_task:
        task_info['failed_count'] = 0
        # process_single_task(task_info, manager, gen_video=True)
        # break
    manager.upsert_tasks(all_task)


if __name__ == '__main__':
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    # tasks_to_process = query_need_process_tasks()
    # tasks_to_process = manager.find_task_by_exact_video_ids([
    #     "7590735140998101617",
    #     "7593362953408105780"
    # ])
    # query_2 = {
    #     "status": {"$ne": "666"}
    # }

    query_2 = {
        # "userName": {"$in": ["jie", "qiqixiao"]},
        "status": "待投稿"
    }

    # query_2 = {
    #     "upload_params.title": {
    #         "$regex": "旭旭宝宝"
    #     }
    # }
    #
    # query_2 = {
    #     '_id': ObjectId("6981db70bfaf783377cf4407")
    # }
    recover_task()

    all_task = manager.find_by_custom_query(manager.tasks_collection, query_2)
    print()
    for task_info in all_task:
        upload_info = task_info.get('upload_info', {})
        if '旭旭宝宝' in str(upload_info):
            for value in upload_info:
               value['category_id'] = 1008
            manager.upsert_tasks([task_info])

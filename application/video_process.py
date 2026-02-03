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
import hashlib
import os
import random
import re
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from application.llm_generator import get_best_valid_text, fix_owner_asr_by_subtitle
from application.video_common_config import find_best_solution, VIDEO_TASK_BASE_PATH, build_video_paths, ERROR_STATUS, \
    check_failure_details, correct_owner_timestamps, build_task_video_paths, correct_consecutive_owner_timestamps
from utils.common_utils import is_valid_target_file_simple, merge_intervals, ms_to_time, save_json, read_json, \
    time_to_ms, first_greater, remove_last_punctuation, safe_process_limit
from utils.edge_tts_utils import parse_tts_filename, all_voice_name_list
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from utils.paddle_ocr import find_overall_subtitle_box_target_number, adjust_subtitle_box, analyze_and_filter_boxes
from utils.paddle_ocr_base import run_subtitle_ocr
from utils.video_utils import clip_video_ms, merge_videos_ffmpeg, probe_duration, cover_subtitle, \
    add_text_overlays_to_video, gen_video, text_image_to_video_with_subtitles, get_frame_at_time_safe, \
    add_text_adaptive_padding, add_bgm_to_video, gen_ending_video, add_transparent_watermark, \
    save_frames_around_timestamp, save_frames_around_timestamp_ffmpeg, get_scene


def gen_owner_time_range(owner_asr_info, video_duration_ms):
    """
    生成作者说话时间段
    :return:
    """
    duration_list = []
    for asr_info in owner_asr_info:
        final_text = asr_info.get('final_text', '').strip()
        speaker = asr_info.get('speaker', 'unknown')
        if speaker != 'owner':
            continue
        if not final_text:
            continue
        asr_start = asr_info.get('fixed_start')
        asr_start = max(0, asr_start - 50)
        asr_end = asr_info.get('fixed_end')
        asr_end = min(video_duration_ms, asr_end + 50)
        duration_list.append((asr_start, asr_end))
    merge_intervals_list = merge_intervals(duration_list)
    return merge_intervals_list


def get_owner_asr_info_list(video_info):
    """
    获取作者说话的 ASR 信息列表，并过滤掉与 deleted_scene 有交集的片段
    :param video_info:
    :return: List[dict]
    """
    owner_asr_info_list = video_info.get('owner_asr_info', [])
    logical_scene_info = video_info.get('logical_scene_info', {})
    deleted_scene_list = logical_scene_info.get('deleted_scene', [])

    # 如果没有需要删除的场景，直接返回原始列表，节省计算
    if not deleted_scene_list:
        return owner_asr_info_list

    valid_asr_list = []

    for asr in owner_asr_info_list:
        asr_start = asr.get('start')
        asr_end = asr.get('end')

        # 简单的防御性编程：确保时间戳存在
        if asr_start is None or asr_end is None:
            continue

        is_intersected = False

        # 遍历所有被删除的场景，检查是否有交集
        for deleted in deleted_scene_list:
            del_start = deleted.get('start')
            del_end = deleted.get('end')

            if del_start is None or del_end is None:
                continue

            # 核心逻辑：判断两个时间段是否有重叠
            # 如果 max(开始时间) < min(结束时间)，则说明有重叠
            if max(asr_start, del_start) < min(asr_end, del_end):
                is_intersected = True
                break  # 只要和一个被删片段有交集，这个 ASR 就不要了，跳出内层循环

        # 如果没有和任何 deleted_scene 产生交集，则保留
        if not is_intersected:
            valid_asr_list.append(asr)

    return valid_asr_list

def _process_single_video(video_id, video_info, is_need_narration):
    """
    处理单个视频的字幕遮挡逻辑
    :return: 如果出错返回错误字典，成功返回 None
    """
    log_pre = f"{video_id} 字幕遮挡逻辑 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    # 1. 准备路径和基本信息
    paths = build_video_paths(video_id)
    video_path = paths.get('static_cut_video_path')
    cover_video_path = paths.get('cover_video_path')
    subtitle_box_path = paths.get('subtitle_box_path')

    if not os.path.exists(video_path):
        return {"error_info": f"原视频文件不存在: {video_path}", "error_level": ERROR_STATUS.ERROR}

    if not is_need_narration:
        print(f"不需要替换解说字幕，直接复制文件: {video_path} {log_pre}")
        shutil.copy2(video_path, cover_video_path)
        return None

    video_size = os.path.getsize(video_path)

    # 2. 检查目标文件是否已存在且有效（跳过机制）
    if is_valid_target_file_simple(cover_video_path, video_size * 0.1):
        print(f"已存在遮挡字幕的视频，跳过: {cover_video_path} {log_pre}")
        return None

    # 3. 获取视频时长
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        error_msg = f"获取视频时长失败: {e} {log_pre}"
        print(error_msg)
        return {"error_info": error_msg, "error_level": ERROR_STATUS.ERROR}

    # 4. 计算作者说话时间段
    owner_asr_info_list = video_info.get('owner_asr_info', [])
    if owner_asr_info_list is None :
        owner_asr_info_list = []
    # owner_asr_info_list = correct_owner_timestamps(owner_asr_info_list, video_duration_ms)
    update_time = video_info.get('update_time', datetime.min)
    # 目标日期：2026-02-04 00:00:00
    target_date = datetime(2026, 2, 4)
    # 判断条件：只要 owner_asr_info_list 里没有 "fixed"，或者更新时间在 2026-02-04 之前
    if 'fixed' not in str(owner_asr_info_list) or update_time < target_date:
        owner_asr_info_list = fix_owner_asr_by_subtitle(video_info)
    owner_asr_info_list = correct_consecutive_owner_timestamps(owner_asr_info_list)
    merge_intervals_list = gen_owner_time_range(owner_asr_info_list, video_duration_ms)

    # 5. Case: 如果没有作者说话，直接复制原视频
    if not merge_intervals_list:
        print(f"视频中无作者说话时间段，直接复制文件: {video_path} {log_pre}")
        shutil.copy2(video_path, cover_video_path)
        return None

    # 准备时间数据
    # merged_timerange_list: 用于传给检测算法 [{'startTime': '00:01', 'endTime': '00:05'}]
    # time_ranges: 用于传给ffmpeg处理 [(1.0, 5.0)]
    merged_timerange_list = [
        {"startTime": ms_to_time(start), "endTime": ms_to_time(end)}
        for start, end in merge_intervals_list
    ]
    time_ranges = [(start / 1000, end / 1000) for start, end in merge_intervals_list]

    # 6. 字幕区域检测（如果缓存不存在则计算）
    if not is_valid_target_file_simple(subtitle_box_path, 10):
        box_dir = os.path.dirname(subtitle_box_path)
        detected_box = find_overall_subtitle_box_target_number(
            video_path, merged_timerange_list, output_dir=box_dir,video_duration_ms=video_duration_ms
        )
        if not detected_box:
            return {"error_info": "字幕区域检测失败，未能获取有效的字幕区域坐标。",
                "error_level": ERROR_STATUS.ERROR
            }
        save_json(subtitle_box_path, detected_box)

    # 7. 读取并调整字幕区域坐标
    raw_box = read_json(subtitle_box_path)
    top_left, bottom_right, _, _ = adjust_subtitle_box(video_path, raw_box)

    # 8. 执行遮挡操作
    print(f"开始生成遮挡字幕视频: {cover_video_path} | Box: {raw_box} {log_pre}")
    start_time = time.time()

    cover_subtitle(video_path, cover_video_path, top_left, bottom_right, time_ranges=time_ranges)

    elapsed_time = time.time() - start_time
    print(f"完成生成，耗时: {elapsed_time:.2f} 秒 {log_pre}")

    # 9. 结果校验
    if not is_valid_target_file_simple(cover_video_path, video_size * 0.1):
        current_size_mb = os.path.getsize(cover_video_path) / (1024 * 1024) if os.path.exists(cover_video_path) else 0
        original_size_mb = video_size / (1024 * 1024)

        error_info = (f"生成失败: 文件大小异常。当前: {current_size_mb:.2f}Mb, "
                      f"原始: {original_size_mb:.2f}Mb (需大于原始10%)")

        return {
            "error_info": error_info,
            "error_level": ERROR_STATUS.ERROR
        }

    return None


def gen_subtitle_box_and_cover_subtitle(video_info_dict, is_need_narration):
    """
    批量生成遮挡作者字幕的视频
    :param video_info_dict: 视频信息字典 {video_id: video_info}
    :param manager: 数据库管理器
    :return: 失败详情字典 failure_details
    """
    failure_details = {}

    for video_id, video_info in video_info_dict.items():
        try:
            # 处理单个视频，返回错误信息（如果有）
            error_result = _process_single_video(video_id, video_info, is_need_narration)

            if error_result:
                failure_details[video_id] = error_result


        except Exception as e:
            traceback.print_exc()
            # 捕获未预料的异常，防止整个任务中断
            error_msg = f"处理视频 {video_id} 时发生未知错误: {str(e)}"
            failure_details[video_id] = {
                "error_info": error_msg,
                "error_level": ERROR_STATUS.ERROR
            }

    return failure_details


def add_image_text_to_video(video_path, video_info, optimized_video_plan_info, output_path, output_dir):
    """
    为视频添加图片文字
    """
    try:
        is_fun = random.choice([True, False])
        all_scene_timestamp_list = []
        overlays = optimized_video_plan_info.get('overlays', [])
        logical_scene_info = video_info.get('logical_scene_info', {})
        new_scene_info_list = logical_scene_info.get('new_scene_info', [])
        for scene in new_scene_info_list:
            start = scene.get('start')
            end = scene.get('end')
            all_scene_timestamp_list.append(start)
            all_scene_timestamp_list.append(end)

        all_scene_timestamp_list = sorted(set(all_scene_timestamp_list))
        # 遍历overlays，找到每个时间段内的场景
        texts_list = []
        for overlay in overlays:
            text = overlay.get('text', '').strip()
            start = overlay.get('start')
            position = overlay.get('position', 'TC')
            start_ms = time_to_ms(start)
            next_timestamp = first_greater(start_ms, all_scene_timestamp_list)
            if not next_timestamp:
                continue
            duration = next_timestamp - start_ms
            duration = min(duration, 5000)
            texts_list.append({
                'text': text,
                'start': start_ms / 1000.0,
                'duration': duration / 1000.0,
                'position': position})

        if not texts_list:
            print(f"{video_path} 没有找到任何需要添加的标题文案。")
            return True
        print(f"准备添加 {len(texts_list)} 条文案图片到视频中。is_fun {is_fun} {video_path}")

        add_text_overlays_to_video(video_path, texts_list, output_path, output_dir, is_fun)
    except Exception as e:
        error_info = f"为视频添加文案图片失败: {e}"
        print(error_info)


def add_image_to_video(video_info_dict):
    """
    在视频上增加图片，目前主要是图片文字以及表情包(待实现)
    :return:
    """
    failure_details = {}
    for video_id, video_info in video_info_dict.items():
        all_path_info = build_video_paths(video_id)
        video_path = all_path_info.get('cover_video_path')
        output_dir = os.path.dirname(video_path)
        video_size = os.path.getsize(video_path)

        is_requires_text = video_info.get('extra_info', {}).get('is_requires_text', True)
        if is_requires_text:
            video_overlays_text_info_list = video_info.get('video_overlays_text_info', [])
            image_text_video_path = all_path_info.get('image_text_video_path')
            if not is_valid_target_file_simple(image_text_video_path, video_size * 0.1):
                add_image_text_to_video(video_path, video_info, video_overlays_text_info_list, image_text_video_path,
                                        output_dir)

            if not is_valid_target_file_simple(image_text_video_path, video_size * 0.1):
                error_info = f"添加图片文字后的视频文件大小异常，生成失败。"
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.WARNING
                }
    return failure_details


def process_narration_clips(segment_list, data, min_duration=500):
    """
    (主入口函数) 处理多个时间段的旁白数据。

    核心逻辑：
    1. 归属权分配：遍历原始数据，根据重叠时长，将每个片段分配给且仅分配给一个最匹配的时间段。
    2. 独立处理：对每个时间段内的数据进行裁剪、填充和合并。
    3. 结果汇总：返回所有处理好的片段列表。

    参数:
        segment_list (list): 时间段列表，例如 [(0, 5000), (5200, 10033)]。
        data (list): 原始数据列表。
        min_duration (int): 片段的最小允许时长。

    返回:
        list: 经过所有处理后的最终列表。
    """
    # 1. 初始化桶：为每个 segment 创建一个空的列表来存放属于它的 clips
    # 结构示例：[ [clip_for_seg1...], [clip_for_seg2...] ]
    segmented_data_buckets = [[] for _ in segment_list]

    # 2. 归属权分配：遍历原始数据，决定每个 clip 到底属于哪个 segment
    for clip in data:
        best_segment_index = -1
        max_overlap_duration = 0

        c_start = clip["narration_script_start"]
        c_end = clip["narration_script_end"]

        # 遍历所有目标时间段，看跟谁重叠最多
        for i, (s_start, s_end) in enumerate(segment_list):
            # 计算重叠部分的起始和结束
            overlap_start = max(c_start, s_start)
            overlap_end = min(c_end, s_end)

            # 计算重叠时长
            overlap = max(0, overlap_end - overlap_start)

            # 如果当前重叠时长比之前的更长，更新“最佳归属”
            # 注意：如果重叠时长相等，这里保留在先找到的那个段中
            if overlap > max_overlap_duration:
                max_overlap_duration = overlap
                best_segment_index = i

        # 3. 如果找到了归属（重叠时长 > 0），放入对应的桶中
        if best_segment_index != -1 and max_overlap_duration > 0:
            segmented_data_buckets[best_segment_index].append(clip)

    # 4. 对分配好数据的每个段进行独立处理
    final_result = []

    # 将时间段和对应的专属数据配对进行处理
    for (start, end), assigned_clips in zip(segment_list, segmented_data_buckets):
        # 调用单段处理逻辑
        segment_result = process_single_range_logic(start, end, assigned_clips, min_duration)
        final_result.extend(segment_result)

    return final_result


def process_single_range_logic(start, end, data, min_duration):
    """
    处理单个连续时间段的裁剪、填充、合并。
    """
    # 步骤 1: 过滤和裁剪所有片段，确保它们在[start, end]范围内
    adjusted_data = adjust_clips_to_range(data, start, end)

    # 步骤 2: 使用处理过的数据来填充[start, end]范围内的空白
    filled_clips = fill_time_gaps(start, end, adjusted_data)

    # 步骤 3: 合并所有时长过短的片段
    final_clips = merge_short_clips(filled_clips, min_duration)

    return final_clips


def adjust_clips_to_range(data, start, end):
    """
    预处理函数：过滤和裁剪片段，确保它们严格在[start, end]范围内。
    """
    adjusted_data = []
    for clip in data:
        clip_start = clip["narration_script_start"]
        clip_end = clip["narration_script_end"]

        # 检查片段与[start, end]范围是否有重叠
        if clip_end > start and clip_start < end:
            # 计算裁剪后的新起始和结束时间
            new_start = max(clip_start, start)
            new_end = min(clip_end, end)

            # 只有当裁剪后仍然是有效的时间段时才添加
            if new_start < new_end:
                new_clip = clip.copy()
                new_clip["narration_script_start"] = new_start
                new_clip["narration_script_end"] = new_end
                adjusted_data.append(new_clip)

    return adjusted_data


def fill_time_gaps(start, end, data):
    """
    填充给定时间段内的空白部分。
    """
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        return []

    sorted_data = sorted(data, key=lambda x: x["narration_script_start"])

    result = []
    current_time = start

    for item in sorted_data:
        item_start = item["narration_script_start"]
        item_end = item["narration_script_end"]

        if current_time < item_start:
            result.append({
                "new_narration_script_list": "",
                "narration_script_start": current_time,
                "narration_script_end": item_start
            })

        result.append({
            "new_narration_script_list": item.get("new_narration_script_list", item.get("narration_script", "")),
            "narration_script_start": item_start,
            "narration_script_end": item_end
        })

        current_time = item_end

    if current_time < end:
        result.append({
            "new_narration_script_list": "",
            "narration_script_start": current_time,
            "narration_script_end": end
        })

    return result


def merge_short_clips(clips, min_duration=500):
    """
    合并列表中时长过短的片段。
    """
    if not clips:
        return []

    merged_list = [clips[0]]

    for i in range(1, len(clips)):
        last_clip = merged_list[-1]
        current_clip = clips[i]

        last_duration = last_clip["narration_script_end"] - last_clip["narration_script_start"]
        current_duration = current_clip["narration_script_end"] - current_clip["narration_script_start"]

        if last_duration < min_duration:
            current_clip["narration_script_start"] = last_clip["narration_script_start"]
            if last_clip["new_narration_script_list"] and not current_clip["new_narration_script_list"]:
                current_clip["new_narration_script_list"] = last_clip["new_narration_script_list"]
            merged_list[-1] = current_clip
        elif current_duration < min_duration:
            last_clip["narration_script_end"] = current_clip["narration_script_end"]
            if current_clip["new_narration_script_list"] and not last_clip["new_narration_script_list"]:
                last_clip["new_narration_script_list"] = current_clip["new_narration_script_list"]
        else:
            merged_list.append(current_clip)

    return merged_list


def build_all_need_data_map(video_info_dict):
    """
    生成具体素材的场景信息字段以及说话人语音对应原始信息的字典
    :param video_info_dict:
    :return:
    """
    all_logical_scene_dict = {}
    all_owner_asr_info_dict = {}
    for video_id, video_info in video_info_dict.items():
        logical_scene_info = video_info.get('logical_scene_info')
        new_scene_info_list = logical_scene_info.get('new_scene_info', [])
        for scene in new_scene_info_list:
            scene_number = scene.get('scene_number')
            scene_key = f"{video_id}_{scene_number}"
            all_logical_scene_dict[scene_key] = scene

        owner_asr_info_list = video_info.get('owner_asr_info', [])
        if owner_asr_info_list is None:
            owner_asr_info_list = []

        for asr_info in owner_asr_info_list:
            speaker = asr_info.get('speaker')
            if speaker != 'owner':
                continue
            final_text = asr_info.get('final_text')
            all_owner_asr_info_dict[final_text] = asr_info
    return all_logical_scene_dict, all_owner_asr_info_dict


def process_video_with_owner_text(video_path, split_scene, output_dir, subtitle_box, voice_info):
    new_narration_script_list = split_scene.get('new_narration_script_list', '')
    narration_script_start = split_scene.get('narration_script_start', 0)
    narration_script_end = split_scene.get('narration_script_end', 0)
    segment_output_scene_file = os.path.join(output_dir,
                                             'split_scene/' f'{narration_script_start}_{narration_script_end}.mp4')
    start_time = time.time()

    if narration_script_start >= narration_script_end - 100:
        print(f"跳过无效时间段: {narration_script_start}-{narration_script_end}")
        return None

    if not is_valid_target_file_simple(segment_output_scene_file):
        clip_video_ms(video_path, narration_script_start, narration_script_end, segment_output_scene_file)

    if new_narration_script_list.strip() != '':
        output_path = segment_output_scene_file.replace('.mp4', '_with_text.mp4')
        origin_video_path = segment_output_scene_file
        keep_original_audio = False
        if not is_valid_target_file_simple(output_path):
            # audio_path = gen_audio_path(video_path).replace("vocals.wav", "no_vocals.wav")
            # pure_audio_path = gen_audio_path(video_path).replace(".wav", "_pure.wav")
            # if not is_valid_target_file_simple(pure_audio_path):
            #     process_media_by_volume(audio_path, pure_audio_path)
            # segment_output_scene_background_file = segment_output_scene_file.replace('.mp4', '_with_background.mp4')
            # replace_video_audio(segment_output_scene_file,seg_start, seg_end, pure_audio_path, segment_output_scene_background_file)
            # origin_video_path = segment_output_scene_background_file
            # keep_original_audio = True
            gen_video(new_narration_script_list, output_path, origin_video_path, keep_original_audio=keep_original_audio,
                      fixed_rect=subtitle_box, voice_info=voice_info)
        need_merge_video_file = output_path
    else:
        need_merge_video_file = segment_output_scene_file

    print(f"处理片段 {narration_script_start}_{narration_script_end} 完成，耗时 {time.time() - start_time:.2f} 秒\n")
    return need_merge_video_file


def get_voice_info(tags=None):
    # 1. 处理 tags：如果有传入则展平，没传入(None)则设为空集合
    # 这样 tags 为 None 时，user_tags 就是空集，不会报错
    user_tags = {t for sublist in tags.values() for t in sublist} if tags else set()

    # 2. 读取数据 (保持不变)
    voice_info = read_json(r"W:\project\python_project\watermark_remove\content_community\app\voice_info.json")

    # 3. 计算分数 (保持不变)
    # 如果 user_tags 是空集，len(user_tags & ...) 的结果全是 0
    scores = [
        (name, len(user_tags & {t for v in data.values() for t in v}))
        for name, data in voice_info.items()
    ]

    # 4. 排序与返回 (保持不变)
    # 注意：如果所有分数都是0，这里会取列表的前两个（即json文件里的前两个）进行随机
    final_voice_name = random.choice(sorted(scores, key=lambda x: x[1], reverse=True)[:2])[0] if scores else None

    # 这里的 parse_tts_filename 和 all_voice_name_list 假设你在外部定义了
    original_voice_name, pitch, rate = parse_tts_filename(final_voice_name, all_voice_name_list)
    final_voice_infp = {
        "voice_name": original_voice_name,
        "pitch": pitch,
        "rate": rate
    }
    print(f"ⓘ 根据标签推荐音色: {final_voice_infp}")
    return final_voice_infp


def gen_transition_video(video_path, split_scene_list, transition_text, voice_info, subtitle_box):
    """
    生成转场视频
    :return:
    """
    base_dir = os.path.dirname(video_path)
    image_time = split_scene_list[0][0] + 20
    image_time_str = ms_to_time(image_time)
    image_path = os.path.join(base_dir, 'split_scene', f"{image_time}.jpg")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    transition_video_output_path = os.path.join(base_dir, 'split_scene', f"{image_time}.mp4")
    target_frame = get_frame_at_time_safe(video_path=video_path, time_str=image_time_str)
    cv2.imwrite(image_path, target_frame)
    # 获取图片的分辨率
    height, width, _ = target_frame.shape
    resolution = (width, height)
    text_image_to_video_with_subtitles(transition_text, image_path, transition_video_output_path,
                                       short_text=transition_text, resolution=resolution, voice_info=voice_info,
                                       fixed_rect=subtitle_box)
    return transition_video_output_path


def gen_scene_video(video_path, new_script_scene, narration_detail_info, merged_segment_list, subtitle_box, voice_info):
    """
    生成单个场景视频
    :return:
    """
    failure_details = {}
    need_merge_video_file_list = []

    # 如果场景数据已经存在就直接返回
    output_dir = os.path.dirname(video_path)
    final_scene_output_path = os.path.join(output_dir, 'split_scene', f"{new_script_scene.get('scene_id')}_remake.mp4")
    os.makedirs(os.path.dirname(final_scene_output_path), exist_ok=True)

    if is_valid_target_file_simple(final_scene_output_path):
        print(f"已存在最终单个场景视频，跳过生成: {final_scene_output_path}")
        return failure_details, final_scene_output_path

    # 生成转场视频
    transition_text = new_script_scene.get('transition_text', '')
    if transition_text:
        transition_video_output_path = gen_transition_video(video_path, merged_segment_list, transition_text,
                                                            voice_info, subtitle_box)
        if is_valid_target_file_simple(transition_video_output_path):
            need_merge_video_file_list.append(transition_video_output_path)

    # 生成不同的分割段，然后进行每一段视频的生成
    new_narration_script_list = new_script_scene.get('new_narration_script_list', [])
    new_narration_script_list_info_list = []
    for new_narration_script_list in new_narration_script_list:
        asr_info = narration_detail_info.get(new_narration_script_list)
        new_narration_script_list_info_list.append({
            'new_narration_script_list': new_narration_script_list,
            'narration_script_start': asr_info.get('fixed_start', asr_info.get('start')),
            'narration_script_end': asr_info.get('fixed_end', asr_info.get('end'))
        })
    split_scene_list = process_narration_clips(merged_segment_list, new_narration_script_list_info_list, min_duration=500)
    count = 0
    for split_scene in split_scene_list:
        count += 1
        need_merge_video_file = process_video_with_owner_text(video_path, split_scene, output_dir, subtitle_box,
                                                              voice_info)
        if need_merge_video_file:
            need_merge_video_file_list.append(need_merge_video_file)

    merge_videos_ffmpeg(need_merge_video_file_list, output_path=final_scene_output_path)

    return failure_details, final_scene_output_path


def gen_title_video(is_need_scene_title, all_scene_video_path, all_scene_video_file_list, best_script,
                    video_with_title_output_path):
    """
    生成带有标题以及场景的视频
    :param all_scene_video_path:
    :param all_scene_video_file_list:
    :param best_script:
    :param video_with_title_output_path:
    :return:
    """
    failure_details = {}
    if is_valid_target_file_simple(video_with_title_output_path):
        print(f"已存在带标题视频，跳过生成: {video_with_title_output_path}")
        return failure_details, video_with_title_output_path

    if is_need_scene_title is False:
        print(f"{all_scene_video_path} 不需要生成带标题视频，直接返回原视频路径")
        return failure_details, all_scene_video_path

    new_script_scenes = best_script.get('场景顺序与新文案', [])
    video_abstract = best_script.get('video_abstract')
    video_abstract_list = [video_abstract]
    video_abstract_list = [remove_last_punctuation(text) for text in video_abstract_list if text.strip()]

    PALETTE = ['#FFFFFF', '#FF4C4C', '#FFD700']  # 白 / 黑 / 金
    fontcolor = random.choice(PALETTE)
    color_config = {
        'fontcolor': fontcolor
    }

    if len(new_script_scenes) != len(all_scene_video_file_list):
        error_info = f"新脚本场景数量与生成视频数量不匹配，无法生成带标题视频 分别为 场景数量{len(new_script_scenes)} vs 场景视频数量{len(all_scene_video_file_list)}"
        print(error_info)
        failure_details['error_info'] = {
            "error_info": error_info,
            "error_level": ERROR_STATUS.ERROR
        }
        return failure_details, all_scene_video_path
    # 准备生成标题数据
    texts_list = []
    scene_start = 0
    for i, scene_video_path in enumerate(all_scene_video_file_list):
        temp_text_list = video_abstract_list.copy()

        scene_duration_s = probe_duration(scene_video_path)
        scene_end = scene_start + scene_duration_s
        new_script_scene = new_script_scenes[i]
        on_screen_text = new_script_scene.get('on_screen_text')
        if on_screen_text:
            temp_text_list.append(on_screen_text)
        temp_dict = {}
        temp_dict['text_list'] = temp_text_list
        temp_dict['start_time'] = scene_start
        temp_dict['end_time'] = scene_end
        temp_dict['color_config'] = color_config
        texts_list.append(temp_dict)
        scene_start = scene_end

    if not texts_list:
        print(f"{all_scene_video_path} 没有找到任何需要添加的标题文案。")
        failure_details['error_info'] = {
            "error_info": "没有找到任何需要添加的标题文案。",
            "error_level": ERROR_STATUS.WARNING
        }
        return failure_details, all_scene_video_path
    print(f"准备添加 {len(texts_list)} 条标题文案到视频中。")

    add_text_adaptive_padding(all_scene_video_path, video_with_title_output_path, texts_list)
    return failure_details, video_with_title_output_path


def get_bgm_path(tags={}):
    """
    根据标签匹配数量对BGM进行排序，并选择一个合适的BGM路径。

    Args:
        tags (dict): 输入的标签字典，例如 {'style': ['清新'], 'mood': ['愉快']}
        logger: 日志记录器实例

    Returns:
        str: 选中的BGM文件路径。
    """
    all_tags = []
    for key, value in tags.items():
        all_tags.extend(value)
    # 使用集合以便快速计算交集
    all_tags_set = set(all_tags)

    bgm_dir = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio"
    bgm_info_list = read_json(r"W:\project\python_project\watermark_remove\content_community\app\bgm_info.json")

    bgm_info_map = {}
    for bgm_info in bgm_info_list:
        bgm_name = bgm_info.get('bgm_name', '未知').split('.')[0]
        bgm_tags_dict = bgm_info.get('selected_tags', {})
        bgm_all_tags = []
        for key, bgm_tag_list in bgm_tags_dict.items():
            bgm_all_tags.extend(bgm_tag_list)
        bgm_info_map[bgm_name] = bgm_all_tags

    # 获取所有有效的BGM文件
    bgm_files = [f for f in os.listdir(bgm_dir) if f.lower().endswith('.wav')]
    bgm_file_names = [os.path.splitext(f)[0] for f in bgm_files]

    bgm_with_match_count = []
    for bgm_file_name in bgm_file_names:
        bgm_tags = bgm_info_map.get(bgm_file_name, [])
        if not bgm_tags:
            continue

        # 计算交集，获取匹配的标签数量
        match_count = len(all_tags_set.intersection(set(bgm_tags)))

        if match_count > 0:
            bgm_path = os.path.join(bgm_dir, f"{bgm_file_name}.wav")
            bgm_with_match_count.append({'path': bgm_path, 'match_count': match_count})

    if not bgm_with_match_count:
        # 如果没有任何匹配的BGM，可以采取备用策略，例如随机选择一个BGM
        print(f"在 {bgm_dir} 目录下未找到任何与给定标签匹配的音频文件，将随机选择一个文件。")
        if not bgm_files:
            raise FileNotFoundError(f"在 {bgm_dir} 目录下找不到任何音频文件！")
        return os.path.join(bgm_dir, random.choice(bgm_files))

    # 根据匹配数量进行降序排序
    bgm_with_match_count.sort(key=lambda x: x['match_count'], reverse=True)

    # --- 选择策略 ---
    # 策略2：在匹配度最高的几个BGM中随机选择一个（例如前3个）
    top_n = 2
    top_choices = bgm_with_match_count[:top_n]
    if not top_choices:
        # 理论上，如果bgm_with_match_count不为空，这里就不会为空
        raise ValueError("未能确定顶部的BGM选项。")

    selected_bgm = random.choice(top_choices)

    print(f"最终选择的BGM: {selected_bgm['path']} (匹配数: {selected_bgm['match_count']})")
    return selected_bgm['path']


def _calculate_bgm_ratio(new_script_scenes, video_info_dict):
    """
    辅助函数：计算需要添加BGM的场景比例
    """
    if not new_script_scenes:
        return 0.0

    need_bgm_count = 0
    for scene in new_script_scenes:
        # 1. 如果加了新旁白，原声肯定没了，需要BGM
        if len(scene.get('new_narration_script_list', [])) > 0:
            need_bgm_count += 1
            continue

        # 2. 检查原视频片段是否自带BGM
        scene_id = scene.get('scene_id')
        video_id = scene_id.split('_')[0]
        # 使用链式get避免KeyError，更加优雅
        has_bgm = video_info_dict.get(video_id, {}) \
            .get('extra_info', {}) \
            .get('is_contains_bgm', False)

        if not has_bgm:
            need_bgm_count += 1

    return need_bgm_count / len(new_script_scenes)


def gen_video_with_bgm(video_path, output_video_path, video_info_dict, new_script_scenes, tags):
    """
    尝试生成带有bgm的视频。
    如果需要BGM的场景超过50%，则合成新视频；否则直接使用原视频。
    """
    failure_details = {}
    if is_valid_target_file_simple(output_video_path):
        print(f"已存在带BGM视频，跳过生成: {output_video_path}")
        return failure_details, output_video_path

    # --- 1. 决策阶段：判断是否需要加BGM ---
    bgm_ratio = _calculate_bgm_ratio(new_script_scenes, video_info_dict)
    print(f"需要添加BGM的场景比例: {bgm_ratio:.2f} {video_path} ")
    # 如果比例不足 0.5，直接返回原视频（不进行处理）
    if bgm_ratio <= 0.5:
        return failure_details, video_path

    # --- 2. 准备阶段：获取资源 ---
    bgm_path = get_bgm_path(tags)

    # 修复原代码Bug：如果需要BGM但找不到BGM文件，应该降级返回原视频
    if not bgm_path or not os.path.exists(bgm_path):
        print(f"警告：需要添加BGM (比例 {bgm_ratio:.2f}) 但未找到BGM文件。")
        return failure_details, video_path

    # --- 3. 执行阶段：合成视频 ---
    try:
        add_bgm_to_video(
            video_path,
            bgm_path,
            str(output_video_path),
            auto_compute=True,
            rate=bgm_ratio
        )
    except Exception as e:
        # 捕获潜在的转换异常，防止程序崩溃
        print(f"添加背景音乐执行出错: {e}")
        failure_details['error_info'] = {"error_info": str(e), "error_level": ERROR_STATUS.WARNING}
        return failure_details, video_path

    # --- 4. 验证阶段：检查结果 ---
    original_size = os.path.getsize(video_path)
    # 假设最小阈值为原视频大小的 10%
    if is_valid_target_file_simple(output_video_path, original_size * 0.1):
        print(f"成功：已添加背景音乐。BGM: {bgm_path}, 输出: {output_video_path}")
        return failure_details, output_video_path
    else:
        # 虽然执行了但文件无效（如大小为0），回退到原视频
        error_msg = f"添加背景音乐失败(文件校验未通过)，回退到无BGM版本: {output_video_path}"
        print(error_msg)
        failure_details['error_info'] = {
            "error_info": error_msg,
            "error_level": ERROR_STATUS.WARNING
        }
        return failure_details, video_path


def process_single_scene(new_script_scene, all_final_scene_dict, all_owner_asr_info_dict, all_logical_scene_dict,
                         voice_info):
    """
    处理单个场景的视频生成逻辑
    Returns:
        tuple: (video_id, final_output_path, error_msg)
        如果成功，error_msg 为 None；如果失败，前两个值为 None。
    """
    failure_details = {}
    scene_id = new_script_scene.get('scene_id')
    scene_info = all_final_scene_dict.get(scene_id, {})
    video_id = scene_info.get('source_video_id')

    # 1. 准备视频和字幕路径信息
    all_path_info = build_video_paths(video_id)
    video_path = all_path_info.get('cover_video_path')
    subtitle_box_path = all_path_info.get('subtitle_box_path')

    # 2. 处理字幕框
    subtitle_box = read_json(subtitle_box_path)
    real_subtitle_box = None
    if subtitle_box:
        top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(video_path, subtitle_box)
        real_subtitle_box = [top_left, bottom_right]

    # 3. 检查是否使用图文视频路径
    if is_valid_target_file_simple(all_path_info.get('image_text_video_path')):
        video_path = all_path_info.get('image_text_video_path')

    # 4. 处理旁白脚本对应关系
    scene_number_list = scene_info.get('scene_number_list')
    narration_script_list = scene_info.get('narration_script_list', [])
    target_narration_scripts = new_script_scene.get('new_narration_script_list', [])  # 重命名变量以避免混淆

    new_narration_detail_info = {}

    for i, current_narration in enumerate(target_narration_scripts):
        original_narration_script = narration_script_list[i]
        asr_info = all_owner_asr_info_dict.get(original_narration_script)

        if not asr_info:
            error_info = f"未找到对应的ASR信息，场景ID: {scene_id}, 旁白脚本: {original_narration_script} 第{i}段"
            print(error_info)
            failure_details['error_info'] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            return failure_details, None

        new_narration_detail_info[current_narration] = asr_info

    # 5. 合并时间片段
    all_segment_list = []
    for scene_number in scene_number_list:
        scene_key = f"{video_id}_{scene_number}"
        scene_logical_info = all_logical_scene_dict.get(scene_key)
        if scene_logical_info:
            start = scene_logical_info.get('start')
            end = scene_logical_info.get('end')
            all_segment_list.append((start, end))

    merged_segment_list = merge_intervals(all_segment_list)

    # 6. 生成视频
    failure_details, final_scene_output_path = gen_scene_video(
        video_path,
        new_script_scene,
        new_narration_detail_info,
        merged_segment_list,
        real_subtitle_box,
        voice_info
    )

    return failure_details, final_scene_output_path


def get_watermark_path(user_name: str) -> str:
    """
    生成合适的水印图片路径。
    从 asset/ 目录中筛选包含 user_type 的 .png，按 user_name 的哈希稳定选择。
    """
    asset_dir = r'W:\project\python_project\auto_video\config\asset'
    try:
        all_files = os.listdir(asset_dir)
    except FileNotFoundError:
        print("⚠️ 未找到 asset 目录，使用默认水印。")
        return r"W:\project\python_project\auto_video\config\asset\default_watermark.png"

    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type = "fun"
    user_type_info = user_config.get('user_type_info')
    for temp_user_type, user_list in user_type_info.items():
        if user_name in user_list:
            user_type = temp_user_type
            break

    filtered_files = [f for f in all_files if user_type in f and f.endswith(".png")]
    if not filtered_files:
        print("⚠️ 未找到符合条件的水印图片，使用默认水印。")
        return r"W:\project\python_project\auto_video\config\asset\default_watermark.png"

    filtered_files.sort()
    user_hash_hex = hashlib.sha256(user_name.encode("utf-8")).hexdigest()
    user_hash_int = int(user_hash_hex, 16)
    selected_index = user_hash_int % len(filtered_files)
    selected_file = filtered_files[selected_index]
    watermark_path = os.path.join(asset_dir, selected_file)
    print(
        f"{user_name} ✅ 使用水印图片 {watermark_path} 筛选池大小 {len(filtered_files)} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return watermark_path


def add_watermark_and_ending(video_path, watermark_path, ending_video_path, voice_info, best_script, user_name,
                             user_type):
    failure_details = {}
    base_dir = os.path.dirname(video_path)
    current_video_path = video_path
    temp_ending_video_path = os.path.join(base_dir, 'temp_ending_video.mp4')
    origin_ending_video_path = r'W:\project\python_project\auto_video\config\origin_ending_video.mp4'
    ending_text = best_script.get('upload_info', {}).get('introduction', {}).get('warm_close',
                                                                                 "感谢观看本视频，欢迎点赞、评论、关注、投币、分享！")
    start_time = time.time()
    gen_ending_video(ending_text, temp_ending_video_path, origin_ending_video_path, voice_info)
    merge_videos_ffmpeg([video_path, temp_ending_video_path], output_path=ending_video_path)
    print(f"生成结尾视频完成，耗时 {time.time() - start_time:.2f} 秒，输出路径: {ending_video_path}")

    # 尝试生成结尾视频
    if is_valid_target_file_simple(ending_video_path):
        print(f"成功添加片尾视频，输出路径: {ending_video_path}")
        current_video_path = ending_video_path

    # wm_path = get_watermark_path(user_name)
    # start_time = time.time()
    # add_transparent_watermark(current_video_path, wm_path, watermark_path)
    # print(f"添加水印完成，耗时 {time.time() - start_time:.2f} 秒，输出路径: {watermark_path}")
    #
    # if is_valid_target_file_simple(watermark_path):
    #     print(f"成功添加水印，输出路径: {watermark_path}")
    #     current_video_path = watermark_path

    return failure_details, current_video_path


def merge_script_and_upload_info(video_script_info_list, upload_info_list):
    """
    合并视频脚本信息和上传信息
    :param video_script_info:
    :param upload_info:
    :return:
    """
    upload_info_dict = {}
    for upload_info in upload_info_list:
        title = upload_info.get('title', '')
        upload_info_dict[title] = upload_info

    for video_script_info in video_script_info_list:
        title = video_script_info.get('title')
        upload_info = upload_info_dict.get(title)
        if not upload_info:
            raise ValueError(f"视频方案未找到对应的上传信息，标题: {title}")
        video_script_info['upload_info'] = upload_info
    return video_script_info_list

def clear_exist_split_scene(video_info_dict):
    """
    清理已经存在的 split_scene 目录
    :param video_info_dict:
    :return:
    """
    try:
        for video_id, video_info in video_info_dict.items():
            all_path_info = build_video_paths(video_id)
            video_path = all_path_info.get('cover_video_path')
            base_dir = os.path.dirname(video_path)
            split_scene_dir = os.path.join(base_dir, 'split_scene')
            if os.path.exists(split_scene_dir):
                shutil.rmtree(split_scene_dir)
                print(f"清理已存在的 split_scene 目录: {split_scene_dir}")
    except Exception as e:
        print(f"清理 split_scene 目录时出错: {e}")



def gen_new_video(task_info, video_info_dict):
    """
    生成新的视频
    :param task_info:
    :param video_info_dict:
    :return:
    """
    failure_details = {}
    all_task_video_path_info = build_task_video_paths(task_info)
    final_output_path = all_task_video_path_info.get('final_output_path')
    # 保证final_output_path所在目录存在
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)



    # 准备好一些数据
    final_scene_info = task_info.get('final_scene_info', {})

    video_script_info = task_info.get('video_script_info', {})

    upload_info_list = task_info.get('upload_info', {})
    final_video_script_info = merge_script_and_upload_info(video_script_info, upload_info_list)

    all_logical_scene_dict, all_owner_asr_info_dict = build_all_need_data_map(video_info_dict)
    best_script = find_best_solution(final_video_script_info)

    if is_valid_target_file_simple(final_output_path):
        print(f"已存在最终生成视频，跳过生成: {final_output_path}")
        return failure_details, best_script

    upload_info = best_script.get('upload_info', {})
    # 只保留upload_info的 mood_tags theme_tags pacing_tags 这三个字段
    tags = {
        'mood_tags': upload_info.get('mood_tags', []),
        'theme_tags': upload_info.get('theme_tags', []),
        'pacing_tags': upload_info.get('pacing_tags', [])
    }

    voice_info = get_voice_info(tags)

    new_script_scenes = best_script.get('场景顺序与新文案', [])
    all_scene_list = final_scene_info.get('all_scenes', [])
    all_final_scene_dict = {}
    for scene in all_scene_list:
        scene_id = scene.get('scene_id')
        all_final_scene_dict[scene_id] = scene

    clear_exist_split_scene(video_info_dict)
    # 生成好了的每一个场景视频
    all_scene_video_file_list = []
    for new_script_scene in new_script_scenes:
        failure_details, final_scene_output_path = process_single_scene(
            new_script_scene,
            all_final_scene_dict,
            all_owner_asr_info_dict,
            all_logical_scene_dict,
            voice_info
        )
        if check_failure_details(failure_details):
            return failure_details, best_script
        all_scene_video_file_list.append(final_scene_output_path)

    # 合并所有场景视频
    all_scene_video_path = all_task_video_path_info.get('all_scene_video_path')
    if not is_valid_target_file_simple(all_scene_video_path):
        merge_videos_ffmpeg(all_scene_video_file_list, output_path=all_scene_video_path)

    # 生成带有标题的视频
    is_need_scene_title = task_info.get('creation_guidance_info', {}).get('is_need_scene_title', True)
    video_with_title_output_path = all_task_video_path_info.get('video_with_title_output_path')
    failure_details, current_video_path = gen_title_video(is_need_scene_title, all_scene_video_path,
                                                          all_scene_video_file_list, best_script,
                                                          video_with_title_output_path)
    if check_failure_details(failure_details):
        return failure_details, best_script

    # 生成带有bgm的视频
    video_with_bgm_output_path = all_task_video_path_info.get('video_with_bgm_output_path')
    failure_details, current_video_path = gen_video_with_bgm(current_video_path, video_with_bgm_output_path,
                                                             video_info_dict, new_script_scenes, tags)
    if check_failure_details(failure_details):
        return failure_details, best_script

    # 增加结尾祝福以及水印
    video_with_ending_output_path = all_task_video_path_info.get('video_with_ending_output_path')
    video_with_watermark_output_path = all_task_video_path_info.get('video_with_watermark_output_path')
    failure_details, current_video_path = add_watermark_and_ending(current_video_path,
                                                                   watermark_path=video_with_watermark_output_path,
                                                                   ending_video_path=video_with_ending_output_path,
                                                                   voice_info=voice_info, best_script=best_script,
                                                                   user_name=task_info.get('userName', 'user'),
                                                                   user_type=task_info.get('creation_guidance_info',
                                                                                           {}).get('video_type',
                                                                                                   '娱乐'))
    if check_failure_details(failure_details):
        return failure_details, best_script

    # 将current_video_path复制到final_output_path
    shutil.copyfile(current_video_path, final_output_path)
    return failure_details, best_script


def gen_video_by_script(task_info, video_info_dict):
    """
    通过视频脚本生成新的视频
    :param task_info:
    :param video_info_dict:
    :return:
    """
    cost_time_info = {}
    start_time = time.time()
    chosen_script = {}
    creation_guidance_info = task_info.get('creation_guidance_info', {})
    is_need_narration = creation_guidance_info.get('is_need_audio_replace', False)


    # 生成字幕遮挡视频
    failure_details = gen_subtitle_box_and_cover_subtitle(video_info_dict, is_need_narration)
    if check_failure_details(failure_details):
        return failure_details, chosen_script, cost_time_info

    # 生成有了图片文字的视频
    failure_details = add_image_to_video(video_info_dict)
    if check_failure_details(failure_details):
        return failure_details, chosen_script, cost_time_info

    failure_details, chosen_script = gen_new_video(task_info, video_info_dict)
    if check_failure_details(failure_details):
        return failure_details, chosen_script, cost_time_info

    cost_time_info['生成视频耗时'] = time.time() - start_time
    return failure_details, chosen_script, cost_time_info

def save_frame_demo():
    merged_timestamps = read_json(r"W:\project\python_project\auto_video\videos\material\7597766646886927679\7597766646886927679_low_resolution_scenes\merged_timestamps.json")
    valid_camera_shots = [c for c in merged_timestamps if c and c[0] is not None and c[1] > 0]
    max_delta_ms = 1000
    target_ts = 11750
    # 2. 筛选候选者
    candidates = [
        shot for shot in valid_camera_shots
        if abs(shot[0] - target_ts) <= max_delta_ms
    ]

    # 3. 寻找最佳匹配 (Visual)
    best_shot = None
    if candidates:
        # --- 步骤 1: 预处理 (加工数据) ---
        # 我们创建一个新的列表，把 score 和 diff 算好放进去
        # 新的结构变成了: (原始ts, 原始count, score, diff)
        processed_candidates = []

        for shot in candidates:
            ts = shot[0]
            count = shot[1]

            # 计算逻辑
            diff = ts - target_ts
            ratio = 1
            if diff > 0:
                ratio = 1
            diff = abs(ts - target_ts) * ratio
            score = diff / count if count > 0 else float('inf')

            # 【关键点】把算好的 score 和 diff 加到元组后面
            # 现在的结构是: index 0=ts, 1=count, 2=score, 3=diff
            processed_candidates.append((ts, count, score, diff))

        # --- 步骤 2: 评选 (直接根据 index 2 的 score 选) ---
        # x[2] 就是我们在上面算好的 score
        best_shot = min(processed_candidates, key=lambda x: x[2])
    print()


def batch_cleanup_mp4(directory_path, days=7, dry_run=True):
    """
    扫描并清理 MP4 文件，同时计算释放的磁盘空间（MB）。
    """
    target_dir = Path(directory_path)
    if not target_dir.exists():
        print(f"错误：目录 '{directory_path}' 不存在。")
        return

    # 1. 设定时间阈值
    cutoff_time = time.time() - (days * 24 * 60 * 60)

    # 存储待删除文件信息的列表
    files_to_delete = []
    total_size_bytes = 0  # 累计大小（字节）

    print(f"========== 阶段 1: 扫描目录 ==========")
    print(f"目标路径: {target_dir.absolute()}")
    print(f"保留策略: {days}天内修改 或 文件名以_origin结尾\n")

    total_scanned = 0
    # 使用 rglob 递归扫描所有子文件夹
    for file_path in target_dir.rglob('*.mp4'):
        total_scanned += 1

        # 获取文件状态（一次性获取大小和时间）
        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            f_size = stat.st_size
        except OSError:
            # 如果文件在扫描时被占用或无法读取，跳过
            continue

        # 判断逻辑
        is_recent = mtime > cutoff_time
        is_origin = file_path.name.endswith('_origin.mp4')

        # 既不是最近修改，也不是 origin，加入待删除列表
        if not is_recent and not is_origin:
            files_to_delete.append({
                'path': file_path,
                'size': f_size,
                'mtime': mtime
            })
            total_size_bytes += f_size

    # 计算总 MB
    total_size_mb = total_size_bytes / (1024 * 1024)

    # 2. 报告扫描结果
    print(f"扫描完成。共扫描 MP4 文件: {total_scanned} 个")
    print(f"符合删除条件的文件: {len(files_to_delete)} 个")
    print(f"预计释放空间: {total_size_mb:.2f} MB")  # 核心修改：打印总大小

    if len(files_to_delete) == 0:
        print("没有发现需要清理的文件。")
        return

    # 3. 执行阶段
    print(f"\n========== 阶段 2: 执行操作 ==========")
    if dry_run:
        print("当前为 [预演模式 dry_run=True]，未执行任何删除。")
        print(f"确认无误后（共 {total_size_mb:.2f} MB），请将 dry_run 改为 False 执行。")
    else:
        print("正在执行批量删除...")
        success_count = 0
        fail_count = 0

        for item in files_to_delete:
            f_path = item['path']
            try:
                f_path.unlink()
                print(f"[已删除] {f_path.name}")
                success_count += 1
            except Exception as e:
                print(f"[删除失败] {f_path.name} -> {e}")
                fail_count += 1

        print(f"\n操作结束: 成功删除 {success_count} 个，失败 {fail_count} 个。")


def crop_and_save_images(image_paths, output_dir, box_info):
    """
    根据框信息裁剪图片并保存到指定目录。

    :param image_paths: 图片路径列表，例如 ['1.jpg', '2.png']
    :param output_dir: 保存裁剪后图片的目录
    :param box_info: 框坐标列表 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 从四点坐标中提取矩形的左上角和右下角
    # 假设 box_info 格式为: [[左上], [右上], [右下], [左下]]
    x_coords = [point[0] for point in box_info]
    y_coords = [point[1] for point in box_info]

    left = min(x_coords)
    top = min(y_coords)
    right = max(x_coords)
    bottom = max(y_coords)

    crop_box = (left, top, right, bottom)

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                # 执行裁剪
                cropped_img = img.crop(crop_box)

                # 构建保存路径
                file_name = os.path.basename(img_path)
                save_path = os.path.join(output_dir, f"cropped_{file_name}")

                # 保存图片
                cropped_img.save(save_path)
                print(f"成功处理并保存: {save_path}")
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")

if __name__ == "__main__":
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    video_path = r"W:\project\python_project\auto_video\videos\material\7597599415717615476\7597599415717615476_origin.mp4"
    # output_dir = os.path.join(os.path.dirname(video_path), f'test_scenes')
    # output_path = os.path.join(output_dir, 'clip_video.mp4')
    # clip_video_ms(video_path, 0, 5000, output_path)

    box_dir = os.path.join(os.path.dirname(video_path), f'test_subtitle_box')
    # merged_timerange_list = [{"startTime": 0, "endTime": 78000}]
    # detected_box = find_overall_subtitle_box_target_number(
    #     video_path, merged_timerange_list, output_dir=box_dir, video_duration_ms=78000
    # )
    # print(detected_box)

    # 获取box_dir下面的所有jpg列表
    image_paths = []
    for file_name in os.listdir(box_dir):
        if file_name.lower().endswith('.jpg'):
            image_paths.append(os.path.join(box_dir, file_name))

    box = [[208, 963], [1712, 963], [1712, 1042], [208, 1042]]
    crop_and_save_images(image_paths, os.path.join(box_dir, 'cropped_images'), box)



    # video_info_list = manager.find_materials_by_ids(['7598869943144172846'])
    # for video_info in video_info_list:
    #     video_id = video_info.get('video_id')
    #     fix_owner_asr_by_subtitle(video_info)
        # _process_single_video(video_id, video_info, is_need_narration=True)

    # time_list = [11167, 12433, 11750]
    # my_video_path = r"W:\project\python_project\auto_video\videos\material\7597766646886927679\7597766646886927679_low_resolution.mp4"
    # for timestamp in time_list:
    #     timestamp = timestamp / 1000
    #     save_frames_around_timestamp(my_video_path, timestamp, 3, str(os.path.join(os.path.dirname(my_video_path), 'scenes', f"{timestamp}")))
    # video_dir = r"W:\project\python_project\auto_video\videos"
    # # 删除所有超时的MP4
    # batch_cleanup_mp4(video_dir, dry_run=False)


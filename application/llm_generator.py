# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 22:40
:last_date:
    2025/12/15 22:40
:description:
    主要是调用大模型生成内容的业务代码
"""
import copy
import os
import re
import time
from collections import Counter

import numpy as np

from utils.auto_web.gemini_auto import generate_gemini_content_playwright
from utils.common_utils import read_file_to_str, string_to_object, time_to_ms
from utils.gemini import get_llm_content_gemini_flash_video
from utils.paddle_ocr import SubtitleOCR, analyze_and_filter_boxes
from utils.video_utils import probe_duration, get_scene, save_frames_around_timestamp


def check_logical_scene(logical_scene_info: dict, video_duration_ms: int, max_scenes, need_remove_frames) -> tuple[bool, str]:
    """
     检查 logical_scene_info 的有效性，并在检查过程中将时间字符串转换为毫秒整数（in-place-modification）。

     Args:
         logical_scene_info (dict): 包含 'new_scene_info' 和 'deleted_scene' 的字典。
                                    此字典中的时间格式将被直接修改。
         video_duration_ms (int): 视频总时长（毫秒）。

     Returns:
         tuple[bool, str]: 一个元组，第一个元素是检查结果 (True/False)，
                            第二个元素是具体的检查信息。
     """
    # 临时列表，用于存储转换后的时间信息以进行排序和连续性检查
    all_scenes_for_sorting = []

    # 待处理的场景列表（new_scene_info 和 deleted_scene）
    scene_lists_to_process = [
        logical_scene_info.get('new_scene_info', []),
        logical_scene_info.get('deleted_scene', [])
    ]
    deleted_scene = logical_scene_info.get('deleted_scene', [])
    new_scene_info = logical_scene_info.get('new_scene_info', [])

    # 检查 deleted_scene 中的场景数量，不能超过3个
    if len(deleted_scene) > 3:
        return False, "检查失败：deleted_scene 中的场景数量超过3个，可能存在误操作。"

    if need_remove_frames == 'yes':
        if len(deleted_scene) == 0:
            return False, "需要删除场景但是没有检测出待删除的场景"

    if need_remove_frames == 'no':
        if len(deleted_scene) > 0:
            return False, "不需要  删除场景但是检测出待删除的场景"

    # 检查 new_scene_info 中的场景数量，不能超过15个
    if len(new_scene_info) > max_scenes and max_scenes > 0:
        return False, f"检查失败：new_scene_info 中的场景数量超过{max_scenes}个，可能存在误操作。"
    # 1. 遍历并转换所有场景，同时进行初步检查
    for scene_list in scene_lists_to_process:
        for i, scene in enumerate(scene_list):
            try:
                start_str, end_str = scene['start'], scene['end']

                # 确保 start 和 end 都是字符串，如果已经是数字则跳过转换
                if not isinstance(start_str, str) or not isinstance(end_str, str):
                    return False, f"检查失败：场景 {i + 1} 的时间格式不正确，期望是字符串但不是。场景: {scene}"

                start_ms = time_to_ms(start_str)
                end_ms = time_to_ms(end_str)

                # --- 核心修改步骤 ---
                # 直接在原始字典上更新值为毫秒整数
                scene['start'] = start_ms
                scene['end'] = end_ms
                # --------------------

                # 要求1：start < end
                if start_ms >= end_ms:
                    return False, f"检查失败：场景 {i + 1} 的开始时间 {start_str} ({start_ms}ms) 必须小于结束时间 {end_str} ({end_ms}ms)。"

                # 要求3：在视频时长范围内
                if not (0 <= start_ms <= video_duration_ms and 0 <= end_ms <= video_duration_ms + 2000):
                    return False, f"检查失败：场景 {i + 1} 的时间范围 [{start_str}, {end_str}] 超出视频时长 [0, {video_duration_ms}ms]。"

                # 将信息存入临时列表，用于后续排序和检查
                all_scenes_for_sorting.append({
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'original_start': start_str,  # 保留原始字符串用于错误报告
                    'original_end': end_str,
                })

            except (ValueError, TypeError) as e:
                return False, f"检查失败：场景 {i + 1} 的时间格式无效。原始场景: {scene}, 错误: {e}"

    # 如果视频时长为0，且没有场景，这是有效情况
    if not all_scenes_for_sorting and video_duration_ms == 0:
        return True, "OK. 视频时长为0，且没有场景。"

    if not all_scenes_for_sorting:
        return False, "检查失败：未提供任何场景信息，但视频时长大于0。"

    # 2. 按开始时间排序，为连续性检查做准备
    all_scenes_for_sorting.sort(key=lambda x: x['start_ms'])

    # 3. 检查时间轴的完整性
    if all_scenes_for_sorting[0]['start_ms'] != 0:
        return False, f"检查失败：时间轴不连续。第一个场景从 {all_scenes_for_sorting[0]['original_start']} 开始，而不是从 00:00.000 开始。"

    if abs(all_scenes_for_sorting[-1]['end_ms'] - video_duration_ms) > 2000:
        return False, f"检查失败：时间轴不完整。最后一个场景在 {all_scenes_for_sorting[-1]['original_end']} ({all_scenes_for_sorting[-1]['end_ms']}ms) 结束，与视频总时长 {video_duration_ms}ms 不匹配。"

    # 4. 遍历排序后的场景，检查重叠和间隔
    for i in range(len(all_scenes_for_sorting) - 1):
        current = all_scenes_for_sorting[i]
        next_s = all_scenes_for_sorting[i + 1]

        # 要求2：不能重叠
        if current['end_ms'] > next_s['start_ms']:
            return False, (f"检查失败：场景之间存在重叠。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 重叠。")

        # 要求4：不能有间隔
        if current['end_ms'] < next_s['start_ms']:
            return False, (f"检查失败：场景之间存在间隔。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 之后与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 之前有时间空缺。")

    # 为logical_scene_info增加一个字段，表示scene_number
    scene_number = 1
    for scene_list in scene_lists_to_process:
        for scene in scene_list:
            scene['scene_number'] = scene_number
            scene_number += 1

    return True, "检查并转换成功：所有场景的时间有效、连续且无重叠，格式已更新为毫秒。"

def gen_base_prompt(video_path, video_info):
    """
    生成基础的通用提示词
    """
    duration = probe_duration(video_path)
    video_title = video_info.get('base_info', {}).get('video_title', '')

    base_prompt = f"\n视频相关信息如下:\n视频时长为: {duration}"
    if video_title:
        base_prompt += f"\n视频描述为: {video_title}"
    # if comment_list:
    #     base_prompt += f"\n视频已有评论列表 (数字表示已获赞数量): {comment_list}"
    return base_prompt


def is_box_valid(sub_box, final_box_coords):
    """
    判断字幕框的中心点是否在 final_box 范围内
    """
    # 计算字幕框中心点
    cx = sum(p[0] for p in sub_box) / 4
    cy = sum(p[1] for p in sub_box) / 4

    min_x, max_x, min_y, max_y = final_box_coords

    # 稍微放宽一点边界容错 (可选)
    margin = 5
    return (min_x - margin <= cx <= max_x + margin) and \
        (min_y - margin <= cy <= max_y + margin)


def gen_precise_scene_timestamp_by_subtitle(video_path, timestamp):
    """
    通过字幕生成更精确的场景时间戳
    :param video_path: 视频路径
    :param timestamp: 初始时间戳 (单位: ms)
    :return: 精确后的时间戳 (单位: ms)
    """
    dir_path = os.path.dirname(video_path)
    output_dir = os.path.join(dir_path, "temp_scene_timestamp_frames")

    # 1. 保存关键帧 (假设这个函数现在使用 ms 单位保存文件名)
    image_path_list = save_frames_around_timestamp(video_path, timestamp / 1000, 5, output_dir)

    # 2. 运行OCR
    ocr = SubtitleOCR(use_gpu=True)
    result_json = ocr.run_batch(image_path_list)

    # 提取所有原始框用于计算范围
    detected_boxes = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]

    if not detected_boxes:
        print("未找到任何字幕框。")
        return None

    # --- 阶段 3: 分析并计算最终包围框 ---
    print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)
    if not good_boxes:
        print("\n[结果] 所有检测到的框都被过滤为异常值。")
        return None

    all_points = np.array([point for box in good_boxes for point in box])

    # 计算最终包围盒的极值
    min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
    max_x, max_y = np.max(all_points[:, 0]), np.max(all_points[:, 1])

    final_box_coords = (min_x, max_x, min_y, max_y)

    # 可视化打印一下 final_box (min_x, min_y, max_x, max_y)
    print(f"[阶段 3] 最终有效字幕区域 (x: {min_x}~{max_x}, y: {min_y}~{max_y})")

    # --- 阶段 4: 数据清洗与字典构建 ---
    print("\n[阶段 4] 生成 {时间戳: 文本} 映射...")

    timestamp_text_map = {}

    for item in result_json.get('data', []):
        file_path = item.get('file_path', '')

        # 正则解析文件名中的时间戳 (frame_33.png -> 33)
        match = re.search(r'frame_(\d+)\.png', file_path)
        if not match:
            continue

        current_ms = int(match.group(1))  # 直接作为毫秒时间戳

        # 筛选落在 final_box 内的字幕
        valid_texts = []
        for sub in item.get('subtitles', []):
            if is_box_valid(sub['box'], final_box_coords):
                valid_texts.append(sub['text'])

        # 存入字典
        if valid_texts:
            timestamp_text_map[current_ms] = "".join(valid_texts)
        else:
            timestamp_text_map[current_ms] = ""  # 该时刻指定区域无字幕

    # 按时间戳排序
    sorted_timestamps = sorted(timestamp_text_map.keys())
    if not sorted_timestamps:
        print("警告：在指定区域内未提取到有效文本。")
        return timestamp

    print(f"解析到的时间序列 (ms): {sorted_timestamps}")
    # print(f"内容映射: {timestamp_text_map}")

    # --- 阶段 5: 前后寻找文本变化点 ---

    # 找到距离用户输入 timestamp 最近的那个时间点作为锚点
    closest_ms = min(sorted_timestamps, key=lambda x: abs(x - timestamp))
    anchor_text = timestamp_text_map[closest_ms]

    print(f"锚点时间: {closest_ms}ms, 锚点文本: '{anchor_text}'")

    if not anchor_text:
        print("警告：中心时间点附近无字幕文本，无法进行精确定位，返回原始时间。")
        return timestamp

    # 向前倒推：寻找这段文本最早出现的时间
    start_ms = closest_ms
    idx = sorted_timestamps.index(closest_ms)

    for i in range(idx, -1, -1):
        curr_ms = sorted_timestamps[i]
        text = timestamp_text_map[curr_ms]

        # 判断文本是否一致 (包含或相等)
        if text == anchor_text or (len(text) > 0 and text in anchor_text):
            start_ms = curr_ms
        else:
            # 文本变了（变成空，或者变成了上一句话），停止
            break

    print(f"定位到字幕开始时间: {start_ms}ms")

    # 如果需要还可以做一个简单的向后查找(find end)，这里只按需求找了“更精确的时间戳(通常指开始)”

    return start_ms




def fix_logical_scene_info(video_path, merged_timestamps, logical_scene_info, max_delta_ms=1000):
    """
     将 logical_scene_info 中的每个 scene 的 start/end 对齐到 camera shot。
     对齐原则：
     1. 寻找 max_delta_ms 毫秒容差范围内的所有 camera shot。
     2. 筛选出最佳匹配（优先高频次，其次近距离）。
     3. 如果最佳匹配的 camera shot 次数 (count) < 2，则强制调用 gen_precise_scene_timestamp_by_subtitle 进行字幕对齐。
     4. 否则使用 camera shot 的时间。

     Args:
         merged_timestamps (list): camera shot 信息，每个元素是 [timestamp_ms, count]。
         logical_scene_info (dict): 包含 'new_scene_info' 的逻辑场景信息。
         max_delta_ms (int): 查找候选 camera shot 的最大时间差（毫秒）。
     """

    camera_shots_with_counts = [c for c in merged_timestamps if c and c[0] is not None and c[1] > 0]
    if not camera_shots_with_counts:
        print("⚠️ 无有效 camera_shot 时间戳，跳过调整。")
        return logical_scene_info

    for i, scene in enumerate(logical_scene_info.get('new_scene_info', [])):
        for key in ('start', 'end'):
            orig_ts = scene.get(key)
            if orig_ts is None:
                print(f"[Scene {i}] {key}: 无法解析原始时间 ({orig_ts})，跳过。")
                continue

            # 步骤 1: 筛选候选者
            candidates = [
                shot for shot in camera_shots_with_counts
                if abs(shot[0] - orig_ts) <= max_delta_ms
            ]

            if not candidates:
                print(f"[Scene {i}] {key}: 保持不变 {orig_ts} (在 {max_delta_ms}ms 范围内无候选 camera shot)")
            else:
                # 步骤 2: 找到最佳匹配的 camera shot
                def calculate_key(shot):
                    diff = abs(shot[0] - orig_ts)
                    count = shot[1]
                    score = diff / count if count > 0 else float('inf')
                    return (score, diff)

                best_shot = min(candidates, key=calculate_key)
                count = best_shot[1]

                # 步骤 3: 决策逻辑
                if count < 10:
                    # 如果 camera shot 置信度低，使用字幕对齐
                    new_ts = gen_precise_scene_timestamp_by_subtitle(video_path, orig_ts)
                    print(f"[Scene {i}] {key}: {orig_ts} -> {new_ts} (CameraShot count={count}<2, 已通过字幕修正)")
                else:
                    # 否则使用 camera shot 对齐
                    new_ts = int(best_shot[0])
                    diff = abs(new_ts - orig_ts)
                    score = diff / count if count > 0 else float('inf')
                    print(f"[Scene {i}] {key}: {orig_ts} -> {new_ts} (最佳分={score:.2f}, 次数={count}, 差值={diff}ms, 已通过视觉修正)")

                scene[key] = new_ts

    return logical_scene_info

def gen_logical_scene_llm(video_path, video_info):
    """
    生成新的视频方案
    """
    need_remove_frames = video_info.get('extra_info', {}).get('has_ad_or_face', 'auto')

    base_prompt = gen_base_prompt(video_path, video_info)
    max_scenes = video_info.get('base_info', {}).get('max_scenes', 0)
    log_pre = f"{video_path} 逻辑性场景划分 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return "获取视频时长失败", None

    retry_delay = 10
    max_retries = 3
    prompt_file_path = './prompt/视频素材切分.txt'
    full_prompt = read_file_to_str(prompt_file_path)
    full_prompt += f'\n{base_prompt}'
    if max_scenes > 0:
        full_prompt += f'\n请将生成的场景数量控制在 {max_scenes} 个以内。'

    error_info = ""
    gen_error_info = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries}) {log_pre}")
            gen_error_info, raw = generate_gemini_content_playwright(full_prompt, file_path=video_path, model_name="gemini-2.5-pro")

            logical_scene_info = string_to_object(raw)
            check_result, check_info = check_logical_scene(logical_scene_info, video_duration_ms, max_scenes, need_remove_frames)
            if not check_result:
                error_info = f"逻辑性场景划分检查未通过: {check_info} {raw} {log_pre}"
                raise ValueError(f"逻辑性场景划分检查未通过: {check_info} {raw}")

            merged_timestamps = get_scene(video_path, min_final_scenes=max_scenes)
            logical_scene_info = fix_logical_scene_info(video_path, merged_timestamps, logical_scene_info, max_delta_ms=1000)

            return None, logical_scene_info
        except Exception as e:
            error_str = f"{error_info} {str(e)}"
            print(f"生成逻辑性场景划分失败 (尝试 {attempt}/{max_retries}): {error_str} {log_pre} {gen_error_info}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None  # 达到最大重试次数后返回 None

def check_overlays_text(optimized_video_plan, video_duration_ms):
    """
    检查优化的方案
    """

    overlays = optimized_video_plan.get('overlays', [])
    # 长度要大于2
    if len(overlays) < 2:
        return False, f"优化方案检查失败：overlays 长度必须至少为 2。当前长度为 {len(overlays)}。"
    # 每个start必须都在视频时长范围内
    for i, overlay in enumerate(overlays):
        start = overlay.get('start')
        start_ms = time_to_ms(start)
        text = overlay.get('text', '').strip()
        if len(text) > 12:
            return False, f"优化方案检查失败：第 {i + 1} 个 overlay 的文本长度过长（>=10）。文本: {text}"
        if not (0 <= start_ms <= video_duration_ms):
            return False, f"优化方案检查失败：第 {i + 1} 个 overlay 的 start 时间 {start} 超出视频时长范围 [0, {video_duration_ms}ms]。"
    return True, "优化方案检查通过。"


def gen_overlays_text_llm(video_path, video_info):
    """
    生成新的视频优化方案
    """
    log_pre = f"{video_path} 视频覆盖文字生成 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    base_prompt = gen_base_prompt(video_path, video_info)
    error_info = ""
    # --- 2. 初始化和预处理 ---
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        error_info = f"获取视频时长失败: {e} {log_pre}"
        return error_info, None

    retry_delay = 10
    max_retries = 5
    prompt_file_path = './prompt/视频质量提高生成画面文字.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}'
    full_prompt += f'\n{base_prompt}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-flash-latest"
            # model_name = "gemini-flash-latest"
            print(f"正在视频覆盖文字生成 (尝试 {attempt}/{max_retries}) {log_pre}")
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            video_overlays_text_info = string_to_object(raw)
            check_result, check_info = check_overlays_text(video_overlays_text_info, video_duration_ms)
            if not check_result:
                error_info = f"优化方案检查未通过: {check_info} {raw} {log_pre}"
                raise ValueError(error_info)
            return error_info, video_overlays_text_info
        except Exception as e:
            error_str = f"{error_info} {str(e)}"
            print(f"视频覆盖文字方案检查未通过 (尝试 {attempt}/{max_retries}): {e} {raw} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None  # 达到最大重试次数后返回 None

def check_owner_asr(owner_asr_info, video_duration):
    """
        检查生成的asr文本是否正确，第一是验证每个时间是否合理（1.最长跨度不能够超过20s 2.时长的合理性（也就是最快和最慢的语速就能够知道文本对应的时长是否合理） 3.owner语音和本地speaker说话人日志的差异不能够太大）

    :param owner_asr_info: 包含 ASR 信息的字典列表
    :return: 错误信息列表，若没有错误则返回空列表
    """
    max_end_time_ms = 0
    error_info = 'asr文本检查通过'
    # 使用 enumerate 获取索引和元素，便于日志记录
    for i, segment in enumerate(owner_asr_info):
        try:
            start_str = segment.get("start")
            end_str = segment.get("end")

            # 检查 start 和 end 是否为字符串，如果不是，则格式错误
            if not isinstance(start_str, str) or not isinstance(end_str, str):
                error_info = f"[ERROR] 片段 {i} 的时间格式不正确，应为字符串。数据: {segment}"
                return False, error_info

            start_time_ms = time_to_ms(start_str)
            end_time_ms = time_to_ms(end_str)

            # --- 核心修改步骤：原地更新字典 ---
            segment["start"] = start_time_ms
            segment["end"] = end_time_ms
            # ------------------------------------

            # 更新整个 ASR 列表的最大结束时间
            max_end_time_ms = max(max_end_time_ms, end_time_ms)

            duration_ms = end_time_ms - start_time_ms

            # 1. 最大文案长度不能超过 20s
            if len(owner_asr_info[i]['final_text']) > 200 and owner_asr_info[i]['speaker'] == 'owner':
                error_info = f"[ERROR] 片段 {i} 文案长度：{len(owner_asr_info[i]['final_text'])} 跨度过长: {duration_ms} ms 文案为:{owner_asr_info[i]['final_text']}"
                return False, error_info

        except (ValueError, TypeError) as e:
            error_info = f"[ERROR] 处理片段 {i} 时发生时间转换错误: {e}. 数据: {segment}"
            return False, error_info

    # 循环结束后，检查 ASR 的最大时间是否超过视频总时长（允许1秒的误差）
    if max_end_time_ms > video_duration + 1000:
        error_info = f"[ERROR] ASR 最大结束时间 {max_end_time_ms} ms 超过视频总时长 {video_duration} ms"
        return False, error_info

    # 为owner_asr_info增加source_clip_id字段，从1开始
    source_clip_id = 0
    for segment in owner_asr_info:
        source_clip_id += 1
        segment['source_clip_id'] = source_clip_id

    return True, error_info


def check_video_script(video_script_info, final_scene_info, allow_commentary, is_need_narration):
    """
    检查 video_script_info 列表中的每个方案是否符合预设的规则。

    Args:
        video_script_info (list): 包含一个或多个视频脚本方案的列表。
        final_scene_info (dict): 包含有效场景ID列表等信息的字典。
                                示例: {'all_scenes': [{'scene_id': 'id1'}, ...]}
        allow_commentary (bool): 是否允许/包含解说词。如果为True，需要检查 transition_text。

    Returns:
        tuple: (bool, str)
               第一个元素为布尔值，True表示检查通过，False表示检查失败。
               第二个元素为字符串，检查通过时为空字符串，失败时为具体的错误信息。
    """
    try:
        all_scene_list = final_scene_info.get('all_scenes', [])
        all_scene_dict = {}
        for scene in all_scene_list:
            scene_id = scene.get('scene_id')
            all_scene_dict[scene_id] = scene

        # 0. 预处理和基本结构检查
        if not isinstance(video_script_info, list):
            return False, "输入的数据 'video_script_info' 不是一个列表。"

        if 'all_scenes' not in final_scene_info or not isinstance(final_scene_info['all_scenes'], list):
            return False, "输入的数据 'final_info_list' 格式错误，缺少 'all_scenes' 列表。"

        # 提取final_info_list中的所有scene_id，并放入set中以提高查询效率
        valid_scene_ids = {scene['scene_id'] for scene in final_scene_info['all_scenes']}

        # 遍历每个方案进行检查
        for i, solution in enumerate(video_script_info):
            solution_num = i + 1  # 方案编号，从1开始

            if not isinstance(solution, dict):
                return False, f"方案 {solution_num} 的数据格式不是一个字典。"

            # 1. 检查每个方案的 title, cover_text, video_abstract, 方案整体评分, 场景顺序与新文案 字段不能为空。
            required_fields = ['title', 'cover_text', 'video_abstract', '方案整体评分', '场景顺序与新文案']
            for field in required_fields:
                value = solution.get(field)
                if value is None:
                    return False, f"方案 {solution_num} 缺少必要字段: '{field}'。"
                # 检查字符串或列表是否为空
                if isinstance(value, (str, list)) and not value:
                    return False, f"方案 {solution_num} 的字段 '{field}' 的值不能为空。"

            # 2. 检查 方案整体评分 字段必须是数字且在0到10之间。
            score = solution.get('方案整体评分')
            if not isinstance(score, (int, float)):
                return False, f"方案 {solution_num} 的 '方案整体评分' ({score}) 不是数字类型。"
            if not (0 <= score <= 10):
                return False, f"方案 {solution_num} 的 '方案整体评分' ({score}) 不在 0 到 10 的范围内。"

            # 开始检查 '场景顺序与新文案' 内部的细节
            scenes = solution.get('场景顺序与新文案', [])
            if not isinstance(scenes, list):
                return False, f"方案 {solution_num} 的 '场景顺序与新文案' 不是一个列表。"

            seen_scene_ids = set()
            expected_scene_number = 1

            for j, scene in enumerate(scenes):
                scene_num = j + 1  # 场景编号，从1开始

                if not isinstance(scene, dict):
                    return False, f"方案 {solution_num} 的场景 {scene_num} 的数据格式不是一个字典。"

                # 4. 检查 new_scene_number 是否从1开始连续增长。
                current_scene_number = scene.get('new_scene_number')
                if current_scene_number != expected_scene_number:
                    return False, f"方案 {solution_num} 的 'new_scene_number' 不连续。在场景 {scene_num} 期望值为 {expected_scene_number}，实际值为 {current_scene_number}。"

                # 3. 检查 scene_id 是否有效且无重复。
                scene_id = scene.get('scene_id')
                if not scene_id:
                    return False, f"方案 {solution_num} 的场景 {scene_num} 缺少 'scene_id' 或其值为空。"
                if scene_id not in valid_scene_ids:
                    return False, f"方案 {solution_num} 的场景 {scene_num} 中，scene_id '{scene_id}' 是无效的，不存在于 valid_scene_ids 列表中。"
                if scene_id in seen_scene_ids:
                    return False, f"方案 {solution_num} 中存在重复的 scene_id: '{scene_id}'。"

                # 5. 检查 on_screen_text 字段是否存在。
                if 'on_screen_text' not in scene:  # 检查key是否存在，允许其值为空字符串
                    return False, f"方案 {solution_num} 的场景 {scene_num} 缺少 'on_screen_text' 字段。"

                # =================== 新增修改开始 ===================
                # 6. 如果 allow_commentary 为 True，检查 transition_text 字段是否存在
                if allow_commentary:
                    if 'transition_text' not in scene:
                        return False, f"方案 {solution_num} 的场景 {scene_num} 缺少 'transition_text' 字段 (因为 allow_commentary=True)。"

                if is_need_narration:
                    scene_info = all_scene_dict.get(scene_id, {})
                    narration_script_list = scene_info.get('narration_script_list', [])
                    new_narration_script_list = scene.get('new_narration_script', [])
                    # 检查 narration_script_list 和 new_narration_script_list 的长度是否一致
                    if len(narration_script_list) != len(new_narration_script_list):
                        return False, f"方案 {solution_num} 的场景 {scene_num}  {scene_id} 中，narration_script_list 长度 ({len(narration_script_list)}) 与 new_narration_script 长度 ({len(new_narration_script_list)}) 不一致。"


                # 更新检查状态
                seen_scene_ids.add(scene_id)
                expected_scene_number += 1

        # 如果所有检查都通过
        return True, ""

    except (KeyError, TypeError, AttributeError) as e:
        # 捕获可能的数据结构错误，确保程序不崩溃
        error_info = f"处理数据时发生结构性错误，请检查输入格式是否正确。错误详情: {type(e).__name__}: {e}"
        return False, error_info


def gen_owner_asr_by_llm(video_path, video_info):
    """
    通过大模型生成带说话人识别的ASR文本。
    """
    log_pre = f"{video_path} owner asr 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    base_prompt = gen_base_prompt(video_path, video_info)
    error_info = ""
    gen_error_info = ""
    # --- 1. 配置常量 ---
    max_retries = 3
    retry_delay = 10  # 秒
    PROMPT_FILE_PATH = './prompt/视频分解素材_直接进行asr转录与owner识别严格.txt'

    # --- 2. 初始化和预处理 ---
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        error_info = f"获取视频时长失败: {e} {log_pre}"
        return error_info, None
    # --- 4. 准备Prompt ---
    try:
        prompt = read_file_to_str(PROMPT_FILE_PATH)
    except Exception as e:
        error_info = f"读取Prompt文件失败: {PROMPT_FILE_PATH}, 错误: {e} {log_pre}"
        print(error_info)
        return error_info, None
    prompt = f"{prompt}{base_prompt}"
    # --- 5. 带重试机制的核心逻辑 ---
    for attempt in range(1, max_retries + 1):
        print(f"尝试生成ASR信息... (第 {attempt}/{max_retries} 次) {log_pre}")
        raw_response = ""
        try:
            gen_error_info, raw_response = generate_gemini_content_playwright(prompt, file_path=video_path, model_name="gemini-2.5-pro")


            # 解析和校验
            owner_asr_info = string_to_object(raw_response)
            check_result, check_info = check_owner_asr(owner_asr_info, video_duration_ms)
            if not check_result:
                error_info = f"asr 检查未通过: {check_info} {raw_response} {log_pre}"
                raise ValueError(error_info)
            return error_info, owner_asr_info
        except Exception as e:
            error_str = f"{error_info} {str(e)} {gen_error_info}"
            print(f"asr 生成 未通过 (尝试 {attempt}/{max_retries}): {e} {raw_response} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None  # 达到最大重试次数后返回 None

def validate_danmu_result(result: any):
    """
    检测LLM返回结果的正确性。

    Args:
        result: 从LLM返回并经过解析后的对象。

    Returns:
        bool: 如果结果正确则返回 True，否则返回 False。
    """
    error_info = ""
    if not isinstance(result, dict):
        error_info = f"验证失败：结果不是一个字典 (dict)，实际类型为 {type(result)}。 {result}"
        return False, error_info

    if "视频分析" not in result:
        error_info = f"验证失败：结果字典中缺少 '视频分析' 字段。{result}"
        return False, error_info

    video_analysis = result.get("视频分析")

    if not isinstance(video_analysis, dict):
        error_info = (f"验证失败：'视频分析' 字段不是字典：{video_analysis}")
        return False, error_info

    # 检查题材字段必须存在并非空
    genre = video_analysis.get("题材")
    if not genre:
        error_info = (f"验证失败：'视频分析' 下缺少 '题材' 字段，或其值为空。当前值：{genre}")
        return False, error_info

    return True, error_info



def gen_hudong_by_llm(video_path, video_info):
    """
    通过视频和描述生成弹幕，带有重试和验证机制。
    """
    MAX_RETRIES = 3  # 设置最大重试次数
    prompt_file_path = './prompt/筛选出合适的弹幕.txt'
    base_prompt = gen_base_prompt(video_path, video_info)
    log_pre = f"{video_path} 生成弹幕互动信息 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    try:
        prompt = read_file_to_str(prompt_file_path)
        duration = probe_duration(video_path)
    except Exception as e:
        print(f"初始化prompt或获取视频时长时出错: {e} {log_pre}")
        return None

    prompt_with_duration = f"{prompt}{base_prompt}"
    comment_list = video_info.get('base_info', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    desc = f"\n已有评论列表 (数字表示已获赞数量): {temp_comments}"
    # 模型选择逻辑（与原版保持一致）
    max_duration = 600
    model_name = "gemini-flash-latest"
    if duration > max_duration:
        # 即使超过时长，模型名也没变，但保留打印语句
        print(f"视频时长 {duration} 秒超过最大限制 {max_duration} 秒，使用默认处理方式。  {log_pre}")
    error_info = ""
    # 开始重试循环
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n--- [第 {attempt}/{MAX_RETRIES} 次尝试] ---  {log_pre}")

        # 策略：首次尝试带 desc，后续重试不带 desc
        if attempt == 1:
            current_prompt = f"{prompt_with_duration}\n{desc}"
            print(f"生成弹幕互动信息 首次尝试：使用包含 `desc` 的完整 prompt。 {log_pre}")
        else:
            current_prompt = prompt_with_duration
            print(f"生成弹幕互动信息 重试尝试：使用不包含 `desc` 的基础 prompt。 {log_pre}")

        try:
            # 1. 调用 LLM 获取原始文本
            raw = get_llm_content_gemini_flash_video(
                prompt=current_prompt,
                video_path=video_path,
                model_name=model_name
            )

            # 2. 尝试解析文本为对象
            try:
                result = string_to_object(raw)
                check_result, check_info = validate_danmu_result(result)
                if not check_result:
                    raise ValueError(f"生成弹幕互动信息 结果验证未通过: {check_info} {raw} {log_pre}")
                return error_info, result
            except Exception as e:
                error_info = f"生成弹幕互动信息 解析返回结果时出错: {str(e)}"
                print(f"生成弹幕互动信息 解析返回结果时出错: {str(e)}")
                return error_info, None

        except Exception as e:
            error_info = f"生成弹幕互动信息 解析返回结果时出错: {str(e)}"
            print(f"生成弹幕互动信息 在第 {attempt} 次调用 LLM API 时发生严重错误: {e}")
            # 如果API调用本身就失败了，也计为一次失败的尝试
            if 'PROHIBITED_CONTENT' in str(e): # <--- 修复在这里
                print("生成弹幕互动信息 遇到内容禁止错误，停止重试。")
                break  # 使用 break 更清晰地跳出循环
            return error_info, None


def analyze_scene_content(scene_list, owner_asr_info, top_k=3, merge_mode='global'):
    """
    分析场景列表。
    """

    if not scene_list:
        return []

    # --- 1. 内部辅助函数：处理一组场景 ---
    # 这部分函数是修改的重点
    def _process_segment(segment_scenes):
        scene_summaries = []
        all_tags = []
        scene_number_list = []

        if not segment_scenes:
            return {}

        # 1. 确定当前场景段落的整体时间范围
        segment_start = min(s['start'] for s in segment_scenes)
        segment_end = max(s['end'] for s in segment_scenes)

        original_script_list = []
        narration_script_list = []

        # 2. 遍历 ASR，使用中心点判定归属
        if owner_asr_info:
            for asr_item in owner_asr_info:
                asr_start = asr_item.get('start', 0)
                asr_end = asr_item.get('end', 0)
                asr_mid = (asr_start + asr_end) / 2
                if segment_start <= asr_mid < segment_end:
                    text = asr_item.get('final_text', '')
                    speaker = asr_item.get('speaker', '')
                    if speaker == 'owner':
                        narration_script_list.append(text)
                    else:
                        original_script_list.append(text)

        # 遍历 segment_scenes 来提取所需信息
        for scene in segment_scenes:
            # 提取 visual_description
            v_desc = scene.get('scene_summary')
            if v_desc:
                scene_summaries.append(v_desc)

            # 提取 tags
            tags = scene.get('tags', [])
            if tags:
                all_tags.extend(tags)

            # --- 新增代码 Start ---
            # 提取 scene_number_list
            # 使用 .get() 方法以防该字段不存在
            origin_id = scene.get('scene_number')
            if origin_id is not None:  # 确保ID存在才添加
                scene_number_list.append(origin_id)
            # --- 新增代码 End ---

        # 计数并取 Top K
        top_all_tags = [tag for tag, count in Counter(all_tags).most_common(top_k)]

        # 在返回的字典中增加 origin_scene_id_list 字段
        return {
            'scene_summary': scene_summaries,
            'scene_number_list': scene_number_list,
            'tags': top_all_tags,
            'original_script_list': original_script_list,
            'narration_script_list': narration_script_list
        }

    # --- 2. 根据模式构建 Segments ---
    # 这部分逻辑保持不变
    segments = []

    if merge_mode == 'global':
        segments.append(scene_list)

    elif merge_mode == 'none':
        for scene in scene_list:
            segments.append([scene])

    elif merge_mode == 'smart':
        if not scene_list:
            return []

        buffer = []
        for scene in scene_list:
            if not buffer:
                buffer.append(scene)
                continue

            is_adjustable = scene.get('sequence_info', {}).get('is_adjustable', False)

            if is_adjustable:
                segments.append(buffer)
                buffer = [scene]
            else:
                buffer.append(scene)

        if buffer:
            segments.append(buffer)

    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")

    # --- 3. 处理结果 ---
    # 这部分逻辑保持不变
    result_list = []
    for segment in segments:
        result_list.append(_process_segment(segment))

    return result_list



def is_contain_owner_speaker(owner_asr_info):
    """
    检查是否包含owner的文本
    """
    for asr_info in owner_asr_info:
        speaker = asr_info.get('speaker', 'unknown')
        final_text = asr_info.get('final_text', '').strip()
        if speaker == 'owner' and final_text:
            return True
    return False


def build_prompt_data(task_info, video_info_dict):
    """
    组织好最终的数据，不同的选项有不同的组织方式
    :param task_info:
    :param video_info_dict:
    :return:
    """
    creation_guidance_info = task_info.get('creation_guidance_info', {})
    creative_guidance = creation_guidance_info.get('creative_guidance', '')
    material_usage_mode = creation_guidance_info.get('retention_ratio', 'free')
    is_need_narration = creation_guidance_info.get('is_need_audio_replace', False)
    video_summary_info = {}
    all_scene_info_list = []



    for video_id, video_info in video_info_dict.items():
        max_scenes = video_info.get('base_info', {}).get('max_scenes', 0)
        owner_asr_info = video_info.get('owner_asr_info', {})
        if max_scenes == 1: # 如果不需要原创就应该全量保留而且不能够改变顺序
            merge_mode = 'global'
        else:
            if is_need_narration and is_contain_owner_speaker(owner_asr_info):
                merge_mode = 'none'
            else:
                merge_mode = 'smart'

        logical_scene_info = video_info.get('logical_scene_info')
        video_summary = logical_scene_info.get('video_summary', '')
        tags = logical_scene_info.get('tags', '')
        video_summary_info[video_id] ={
                "source_video_id": video_id,
                "summary": video_summary,
                "tags":tags
            }
        new_scene_info = logical_scene_info.get('new_scene_info', [])

        # 获取new_scene_info每个元素的visual_description，放入一个列表中
        merged_scene_list = analyze_scene_content(new_scene_info, owner_asr_info, merge_mode=merge_mode)
        counted_scene = 0
        for scene in merged_scene_list:
            counted_scene += 1
            scene['scene_id'] = f"{video_id}_part{counted_scene}"
            scene['source_video_id'] = video_id
        all_scene_info_list.extend(merged_scene_list)

    final_info = {
        "creative_guidance": creative_guidance,
        "material_usage_mode": material_usage_mode,
        "video_summaries": video_summary_info,
        "all_scenes": all_scene_info_list

    }
    return final_info


def gen_video_script_llm(task_info, video_info_dict):
    """
    生成新的脚本
    :param task_info:
    :param video_info_dict:
    :return:
    """
    creation_guidance_info = task_info.get('creation_guidance_info', {})
    final_info_list = build_prompt_data(task_info, video_info_dict)
    origin_final_scene_info = copy.deepcopy(final_info_list)
    # 剔除all_scenes中的start和end字段
    for scene in final_info_list.get('all_scenes', []):
        if 'start' in scene:
            del scene['start']
        if 'end' in scene:
            del scene['end']
        if 'scene_number_list' in scene:
            del scene['scene_number_list']

    print("生成场景的最终数据成功")

    prompt_path = './prompt/多素材视频生成不加过度.txt'
    allow_commentary = creation_guidance_info.get('creation_guidance_info', {}).get('is_need_commentary', False)
    is_need_narration = creation_guidance_info.get('is_need_audio_replace', False)

    if allow_commentary == True and is_need_narration == True:
        prompt_path = './prompt/多素材视频生成.txt'

    if allow_commentary == True and is_need_narration == False:
        prompt_path = './prompt/多素材视频生成非替换语音.txt'

    if allow_commentary == False and is_need_narration == False:
        prompt_path = './prompt/多素材视频生成非替换语音不加过度.txt'

    if allow_commentary == False and is_need_narration == True:
        prompt_path = './prompt/多素材视频生成不加过度.txt'

    full_prompt = read_file_to_str(prompt_path)

    full_prompt = f"{full_prompt}\n下面是相应的视频场景数据：\n{final_info_list}"
    max_retries = 3
    log_pre = f"多素材视频生成脚本 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    print("生成脚本的最终prompt成功")
    retry_delay = 10
    for attempt in range(1, max_retries + 1):
        print(f"尝试生成新视频脚本信息... (第 {attempt}/{max_retries} 次) {log_pre}")
        raw_response = ""
        error_info = ""
        try:
            gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name="gemini-2.5-pro")


            # 解析和校验
            video_script_info = string_to_object(raw_response)
            check_result, check_info = check_video_script(video_script_info, final_info_list, allow_commentary, is_need_narration)
            if not check_result:
                error_info = f"新视频脚本 检查未通过: {check_info} {raw_response} {log_pre}"
                raise ValueError(error_info)
            return error_info, video_script_info, origin_final_scene_info
        except Exception as e:
            error_str = f"{error_info} {str(e)} {gen_error_info}"
            print(f"asr 生成 未通过 (尝试 {attempt}/{max_retries}): {e} {raw_response} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None, None  # 达到最大重试次数后返回 None







if __name__ == '__main__':
    video_path = r"W:\project\python_project\auto_video\videos\material\7586639820693179690\7586639820693179690_static_cut.mp4"
    merged_timestamps = get_scene(video_path)






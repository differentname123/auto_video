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
import random
import re
import time
import traceback
from collections import Counter
import pypinyin
from utils.paddle_ocr_base import run_subtitle_ocr

# 定义全局信号量，限制 fix_logical_scene_info 的最大并发数为 3
import numpy as np

from application.video_common_config import correct_owner_timestamps, build_video_paths, is_contain_owner_speaker, \
    analyze_scene_content
from utils.auto_web.gemini_auto import generate_gemini_content_playwright
from utils.bilibili.find_paid_topics import get_all_paid_topics
from utils.common_utils import read_file_to_str, string_to_object, time_to_ms, ms_to_time, get_top_comments, read_json, \
    safe_process_limit, simple_cipher
from utils.gemini import get_llm_content_gemini_flash_video, get_llm_content
from utils.gemini_web import generate_gemini_content_managed
from utils.paddle_ocr import analyze_and_filter_boxes
from utils.video_utils import probe_duration, get_scene, \
    save_frames_around_timestamp_ffmpeg


def check_logical_scene(logical_scene_info: dict, video_duration_ms: int, max_scenes, need_remove_frames,
                        split_time_ms_points) -> tuple[bool, str]:
    """
     检查 logical_scene_info 的有效性，并在检查过程中将时间字符串转换为毫秒整数（in-place-modification）。

     Args:
         logical_scene_info (dict): 包含 'new_scene_info' 和 'deleted_scene' 的字典。
                                    此字典中的时间格式将被直接修改。
         video_duration_ms (int): 视频总时长（毫秒）。
         max_scenes (int): 允许的最大场景数量。
         need_remove_frames (str): 是否需要删除帧 ('yes'/'no')。
         split_time_ms_points (list): 关键分割点时间戳列表（毫秒）。

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
                traceback.print_exc()

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

    # 5. [新增] 检查 split_time_ms_points 中的时间戳是否在场景分割点附近
    if split_time_ms_points:
        # 提取当前所有逻辑场景的内部分割点（即每个场景的结束时间，排除视频本身的结束时间）
        # 此时场景已排序且连续，current['end_ms'] 即为分割点
        logical_split_points = [s['end_ms'] for s in all_scenes_for_sorting[:-1]]

        for required_split in split_time_ms_points:
            # 检查 required_split 是否在任意一个 logical_split_point 的 ±1000ms 范围内
            found_match = False
            for logical_pt in logical_split_points:
                if abs(required_split - logical_pt) <= 1000:
                    found_match = True
                    break

            if not found_match:
                return False, f"检查失败：在 split_time_ms_points 中的时间点 {required_split}ms 附近（±1000ms）未找到对应的场景分割点。"

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
    temp_comment = [(c[0], c[1]) for c in video_info.get('comment_list')][:10]
    base_prompt = f"\n视频相关信息如下:\n视频时长为: {duration}"
    if video_title:
        base_prompt += f"\n视频描述为: {video_title}"
    # if comment_list:
    #     base_prompt += f"\n视频已有评论列表 (数字表示已获赞数量): {comment_list}"
    return base_prompt


def get_best_valid_text(subtitles, final_box_coords, margin=5):
    """
    从字幕列表中找出唯一一个最符合条件（在范围内且离中心最近）的字幕文本。
    如果没有符合条件的，返回 None。
    """
    if not subtitles:
        return None

    min_x, max_x, min_y, max_y = final_box_coords

    # 1. 计算 final_box 的几何中心
    target_cx = (min_x + max_x) / 2
    target_cy = (min_y + max_y) / 2

    best_text = None
    min_dist_sq = float('inf')  # 初始化最小距离平方为无穷大

    for sub in subtitles:
        sub_box = sub['box']
        # 计算字幕框中心点
        cx = sum(p[0] for p in sub_box) / 4
        cy = sum(p[1] for p in sub_box) / 4

        # 2. 判断是否 valid (在范围内)
        if (min_x - margin <= cx <= max_x + margin) and \
                (min_y - margin <= cy <= max_y + margin):

            # 3. 计算离中心点的距离平方
            dist_sq = (cx - target_cx) ** 2 + (cy - target_cy) ** 2

            # 4. 擂台法：保留距离最小的那个
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_text = sub['text']

    return best_text


def calculate_closest_cut_point(timestamp_text_map, anchor_timestamp):
    """
    根据字幕内容图，修复缺失数据，并找到距离 anchor_timestamp 最近的字幕跳变点。
    跳变点定义为：内容发生变化的上一个时间戳。
    """
    sorted_timestamps = sorted(timestamp_text_map.keys())
    if not sorted_timestamps:
        return anchor_timestamp

    # --- 步骤 1: 修复遗漏掉的文字 (逻辑 3) ---
    # 我们创建一个副本以免影响原始数据（如果需要保留原始数据的话）
    # 这里直接在逻辑中处理，生成一个 cleaned_map
    cleaned_map = timestamp_text_map.copy()

    # 遍历列表（排除首尾，因为需要前后对比）
    for i in range(1, len(sorted_timestamps) - 1):
        prev_t = sorted_timestamps[i - 1]
        curr_t = sorted_timestamps[i]
        next_t = sorted_timestamps[i + 1]

        prev_text = cleaned_map[prev_t]
        curr_text = cleaned_map[curr_t]
        next_text = cleaned_map[next_t]

        # 如果当前为空，但前后一致且不为空，则修复
        if curr_text == "" and prev_text == next_text and prev_text != "":
            cleaned_map[curr_t] = prev_text

    # --- 步骤 2: 寻找所有的跳变点 (逻辑 4) ---
    # 跳变点 candidates 列表
    jump_candidates = []

    # 遍历直到倒数第二个，比较 i 和 i+1
    for i in range(len(sorted_timestamps) - 1):
        curr_t = sorted_timestamps[i]
        next_t = sorted_timestamps[i + 1]

        curr_text = cleaned_map[curr_t]
        next_text = cleaned_map[next_t]

        # 简单的文本不相等判断，也可以根据需要加 fuzzy matching
        if curr_text != next_text:
            # 记录跳变的上一个时间点
            jump_candidates.append(curr_t)

    # 如果没有发现任何跳变（全程文字一样），返回 anchor 或 序列起点
    if not jump_candidates:
        print("未检测到字幕内容变化，返回原始锚点。")
        return anchor_timestamp

    # --- 步骤 3: 找到距离锚点最近的跳变点 (逻辑 1 & 2) ---
    # 使用 min 函数，key 为与 anchor_timestamp 的绝对距离
    closest_point = min(jump_candidates, key=lambda t: abs(t - anchor_timestamp))
    return closest_point


def gen_precise_scene_timestamp_by_subtitle(video_path, timestamp):
    """
    通过字幕生成更精确的场景时间戳
    :param video_path: 视频路径
    :param timestamp: 初始时间戳 (单位: ms)
    :return: 精确后的时间戳 (单位: ms)
    """
    # 【修改点 1】在函数最外层加入 try 块，包裹所有逻辑
    try:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f'{video_filename}_scenes')
        # 1. 保存关键帧 (涉及IO，易报错)
        image_path_list = save_frames_around_timestamp_ffmpeg(video_path, timestamp / 1000, 30, output_dir, time_duration_s=1)

        result_json = run_subtitle_ocr(image_path_list, only_best=False)

        # 提取所有原始框用于计算范围
        detected_boxes = [sub.get("box", []) for item in result_json.get("data", []) for sub in
                          item.get("subtitles", [])]

        if not detected_boxes:
            print("未找到任何字幕框。")
            return timestamp

        # --- 阶段 3: 分析并计算最终包围框 ---
        # print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
        good_boxes = analyze_and_filter_boxes(detected_boxes)
        if not good_boxes:
            print("\n[结果] 所有检测到的框都被过滤为异常值。")
            return timestamp

        all_points = np.array([point for box in good_boxes for point in box])
        min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
        max_x, max_y = np.max(all_points[:, 0]), np.max(all_points[:, 1])
        final_box_coords = (min_x, max_x, min_y, max_y)

        # print(f"[阶段 3] 最终有效字幕区域 (x: {min_x}~{max_x}, y: {min_y}~{max_y})")

        # --- 阶段 4: 生成 {时间戳: 文本} 映射 ---
        # print("\n[阶段 4] 生成 {时间戳: 文本} 映射...")
        timestamp_text_map = {}

        for item in result_json.get('data', []):
            file_path = item.get('file_path', '')
            match = re.search(r'frame_(\d+)\.png', file_path)
            if not match:
                continue
            current_ms = int(match.group(1))

            best_text = get_best_valid_text(item.get('subtitles', []), final_box_coords)
            # 构造 valid_texts 列表：如果有结果就是 [text]，没有就是 []
            valid_texts = [best_text] if best_text else []

            # 去除首尾空格，避免 OCR 带来的微小差异影响比对
            text_content = "".join(valid_texts).strip()
            timestamp_text_map[current_ms] = text_content

        if not timestamp_text_map:
            print("警告：在指定区域内未提取到有效文本。")
            return timestamp

        # --- 阶段 5: 调用独立函数计算最终时间点 ---
        # print(f"\n[阶段 5] 计算最近的字幕切分点...字幕长度为：{len(timestamp_text_map)}")

        # 计算逻辑也可能出错，放在 try 块中很安全
        final_timestamp = calculate_closest_cut_point(timestamp_text_map, timestamp)

        # print(f"初始时间: {timestamp}ms -> 精确时间: {final_timestamp}ms")

        return final_timestamp

    # 【修改点 2】捕获所有异常，打印日志并强制返回原始 timestamp
    except Exception as e:
        print(f"[Error] gen_precise_scene_timestamp_by_subtitle 发生错误: {e}")
        traceback.print_exc()
        return timestamp


def align_single_timestamp(target_ts, merged_timestamps, video_path, max_delta_ms=1000):
    """
    输入一个目标时间戳和原始的时间戳列表，计算出修正后的时间戳。
    该函数内部会自动清洗 merged_timestamps。
    target_ts: ms
    """
    # 1. 数据清洗：在函数内部处理，对调用方透明
    # 只保留有效的时间戳 (timestamp exists, count > 0)
    valid_camera_shots = [c for c in merged_timestamps if c and c[0] is not None and c[1] > 0]

    # 2. 筛选候选者
    candidates = [
        shot for shot in valid_camera_shots
        if abs(shot[0] - target_ts) <= max_delta_ms
    ]

    # 3. 寻找最佳匹配 (Visual)
    best_shot = None
    if candidates:
        # 计算所有候选者的分数并存储为 (score, shot)
        scored_candidates = []
        for shot in candidates:
            diff = abs(shot[0] - target_ts)
            count = shot[1]
            # 评分逻辑：Diff 越小越好，Count 越大越好
            score = diff / count if count > 0 else float('inf')
            scored_candidates.append((score, shot))

        # 按分数升序排序（分数越小越好）
        scored_candidates.sort(key=lambda x: x[0])

        if len(scored_candidates) == 1:
            # 只有一个候选项，直接选中
            best_shot = scored_candidates[0][1]
        else:
            best_score = scored_candidates[0][0]
            second_score = scored_candidates[1][0]

            # 逻辑：
            # 1. 如果最好的分数 < 150，直接选中 (绝对优秀，无需比较)
            # 2. 如果最好的分数 >= 150，则要求它比第二名小 50 以上 (相对优势)
            if best_score < 150 or (second_score - best_score > 50):
                best_shot = scored_candidates[0][1]
            else:
                best_shot = None  # 既不够优秀，优势也不够大，放弃

    # 4. 决策与执行
    # 策略 A: 视觉对齐 (找到且 count >= 2)
    if best_shot and best_shot[1] >= 2:
        new_ts = int(best_shot[0])
        count = best_shot[1]
        diff = abs(new_ts - target_ts)
        score = diff / count if count > 0 else 0

        return new_ts, 'visual', {
            'count': count,
            'diff': diff,
            'score': score
        }

    # 策略 B: 字幕对齐 (无候选 或 count < 2)
    else:
        reason = "无候选 Camera Shot" if not candidates else f"Camera Shot 置信度低 (count={best_shot[1]}<2)"

        # 调用字幕对齐函数
        new_ts = gen_precise_scene_timestamp_by_subtitle(video_path, target_ts)

        if new_ts is not None:
            return new_ts, 'subtitle', {'reason': reason}
        else:
            # 字幕对齐也失败，返回原始时间
            return target_ts, 'failed', {'reason': reason}

@safe_process_limit(limit=3, name="fix_logical_scene_info")
def fix_logical_scene_info(video_path, merged_timestamps, logical_scene_info, max_delta_ms=1000):
    strat_time = time.time()
    time_map = {}  # 用于缓存已处理的时间戳，避免重复计算
    print(f"🔧 开始修正开始 {video_path} 的逻辑场景时间戳...")
    # 检查是否有数据（仅用于打印一条全局警告，不影响逻辑运行）
    has_valid_data = any(c and c[0] is not None and c[1] > 0 for c in merged_timestamps)
    if not has_valid_data:
        print("⚠️ 无有效 camera_shot 时间戳，后续将全部依赖字幕对齐逻辑。")

    scenes = logical_scene_info.get('new_scene_info', [])

    for i, scene in enumerate(scenes):
        for key in ('start', 'end'):
            orig_ts = scene.get(key)
            if orig_ts is None:
                print(f"[Scene {i}] {key}: 无法解析原始时间，跳过。")
                continue

            # 1. 查缓存
            if orig_ts in time_map:
                scene[key] = time_map[orig_ts]
                continue

            # 2. 核心计算：直接传入原始 merged_timestamps，不用管怎么洗数据
            new_ts, strategy, info = align_single_timestamp(
                orig_ts, merged_timestamps, video_path, max_delta_ms
            )

            # 3. 打印日志
            if strategy == 'visual':
                print(f"[Scene {i}] {key}: {orig_ts} -> {new_ts} "
                      f"(🖼️ 视觉修正: count={info['count']}, diff={info['diff']}ms, score={info['score']:.2f})")

            elif strategy == 'subtitle':
                print(f"[Scene {i}] {key}: {orig_ts} -> {new_ts} "
                      f"(🛠️ 字幕修正: {info['reason']})")

            elif strategy == 'failed':
                print(f"[Scene {i}] {key}: {orig_ts} (保持不变, 字幕对齐失败, 原因: {info['reason']})")

            # 4. 更新与缓存
            time_map[orig_ts] = new_ts
            scene[key] = new_ts
    print(f"🎯  {video_path} 完成修正完成，总耗时 {time.time() - strat_time:.2f} 秒。 场景数量为{len(scenes)}")

    return logical_scene_info


def append_segmentation_constraints(full_prompt, fixed_points, max_scenes, guidance_text):
    # 如果没有任何动态约束，直接返回原提示词
    if not any([fixed_points, max_scenes, guidance_text]):
        return full_prompt

    blocks = []

    # ------------------------------------------------------------------
    # 1. 强制分割点 (Fixed Points) - 解决“只切这几刀”的问题
    # ------------------------------------------------------------------
    if fixed_points:
        # 格式化时间戳
        points_str = " / ".join([f"[{ms_to_time(tp)}]" for tp in fixed_points])
        blocks.append(f"""
    **[指令A] 强制物理断点（Mandatory Breakpoints）**
    *   **关键数据**：{points_str}
    *   **操作逻辑**：
        1.  **叠加原则**：这些时间点是必须执行的“硬性切刀”。
        2.  **持续细分**：在执行完上述硬性切割后，**必须**继续在这些时间点形成的区间内部，依据原有的“语义/话题/动作”逻辑进行常规切分。
        3.  **禁止偷懒**：严禁只输出由上述时间点构成的宽泛片段，必须保证常规的颗粒度。
        4.  **对齐要求**：输出的JSON中，必须有场景的 `end` 和下一个场景的 `start` 精确落在这些时间点上。""")

    # ------------------------------------------------------------------
    # 2. 场景数量约束 (Quantity Constraint) - 保持专业术语
    # ------------------------------------------------------------------
    if max_scenes and max_scenes > 0:
        if max_scenes == 1:
            instruction = (
                "**单场景聚合模式**：在严格执行完“删除判定（广告/作者身份）”后，"
                "将剩余的所有保留内容合并为一个唯一的叙事单元，忽略内部的细微转折。"
            )
        else:
            instruction = (
                f"**目标场景量：约 {max_scenes} 个**。\n"
                f"        请调整你的【剪辑颗粒度】。如果自然切分结果远超此数，请按“大事件/大篇章”进行合并；"
                f"如果远少于此数，请按“微动作/单句台词”进行细分。"
            )

        blocks.append(f"""
    **[指令B] 场景颗粒度控制（Granularity Control）**
    *   **目标参数**：{max_scenes}
    *   **操作逻辑**：{instruction}""")

    # ------------------------------------------------------------------
    # 3. 逻辑指导 (Guidance) - 融入“专家人设”
    # ------------------------------------------------------------------
    if guidance_text:
        blocks.append(f"""
    **[指令C] 特殊叙事策略（Special Narrative Strategy）**
    *   **策略描述**："{guidance_text}"
    *   **操作逻辑**：
        1.  **优先级覆写**：在判断“场景边界”时，请优先采用上述策略（例如用户要求按情绪切分，则忽略物理位置变化）。
        2.  **安全底线**：此策略仅影响“如何切分保留内容”，**绝不可**因此保留原定应删除的“广告”或“作者身份暴露”片段。
        3.  **格式维持**：JSON输出结构与字段定义保持不变。""")

    # ------------------------------------------------------------------
    # 组合最终提示词 - 使用“补充协议”的口吻
    # ------------------------------------------------------------------
    if blocks:
        # 这里用一种“附加备忘录”的风格，与你的主Prompt无缝衔接
        header = (
            "\n\n"
            "----------------------------------------------------------------\n"
            "### **特别剪辑任务增补 (Supplementary Editorial Mandates)**\n"
            "注意：在执行上述标准流程前，收到即时更新的剪辑需求。请将以下指令**叠加**到你的分析逻辑中，若与默认切分逻辑冲突，以以下指令为准：\n"
        )
        return full_prompt + header + "\n".join(blocks)

    return full_prompt



def gen_logical_scene_llm(video_path, video_info, all_path_info):
    """
    生成新的视频方案
    """
    cost_time_info = {}
    need_remove_frames = video_info.get('extra_info', {}).get('has_ad_or_face', 'auto')
    static_cut_video_path = all_path_info.get('static_cut_video_path', '')
    base_prompt = gen_base_prompt(video_path, video_info)
    log_pre = f"{video_path} 逻辑性场景划分 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        traceback.print_exc()

        print(f"获取视频时长失败: {e}")
        return "获取视频时长失败", None, {}

    retry_delay = 10
    max_retries = 3
    prompt_file_path = './prompt/视频素材切分.txt'
    full_prompt = read_file_to_str(prompt_file_path)
    full_prompt += f'\n{base_prompt}'
    extra = video_info.get('extra_info', {})

    fixed_points = extra.get('fixed_split_time_points', [])
    max_scenes = extra.get('max_scenes', 0)
    guidance_text = extra.get('split_guidance', '')
    full_prompt = append_segmentation_constraints(full_prompt, fixed_points, max_scenes, guidance_text)
    if need_remove_frames == 'no':
        full_prompt += "\n注意：本次不需要删除任何场景中的广告或作者身份暴露内容。"

    error_info = ""
    gen_error_info = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries}) {log_pre}")
            start_time = time.time()

            random_value = random.random()
            if random_value < 0.9:
                gen_error_info, raw = generate_gemini_content_playwright(full_prompt, file_path=video_path,
                                                                         model_name="gemini-3.1-pro-preview", fallback_model="gemini-3-pro-preview")
            else:
                gen_error_info, raw = generate_gemini_content_playwright(full_prompt, file_path=video_path,
                                                                         model_name="gemini-2.5-pro")
            cost_time_info['llm_generate_time'] = time.time() - start_time

            logical_scene_info = string_to_object(raw)
            check_result, check_info = check_logical_scene(logical_scene_info, video_duration_ms, max_scenes, need_remove_frames, fixed_points)
            if not check_result:
                error_info = f"逻辑性场景划分检查未通过: {check_info} {raw} {log_pre}"
                raise ValueError(f"逻辑性场景划分检查未通过: {check_info} {raw}")
            start_time = time.time()
            merged_timestamps = get_scene(video_path, min_final_scenes=max_scenes)
            cost_time_info['get_scene_time'] = time.time() - start_time

            # 使用信号量控制并发，最多3个线程同时进入此代码块
            start_time = time.time()
            logical_scene_info = fix_logical_scene_info(video_path, merged_timestamps, logical_scene_info, max_delta_ms=1000)
            cost_time_info['fix_scene_time'] = time.time() - start_time

            return None, logical_scene_info, cost_time_info
        except Exception as e:
            error_str = f"{error_info} {str(e)} {log_pre}"
            print(f"生成逻辑性场景划分失败 (尝试 {attempt}/{max_retries}): {error_str} {log_pre} {gen_error_info}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None, {}  # 达到最大重试次数后返回 None


def check_overlays_text(optimized_video_plan, video_duration_ms):
    """
    检查优化的方案
    逻辑更新：先过滤掉文本长度 > 15 的条目，再检查剩余条目的数量和时间范围。
    """

    overlays = optimized_video_plan.get('overlays', [])

    # -------------------------------------------------
    # 1. 过滤：移除文本长度大于 15 的 overlay
    # -------------------------------------------------
    filtered_overlays = [
        overlay for overlay in overlays
        if len(overlay.get('text', '').strip()) <= 15
    ]

    # 修改原本的 optimized_video_plan
    optimized_video_plan['overlays'] = filtered_overlays
    # 更新局部变量 overlays 用于后续检查
    overlays = filtered_overlays

    # -------------------------------------------------
    # 2. 常规检查
    # -------------------------------------------------

    # 检查：过滤后长度是否还大于等于 2
    if len(overlays) < 2:
        return False, f"优化方案检查失败：经过长文本过滤后，overlays 长度必须至少为 2。当前长度为 {len(overlays)}。"

    # 检查：每个start必须都在视频时长范围内
    for i, overlay in enumerate(overlays):
        start = overlay.get('start')
        # 注意：这里假设 time_to_ms 函数在外部已定义
        start_ms = time_to_ms(start)

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
            model_name_list = ["gemini-2.5-flash", "gemini-3-flash-preview"]
            model_name = random.choice(model_name_list)
            print(f"正在视频覆盖文字生成 (尝试 {attempt}/{max_retries}) {log_pre}")
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            video_overlays_text_info = string_to_object(raw)
            check_result, check_info = check_overlays_text(video_overlays_text_info, video_duration_ms)
            if not check_result:
                error_info = f"优化方案检查未通过: {check_info} {raw} {log_pre} {check_info}"
                raise ValueError(error_info)
            return error_info, video_overlays_text_info
        except Exception as e:
            error_str = f"{str(e)} {log_pre}"
            print(f"视频覆盖文字方案检查未通过 (尝试 {attempt}/{max_retries}): {e} {raw} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None  # 达到最大重试次数后返回 None

def check_owner_asr(owner_asr_info, video_duration, check_owner):
    """
        检查生成的asr文本是否正确，第一是验证每个时间是否合理（1.最长跨度不能够超过20s 2.时长的合理性（也就是最快和最慢的语速就能够知道文本对应的时长是否合理） 3.owner语音和本地speaker说话人日志的差异不能够太大）

    :param owner_asr_info: 包含 ASR 信息的字典列表
    :return: 错误信息列表，若没有错误则返回空列表
    """
    max_end_time_ms = 0
    error_info = 'asr文本检查通过'
    has_owner = False
    if not check_owner:
        has_owner = True
    # 使用 enumerate 获取索引和元素，便于日志记录
    for i, segment in enumerate(owner_asr_info):
        try:
            start_str = segment.get("start")
            end_str = segment.get("end")
            speaker = segment.get("speaker", "")
            if speaker == 'owner':
                has_owner = True

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
    if has_owner is False:
        error_info = f"[ERROR] ASR 文本中未检测到任何 'owner' 说话人内容"
        return False, error_info

    # 为owner_asr_info增加source_clip_id字段，从1开始
    source_clip_id = 0
    for segment in owner_asr_info:
        source_clip_id += 1
        segment['source_clip_id'] = source_clip_id

    return True, error_info


def check_video_script(video_script_info, final_scene_info, is_need_narration=True):
    """
    检查 video_script_info 列表中的每个方案是否符合预设的规则。
    如果违反多样性规则，会直接从列表种移除不合规的方案。

    Args:
        video_script_info (list): 包含一个或多个视频脚本方案的列表。**注意：此列表可能会在函数内部被修改（删除元素）**。
        final_scene_info (dict): 包含有效场景ID列表等信息的字典。

    Returns:
        tuple: (bool, str)
    """
    try:
        all_scene_list = final_scene_info.get('all_scenes', [])
        material_usage_mode = final_scene_info.get('material_usage_mode', 'free')
        all_scene_dict = {}
        for scene in all_scene_list:
            scene_id = scene.get('scene_id')
            all_scene_dict[scene_id] = scene

        # 0. 预处理和基本结构检查
        if not isinstance(video_script_info, list):
            return False, "输入的数据 'video_script_info' 不是一个列表。"

        if 'all_scenes' not in final_scene_info or not isinstance(final_scene_info['all_scenes'], list):
            return False, "输入的数据 'final_info_list' 格式错误，缺少 'all_scenes' 列表。"

        valid_scene_ids = {scene['scene_id'] for scene in final_scene_info['all_scenes']}
        valid_source_video_ids = {scene['source_video_id'] for scene in final_scene_info['all_scenes']}

        # 用于记录每个方案使用的素材源数量，用于后续的多样性过滤
        # 这个列表的索引与 video_script_info 的索引是一一对应的
        solutions_source_usage = []

        # === 阶段一：遍历检查基本规则并收集信息 ===
        # 注意：在此循环中严禁修改 video_script_info 的长度，否则会导致索引错乱
        for i, solution in enumerate(video_script_info):
            solution_num = i + 1

            if not isinstance(solution, dict):
                return False, f"方案 {solution_num} 的数据格式不是一个字典。"

            # 1. 检查必要字段
            required_fields = ['title', 'cover_text', 'video_abstract', '方案整体评分', '场景顺序与新文案']
            for field in required_fields:
                value = solution.get(field)
                if value is None:
                    return False, f"方案 {solution_num} 缺少必要字段: '{field}'。"
                if isinstance(value, (str, list)) and not value:
                    return False, f"方案 {solution_num} 的字段 '{field}' 的值不能为空。"

            # 2. 检查评分
            score = solution.get('方案整体评分')
            if not isinstance(score, (int, float)):
                return False, f"方案 {solution_num} 的 '方案整体评分' ({score}) 不是数字类型。"
            if not (0 <= score <= 10):
                return False, f"方案 {solution_num} 的 '方案整体评分' ({score}) 不在 0 到 10 的范围内。"

            # 3. 检查场景细节
            scenes = solution.get('场景顺序与新文案', [])
            if not isinstance(scenes, list):
                return False, f"方案 {solution_num} 的 '场景顺序与新文案' 不是一个列表。"

            seen_scene_ids = set()
            source_video_ids_in_solution = set()  # 当前方案用到的所有源视频ID
            expected_scene_number = 1

            for j, scene in enumerate(scenes):
                scene_num = j + 1
                if not isinstance(scene, dict):
                    return False, f"方案 {solution_num} 的场景 {scene_num} 的数据格式不是一个字典。"

                # 4. 检查编号连续性
                current_scene_number = scene.get('new_scene_number')
                if current_scene_number != expected_scene_number:
                    return False, f"方案 {solution_num} 的 'new_scene_number' 不连续。期望 {expected_scene_number}，实际 {current_scene_number}。"

                # 3. 检查 scene_id 有效性
                scene_id = scene.get('scene_id')
                if not scene_id:
                    return False, f"方案 {solution_num} 的场景 {scene_num} 缺少 'scene_id'。"
                if scene_id not in valid_scene_ids:
                    return False, f"方案 {solution_num} 的场景 {scene_num} ID无效。"
                if scene_id in seen_scene_ids:
                    return False, f"方案 {solution_num} 中存在重复的 scene_id: '{scene_id}'。"

                # 5. 检查 on_screen_text
                if 'on_screen_text' not in scene:
                    return False, f"方案 {solution_num} 的场景 {scene_num} 缺少 'on_screen_text'。"

                # 检查旁白
                if is_need_narration:
                    scene_info = all_scene_dict.get(scene_id, {})
                    narration_script_list = scene_info.get('narration_script_list', [])
                    new_narration_script_list = scene.get('new_narration_script_list', [])
                    if len(narration_script_list) != len(new_narration_script_list):
                        return False, f"方案 {solution_num} 的场景 {scene_num} 旁白列表长度不一致 (原始: {len(narration_script_list)}，新: {len(new_narration_script_list)})。"

                seen_scene_ids.add(scene_id)
                source_video_ids_in_solution.add(all_scene_dict[scene_id].get('source_video_id'))
                expected_scene_number += 1

            # 记录该方案使用的素材源数量
            solutions_source_usage.append(len(source_video_ids_in_solution))

            # 检查素材使用模式 (full/major)
            if material_usage_mode == 'full':
                if len(seen_scene_ids) != len(valid_scene_ids):
                    return False, f"方案 {solution_num} 违反 'full' 模式规则：需使用全部场景。"
            elif material_usage_mode == 'major':
                if len(seen_scene_ids) < (len(valid_scene_ids) / 2):
                    return False, f"方案 {solution_num} 违反 'major' 模式规则：需使用过半场景。"

        # === 阶段二：多样性规则过滤与删除逻辑 ===
        # 逻辑：当总有效素材源 >= 2 时，方案内必须混合使用 >= 2 个素材源。
        # 我们在这里进行“后处理”，安全地移除不满足条件的方案。

        if len(valid_source_video_ids) >= 2:
            filtered_solutions = []

            # 使用平行列表 solutions_source_usage 来判断保留哪些方案
            # 因为目前为止 video_script_info 没有被修改，索引 k 是对应的
            for k, usage_count in enumerate(solutions_source_usage):
                if usage_count >= 2:
                    filtered_solutions.append(video_script_info[k])

            # 检查是否有方案被过滤掉
            if len(filtered_solutions) != len(video_script_info):
                # 使用切片赋值，在原内存地址上修改列表内容，这是安全的修改方式
                video_script_info[:] = filtered_solutions

            # 如果过滤后列表为空，说明所有方案都不合格
            if not video_script_info:
                return False, "违反多样性规则：当有效素材视频源数量大于等于2时，所有方案均未混合使用多种素材源。"

        return True, ""

    except (KeyError, TypeError, AttributeError) as e:
        error_info = f"处理数据时发生结构性错误，请检查输入格式。错误详情: {type(e).__name__}: {e}"
        return False, error_info


def gen_owner_asr_by_llm(video_path, video_info):
    """
    通过大模型生成带说话人识别的ASR文本。
    """
    check_owner = video_info.get('extra_info', {}).get('check_owner', True)
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

            random_value = random.random()
            if random_value < 0.01:
                # gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name="gemini-2.5-pro")
                gen_error_info, raw_response = generate_gemini_content_playwright(prompt, file_path=video_path, model_name="gemini-3-pro-preview")
            else:
                gen_error_info, raw_response = generate_gemini_content_playwright(prompt, file_path=video_path, model_name="gemini-2.5-pro")

            # 解析和校验
            owner_asr_info = string_to_object(raw_response)
            check_result, check_info = check_owner_asr(owner_asr_info, video_duration_ms, check_owner)
            if not check_result:
                error_info = f"asr 检查未通过: {check_info} {raw_response} {check_info} {log_pre}"
                raise ValueError(error_info)
            # owner_asr_info = correct_owner_timestamps(owner_asr_info, video_duration_ms)
            video_info['owner_asr_info'] = owner_asr_info
            owner_asr_info = fix_owner_asr_by_subtitle(video_info)

            return error_info, owner_asr_info
        except Exception as e:
            error_str = f"{error_info} {str(e)} {gen_error_info} {log_pre}"
            print(f"asr 生成 未通过 (尝试 {attempt}/{max_retries}): {e}")
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
        error_info = f"{log_pre}初始化prompt或获取视频时长时出错: {e} "
        print(error_info)
        return error_info, None

    prompt_with_duration = f"{prompt}{base_prompt}"
    comment_list = video_info.get('base_info', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    desc = f"\n已有评论列表 (数字表示已获赞数量): {temp_comments}"
    # 模型选择逻辑（与原版保持一致）
    max_duration = 600
    model_name_list = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    model_name = random.choice(model_name_list)

    if duration > max_duration:
        # 即使超过时长，模型名也没变，但保留打印语句
        print(f"{log_pre} 视频时长 {duration} 秒超过最大限制 {max_duration} 秒，使用默认处理方式。  ")
    error_info = ""
    # 开始重试循环
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{log_pre}--- [第 {attempt}/{MAX_RETRIES} 次尝试] ---  ")

        # 策略：首次尝试带 desc，后续重试不带 desc
        if attempt == 1:
            current_prompt = f"{prompt_with_duration}\n{desc}"
            print(f" {log_pre}生成弹幕互动信息 首次尝试：使用包含 `desc` 的完整 prompt。")
        else:
            current_prompt = prompt_with_duration
            print(f"{log_pre}生成弹幕互动信息 重试尝试：使用不包含 `desc` 的基础 prompt。 ")

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
                    raise ValueError(f"{log_pre}生成弹幕互动信息 结果验证未通过: {check_info} {raw} ")
                return error_info, result
            except Exception as e:
                error_info = f"生成弹幕互动信息 解析返回结果时出错: {str(e)}"
                print(f"生成弹幕互动信息 解析返回结果时出错: {str(e)}")
                # return error_info, None

        except Exception as e:
            error_info = f" {log_pre} {str(e)}"
            print(f"{log_pre} ⚠️生成弹幕互动信息 在第 {attempt} 次调用 LLM API 时发生严重错误: {e}")
            # 如果API调用本身就失败了，也计为一次失败的尝试
            if 'PROHIBITED_CONTENT' in str(e): # <--- 修复在这里
                print("生成弹幕互动信息 遇到内容禁止错误，停止重试。")
                break  # 使用 break 更清晰地跳出循环
    return error_info, None


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
    video_id_list = task_info.get('video_id_list', [])



    for video_id in video_id_list:
        video_info = video_info_dict.get(video_id, {})
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

            suffix = f'_part{counted_scene}'

            new_suffix = simple_cipher(suffix)
            scene['scene_id'] = f"{video_id}{new_suffix}"
            scene['source_video_id'] = video_id
        all_scene_info_list.extend(merged_scene_list)

    final_info = {
        "creative_guidance": creative_guidance,
        "material_usage_mode": material_usage_mode,
        "video_summaries": video_summary_info,
        "all_scenes": all_scene_info_list

    }
    return final_info


def convert_video_data(input_data):
    """
    将原始视频分析数据转换为以 video_id 为键的精简格式。

    Args:
        input_data (dict): 包含 'video_summaries' 和 'all_scenes' 的原始字典数据。

    Returns:
        dict: 转换后的字典，key 为 video_id，value 为包含摘要和场景列表的字典。
    """
    output_data = {}

    # 1. 初始化视频主信息 (处理 video_summaries)
    video_summaries = input_data.get('video_summaries', {})
    for vid_id, vid_info in video_summaries.items():
        output_data[vid_id] = {
            'video_id': vid_id,
            'video_summary': vid_info.get('summary', ''),
            'scene_summary_list': []
        }

    # 2. 收集并处理场景信息 (处理 all_scenes)
    # 使用临时字典按 video_id 分组存储场景，便于后续排序
    scenes_by_video = {}

    all_scenes = input_data.get('all_scenes', [])
    for scene in all_scenes:
        vid_id = scene.get('source_video_id')
        if not vid_id:
            continue

        # 如果 video_summaries 中缺少该 ID，则初始化一个默认结构
        if vid_id not in output_data:
            output_data[vid_id] = {
                'video_id': vid_id,
                'video_summary': '',
                'scene_summary_list': []
            }

        if vid_id not in scenes_by_video:
            scenes_by_video[vid_id] = []

        # 格式转换：scene_summary 从 list 转换为 string
        raw_summary = scene.get('scene_summary', [])
        summary_text = ""
        if isinstance(raw_summary, list):
            summary_text = "".join(raw_summary)
        elif isinstance(raw_summary, str):
            summary_text = raw_summary

        # 获取排序依据 (scene_number_list 的第一个元素)
        scene_num_list = scene.get('scene_number_list', [0])
        sort_key = scene_num_list[0] if scene_num_list else 0

        # 构建目标场景对象
        scene_item = {
            'scene_id': scene.get('scene_id'),
            'scene_summary': summary_text
        }

        # 存入临时列表：(排序键, 场景对象)
        scenes_by_video[vid_id].append((sort_key, scene_item))

    # 3. 排序并组装最终数据
    for vid_id, scenes_list in scenes_by_video.items():
        # 根据 scene_number 从小到大排序
        scenes_list.sort(key=lambda x: x[0])

        # 提取排序后的场景对象并赋值
        output_data[vid_id]['scene_summary_list'] = [item[1] for item in scenes_list]

    return output_data


def check_draft_video_plan(video_content_plans, all_scene_id_list, material_usage_mode='free'):
    """
    检查视频内容计划的有效性。
    修改后逻辑：遍历所有方案，剔除不合格方案，记录错误。
    如果剔除后列表为空，则返回失败和汇总错误；如果仍有方案保留，则视为成功。

    Args:
        video_content_plans (list): 模型生成的计划列表 (将被原地修改)
        all_scene_id_list (list/set/dict): 原始有效的视频ID集合，用于校验是否存在

    Returns:
        tuple: (bool, str) -> (是否通过, 错误信息)
    """

    # 0. 基础类型检查 (如果整体都不是列表，直接返回失败，无法进行后续过滤)
    if not isinstance(video_content_plans, list):
        return False, "返回数据格式错误：video_content_plans 必须是一个列表"

    # 为了提高查找效率，将 valid_video_list 转换为集合 (Set)
    valid_keys_set = set(all_scene_id_list)

    valid_video_id_set = set()

    for scene_id in all_scene_id_list:
        video_id = scene_id.split('_')[0]
        valid_video_id_set.add(video_id)

    # 定义必须存在的字段
    required_fields = {
        'scene_sourcing_plan',
        'video_theme',
        'score'
    }

    # 用于存储经过筛选后的有效方案
    valid_plans = []
    # 用于记录被剔除方案的错误信息
    accumulated_errors = []

    for index, plan in enumerate(video_content_plans):
        # 1. 检查是否为字典
        if not isinstance(plan, dict):
            accumulated_errors.append(f"第 {index + 1} 个方案格式错误：列表元素必须是字典")
            continue  # 跳过此方案，不加入 valid_plans

        # 2. 检查必须包含的字段 (Missing Keys)
        missing_keys = required_fields - set(plan.keys())
        if missing_keys:
            accumulated_errors.append(f"第 {index + 1} 个方案缺失字段：{', '.join(missing_keys)}")
            continue

        # 3. 检查 video_keys 的有效性
        scene_sourcing_plan_list = plan.get('scene_sourcing_plan', [])

        # 3.1 检查类型是否为列表
        if not isinstance(scene_sourcing_plan_list, list):
            accumulated_errors.append(f"第 {index + 1} 个方案的 'video_keys' 必须是一个列表")
            continue

        # 3.2 检查长度是否大于 1 (剪辑至少需要2个视频)
        if len(scene_sourcing_plan_list) <= 0:
            accumulated_errors.append(
                f"第 {index + 1} 个方案无效：'video_keys' 长度为 {len(scene_sourcing_plan_list)}，必须包含至少 2 个场景")
            continue

        if material_usage_mode == 'major':
            if len(scene_sourcing_plan_list) < len(valid_keys_set) / 2:
                accumulated_errors.append(
                    f"第 {index + 1} 个方案无效：'scene_sourcing_plan' 长度为 {len(scene_sourcing_plan_list)}，'major' 模式下必须包含超过半数的场景 (至少 {len(valid_keys_set) / 2})")
                continue

        if material_usage_mode == 'full':
            if len(scene_sourcing_plan_list) < len(valid_keys_set):
                accumulated_errors.append(
                    f"第 {index + 1} 个方案无效：'scene_sourcing_plan' 长度为 {len(scene_sourcing_plan_list)}，'full' 模式下必须包含全部场景 (至少 {len(valid_keys_set)})")
                continue

        plan_id_set = set()
        plan_scene_id_list = []
        for scene_sourcing_plan in scene_sourcing_plan_list:
            scene_id = scene_sourcing_plan.get('source_scene_id', '')
            if scene_id in plan_scene_id_list:
                accumulated_errors.append(
                    f"第 {index + 1} 个方案无效：'scene_sourcing_plan' 中存在重复的 scene_id '{scene_id}'")
                break
            plan_scene_id_list.append(scene_id)
            video_id = scene_id.split('_')[0]
            plan_id_set.add(video_id)

        # 如果valid_video_id_set大于等于2，那么plan_id_set必须大于等于2
        if len(valid_video_id_set) >= 2 and len(plan_id_set) < 2:
            accumulated_errors.append(
                f"第 {index + 1} 个方案无效：涉及的视频源数量为 {len(plan_id_set)}，当原始数据中视频源数量大于等于2时，方案中必须涉及至少2个视频源")
            continue

        # 3.3 检查 key 是否都在 valid_video_list 中 (防止模型臆造 ID)
        invalid_id_found = False
        for scene_sourcing_plan in scene_sourcing_plan_list:
            source_scene_id = scene_sourcing_plan.get('source_scene_id', '')
            if source_scene_id not in valid_keys_set:
                accumulated_errors.append(
                    f"第 {index + 1} 个方案包含无效的视频 ID：'{source_scene_id}' (不在原始数据中)")
                invalid_id_found = True
                break  # 只要发现一个无效ID，该方案即作废
        if invalid_id_found:
            continue

        # 4. 检查字段内容是否为空
        if not plan.get('video_theme') or not isinstance(plan.get('video_theme'), str):
            accumulated_errors.append(f"第 {index + 1} 个方案 'new_video_theme' 为空或类型错误")
            continue

        # 5. 检查 score 是否能转换为 float
        score_val = plan.get('score')
        try:
            float(score_val)
        except (ValueError, TypeError):
            accumulated_errors.append(f"第 {index + 1} 个方案 'score' 格式错误：'{score_val}' 无法转换为浮点数")
            continue

        # 如果所有检查都通过，加入有效列表
        valid_plans.append(plan)

    # 原地修改 video_content_plans，保留仅有效的方案
    video_content_plans[:] = valid_plans

    # 最终判断：如果过滤后没有任何方案剩余，才算失败
    if not video_content_plans:
        # 拼接所有的错误信息返回
        error_report = "\n".join(accumulated_errors)
        return False, f"所有方案均未通过校验，错误详情：\n{error_report}"

    if accumulated_errors:
        print(f"部分方案未通过校验，已剔除无效方案，剩余有效方案数量：{len(video_content_plans)}。错误详情：\n{accumulated_errors}")
    # 如果还有剩余方案，返回成功 (错误信息可置空，或者根据需求返回警告，此处按惯例置空)
    return True, ""


def gen_draft_video_script_llm(final_info_list):
    """
    生成初步的视频脚本
    :param task_info:
    :param video_info_dict:
    :return:
    """

    try:
        creative_guidance = final_info_list.get('creative_guidance', '')
        material_usage_mode = final_info_list.get('material_usage_mode', 'free')
        prompt_path = './prompt/挖掘热门视频规定scene_id.txt'
        full_prompt = read_file_to_str(prompt_path)
        draft_final_info_list = convert_video_data(final_info_list)


        if material_usage_mode == 'major':
            material_usage_prompt_path = './prompt/补丁_素材使用一半以上.txt'
            material_usage_prompt = read_file_to_str(material_usage_prompt_path)
            full_prompt = f"{full_prompt}\n{material_usage_prompt}\n"

        if material_usage_mode == 'full':
            material_usage_prompt_path = './prompt/补丁_素材全部使用.txt'
            material_usage_prompt = read_file_to_str(material_usage_prompt_path)
            full_prompt = f"{full_prompt}\n{material_usage_prompt}\n"

        if creative_guidance:
            creative_guidance_prompt_path = './prompt/补丁_创作指导.txt'
            creative_guidance_prompt = read_file_to_str(creative_guidance_prompt_path)
            full_prompt = f"{full_prompt}\n{creative_guidance_prompt}\n {creative_guidance}\n"

        all_scene_id_list = []
        for info in draft_final_info_list.values():
            scene_summary_list = info.get('scene_summary_list', [])
            for scene_summary in scene_summary_list:
                scene_id = scene_summary.get('scene_id', '')
                all_scene_id_list.append(scene_id)



        full_prompt = f"{full_prompt}\n尽量每个视频都有场景被采用到最终的方案\n尽量每个视频都有场景被采用到最终的方案\n下面是相应的视频场景数据：\n{draft_final_info_list}"
        max_retries = 3
        log_pre = f"视频生成初步脚本 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        retry_delay = 10
        for attempt in range(1, max_retries + 1):
            print(f"视频生成初步脚本... (第 {attempt}/{max_retries} 次) {log_pre}")
            error_info = ""
            gen_error_info = ""
            try:
                random_value = random.random()
                if random_value < 0.8:
                    gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name="gemini-3.1-pro-preview", fallback_model="gemini-3-pro-preview")
                else:
                    # model_name = "gemini-2.5-flash"
                    model_name = "gemini-3-flash-preview"
                    gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name=model_name, fallback_model="gemini-2.5-flash")
                    # raw_response = get_llm_content(prompt=full_prompt, model_name=model_name)

                draft_video_script_info = string_to_object(raw_response)

                check_result, check_info = check_draft_video_plan(draft_video_script_info, all_scene_id_list,  material_usage_mode=material_usage_mode)


                if not check_result:
                    error_info = f"{check_info} {raw_response} {log_pre} {check_info}  "
                    raise ValueError(error_info)

                # 将draft_video_script_info 按照score降序排序
                draft_video_script_info.sort(key=lambda x: float(x.get('score', 0)), reverse=True)

                return error_info, draft_video_script_info
            except Exception as e:
                error_str = f"{str(e)} {gen_error_info}  {log_pre}"
                print(f"初步视频脚本 生成 未通过 (尝试 {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                    time.sleep(retry_delay)  # 等待一段时间后再重试
                else:
                    print(f"达到最大重试次数，失败. {log_pre}")
                    return error_str, None  # 达到最大重试次数后返回 None

    except Exception as e:
        traceback.print_exc()
        error_info = f"生成初步视频脚本失败: {e} "
        print(error_info)
        return error_info, None

def build_final_prompt_data(draft_video_script_info, final_info_list):
    """
    生成最终方案的数据
    :param draft_video_script_info:
    :param final_info_list:
    :return:
    """
    final_prompt_data = {}
    need_filed_list = ['video_theme', 'one_sentence_summary', 'content_logic_description', 'scene_sourcing_plan']

    # 从draft_video_script_info中提取需要的字段，如果没有直接抛错
    for need_filed in need_filed_list:
        if need_filed not in draft_video_script_info:
            raise ValueError(f"初步视频脚本信息缺失必要字段: {need_filed}")
        final_prompt_data[need_filed] = draft_video_script_info[need_filed]

    scene_sourcing_plan_list = final_prompt_data.get('scene_sourcing_plan', [])

    for scene_sourcing_plan in scene_sourcing_plan_list:
        source_scene_id = scene_sourcing_plan.get('source_scene_id', '')
        for scene_info in final_info_list.get('all_scenes', []):
            if scene_info.get('scene_id') == source_scene_id:
                scene_summary_list = scene_info.get('scene_summary', [])
                scene_summary = ','.join(scene_summary_list) if isinstance(scene_summary_list, list) else scene_summary_list
                narration_script_list = scene_info.get('narration_script_list', [])
                original_script_list = scene_info.get('original_script_list', [])
                scene_sourcing_plan['scene_summary'] = scene_summary
                scene_sourcing_plan['narration_script_list'] = narration_script_list
                scene_sourcing_plan['original_script_list'] = original_script_list
                break

    return final_prompt_data


def convert_plan_format(new_data_list, default_score=0.0):
    """
    将新版脚本方案数据结构转换为旧版格式。

    Args:
        new_data_list (list): 包含新格式字典的列表。
        default_score (float): 原格式中'方案整体评分'字段的默认值，因为新数据中没有此字段。

    Returns:
        list: 转换后的旧格式列表。
    """
    original_format_list = []

    for item in new_data_list:
        # 1. 构建外层字典
        converted_item = {
            "title": item.get("title", ""),
            "cover_text": item.get("cover_text", ""),
            "video_abstract": item.get("video_abstract", ""),
            "solution_idea": item.get("solution_idea", ""),
            "方案整体评分": default_score,  # 填充缺失字段
            "场景顺序与新文案": []
        }

        # 2. 遍历并转换内部的场景列表
        # 新格式键名为 'refined_plan'
        refined_plan = item.get("refined_plan", [])

        for scene in refined_plan:
            new_scene_entry = {
                # 映射：order -> new_scene_number
                "new_scene_number": scene.get("order"),

                # 映射：source_scene_id -> scene_id
                "scene_id": scene.get("source_scene_id"),

                # 映射：narrative_function -> selection_logic
                # 这里的逻辑是：场景的"叙事功能"就是选择它的"逻辑"
                "selection_logic": scene.get("narrative_function", ""),

                "transition_text": scene.get("transition_text", ""),
                "transition_necessity_score": scene.get("transition_necessity_score", 0),
                "on_screen_text": scene.get("on_screen_text", ""),
                "new_narration_script_list": scene.get("new_narration_script_list", [])
            }

            converted_item["场景顺序与新文案"].append(new_scene_entry)

        original_format_list.append(converted_item)

    return original_format_list

def gen_video_script_llm(task_info, video_info_dict):
    """
    生成新的脚本
    :param task_info:
    :param video_info_dict:
    :return:
    """
    final_info_list = build_prompt_data(task_info, video_info_dict)
    origin_final_scene_info = copy.deepcopy(final_info_list)

    # 生成初步的方案
    draft_video_script_info = task_info.get('draft_video_script_info', {})
    draft_video_script_info = []
    if not draft_video_script_info:
        error_info, draft_video_script_info = gen_draft_video_script_llm(final_info_list)
        if error_info:
            return error_info, None, None
        task_info['draft_video_script_info'] = draft_video_script_info
    print("生成初步视频脚本成功")

    # print("生成场景的最终数据成功")

    prompt_path = './prompt/多素材视频生成_指定场景.txt'
    task_info['final_scene_info'] = origin_final_scene_info
    full_prompt = read_file_to_str(prompt_path)

    best_draft_video_script_info = draft_video_script_info[0]
    final_prompt_data = build_final_prompt_data(best_draft_video_script_info, final_info_list)

    creative_guidance = final_info_list.get('creative_guidance', '')
    if creative_guidance:
        creative_guidance_prompt_path = './prompt/补丁_创作指导.txt'
        creative_guidance_prompt = read_file_to_str(creative_guidance_prompt_path)
        full_prompt = f"{full_prompt}\n{creative_guidance_prompt}\n {creative_guidance}\n"


    full_prompt = f"{full_prompt}\n下面是初步的方案数据\n{final_prompt_data}"
    max_retries = 3
    log_pre = f"多素材视频生成脚本 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    retry_delay = 10
    for attempt in range(1, max_retries + 1):
        print(f"尝试生成新视频脚本信息... (第 {attempt}/{max_retries} 次) {log_pre}")
        raw_response = ""
        error_info = ""
        try:

            # 50%调用playwright，50%调用managed
            random_value = random.random()
            if random_value < 0.01:
                # gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name="gemini-2.5-pro")
                gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None, model_name="gemini-3-flash-preview", fallback_model="gemini-2.5-flash")
            else:
                gen_error_info, raw_response = generate_gemini_content_managed(full_prompt)
                if gen_error_info:
                    gen_error_info, raw_response = generate_gemini_content_managed(full_prompt, model_name='gemini-3.0-flash')
                if gen_error_info:
                    gen_error_info, raw_response = generate_gemini_content_playwright(full_prompt, file_path=None,
                                                                                      model_name="gemini-3-pro-preview",
                                                                                      fallback_model="gemini-3-flash-preview")

            # 解析和校验
            new_video_script_info = string_to_object(raw_response)
            task_info['origin_video_script_info'] = new_video_script_info
            video_script_info = convert_plan_format(new_video_script_info, default_score=best_draft_video_script_info.get('score', 0.0) / 10)

            check_result, check_info = check_video_script(video_script_info, final_info_list)
            if not check_result:
                error_info = f"新视频脚本 检查未通过: {check_info} {raw_response} {log_pre} {check_info}  "
                raise ValueError(error_info)
            return error_info, video_script_info, origin_final_scene_info
        except Exception as e:
            error_str = f"{str(e)} {gen_error_info}  {log_pre}"
            print(f"新视频脚本 生成 未通过 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None, None  # 达到最大重试次数后返回 None


def get_top_topics(topic_info_list, top_n=20):
    """
    处理topic列表：去重 -> 按播放量降序排序 -> 取Top N -> 格式化字段
    """
    if not topic_info_list:
        return []

    # 1. 按照 topic_id 进行去重
    seen_ids = set()
    deduplicated_list = []

    for topic in topic_info_list:
        t_id = topic.get('topic_id')
        # 如果没有topic_id或者已经存在，则跳过
        if t_id is not None and t_id not in seen_ids:
            seen_ids.add(t_id)
            deduplicated_list.append(topic)

    # 2. 按照 arc_play_vv 进行降序排序 (处理可能缺失该字段的情况，默认为0)
    deduplicated_list.sort(key=lambda x: x.get('arc_play_vv', 0), reverse=True)

    # 3. 获取前 top_n 个 topic
    top_topics = deduplicated_list[:top_n]

    # 4. 构建最终结果，保留 topic_name 和 拼接后的 topic_desc
    result = []
    for topic in top_topics:
        # 获取字段，如果为None则转为空字符串，防止拼接报错
        t_name = topic.get('topic_name') or ""
        t_desc = topic.get('topic_description') or ""
        a_text = topic.get('activity_text') or ""
        a_desc = topic.get('activity_description') or ""

        # 拼接字段。这里使用列表推导式过滤掉空字符串，并用空格连接，
        # 避免出现 "名字  描述" 这种中间有多个空格的情况。
        # 如果你需要无缝拼接（不加空格），可以将 " ".join 改为 "".join
        parts = [str(t_name), str(t_desc), str(a_text), str(a_desc)]
        topic_desc_str = " ".join([p for p in parts if p])

        result.append({
            "topic_id": topic.get('topic_id'),
            "topic_desc": topic_desc_str
        })

    return result


def get_proper_topics(video_info_dict):
    """
    生成合适的topic数据
    :param video_info_dict:
    :return:
    """
    paid_topic_all, category_data_all = get_all_paid_topics()
    category_name_list = []
    for video_id, video_info in video_info_dict.items():
        hudong_info = video_info.get('hudong_info', {})
        category_id_list = hudong_info.get('视频分析', {}).get('category_id_list', [])
        for category_id in category_id_list:
            if str(category_id) in category_data_all.keys():
                category_name = category_data_all[str(category_id)]['name']
                if category_name not in category_name_list:
                    category_name_list.append(category_name)


    if category_name_list == []:
        category_name_list = paid_topic_all.keys()

    # 获取category_name_list对应的paid_topic
    proper_topics = []
    for category_name in category_name_list:
        if category_name in paid_topic_all.keys():
            proper_topics.extend(paid_topic_all[category_name])

    proper_topics = get_top_topics(proper_topics, top_n=40)
    return proper_topics




def build_upload_info_prompt(prompt, task_info, video_info_dict):
    """
    组织好最终的数据，不同的选项有不同的组织方式
    :param task_info:
    :param video_info_dict:
    :return:
    """
    full_prompt = prompt
    video_script_info = task_info.get('video_script_info', {})
    full_prompt += f"\n视频方案信息如下：\n{video_script_info}"


    selected_comments = get_top_comments(video_info_dict)
    full_prompt += f"\n评论列表信息如下:\n{selected_comments}"

    proper_topics = get_proper_topics(video_info_dict)

    full_prompt += f"\n话题列表信息如下:\n{proper_topics}"
    return full_prompt

def check_upload_info(upload_info_list, video_script_info, full_prompt):

    origin_title_list = []
    for video_script in video_script_info:
        title = video_script.get('title', '')
        origin_title_list.append(title)

    for upload_info in upload_info_list:
        title = upload_info.get('title')
        if title not in origin_title_list:
            error_info = f"上传信息 校验未通过: 生成的标题 '{title}' 不在原始脚本标题列表中 {origin_title_list}"
            return False, error_info

        topic_id = upload_info.get('topic_id')
        # 要求topic_id必须是整数
        if not isinstance(topic_id, int):
            error_info = f"上传信息 校验未通过: 生成的话题ID '{topic_id}' 不是整数类型"
            return False, error_info

        if str(topic_id) not in str(full_prompt):
            error_info = f"上传信息 校验未通过: 生成的话题ID '{topic_id}' 不在提供的话题列表中"
            return False, error_info


        category_id = upload_info.get('category_id')
        if str(category_id) not in str(full_prompt):
            error_info = f"上传信息 校验未通过: 生成的分类ID '{category_id}' 不在提供的分类列表中"
            return False, error_info

        tags = upload_info.get('tags')
        if not isinstance(tags, list) or len(tags) == 0:
            error_info = f"上传信息 校验未通过: 生成的标签 '{tags}' 不能是一个非空列表"
            return False, error_info

        introduction = upload_info.get('introduction')
        if not introduction:
            error_info = f"上传信息 校验未通过: 生成的视频简介不能为空"
            return False, error_info
    return True, ""




def gen_upload_info_llm(task_info, video_info_dict):
    """
    生成上传信息
    :param task_info:
    :param video_info_dict:
    :return:
    """
    log_pre = f"生成上传信息 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    MAX_RETRIES = 3  # 设置最大重试次数
    prompt_file_path = './prompt/投稿相关信息的生成.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = build_upload_info_prompt(prompt, task_info, video_info_dict)
    model_name = "gemini-2.5-flash"
    # model_name = "gemini-3-flash-preview"

    video_script_info = task_info.get('video_script_info', [])

    error_info = ""
    # 开始重试循环
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n--- [第 {attempt}/{MAX_RETRIES} 次尝试] ---  {log_pre}")
        try:
            # 1. 调用 LLM 获取原始文本
            raw = get_llm_content(prompt=full_prompt, model_name=model_name)

            # 2. 尝试解析文本为对象
            try:
                upload_info_list = string_to_object(raw)
                check_result, check_info = check_upload_info(upload_info_list,video_script_info, full_prompt)
                if not check_result:
                    raise ValueError(f"生成上传信息 结果验证未通过: {check_info} {raw} {log_pre}")
                return error_info, upload_info_list
            except Exception as e:
                traceback.print_exc()
                error_info = f" {log_pre} {str(e)}"
                print(f"生成上传信息 解析返回结果时出错: {str(e)}")
                # return error_info, None

        except Exception as e:
            traceback.print_exc()

            error_info = f"生成上传信息 解析返回结果时出错: {str(e)}"
            print(f"生成上传信息 在第 {attempt} 次调用 LLM API 时发生严重错误: {e}")
            # 如果API调用本身就失败了，也计为一次失败的尝试
            if 'PROHIBITED_CONTENT' in str(e): # <--- 修复在这里
                print("生成弹幕互动信息 遇到内容禁止错误，停止重试。")
                break  # 使用 break 更清晰地跳出循环
    return error_info, None


def find_boundary_pairs(owner_info_list: list) -> list:
    """
    遍历一个包含字典的列表，找到'speaker'为'owner'和非'owner'的交界处。
    将交界处的相邻两个元素组成一个元组，并添加到结果列表中。

    Args:
        data_list: 输入的列表，每个元素是一个包含'speaker'键的字典。

    Returns:
        一个包含元组的列表，每个元组都是一个'owner'和非'owner'的交界对。
    """
    data_list = owner_info_list
    boundary_pairs = []

    # 如果第一句就是owner，增加一个虚构的'other'交界对
    if data_list and data_list[0].get('speaker') == 'owner':
        virtual_other = {'speaker': 'other', 'start': 0, 'end': 0}
        boundary_pairs.append((virtual_other, data_list[0]))

    # 遍历到倒数第二个元素，以便安全地访问 i+1
    for i in range(len(data_list) - 1):
        current_item = data_list[i]
        next_item = data_list[i+1]

        # 判断当前元素的speaker是否为'owner'
        is_current_owner = (current_item.get('speaker') == 'owner')
        # 判断下一个元素的speaker是否为'owner'
        is_next_owner = (next_item.get('speaker') == 'owner')

        # 如果一个是owner而另一个不是，则它们是交界对
        if is_current_owner != is_next_owner:
            boundary_pairs.append((current_item, next_item))

    return boundary_pairs


def is_char_match_strict(c1: str, c2: str) -> bool:
    """
    严谨的字符匹配规则：追求判断的绝对正确。
    """
    # 1. 忽略大小写的精确匹配（解决 BGM vs bgm, 以及数字和标点）
    if c1.lower() == c2.lower():
        return True

    # 2. 只有当两个字符都是纯中文字符时，才进行拼音的容错匹配
    # \u4e00 到 \u9fa5 是基本汉字的 Unicode 范围
    if not ('\u4e00' <= c1 <= '\u9fa5' and '\u4e00' <= c2 <= '\u9fa5'):
        return False

    # 3. 获取中文字符的【所有】可能读音（开启多音字全量模式）
    # style=NORMAL 表示不带声调（比如 '木' -> 'mu'）
    # 返回结果类似 [['mu']] 或者多音字 [['hang', 'xing']]
    p1_list = pypinyin.pinyin(c1, style=pypinyin.Style.NORMAL, heteronym=True)
    p2_list = pypinyin.pinyin(c2, style=pypinyin.Style.NORMAL, heteronym=True)

    if not p1_list or not p2_list:
        return False

    # 将读音列表转换为集合
    p1_set = set(p1_list[0])
    p2_set = set(p2_list[0])

    # 4. 只要两个字的读音集合有交集（哪怕是多音字碰巧有一个音一样），就视为同音匹配
    return bool(p1_set & p2_set)


def find_longest_common_substring(s1: str, s2: str) -> int:
    """
    计算两个字符串的最长公共子串的长度（支持模糊匹配）。
    """
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest = 0

    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if is_char_match_strict(s1[x - 1], s2[y - 1]):
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
            else:
                m[x][y] = 0

    return longest


def find_text_timestamp_range(timestamp_text_map, target_text):
    """
    在带时间戳的字幕中，根据模糊匹配找到目标文本的起始和结束时间戳。

    Args:
        timestamp_text_map (dict): 时间戳到字幕文本的映射。
        target_text (str): 希望在字幕中查找的目标文本。

    Returns:
        tuple: 一个包含 (start_time, end_time) 的元组，如果未找到则返回 None。
    """
    if not timestamp_text_map or not target_text:
        return None

    sorted_timestamps = sorted(timestamp_text_map.keys())

    # --- 状态变量 ---
    start_time = None  # 记录当前匹配序列的开始时间
    end_time = None  # 记录当前匹配序列中，最后一个成功匹配的时间
    miss_count = 0  # 连续不匹配的帧数计数器
    is_in_match_seq = False  # 标记是否已经进入一个匹配序列

    # --- 遍历所有时间戳 ---
    for i, current_t in enumerate(sorted_timestamps):
        current_text = timestamp_text_map[current_t]

        # --- 核心匹配逻辑 ---
        # 1. 计算匹配阈值
        min_len = min(len(current_text), len(target_text))
        # 匹配数量至少是2，或者达到最短长度的60%
        threshold = max(2, min_len * 0.6)

        # 2. 计算最长公共子串长度
        common_len = find_longest_common_substring(current_text, target_text)

        # 3. 判断是否匹配成功
        is_match = common_len >= threshold

        # --- 状态机处理 ---
        if is_match:
            miss_count = 0  # 重置不匹配计数
            if not is_in_match_seq:
                is_in_match_seq = True
                start_time = current_t
            end_time = current_t
        else:
            if is_in_match_seq:
                miss_count += 1
                if miss_count >= 2:
                    return (start_time, end_time)

    if is_in_match_seq:
        return (start_time, end_time)

    return None


def gen_precise_owner_timestamp_by_subtitle(video_path, timestamp, target_text):
    """
    通过字幕生成更精确的场景时间戳
    :param video_path: 视频路径
    :param timestamp: 初始时间戳 (单位: ms)
    :return: 精确后的时间戳 (单位: ms)
    """
    # 【修改点 1】在函数最外层加入 try 块，包裹所有逻辑
    try:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f'{video_filename}_scenes')
        # 1. 保存关键帧 (涉及IO，易报错)
        image_path_list = save_frames_around_timestamp_ffmpeg(video_path, timestamp / 1000, 30, output_dir, time_duration_s=1)

        result_json = run_subtitle_ocr(image_path_list, only_best=False)

        # 提取所有原始框用于计算范围
        detected_boxes = [sub.get("box", []) for item in result_json.get("data", []) for sub in
                          item.get("subtitles", [])]

        if not detected_boxes:
            print("未找到任何字幕框。")
            return (timestamp, timestamp)

        # --- 阶段 3: 分析并计算最终包围框 ---
        # print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
        good_boxes = analyze_and_filter_boxes(detected_boxes)
        if not good_boxes:
            print("\n[结果] 所有检测到的框都被过滤为异常值。")
            return  (timestamp, timestamp)

        all_points = np.array([point for box in good_boxes for point in box])
        min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
        max_x, max_y = np.max(all_points[:, 0]), np.max(all_points[:, 1])
        final_box_coords = (min_x, max_x, min_y, max_y)

        # print(f"[阶段 3] 最终有效字幕区域 (x: {min_x}~{max_x}, y: {min_y}~{max_y})")

        # --- 阶段 4: 生成 {时间戳: 文本} 映射 ---
        # print("\n[阶段 4] 生成 {时间戳: 文本} 映射...")
        timestamp_text_map = {}

        for item in result_json.get('data', []):
            file_path = item.get('file_path', '')
            match = re.search(r'frame_(\d+)\.png', file_path)
            if not match:
                continue
            current_ms = int(match.group(1))

            best_text = get_best_valid_text(item.get('subtitles', []), final_box_coords)
            # 构造 valid_texts 列表：如果有结果就是 [text]，没有就是 []
            valid_texts = [best_text] if best_text else []

            # 去除首尾空格，避免 OCR 带来的微小差异影响比对
            text_content = "".join(valid_texts).strip()
            timestamp_text_map[current_ms] = text_content

        if not timestamp_text_map:
            print("警告：在指定区域内未提取到有效文本。")
            return  (timestamp, timestamp)

        # 计算逻辑也可能出错，放在 try 块中很安全
        final_timestamp = find_text_timestamp_range(timestamp_text_map, target_text)
        if not final_timestamp:
            print("未找到匹配的字幕时间范围。")
            return (timestamp, timestamp)
        return final_timestamp

    # 【修改点 2】捕获所有异常，打印日志并强制返回原始 timestamp
    except Exception as e:
        print(f"[Error] gen_precise_scene_timestamp_by_subtitle 发生错误: {e}")
        traceback.print_exc()
        return (timestamp, timestamp)

def align_owner_timestamp(target_ts, target_text,  merged_timestamps, video_path, max_delta_ms=500, direction=None):
    """
    输入一个目标时间戳和原始的时间戳列表，计算出修正后的时间戳。
    该函数内部会自动清洗 merged_timestamps。
    target_ts: ms
    direction: None (无要求), 'before' (要求 <= target_ts), 'after' (要求 >= target_ts)
    """
    # 1. 数据清洗：在函数内部处理，对调用方透明
    # 只保留有效的时间戳 (timestamp exists, count > 0)
    valid_camera_shots = [c for c in merged_timestamps if c and c[0] is not None and c[1] > 0]

    # 2. 筛选候选者
    # 修改说明：增加了 direction 的判断逻辑
    candidates = [
        shot for shot in valid_camera_shots
        if abs(shot[0] - target_ts) <= max_delta_ms
        and (direction is None or
             (direction == 'before' and shot[0] <= target_ts) or
             (direction == 'after' and shot[0] >= target_ts))
    ]

    # 3. 寻找最佳匹配 (Visual)
    best_shot = None
    if candidates:
        # 计算所有候选者的分数并存储为 (score, shot)
        scored_candidates = []
        for shot in candidates:
            diff = abs(shot[0] - target_ts)
            count = shot[1]
            # 评分逻辑：Diff 越小越好，Count 越大越好
            score = diff / count if count > 0 else float('inf')
            scored_candidates.append((score, shot))

        # 按分数升序排序（分数越小越好）
        scored_candidates.sort(key=lambda x: x[0])

        if len(scored_candidates) == 1:
            # 只有一个候选项，直接选中
            best_shot = scored_candidates[0][1]
        else:
            best_score = scored_candidates[0][0]
            second_score = scored_candidates[1][0]

            # 逻辑：
            # 1. 如果最好的分数 < 150，直接选中 (绝对优秀，无需比较)
            # 2. 如果最好的分数 >= 150，则要求它比第二名小 50 以上 (相对优势)
            if best_score < 150 or (second_score - best_score > 50):
                best_shot = scored_candidates[0][1]
            else:
                best_shot = None  # 既不够优秀，优势也不够大，放弃

    # 4. 决策与执行
    # 策略 A: 视觉对齐 (找到且 count >= 2)
    if best_shot and best_shot[1] >= 2:
        new_ts = int(best_shot[0])
        count = best_shot[1]
        diff = abs(new_ts - target_ts)
        score = diff / count if count > 0 else 0
        ts_range = (new_ts, new_ts)

        return ts_range, 'visual', {
            'count': count,
            'diff': diff,
            'score': score
        }

    # 策略 B: 字幕对齐 (无候选 或 count < 2)
    else:
        # 修改说明：提示信息中若因方向过滤导致无候选，此处逻辑依然适用
        reason = "无候选 Camera Shot" if not candidates else f"Camera Shot 置信度低 (count={best_shot}<2)"

        # 调用字幕对齐函数
        ts_range = gen_precise_owner_timestamp_by_subtitle(video_path, target_ts, target_text)

        if ts_range is not None:
            return ts_range, 'subtitle', {'reason': reason}
        else:
            # 字幕对齐也失败，返回原始时间
            return (target_ts, target_ts) , 'failed', {'reason': reason}


@safe_process_limit(limit=3, name="fix_owner_asr_by_subtitle")
def fix_owner_asr_by_subtitle(video_info):
    """
    通过字幕和视觉场景纠正说话人（Owner）语音片段的边界时间戳。

    该函数会识别出 Owner 与其他说话人交替的边界，并尝试使用算法
    (`align_owner_timestamp`) 对这些边界进行精细化调整。
    调整依据包括文本的断句和视频的场景切换点。

    重构说明:
    1. **预初始化**: 在处理前，为所有语音片段添加 'fixed_start' 和 'fixed_end' 字段，并赋予初始值，确保最终返回的每个元素都包含这些字段。
    2. **逻辑抽象**: 提取日志打印为辅助函数 `_log_alignment_result`，使主循环更聚焦于核心逻辑。
    3. **常量统一**: 将标点符号的正则表达式统一为一个常量 `PUNCTUATION_REGEX`，增强代码一致性和可维护性。
    4. **修复上下文**: 修复了原代码中 video_id 未定义的潜在问题，假设其可从 video_info 中获取。
    """
    owner_asr_info_list = video_info.get('owner_asr_info', [])
    if not owner_asr_info_list:
        return []

    # 关键调整 2: 预初始化所有片段的 fixed 字段，确保最终返回的数据结构一致
    for asr_info in owner_asr_info_list:
        asr_info['fixed_start'] = asr_info.get('start')
        asr_info['fixed_end'] = asr_info.get('end')

    # 判断是否有owner的asr信息，若无，则直接返回已添加fixed字段的列表
    if not any(asr.get('speaker') == 'owner' for asr in owner_asr_info_list):
        return owner_asr_info_list

    # --- 准备阶段 ---
    video_id = video_info.get('video_id', '')

    all_video_path_info = build_video_paths(video_id)
    video_path = all_video_path_info.get('low_resolution_video_path')
    merged_timestamps = get_scene(video_path)
    boundary_pairs = find_boundary_pairs(owner_asr_info_list)

    # 关键调整 1: 定义统一的标点符号正则常量
    PUNCTUATION_REGEX = r'[，。！？；,.!?;]+'

    # --- 核心处理循环 ---
    for pair_start_info, pair_end_info in boundary_pairs:
        start_speaker = pair_start_info.get('speaker')

        if start_speaker == 'owner':
            direction = 'after'
            # --- 场景 1: 前一个片段是 Owner，修正其【结束时间】 ---
            timestamp = pair_start_info.get('end', 0)
            final_text = pair_start_info.get('final_text', '').strip()
            source_clip_id = pair_start_info.get('source_clip_id', '')

            # 提取最后一句文本用于对齐
            parts = [p.strip() for p in re.split(PUNCTUATION_REGEX, final_text) if p.strip()]
            target_text = parts[-1] if parts else final_text

            ts_range, strategy, info = align_owner_timestamp(
                timestamp, target_text, merged_timestamps, video_path, max_delta_ms=500, direction=direction
            )
            new_ts = ts_range[1]  # 修正结束时间，所以取范围的第二个值

            _log_alignment_result(source_clip_id, 'end', timestamp, new_ts, strategy, info, final_text)

            # 更新数据并处理与下一个片段的冲突
            pair_start_info['fixed_end'] = new_ts
            if new_ts > pair_end_info.get('start', 0):
                pair_end_info['fixed_start'] = new_ts

        else:
            # --- 场景 2: 后一个片段是 Owner，修正其【开始时间】 ---
            direction = 'before'
            timestamp = pair_end_info.get('start', 0)
            final_text = pair_end_info.get('final_text', '')
            source_clip_id = pair_end_info.get('source_clip_id', '')

            # 提取第一句文本用于对齐
            parts = [p.strip() for p in re.split(PUNCTUATION_REGEX, final_text) if p.strip()]
            target_text = parts[0] if parts else final_text

            ts_range, strategy, info = align_owner_timestamp(
                timestamp, target_text, merged_timestamps, video_path, max_delta_ms=500, direction=direction
            )
            new_ts = ts_range[0]  # 修正开始时间，所以取范围的第一个值

            _log_alignment_result(source_clip_id, 'start', timestamp, new_ts, strategy, info, final_text)

            # 更新数据并处理与上一个片段的冲突
            pair_end_info['fixed_start'] = new_ts
            if new_ts < pair_start_info.get('end', 0):
                pair_start_info['fixed_end'] = new_ts

    print()  # 保持原格式空行
    video_info['owner_asr_info'] = owner_asr_info_list
    return owner_asr_info_list


def _log_alignment_result(clip_id, key_type, old_ts, new_ts, strategy, info, text_preview):
    """辅助函数：统一日志打印逻辑，保持控制台输出整洁。"""
    prefix = f"[source_clip_id {clip_id}] {key_type}: {old_ts}"

    if strategy == 'visual':
        log_msg = (f"{prefix} -> {new_ts} "
                   f"(🖼️ 视觉修正: count={info.get('count')}, "
                   f"diff={info.get('diff')}ms, score={info.get('score', 0):.2f})")
    elif strategy == 'subtitle':
        log_msg = (f"{prefix} -> {new_ts} {text_preview} "
                   f"(🛠️ 字幕修正: {info.get('reason')})")
    elif strategy == 'failed':
        log_msg = f"{prefix} (保持不变, 字幕对齐失败, 原因: {info.get('reason')})"
    else:
        log_msg = f"{prefix} (未知策略: {strategy})"  # 增加一个默认分支，以防万一

    print(log_msg)



if __name__ == '__main__':
    video_path = r"W:\project\python_project\auto_video\videos\material\7586639820693179690\7586639820693179690_static_cut.mp4"
    merged_timestamps = get_scene(video_path)






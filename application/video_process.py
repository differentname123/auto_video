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
import os
import random
import shutil
import time

from application.video_common_config import find_best_solution, VIDEO_TASK_BASE_PATH, build_video_paths, ERROR_STATUS, \
    check_failure_details
from utils.common_utils import is_valid_target_file_simple, merge_intervals, ms_to_time, save_json, read_json, \
    time_to_ms, first_greater
from utils.edge_tts_utils import parse_tts_filename, all_voice_name_list
from utils.paddle_ocr import find_overall_subtitle_box_target_number, adjust_subtitle_box
from utils.video_utils import clip_video_ms, merge_videos_ffmpeg, probe_duration, cover_subtitle, \
    add_text_overlays_to_video, gen_video


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
        asr_start = asr_info.get('start')
        asr_start = max(0, asr_start-500)
        asr_end = asr_info.get('end')
        asr_end = min(video_duration_ms, asr_end+500)
        duration_list.append((asr_start, asr_end))
    merge_intervals_list = merge_intervals(duration_list)
    return merge_intervals_list





def _process_single_video(video_id, video_info):
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
    owner_asr_info_list = video_info.get('owner_asr_info')
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
            video_path, merged_timerange_list, output_dir=box_dir
        )
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





def gen_subtitle_box_and_cover_subtitle(video_info_dict):
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
            error_result = _process_single_video(video_id, video_info)

            if error_result:
                failure_details[video_id] = error_result


        except Exception as e:
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
                add_image_text_to_video(video_path, video_info, video_overlays_text_info_list, image_text_video_path, output_dir)

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
                "new_narration_script": "",
                "narration_script_start": current_time,
                "narration_script_end": item_start
            })

        result.append({
            "new_narration_script": item.get("new_narration_script", item.get("narration_script", "")),
            "narration_script_start": item_start,
            "narration_script_end": item_end
        })

        current_time = item_end

    if current_time < end:
        result.append({
            "new_narration_script": "",
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
            if last_clip["new_narration_script"] and not current_clip["new_narration_script"]:
                current_clip["new_narration_script"] = last_clip["new_narration_script"]
            merged_list[-1] = current_clip
        elif current_duration < min_duration:
            last_clip["narration_script_end"] = current_clip["narration_script_end"]
            if current_clip["new_narration_script"] and not last_clip["new_narration_script"]:
                last_clip["new_narration_script"] = current_clip["new_narration_script"]
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


        owner_asr_info_list = video_info.get('owner_asr_info')

        for asr_info in owner_asr_info_list:
            speaker = asr_info.get('speaker')
            if speaker != 'owner':
                continue
            final_text = asr_info.get('final_text')
            all_owner_asr_info_dict[final_text] = asr_info
    return all_logical_scene_dict, all_owner_asr_info_dict




def process_video_with_owner_text(video_path, split_scene, output_dir, subtitle_box, voice_info):
    new_narration_script = split_scene.get('new_narration_script', '')
    narration_script_start = split_scene.get('narration_script_start', 0)
    narration_script_end = split_scene.get('narration_script_end', 0)
    segment_output_scene_file = os.path.join(output_dir, 'split_scene/' f'{narration_script_start}_{narration_script_end}.mp4')
    start_time = time.time()

    if narration_script_start >= narration_script_end - 100:
        print(f"跳过无效时间段: {narration_script_start}-{narration_script_end}")
        return None

    if not is_valid_target_file_simple(segment_output_scene_file):
        clip_video_ms(video_path, narration_script_start, narration_script_end, segment_output_scene_file)

    if new_narration_script.strip() != '':
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
            gen_video(new_narration_script, output_path, origin_video_path, keep_original_audio=keep_original_audio, fixed_rect=subtitle_box, voice_info=voice_info)
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




def gen_scene_video(video_path, new_script_scene, narration_detail_info, merged_segment_list, subtitle_box, voice_info):
    """
    生成单个场景视频
    :return:
    """
    output_dir = os.path.dirname(video_path)

    need_merge_video_file_list = []
    new_narration_script_list = new_script_scene.get('new_narration_script')
    new_narration_script_info_list = []
    for new_narration_script in new_narration_script_list:
        asr_info = narration_detail_info.get(new_narration_script)
        new_narration_script_info_list.append({
            'new_narration_script': new_narration_script,
            'narration_script_start': asr_info.get('start'),
            'narration_script_end': asr_info.get('end')
        })
    split_scene_list = process_narration_clips(merged_segment_list, new_narration_script_info_list, min_duration=500)

    count = 0
    for split_scene in split_scene_list:
        count += 1
        need_merge_video_file = process_video_with_owner_text(video_path, split_scene, output_dir, subtitle_box, voice_info)
        if need_merge_video_file:
            need_merge_video_file_list.append(need_merge_video_file)
    return need_merge_video_file_list


def gen_new_video(task_info, video_info_dict):
    """
    生成新的视频
    :param task_info:
    :param video_info_dict:
    :return:
    """
    failure_details = {}
    voice_info = get_voice_info()
    video_id_str = '_'.join(task_info.get('video_id_list', []))

    final_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, 'remake.mp4')
    # 保证final_output_path所在目录存在
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    if is_valid_target_file_simple(final_output_path):
        print(f"已存在最终生成视频，跳过生成: {final_output_path}")
        return failure_details

    final_scene_info = task_info.get('final_scene_info', {})
    all_logical_scene_dict, all_owner_asr_info_dict = build_all_need_data_map(video_info_dict)
    video_script_info = task_info.get('video_script_info', {})
    best_script = find_best_solution(video_script_info)
    new_script_scenes = best_script.get('场景顺序与新文案', [])
    all_scene_list = final_scene_info.get('all_scenes', [])

    all_final_scene_dict = {}
    for scene in all_scene_list:
        scene_id = scene.get('scene_id')
        all_final_scene_dict[scene_id] = scene

    all_need_merge_video_file_list = []
    for new_script_scene in new_script_scenes:

        scene_id = new_script_scene.get('scene_id')
        scene_info = all_final_scene_dict.get(scene_id, {})
        video_id = scene_info.get('source_video_id')
        all_path_info = build_video_paths(video_id)
        video_path = all_path_info.get('cover_video_path')
        subtitle_box_path = all_path_info.get('subtitle_box_path')
        subtitle_box = read_json(subtitle_box_path)
        top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(video_path, subtitle_box)
        real_subtitle_box = [top_left, bottom_right]
        if is_valid_target_file_simple(all_path_info.get('image_text_video_path')):
            video_path = all_path_info.get('image_text_video_path')
        scene_number_list = scene_info.get('scene_number_list')
        narration_script_list = scene_info.get('narration_script_list', [])
        new_narration_script = new_script_scene.get('new_narration_script', [])
        new_narration_detail_info = {}
        for i, new_narration_script in enumerate(new_narration_script):
            original_narration_script = narration_script_list[i]
            asr_info = all_owner_asr_info_dict.get(original_narration_script)
            if not asr_info:
                error_info = f"未找到对应的ASR信息，场景ID: {scene_id}, 旁白脚本: {original_narration_script} 第{i}段"
                return error_info, None
            new_narration_detail_info[new_narration_script] = asr_info

        all_segment_list = []  # 最终场景的时间段
        for scene_number in scene_number_list:
            scene_key = f"{video_id}_{scene_number}"
            scene_logical_info = all_logical_scene_dict.get(scene_key)
            start = scene_logical_info.get('start')
            end = scene_logical_info.get('end')
            all_segment_list.append((start, end))
        merged_segment_list = merge_intervals(all_segment_list)
        need_merge_video_file_list = gen_scene_video(video_path, new_script_scene, new_narration_detail_info, merged_segment_list, real_subtitle_box, voice_info)
        all_need_merge_video_file_list.extend(need_merge_video_file_list)

    merge_videos_ffmpeg(all_need_merge_video_file_list, output_path=final_output_path)
    return failure_details








def gen_video_by_script(task_info, video_info_dict):
    """
    通过视频脚本生成新的视频
    :param task_info:
    :param video_info_dict:
    :return:
    """

    # 生成字幕遮挡视频
    failure_details = gen_subtitle_box_and_cover_subtitle(video_info_dict)
    if check_failure_details(failure_details):
        return failure_details

    # 生成有了图片文字的视频
    failure_details = add_image_to_video(video_info_dict)
    if check_failure_details(failure_details):
        return failure_details


    failure_details = gen_new_video(task_info, video_info_dict)
    if check_failure_details(failure_details):
        return failure_details

    return failure_details


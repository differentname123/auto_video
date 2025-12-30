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
from utils.paddle_ocr import find_overall_subtitle_box_target_number, adjust_subtitle_box
from utils.video_utils import clip_video_ms, merge_videos_ffmpeg, probe_duration, cover_subtitle, \
    add_text_overlays_to_video


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

    #
    #
    # all_logical_scene_dict = {}
    # all_owner_asr_info_dict = {}
    # for video_id, video_info in video_info_dict.items():
    #     logical_scene_info = video_info.get('logical_scene_info')
    #     new_scene_info_list = logical_scene_info.get('new_scene_info', [])
    #     for scene in new_scene_info_list:
    #         scene_number = scene.get('scene_number')
    #         scene_key = f"{video_id}_{scene_number}"
    #         all_logical_scene_dict[scene_key] = scene
    #
    #
    #     owner_asr_info_list = video_info.get('owner_asr_info')
    #
    #     for asr_info in owner_asr_info_list:
    #         speaker = asr_info.get('speaker')
    #         if speaker != 'owner':
    #             continue
    #         final_text = asr_info.get('final_text')
    #         all_owner_asr_info_dict[final_text] = asr_info
    #
    # video_script_info = task_info.get('video_script_info', {})
    # best_script = find_best_solution(video_script_info)
    # new_script_scenes = best_script.get('场景顺序与新文案', [])
    # final_scene_info = task_info.get('final_scene_info', {})
    # all_scene_list = final_scene_info.get('all_scenes', [])
    # all_final_scene_dict = {}
    # for scene in all_scene_list:
    #     scene_id = scene.get('scene_id')
    #     all_final_scene_dict[scene_id] = scene
    #
    # video_id_list = task_info.get('video_id_list', [])
    # video_id_str = '_'.join(video_id_list)
    # output_path_dir = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str)
    #
    # need_merge_video_file_list = []
    # for new_script_scene in new_script_scenes:
    #     scene_id = new_script_scene.get('scene_id')
    #     scene_info = all_final_scene_dict.get(scene_id, {})
    #     video_id = scene_info.get('source_video_id')
    #     scene_number_list = scene_info.get('scene_number_list')
    #     narration_script_list = scene_info.get('narration_script_list', [])
    #     new_narration_script = new_script_scene.get('new_narration_script', [])
    #     new_narration_detail_info = {}
    #     for i, new_narration_script in enumerate(new_narration_script):
    #         original_narration_script = narration_script_list[i]
    #         asr_info = all_owner_asr_info_dict.get(original_narration_script)
    #         if not asr_info:
    #             error_info = f"未找到对应的ASR信息，场景ID: {scene_id}, 旁白脚本: {original_narration_script} 第{i}段"
    #             return error_info, None
    #         new_narration_detail_info[new_narration_script] = asr_info
    #     all_segment_list = []
    #     for scene_number in scene_number_list:
    #         scene_key = f"{video_id}_{scene_number}"
    #         scene_logical_info = all_logical_scene_dict.get(scene_key)
    #         start = scene_logical_info.get('start')
    #         end = scene_logical_info.get('end')
    #         all_segment_list.append((start, end))
    #
    #     segment_output_scene_file = os.path.join(output_path_dir, "scenes", f"scene_{scene_id}.mp4")
    #     need_merge_video_file_list.append(segment_output_scene_file)
    #     # if is_valid_target_file_simple(segment_output_scene_file, min_size_bytes=1024):
    #     #     print(f"场景视频已存在，跳过生成: {segment_output_scene_file}")
    #     #     continue
    #     start = scene_info.get('start')
    #     end = scene_info.get('end')
    #     source_video_id = scene_info.get('source_video_id')
    #     all_path = build_video_paths(source_video_id)
    #     video_path = all_path.get('origin_video_path')
    #     clip_video_ms(video_path, start, end, segment_output_scene_file)
    #
    # final_output_path = os.path.join(output_path_dir, f"{video_id_str}_final_output.mp4")
    # merge_videos_ffmpeg(need_merge_video_file_list, output_path=final_output_path)
    #
    #
    return failure_details


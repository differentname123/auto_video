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

from application.video_common_config import find_best_solution, VIDEO_TASK_BASE_PATH, build_video_paths
from utils.common_utils import is_valid_target_file_simple
from utils.video_utils import clip_video_ms, merge_videos_ffmpeg






def gen_video_by_script(task_info, video_info_dict):
    """
    通过视频脚本生成新的视频
    :param task_info:
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

    video_script_info = task_info.get('video_script_info', {})
    best_script = find_best_solution(video_script_info)
    new_script_scenes = best_script.get('场景顺序与新文案', [])
    final_scene_info = task_info.get('final_scene_info', {})
    all_scene_list = final_scene_info.get('all_scenes', [])
    all_final_scene_dict = {}
    for scene in all_scene_list:
        scene_id = scene.get('scene_id')
        all_final_scene_dict[scene_id] = scene

    video_id_list = task_info.get('video_id_list', [])
    video_id_str = '_'.join(video_id_list)
    output_path_dir = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str)

    need_merge_video_file_list = []
    for new_script_scene in new_script_scenes:
        scene_id = new_script_scene.get('scene_id')
        scene_info = all_final_scene_dict.get(scene_id, {})
        video_id = scene_info.get('source_video_id')
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
        all_segment_list = []
        for scene_number in scene_number_list:
            scene_key = f"{video_id}_{scene_number}"
            scene_logical_info = all_logical_scene_dict.get(scene_key)
            start = scene_logical_info.get('start')
            end = scene_logical_info.get('end')
            all_segment_list.append((start, end))

        segment_output_scene_file = os.path.join(output_path_dir, "scenes", f"scene_{scene_id}.mp4")
        need_merge_video_file_list.append(segment_output_scene_file)
        # if is_valid_target_file_simple(segment_output_scene_file, min_size_bytes=1024):
        #     print(f"场景视频已存在，跳过生成: {segment_output_scene_file}")
        #     continue
        start = scene_info.get('start')
        end = scene_info.get('end')
        source_video_id = scene_info.get('source_video_id')
        all_path = build_video_paths(source_video_id)
        video_path = all_path.get('origin_video_path')
        clip_video_ms(video_path, start, end, segment_output_scene_file)

    final_output_path = os.path.join(output_path_dir, f"{video_id_str}_final_output.mp4")
    merge_videos_ffmpeg(need_merge_video_file_list, output_path=final_output_path)



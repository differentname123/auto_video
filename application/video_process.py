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

from application.video_common_config import VIDEO_MATERIAL_BASE_PATH, VIDEO_TASK_BASE_PATH
from utils.common_utils import is_valid_target_file_simple
from utils.video_utils import clip_video_ms, merge_videos_ffmpeg


def find_best_solution(video_script_info: list):
    """
    从视频脚本方案列表中，根据“方案整体评分”找出并返回得分最高的方案。

    Args:
        video_script_info (list): 包含多个方案的列表，每个方案是一个字典。
                                  每个字典必须包含一个名为 '方案整体评分' 的键。

    Returns:
        dict: 列表中“方案整体评分”最高的那个方案字典。
              如果输入列表为空，则返回 None。
    """
    # 检查输入列表是否为空，避免对空列表调用max()时出错
    if not video_script_info:
        return None
    best_solution = max(video_script_info, key=lambda solution: solution['方案整体评分'])

    return best_solution

def build_video_paths(video_id):
    """
    生成一个视频id的所有相关地址dict

    :param video_id:
    :return:
    """
    origin_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}/{video_id}_origin.mp4")  # 直接下载下来的原始视频，没有任何的加工
    static_cut_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_static_cut.mp4")  # 静态剪辑后的视频,也就是去除视频画面没有改变的部分，这个是用于后续的剪辑
    low_resolution_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                             f"{video_id}/{video_id}_low_resolution.mp4")  # 这个是静态剪辑后视频再进行降低分辨率和降低帧率后的数据，用于和大模型交互
    return {
        'origin_video_path': origin_video_path,
        'static_cut_video_path': static_cut_video_path,
        'low_resolution_video_path': low_resolution_video_path
    }



def gen_video_by_script(task_info, video_info_dict):
    """
    通过视频脚本生成新的视频
    :param task_info:
    :param video_info_dict:
    :return:
    """

    video_script_info = task_info.get('video_script_info', {})
    best_script = find_best_solution(video_script_info)
    new_script_scenes = best_script.get('场景顺序与新文案', [])
    final_scene_info = task_info.get('final_scene_info', {})
    all_scene_list = final_scene_info.get('all_scenes', [])
    all_scene_dict = {}
    for scene in all_scene_list:
        scene_id = scene.get('scene_id')
        all_scene_dict[scene_id] = scene

    video_id_list = task_info.get('video_id_list', [])
    video_id_str = '_'.join(video_id_list)
    output_path_dir = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str)

    need_merge_video_file_list = []
    for new_script_scene in new_script_scenes:
        scene_id = new_script_scene.get('scene_id')
        segment_output_scene_file = os.path.join(output_path_dir, "scenes", f"scene_{scene_id}.mp4")
        need_merge_video_file_list.append(segment_output_scene_file)
        if is_valid_target_file_simple(segment_output_scene_file, min_size_bytes=1024):
            print(f"场景视频已存在，跳过生成: {segment_output_scene_file}")
            continue
        scene_info = all_scene_dict.get(scene_id, {})
        start = scene_info.get('start')
        end = scene_info.get('end')
        source_video_id = scene_info.get('source_video_id')
        all_path = build_video_paths(source_video_id)
        video_path = all_path.get('origin_video_path')
        clip_video_ms(video_path, start, end, segment_output_scene_file)

    final_output_path = os.path.join(output_path_dir, f"{video_id_str}_final_output.mp4")
    merge_videos_ffmpeg(need_merge_video_file_list, output_path=final_output_path)



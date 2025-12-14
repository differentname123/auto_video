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

from utils.common_utils import is_valid_target_file_simple
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from video_common_config import VIDEO_MAX_RETRY_TIMES, VIDEO_MATERIAL_BASE_PATH


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

def run():
    """
    主运行函数，处理所有需要处理的任务
    :return:
    """
    tasks_to_process = query_need_process_tasks()

    for task_info in tasks_to_process:
        process_single_task(task_info)

def build_video_paths(video_id):
    """
    生成一个视频id的所有相关地址dict

    :param video_id:
    :return:
    """
    origin_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}_origin.mp4")  # 直接下载下来的原始视频，没有任何的加工
    static_cut_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}_static_cut.mp4")  # 静态剪辑后的视频,也就是去除视频画面没有改变的部分，这个是用于后续的剪辑
    low_resolution_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}_low_resolution.mp4") # 这个是静态剪辑后视频再进行降低分辨率和降低帧率后的数据，用于和大模型交互
    return {
        'origin_video_path': origin_video_path,
        'static_cut_video_path': static_cut_video_path,
        'low_resolution_video_path': low_resolution_video_path
    }



def process_single_task(task_info):
    """
    处理单个任务的逻辑
    """
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    video_id_list = task_info.get('video_id_list', [])

    video_info_list = manager.find_materials_by_ids(video_id_list)
    video_info_dict = {video_info['video_id']: video_info for video_info in video_info_list}

    for video_id in video_id_list:
        video_path_info = build_video_paths(video_id)
        origin_video_path = video_path_info['origin_video_path']
        video_info = video_info_dict.get(video_id)
        if not is_valid_target_file_simple(origin_video_path) or not video_info:
            print(f"视频 {video_id} 的原始文件存在，准备进行处理")






    print()
    # 先检查前置信息是否满足
    pass  # 这里实现具体的任  务处理逻辑




if __name__ == '__main__':
    run()
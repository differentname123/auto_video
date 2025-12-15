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
import json
import os
import time

from video_common_config import VIDEO_MAX_RETRY_TIMES, VIDEO_MATERIAL_BASE_PATH, VIDEO_ERROR, \
    _configure_third_party_paths, TaskStatus, NEED_REFRESH_COMMENT

_configure_third_party_paths()

from third_party.TikTokDownloader.douyin_downloader import download_douyin_video_sync, get_comment
from utils.common_utils import is_valid_target_file_simple
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


def build_video_paths(video_id):
    """
    生成一个视频id的所有相关地址dict

    :param video_id:
    :return:
    """
    origin_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}_origin.mp4")  # 直接下载下来的原始视频，没有任何的加工
    static_cut_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}_static_cut.mp4")  # 静态剪辑后的视频,也就是去除视频画面没有改变的部分，这个是用于后续的剪辑
    low_resolution_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                             f"{video_id}_low_resolution.mp4")  # 这个是静态剪辑后视频再进行降低分辨率和降低帧率后的数据，用于和大模型交互
    return {
        'origin_video_path': origin_video_path,
        'static_cut_video_path': static_cut_video_path,
        'low_resolution_video_path': low_resolution_video_path
    }


def run():
    """
    主运行函数，处理所有需要处理的任务
    :return:
    """
    tasks_to_process = query_need_process_tasks()
    print(f"找到 {len(tasks_to_process)} 个需要处理的任务。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    for task_info in tasks_to_process:
        process_single_task(task_info, manager)


def process_single_task(task_info, manager):
    """
    处理单个任务的逻辑，此函数经过了全面的健壮性和效率优化。

    - manager: 外部传入的 MongoManager 实例，用于数据库操作。
    """
    video_id_list = task_info.get('video_id_list', [])
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志

    if not video_id_list:
        print(f"任务 {task_id} 的 video_id_list 为空，直接标记为失败。")
        task_info['status'] = TaskStatus.FAILED
        task_info['failed_count'] = VIDEO_MAX_RETRY_TIMES + 1
        task_info['last_error'] = "Empty video_id_list"
        task_info['finished_time'] = time.time()
        manager.upsert_tasks([task_info])
        return

    # 1. 批量获取所有需要的物料信息
    video_info_list = manager.find_materials_by_ids(video_id_list)
    video_info_dict = {video_info['video_id']: video_info for video_info in video_info_list}

    materials_to_update = []
    failure_details = {}  # 使用字典记录每个失败视频的详细原因
    successful_video_ids = []

    # 2. 循环处理每个视频，并在循环内部处理异常
    for video_id in video_id_list:
        try:
            video_info = video_info_dict.get(video_id)

            if not video_info:
                print(f"严重错误: 视频 {video_id} 在物料库中不存在，已跳过。")
                failure_details[video_id] = "Material not found in database"
                continue

            needs_db_update = False

            # --- 修改点 1: 清理通用错误，将字段值设为 None ---
            # 检查 'processing_error' 是否有旧值。如果有，则清空它并标记需要更新数据库。
            if video_info.get('processing_error'):
                video_info['processing_error'] = None
                needs_db_update = True

            # 准备路径和URL
            video_path_info = build_video_paths(video_id)
            origin_video_path = video_path_info['origin_video_path']
            video_url = f"https://www.douyin.com/video/{video_id}"

            # 步骤A: 保证视频文件存在，并清理相关的错误状态
            if not is_valid_target_file_simple(origin_video_path):
                print(f"视频 {video_id} 的原始文件不存在，准备下载...")
                result = download_douyin_video_sync(video_url)

                if not result:
                    print(f"错误: 视频 {video_id} 下载失败。")
                    video_info["download_error"] = VIDEO_ERROR.DOWNLOAD_FAILED
                    failure_details[video_id] = "Video download failed"
                    materials_to_update.append(video_info)
                    continue

                # 下载成功
                original_file_path, metadata = result
                os.makedirs(os.path.dirname(origin_video_path), exist_ok=True)
                os.replace(original_file_path, origin_video_path)
                print(f"视频 {video_id} 下载并移动成功。")
                video_info['metadata'] = metadata
                needs_db_update = True

                # --- 新增修改 2: 下载成功后，同样确保 download_error 字段为空 ---
                if video_info.get('download_error'):
                    video_info['download_error'] = None
                    # needs_db_update 已经是 True, 无需重复设置
            else:
                # --- 修改点 3: 文件已存在，清理下载错误字段 ---
                # 如果文件已存在，意味着下载是成功的，将任何历史 download_error 值设为 None。
                if video_info.get('download_error'):
                    print(f"视频 {video_id} 文件已存在，清理历史下载错误标记。")
                    video_info['download_error'] = None
                    needs_db_update = True

            # 步骤B: 保证评论信息完整
            # (这部分逻辑没有错误字段，保持不变)
            comment_list = video_info.get('base_info', {}).get('comment_list', [])
            if not comment_list or NEED_REFRESH_COMMENT:
                print(f"视频 {video_id} 的评论需要获取或刷新...")
                fetched_comments = get_comment(video_id, comment_limit=100)
                if 'base_info' not in video_info: video_info['base_info'] = {}
                video_info['base_info']['comment_list'] = fetched_comments
                needs_db_update = True

            successful_video_ids.append(video_id)
            if needs_db_update:
                materials_to_update.append(video_info)

        except Exception as e:
            print(f"严重错误: 处理视频 {video_id} 时发生未知异常: {e}")
            failure_details[video_id] = f"Unexpected error: {str(e)}"
            if 'video_info' in locals() and video_info:
                video_info['processing_error'] = str(e)  # 这里是设置错误，保持不变
                materials_to_update.append(video_info)
            continue

    # 3. 批量更新物料信息
    if materials_to_update:
        unique_materials = {v['video_id']: v for v in materials_to_update}.values()
        manager.ups_ert_materials(list(unique_materials))
        print(f"任务 {task_id}: 批量更新了 {len(unique_materials)} 个物料信息。")

    # 4. 根据处理结果更新任务状态 (这部分逻辑无需修改)
    if not failure_details:
        print(f"任务 {task_id} 全部成功。")
        task_info['status'] = TaskStatus.COMPLETED
        task_info.pop('last_error', None)
    else:
        error_msg = json.dumps(failure_details, ensure_ascii=False, indent=2)
        print(f"任务 {task_id} 处理完成，但存在失败项:\n{error_msg}")
        task_info['last_error'] = error_msg

        if not successful_video_ids:
            current_failures = task_info.get('failed_count', 0) + 1
            task_info['failed_count'] = current_failures
            if current_failures > VIDEO_MAX_RETRY_TIMES:
                task_info['status'] = TaskStatus.FAILED
                print(f"任务 {task_id} 已达到最大重试次数，标记为最终失败。")
            else:
                print(f"任务 {task_id} 本次全部失败，失败次数增加到 {current_failures}。")
        else:
            task_info['status'] = TaskStatus.COMPLETED

    task_info['finished_time'] = time.time()
    manager.upsert_tasks([task_info])


if __name__ == '__main__':
    run()

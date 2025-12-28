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
import shutil
import time

from application.llm_generator import gen_logical_scene_llm, gen_overlays_text_llm, gen_owner_asr_by_llm, \
    gen_hudong_by_llm, gen_video_script_llm
from application.video_process import gen_video_by_script
from utils.video_utils import remove_static_background_video, reduce_and_replace_video, probe_duration
from video_common_config import VIDEO_MAX_RETRY_TIMES, VIDEO_MATERIAL_BASE_PATH, VIDEO_ERROR, \
    _configure_third_party_paths, TaskStatus, NEED_REFRESH_COMMENT, ERROR_STATUS, build_video_paths

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

def process_origin_video(video_id):
    """
    处理原始视频生成后续需要处理的视频
    :param video_id:
    :return:
    """
    video_path_info = build_video_paths(video_id)
    origin_video_path = video_path_info['origin_video_path']
    static_cut_video_path = video_path_info['static_cut_video_path']
    low_resolution_video_path = video_path_info['low_resolution_video_path']

    if not is_valid_target_file_simple(origin_video_path):
        raise FileNotFoundError(f"原始视频文件不存在: {origin_video_path}")

    if not is_valid_target_file_simple(static_cut_video_path):
        # 第一步先进行降低分辨率和帧率
        params = {
            'crf': 23,
            'target_width': 2560,
            'target_fps': 30
            }
        reduce_and_replace_video(origin_video_path, **params)

        # 第二步进行静态背景去除
        crop_result, crop_path = remove_static_background_video(origin_video_path)
        shutil.copy2(crop_path, static_cut_video_path)

    if not is_valid_target_file_simple(low_resolution_video_path):
        # 第三步进行降低分辨率和帧率
        shutil.copy2(static_cut_video_path, low_resolution_video_path)
        reduce_and_replace_video(low_resolution_video_path)
    print(f"视频 {video_id} 的原始视频处理完成。")


def gen_extra_info(task_info, video_info_dict, manager):
    """
    为每个视频生成额外信息 逻辑场景划分 覆盖文字识别 作者语音识别
    :param task_info:
    :param video_info_dict:
    :return:
    """
    failure_details = {}
    original_url_info_list = task_info.get('original_url_info_list', [])

    for original_url_info in original_url_info_list:
        video_id = original_url_info.get('video_id')
        if not video_id:
            error_info = "Missing video_id in original_url_info"
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            continue
        video_info = video_info_dict.get(video_id)
        if not video_info:
            error_info = "Video info not found in database"
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            continue
        path_info = build_video_paths(video_id)
        logical_scene_info = video_info.get('logical_scene_info')
        video_path = path_info['low_resolution_video_path']
        error_info = ""
        if not logical_scene_info:
            error_info, logical_scene_info = gen_logical_scene_llm(video_path, video_info)
            if not error_info:
                video_info['logical_scene_info'] = logical_scene_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
                video_info["logical_error"] = error_info

            manager.upsert_materials([video_info])
            if error_info:
                continue
        print(f"视频 {video_id} logical_scene_info生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} {error_info}")




        video_overlays_text_info = video_info.get('video_overlays_text_info', {})
        if not video_overlays_text_info:
            error_info, video_overlays_text_info = gen_overlays_text_llm(video_path, video_info)
            if not error_info:
                video_info['video_overlays_text_info'] = video_overlays_text_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.WARNING
                }
                video_info["overlays_text_error"] = error_info

            manager.upsert_materials([video_info])
            if error_info:
                print(f"视频 {video_id} overlays_text_info 生成失败: {error_info} 但是非必要可以继续运行")
        print(f"视频 {video_id} overlays_text_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} {error_info}")





        owner_asr_info = video_info.get('owner_asr_info', {})
        is_contains_author_voice = video_info.get('is_contains_author_voice', True)
        if is_contains_author_voice and not owner_asr_info:
            error_info, owner_asr_info = gen_owner_asr_by_llm(video_path, video_info)
            if not error_info:
                video_info['owner_asr_info'] = owner_asr_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
                video_info["owner_asr_error"] = error_info
            manager.upsert_materials([video_info])
            if error_info:
                continue
        print(f"视频 {video_id} owner_asr_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} {error_info}")



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
                video_info["hudong_error"] = error_info
            manager.upsert_materials([video_info])
            if error_info:
                continue
        print(f"视频 {video_id} hudong_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} {error_info}")

    return failure_details


def check_failure_details(failure_details):
    """
    判断failure_details中错误等级是否有超过ERROR的
    :param failure_details:
    :return:
    """
    for video_id, detail in failure_details.items():
        if detail.get('error_level') in [ERROR_STATUS.ERROR, ERROR_STATUS.CRITICAL]:
            print(f"检测到严重错误，停止后续处理。视频ID: {video_id} 错误详情: {detail}")
            return True
    return False



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
                failure_details[video_id] = {
                    "error_info": "Video info not found in database",
                    "error_level": ERROR_STATUS.CRITICAL
                }
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
                    failure_details[video_id] = {
                        "error_info": "Download failed",
                        "error_level": ERROR_STATUS.ERROR
                    }
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
            failure_details[video_id] = {
                "error_info": f"Unexpected error: {str(e)}",
                "error_level": ERROR_STATUS.ERROR
            }
            if 'video_info' in locals() and video_info:
                video_info['processing_error'] = str(e)  # 这里是设置错误，保持不变
                materials_to_update.append(video_info)
            continue

    # 3. 批量更新物料信息
    if materials_to_update:
        unique_materials = {v['video_id']: v for v in materials_to_update}.values()
        manager.upsert_materials(list(unique_materials))
        print(f"任务 {task_id}: 批量更新了 {len(unique_materials)} 个物料信息。")

    # 4. 根据处理结果更新任务状态 (这部分逻辑无需修改)
    if not failure_details:
        task_info['last_error'] = None
    else:
        error_msg = json.dumps(failure_details, ensure_ascii=False, indent=2)
        print(f"任务 {task_id} 处理完成，但存在失败项:\n{error_msg}")
        task_info['last_error'] = error_msg

        if not successful_video_ids:
            current_failures = task_info.get('failed_count', 0) + 1
            task_info['failed_count'] = current_failures
            task_info['status'] = TaskStatus.FAILED

    task_info['finished_time'] = time.time()
    # 完成数据拉取阶段，可以先保存一份数据
    manager.upsert_tasks([task_info])

    # 如果上一步出现过错误直接结束
    if check_failure_details(failure_details):
        return

    # 进行原始视频的加工，生成后续要处理的视频
    for video_id in successful_video_ids:
        try:
            process_origin_video(video_id)
        except Exception as e:
            print(f"严重错误: 处理视频 {video_id} 的原始视频时发生异常: {e}")
            failure_details[video_id] = {
                "error_info": f"Error processing origin video: {str(e)}",
                "error_level": ERROR_STATUS.ERROR
            }
            video_info = video_info_dict.get(video_id)
            if video_info:
                video_info['processing_error'] = str(e)
                manager.upsert_materials([video_info])
            continue

    if not failure_details:
        task_info['last_error'] = None
    else:
        error_msg = json.dumps(failure_details, ensure_ascii=False, indent=2)
        print(f"任务 {task_id} 原始视频处理完成，但存在失败项:\n{error_msg}")
        task_info['last_error'] = error_msg

        if not successful_video_ids:
            current_failures = task_info.get('failed_count', 0) + 1
            task_info['failed_count'] = current_failures
            task_info['status'] = TaskStatus.FAILED

    manager.upsert_tasks([task_info])

    # 如果上一步出现过错误直接结束
    if check_failure_details(failure_details):
        return

    failure_details = gen_extra_info(task_info, video_info_dict, manager)

    if check_failure_details(failure_details):
        return


    video_script_info = task_info.get('video_script_info', {})
    error_info = ""
    if video_script_info:
        error_info, video_script_info, final_scene_info = gen_video_script_llm(task_info, video_info_dict)
        if not error_info:
            task_info['video_script_info'] = video_script_info
            task_info['final_scene_info'] = final_scene_info
            task_info['status'] = TaskStatus.PLAN_GENERATED
        else:
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            task_info["script_error"] = error_info
        manager.upsert_tasks([task_info])

    if check_failure_details(failure_details):
        return
    print(f"任务 {task_id} 脚本生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} {error_info}")

    gen_video_by_script(task_info, video_info_dict)

    print(f"任务 {task_id} 全部处理完成。\n")


if __name__ == '__main__':
    run()

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
import traceback

from application.llm_generator import gen_logical_scene_llm, gen_overlays_text_llm, gen_owner_asr_by_llm, \
    gen_hudong_by_llm, gen_video_script_llm, align_single_timestamp, gen_upload_info_llm
from application.video_process import gen_video_by_script
from utils.bilibili.bili_utils import check_duplicate_video
from utils.video_utils import remove_static_background_video, reduce_and_replace_video, probe_duration, get_scene, \
    clip_and_merge_segments
from video_common_config import VIDEO_MAX_RETRY_TIMES, VIDEO_MATERIAL_BASE_PATH, VIDEO_ERROR, \
    _configure_third_party_paths, TaskStatus, NEED_REFRESH_COMMENT, ERROR_STATUS, build_video_paths, \
    check_failure_details, fix_split_time_points

_configure_third_party_paths()

from third_party.TikTokDownloader.douyin_downloader import download_douyin_video_sync, get_comment
from utils.common_utils import is_valid_target_file_simple, time_to_ms, merge_intervals, get_remaining_segments
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
    # 引入线程池执行器 (放在此处是为了不修改函数外部代码，也可移至文件顶部)
    from concurrent.futures import ThreadPoolExecutor

    tasks_to_process = query_need_process_tasks()
    print(f"找到 {len(tasks_to_process)} 个需要处理的任务。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)

    def _task_worker(task_info):
        """
        线程工作函数，封装原本循环体内的逻辑
        """
        failure_details = {}
        try:
            failure_details, video_info_dict, chosen_script = process_single_task(task_info, manager)
        except Exception as e:
            traceback.print_exc()
            error_info = f"严重错误: 处理任务 {task_info.get('_id', 'N/A')} 时发生未知异常: {str(e)}"
            print(error_info)
            failure_details[str(task_info.get('_id', 'N/A'))] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.CRITICAL
            }
            # 原代码在循环中使用了 continue，此处函数执行完异常处理后会自动进入 finally，效果一致
        finally:
            if check_failure_details(failure_details):
                failed_count = task_info.get('failed_count', 0)
                task_info['failed_count'] = failed_count + 1
                task_info['status'] = TaskStatus.FAILED
            else:
                # task_info['status'] = TaskStatus.COMPLETED
                pass

            task_info['failure_details'] = str(failure_details)
            manager.upsert_tasks([task_info])

    # 使用线程池并发处理，设置线程数量为 5
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(_task_worker, tasks_to_process)

def cutoff_target_segment(video_path, remove_time_segments, output_path):
    """
    按照期望的时间段，剔除指定时间段的视频
    :param video_path:
    :param remove_time_segments:
    :param output_path:
    :return:
    """
    all_timestamp_list = []
    for remove_time_segment in remove_time_segments:
        # 简单校验格式，防止 crash
        if '-' in remove_time_segment:
            start_str, end_str = remove_time_segment.split('-')
            # 确保转换为整数或浮点数
            start_ms = time_to_ms(start_str)
            end_ms = time_to_ms(end_str)
            all_timestamp_list.append(start_ms)
            all_timestamp_list.append(end_ms)
    # 对all_timestamp_list进行去重和排序
    all_timestamp_list = sorted(set(all_timestamp_list))
    if len(all_timestamp_list) > 0:
        print(f"准备剔除视频 {video_path} 的时间段: {remove_time_segments}，对应的时间戳列表: {all_timestamp_list}")
    else:
        # 复制一份video_path到output_path
        shutil.copy2(video_path, output_path)
        return []
    time_map = {}
    merged_timestamps = get_scene(video_path)

    for target_ts in all_timestamp_list:
        new_ts, strategy, info = align_single_timestamp(target_ts, merged_timestamps, video_path)
        # 3. 打印日志
        if strategy == 'visual':
            print(f"[Scene: {target_ts} -> {new_ts} "
                  f"(✅ 视觉修正: count={info['count']}, diff={info['diff']}ms, score={info['score']:.2f})")

        elif strategy == 'subtitle':
            print(f": {target_ts} -> {new_ts} "
                  f"(🛠️ 字幕修正: {info['reason']})")

        elif strategy == 'failed':
            print(f" {target_ts} (保持不变, 字幕对齐失败, 原因: {info['reason']})")
        time_map[target_ts] = new_ts

    # 根据时间映射生成新的剔除时间段

    fixed_remove_time_segments = []
    for remove_time_segment in remove_time_segments:
        # 简单校验格式，防止 crash
        if '-' in remove_time_segment:
            start_str, end_str = remove_time_segment.split('-')
            # 确保转换为整数或浮点数
            start_ms = time_to_ms(start_str)
            end_ms = time_to_ms(end_str)
            fixed_start_ms = time_map.get(start_ms, start_ms)
            fixed_end_ms = time_map.get(end_ms, end_ms)
            fixed_remove_time_segments.append((fixed_start_ms, fixed_end_ms))

    merged_fixed_remove_time_segments = merge_intervals(fixed_remove_time_segments)
    duration_ms = probe_duration(video_path) * 1000

    remaining_segments = get_remaining_segments(duration_ms, merged_fixed_remove_time_segments)

    # 使用ffmpeg命令行工具进行视频剪辑
    clip_and_merge_segments(video_path, remaining_segments, output_path)
    return fixed_remove_time_segments




def process_origin_video(video_id, video_info):
    """
    处理原始视频生成后续需要处理的视频
    :param video_id:
    :return:
    """
    video_path_info = build_video_paths(video_id)
    origin_video_path = video_path_info['origin_video_path']
    origin_video_delete_part_path = video_path_info['origin_video_delete_part_path']
    low_origin_video_path = video_path_info['low_origin_video_path']
    static_cut_video_path = video_path_info['static_cut_video_path']
    low_resolution_video_path = video_path_info['low_resolution_video_path']


    if not is_valid_target_file_simple(origin_video_path):
        raise FileNotFoundError(f"原始视频文件不存在: {origin_video_path}")

    # 使用一个变量标识文件状态是否发生变更，利用连锁反应触发后续更新
    file_changed = False

    # 1. 剪切片段处理
    # 如果文件不存在，或者强制要求重剪，则执行
    if not is_valid_target_file_simple(origin_video_delete_part_path) or video_info.get('need_recut', True):
        remove_time_segments = video_info.get('extra_info', {}).get('remove_time_segments', [])
        fixed_remove_time_segments = cutoff_target_segment(origin_video_path, remove_time_segments, origin_video_delete_part_path)
        video_info['extra_info']['fixed_remove_time_segments'] = fixed_remove_time_segments
        video_info['need_recut'] = False
        split_time_points = video_info.get('extra_info', {}).get('split_time_points', [])
        fixed_split_time_points = fix_split_time_points(fixed_remove_time_segments, split_time_points)
        video_info['extra_info']['fixed_split_time_points'] = fixed_split_time_points
        # 标记变动，后续步骤将强制执行
        file_changed = True


    # 2. 生成 low_origin_video
    # 如果文件不存在，或者上一步发生了变动(file_changed为True)，则执行
    if not is_valid_target_file_simple(low_origin_video_path) or file_changed:
        shutil.copy2(origin_video_delete_part_path, low_origin_video_path)
        file_changed = True


    # 3. 生成 static_cut_video
    # 如果文件不存在，或者上一步发生了变动，则执行
    if not is_valid_target_file_simple(static_cut_video_path) or file_changed:
        # 第一步先进行降低分辨率和帧率(初步)
        params = {
            'crf': 23,
            'target_width': 2560,
            'target_fps': 30
            }
        reduce_and_replace_video(low_origin_video_path, **params)

        # 第二步进行静态背景去除
        crop_result, crop_path = remove_static_background_video(low_origin_video_path)
        shutil.copy2(crop_path, static_cut_video_path)
        file_changed = True

    # 4. 生成 low_resolution_video
    # 如果文件不存在，或者上一步发生了变动，则执行
    if not is_valid_target_file_simple(low_resolution_video_path) or file_changed:
        # 第三步进行降低分辨率和帧率（超级压缩）
        shutil.copy2(static_cut_video_path, low_resolution_video_path)
        reduce_and_replace_video(low_resolution_video_path)
        file_changed = True

    print(f"视频 {video_id} 的原始视频处理完成。")


def gen_extra_info(video_info_dict, manager):
    """
    为每个视频生成额外信息 逻辑场景划分 覆盖文字识别 作者语音识别
    :param video_info_dict:
    :return:
    """
    failure_details = {}

    for video_id, video_info in video_info_dict.items():
        all_path_info = build_video_paths(video_id)

        # 生成逻辑性的场景划分
        logical_scene_info = video_info.get('logical_scene_info')
        video_path = all_path_info['low_resolution_video_path']
        if not logical_scene_info:
            error_info, logical_scene_info = gen_logical_scene_llm(video_path, video_info, all_path_info)
            if not error_info:
                video_info['logical_scene_info'] = logical_scene_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
            update_video_info(video_info_dict, manager, failure_details, error_key='logical_error')
        if check_failure_details(failure_details):
            return failure_details
        print(f"视频 {video_id} logical_scene_info生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")



        # 生气情绪性花字
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
            update_video_info(video_info_dict, manager, failure_details, error_key='overlays_text_error')

        if check_failure_details(failure_details):
            return failure_details
        failure_details = {}
        print(f"视频 {video_id} overlays_text_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")




        # 生成asr识别结果
        owner_asr_info = video_info.get('owner_asr_info', None)
        is_contains_author_voice = video_info.get('extra_info', {}).get('is_contains_author_voice', True)
        if is_contains_author_voice and owner_asr_info is None:
            error_info, owner_asr_info = gen_owner_asr_by_llm(video_path, video_info)
            if not error_info:
                video_info['owner_asr_info'] = owner_asr_info
            else:
                failure_details[video_id] = {
                    "error_info": error_info,
                    "error_level": ERROR_STATUS.ERROR
                }
            update_video_info(video_info_dict, manager, failure_details, error_key='owner_asr_error')

        if check_failure_details(failure_details):
            return failure_details
        print(f"视频 {video_id} owner_asr_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")


        # 生成互动信息
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
            update_video_info(video_info_dict, manager, failure_details, error_key='hudong_error')
        if check_failure_details(failure_details):
            return failure_details
        print(f"视频 {video_id} hudong_info 生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return failure_details


def gen_video_info_dict(task_info, manager):
    """
    生成相应的单视频信息字典，key为video_id，值为物料表的视频信息
    :param task_info:
    :param manager:
    :return:
    """
    failure_details = {}  # 使用字典记录每个失败视频的详细原因

    video_id_list = task_info.get('video_id_list', [])
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志

    if not video_id_list:
        error_info = f"任务 {task_id} 的 video_id_list 为空，直接标记为失败。"
        print(error_info)
        failure_details[task_id] = {
            "error_info": error_info,
            "error_level": ERROR_STATUS.CRITICAL
        }
        return failure_details, {}

    # 1. 批量获取所有需要的物料信息
    video_info_list = manager.find_materials_by_ids(video_id_list)
    video_info_dict = {video_info['video_id']: video_info for video_info in video_info_list}
    for video_id in video_id_list:
        video_info = video_info_dict.get(video_id)
        if not video_info:
            print(f"任务 {task_id} 严重错误: 视频 {video_id} 在物料库中不存在，已跳过。")
            failure_details[video_id] = {
                "error_info": "Video info not found in database",
                "error_level": ERROR_STATUS.CRITICAL
            }


    return failure_details, video_info_dict


def prepare_basic_video_info(video_info_dict):
    """
    准备基础视频信息，比如评论，原始视频，等
    :param video_info_dict:
    :return:
    """
    log_pre = f"准备基础视频信息  当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    failure_details = {}
    for video_id, video_info in video_info_dict.items():
        try:
            # 准备路径和URL
            video_path_info = build_video_paths(video_id)
            origin_video_path = video_path_info['origin_video_path']
            video_url = f"https://www.douyin.com/video/{video_id}"

            # 步骤A: 保证视频文件存在，并清理相关的错误状态
            if not is_valid_target_file_simple(origin_video_path):
                print(f"视频 {video_id} 的原始文件不存在，准备下载...{log_pre}")
                result = download_douyin_video_sync(video_url)

                if not result:
                    error_info = f"错误: 视频 {video_id} 下载失败。{log_pre}"
                    print(error_info)
                    failure_details[video_id] = {
                        "error_info": error_info,
                        "error_level": ERROR_STATUS.ERROR
                    }
                    continue

                # 下载成功
                original_file_path, metadata = result
                os.makedirs(os.path.dirname(origin_video_path), exist_ok=True)
                os.replace(original_file_path, origin_video_path)
                print(f"视频 {video_id} 下载并移动成功。{log_pre}")
                video_info['metadata'] = metadata


            # 步骤B: 保证评论信息完整
            comment_list = video_info.get('comment_list', [])
            if not comment_list or NEED_REFRESH_COMMENT:
                print(f"视频 {video_id} 的评论需要获取或刷新...{log_pre}")
                fetched_comments = get_comment(video_id, comment_limit=100)
                video_info['comment_list'] = fetched_comments
            print(f"视频 {video_id} 的基础信息准备完成。{log_pre}")

            # 判断is_duplicate是否已经存在，避免重复计算
            is_duplicate = video_info.get('is_duplicate')
            if is_duplicate is None:
                is_duplicate = check_duplicate_video(video_info.get('metadata')[0])
                video_info['is_duplicate'] = is_duplicate






        except Exception as e:
            error_info = f"严重错误: 处理视频 {video_id} 时发生未知异常: {str(e)}"
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
    return failure_details, video_info_dict


def update_video_info(video_info_dict, manager, failure_details, error_key='last_error'):
    """
    更新物料视频信息
    :param video_info_dict:
    :param manager:
    :param failure_details:
    :param error_key:
    :return:
    """
    for video_id, detail in failure_details.items():
        video_info = video_info_dict.get(video_id)
        # 注释掉下面的一行代码就能够保存历史错误，而不至于覆盖
        video_info[error_key] = ""
        if video_info:
            video_info[error_key] = detail.get('error_info', 'Unknown error')
    manager.upsert_materials(video_info_dict.values())


def gen_derive_videos(video_info_dict):
    """
    生成后续需要处理的派生视频，主要是静态去除以及降低分辨率后的视频
    :param video_info_dict:
    :return:
    """
    failure_details = {}
    for video_id, video_info in video_info_dict.items():
        try:
            process_origin_video(video_id, video_info)
        except Exception as e:
            error_info = f"严重错误: 处理视频 {video_id} 的原始视频时发生异常: {str(e)}"
            print(error_info)
            failure_details[video_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
    return failure_details

def gen_video_script(task_info, video_info_dict, manager):
    """
    生成多素材的方案
    :param task_info:
    :param video_info_dict:
    :param manager:
    :return:
    """
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志
    failure_details = {}
    video_script_info = task_info.get('video_script_info', {})
    if not video_script_info:
        error_info, video_script_info, final_scene_info = gen_video_script_llm(task_info, video_info_dict)
        if not error_info:
            task_info['video_script_info'] = video_script_info
            task_info['final_scene_info'] = final_scene_info
        else:
            failure_details[task_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            task_info["script_error"] = error_info
        manager.upsert_tasks([task_info])
    return failure_details


def gen_upload_info(task_info, video_info_dict, manager):
    """
    生成投稿需要的信息
    :param task_info:
    :param video_info_dict:
    :return:
    """
    task_id = task_info.get('_id', 'N/A')  # 获取任务ID用于日志
    failure_details = {}
    upload_info = task_info.get('upload_info', {})
    if not upload_info:
        error_info, upload_info = gen_upload_info_llm(task_info, video_info_dict)
        if not error_info:
            task_info['upload_info'] = upload_info
            task_info['status'] = TaskStatus.PLAN_GENERATED
        else:
            failure_details[task_id] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.ERROR
            }
            task_info["upload_info_error"] = error_info
        manager.upsert_tasks([task_info])
    return failure_details


def process_single_task(task_info, manager, gen_video=False):
    """
    处理单个任务的逻辑，此函数经过了全面的健壮性和效率优化。

    - manager: 外部传入的 MongoManager 实例，用于数据库操作。
    """

    chosen_script = None
    # 准备好相应的视频数据
    failure_details, video_info_dict = gen_video_info_dict(task_info, manager)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 确保基础数据存在，比如视频文件，评论等
    failure_details, video_info_dict = prepare_basic_video_info(video_info_dict)
    update_video_info(video_info_dict, manager, failure_details, error_key='prepare_basic_video_error')
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 生成后续需要处理的派生视频，删除指定片段主要是静态去除以及降低分辨率后的视频
    failure_details = gen_derive_videos(video_info_dict)
    update_video_info(video_info_dict, manager, failure_details, error_key='gen_derive_error')
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script

    # 为每一个视频生成需要的大模型信息 场景切分 asr识别， 图片文字等
    failure_details = gen_extra_info(video_info_dict, manager)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    print(f"任务 {video_info_dict.keys()} 单视频信息生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")



    # 生成新的视频脚本方案
    failure_details = gen_video_script(task_info, video_info_dict, manager)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    print(f"任务 {video_info_dict.keys()} 脚本生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")


    # 生成投稿所需的信息
    failure_details = gen_upload_info(task_info, video_info_dict, manager)
    if check_failure_details(failure_details):
        return failure_details, video_info_dict, chosen_script
    task_info['status'] = TaskStatus.PLAN_GENERATED
    print(f"任务 {video_info_dict.keys()} 投稿信息生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if gen_video:
        # 根据方案生成最终视频
        failure_details, chosen_script = gen_video_by_script(task_info, video_info_dict)
        if check_failure_details(failure_details):
            return failure_details, video_info_dict, chosen_script
        print(f"任务 {video_info_dict.keys()} 最终视频生成完成。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return failure_details, video_info_dict, chosen_script




if __name__ == '__main__':
    run()


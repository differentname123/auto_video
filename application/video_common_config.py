# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/14 21:34
:last_date:
    2025/12/14 21:34
:description:
    
"""
import os
import sys

from utils.common_utils import time_to_ms, ms_to_time, read_json


class ResponseStatus:
    """API响应状态常量"""
    SUCCESS = 'success'
    ERROR = 'error'


class ErrorMessage:
    """错误消息常量"""
    EMPTY_REQUEST_BODY = '请求体为空'
    MISSING_REQUIRED_FIELDS = '用户名或视频列表为空'
    PARTIAL_PARSE_FAILURE = '部分视频解析失败，任务未创建。'
    TASK_ALREADY_EXISTS = '任务已存在，无需重复创建。'
    PARSE_NO_METADATA = '解析失败：未能从链接中提取到任何元数据'

user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')

ALLOWED_USER_LIST = user_config.get('allowed_user_list', [])

LOCAL_ORIGIN_URL_ID_INFO_PATH = r'W:/project/python_project/auto_video/videos/origin_url_id_info.json'

class TaskStatus:
    """任务状态常量"""
    PROCESSING = '处理中'
    COMPLETED = '已完成'
    FAILED = '失败'
    DOWNLOADED = '已下载'
    PLAN_GENERATED = '方案已生成'
    TO_UPLOADED = '待投稿'
    UPLOADED = '已投稿'


SINGLE_UPLOAD_COUNT = 10      #一轮循环最大投稿数量
SINGLE_DAY_UPLOAD_COUNT = 20    #单账号单日最大投稿数量


class ERROR_STATUS:
    """
    错误的严重等级
    """
    WARNING = 'warning'  # 警告级别错误
    ERROR = 'error'      # 错误级别
    CRITICAL = 'critical'  # 严重错误




def _configure_third_party_paths() -> None:
    """
    配置第三方库路径。

    TikTokDownloader 库内部使用 'from src...' 的导入方式，
    需要将其根目录添加到 sys.path 中才能正常工作。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    downloader_root = os.path.join(project_root, 'third_party', 'TikTokDownloader')

    if downloader_root not in sys.path:
        sys.path.insert(0, downloader_root)


class VIDEO_ERROR:
    """视频常见错误"""
    DOWNLOAD_FAILED = '下载失败'
    PROCESSING_FAILED = '处理失败'


VIDEO_MAX_RETRY_TIMES = 5  # 视频处理最大重试次数

VIDEO_MATERIAL_BASE_PATH = "W:/project/python_project/auto_video/videos/material"

VIDEO_TASK_BASE_PATH = "W:/project/python_project/auto_video/videos/task"


NEED_REFRESH_COMMENT = False


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
    origin_video_delete_part_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}/{video_id}_origin_delete.mp4")  # 直接下载下来的原始视频，没有任何的加工


    low_origin_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}/{video_id}_origin_low.mp4")  # 直接下载下来的原始视频，只进行了降分辨率和降帧率处理
    static_cut_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_static_cut.mp4")  # 静态剪辑后的视频,也就是去除视频画面没有改变的部分，这个是用于后续的剪辑
    low_resolution_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                             f"{video_id}/{video_id}_low_resolution.mp4")  # 这个是静态剪辑后视频再进行降低分辨率和降低帧率后的数据，用于和大模型交互

    subtitle_box_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/frame/subtitle_box.json")  # 作者字幕遮挡区域的json文件

    cover_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_cover.mp4")  # 封面视频地址

    image_text_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_image_text.mp4")  # 图片文字表情包等添加后的视频地址
    return {
        'origin_video_path': origin_video_path,
        'origin_video_delete_part_path': origin_video_delete_part_path,
        'low_origin_video_path': low_origin_video_path,
        'static_cut_video_path': static_cut_video_path,
        'low_resolution_video_path': low_resolution_video_path,
        'subtitle_box_path':subtitle_box_path,
        'cover_video_path':cover_video_path,
        'image_text_video_path':image_text_video_path
    }

def build_task_video_paths(task_info):
    """
    为任务生成相应的视频路径字典
    :param task_info:
    :return:
    """
    video_id_list = sorted(task_info.get('video_id_list') or [])
    video_id_str = '_'.join(video_id_list)
    str_id = str(task_info.get('_id'))


    final_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id,'final_remake.mp4')
    video_with_title_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id, 'title.mp4')
    all_scene_video_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id, 'all_scene.mp4')
    video_with_bgm_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id, 'bgm.mp4')
    video_with_ending_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id, 'ending.mp4')
    video_with_watermark_output_path = os.path.join(VIDEO_TASK_BASE_PATH, video_id_str, str_id, 'watermark.mp4')



    return {
        'final_output_path': final_output_path,
        'video_with_title_output_path': video_with_title_output_path,
        'all_scene_video_path': all_scene_video_path,
        'video_with_bgm_output_path': video_with_bgm_output_path,
        'video_with_ending_output_path': video_with_ending_output_path,
        'video_with_watermark_output_path': video_with_watermark_output_path
    }


def check_failure_details(failure_details):
    """
    判断failure_details中错误等级是否有超过ERROR的
    :param failure_details:
    :return:
    """
    for video_id, detail in failure_details.items():
        if detail.get('error_level') in [ERROR_STATUS.ERROR, ERROR_STATUS.CRITICAL]:
            print(f"❌ 失败视频失败  结束视频结束 错误视频错误 检测到严重错误，停止后续处理。视频ID: {video_id} 错误详情: {detail}")
            return True
    return False


def correct_owner_timestamps(asr_result, duration):
    """
    对ASR结果列表中speaker为owner的文本时间进行纠正。

    Args:
        asr_result: ASR结果列表。
        duration: 视频总时长（毫秒）。

    Returns:
        带有 'fix_start' 和 'fix_end' 字段的ASR结果列表。
    """
    # 1. 初始化 fix_start 和 fix_end 字段
    for segment in asr_result:
        segment['fix_start'] = segment['start']
        segment['fix_end'] = segment['end']

    # 2. 遍历列表，应用修正逻辑
    for i in range(len(asr_result)):
        current_segment = asr_result[i]

        # 只处理 speaker 为 'owner' 的情况
        if current_segment['speaker'] == 'owner':

            # --- 向前修正逻辑 (修正 start) ---
            # 查看上一个文本
            if i > 0:
                prev_segment = asr_result[i - 1]
                # 如果上一个不是 owner，则尝试移动 start
                if prev_segment['speaker'] != 'owner':
                    gap = current_segment['start'] - prev_segment['end']
                    if gap > 0:
                        # 最多移动500ms
                        movement = min(500, gap / 2)
                        current_segment['fix_start'] = current_segment['start'] - movement

            # --- 向后修正逻辑 (修正 end) ---
            # 查看下一个文本是否存在
            if i < len(asr_result) - 1:
                next_segment = asr_result[i + 1]

                # 如果下一个也是 owner
                if next_segment['speaker'] == 'owner':
                    gap = next_segment['start'] - current_segment['end']
                    if gap > 0:
                        if gap < 1000:
                            # 间隔小于1000ms，取中点
                            midpoint = round(current_segment['end'] + gap / 2)
                            current_segment['fix_end'] = midpoint
                            # 注意：这里直接修正了下一个owner的fix_start
                            next_segment['fix_start'] = midpoint
                        else:
                            # 间隔大于等于1000ms，各自移动，但最多500ms
                            # 同时要保证移动后两者间隔至少500ms
                            movement = min(500, (gap - 500) / 2)
                            if movement > 0:
                                current_segment['fix_end'] = round(current_segment['end'] + movement)
                                # 注意：这里直接修正了下一个owner的fix_start
                                next_segment['fix_start'] = round(next_segment['start'] - movement)

                # 如果下一个不是 owner
                else:
                    gap = next_segment['start'] - current_segment['end']
                    if gap > 0:
                        # 最多移动500ms
                        movement = min(500, gap / 2)
                        current_segment['fix_end'] = current_segment['end'] + movement

            else:
                # --- 新增逻辑：这是最后一个片段 ---
                # 如果当前片段是 owner，且是整个列表的最后一个
                gap = duration - current_segment['end']

                # 只有当视频还有剩余时间时才延伸
                if gap > 0:
                    # 向后延伸最多 500ms，或者直到视频结束 (取较小值)
                    movement = min(500, gap)
                    current_segment['fix_end'] = current_segment['end'] + movement

    return asr_result


def fix_split_time_points(remove_time_segments_ms, split_time_points):
    """
    修复分割点，因为会移除一些时间段，就会影响到分割点的位置
    根据 remove_time_segments 计算 split_time_points 的新位置
    如果 split_time_point 在某个 remove_time_segment 内，则移除该 split_time_point
    :param video_item: 包含 remove_time_segments 和 split_time_points 的字典
    :return: 增加 fixed_split_time_points 字段后的 video_item
    """
    split_time_points_ms = []

    # 2. 关键步骤：按开始时间对移除段进行排序
    # 如果不排序，计算累计删除时长时会出错
    remove_time_segments_ms.sort(key=lambda x: x[0])

    # 3. 解析原始分割点
    for split_time_point in split_time_points:
        split_time_points_ms.append(time_to_ms(split_time_point))

    # 对原始分割点排序（可选，但推荐，方便逻辑处理）
    split_time_points_ms.sort()

    fixed_split_time_points = []

    # 4. 计算新的分割点位置
    for point in split_time_points_ms:
        deleted_duration_before = 0  # 在当前点之前累计被删除的时长
        is_inside_removed_segment = False  # 标记当前点是否在被删除的片段内

        for r_start, r_end in remove_time_segments_ms:
            if point < r_start:
                # 移除段在分割点之后，不影响当前分割点的位置，且由于已排序，后续的移除段也一定在之后
                break
            elif point > r_end:
                # 移除段完全在分割点之前，累加被删除的时长
                deleted_duration_before += (r_end - r_start)
            else:
                # r_start <= point <= r_end
                # 分割点正好落在被移除的时间段内（包含边界），该点应被丢弃
                is_inside_removed_segment = True
                break

        # 只有未落入移除区间的点才保留
        if not is_inside_removed_segment:
            # 新位置 = 原始位置 - 之前所有被删掉的时长
            new_point = point - deleted_duration_before
            # 只有当新位置大于等于0才有效（理论上一定是>=0）
            if new_point >= 0:
                fixed_split_time_points.append(new_point)

    return fixed_split_time_points


BVID_FILE_PATH = r'W:\project\python_project\auto_video\config\bvid_file.json' # 用于存放拉取到的平台视频数据，方便进行播放量统计以及计算时间投稿成功的数量

USER_STATISTIC_INFO_PATH = r'W:\project\python_project\auto_video\config\user_upload_info.json' # 用于存放用户的统计信息


ALL_BILIBILI_EMOTE_PATH = r'W:\project\python_project\auto_video\config\all_emote.json'
USER_BVID_FILE = r'W:\project\python_project\auto_video\config\user_bvid_file.json'
ALL_BVID_FILE = r'W:\project\python_project\auto_video\config\all_bvid_file.json'

COMMENTER_USAFE_FILE = r'W:\project\python_project\auto_video\config\commenter_usage.json'

STATISTIC_PLAY_COUNT_FILE = r'W:\project\python_project\auto_video\config\statistic_play_count.json'


BLOCK_VIDEO_BVID_FILE = r'W:\project\python_project\auto_video\config\block_video_bvid_file.json'  # 用于存放被屏蔽的视频bvid列表
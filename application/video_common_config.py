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


class TaskStatus:
    """任务状态常量"""
    PROCESSING = '处理中'
    COMPLETED = '已完成'
    FAILED = '失败'
    DOWNLOADED = '已下载'
    PLAN_GENERATED = '方案已生成'  # 新增：方案已生成



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
    static_cut_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_static_cut.mp4")  # 静态剪辑后的视频,也就是去除视频画面没有改变的部分，这个是用于后续的剪辑
    low_resolution_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                             f"{video_id}/{video_id}_low_resolution.mp4")  # 这个是静态剪辑后视频再进行降低分辨率和降低帧率后的数据，用于和大模型交互

    subtitle_box_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/frame/subtitle_box.json")  # 作者字幕遮挡区域的json文件

    cover_video_path = os.path.join(VIDEO_MATERIAL_BASE_PATH,
                                         f"{video_id}/{video_id}_cover.mp4")  # 封面视频地址
    return {
        'origin_video_path': origin_video_path,
        'static_cut_video_path': static_cut_video_path,
        'low_resolution_video_path': low_resolution_video_path,
        'subtitle_box_path':subtitle_box_path,
        'cover_video_path':cover_video_path
    }


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
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
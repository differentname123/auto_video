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
from collections import Counter

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

    origin_video_path_blur = os.path.join(VIDEO_MATERIAL_BASE_PATH, f"{video_id}/{video_id}_origin_blur.mp4")  # 直接下载下来的原始视频，没有任何的加工

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
        'origin_video_path_blur': origin_video_path_blur,
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


def correct_consecutive_owner_timestamps(asr_result):
    """
    仅对ASR结果列表中连续为'owner'的片段之间的时间进行纠正。

    此函数只调整相邻的两个'owner'片段之间的间隙，
    不会调整'owner'与非'owner'片段之间的时间，也不会调整'owner'在视频首尾的时间。

    Args:
        asr_result: ASR结果列表，每个元素是一个包含'start', 'end', 'speaker'的字典。

    Returns:
        带有 'fix_start' 和 'fix_end' 字段的ASR结果列表。
    """
    """
    基于已有的 'fixed_start' 和 'fixed_end' 字段，仅对连续为 'owner' 的片段之间的时间进行再次纠正。

    此函数直接读取和修改 'fixed_start' 和 'fixed_end' 字段，
    用于对相邻的两个'owner'片段之间的间隙进行微调。

    Args:
        asr_result: ASR结果列表。每个元素必须已包含 'fixed_start' 和 'fixed_end' 字段。

    Returns:
        在原有对象上修改了 'fixed_start' 和 'fixed_end' 字段后的ASR结果列表。
    """
    # 遍历列表，应用 owner 和 owner 之间的修正逻辑
    # 我们只需要遍历到倒数第二个元素，因为总是要看下一个
    for i in range(len(asr_result) - 1):
        current_segment = asr_result[i]
        next_segment = asr_result[i + 1]

        # 核心条件：当前和下一个片段的 speaker 都必须是 'owner'
        if current_segment['speaker'] == 'owner' and next_segment['speaker'] == 'owner':

            # 使用 fixed_start 和 fixed_end 来计算间隙
            gap = next_segment['fixed_start'] - current_segment['fixed_end']

            # 确保它们之间确实存在时间间隙
            if gap > 0:
                # 调整规则与原始代码保持一致
                if gap < 1000:
                    # 间隔小于1000ms，取中点将两者连接起来
                    midpoint = round(current_segment['fixed_end'] + gap / 2)
                    current_segment['fixed_end'] = midpoint
                    next_segment['fixed_start'] = midpoint
                else:
                    # 间隔大于等于1000ms，各自向中间移动，但最多500ms
                    # 同时要保证移动后两者间隔至少500ms
                    movement = min(500, (gap - 500) / 2)
                    if movement > 0:
                        # 所有计算都基于 fixed_end 和 fixed_start
                        current_segment['fixed_end'] = round(current_segment['fixed_end'] + movement)
                        next_segment['fixed_start'] = round(next_segment['fixed_start'] - movement)

    return asr_result

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


def get_tags_info(data: dict) -> dict:
    """
     从一个结构可能不规范的字典中，极其健壮地总结所有标签（tags）的出现次数。

     该函数经过特殊设计，可以处理各种形式的错误输入而不会抛出异常：
     - 输入的 data 不是字典。
     - 'tags', 'new_scene_info', 'deleted_scene' 键不存在。
     - 'tags', 'new_scene_info', 'deleted_scene' 键对应的值不是列表。
     - 'new_scene_info' 或 'deleted_scene' 列表中的元素不是字典。
     - 场景字典中的 'tags' 键不存在或其值不是列表。
     - 标签列表中包含非字符串等不可哈希的元素。

     在任何错误情况下，它都会安全地跳过问题数据并继续执行，最终返回一个有效的字典。

     Args:
         data (dict): 包含视频信息的源数据，可能不符合预期结构。

     Returns:
         dict: 一个字典，其中 key 是标签名，value 是该标签出现的总次数。
               如果无法提取任何标签，则返回一个空字典 {}。
     """
    # 最终存放所有标签的列表
    all_tags = []

    # 1. 首先，确保输入的数据本身就是一个字典。如果不是，直接返回空字典。
    if not isinstance(data, dict):
        return {}

    # 2. 安全地处理顶层的 'tags'
    # .get() 避免了 KeyError，isinstance() 确保了它是一个列表
    top_level_tags = data.get('tags')
    if isinstance(top_level_tags, list):
        all_tags.extend(top_level_tags)

    # 3. 安全地处理 'new_scene_info' 和 'deleted_scene' 中的 tags
    scene_keys = ['new_scene_info', 'deleted_scene']
    for key in scene_keys:
        scene_list = data.get(key)
        if isinstance(scene_list, list):
            for scene in scene_list:
                if isinstance(scene, dict):
                    scene_tags = scene.get('tags')
                    if isinstance(scene_tags, list):
                        all_tags.extend(scene_tags)
    hashable_tags = [tag for tag in all_tags if isinstance(tag, str)]
    if not hashable_tags:
        return {}

    # 5. 使用 Counter 进行统计，现在绝对安全
    tags_info = Counter(hashable_tags)

    return dict(tags_info)


BVID_FILE_PATH = r'W:\project\python_project\auto_video\config\bvid_file.json' # 用于存放拉取到的平台视频数据，方便进行播放量统计以及计算时间投稿成功的数量

USER_STATISTIC_INFO_PATH = r'W:\project\python_project\auto_video\config\user_upload_info.json' # 用于存放用户的统计信息


ALL_BILIBILI_EMOTE_PATH = r'W:\project\python_project\auto_video\config\all_emote.json'
USER_BVID_FILE = r'W:\project\python_project\auto_video\config\user_bvid_file.json'
ALL_BVID_FILE = r'W:\project\python_project\auto_video\config\all_bvid_file.json'

COMMENTER_USAFE_FILE = r'W:\project\python_project\auto_video\config\commenter_usage.json'

STATISTIC_PLAY_COUNT_FILE = r'W:\project\python_project\auto_video\config\statistic_play_count.json'


BLOCK_VIDEO_BVID_FILE = r'W:\project\python_project\auto_video\config\block_video_bvid_file.json'  # 用于存放被屏蔽的视频bvid列表


ALL_MATERIAL_VIDEO_INFO_PATH = r'W:\project\python_project\auto_video\config\all_material_video_info.json'  # 用于存放所有素材视频的信息，方便进行快速查询

DIG_HOT_VIDEO_PLAN_FILE = r'W:\project\python_project\auto_video\config\dig_hot_video_plan.json'  # 用于存放挖热点视频的计划信息

SNAPSHOT_CACHE_DIR = r'W:\project\python_project\auto_video\videos\snapshot_cache'


def is_contain_owner_speaker(owner_asr_info):
    """
    检查是否包含owner的文本
    """
    if owner_asr_info:
        for asr_info in owner_asr_info:
            speaker = asr_info.get('speaker', 'unknown')
            final_text = asr_info.get('final_text', '').strip()
            if speaker == 'owner' and final_text:
                return True
    return False


def analyze_scene_content(scene_list, owner_asr_info, top_k=3, merge_mode='global'):
    """
    分析场景列表。
    """

    if not scene_list:
        return []

    # --- 1. 内部辅助函数：处理一组场景 ---
    # 这部分函数是修改的重点
    def _process_segment(segment_scenes):
        scene_summaries = []
        all_tags = []
        scene_number_list = []

        if not segment_scenes:
            return {}

        # 1. 确定当前场景段落的整体时间范围
        segment_start = min(s['start'] for s in segment_scenes)
        segment_end = max(s['end'] for s in segment_scenes)

        original_script_list = []
        narration_script_list = []

        # 2. 遍历 ASR，使用中心点判定归属
        if owner_asr_info:
            for asr_item in owner_asr_info:
                asr_start = asr_item.get('start', 0)
                asr_end = asr_item.get('end', 0)
                asr_mid = (asr_start + asr_end) / 2
                if segment_start <= asr_mid < segment_end:
                    text = asr_item.get('final_text', '')
                    speaker = asr_item.get('speaker', '')
                    if speaker == 'owner':
                        narration_script_list.append(text)
                    else:
                        original_script_list.append(text)

        # 遍历 segment_scenes 来提取所需信息
        for scene in segment_scenes:
            # 提取 visual_description
            v_desc = scene.get('scene_summary')
            if v_desc:
                scene_summaries.append(v_desc)

            # 提取 tags
            tags = scene.get('tags', [])
            if tags:
                all_tags.extend(tags)

            # --- 新增代码 Start ---
            # 提取 scene_number_list
            # 使用 .get() 方法以防该字段不存在
            origin_id = scene.get('scene_number')
            if origin_id is not None:  # 确保ID存在才添加
                scene_number_list.append(origin_id)
            # --- 新增代码 End ---

        # 计数并取 Top K
        top_all_tags = [tag for tag, count in Counter(all_tags).most_common(top_k)]

        # 在返回的字典中增加 origin_scene_id_list 字段
        return {
            'scene_summary': scene_summaries,
            'scene_number_list': scene_number_list,
            'tags': top_all_tags,
            'narration_script_list': narration_script_list,
            'original_script_list': original_script_list,
        }

    # --- 2. 根据模式构建 Segments ---
    # 这部分逻辑保持不变
    segments = []

    if merge_mode == 'global':
        segments.append(scene_list)

    elif merge_mode == 'none':
        for scene in scene_list:
            segments.append([scene])

    elif merge_mode == 'smart':
        if not scene_list:
            return []

        buffer = []
        for scene in scene_list:
            if not buffer:
                buffer.append(scene)
                continue

            is_adjustable = scene.get('sequence_info', {}).get('is_adjustable', False)

            if is_adjustable:
                segments.append(buffer)
                buffer = [scene]
            else:
                buffer.append(scene)

        if buffer:
            segments.append(buffer)

    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")

    # --- 3. 处理结果 ---
    # 这部分逻辑保持不变
    result_list = []
    for segment in segments:
        result_list.append(_process_segment(segment))

    return result_list

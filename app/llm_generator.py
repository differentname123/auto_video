# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 22:40
:last_date:
    2025/12/15 22:40
:description:
    主要是调用大模型生成内容的业务代码
"""
import time

from utils.auto_web.gemini_auto import generate_gemini_content_playwright
from utils.common_utils import read_file_to_str, string_to_object, time_to_ms
from utils.video_utils import probe_duration


def check_logical_scene(logical_scene_info: dict, video_duration_ms: int) -> tuple[bool, str]:
    """
     检查 logical_scene_info 的有效性，并在检查过程中将时间字符串转换为毫秒整数（in-place-modification）。

     Args:
         logical_scene_info (dict): 包含 'new_scene_info' 和 'deleted_scene' 的字典。
                                    此字典中的时间格式将被直接修改。
         video_duration_ms (int): 视频总时长（毫秒）。

     Returns:
         tuple[bool, str]: 一个元组，第一个元素是检查结果 (True/False)，
                            第二个元素是具体的检查信息。
     """
    # 临时列表，用于存储转换后的时间信息以进行排序和连续性检查
    all_scenes_for_sorting = []

    # 待处理的场景列表（new_scene_info 和 deleted_scene）
    scene_lists_to_process = [
        logical_scene_info.get('new_scene_info', []),
        logical_scene_info.get('deleted_scene', [])
    ]
    deleted_scene = logical_scene_info.get('deleted_scene', [])
    new_scene_info = logical_scene_info.get('new_scene_info', [])
    # 检查 deleted_scene 中的场景数量，不能超过3个
    if len(deleted_scene) > 3:
        return False, "检查失败：deleted_scene 中的场景数量超过3个，可能存在误操作。"

    # 检查 new_scene_info 中的场景数量，不能超过15个
    if len(new_scene_info) > 15:
        return False, "检查失败：new_scene_info 中的场景数量超过15个，可能存在误操作。"
    # 1. 遍历并转换所有场景，同时进行初步检查
    for scene_list in scene_lists_to_process:
        for i, scene in enumerate(scene_list):
            try:
                start_str, end_str = scene['start'], scene['end']

                # 确保 start 和 end 都是字符串，如果已经是数字则跳过转换
                if not isinstance(start_str, str) or not isinstance(end_str, str):
                    return False, f"检查失败：场景 {i + 1} 的时间格式不正确，期望是字符串但不是。场景: {scene}"

                start_ms = time_to_ms(start_str)
                end_ms = time_to_ms(end_str)

                # --- 核心修改步骤 ---
                # 直接在原始字典上更新值为毫秒整数
                scene['start'] = start_ms
                scene['end'] = end_ms
                # --------------------

                # 要求1：start < end
                if start_ms >= end_ms:
                    return False, f"检查失败：场景 {i + 1} 的开始时间 {start_str} ({start_ms}ms) 必须小于结束时间 {end_str} ({end_ms}ms)。"

                # 要求3：在视频时长范围内
                if not (0 <= start_ms <= video_duration_ms and 0 <= end_ms <= video_duration_ms + 2000):
                    return False, f"检查失败：场景 {i + 1} 的时间范围 [{start_str}, {end_str}] 超出视频时长 [0, {video_duration_ms}ms]。"

                # 将信息存入临时列表，用于后续排序和检查
                all_scenes_for_sorting.append({
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'original_start': start_str,  # 保留原始字符串用于错误报告
                    'original_end': end_str,
                })

            except (ValueError, TypeError) as e:
                return False, f"检查失败：场景 {i + 1} 的时间格式无效。原始场景: {scene}, 错误: {e}"

    # 如果视频时长为0，且没有场景，这是有效情况
    if not all_scenes_for_sorting and video_duration_ms == 0:
        return True, "OK. 视频时长为0，且没有场景。"

    if not all_scenes_for_sorting:
        return False, "检查失败：未提供任何场景信息，但视频时长大于0。"

    # 2. 按开始时间排序，为连续性检查做准备
    all_scenes_for_sorting.sort(key=lambda x: x['start_ms'])

    # 3. 检查时间轴的完整性
    if all_scenes_for_sorting[0]['start_ms'] != 0:
        return False, f"检查失败：时间轴不连续。第一个场景从 {all_scenes_for_sorting[0]['original_start']} 开始，而不是从 00:00.000 开始。"

    if abs(all_scenes_for_sorting[-1]['end_ms'] - video_duration_ms) > 2000:
        return False, f"检查失败：时间轴不完整。最后一个场景在 {all_scenes_for_sorting[-1]['original_end']} ({all_scenes_for_sorting[-1]['end_ms']}ms) 结束，与视频总时长 {video_duration_ms}ms 不匹配。"

    # 4. 遍历排序后的场景，检查重叠和间隔
    for i in range(len(all_scenes_for_sorting) - 1):
        current = all_scenes_for_sorting[i]
        next_s = all_scenes_for_sorting[i + 1]

        # 要求2：不能重叠
        if current['end_ms'] > next_s['start_ms']:
            return False, (f"检查失败：场景之间存在重叠。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 重叠。")

        # 要求4：不能有间隔
        if current['end_ms'] < next_s['start_ms']:
            return False, (f"检查失败：场景之间存在间隔。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 之后与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 之前有时间空缺。")

    # 为logical_scene_info增加一个字段，表示scene_number
    scene_number = 1
    for scene_list in scene_lists_to_process:
        for scene in scene_list:
            scene['scene_number'] = scene_number
            scene_number += 1

    return True, "检查并转换成功：所有场景的时间有效、连续且无重叠，格式已更新为毫秒。"

def gen_base_prompt(video_path, video_info):
    """
    生成基础的通用提示词
    """
    duration = probe_duration(video_path)
    video_title = video_info.get('base_info', {}).get('video_title', '')

    base_prompt = f"\n视频相关信息如下:\n视频时长为: {duration}"
    if video_title:
        base_prompt += f"\n视频描述为: {video_title}"
    # if comment_list:
    #     base_prompt += f"\n视频已有评论列表 (数字表示已获赞数量): {comment_list}"
    return base_prompt


def gen_logical_scene_llm(video_path, video_info):
    """
    生成新的视频方案
    """
    base_prompt = "base_prompt"
    log_pre = f"{video_path} 逻辑性场景划分 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return "获取视频时长失败", None

    retry_delay = 10
    max_retries = 3
    prompt_file_path = './prompt/视频场景逻辑切分只根据视频内容.txt'
    full_prompt = read_file_to_str(prompt_file_path)
    full_prompt += f'\n{base_prompt}'
    error_info = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries}) {log_pre}")
            error_info, raw = generate_gemini_content_playwright(full_prompt, file_path=video_path, model_name="gemini-2.5-pro")

            logical_scene_info = string_to_object(raw)
            check_result, check_info = check_logical_scene(logical_scene_info, video_duration_ms)
            if not check_result:
                raise ValueError(f"逻辑性场景划分检查未通过: {check_info} {raw}")
            return None, logical_scene_info
        except Exception as e:
            error_str = f"{error_info} {str(e)}"
            print(f"生成逻辑性场景划分失败 (尝试 {attempt}/{max_retries}): {error_str} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return error_str, None  # 达到最大重试次数后返回 None

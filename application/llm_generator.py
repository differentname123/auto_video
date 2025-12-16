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
from utils.video_utils import probe_duration, get_scene


def check_logical_scene(logical_scene_info: dict, video_duration_ms: int, max_scenes) -> tuple[bool, str]:
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
    if len(new_scene_info) > max_scenes and max_scenes > 0:
        return False, f"检查失败：new_scene_info 中的场景数量超过{max_scenes}个，可能存在误操作。"
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


def fix_logical_scene_info(merged_timestamps, logical_scene_info, max_delta_ms=1000):
    """
     将 logical_scene_info 中的每个 scene 的 start/end 对齐到 camera shot。
     对齐原则：
     1. 寻找 max_delta_ms 毫秒容差范围内的所有 camera shot。
     2. 在这些候选中，选择出现次数最多的那个。
     3. 如果出现次数相同，则选择与原始时间戳差值最小（最近）的那个。
     打印每个时间戳调整前后的对比。返回修改后的 logical_scene_info。

     Args:
         video_path (str): 视频文件路径。
         scenes (list of list): camera shot 信息，每个元素是 [timestamp_ms, count]。
         logical_scene_info (dict): 包含 'new_scene_info' 的逻辑场景信息。
         output_dir (str): 输出目录，用于保存调试帧。
         max_delta_ms (int): 查找候选 camera shot 的最大时间差（毫秒）。
     """

    camera_shots_with_counts = [c for c in merged_timestamps if c and c[0] is not None and c[1] > 0]
    if not camera_shots_with_counts:
        print("⚠️ 无有效 camera_shot 时间戳，跳过调整。")
        return logical_scene_info

    for i, scene in enumerate(logical_scene_info.get('new_scene_info', [])):
        for key in ('start', 'end'):
            orig_ts = scene.get(key)
            if orig_ts is None:
                print(f"[Scene {i}] {key}: 无法解析原始时间 ({orig_ts})，跳过。")
                continue

            # 步骤 1: 筛选候选者
            candidates = [
                shot for shot in camera_shots_with_counts
                if abs(shot[0] - orig_ts) <= max_delta_ms
            ]

            if not candidates:
                print(f"[Scene {i}] {key}: 保持不变 {orig_ts} (在 {max_delta_ms}ms 范围内无候选 camera shot)")
            else:
                # 步骤 2: 使用 min() 和一个计算分数的 key 来找到最佳匹配
                # key 返回一个元组，min() 会依次比较元组中的元素
                # 1. 主要比较分数: (差值 / 次数)
                # 2. 次要比较（分数相同时）: 差值本身，确保选择更近的
                def calculate_key(shot):
                    diff = abs(shot[0] - orig_ts)
                    count = shot[1]
                    # 安全起见，虽然前面过滤了，但这里处理 count=0 的情况
                    score = diff / count if count > 0 else float('inf')
                    return (score, diff)

                best_shot = min(candidates, key=calculate_key)

                new_ts = int(best_shot[0])
                count = best_shot[1]
                diff = abs(new_ts - orig_ts)
                score = diff / count if count > 0 else float('inf')

                scene[key] = new_ts
                print(f"[Scene {i}] {key}: {orig_ts} -> {new_ts} (最佳分={score:.2f}, 次数={count}, 差值={diff}ms, 已调整)")

                # 如果需要调试，可以取消下面的注释
                # save_frames_around_timestamp(video_path, ms_to_time(orig_ts), 5, str(os.path.join(output_dir, 'orig', str(orig_ts))))
                # save_frames_around_timestamp(video_path, ms_to_time(new_ts), 5, str(os.path.join(output_dir,'closest', str(new_ts))))

    return logical_scene_info

def gen_logical_scene_llm(video_path, video_info):
    """
    生成新的视频方案
    """
    base_prompt = gen_base_prompt(video_path, video_info)
    max_scenes = video_info.get('base_info', {}).get('max_scenes', 0)
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
    if max_scenes > 0:
        full_prompt += f'\n请将生成的场景数量控制在 {max_scenes} 个以内。'

    error_info = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries}) {log_pre}")
            error_info, raw = generate_gemini_content_playwright(full_prompt, file_path=video_path, model_name="gemini-2.5-pro")

            logical_scene_info = string_to_object(raw)
            check_result, check_info = check_logical_scene(logical_scene_info, video_duration_ms, max_scenes)
            if not check_result:
                raise ValueError(f"逻辑性场景划分检查未通过: {check_info} {raw}")

            merged_timestamps = get_scene(video_path, min_final_scenes=max_scenes)
            logical_scene_info = fix_logical_scene_info(merged_timestamps, logical_scene_info, max_delta_ms=1000)

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

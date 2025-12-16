# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 18:13
:last_date:
    2025/12/15 18:13
:description:
    
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils.common_utils import is_valid_target_file_simple, read_json, save_json, time_to_ms
from utils.split_scenes import split_scenes_json


def probe_video_new(path):
    """用 ffprobe 获取视频的宽、高、帧率和 SAR。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,sample_aspect_ratio",
        "-of", "json",
        path
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(proc.stdout)["streams"][0]
    # 解析帧率，比如 "30000/1001" -> float
    num, den = map(int, info["r_frame_rate"].split("/"))
    fps = num / den
    sar = info.get("sample_aspect_ratio", "1:1")
    w, h = info["width"], info["height"]
    return w, h, fps, sar

def reduce_video_size_robust(
        input_path: str,
        output_path: str,
        target_width: int = 480,
        crf: int = 23,
        target_fps: int = 30,
        force_fps: bool = False,
        faststart: bool = True
) -> bool:
    """
    使用 FFmpeg 优化地处理视频。
    - 只有当输入文件 > 20MB 时才进行处理，否则直接复制。
    - 只有当目标宽度 < 原始宽度时，才进行等比缩放（避免放大）。
    - 如果无法复制音频，操作将直接失败，不会产生静音视频。
    - 返回 True 表示成功，False 表示失败。
    """
    # 检查 ffmpeg/ffprobe
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("错误: FFmpeg/ffprobe 未找到。请确保已安装并加入 PATH。")
        return False

    if not os.path.exists(input_path):
        print(f"错误: 输入文件未找到: {input_path}")
        return False

    # 检查文件大小，小于等于20MB则直接复制
    size_threshold_mb = 10
    threshold_bytes = size_threshold_mb * 1024 * 1024
    file_size_bytes = os.path.getsize(input_path)

    if file_size_bytes <= threshold_bytes:
        print(
            f"文件大小 {(file_size_bytes / (1024 * 1024)):.2f} MB <= 阈值 {size_threshold_mb} MB，不进行压缩，直接复制。")
        try:
            shutil.copy2(input_path, output_path)
            print(f"成功：文件已复制到 {output_path}")
            return True
        except Exception as e:
            print(f"错误：复制文件时失败: {e}")
            return False

    print(f"文件大小 {(file_size_bytes / (1024 * 1024)):.2f} MB > 阈值 {size_threshold_mb} MB，开始压缩...")
    print(f"开始处理: {input_path}")

    # 探测视频信息
    w, h, fps, sar = probe_video_new(input_path)
    if w is None or h is None:
        print("[warn] 无法探测视频宽高，将按设定缩放，但请注意可能发生放大。")
    else:
        print(f"[probe] 宽x高: {w}x{h}, SAR: {sar}, fps: {fps}")

    # 帧率处理逻辑 (保持不变)
    reduce_fps = False
    if force_fps or (fps is not None and fps > float(target_fps)):
        reduce_fps = True

    # 1. 定义是否需要调整的条件
    needs_scaling = target_width != -1 and w is not None and target_width < w
    needs_fps_reduction = force_fps or (fps is not None and fps > float(target_fps))

    # 2. 如果两个条件都不满足，则直接复制并返回
    if not needs_scaling and not needs_fps_reduction:
        print("[跳过] 视频分辨率和帧率均无需调整，直接复制文件。")
        try:
            shutil.copy2(input_path, output_path)
            print(f"成功：文件已复制到 {output_path}")
            return True
        except Exception as e:
            print(f"错误：复制文件时失败: {e}")
            return False

    # --- 修改 ---: 构建 vf 过滤器，集成智能缩放逻辑
    vf_parts = []
    # 只有当 target_width 有效，且我们知道原始宽度，并且目标宽度小于原始宽度时，才缩放。
    if target_width != -1 and w is not None and target_width < w:
        print(f"[scale] 目标宽度 {target_width} < 原始宽度 {w}，将进行等比缩放。")
        # 使用 -2 确保输出高度为偶数，提高兼容性
        vf_parts.append(f"scale={target_width}:-2")
    else:
        print("[scale] 目标宽度不小于原始宽度或无法探测，将保持原分辨率。")

    if reduce_fps:
        vf_parts.append(f"fps={target_fps}")
        print(f"[fps] 检测到输入帧率 {fps:.6g} > {target_fps}，将降帧到 {target_fps} fps。")
    else:
        print(f"[fps] 输入帧率 {fps:.6g} <= {target_fps}，保持原帧率。")

    vf_arg = []
    if vf_parts:
        vf_arg = ["-vf", ",".join(vf_parts)]

    # 基础命令构建 (保持不变)
    base_command = [
        "ffmpeg", "-i", input_path,
        *vf_arg,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
    ]
    if faststart:
        base_command += ["-movflags", "+faststart"]

    # 严格的音频处理：只尝试一次，失败则直接返回 False
    print(f"{input_path}[原地压缩处理中] 尝试转换视频并复制音频流...")
    command_with_audio = base_command + ["-c:a", "copy", "-y", output_path]
    try:
        print(f"[cmd] {' '.join(command_with_audio)}")
        p = subprocess.Popen(command_with_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding='utf-8')
        _, err = p.communicate()
        if p.returncode == 0:
            print(f"成功：输出文件已保存为 {output_path}")
            return True
        else:
            print("[失败] FFmpeg处理失败（可能是音频流不兼容无法复制）。操作已中止。")
            print("FFmpeg stderr:\n", err)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"[清理] 已删除失败的输出文件: {output_path}")
                except OSError as e:
                    print(f"[警告] 无法删除失败的输出文件: {e}")
            return False
    except Exception as e:
        print(f"[错误] 执行 FFmpeg 命令时出错: {e}")
        return False



def find_motion_bbox(video_path, start_frame=60, end_frame_offset=60, num_samples=20, motion_threshold=30, padding=10):
    """
    分析视频指定片段，通过均匀采样固定数量的帧来找到运动区域的边界框。

    :param video_path: 视频文件路径
    :param start_frame: 开始分析的绝对帧号 (默认为0)
    :param end_frame_offset: 从视频末尾向前偏移的帧数。0表示分析到最后一帧。(默认为0)
    :param num_samples: 在指定范围内均匀采样的帧数 (默认为20)
    :param motion_threshold: 像素差异多大时算作运动
    :param padding: 在计算出的边界框外围增加的像素边距
    :return: (x, y, w, h) 的边界框元组，如果失败则返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}", file=sys.stderr)
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频信息: {frame_width}x{frame_height}, {total_frames} 帧, {fps:.2f} FPS")
    if total_frames < 2:
        print("错误: 视频文件帧数不足，无法进行分析。", file=sys.stderr)
        cap.release()
        return None

    # --- 新逻辑: 计算实际的分析范围 ---
    actual_end_frame = total_frames - end_frame_offset

    # --- 参数有效性检查 ---
    if start_frame < 0 or start_frame >= total_frames:
        print(f"错误: 起始帧 {start_frame} 超出范围 (0-{total_frames - 1})。", file=sys.stderr)
        return None
    if actual_end_frame <= start_frame:
        print(f"错误: 计算出的结束帧({actual_end_frame})必须大于起始帧({start_frame})。", file=sys.stderr)
        return None
    if num_samples < 2:
        print(f"错误: 采样帧数 {num_samples} 必须至少为2。", file=sys.stderr)
        return None

    # --- 新逻辑: 使用 linspace 生成均匀分布的采样帧索引 ---
    # np.linspace 包含端点，所以我们从 start_frame 到 actual_end_frame - 1
    sample_indices = np.linspace(start_frame, actual_end_frame - 1, num=num_samples, dtype=int)
    print(f"将在第 {start_frame} 帧到第 {actual_end_frame} 帧之间，均匀采样 {len(sample_indices)} 帧进行分析。")

    motion_accumulator = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # --- 新逻辑: 处理第一个采样帧来初始化 prev_gray ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_indices[0])
    ret, prev_frame = cap.read()
    if not ret:
        print(f"错误: 无法读取帧 {sample_indices[0]}。", file=sys.stderr)
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # --- 新逻辑: 循环遍历剩余的采样帧 ---
    for i, frame_index in enumerate(sample_indices[1:]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"\n警告: 无法读取帧 {frame_index}，跳过此帧。")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_accumulator = cv2.bitwise_or(motion_accumulator, thresh)
        prev_gray = gray  # 更新 prev_gray 以便下次比较

        # 进度条基于已处理的采样帧数
        progress = ((i + 2) / len(sample_indices)) * 100
        sys.stdout.write(f"\r正在分析... {progress:.2f}% (已处理 {i + 2}/{len(sample_indices)} 帧)")
        sys.stdout.flush()

    print("\n分析完成！")
    cap.release()

    points = cv2.findNonZero(motion_accumulator)
    if points is None:
        print("警告：在指定片段内未检测到任何运动。将返回整个视频区域。")
        return 0, 0, frame_width, frame_height

    # 后续处理与之前相同
    x, y, w, h = cv2.boundingRect(points)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_width - x, w + 2 * padding)
    h = min(frame_height - y, h + 2 * padding)
    w = w + (w % 2)
    h = h + (h % 2)
    if x + w > frame_width: w = frame_width - x
    if y + h > frame_height: h = frame_height - y

    return ((x, y, w, h), frame_width, frame_height)

def crop_video(input_path, output_path, bbox, crf=23):
    """
    使用 FFmpeg 调用来裁剪视频，并使用 CRF 控制输出质量和文件大小。

    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param bbox: (x, y, w, h) 的边界框
    :param crf: Constant Rate Factor (CRF)。范围 0-51，默认 23。
                数值越小，质量越高，文件越大。
    """
    x, y, w, h = bbox
    print(f"\n检测到的活动区域 (x, y, w, h): ({x}, {y}, {w}, {h})")

    # 构建 FFmpeg 命令列表
    command = [
        'ffmpeg',
        '-y',  # 自动覆盖输出文件
        '-i', input_path,
        '-vf', f'crop={w}:{h}:{x}:{y}',
        '-c:v', 'libx264',  # 指定视频编码器为 H.264
        '-crf', str(crf),   # 指定质量因子，23 是一个很好的平衡值
        '-preset', 'ultrafast',# 预设，影响编码速度和压缩率的平衡。'medium' 是默认值，通常无需更改。
        '-c:a', 'copy',     # 直接复制音频流，不做重新编码
        output_path
    ]

    print("\n将要执行的 FFmpeg 命令:")
    # 为了清晰地打印命令
    command_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in command)
    print(command_str)
    print("-" * 50)

    try:
        # 执行命令
        print("正在执行裁剪，请稍候...")
        # 使用 PIPE 捕获输出，可以在出错时提供更详细的信息
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True # Python 3.7+
        )
        print(f"\n裁剪成功！输出文件已保存至: {output_path}")

    except FileNotFoundError:
        print("错误: 'ffmpeg' 命令未找到。", file=sys.stderr)
        print("请确保 FFmpeg 已安装并配置在系统的 PATH 环境变量中。", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"错误: FFmpeg 执行失败，返回码 {e.returncode}", file=sys.stderr)
        print("\n--- FFmpeg 标准输出 ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- FFmpeg 错误输出 ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)


def remove_static_background_video(video_path, area_threshold_ratio=0.9, **kwargs):
    """
    分析视频中的运动区域，如果运动区域显著小于整个画面，则进行裁剪。

    :param video_path: 待处理的视频文件路径
    :param area_threshold_ratio: 面积阈值比例。当运动区域面积小于原面积的该比例时，触发裁剪。
                                 例如, 0.8 表示小于80%。
    :param kwargs: 传递给 find_motion_bbox 的其他参数，如 start_frame, num_samples 等。
    :return: 元组 (was_cropped, final_path)。
             was_cropped: 布尔值，True表示已裁剪，False表示未裁剪。
             final_path: 最终视频文件的路径（可能是裁剪后的新路径，也可能是原始路径）。
    """
    print(f"\n--- 开始裁剪静态区域处理视频: {video_path} ---")

    # 1. 查找运动边界框
    analysis_result = find_motion_bbox(video_path, **kwargs)

    if analysis_result is None:
        print("分析失败，无法获取边界框。")
        return (False, video_path)

    bbox, original_w, original_h = analysis_result
    x, y, w, h = bbox

    # 2. 判断是否需要裁剪
    original_area = original_w * original_h
    crop_area = w * h

    # 避免除以零的错误
    if original_area == 0:
        print("视频原始面积为0，无法计算比例。")
        return (False, video_path)

    current_ratio = crop_area / original_area
    print(f"运动区域面积占总面积的 {current_ratio:.2%}")

    # 条件：当前比例小于阈值，并且裁剪区域不等于整个视频（这是 find_motion_bbox 的回退情况）
    if current_ratio < area_threshold_ratio and (w, h) != (original_w, original_h):
        print(f"面积比例 ({current_ratio:.2%}) 小于阈值 ({area_threshold_ratio:.2%})，将执行裁剪。")

        # 构造输出文件名，例如 a.mp4 -> a_crop.mp4
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_crop{ext}"

        # 3. 执行裁剪
        crop_video(video_path, output_path, bbox)

        return (True, output_path)
    else:
        print(f"面积比例不小于阈值或与原尺寸相同，无需裁剪。")
        return (False, video_path)



def reduce_and_replace_video(video_path: str, **kwargs) -> bool:
    """
    就地减小视频文件的大小。

    此函数会尝试压缩指定的视频文件。只有在压缩成功 **并且**
    生成的新文件严格小于原始文件时，才会用新文件替换原始文件。
    如果压缩失败或文件没有变小，原始文件将保持不变。

    Args:
        video_path (str): 要处理的视频文件的路径。
        **kwargs: 传递给 `reduce_video_size_robust` 的其他参数,
                  例如 target_width, crf, target_fps 等。

    Returns:
        bool: 如果成功减小并替换了文件，则返回 True，否则返回 False。
    """
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在: {video_path}")
        return False

    # 1. 创建一个唯一的临时文件名
    path_obj = Path(video_path)
    temp_path = str(path_obj.with_suffix(f".temp_compression{path_obj.suffix}"))
    print("\n")
    print("-" * 50)
    print(f"开始为 '{path_obj.name}' 尝试就地压缩...")

    try:
        # 2. 调用核心压缩函数，输出到临时文件
        success = reduce_video_size_robust(
            input_path=video_path,
            output_path=temp_path,
            **kwargs
        )

        # 3. 如果 FFmpeg 处理失败，直接返回 False
        if not success:
            print(f"压缩过程失败，原始文件 '{video_path}' 保持不变。")
            return False

        # 4. FFmpeg 成功，现在比较文件大小
        original_size = os.path.getsize(video_path)
        new_size = os.path.getsize(temp_path)

        # 转换为 MB 以便阅读
        original_size_mb = original_size / (1024 * 1024)
        new_size_mb = new_size / (1024 * 1024)

        # 5. 只有新文件更小时才替换
        if new_size < original_size:
            print(f"压缩有效！大小从 {original_size_mb:.2f} MB 减小到 {new_size_mb:.2f} MB。")
            shutil.move(temp_path, video_path)
            print(f"成功：原始文件 '{video_path}' 已被替换。")
            # shutil.move 已经重命名，temp_path 不再存在，finally 块不会出错
            return True
        else:
            print(f"压缩后文件未变小（或变大）。原始大小: {original_size_mb:.2f} MB, 新大小: {new_size_mb:.2f} MB。")
            print(f"保留原始文件 '{video_path}'。")
            return False

    finally:
        # 6. 清理：无论发生什么，都尝试删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"临时文件已清理: {temp_path}")
        print("-" * 50)


def probe_duration(path):
    """返回视频时长（秒）"""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out)

def merge_scene_timestamps(scene_dict, min_count=3, count_by_threshold=True):
    """
    合并不同阈值下的场景时间点。

    行为说明：
      - kept_sorted: 返回**未过滤**的时间戳及其出现次数，类型为列表 [(timestamp_str, count), ...]，
                    并按真实时间升序排序。
      - pairs: 仍然基于满足 min_count 的时间戳构建相邻配对区间，格式为
               {'场景1': {'start': s, 'end': e, 'duration': ms}, ...}

    参数:
      scene_dict: 嵌套字典，外层 key 为阈值（例如 40,50,60），内层为场景名 -> [start, end]
      min_count: 只把出现次数 >= min_count 的时间戳用于构建 pairs（默认 3）
      count_by_threshold: True 时在每个阈值内先去重再计数（推荐），
                          False 时把所有出现次数都计入（同阈值重复会被计多次）

    返回:
      (kept_sorted, pairs)
        - kept_sorted: [(timestamp_str, count), ...] （未过滤，按时间升序）
        - pairs: dict，键为 '场景1','场景2',...，值为 {'start','end','duration'}
    """
    from collections import Counter

    counts = Counter()

    for thr, scenes in scene_dict.items():
        ts_list = []
        for scene_name, bounds in scenes.items():
            if not bounds:
                continue
            # 期望 bounds = [start, end]
            ts_list.extend(time_to_ms(t) for t in bounds if isinstance(t, str) and t.strip())
        if count_by_threshold:
            for ts in set(ts_list):
                counts[ts] += 1
        else:
            for ts in ts_list:
                counts[ts] += 1

    # kept_sorted: 未过滤，包含每个时间戳的出现次数，按时间升序
    kept_sorted = sorted(counts.items(), key=lambda kv: time_to_ms(kv[0]))  # [(ts, count), ...]

    # 下面为构建 pairs：仍然只使用出现次数 >= min_count 的时间戳（按时间排序）
    filtered_ts = [ts for ts, c in counts.items() if c >= min_count]
    filtered_sorted = sorted(filtered_ts, key=time_to_ms)

    pairs = {}
    n = len(filtered_sorted)
    if n == 0:
        return kept_sorted, pairs
    if n == 1:
        start = filtered_sorted[0]
        end = filtered_sorted[0]
        td = time_to_ms(end) - time_to_ms(start)
        pairs['场景1'] = {
            'start': start,
            'end': end,
            'duration': td
        }
        return kept_sorted, pairs

    for i in range(n - 1):
        key = f"场景{i + 1}"
        start = filtered_sorted[i]
        end = filtered_sorted[i + 1]
        td = time_to_ms(end) - time_to_ms(start)
        pairs[key] = {
            'start': start,
            'end': end,
            'duration': td
        }

    return kept_sorted, pairs


def get_scene(video_path, min_final_scenes=20):
    # --- 新增的配置项 ---
    initial_thresholds = [30,40, 50,60, 70]  # 初始阈值列表
    adjustment_step = 10  # 每次调整的步长
    max_attempts = 3  # 最多尝试次数
    # output_dir的逻辑为和video_path同目录下面的scene
    output_dir = os.path.join(os.path.dirname(video_path), 'scenes')



    # --- 将你原有的逻辑放入一个循环中 ---
    thresholds = list(initial_thresholds)  # 创建一个可修改的副本
    kept_sorted = []  # 初始化一个空列表
    all_scene_info_dict = {}
    merged_timestamps_path = os.path.join(output_dir, 'merged_timestamps.json')

    if is_valid_target_file_simple(merged_timestamps_path, 1):
        kept_sorted = read_json(merged_timestamps_path)
        print(f"检测到已存在的合并时间戳文件，直接加载返回，场景数量为: {len(kept_sorted)} {kept_sorted}")
        return kept_sorted


    for attempt in range(max_attempts):
        print(f"--- 第 {attempt + 1}/{max_attempts} 次尝试 ---")
        print(f"当前使用的阈值列表: {thresholds}")

        # 你的核心逻辑基本不变，只是把固定的 [30, 50, 70] 换成了可变的 thresholds
        for high_threshold in thresholds:
            start_time = time.time()
            scene_info_file = os.path.join(output_dir, 'scenes', f'scenes_{high_threshold}', 'scene_info.json')


            if is_valid_target_file_simple(scene_info_file):
                all_scene_info_dict[high_threshold] = read_json(scene_info_file)
                continue

            scene_info_dict = split_scenes_json(
                video_path,
                high_threshold=high_threshold,
                min_scene_len=25,
            )
            print(
                f"阈值为 {high_threshold} 场景信息字典已生成。共 {len(scene_info_dict)} 个场景。 耗时: {time.time() - start_time:.2f} 秒\n")

            save_json(scene_info_file, scene_info_dict)
            all_scene_info_dict[high_threshold] = scene_info_dict

        # 合并逻辑不变
        kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=3)
        print(f"场景识别合并完成: 本次尝试生成场景数量为: {len(kept_sorted)}")

        # --- 新增的核心判断逻辑 ---
        if len(kept_sorted) >= min_final_scenes:
            print(f"成功！生成的场景数量 ({len(kept_sorted)}) 满足要求 (>= {min_final_scenes})。")
            break  # 达到目标，跳出重试循环
        else:
            print(f"警告：生成的场景数量 ({len(kept_sorted)}) 过少，不满足要求 (>= {min_final_scenes})。")
            # 如果不是最后一次尝试，则降低阈值准备重试
            if attempt < max_attempts - 1:
                print(f"准备降低阈值后重试...")
                # 将列表中的每个阈值都减小，并确保不低于某个下限（例如10）
                thresholds = [max(10, t - adjustment_step) for t in thresholds]
                # 如果阈值已经降到最低无法再降，也提前退出
                if all(t == 10 for t in thresholds):
                    print("阈值已降至最低，无法继续。")
                    break
            else:
                print("已达到最大尝试次数，将使用当前结果。")

    # 循环结束后的收尾工作（保存文件等）
    print(f"\n--- 最终处理结果 ---")
    print(f"最终场景数量为: {len(kept_sorted)} {kept_sorted}")
    save_json(
        merged_timestamps_path,
        kept_sorted)

    return kept_sorted
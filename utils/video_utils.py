# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 18:13
:last_date:
    2025/12/15 18:13
:description:
    
"""
import tempfile
import uuid
from typing import Union, List

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

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

def _format_time_for_ffmpeg(seconds: float) -> str:
    # 辅助函数：将秒数格式化回 FFmpeg 需要的格式
    return f"{seconds:.3f}"

def _format_time(time_value: Union[str, int, float]) -> str:
    """将时间值格式化为 ffmpeg 兼容的字符串。"""
    if isinstance(time_value, str):
        # 如果是字符串，直接返回，假设用户提供了正确的格式
        return time_value
    if isinstance(time_value, (int, float)):
        # 如果是数字，我们假定单位是毫秒
        # 将其转换为秒，并格式化为 SS.mmm
        return f"{time_value / 1000:.3f}"
    raise TypeError(
        f"不支持的时间格式: {type(time_value)}。请输入字符串或代表毫秒的数字。"
    )

def clip_video_ms(
        input_path: str,
        start_time: Union[str, int, float],
        end_time: Union[str, int, float],
        output_path: str
):
    """
    使用 ffmpeg 精确截取指定时间段的视频片段。

    此函数通过重新编码视频来确保截取的起点绝对精确，从而避免
    因关键帧问题导致的“有声无画”现象。

    优点:
    - 时间点精确到毫秒。
    - 保证输出的视频文件能正常播放。

    缺点:
    - 处理速度比流复制慢，因为它需要CPU进行视频编码计算。

    Args:
        input_path (str): 输入视频文件的完整路径。
        start_time (Union[str, int, float]):
            截取片段的开始时间 (毫秒数或 "HH:MM:SS.mmm" 字符串)。
        end_time (Union[str, int, float]):
            截取片段的结束时间，格式同 start_time。
        output_path (str): 输出视频文件的保存路径。

    Returns:
        bool: 如果截取成功返回 True，否则返回 False。
        str: 返回 ffmpeg 的输出信息（成功时）或错误信息（失败时）。
    """
    current_time = time.time()
    try:
        start_formatted = _format_time(start_time)
        end_formatted = _format_time(end_time)
    except TypeError as e:
        print(e)
        return False, str(e)

    # 【修改部分】构建 ffmpeg 命令列表，以避免 shell 解析特殊字符（如'#'）
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', start_formatted,
        '-to', end_formatted,
        '-c:v', 'libx264',  # 或者使用 'libx265' 如果需要 HEVC 编码
        '-crf', '23',  # 推荐值，可以调整
        '-c:a', 'copy',  # 复制音频，避免处理和质量损失
        '-preset', 'ultrafast',  # 编码速度和压缩率的平衡，'veryfast', 'fast', 'medium', 'slow'
        '-y', output_path
    ]

    # print("模式: 精确重编码 (速度较慢)")
    # print(f"正在执行命令: {' '.join(command)}") # 如果需要调试，可以用这种方式打印
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 【修改部分】执行命令并等待完成，移除 shell=True
        result = subprocess.run(
            command,
            # shell=True,  <-- 已移除
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        success_message = f"视频截取完成：已保存至: {output_path} 截取时长: {float(end_formatted) - float(start_formatted)} 时间段为 {start_formatted} - {end_formatted} 耗时 {time.time() - current_time:.2f} 秒"
        print(success_message)
        # ffmpeg 正常运行时会将大量信息输出到 stderr
        return True, result.stderr or success_message

    except FileNotFoundError:
        error_message = "错误：找不到 ffmpeg 命令。请确保 ffmpeg 已正确安装并已添加到系统环境变量 PATH 中。"
        print(error_message)
        return False, error_message

    except subprocess.CalledProcessError as e:
        # 如果 ffmpeg 返回非零退出码，说明出错了
        error_message = f"ffmpeg 执行出错：\n{e.stderr}"
        print(error_message)
        return False, error_message

    except Exception as e:
        # 捕获其他可能的异常
        error_message = f"发生未知错误: {e}"
        print(error_message)
        return False, error_message


def _merge_chunk_ffmpeg(video_paths, output_path, probe_fn):
    """
    使用 filter_complex 将一小批 video_paths 合并为 output_path。
    probe_fn 是你现有的 probe_video_new，接受路径返回 (w,h,fps,sar)
    """
    if not video_paths:
        raise ValueError("video_paths 不能为空")
    for p in video_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到文件: {p}")

    # 单文件直接复制（避免重新编码）
    if len(video_paths) == 1:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_paths[0], "-c", "copy", output_path]
        print("[INFO] 单文件直接复制:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    # 参考第一个视频参数
    ref_w, ref_h, ref_fps, ref_sar = probe_fn(video_paths[0])
    print(f"[INFO] 参考视频参数: {ref_w}×{ref_h}, fps={ref_fps:.2f}, SAR={ref_sar}")

    inputs = []
    vf_filters = []

    for idx, path in enumerate(video_paths):
        inputs += ["-i", path]

        if idx == 0:
            vf_filters.append(
                f"[{idx}:v]"
                f"setsar=1,"
                f"format=yuv420p,"
                f"setpts=PTS-STARTPTS[v{idx}]"
            )
        else:
            vf_filters.append(
                f"[{idx}:v]"
                f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease"
                f":in_color_matrix=auto:out_color_matrix=bt709"
                f":in_range=auto:out_range=limited,"
                f"pad={ref_w}:{ref_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"format=yuv420p,"
                f"setpts=PTS-STARTPTS[v{idx}]"
            )

        vf_filters.append(
            f"[{idx}:a]"
            f"volume=1,"
            f"aresample=48000,"
            f"aformat=sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{idx}]"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(video_paths)))
    vf_filters.append(f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=1[outv][outa]")

    filter_complex = "; ".join(vf_filters)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-r", f"{ref_fps:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    print("[INFO] 执行 ffmpeg 合并（小批次）:", " ".join(cmd))
    start_time = time.time()
    subprocess.run(cmd, check=True)
    print(f" 耗时 {time.time() - start_time:.2f} 秒 [SUCCESS] 小批次合并完成：{output_path}")


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _short_tempfile_name(temp_dir, prefix="ffmpeg_part_", suffix=".mp4"):
    name = prefix + uuid.uuid4().hex[:8] + suffix
    return os.path.join(temp_dir, name)

def _safe_remove(path, retries=5, delay=0.5):
    """尝试删除文件或目录，重试若干次；失败则抛出异常"""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                # 目录
                shutil.rmtree(path)
            else:
                # 文件：先清除只读位
                if os.path.exists(path):
                    try:
                        os.chmod(path, 0o666)
                    except Exception:
                        pass
                    os.remove(path)
            return
        except FileNotFoundError:
            return  # 已经不存在了，视作成功
        except Exception as e:
            last_exc = e
            time.sleep(delay * attempt)  # 指数回退
    # 多次尝试后仍未删除，抛出异常
    raise RuntimeError(f"无法删除临时路径: {path}. 最后异常: {last_exc}")

def merge_videos_ffmpeg(video_paths, output_path="merged_video_original_volume.mp4",
                        batch_size=20, temp_dir=None, probe_fn=None,
                        cleanup_temp=True, cleanup_retries=5, cleanup_delay=0.5):
    """
    分批合并并保证临时文件被清理（若无法删除则抛出异常，避免遗漏）。
    cleanup_retries / cleanup_delay 控制删除重试策略。
    """
    if probe_fn is None:
        probe_fn = globals().get("probe_video_new")
        if probe_fn is None:
            raise RuntimeError("必须提供 probe_fn（如 probe_video_new） 或 确保全局有 probe_video_new 函数")

    if not video_paths:
        raise ValueError("视频路径列表不能为空")
    for p in video_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到文件: {p}")

    if len(video_paths) == 1:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_paths[0], "-c", "copy", output_path]
        subprocess.run(cmd, check=True)
        return

    created_temp_dir = False
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        tmpdir = temp_dir
    else:
        tmpdir = tempfile.mkdtemp(prefix="ffmpeg_merge_")
        created_temp_dir = True

    temp_files = []  # 记录所有由本次调用创建的临时文件（绝对路径）

    try:
        if len(video_paths) <= batch_size:
            _merge_chunk_ffmpeg(video_paths, output_path, probe_fn)
            return

        chunks = list(_chunked(video_paths, batch_size))
        for i, chunk in enumerate(chunks):
            tmp_out = _short_tempfile_name(tmpdir, prefix=f"batch{i}_")
            # 记录：即便用户指定 temp_dir，也会删除我们创建的这些临时文件
            temp_files.append(tmp_out)
            _merge_chunk_ffmpeg(chunk, tmp_out, probe_fn)

        # 递归合并临时文件（如果数量超出 batch_size 会继续分批）
        merge_videos_ffmpeg(temp_files, output_path=output_path,
                            batch_size=batch_size, temp_dir=tmpdir, probe_fn=probe_fn,
                            cleanup_temp=False, cleanup_retries=cleanup_retries, cleanup_delay=cleanup_delay)

    finally:
        # ---------- 强化的清理逻辑 ----------
        # 1) 始终尝试删除 temp_files 列表中记录的每个临时文件（这是我们创建的）
        # 2) 如果本函数创建了 tmpdir（created_temp_dir），并且 cleanup_temp=True，则删除整个目录
        # 3) 如果删除失败（重试后仍失败），抛出异常（避免静默遗漏）
        errs = []
        for p in temp_files:
            try:
                _safe_remove(p, retries=cleanup_retries, delay=cleanup_delay)
            except Exception as e:
                errs.append((p, str(e)))

        if created_temp_dir and cleanup_temp:
            try:
                _safe_remove(tmpdir, retries=cleanup_retries, delay=cleanup_delay)
            except Exception as e:
                errs.append((tmpdir, str(e)))

        if errs:
            # 汇总错误并抛出，提醒调用方有未被删除的临时对象
            msg_lines = ["清理临时文件/目录时出现错误："]
            for p, e in errs:
                msg_lines.append(f"  - {p} -> {e}")
            # 将信息打印（方便 debug）并抛出异常
            err_msg = "\n".join(msg_lines)
            print(err_msg)
            raise RuntimeError(err_msg)

def probe_video(path: str) -> dict:
    """
    使用 ffprobe 获取视频的详细信息（宽、高、帧率、SAR、时长）。
    """
    if not shutil.which("ffprobe"):
        raise FileNotFoundError("ffprobe command not found. Please install FFmpeg.")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,sample_aspect_ratio,duration",
        "-of", "json",
        path
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        info = json.loads(proc.stdout)["streams"][0]

        # 解析帧率 "30000/1001" -> float
        num, den = map(int, info["r_frame_rate"].split("/"))
        fps = num / den if den != 0 else 0

        return {
            "width": int(info["width"]),
            "height": int(info["height"]),
            "fps": fps,
            "sar": info.get("sample_aspect_ratio", "1:1"),
            "duration": float(info.get("duration", "0"))
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError) as e:
        raise RuntimeError(f"Failed to probe video file: {path}. Error: {e}")



def _build_enable_expr(time_ranges) -> str:
    """
    构建 FFmpeg 的 enable 表达式，支持多个时间段。
    返回格式: ":enable='between(t,s1,e1)+between(t,s2,e2)+...'"
    如果 time_ranges 为 None 或空，则返回空字符串（全程生效）
    """
    if not time_ranges:
        return ""

    conditions = []
    for start, end in time_ranges:
        # 确保 start <= end
        if start >= end:
            continue  # 跳过无效区间
        conditions.append(f"between(t,{start},{end})")

    if not conditions:
        return ""  # 没有有效区间，全程不遮挡？但通常不会这样用

    expr = "+".join(conditions)
    return f":enable='{expr}'"


def cover_video_area_blur_super_robust(
        video_path: str,
        output_path: str,
        top_left,
        bottom_right,
        time_ranges=None,
        blur_strength: int = 15,
        crf: int = 23,
        preset: str = "ultrafast"
):
    """
    为一个视频的指定区域应用模糊，并进行大量的预检查和参数修正以确保成功。
    """
    # --- 步骤 1: 检查环境依赖 ---
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg command not found. Please install FFmpeg.")

    # --- 步骤 2: 获取视频元数据 ---
    try:
        video_info = probe_video(video_path)
        video_w, video_h = video_info["width"], video_info["height"]
        print(f"Probed video: {video_w}x{video_h}, duration: {video_info['duration']:.2f}s")
    except RuntimeError as e:
        # 如果探测失败，直接中止
        raise RuntimeError(f"Could not process video, probing failed. Reason: {e}")

    # --- 步骤 3: 验证和修正坐标 ---
    x1_orig, y1_orig = top_left
    x2_orig, y2_orig = bottom_right

    # 将坐标钳位在 [0, video_dimension] 范围内
    x1 = max(0, x1_orig)
    y1 = max(0, y1_orig)
    x2 = min(video_w, x2_orig)
    y2 = min(video_h, y2_orig)

    if (x1, y1, x2, y2) != (x1_orig, y1_orig, x2_orig, y2_orig):
        print(f"[INFO] Original coordinates ({x1_orig},{y1_orig})-({x2_orig},{y2_orig}) were out of bounds.")
        print(f"[INFO] Clamped to valid area: ({x1},{y1})-({x2},{y2})")

    # --- 步骤 4: 计算和修正裁剪尺寸 ---
    w, h = x2 - x1, y2 - y1

    # 确保宽高为偶数，这是很多编码器（特别是yuv420p）的要求
    if w % 2 != 0:
        w -= 1
        print(f"[INFO] Adjusted width to be even: {w + 1} -> {w}")
    if h % 2 != 0:
        h -= 1
        print(f"[INFO] Adjusted height to be even: {h + 1} -> {h}")

    # --- 步骤 5: 最终有效性检查 ---
    if w <= 0 or h <= 0:
        raise ValueError(
            f"The specified or corrected area has non-positive dimensions (w={w}, h={h}). "
            "This can happen if the requested area is completely outside the video frame. Aborting."
        )

    # --- 步骤 6: 构建滤镜图，并安全地处理 boxblur 参数 ---
    enable_expr = _build_enable_expr(time_ranges)

    # **核心修复：解决 boxblur 的参数问题**
    # luma_radius (亮度模糊) 可以是 blur_strength
    # chroma_radius (色度模糊) 必须被限制在一个较小的范围内
    luma_radius = blur_strength
    chroma_radius = min(blur_strength, 9)  # 9 是一个非常安全的值
    power = 2  # 模糊迭代次数，2通常效果不错
    boxblur_params = f"{luma_radius}:{power}:{chroma_radius}:{power}"

    # 也可以考虑直接换成 gblur，它没有这个问题，参数更简单
    # gblur_params = f"gblur=sigma={blur_strength}"

    vf = (
        f"[0:v]split=2[orig][crop];"
        f"[crop]crop={w}:{h}:{x1}:{y1},boxblur={boxblur_params}[blurred];"
        f"[orig][blurred]overlay={x1}:{y1}{enable_expr}"
    )

    print(f"Final crop area: x={x1}, y={y1}, w={w}, h={h}")
    print(f"Using boxblur with params: {boxblur_params}")

    # --- 步骤 7: 执行 FFmpeg 命令 ---
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-filter_complex", vf,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        error_message = (
            f"FFmpeg failed with return code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr:\n{proc.stderr}\n"
        )
        raise RuntimeError(error_message)

    print(f"[SUCCESS] Output saved to {output_path}")


def cover_video_area_simple(
    video_path: str,
    output_path: str,
    top_left,
    bottom_right,
    time_ranges = None,
    color: str = "black@1.0"
):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1

    enable_expr = _build_enable_expr(time_ranges)
    vf = f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color={color}:t=fill{enable_expr}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr}")
    print(f"[SUCCESS] Output saved to {output_path}")


def cover_subtitle(
    video_path: str,
    output_path: str,
    top_left,
    bottom_right,
    time_ranges=None
):
    """
    覆盖视频中的字幕区域（支持多个时间段）
    :param time_ranges: 多个 (start_sec, end_sec) 元组列表，例如 [(5,10), (20,25)]
                        若为 None 或空列表，则全程遮挡
    """
    start_time = time.time()
    try:
        cover_video_area_blur_super_robust(
            video_path=video_path,
            output_path=output_path,
            top_left=top_left,
            bottom_right=bottom_right,
            time_ranges=time_ranges
        )
    except Exception as e:
        print(f"覆盖字幕区域失败: {e} 尝试使用备用方法...")
        cover_video_area_simple(
            video_path=video_path,
            output_path=output_path,
            top_left=top_left,
            bottom_right=bottom_right,
            time_ranges=time_ranges
        )
        return
    print(f"覆盖字幕区域完成，输出文件: {output_path} 耗时: {time.time() - start_time:.2f} 秒")


def save_frames_around_timestamp(
        video_path: str,
        timestamp,
        num_frames: int,
        output_dir: str
) -> List[str]:
    """
    从视频中在给定时间戳前后各截取 num_frames 帧并保存为图片。
    输出文件命名格式：frame_{timestamp_ms}.png
    返回：保存的图片路径列表
    """

    # 将输入的时间戳转换为秒（用于定位目标位置）
    # 注意：需确保外部有 time_to_ms 函数，或者传入的 timestamp 已经是 int/float
    ts_sec = time_to_ms(timestamp) / 1000

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("无法获取视频帧率或总帧数")

    # 目标帧序号
    target_idx = int(round(ts_sec * fps))
    # 计算要截取的帧索引区间
    start_idx = max(0, target_idx - num_frames)
    end_idx = min(total_frames - 1, target_idx + num_frames)

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []  # 用于存储图片路径的列表

    print(f"FPS: {fps}, 目标时间: {ts_sec}s, 截取范围: {start_idx} - {end_idx}")

    for idx in range(start_idx, end_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告：读取帧 {idx} 失败，跳过")
            continue

        # --- 修改：计算当前帧的毫秒时间戳 ---
        current_sec = idx / fps
        current_ms = int(current_sec * 1000)

        # 命名格式变为 frame_ + 毫秒时间戳
        filename = f"frame_{current_ms}.png"
        out_path = os.path.join(output_dir, filename)
        # -----------------------------

        # BGR 转 RGB 并保存
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img.save(out_path)

        # 将路径加入列表
        saved_paths.append(out_path)
        print(f"已保存: {filename} (时间戳: {current_ms}ms)")

    cap.release()

    # 返回图片地址列表
    return saved_paths
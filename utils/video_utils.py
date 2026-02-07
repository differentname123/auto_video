
# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 18:13
:last_date:
    2025/12/15 18:13
:description:
    
"""
import pathlib
import platform
import random
import shlex
import tempfile
import uuid
from datetime import datetime
from typing import Union, List

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import cv2
import numpy as np

from utils.common_utils import is_valid_target_file_simple, read_json, save_json, time_to_ms, ms_to_time
from utils.edge_tts_utils import generate_audio_and_get_duration_sync
from utils.split_scenes import split_scenes_json

from pydub import AudioSegment

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
        return ((0, 0, frame_width, frame_height), frame_width, frame_height)
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


def remove_static_background_video(video_path, area_threshold_ratio=0.9, bbox=None, **kwargs):
    """
    分析视频中的运动区域，如果运动区域显著小于整个画面，则进行裁剪。
    如果指定了 bbox，则直接根据 bbox 进行裁剪（支持归一化坐标）。

    :param video_path: 待处理的视频文件路径
    :param area_threshold_ratio: 面积阈值比例。当运动区域面积小于原面积的该比例时，触发裁剪。
                                 例如, 0.8 表示小于80%。
    :param bbox: (可选) 手动指定裁剪框 (x, y, w, h)。
                 如果是归一化坐标(0-1之间)，会自动还原为像素坐标。
    :param kwargs: 传递给 find_motion_bbox 的其他参数，如 start_frame, num_samples 等。
    :return: 元组 (was_cropped, final_path)。
             was_cropped: 布尔值，True表示已裁剪，False表示未裁剪。
             final_path: 最终视频文件的路径（可能是裁剪后的新路径，也可能是原始路径）。
    """
    print(f"\n--- 开始裁剪静态区域处理视频: {video_path} ---")

    # --- 新增逻辑：如果提供了 bbox，直接还原并裁剪 ---
    if bbox is not None:
        # 获取视频原始尺寸用于还原
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频以获取尺寸。")
            return (False, video_path)
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        try:
            # 【修复点】判断 bbox 类型，正确提取数值并转为 float
            if isinstance(bbox, dict):
                bx = float(bbox.get('x', 0))
                by = float(bbox.get('y', 0))
                bw = float(bbox.get('w', 0))
                bh = float(bbox.get('h', 0))
            else:
                # 假设是列表或元组 (x, y, w, h)
                bx, by, bw, bh = [float(v) for v in bbox]
        except ValueError as e:
            print(f"bbox 数据格式错误，无法转换为数字: {bbox}, 错误: {e}")
            return (False, video_path)


        # 判断是否需要还原归一化坐标
        # 简单判断：如果所有值都在 0.0 到 1.0 之间（且宽高不全为0），则视为归一化坐标
        if all(0.0 <= v <= 1.0 for v in [bx, by, bw, bh]) and (bw > 0 or bh > 0):
            print(f"检测到归一化 bbox {bbox}，正在还原...")
            x = int(bx * original_w)
            y = int(by * original_h)
            w = int(bw * original_w)
            h = int(bh * original_h)
            # 确保不越界
            w = min(w, original_w - x)
            h = min(h, original_h - y)
            final_bbox = (x, y, w, h)
            print(f"还原后的 bbox: {final_bbox}")
        else:
            final_bbox = (int(bx), int(by), int(bw), int(bh))
            print(f"使用指定 bbox: {final_bbox}")

        # 构造输出文件名，例如 a.mp4 -> a_crop.mp4
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_crop{ext}"

        # 直接执行裁剪
        crop_video(video_path, output_path, final_bbox)
        return (True, output_path)
    # -----------------------------------------------

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
    if current_ratio < area_threshold_ratio and (w, h) != (original_w, original_h) and current_ratio > 0.2:
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
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.join(os.path.dirname(video_path), f'{video_filename}_scenes')

    # 获取video_path的文件名，不需要后缀



    # --- 将你原有的逻辑放入一个循环中 ---
    thresholds = list(initial_thresholds)  # 创建一个可修改的副本
    kept_sorted = []  # 初始化一个空列表
    all_scene_info_dict = {}
    merged_timestamps_path = os.path.join(output_dir, 'merged_timestamps.json')

    if is_valid_target_file_simple(merged_timestamps_path, 1):
        kept_sorted = read_json(merged_timestamps_path)
        print(f"检测到已存在的合并时间戳文件，直接加载返回，场景数量为: {len(kept_sorted)} {kept_sorted}")
        return kept_sorted

    all_start_time = time.time()
    for attempt in range(max_attempts):
        # print(f"--- 第 {attempt + 1}/{max_attempts} 次尝试 ---")
        # print(f"当前使用的阈值列表: {thresholds} {video_path}")

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
            # print(f"阈值为 {high_threshold} 场景信息字典已生成。共 {len(scene_info_dict)} 个场景。 耗时: {time.time() - start_time:.2f} 秒\n")

            save_json(scene_info_file, scene_info_dict)
            all_scene_info_dict[high_threshold] = scene_info_dict

        # 合并逻辑不变
        kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=3)
        # print(f"场景识别合并完成: 本次尝试生成场景数量为: {len(kept_sorted)}")

        # --- 新增的核心判断逻辑 ---
        if len(kept_sorted) >= min_final_scenes:
            print(f"成功！生成的场景数量  {video_path} ({len(kept_sorted)}) 满足要求 (>= {min_final_scenes})。")
            break  # 达到目标，跳出重试循环
        else:
            print(f"警告：生成的场景数量  {video_path} ({len(kept_sorted)}) 过少，不满足要求 (>= {min_final_scenes})。")
            # 如果不是最后一次尝试，则降低阈值准备重试
            if attempt < max_attempts - 1:
                print(f"准备降低阈值后重试...")
                # 将列表中的每个阈值都减小，并确保不低于某个下限（例如10）
                thresholds = [max(10, t - adjustment_step) for t in thresholds]
                # 如果阈值已经降到最低无法再降，也提前退出
                if all(t == 10 for t in thresholds):
                    print(f" {video_path}阈值已降至最低，无法继续。")
                    break
            else:
                print(f" {video_path}已达到最大尝试次数，将使用当前结果。")

    # 循环结束后的收尾工作（保存文件等）
    print(f"{video_path} 最终场景数量为: {len(kept_sorted)} 总共耗时: {time.time() - all_start_time:.2f} 秒 {kept_sorted} ")
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
    - 【新增】如果截取全长，会自动切换为无损流复制，极大提升速度。

    缺点:
    - 处理速度比流复制慢，因为它需要CPU进行视频编码计算（仅在非全长截取时）。

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

    # 【新增部分】判断是否为全视频截取，如果是则使用流复制以节省时间
    is_full_video = False
    try:
        # 使用 ffprobe 获取视频总时长
        probe_command = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', input_path
        ]
        # capture_output=True 捕获输出用于计算
        probe_result = subprocess.run(
            probe_command,
            check=True,
            capture_output=True,
            text=True
        )
        total_duration = float(probe_result.stdout.strip())

        # 转换为浮点数进行比较
        start_sec = float(start_formatted)
        end_sec = float(end_formatted)

        # 判断逻辑：起点接近0，且终点大于等于视频总时长（给予 0.1s 的浮点误差容忍度）
        if start_sec <= 0.2 and end_sec >= (total_duration - 0.2):
            is_full_video = True

    except Exception as e:
        # 如果获取时长失败（例如没有 ffprobe），则静默失败，回退到默认的重编码模式
        # print(f"时长检测跳过: {e}")
        pass

    # 【修改部分】根据是否全长截取，构建不同的 ffmpeg 命令
    if is_full_video:
        print("检测到截取范围覆盖整个视频，已切换至【流复制模式】(速度极快，无画质损失)...")
        command = [
            'ffmpeg',
            '-i', input_path,
            '-c', 'copy',  # 视频和音频直接复制，不进行编码
            '-y', output_path
        ]
    else:
        # 原有的精确重编码逻辑
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
        error_message = "错误：找不到 ffmpeg 或 ffprobe 命令。请确保它们已正确安装并已添加到系统环境变量 PATH 中。"
        print(error_message)
        return False, error_message

    except subprocess.CalledProcessError as e:
        # 如果 ffmpeg 返回非零退出码，说明出错了
        error_message = f"ffmpeg 执行出错：\n{e.stderr} 完整命令\n{command}"
        print(error_message)
        return False, error_message

    except Exception as e:
        # 捕获其他可能的异常
        error_message = f"发生未知错误: {e}"
        print(error_message)
        return False, error_message


def _merge_chunk_ffmpeg(video_paths, output_path, probe_fn, preset='ultrafast'):
    """
    使用 filter_complex 将一小批 video_paths 合并为 output_path。
    probe_fn 是你现有的 probe_video_new，接受路径返回 (w,h,fps,sar)
    """

    def _probe_audio_and_duration(path):
        """
        获取文件是否有音频流以及文件的总时长
        """
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False, 0.0

        streams = data.get('streams', [])
        fmt = data.get('format', {})

        # 判断是否有音频
        has_audio_stream = any(stream['codec_type'] == 'audio' for stream in streams)

        # 获取时长，优先从 format 获取，失败则从 streams 获取
        duration = 0.0
        try:
            if 'duration' in fmt:
                duration = float(fmt['duration'])
        except:
            pass

        if duration <= 0:
            for stream in streams:
                if 'duration' in stream:
                    try:
                        d = float(stream['duration'])
                        if d > 0:
                            duration = d
                            break
                    except:
                        pass

        return has_audio_stream, duration

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

        # 获取音频状态和时长
        has_audio_flag, duration_val = _probe_audio_and_duration(path)

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

        if has_audio_flag:
            vf_filters.append(
                f"[{idx}:a]"
                f"volume=1,"
                f"aresample=48000,"
                f"aformat=sample_rates=48000:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{idx}]"
            )
        else:
            # 修改点：生成静音时必须指定 duration，否则流是无限长的，会导致 concat 卡死
            vf_filters.append(
                f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={duration_val}[a{idx}]"
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
        "-c:v", "libx264", "-preset", preset, "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",

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
                        cleanup_temp=True, cleanup_retries=5, cleanup_delay=0.5, preset="ultrafast"):
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
            _merge_chunk_ffmpeg(video_paths, output_path, probe_fn, preset=preset)
            return

        chunks = list(_chunked(video_paths, batch_size))
        for i, chunk in enumerate(chunks):
            tmp_out = _short_tempfile_name(tmpdir, prefix=f"batch{i}_")
            # 记录：即便用户指定 temp_dir，也会删除我们创建的这些临时文件
            temp_files.append(tmp_out)
            _merge_chunk_ffmpeg(chunk, tmp_out, probe_fn, preset=preset)

        # 递归合并临时文件（如果数量超出 batch_size 会继续分批）
        merge_videos_ffmpeg(temp_files, output_path=output_path,
                            batch_size=batch_size, temp_dir=tmpdir, probe_fn=probe_fn,
                            cleanup_temp=False, cleanup_retries=cleanup_retries, cleanup_delay=cleanup_delay, preset=preset)

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
    使用 ffprobe 获取视频的详细信息（宽、高、帧率、SAR、时长s）。
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


import shutil
import subprocess
import time


def _build_enable_expr(time_ranges):
    """
    根据时间段构建 FFmpeg 的 enable 表达式字符串。
    返回格式例如: ":enable='between(t,0,5)+between(t,10,15)'"
    如果 time_ranges 为空，返回空字符串（意味着全程生效）。
    """
    if not time_ranges:
        return ""

    parts = []
    for start, end in time_ranges:
        parts.append(f"between(t,{start},{end})")

    # 组合表达式
    expr = "+".join(parts)
    return f":enable='{expr}'"


def cover_video_area_blur_super_robust(
        video_path: str,
        output_path: str,
        top_left,
        bottom_right,
        time_ranges=None,
        blur_strength: int = 50,
        crf: int = 23,
        preset: str = "ultrafast"
):
    """
    为一个视频的指定区域应用模糊，并在相同时间段将音频静音。
    """
    # --- 步骤 1: 检查环境依赖 ---
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg command not found. Please install FFmpeg.")

    # --- 步骤 2: 获取视频元数据 ---
    try:
        # 这里假设 probe_video 函数在你原本的 context 中存在
        # 如果不存在，你需要补充该函数，或者直接使用 subprocess 调用 ffprobe
        video_info = probe_video(video_path)
        video_w, video_h = video_info["width"], video_info["height"]
        print(f"Probed video: {video_w}x{video_h}, duration: {video_info['duration']:.2f}s")
    except Exception as e:
        # 为了代码的完整性，如果 probe_video 不存在，这里可能报错
        # 实际使用中请确保 probe_video 可用
        raise RuntimeError(f"Could not process video, probing failed. Reason: {e}")

    # --- 步骤 3: 验证和修正坐标 ---
    x1_orig, y1_orig = top_left
    x2_orig, y2_orig = bottom_right

    x1 = max(0, x1_orig)
    y1 = max(0, y1_orig)
    x2 = min(video_w, x2_orig)
    y2 = min(video_h, y2_orig)

    if (x1, y1, x2, y2) != (x1_orig, y1_orig, x2_orig, y2_orig):
        print(f"[INFO] Original coordinates were out of bounds. Clamped to: ({x1},{y1})-({x2},{y2})")

    # --- 步骤 4: 计算和修正裁剪尺寸 ---
    w, h = x2 - x1, y2 - y1

    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1

    # --- 步骤 5: 最终有效性检查 ---
    if w <= 0 or h <= 0:
        raise ValueError(f"The specified area has non-positive dimensions (w={w}, h={h}). Aborting.")

    # --- 步骤 6: 构建滤镜图 ---

    # 6.1 生成时间控制表达式 (例如: ":enable='between(t,0,5)'")
    enable_expr = _build_enable_expr(time_ranges)

    # 6.2 视频模糊参数
    luma_radius = blur_strength
    chroma_radius = min(blur_strength, 9)
    power = 2
    boxblur_params = f"{luma_radius}:{power}:{chroma_radius}:{power}"

    # 6.3 构建 Filter Complex
    # [orig][blurred]overlay=...:enable='...'
    vf = (
        f"[0:v]split=2[orig][crop];"
        f"[crop]crop={w}:{h}:{x1}:{y1},boxblur={boxblur_params}[blurred];"
        f"[orig][blurred]overlay={x1}:{y1}{enable_expr}"
    )

    # 6.4 构建音频滤镜 (新增)
    # volume=0:enable='...'
    af = f"volume=0{enable_expr}"

    print(f"Final crop area: x={x1}, y={y1}, w={w}, h={h}")
    print(f"Audio Filter: {af}")

    # --- 步骤 7: 执行 FFmpeg 命令 ---
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-filter_complex", vf,  # 视频滤镜图
        "-af", af,  # 音频滤镜 (静音)
        "-c:a", "aac",  # 【修改】必须重编码音频才能生效，不能用 copy
        "-b:a", "192k",  # 指定音频比特率，保证音质
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
        time_ranges=None,
        color: str = "black@1.0"
):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1

    # 1. 生成时间控制表达式
    enable_expr = _build_enable_expr(time_ranges)

    # 2. 视频滤镜
    vf = f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color={color}:t=fill{enable_expr}"

    # 3. 音频滤镜 (新增)
    af = f"volume=0{enable_expr}"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-af", af,  # 音频滤镜
        "-c:a", "aac",  # 【修改】音频重编码
        "-c:v", "libx264",  # drawbox 改变了画面内容，建议显式指定编码器
        "-preset", "ultrafast",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr}")
    print(f"[SUCCESS] Output saved to {output_path}")

def dynamic_video_area_blur(
        video_path: str,
        output_path: str,
        blur_segments: list,
        blur_strength: int = 50,
        target_quality: int = 24,  # NVENC的CQ值，越小画质越好，24-26通常为视觉无损
        use_gpu: bool = True
):
    """
    高性能版：使用 NVIDIA NVENC 加速，使用高斯模糊提升视觉质感。
    """

    # --- 1. 环境与输入检查 ---
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("Error: ffmpeg is not installed.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: Input video not found at {video_path}")

    # --- 2. 获取视频信息 ---
    try:
        cmd_probe = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", video_path
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True)
        info = json.loads(result.stdout)
        video_w = info['streams'][0]['width']
        video_h = info['streams'][0]['height']
    except Exception as e:
        raise RuntimeError(f"Failed to probe video info: {e}")

    # --- 3. 构建 Filter Complex ---
    filter_chains = []
    last_stream = "[0:v]"

    # 【视觉优化】使用高斯模糊 (gblur) 代替 boxblur
    # sigma 是模糊强度，steps 是模糊次数(默认为1即可)
    # 相比 boxblur，gblur 边缘更柔和，像毛玻璃，不像马赛克
    gblur_sigma = blur_strength
    blur_filter = f"gblur=sigma={gblur_sigma}:steps=1"

    valid_segments_count = 0

    for i, seg in enumerate(blur_segments):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        x1, y1, x2, y2 = seg.get("bbox", (0, 0, 0, 0))

        # --- 3.1 坐标修正 (保持原有健壮逻辑) ---
        x1 = max(0, min(x1, video_w - 1))
        y1 = max(0, min(y1, video_h - 1))
        x2 = max(x1 + 1, min(x2, video_w))
        y2 = max(y1 + 1, min(y2, video_h))

        w = x2 - x1
        h = y2 - y1

        # 偶数修正 (YUV420p)
        if x1 % 2 != 0: x1 -= 1
        if y1 % 2 != 0: y1 -= 1
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        if w < 2 or h < 2: continue
        if x1 + w > video_w: w -= 2
        if y1 + h > video_h: h -= 2

        valid_segments_count += 1

        # --- 3.2 生成滤镜链 ---
        split_crop = f"crop_{i}"
        split_main = f"main_{i}"
        blurred = f"blur_{i}"
        out_label = f"v{i + 1}"

        # 逻辑：分离 -> 裁剪 -> 高斯模糊 -> 覆盖回原处
        cmd_part = (
            f"{last_stream}split=2[{split_main}][{split_crop}];"
            f"[{split_crop}]crop={w}:{h}:{x1}:{y1},{blur_filter}[{blurred}];"
            f"[{split_main}][{blurred}]overlay={x1}:{y1}:enable='between(t,{start},{end})':shortest=1[{out_label}]"
        )
        # 注意：overlay 添加 shortest=1 防止因浮点数精度导致视频长度微变

        filter_chains.append(cmd_part)
        last_stream = f"[{out_label}]"

    # --- 4. 编码参数 (核心优化点) ---
    if valid_segments_count == 0:
        print("[INFO] No valid segments. Copying.")
        final_args = ["-c", "copy"]
        filter_complex_args = []
    else:
        filter_str = ";".join(filter_chains)
        filter_complex_args = ["-filter_complex", filter_str, "-map", last_stream]

        if use_gpu:
            # === NVIDIA GPU 编码配置 (HEVC/H.265) ===
            encode_args = [
                "-c:v", "hevc_nvenc",       # 调用 N卡 硬件编码器
                "-preset", "p6",            # p1-p7，p6/p7 为高质量慢速，p4为中等。p6 此时性价比最高
                "-tune", "hq",              # 调优为高质量
                "-rc", "vbr",               # 动态码率
                "-cq", str(target_quality), # 恒定质量参数 (类似CRF)，NVENC下 24-26 约等于 CRF 23
                "-b:v", "0",                # 让 cq 接管码率控制
                "-tag:v", "hvc1",           # Apple 兼容性
                "-pix_fmt", "yuv420p"
            ]
            print(f"[INFO] Using GPU Encoding: hevc_nvenc (CQ: {target_quality})")
        else:
            # === CPU 兜底配置 (libx265) ===
            encode_args = [
                "-c:v", "libx265",
                "-preset", "medium",
                "-crf", str(target_quality + 2), # CPU CRF 和 NVENC CQ 值略有差异，微调补偿
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p"
            ]
            print(f"[INFO] Using CPU Encoding: libx265 (CRF: {target_quality + 2})")

        final_args = filter_complex_args + ["-map", "0:a?", "-c:a", "copy"] + encode_args

    # --- 5. 执行 ---
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path] + final_args + [output_path]

    print(f"Executing FFmpeg with {valid_segments_count} blur segments...")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    if proc.returncode != 0:
        # 如果 NVENC 失败（例如驱动问题或显卡不支持），这里可以尝试降级回 CPU，
        # 但为简单起见，直接抛出详细错误供调试。
        if "hevc_nvenc" in str(proc.stderr):
             raise RuntimeError(f"GPU Encoding failed. Please update drivers or set use_gpu=False.\nDetails: {proc.stderr}")
        raise RuntimeError(f"FFmpeg Error:\n{proc.stderr}")

    print(f"[SUCCESS] Saved to: {output_path}")


def dynamic_video_area_blur_h265(
        video_path: str,
        output_path: str,
        blur_segments: list,
        blur_strength: int = 15,
        crf: int = 26,  # 修改默认值：稍微调高CRF以减小体积
        preset: str = "medium",  # 修改默认值：medium 比 fast 压缩率更好
        codec: str = "libx265"  # 新增：默认使用 H.265 以大幅减小体积
):
    """
    优化版：修复坐标Bug，支持H.265压缩。
    """

    # --- 1. 环境检查 ---
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("Error: ffmpeg is not installed.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: Input video not found at {video_path}")

    # --- 2. 获取视频信息 ---
    try:
        cmd_probe = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", video_path
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True)
        info = json.loads(result.stdout)
        video_w = info['streams'][0]['width']
        video_h = info['streams'][0]['height']
    except Exception as e:
        raise RuntimeError(f"Failed to probe video info: {e}")

    # --- 3. 构建 Filter Complex ---
    filter_chains = []
    last_stream = "[0:v]"

    # 模糊参数
    luma_radius = blur_strength
    chroma_radius = min(blur_strength, 10)
    boxblur_args = f"{luma_radius}:{chroma_radius}"

    valid_segments_count = 0

    for i, seg in enumerate(blur_segments):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        x1, y1, x2, y2 = seg.get("bbox", (0, 0, 0, 0))

        # --- 3.1 坐标健壮性修正 (修复版) ---
        # 1. 边界限制
        x1 = max(0, min(x1, video_w - 1))
        y1 = max(0, min(y1, video_h - 1))
        x2 = max(x1 + 1, min(x2, video_w))
        y2 = max(y1 + 1, min(y2, video_h))

        w = x2 - x1
        h = y2 - y1

        # 2. 【修复 Bug】不仅宽高要是偶数，起始坐标(x,y)也必须是偶数
        # 这是为了满足 yuv420p 的 2x2 子采样要求
        if x1 % 2 != 0: x1 -= 1
        if y1 % 2 != 0: y1 -= 1
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        # 3. 尺寸过小或无效检查
        if w < 2 or h < 2:
            continue

        # 4. 再次检查右下角是否越界 (因为上面减小了 x1, y1 可能导致 w, h 变化)
        if x1 + w > video_w: w -= 2
        if y1 + h > video_h: h -= 2

        valid_segments_count += 1

        # --- 3.2 生成滤镜 ---
        # 优化提示：如果 blur_segments 数量 > 20，建议使用 sendcmd 或 ass 遮罩，
        # 但为了保持 Python 逻辑简单，这里保留 filter 链，只做逻辑修复。

        split_crop = f"crop_{i}"
        split_main = f"main_{i}"
        blurred = f"blur_{i}"
        out_label = f"v{i + 1}"

        # 使用简化的 label 名以减少命令行长度
        cmd_part = (
            f"{last_stream}split=2[{split_main}][{split_crop}];"
            f"[{split_crop}]crop={w}:{h}:{x1}:{y1},boxblur={boxblur_args}[{blurred}];"
            f"[{split_main}][{blurred}]overlay={x1}:{y1}:enable='between(t,{start},{end})'[{out_label}]"
        )

        filter_chains.append(cmd_part)
        last_stream = f"[{out_label}]"

    # --- 4. 编码参数优化 (针对体积) ---
    if valid_segments_count == 0:
        print("[INFO] No valid segments. Copying.")
        final_args = ["-c", "copy"]
        filter_complex_args = []
    else:
        filter_str = ";".join(filter_chains)
        filter_complex_args = ["-filter_complex", filter_str, "-map", last_stream]

        # 智能选择编码参数
        if codec == "libx265":
            # H.265 配置
            # tag:hvc1 确保苹果设备兼容性
            encode_args = [
                "-c:v", "libx265",
                "-preset", preset,
                "-crf", str(crf + 4),  # H.265 的 CRF 28 约等于 H.264 的 23，所以自动补偿一下
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p"
            ]
        else:
            # H.264 配置
            encode_args = [
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p"
            ]

        final_args = filter_complex_args + ["-map", "0:a?", "-c:a", "copy"] + encode_args

    # --- 5. 执行 ---
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path] + final_args + [output_path]

    print(f"Executing FFmpeg with {valid_segments_count} blur segments...")
    # print(" ".join(cmd)) # 调试时可以打开

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg Error:\n{proc.stderr}")

    print(f"[SUCCESS] Saved to: {output_path} (Codec: {codec})")

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
        output_dir: str,
        time_duration_s=None
) -> List[str]:
    """
    从视频中在给定时间戳前后各截取 num_frames 帧并保存为图片。
    如果图片已存在，则跳过保存步骤，直接返回路径。
    输出文件命名格式：frame_{timestamp_ms}.png
    返回：保存的图片路径列表
    """

    # 假设外部有 time_to_ms 函数，或者传入的 timestamp 已经是 int/float
    # 如果没有 time_to_ms，请确保传入的是秒或毫秒并自行转换
    ts_sec = time_to_ms(timestamp) / 1000

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("无法获取视频帧率或总帧数")
    if time_duration_s:
        num_frames = int(fps * time_duration_s)

    # 目标帧序号
    target_idx = int(round(ts_sec * fps))
    # 计算要截取的帧索引区间
    start_idx = max(0, target_idx - num_frames)
    end_idx = min(total_frames - 1, target_idx + num_frames)

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    print(f"FPS: {fps}, 目标时间: {ts_sec}s, 截取范围: {start_idx} - {end_idx}")

    for idx in range(start_idx, end_idx + 1):
        # --- 修改重点 1：先计算路径，不进行任何视频读取操作 ---
        current_sec = idx / fps
        current_ms = int(current_sec * 1000)

        filename = f"frame_{current_ms}.png"
        out_path = os.path.join(output_dir, filename)

        # --- 修改重点 2：检查文件是否存在 ---
        if os.path.exists(out_path):
            # print(f"跳过已存在: {filename}") # 可选：打印日志
            saved_paths.append(out_path)
            continue  # 直接进入下一次循环，不执行下面的 cap.set 和 cap.read

        # --- 如果文件不存在，才执行耗时的视频操作 ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告：读取帧 {idx} 失败，跳过 {video_path}")
            continue

        # BGR 转 RGB 并保存
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img.save(out_path)

        # 将路径加入列表
        saved_paths.append(out_path)
        # print(f"已保存: {filename}")

    cap.release()

    return saved_paths


def create_variety_text(text: str, font_size: int, output_image_path: str, text_type: str = "正式"):
    """
    一个为不同场景优化的、自动化的风格化文字生成函数。

    你只需要关心：【文字内容、字体大小、输出路径、文字类型】。
    颜色会根据类型从内置颜色池随机选择，描边宽度会根据字体大小和类型自动适配。

    Args:
        text (str): 要生成的文字内容。
        font_size (int): 字体大小。
        output_image_path (str): 生成图片的保存路径。
        text_type (str, optional): 文字类型，可选值为 "综艺" 或 "正式"。默认为 "综艺"。
    """
    DEFAULT_FONT_PATH = r'C:\Users\zxh\AppData\Local\Microsoft\Windows\Fonts\AaFengKuangYuanShiRen-2.ttf'

    # --- 1. 样式配置库 ---
    # 将不同类型的配置集中管理，方便扩展
    STYLE_CONFIG = {
        "综艺": {
            "colors": [
                (255, 204, 0),    # 醒目柠檬黄 (经典高对比色)
                (0, 230, 230),    # 能量青色 (科技感、未来感)
                (255, 87, 34),    # 活力亮橙 (温暖、引人注目)
                (236, 64, 122),   # 魅力洋红 (时尚、大胆)
                (124, 252, 0),    # 荧光绿 (赛博朋克、年轻)
                (173, 216, 230),  # 天空浅蓝 (清新、宁静)
                (255, 218, 185),  # 蜜桃粉橙 (温柔、有亲和力)
                (181, 230, 194),  # 薄荷绿 (自然、舒适)
                (220, 190, 240),  # 薰衣草紫 (优雅、梦幻)
                (226, 192, 112),  # 高光香槟金 (比之前的金色更亮，质感更好)
                (205, 127, 50),  # 古铜色 (沉稳、有历史感)
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
                (173, 216, 230),  # 浅天蓝
                (255, 20, 147),  # 深粉色
                (255, 140, 0),  # 深橙色
                (34, 139, 34),  # 森林绿
                (75, 0, 130),  # 靛蓝色
                (199, 21, 133),  # 深洋红色
                (255, 215, 0),  # 金色
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
            ],
            "inner_stroke_ratio": 0.12,  # 内层白色描边，占字号的12%，较粗
            "outer_stroke_ratio": 0.05,  # 最外层深色描边，占字号的5%，较粗
            'font_path': r'C:\Users\zxh\AppData\Local\Microsoft\Windows\Fonts\AaFengKuangYuanShiRen-2.ttf'

        },
        "正式": {
            "colors": [
                (19, 41, 75),  # 深海军蓝
                (218, 165, 32),  # 高级金色
                (139, 0, 0),  # 暗红色
                (0, 100, 0),  # 深绿色
                (255, 204, 0),  # 醒目柠檬黄 (经典高对比色)
                (0, 230, 230),  # 能量青色 (科技感、未来感)
                (255, 87, 34),  # 活力亮橙 (温暖、引人注目)
                (236, 64, 122),  # 魅力洋红 (时尚、大胆)
                (124, 252, 0),  # 荧光绿 (赛博朋克、年轻)
                (173, 216, 230),  # 天空浅蓝 (清新、宁静)
                (255, 218, 185),  # 蜜桃粉橙 (温柔、有亲和力)
                (181, 230, 194),  # 薄荷绿 (自然、舒适)
                (220, 190, 240),  # 薰衣草紫 (优雅、梦幻)
                (226, 192, 112),  # 高光香槟金 (比之前的金色更亮，质感更好)
                (205, 127, 50),  # 古铜色 (沉稳、有历史感)
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
                (173, 216, 230),  # 浅天蓝
                (255, 20, 147),  # 深粉色
                (255, 140, 0),  # 深橙色
                (34, 139, 34),  # 森林绿
                (75, 0, 130),  # 靛蓝色
                (199, 21, 133),  # 深洋红色
                (255, 215, 0),  # 金色
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
            ],
            "inner_stroke_ratio": 0.06,  # 内层白色描边，占字号的6%，更精致
            "outer_stroke_ratio": 0.03,  # 最外层深色描边，占字号的3%，更纤细
            'font_path': 'C:/Windows/Fonts/msyhbd.ttc'
        }
    }

    # --- 2. 自动化参数计算 ---
    # 检查传入的 text_type 是否有效，无效则报错
    if text_type not in STYLE_CONFIG:
        print(f"错误：无效的文字类型 '{text_type}'。可用类型为: {list(STYLE_CONFIG.keys())}")
        return False

    # 根据类型选择配置
    config = STYLE_CONFIG[text_type]
    color_pool = config["colors"]
    font_path = config.get('font_path', DEFAULT_FONT_PATH)
    inner_stroke_ratio = config["inner_stroke_ratio"]
    outer_stroke_ratio = config["outer_stroke_ratio"]

    # 根据字体大小和选择的比例，自动计算描边的最佳宽度
    stroke_width = max(1, int(font_size * inner_stroke_ratio))
    outer_stroke_width = max(1, int(font_size * outer_stroke_ratio))

    # 自动选择颜色
    fill_color = random.choice(color_pool)
    stroke_color = (255, 255, 255)  # 内描边固定为白色
    outer_stroke_color = (40, 40, 40)  # 外描边固定为深灰色/黑色

    # --- 3. 加载字体并准备画布 ---
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"错误：无法加载字体 -> {font_path}")
        print("请检查函数中的 DEFAULT_FONT_PATH 变量是否设置正确！")
        return False

    # 计算画布大小，需要给描边留出足够的“扩张”空间
    padding = (stroke_width + outer_stroke_width) * 2
    # 使用 getbbox 来精确计算文字边界
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    canvas_width = text_width + padding
    canvas_height = text_height + padding

    # --- 4. 绘制、扩张、合成图层 (核心逻辑) ---

    # [底层] 绘制最顶层的文字，用于提取形状
    text_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    # text_bbox[1] 是文字顶部的偏移量，减去它来让文字从画布的(padding/2, 0)位置开始绘制
    draw.text((padding // 2, padding // 2 - text_bbox[1]), text, font=font, fill=fill_color)

    # 提取文字形状的Alpha通道作为蒙版
    alpha_mask = text_layer.getchannel('A')

    # [中层] 创建白色描边
    white_stroke_mask = alpha_mask.filter(ImageFilter.MaxFilter(stroke_width * 2 + 1))
    white_stroke_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    white_stroke_layer.paste(Image.new('RGB', (canvas_width, canvas_height), stroke_color), mask=white_stroke_mask)

    # [顶层] 创建深色外框
    black_stroke_mask = white_stroke_mask.filter(ImageFilter.MaxFilter(outer_stroke_width * 2 + 1))
    black_stroke_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    black_stroke_layer.paste(Image.new('RGB', (canvas_width, canvas_height), outer_stroke_color),
                             mask=black_stroke_mask)

    # 从后往前，完美地合成所有图层 (背景 -> 外描边 -> 内描边 -> 文字)
    final_image = Image.alpha_composite(black_stroke_layer, white_stroke_layer)
    final_image = Image.alpha_composite(final_image, text_layer)

    # --- 5. 裁剪并保存 ---
    bbox = final_image.getbbox()
    if bbox:
        final_image = final_image.crop(bbox)

    final_image.save(output_image_path)
    # print(f"-> 类型 '{text_type}' 的文字已生成：{output_image_path} (颜色: {fill_color})")
    return True



def add_text_overlays_to_video(
        video_path: str,
        text_info_list: list,
        output_video_path: str,
        image_dir_path: str,
        is_fun=False
):
    """
    为视频叠加多个综艺花字图片，并将生成的图片保存到指定目录。

    Args:
        video_path (str): 输入视频的路径。
        text_info_list (list): 花字信息列表。
        output_video_path (str): 输出视频的路径。
        image_dir_path (str): 用于存放生成的所有花字图片的目录路径。
    """
    print("开始处理视频...")

    # --- 步骤 1: 获取视频信息 ---
    try:
        video_w, video_h, _, _ = probe_video_new(video_path)
        print(f"视频分辨率: {video_w}x{video_h}")
    except (TypeError, FileNotFoundError) as e:
        print(f"无法继续处理，因为获取视频信息失败: {e}")
        return

    min_video_size = min(video_w, video_h)
    auto_font_size = int(min_video_size / 15)
    margin = int(min_video_size * 0.15)
    print(f"自动计算字体大小为: {auto_font_size}px, 边距为: {margin}px")

    # --- 步骤 2: 创建指定目录并生成所有花字图片 ---
    # 【改动】确保指定的图片输出目录存在
    os.makedirs(image_dir_path, exist_ok=True)
    print(f"花字图片将保存到目录: {image_dir_path}")

    generated_images = []
    # 【改动】为图片文件名添加视频文件名前缀，避免混淆
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for i, info in enumerate(text_info_list):
        # 【改动】使用指定的目录和新的命名规则
        image_filename = f"{video_basename}_text_{i}.png"
        image_path = os.path.join(image_dir_path, image_filename)
        text_type = "综艺" if is_fun else "正式"
        # print(f"正在生成图片: {image_filename} ...")
        success = create_variety_text(
            text=info['text'],
            font_size=auto_font_size,
            output_image_path=image_path,
            text_type=text_type
        )
        # if is_valid_target_file_simple(image_path):
        #     success = True
        if success:
            generated_images.append({**info, 'path': image_path})
        else:
            print(f"警告: 生成文字 '{info['text']}' 的图片失败，将跳过。")

    if not generated_images:
        print("没有成功生成任何花字图片，处理终止。")
        return

    # --- 步骤 3: 构建并执行 FFmpeg 命令 (逻辑无变化) ---
    position_map = {
        'TL': f"x={margin}:y={margin}", 'TC': f"x=(W-overlay_w)/2:y={margin}",
        'TR': f"x=W-overlay_w-{margin}:y={margin}",
        'ML': f"x={margin}:y=(H-overlay_h)/2", 'MC': f"x=(W-overlay_w)/2:y=(H-overlay_h)/2",
        'MR': f"x=W-overlay_w-{margin}:y=(H-overlay_h)/2",
        'BL': f"x={margin}:y=H-overlay_h-{margin}", 'BC': f"x=(W-overlay_w)/2:y=H-overlay_h-{margin}",
        'BR': f"x=W-overlay_w-{margin}:y=H-overlay_h-{margin}",
    }
    base_cmd = ['ffmpeg', '-y', '-i', video_path]
    for img_info in generated_images:
        base_cmd.extend(['-i', img_info['path']])

    filter_complex = []
    last_video_stream = "[0:v]"

    for i, img_info in enumerate(generated_images):
        image_stream = f"[{i + 1}:v]"
        output_stream = f"[v{i + 1}]"
        start, end = img_info['start'], img_info['start'] + img_info['duration']
        position = img_info['position'].upper()
        overlay_coords = position_map.get(position, position_map['MC'])
        filter_str = f"{last_video_stream}{image_stream}overlay={overlay_coords}:enable='between(t,{start},{end})'{output_stream}"
        filter_complex.append(filter_str)
        last_video_stream = output_stream

    full_cmd = base_cmd + [
        '-filter_complex', ";".join(filter_complex),
        '-map', last_video_stream, '-map', '0:a?',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-c:a', 'copy',
        output_video_path
    ]

    print("\n即将执行 FFmpeg 命令:")
    print(shlex.join(full_cmd))

    try:
        subprocess.run(full_cmd, check=True, capture_output=True, text=True)
        print(f"\n✅ 视频处理成功！输出文件: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg 执行失败! 错误信息:\n", e.stderr)



def gen_video(text, output_path, origin_video_path,keep_original_audio=False, fixed_rect=None, voice_info=None):
    """
    """
    voice_name = "zh-CN-XiaoxiaoNeural"
    rate = "+30%"
    pitch = '+30Hz'
    if voice_info is not None:
        voice_name = voice_info.get('voice_name', voice_name)
        rate = voice_info.get('rate', "+30%")
        pitch = voice_info.get('pitch', '+30Hz')
    output_path = pathlib.Path(output_path)
    audio_path = output_path.with_suffix(".mp3")
    adjust_audio_path = output_path.with_name(output_path.stem + "_adjusted.mp3")

    video_duration = probe_duration(origin_video_path)
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        rate=rate,
        pitch=pitch,
    )
    adjust_duration = adjust_audio_duration(str(audio_path), duration, target_duration=video_duration, output_mp3_path=str(adjust_audio_path))

    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(video_duration * 1000),
        'outputPath': str(adjust_audio_path),
        'trimmedDuration': adjust_duration,
    }]
    with_audio_path = output_path.with_name(output_path.stem + "_with_audio.mp4")
    redub_video_with_ffmpeg(video_path=origin_video_path, segments_info=segments_info, output_path=str(with_audio_path),keep_original_audio=keep_original_audio)

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(adjust_duration * 1000),
        'optimizedText': text
    }]
    add_subtitles_to_video(
        video_path=str(with_audio_path),
        subtitles_info=subtitle_data,
        output_path=str(output_path),
        font_size=70,
        bottom_margin=30,
        fixed_rect=fixed_rect
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(adjust_audio_path):
        os.remove(adjust_audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())


def redub_video_with_ffmpeg(video_path: str,
                            segments_info: list,
                            output_path: str = "final_video_ffmpeg.mp4",
                            keep_original_audio: bool = False,
                            max_speed_up: float = 1.1) -> str:
    """
    使用 FFmpeg 直接为视频重新配音。
    自动调整视频速度以精确匹配音频时长，但增加了最大加速限制。

    :param video_path: 原始视频文件的路径。
    :param segments_info: 片段信息列表。
    :param output_path: 输出路径。
    :param keep_original_audio: 是否保留原始音频并混合。
    :param max_speed_up: 视频最大加速倍率 (默认 1.5)。
                         例如设置为 1.5，则视频最快只能变为原来的 1.5 倍速播放。
                         如果音频过短，视频将不再强制对齐音频，而是保持 1.5 倍速播放完毕。
    :return: 输出视频的路径。
    """
    start_time = time.time()

    # ---------- 内部工具函数 ----------
    def _time_str_to_seconds(ts: str) -> float:
        if not ts:
            return 0.0
        ts = ts.strip()
        neg = ts.startswith("-")
        if neg:
            ts = ts[1:]
        parts = ts.split(":")
        parts = [float(p) for p in parts]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, s = parts[-3], parts[-2], parts[-1]
        val = h * 3600 + m * 60 + s
        return -val if neg else val

    def _probe_has_audio(path: str) -> bool:
        if not shutil.which("ffprobe"):
            return True
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path
            ]
            out = subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
            return len(out) > 0
        except Exception:
            return True

    def _probe_duration_seconds(path: str) -> float:
        if not shutil.which("ffprobe"):
            return 0.0
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ]
            out = subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
            return float(out) if out else 0.0
        except Exception:
            return 0.0

    def build_atempo_filter(tempo: float) -> str:
        if abs(tempo - 1.0) < 1e-6:
            return "anull"
        filters = []
        t = tempo
        while t < 0.5:
            filters.append("atempo=0.5")
            t *= 2.0
        while t > 2.0:
            filters.append("atempo=2.0")
            t /= 2.0
        if abs(t - 1.0) > 1e-6:
            filters.append(f"atempo={t:.6f}")
        return ",".join(filters)

    # ---------- 依赖与输入检查 ----------
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    source_has_audio = _probe_has_audio(video_path)

    # ---------- 处理每个片段 ----------
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files_list = []
        concat_file_path = os.path.join(temp_dir, "file_list.txt")

        # 初始化变量
        original_duration = 0.0
        new_audio_duration = 0.0

        # 这里的变量名修正为 pts_factor 以便理解（原代码叫 speed_multiplier）
        # pts_factor < 1.0 意味着加速（时长变短）
        pts_factor = 1.0

        for i, segment in enumerate(segments_info):
            segment_id = segment.get('id', i + 1)
            start_time_str = segment['startTime']
            end_time_str = segment['endTime']
            audio_path = segment['outputPath']

            if not os.path.exists(audio_path):
                print(f"警告: 音频文件未找到 {audio_path}，跳过此片段。")
                continue

            original_duration = max(0.000001, _time_str_to_seconds(end_time_str) - _time_str_to_seconds(start_time_str))
            new_audio_duration = float(segment.get('trimmedDuration') or 0.0)
            if new_audio_duration <= 0.0:
                new_audio_duration = _probe_duration_seconds(audio_path) or original_duration

            temp_output_path = os.path.join(temp_dir, f"temp_segment_{segment_id}.mp4")
            temp_files_list.append(temp_output_path)

            # [关键修改]：计算速度倍率与上限控制
            # 1. 计算理论上需要的时长缩放系数 (Target PTS Factor)
            target_pts_factor = 1.0
            if original_duration > 0 and new_audio_duration > 0:
                target_pts_factor = new_audio_duration / original_duration

            # 2. 计算允许的最小缩放系数 (对应最大加速倍率)
            # 例如 max_speed_up 为 1.5，则最小允许的时长系数为 1/1.5 = 0.666
            min_pts_factor = 1.0 / max_speed_up

            # 3. 应用限制逻辑
            is_speed_limited = False
            if target_pts_factor < min_pts_factor:
                pts_factor = min_pts_factor
                is_speed_limited = True
            else:
                pts_factor = target_pts_factor

            # 简单的稳健性检查：防止过慢 (防止除以0等极端情况)
            if pts_factor > 100.0: pts_factor = 100.0

            # 实际的物理播放速度 (用于显示和音频调整)
            real_speed_val = 1.0 / pts_factor

            # ---------- 构建滤镜与映射 ----------
            if keep_original_audio and source_has_audio:
                # 视频速度是 real_speed_val，原音频也要匹配这个速度
                audio_tempo = real_speed_val
                atempo_filter = build_atempo_filter(audio_tempo)

                filter_complex = (
                    f"[0:v]setpts={pts_factor:.6f}*PTS[v];"
                    f"[0:a]{atempo_filter},aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
                    f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
                    # normalize=0 且 duration=first 或 longest
                    # 这里依然使用 longest，如果视频因为限速变长了，新配音放完后背景音继续放是合理的
                    f"[a0][a1]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.97[a]"
                )
                map_args = ["-map", "[v]", "-map", "[a]"]

                limit_msg = f" [已触发限速 {max_speed_up}x]" if is_speed_limited else ""
                print(
                    f"片段 {segment_id}: 混音模式。{original_duration:.2f}s -> {original_duration * pts_factor:.2f}s (倍率 {real_speed_val:.2f}x){limit_msg}")

            else:
                # 替换模式
                filter_complex = (
                    f"[0:v]setpts={pts_factor:.6f}*PTS[v];"
                    f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,alimiter=limit=0.97[a]"
                )
                map_args = ["-map", "[v]", "-map", "[a]"]

                limit_msg = f" [已触发限速 {max_speed_up}x]" if is_speed_limited else ""
                if keep_original_audio and not source_has_audio:
                    print(f"片段 {segment_id}: 源视频无音轨，切换为替换模式。{limit_msg}")
                else:
                    print(
                        f"片段 {segment_id}: 替换模式。{original_duration:.2f}s -> {original_duration * pts_factor:.2f}s (倍率 {real_speed_val:.2f}x){limit_msg}")

            base_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", start_time_str, "-to", end_time_str,
                "-i", video_path, "-i", audio_path,
                "-filter_complex", filter_complex,
            ]

            encoding_cmd = [
                "-t", str(new_audio_duration),  # <--- 新增这行：强制限制输出时长为音频时长
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2", temp_output_path
            ]
            cmd = base_cmd + map_args + encoding_cmd

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                )
            except subprocess.CalledProcessError as e:
                print(f"处理片段 {segment_id} 时 FFmpeg 发生错误：")
                print(f"FFmpeg Stderr:\n{e.stderr}")
                raise

        if not temp_files_list:
            print("没有可处理的片段，无法生成最终视频。")
            return ""

        # ---------- 拼接片段 ----------
        with open(concat_file_path, 'w', encoding='utf-8') as f:
            for file_path in temp_files_list:
                safe_path = file_path.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file_path, "-c", "copy", output_path
        ]

        try:
            subprocess.run(
                concat_cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
        except subprocess.CalledProcessError as e:
            print("拼接视频时 FFmpeg 发生错误：")
            print(f"FFmpeg Stderr:\n{e.stderr}")
            raise

    if temp_files_list:
        print(f"处理完成。总耗时: {time.time() - start_time:.2f}s")

    return output_path


def add_subtitles_to_video(
        video_path: str,
        subtitles_info: list,
        output_path: str,
        font_size: int = 48,
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        font_color: str = 'white',
        box_color: str = 'black@0.5',
        bottom_margin: int = 50,
        fixed_rect=None
) -> None:
    """
    将字幕信息“烧录”到视频中，并自动分割过长的字幕行，
    并在每条字幕出现的时段内，先绘制一个固定大小的矩形背景。

    :param video_path: 输入视频的路径。
    :param subtitles_info: 原始字幕信息列表。
    :param output_path: 输出视频的路径。
    :param font_path: 字体文件路径。
    :param font_size: 字体大小，默认 48。
    :param font_color: 字体颜色，默认白色。
    :param box_color: 半透明背景色，默认黑@0.5。
    :param bottom_margin: 距离底部的像素偏移，默认 50。
    :param fixed_rect: 固定矩形区域 [[x1,y1],[x2,y2]]。如果为 None，则自动计算。
    """
    current_start_time = time.time()
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件未找到: {font_path}")

    # ------------------- [ 核心修改开始 ] -------------------

    # 1. 获取视频尺寸
    try:
        video_width, video_height = get_video_dimensions(video_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"警告: 无法获取视频尺寸，将使用默认分辨率 1920x1080 进行计算。错误: {e}")
        video_width, video_height = 1920, 1080
        if fixed_rect is None:
            # 如果既没尺寸也没提供矩形，无法自动计算位置，必须报错
            raise ValueError("无法获取视频尺寸且未提供 fixed_rect，无法继续。") from e

    # 2. 【关键】先确定最终要用的字号 (effective_font_size)
    # 必须在分割字幕前确定，否则检测不到超长
    effective_font_size = font_size  # 默认值

    # 设定最大允许宽度
    if fixed_rect is not None:
        # 如果用户指定了固定区域，根据区域高度计算字号
        x1, y1 = fixed_rect[0]
        x2, y2 = fixed_rect[1]
        rect_h_fixed = y2 - y1
        rect_w_fixed = x2 - x1

        effective_font_size = int(rect_h_fixed * 0.8)
        effective_font_size = max(1, effective_font_size)

        # 最大宽度限制为矩形宽度（减去一点内边距）
        max_subtitle_width = rect_w_fixed * 0.95
    else:
        # 如果没有固定框，最大宽度为视频宽度的 90%
        max_subtitle_width = video_width * 0.9

    # 3. 加载字体（使用最终确定的字号）
    try:
        font = ImageFont.truetype(font_path, effective_font_size)
    except IOError:
        raise FileNotFoundError(f"无法加载字体文件，请检查路径和文件格式: {font_path}")

    # 4. 预处理字幕，分割过长行
    # 此时 font 是大号字体，如果字太长，这里就会检测出来并进行分割
    processed_subtitles = _process_and_split_subtitles(
        subtitles_info,
        font,
        max_subtitle_width
    )

    # 5. 如果 fixed_rect 未指定，则在此处自动计算
    if fixed_rect is None:
        if not processed_subtitles:
            print("警告：没有字幕信息，无法计算矩形。将不绘制背景。")
            fixed_rect = [[0, 0], [0, 0]]
        else:
            max_text_w, max_text_h = 0, 0
            for sub in processed_subtitles:
                # 获取文字尺寸
                try:
                    w = font.getlength(sub['optimizedText'])
                except AttributeError:
                    w = font.getsize(sub['optimizedText'])[0]

                bbox = font.getbbox(sub['optimizedText'])
                h = bbox[3] - bbox[1]

                if w > max_text_w: max_text_w = w
                if h > max_text_h: max_text_h = h

            # 为矩形添加一些内边距（padding）
            padding_x = effective_font_size
            padding_y = effective_font_size // 2

            # 【修复】让背景框宽度跟随实际最长字幕，而不是强制设为视频宽度的90%
            rect_w = max_text_w + padding_x
            rect_h = max_text_h + padding_y

            # 再次检查：防止自动计算出的框偶尔超过屏幕
            if rect_w > video_width:
                rect_w = video_width

            # 计算矩形坐标
            rect_x1 = (video_width - rect_w) / 2
            rect_y1 = video_height - bottom_margin - max_text_h - (padding_y / 2)

            # 确保坐标是整数
            rect_x1 = int(rect_x1)
            rect_y1 = int(rect_y1)
            rect_w = int(rect_w)
            rect_h = int(rect_h)

            fixed_rect = [[rect_x1, rect_y1], [rect_x1 + rect_w, rect_y1 + rect_h]]

    # ------------------- [ 核心修改结束 ] -------------------

    # 为 ffmpeg 的滤镜语法格式化字体路径
    formatted_font_path = font_path.replace('\\', '/')
    if os.name == 'nt':
        formatted_font_path = formatted_font_path.replace(':', '\\:')

    # 计算固定矩形的位置和尺寸 (现在 fixed_rect 必定有值)
    x1, y1 = fixed_rect[0]
    x2, y2 = fixed_rect[1]
    rect_w = x2 - x1
    rect_h = y2 - y1

    # 注意：此处不再重新计算 effective_font_size，因为前面已经算好并用于分割逻辑了

    display_font_size = int(effective_font_size * 1)
    # ================= [ 修改结束 ] =================

    filters = []
    # 只有当矩形有实际大小时才添加绘制指令
    if rect_w > 0 and rect_h > 0:
        for sub in processed_subtitles:
            start_time = _parse_subtitle_time(sub['startTime'])
            end_time = _parse_subtitle_time(sub['endTime'])

            # 1) 先画固定大小的矩形 (这里依然使用基于 effective_font_size 算出来的 rect_w/rect_h)
            drawbox = (
                f"drawbox="
                f"x={x1}:y={y1}:w={rect_w}:h={rect_h}:"
                f"color={box_color}:t=fill:"
                f"enable='between(t,{start_time},{end_time})'"
            )
            filters.append(drawbox)

    # 总是绘制字幕文本
    for sub in processed_subtitles:
        start_time = _parse_subtitle_time(sub['startTime'])
        end_time = _parse_subtitle_time(sub['endTime'])
        text = _escape_ffmpeg_text(sub['optimizedText'])

        if fixed_rect is not None:
            # 文字在 fixed_rect 内水平垂直居中
            # 注意：ffmpeg 运行时使用的是当前 text_w (也就是小字体的宽)，所以依然会完美居中
            text_x_expr = f"x=({x1}+{x2})/2 - text_w/2"
            text_y_expr = f"y=({y1}+{y2})/2 - text_h/2"
        else:
            text_x_expr = "x=(w-text_w)/2"
            text_y_expr = f"y=h-text_h-{bottom_margin}"

        # 2) 再画字幕文字
        drawtext = (
            f"drawtext="
            f"fontfile='{formatted_font_path}':"
            f"text='{text}':"
            f"fontsize={display_font_size}:"  # <--- 这里把 effective_font_size 改为 display_font_size
            f"fontcolor={font_color}:"
            f"{text_x_expr}:"
            f"{text_y_expr}:"
            f"box=0:"
            f"enable='between(t,{start_time},{end_time})'"
        )
        filters.append(drawtext)

    if not filters:
        print("没有可烧录的字幕，将直接复制视频。")
        import shutil
        shutil.copy(video_path, output_path)
        return

    vf_arg = ",".join(filters)

    # ... 后续的 ffmpeg 调用代码保持不变 ...
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8') as temp_filter_file:
        temp_filter_file.write(vf_arg)
        filter_script_path = temp_filter_file.name

    formatted_filter_path = filter_script_path.replace('\\', '/')

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-filter_complex_script", formatted_filter_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]

    try:
        # print("正在为视频添加字幕和矩形背景...")
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"字幕添加成功: 已将带字幕的视频保存至: {output_path} 耗时: {time.time() - current_start_time:.2f} 秒")
    except FileNotFoundError:
        print("[错误] ffmpeg 未安装或未在系统 PATH 中。请先安装 ffmpeg。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[错误] ffmpeg 执行失败。返回码: {e.returncode}")
        print(f"FFMPEG 错误输出:\n{e.stderr}")
        print(f"滤镜脚本内容保存在: {filter_script_path}")
        raise
    finally:
        if os.path.exists(filter_script_path):
            os.remove(filter_script_path)


def get_video_dimensions(video_path: str) -> (int, int):
    """
    使用 ffprobe 获取视频的宽度和高度。
    此版本确保 video_path 被正确传递，并提供详细的错误处理。
    """
    # 检查文件是否存在，提前给出更友好的提示
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到，请检查路径: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path  # <-- 修正：将 video_path 作为命令的一部分
    ]

    try:
        # 使用 check=True, ffprobe 失败时会抛出 CalledProcessError
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        data = json.loads(result.stdout)

        # 健壮性检查：确保返回的 JSON 结构符合预期
        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            if "width" in stream and "height" in stream:
                return stream["width"], stream["height"]

        # 如果 JSON 结构不符
        raise ValueError("在 ffprobe 的输出中未找到有效的视频流信息。")

    except FileNotFoundError:
        # 如果 ffprobe 命令本身就找不到
        print("错误: ffprobe 命令未找到。请确保 ffmpeg (及 ffprobe) 已安装并在系统 PATH 中。")
        raise
    except subprocess.CalledProcessError as e:
        # 捕获 ffprobe 执行失败的错误，并打印其 stderr
        print(f"ffprobe 执行失败。返回码: {e.returncode}")
        # ffprobe 的错误信息通常在 stderr 中，这对于调试至关重要
        print(f"ffprobe 的原始错误输出:\n---\n{e.stderr.strip()}\n---")
        raise ValueError(f"无法从视频 {video_path} 中解析出尺寸。")
    except json.JSONDecodeError:
        # 如果 ffprobe 输出的不是有效的 json
        print(f"ffprobe 输出了非预期的内容，无法解析为JSON。输出内容: {result.stdout}")
        raise ValueError(f"无法解析来自 ffprobe 的视频尺寸信息。")


def _parse_subtitle_time(time_str: str) -> float:
    """
    将各种格式的时间字符串统一转换为秒 (float)。
    这个函数现在是 time_to_ms 的一个包装器，以保证健壮性和一致性。
    """
    # 1. 调用我们已经写好的、非常健壮的 time_to_ms 函数
    milliseconds = time_to_ms(time_str)

    # 2. 将毫秒转换为秒 (float)，以匹配原始函数的返回值类型
    return milliseconds / 1000.0


def _escape_ffmpeg_text(text: str) -> str:
    """
    为ffmpeg的drawtext滤镜转义特殊字符。
    """
    # 转义 \ ' % :
    # 单引号 ' 替换为视觉上相似的 ’，避免破坏滤镜语法
    return text.replace('\\', '\\\\').replace("'", "’").replace('%', r'\%').replace(':', r'\:')


def _process_and_split_subtitles(
        subtitles_info,
        font: ImageFont.FreeTypeFont,
        max_width: int
):
    """
    预处理字幕列表，将过长的字幕分割成多段，直到每段都不超过最大宽度 max_width。
    """
    processed_subs = []
    split_chars = ['，', '。', '？', '！', '；', ',', '.', '?', '!', ';']

    for sub in subtitles_info:
        # 队列：存放 (startTime_str, endTime_str, text) 待处理
        segments_to_process = [(sub['startTime'], sub['endTime'], sub['optimizedText'])]

        while segments_to_process:
            start_str, end_str, text = segments_to_process.pop(0)
            # 计算渲染宽度
            try:
                text_width = font.getlength(text)
            except AttributeError:
                text_width = font.getsize(text)[0]

            # 如果宽度合格，直接输出
            if text_width <= max_width:
                processed_subs.append({
                    'startTime': start_str,
                    'endTime': end_str,
                    'optimizedText': text
                })
                continue  # 处理下一个队列项

            # 否则需要拆分
            t_start = _parse_subtitle_time(start_str)
            t_end = _parse_subtitle_time(end_str)
            duration = t_end - t_start
            if duration <= 0:
                # 畸形时间区间，跳过
                continue

            # --- 1. 找标点做最佳拆分点 ---
            best_split = -1
            min_offset = float('inf')
            for ch in split_chars:
                idx = 0
                while True:
                    pos = text.find(ch, idx)
                    if pos == -1:
                        break
                    # 考虑标点后面作为拆点
                    offset = abs(pos + 1 - len(text) / 2)
                    if offset < min_offset:
                        min_offset = offset
                        best_split = pos + 1
                    idx = pos + 1

            # --- 2. 如果没有标点就硬拆中间 ---
            if best_split == -1:
                best_split = len(text) // 2

            # 切成两段，去除首尾空白
            part1 = text[:best_split].strip()
            part2 = text[best_split:].strip()

            # 特殊情况：如果某段为空，就强制中点拆分一次
            if not part1 or not part2:
                mid = len(text) // 2
                part1 = text[:mid].strip()
                part2 = text[mid:].strip()

            # --- 3. 按字符比例分配时间 ---
            ratio1 = len(part1) / len(text)
            split_time_sec = t_start + duration * ratio1
            split_time_str = _format_time_for_ffmpeg(split_time_sec)

            # **改动点**：不再直接输出 part1，而是把两段都入队再检测
            segments_to_process.append((start_str, split_time_str, part1))
            segments_to_process.append((split_time_str, end_str, part2))

        # end while

    # end for
    return processed_subs




def text_image_to_video_with_subtitles(
    text: str,
    image_path: str,
    output_path: str,
    short_text: str = "",
    voice_info=None,
    cleanup: bool = True,
    resolution: tuple = (1920, 1080),
    fixed_rect=None
) -> str:
    """
    根据文本和图片生成带字幕的视频，并可选添加背景音乐（bgm），并自动清理中间视频文件。

    参数:
        text: 完整文案
        image_path: 图片路径
        output_path: 输出视频路径
        short_text: 简略文案（可选）
        voice_name: 语音合成声音
        bgm_path: 背景音乐文件路径（可选，若存在则在生成最终视频后添加）
        cleanup: 是否在生成最终视频后清理中间视频文件

    返回:
        最终视频路径（若提供了 bgm_path，返回带 bgm 的视频路径；否则返回无 bgm 的视频路径）
    """
    voice_name = "zh-CN-XiaoxiaoNeural"
    rate = "+30%"
    pitch = '+30Hz'
    if voice_info is not None:
        voice_name = voice_info.get('voice_name', voice_name)
        rate = voice_info.get('rate', "+30%")
        pitch = voice_info.get('pitch', '+30Hz')

    min_size = min(resolution)


    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 文本转语音
    audio_path = output_path.with_suffix(".mp3")
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        rate=rate,
        pitch=pitch,
    )

    # 2. 图片转视频
    image_video_path = output_path.with_name(output_path.stem + "_img.mp4")
    create_video_from_image_auto_select(
        image_path=image_path,
        output_path=str(image_video_path),
        duration=duration,
        resolution=resolution
    )

    # 3. 合成语音
    audio_video_path = output_path.with_name(output_path.stem + "_audio.mp4")
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'outputPath': str(audio_path),
        'trimmedDuration': duration,
    }]
    redub_video_with_ffmpeg(str(image_video_path), segments_info, output_path=str(audio_video_path))

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text
    }]
    subtitle_video_path = output_path.with_name(output_path.stem + "_sub.mp4")
    add_subtitles_to_video(
        video_path=str(audio_video_path),
        subtitles_info=subtitle_data,
        output_path=str(subtitle_video_path),
        fixed_rect=fixed_rect
    )

    # 5. 如果有简略文案，加第二层字幕
    if short_text and len(text) > 30:
        subtitle_data = [{
            'startTime': ms_to_time(duration * 0),
            'endTime': ms_to_time(duration * 1000),
            'optimizedText': short_text
        }]
        add_subtitles_to_video(
            video_path=str(subtitle_video_path),
            subtitles_info=subtitle_data,
            output_path=str(output_path),
            font_color='#FFD700',
            font_size=80/1000*min_size,
            bottom_margin=min_size
        )
    else:
        shutil.copy(str(subtitle_video_path), str(output_path))

    final_video_path = str(output_path.resolve())

    # 7. 清理中间视频文件
    if cleanup:
        # 确定需要保留哪一个最终文件
        kept_final_paths = set()

        # 需要清理的中间视频路径
        intermediates = [
            str(audio_path),
            str(image_video_path),
            str(audio_video_path),
            str(subtitle_video_path),
        ]

        # 删除中间视频文件
        for p in intermediates:
            if p and os.path.exists(p) and p not in kept_final_paths:
                try:
                    os.remove(p)
                    # print(f"已清理中间视频：{p}")
                except Exception as e:
                    print(f"警告：无法清理中间视频 {p}，原因: {e}")

            return final_video_path

    return final_video_path



def create_video_from_image_auto_select(
        image_path: str,
        output_path: str,
        duration=5,
        resolution: tuple = (1920, 1080),
        fps: int = 30,
        zoom_factor: float = 1.0,
        scroll_speed: int = 30,
        use_background_fill: bool = True
):
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图片 '{image_path}'")
        return
    # 随机设置use_background_fill
    use_background_fill = random.choice([True, False])  # 随机选择是否使用背景填充
    zoom_factor = random.choice([1.0, 1.02, 1.05])
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"错误: 无法读取图片 '{image_path}'。 错误信息: {e}")
        return

    output_width, output_height = resolution

    # 决策逻辑保持不变，但调用时简化参数传递
    if img_height > 3 * img_width:
        print(f"检测到高图 -> 【垂直滚动】")
        ### <<< 优化：不再需要在此处计算 final_duration，交由子函数处理
        scroll_image_vertically(
            image_path=image_path, output_path=output_path,
            scroll_speed=scroll_speed, output_width=output_width,
            output_height=output_height, fps=fps, target_duration=duration,  # 直接传递 duration
            use_background_fill=use_background_fill
        )
    elif img_width > 3 * img_height:
        print(f"检测到宽图 -> 【水平滚动】")
        scroll_image_horizontally(
            image_path=image_path, output_path=output_path,
            scroll_speed=scroll_speed, output_width=output_width,
            output_height=output_height, fps=fps, target_duration=duration,  # 直接传递 duration
            use_background_fill=use_background_fill
        )
    else:
        print(f"检测到常规图 -> 【平滑缩放】")
        create_video_from_image_smooth(
            image_path=image_path, output_path=output_path,
            duration=duration, resolution=resolution, fps=fps,
            zoom_factor=zoom_factor, use_background_fill=use_background_fill
        )



def scroll_image_horizontally(
        image_path,
        output_path,
        scroll_speed=30,
        output_width=1920,
        output_height=1080,
        fps=30,
        target_duration=None,
        use_background_fill: bool = True
):
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    if img_height == 0: return
    scaled_width = img_width * (output_height / img_height)
    scroll_distance = max(0, scaled_width - output_width)

    # 决定最终视频时长
    if scroll_distance <= 0:
        # 如果图片不够宽，无法滚动，则生成一个静止视频
        final_duration = target_duration if target_duration is not None else 3
        scroll_distance = 0  # 确保滚动距离为0
    else:
        # 如果指定了时长，就用指定的；否则根据滚动速度计算
        calculated_duration = scroll_distance / scroll_speed
        final_duration = target_duration if target_duration is not None else calculated_duration

    speed_per_frame = scroll_speed / fps

    filter_complex = ""
    if use_background_fill:
        filter_complex = (
            f"[0:v]split[original][bg_src];"
            f"[bg_src]scale=-1:{output_height},boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
            f"[original]scale=-1:{output_height},format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x='min({scroll_distance},max(0,n*{speed_per_frame}))':y=0[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )
    else:
        # 方案B: 黑色背景
        filter_complex = (
            ### <<< 修正：为 color 滤镜添加 d={final_duration} 参数
            f"color=c=black:s={output_width}x{output_height}:d={final_duration}[bg];"
            f"[0:v]scale=-1:{output_height},format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x='min({scroll_distance},max(0,n*{speed_per_frame}))':y=0[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )

    ### <<< 优化：采用与垂直滚动相同的、更健壮的命令结构
    cmd = [
        "ffmpeg", "-y", '-loglevel', 'error',
        "-loop", "1", "-i", image_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        # 将 -t 作为输出选项放在最后，确保视频总长
        "-t", str(final_duration),
        output_path
    ]

    print("正在生成水平滚动视频...\n", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频成功保存到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg 执行失败:\n{e.stderr}")


# ==============================================================================
# 垂直滚动（已优化）
# ==============================================================================
def scroll_image_vertically(
        image_path,
        output_path,
        scroll_speed=30,
        output_width=1920,
        output_height=1080,
        fps=30,
        target_duration=None,
        use_background_fill: bool = True
):
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    if img_width == 0: return
    scaled_height = img_height * (output_width / img_width)
    scroll_distance = max(0, scaled_height - output_height)

    if scroll_distance <= 0:
        final_duration = target_duration if target_duration is not None else 3
        scroll_distance = 0
    else:
        calculated_duration = scroll_distance / scroll_speed
        final_duration = target_duration if target_duration is not None else calculated_duration

    speed_per_frame = scroll_speed / fps

    filter_complex = ""
    if use_background_fill:
        filter_complex = (
            f"[0:v]split[original][bg_src];"
            f"[bg_src]scale={output_width}:-1,boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
            f"[original]scale={output_width}:-1,format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )
    else:
        filter_complex = (
            # 你的代码中这里已经正确添加了 d={final_duration}，这里保持
            f"color=c=black:s={output_width}x{output_height}:d={final_duration}[bg];"
            f"[0:v]scale={output_width}:-1,format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )

    # ### <<< 优化：清理了你代码中被注释掉的旧命令，只保留最终的、最正确的版本
    cmd = [
        "ffmpeg", "-y", '-loglevel', 'error',
        "-loop", "1", "-i", image_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-t", str(final_duration),
        output_path
    ]

    print("正在生成垂直滚动视频...\n", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频成功保存到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg 执行失败:\n{e.stderr}")


def create_video_from_image_smooth(
        image_path: str,
        output_path: str,
        duration: int = 5,
        resolution: tuple = (1920, 1080),
        fps: int = 30,
        zoom_factor: float = 1.01,
        use_background_fill: bool = True
):
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图片 '{image_path}'")
        return

    width, height = resolution
    final_filter_complex = ""

    if use_background_fill:
        # 方案A: 使用模糊背景填充
        filter_complex_base = (
            ### <<< 修正：为 split 滤镜明确指定输入流 [0:v]
            f"[0:v]split[bg][fg];"
            f"[bg]scale=w='if(gte(iw/ih,{width}/{height}),-1,{width})':h='if(gte(iw/ih,{width}/{height}),{height},-1)',"
            f"gblur=sigma=20,crop={width}:{height}[bg_pp];"
            f"[fg]scale=w='if(gte(iw/ih,{width}/{height}),{width},-1)':h='if(gte(iw/ih,{width}/{height}),-1,{height})'[fg_pp];"
            "[bg_pp][fg_pp]overlay=(W-w)/2:(H-h)/2[overlay_out];"
        )
    else:
        # 方案B: 使用黑边
        filter_complex_base = (
            ### <<< 优化：为 color 滤镜添加时长，使其与视频总长一致
            f"color=c=black:s={width}x{height}:d={duration}[black_bg];"
            f"[0:v]scale=w='if(gte(iw/ih,{width}/{height}),{width},-2)':h='if(gte(iw/ih,{width}/{height}),-2,{height})'[fg_scaled];"
            f"[black_bg][fg_scaled]overlay=(W-w)/2:(H-h)/2[overlay_out];"
        )

    # 动画滤镜部分作用于 [overlay_out]
    zoom_expr = f"1+({zoom_factor}-1)*t/{duration}"
    filter_complex_animation = (
        f"[overlay_out]scale=w='iw*({zoom_expr})':h='ih*({zoom_expr})':eval=frame,"
        f"crop=w={width}:h={height}:x='(iw-{width})/2':y='(ih-{height})/2',"
        "format=yuv420p"
    )
    final_filter_complex = filter_complex_base + filter_complex_animation

    command = [
        'ffmpeg', '-y',
        '-loglevel', 'error',

        # --- 输入部分 ---
        '-loop', '1', '-i', image_path,  # 输入0: 图片循环

        # <<< 修改1: 添加虚拟静音音频源
        # anullsrc: 生成空音频 (null source)
        # channel_layout=stereo: 立体声
        # sample_rate=44100: 采样率 44.1kHz (通用标准)
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',

        # --- 滤镜部分 ---
        '-filter_complex', final_filter_complex,

        # --- 视频编码设置 ---
        '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '18',

        # <<< 修改2: 添加音频编码设置
        '-c:a', 'aac',  # 使用 AAC 编码 (MP4标准音频)
        '-b:a', '128k',  # 音频码率 (静音其实占用很小，但指定一下比较规范)

        # --- 输出控制 ---
        # <<< 修改3: 确保音频长度被截断
        # -t 已经全局指定了时长，它会同时切断视频流和静音音频流
        '-t', str(duration),
        '-r', str(fps),

        # -shortest 是个双保险，确保以最短的流（通常是受 -t 控制的那个）为准结束
        '-shortest',

        output_path
    ]

    print("正在生成平滑动画视频，请稍候...")
    print(f"执行的 FFmpeg 命令: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频 '{output_path}' 生成成功！")
    except subprocess.CalledProcessError as e:
        print("\n视频生成失败！")
        print(f"FFmpeg 错误信息:\n{e.stderr}")

def get_frame_at_time_safe(video_path: str, time_str: str) -> np.ndarray | None:
    """
    从视频中提取指定时间点的帧，并在发生任何错误时安全地回退到第一帧。

    - 如果成功，返回目标时间的帧。
    - 如果时间格式错误、时间超出范围或读取目标帧失败，则返回视频的第一帧。
    - 如果视频文件无法打开或无法读取第一帧，则返回 None。

    参数:
    - video_path (str): 视频文件的路径。
    - time_str (str): "HH:MM:SS" 或 "MM:SS" 格式的时间字符串。

    返回:
    - np.ndarray: OpenCV格式的图像帧。
    - None: 仅在视频文件无法打开或损坏时返回。
    """
    # 1. 尝试打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"严重错误: 无法打开视频文件: {video_path}。无法获取任何帧。")
        return None

    # 2. 立即读取第一帧作为备用
    ret_first, first_frame = cap.read()
    if not ret_first:
        print(f"严重错误: 视频 '{video_path}' 可打开但无法读取第一帧。")
        cap.release()
        return None

    try:
        # 3. 尝试解析时间并定位目标帧（正常流程）
        try:
            total_seconds = time_to_ms(time_str) / 1000
        except ValueError as e:
            # 如果时间格式解析失败，直接触发回退
            raise ValueError(f"时间格式不正确 ({e})") from e

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise IOError("无法读取视频的帧率 (FPS)。")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        if total_seconds > video_duration:
            raise ValueError(f"指定时间 {time_str} 超出视频总时长 ({video_duration:.2f}s)")

        target_frame_index = int(total_seconds * fps)

        # 对于非常接近第一帧的情况，直接使用已读取的第一帧
        if target_frame_index == 0:
            return first_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ret, target_frame = cap.read()

        if not ret:
            raise IOError(f"无法在时间点 {time_str} 读取到帧")

        # 如果一切顺利，返回目标帧
        return target_frame

    except Exception as e:
        # 4. 如果try块中出现任何异常，执行回退逻辑
        print(f"处理时发生异常: {e}")
        print(">>> 已触发回退机制，将返回视频的第一帧。")
        return first_frame

    finally:
        # 5. 确保无论如何都释放视频捕获对象
        cap.release()



def add_text_adaptive_padding(input_video_path, output_video_path, text_events, font_path=None,
                                    padding_ratio=0.1):
    """
    自适应地为视频添加边框和文字，实现文字靠近视频上边界的“底部对-齐”效果。

    Args:
        input_video_path (str): 输入视频的文件路径。
        output_video_path (str): 输出视频的文件路径。
        text_events (list): 包含文字事件的列表。
        font_path (str, optional): 字体文件路径。
        padding_ratio (float, optional): 添加的边框高度占原始视频高度的比例。
    """
    # --- 1. 参数校验和准备 (不变) ---
    if not os.path.exists(input_video_path):
        print(f"错误：输入视频文件不存在 -> {input_video_path}")
        return
    if font_path is None:
        font_path = 'C:/Windows/Fonts/msyhbd.ttc'
        print(f"提示：未使用指定字体，将尝试使用默认字体 -> {font_path}")
    if not os.path.exists(font_path):
        print(f"错误：字体文件不存在 -> {font_path}")
        return
    original_w, original_h, _, _ = probe_video_new(input_video_path)
    if not all([original_w, original_h]):
        print("错误：无法获取有效的视频尺寸。")
        return

    # --- 2. 计算新画布尺寸和视频位置 (不变) ---
    top_padding = int(original_h * padding_ratio)
    output_w = original_w
    output_h = original_h + top_padding
    video_y_start = top_padding
    if original_w / original_h > 1.5:
        bottom_padding = top_padding // 2
        top_padding = top_padding + bottom_padding
        output_h = original_h + top_padding + bottom_padding
        video_y_start = top_padding

    # --- 3. 构建滤镜链 ---
    base_filter = f"pad={output_w}:{output_h}:0:{video_y_start}:color=black"
    drawtext_filters = []
    escaped_font_path = _escape_ffmpeg_path(font_path)
    for event in text_events:
        text_list = event.get('text_list', [])
        if not text_list: continue

        start_time = event.get('start_time', 0)
        end_time = event.get('end_time', 99999)
        if start_time >= end_time: continue

        colors = event.get('color_config', {})
        PALETTE = ['#FFFFFF', '#FF4C4C', '#FFD700']  # 白 / 黑 / 金
        fontcolor = colors.get('fontcolor', random.choice(PALETTE))
        # fontcolor = colors.get('fontcolor', '#FFD700')
        shadowcolor = colors.get('shadowcolor', 'black@0.8')

        # --- 字体大小计算逻辑 (保持不变，依然健壮) ---
        margin_ratio = 0.0
        line_spacing_ratio = 0.1
        available_width = output_w * 0.9
        available_height = top_padding * (1.0 - margin_ratio * 2)
        longest_text = max(text_list, key=len) if any(text_list) else ''
        fontsize_w = (available_width / len(longest_text)) if longest_text else 9999
        num_lines = len(text_list)
        if num_lines > 1:
            denominator = num_lines + (num_lines - 1) * line_spacing_ratio
            fontsize_h = available_height / denominator
        else:
            fontsize_h = available_height
        fontsize = min(fontsize_w, fontsize_h, top_padding / 2)

        # === NEW: 重新计算文本块起始位置以实现“底部对齐” ===

        # 1. 计算单行高度（字体+行间距）
        line_height = fontsize * (1 + line_spacing_ratio)

        # 2. 计算整个文本块的总高度
        # 总高度 = (行数 - 1) * 行高 + 最后一行的字体高度
        total_text_block_height = (num_lines - 1) * line_height + fontsize

        # 3. 计算文本块的起始Y坐标（反推法）
        # 底部锚点 = 视频上边界 - 安全边距
        bottom_anchor = video_y_start - (top_padding * margin_ratio)
        # 起始Y坐标 = 底部锚点 - 文本块总高度
        text_block_y_start = bottom_anchor - total_text_block_height

        # 确保起始点不为负（在文本极多的情况下）
        text_block_y_start = max(0, text_block_y_start)
        # ======================================================

        for i, text in enumerate(text_list):
            if not text: continue

            # 每行的Y坐标计算方式不变，因为它依赖于起始点
            current_y = text_block_y_start + i * line_height
            escaped_text = _escape_ffmpeg_text(text)

            filter_str = (
                f"drawtext="
                f"fontfile='{escaped_font_path}':"
                f"text='{escaped_text}':"
                f"fontsize={fontsize}:"
                f"fontcolor='{fontcolor}':"
                f"shadowcolor='{shadowcolor}':shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:"
                f"y={current_y}:"
                f"enable='between(t,{start_time},{end_time})'"
            )
            drawtext_filters.append(filter_str)

    # --- 4. 组合最终滤镜并构建命令 (不变) ---
    all_filters = [base_filter] + drawtext_filters
    full_filter_chain = ",".join(all_filters)
    command = [
        'ffmpeg', '-i', input_video_path,
        '-filter_complex', f"[0:v]{full_filter_chain}[outv]",
        '-map', '[outv]', '-map', '0:a?',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-c:a', 'copy', '-y', output_video_path # 改为 copy
    ]
    print("即将执行的 FFmpeg 命令:")
    print(shlex.join(command))

    # --- 5. 执行命令 (不变) ---
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        print(f"\n视频处理成功！输出文件位于: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("\n--- FFmpeg 处理失败! ---")
        print("FFmpeg 返回码:", e.returncode)
        print("FFmpeg 错误信息:\n" + e.stderr)


def _escape_ffmpeg_path(path):
    """为 FFmpeg 滤镜中的文件路径进行转义，特别处理 Windows 路径。"""
    if platform.system() == 'Windows':
        return path.replace('\\', '\\\\').replace(':', '\\:')
    return path



def add_bgm_to_video(video_path: str, bgm_path: str, output_path: str, volume_percentage: int = 20, auto_compute=False, rate=1):
    """
    为视频添加背景音乐(BGM)。

    功能:
    - 如果 BGM 时长小于视频，则循环播放 BGM。
    - 将 BGM 音量调整为指定百分比（例如 20 表示 20% 的原始音量）。
    - 保留视频原声，与 BGM 进行混合。如果视频无原声，则仅添加 BGM。
    - 视频流直接复制，不重新编码，以保证速度和画质。
    - 输出视频的时长与原视频完全一致。

    参数:
    - video_path (str): 输入视频的文件路径。
    - bgm_path (str): BGM 音频文件的路径。
    - output_path (str): 输出视频的文件路径。
    - volume_percentage (int, optional): BGM 的音量百分比 (0-100)。默认为 20。

    返回:
    - bool: 如果成功，返回 True。

    抛出异常:
    - FileNotFoundError: 如果输入文件或 ffmpeg/ffprobe 命令不存在。
    - ValueError: 如果音量百分比无效。
    - subprocess.CalledProcessError: 如果 ffmpeg 命令执行失败。
    """
    # 1. 检查依赖和输入参数
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("错误: ffmpeg 命令未找到。请确保已安装 ffmpeg 并将其添加至系统 PATH。")
    if not shutil.which("ffprobe"):
        raise FileNotFoundError(
            "错误: ffprobe 命令未找到。请确保已安装 ffmpeg (通常包含 ffprobe) 并将其添加至系统 PATH。")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"输入视频文件未找到: {video_path}")
    if not os.path.exists(bgm_path):
        raise FileNotFoundError(f"BGM 文件未找到: {bgm_path}")
    if not 0 <= volume_percentage <= 100:
        raise ValueError("音量百分比必须在 0 到 100 之间。")

    if auto_compute:
        bgm_volume = get_average_volume(bgm_path)
        video_volume = get_average_volume(video_path)
        volume_percentage = bgm_volume / video_volume * 100
        volume_percentage *= 0.2
        volume_percentage *= rate
        if volume_percentage > 100:
            volume_percentage = 100

        print(f"自动计算的 BGM 音量百分比: {volume_percentage:.2f}% bgm平均音量: {bgm_volume:.2f}dBFS, 视频平均音量: {video_volume:.2f}dBFS")

    # 2. 构建 ffmpeg 命令
    # 将百分比转换为 ffmpeg 的音量因子（例如 20 -> 0.2）
    volume_factor = volume_percentage / 100.0

    # 检查视频是否已有音轨
    video_has_audio = _probe_has_audio(video_path)

    # -i video.mp4          -> 输入视频 (流 0)
    # -stream_loop -1       -> 无限循环下一个输入
    # -i bgm.mp3            -> 输入BGM (流 1)
    # -filter_complex       -> 定义复杂的滤镜图
    #   "[1:a]volume=...[bgm]" -> 将BGM(流1的音频)调整音量，并标记为[bgm]
    #   "[0:a][bgm]amix=..."   -> 如果视频有原声(流0的音频)，则将其与[bgm]混合
    # -map 0:v              -> 映射视频流
    # -map "[a_out]"        -> 映射处理后的音频流
    # -c:v copy             -> 复制视频流，不重新编码
    # -shortest             -> 使输出文件的时长与最短的输入流（即原视频）一致
    # -y                    -> 覆盖输出文件

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",  # 只输出错误信息
        "-hide_banner",  # 隐藏启动横幅
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", bgm_path,
    ]

    if video_has_audio:
        # 混合原声和BGM
        filter_complex = f"[0:a]volume=1.0[orig_a]; [1:a]volume={volume_factor}[bgm]; [orig_a][bgm]amix=inputs=2:duration=first:normalize=0[a_out]"
        map_audio = "[a_out]"
    else:
        # 视频无原声，仅处理BGM
        filter_complex = f"[1:a]volume={volume_factor}[a_out]"
        map_audio = "[a_out]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", f"{map_audio}",
        "-c:v", "copy",  # 直接复制视频流，速度快
        "-c:a", "aac",  # 使用高质量的 AAC 音频编码器
        "-b:a", "192k",  # 设置音频比特率为 192k
        "-shortest",
        output_path
    ])

    # 3. 执行命令
    print("--------------------------------------------------")
    print("开始为视频添加背景音乐...")
    print(f"  输入视频: {video_path}")
    print(f"  BGM: {bgm_path}")
    print(f"  输出视频: {output_path}")
    print(f"  BGM 音量: {volume_percentage}%")
    # 使用 ' '.join 打印一个易于阅读和复制的命令版本
    print(f"  执行的 FFmpeg 命令:\n  {' '.join(cmd)}")
    print("--------------------------------------------------")

    try:
        # 使用 Popen 以便实时看到 ffmpeg 的输出，对于长时间任务更友好
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            # 你可以在这里解析ffmpeg的进度，但为了简单起见，我们只打印它
            print(line.strip())

        process.wait()  # 等待命令执行完成

        if process.returncode != 0:
            # 如果ffmpeg返回了错误码
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print(f"\n处理完成！带BGM的视频已保存至: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print("\nFFmpeg 执行失败!")
        print(f"错误码: {e.returncode}")
        # 由于我们重定向了stderr，错误信息会在上面的循环中打印出来
        return False
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        return False


def _probe_has_audio(path):
    """
    一个内部辅助函数，用于检查媒体文件是否包含音频流。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",  # 只选择第一个音频流
        "-show_entries", "stream=codec_type",
        "-of", "json",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 如果有音频流，输出将包含 "audio"
    return "audio" in result.stdout



def get_average_volume(media_path: str) -> float | None:
    """
    计算给定媒体文件（音频或视频）的平均音量。

    该函数会加载媒体文件，并计算其音量的分贝值（dBFS）。
    如果文件是视频，它会自动提取音频部分进行计算。

    :param media_path: 媒体文件的路径（可以是音频或视频）。
    :return: 以dBFS为单位的平均音量。如果文件无法加载或为完全静音，
             对于无法加载的情况返回 None，对于完全静音的情况返回 -inf。
    """
    if not os.path.exists(media_path):
        print(f"错误：文件不存在于路径: {media_path}")
        return None

    try:
        print(f"正在加载媒体文件: {media_path}...")
        # AudioSegment.from_file 可以智能处理音频和视频文件（提取音频）
        audio = AudioSegment.from_file(media_path)
    except Exception as e:
        print(f"错误：无法加载媒体文件。请确保文件路径正确且FFmpeg已正确安装。错误信息: {e}")
        return None

    # .dBFS 属性可以计算出平均音量
    average_dbfs = audio.dBFS

    # -float('inf') 表示完全的静音
    if average_dbfs == -float('inf'):
        print("警告：输入媒体文件似乎是完全静音的。")

    print(f"计算出的平均音量为: {average_dbfs:.2f} dBFS")
    return average_dbfs


def clip_and_merge_segments(video_path, remaining_segments, output_path):
    """
    根据时间段截取视频并合并，处理完后自动清理临时文件。
    remaining_segments 示例: [(1000, 30000), (60000, 90000)]
    """
    temp_files = []

    try:
        # 1. 遍历截取生成临时片段
        for i, (start, end) in enumerate(remaining_segments):
            # 构造临时文件名，放在输出目录，例如: temp_0_output.mp4
            temp_name = f"temp_segment_{i}_{os.path.basename(output_path)}"
            temp_path = os.path.join(os.path.dirname(output_path), temp_name)

            # 调用你的截取函数
            clip_video_ms(video_path, start, end, temp_path)

            # 记录生成的临时文件路径
            temp_files.append(temp_path)

        # 2. 合并片段
        if temp_files:
            merge_videos_ffmpeg(temp_files, output_path)
            print(f"处理完成，输出文件：{output_path}")

    finally:
        # 3. 清理临时文件 (无论成功或报错都会执行)
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)


def gen_ending_video(text, output_path, origin_ending_video_path, voice_info):
    """
    生成结尾视频（测试用），结尾语为txt
    """
    voice_name = "zh-CN-XiaoxiaoNeural"
    rate = "+30%"
    pitch = '+30Hz'
    if voice_info is not None:
        voice_name = voice_info.get('voice_name', voice_name)
        rate = voice_info.get('rate', "+30%")
        pitch = voice_info.get('pitch', '+30Hz')

    output_path = pathlib.Path(output_path)
    audio_path = output_path.with_suffix(".mp3")
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        rate=rate,
        pitch=pitch,
    )
    video_duration = probe_duration(origin_ending_video_path)
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(video_duration * 1000),
        'outputPath': str(audio_path),
        'trimmedDuration': duration,
    }]
    with_audio_path = output_path.with_name(output_path.stem + "_with_audio.mp4")
    redub_video_with_ffmpeg(video_path=origin_ending_video_path, segments_info=segments_info, output_path=str(with_audio_path), keep_original_audio=True)

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text
    }]
    add_subtitles_to_video(
        video_path=str(with_audio_path),
        subtitles_info=subtitle_data,
        output_path=str(output_path),
        font_size=60,
        bottom_margin=30
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())

def get_media_dimensions(file_path):
    """使用 ffprobe 获取媒体文件的宽度和高度。"""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', file_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data['streams'][0]['width'], data['streams'][0]['height']
    except Exception as e:
        print(f"错误: 无法获取 '{file_path}' 的尺寸。错误信息: {e}")
        return None, None


def add_transparent_watermark(
        video_path: str,
        watermark_path: str,
        output_path: str,
        relative_width: float = 0.05,
        opacity: float = 1,
        position: str = "top_left"
):
    """
    使用 ffmpeg 为视频添加一个圆形的、动态缩放的半透明水印。
    它会取水印图片的内切圆部分。
    """
    position_map = {
        "top_left": "10:10",
        "top_right": "W-w-10:10",
        "bottom_left": "10:H-h-10",
        "bottom_right": "W-w-10:H-h-10"
    }

    if position not in position_map:
        raise ValueError("位置参数无效。请从 'top_left', 'top_right', 'bottom_left', 'bottom_right' 中选择。")

    try:
        video_width, _ = get_media_dimensions(video_path)
        watermark_scaled_width = int(video_width * relative_width)

        # 这是修改的核心：构建一个新的 filter_complex 字符串
        filter_complex = (
            # 1. 缩放水印图片，并确保它有 RGBA 格式以便修改 alpha 通道
            f"[1:v]scale={watermark_scaled_width}:-1,format=rgba,"
            # 2. 使用 geq 滤镜创建圆形蒙版并应用透明度
            #    r='r(X,Y)': 保持原始的 R, G, B 通道不变
            #    a='...':     重写 Alpha (透明) 通道
            #    pow(X-W/2,2)+pow(Y-H/2,2) <= pow(min(W,H)/2,2) : 这是圆的方程，判断像素是否在内切圆内
            #    if(condition, true_val, false_val) : 三元表达式
            #    true_val: opacity * 255 (Alpha通道范围是0-255)
            #    false_val: 0 (完全透明)
            "geq=r='r(X,Y)':a='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),"
            f"{opacity}*255,0)'[wm];"
            # 3. 将处理好的圆形水印 [wm] 叠加到主视频 [0:v] 上
            f"[0:v][wm]overlay={position_map[position]}"
        )

        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', watermark_path,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',  # 显式指定编码器
            '-crf', '23',  # 近无差别质量；需要更好则减小（比如16或0）
            '-c:a', 'copy',
            # '-preset', 'ultrafast',  # 均衡速度/质量；开发时可用 veryslow
            '-y',
            output_path
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"圆形动态缩放水印添加成功，已保存至: {output_path}")

    except (FileNotFoundError, ValueError) as e:
        print(f"错误：{e}")
        print("请确保您的系统中已经正确安装了 ffmpeg 和 ffprobe。")
    except subprocess.CalledProcessError as e:
        print("ffmpeg 或 ffprobe 在执行过程中返回了一个错误：")
        print(e.stderr)


def _get_image_dimensions(image_path: str) -> tuple[int, int] or None:
    # (此辅助函数无需修改)
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', image_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        return data['streams'][0]['width'], data['streams'][0]['height']
    except Exception as e:
        print(f"错误: 无法获取图片尺寸 '{image_path}'.")
        print(f"具体错误: {e}")
        return None

def create_enhanced_cover(
        input_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        position: str = 'top_third',
        color_theme: str = 'auto',
        font_size_ratio: float = 1.0,
        line_spacing_ratio: float = 1.4,
        overwrite: bool = True
) -> str or None:
    if not all([os.path.exists(input_image_path), os.path.exists(font_path)]):
        print("错误: 输入文件或字体文件未找到。")
        return None

    dimensions = _get_image_dimensions(input_image_path)
    if not dimensions: return None
    img_w, img_h = dimensions
    true_high = int(img_w * 9 / 16)

    if not text_lines:
        print("警告: 未提供任何文字，将直接复制图片。")
        if overwrite or not os.path.exists(output_image_path):
            shutil.copy(input_image_path, output_image_path)
        return output_image_path

    # !! 关键修改 1: 优化颜色主题，并增强阴影对比度 !!
    color_themes = {
        # # 主题1: 经典白字黑边 (最通用，最清晰)
        'classic_white': {'fontcolor': 'White', 'shadowcolor': 'black@0.8'},
        # # 主题2: 活力黄黑配 (最醒目，适合娱乐内容)
        'vibrant_yellow': {'fontcolor': '#FFD700', 'shadowcolor': 'black@0.85'},
        'cyber_cyan': {'fontcolor': '0x00FFFF', 'shadowcolor': 'black@0.4'},
        'energetic_orange': {'fontcolor': '#FF6347', 'shadowcolor': 'white@0.8'},
        'neon_magenta': {'fontcolor': '#FF00FF', 'shadowcolor': 'black@0.7'},  # 强烈、骚动感，适合潮流/娱乐
        'electric_purple': {'fontcolor': '#8A2BE2', 'shadowcolor': 'black@0.6'},  # 科幻/科技风
        'hot_pink': {'fontcolor': '#FF1493', 'shadowcolor': 'black@0.7'},  # 青年/时尚向
        'neon_orange': {'fontcolor': '#FF4500', 'shadowcolor': 'black@0.75'},  # 活力火爆型（通知/CTA）
        'lime_neon': {'fontcolor': '#CCFF00', 'shadowcolor': 'black@0.8'},  # 非常抓眼球的高亮绿
        'teal_turquoise': {'fontcolor': '#00CED1', 'shadowcolor': 'black@0.5'},  # 清爽又醒目，适合科技/医疗类
        'cobalt_blue': {'fontcolor': '#0047AB', 'shadowcolor': 'white@0.85'},  # 稳重但显眼，适合专业/财经
        'crimson_red': {'fontcolor': '#DC143C', 'shadowcolor': 'black@0.6'},  # 强烈紧迫感（促销/警示）
        'neon_blue': {'fontcolor': '#1E90FF', 'shadowcolor': 'black@0.6'},  # 网络感强，适合视频标题
        'solar_gold': {'fontcolor': '#FFB400', 'shadowcolor': 'black@0.8'},  # 黄金感，传达价值/热度
        'icy_cyan': {'fontcolor': '#B0F2FF', 'shadowcolor': 'black@0.9'},  # 冰爽高亮，适合科技/潮流背景
        'pink_purple_gradient': {'fontcolor': '#FF6EC7', 'shadowcolor': 'black@0.6'},  # 单色代替：建议配合轻微渐变背景

    }

    # 如果指定的主题不存在，或为 'auto'，则从预设中随机选择
    if color_theme not in color_themes or color_theme == 'auto':
        # 默认随机选择，但可以优先选择最经典的
        # chosen_theme = color_themes['classic_white']
        chosen_theme = random.choice(list(color_themes.values()))
    else:
        chosen_theme = color_themes[color_theme]

    longest_line = max(text_lines, key=len)
    longest_line_size = max(8, len(longest_line))  # 避免极端短文本导致字体过大
    target_text_width = img_w * 0.95
    estimated_char_width_ratio = 1.0
    font_size = int(min((target_text_width / longest_line_size), img_h / 4) * font_size_ratio)

    # !! 关键修改 2: 增加阴影偏移量，模拟更厚的描边效果 !!
    # 将偏移量从原来的5%提升到8%
    shadow_offset = max(2, int(font_size * 0.06))

    line_height = int(font_size * line_spacing_ratio)
    total_text_height = line_height * (len(text_lines) - 1) + font_size

    escaped_font_path = font_path.replace(':', '\\:') if os.name == 'nt' else font_path

    position_map = {'center': img_h / 2, 'top_third': (img_h / 2 - true_high / 2 + font_size / 2),
                    'bottom_third': img_h * 0.75}
    block_y_center = position_map.get(position, img_h * 0.5)  # 默认居中
    start_y = block_y_center - total_text_height / 2

    filters = []
    for i, line in enumerate(text_lines):
        line_y = start_y + i * line_height
        x_expr = '(w-text_w)/2'

        drawtext_options = {
            'fontfile': f"'{escaped_font_path}'",
            'text': f"'{line.replace(':', '\\:').replace('%', '\\%').replace('\'', '')}'",
            'fontsize': str(font_size),
            'fontcolor': chosen_theme['fontcolor'],
            'x': x_expr,
            'y': str(line_y),
            'shadowcolor': chosen_theme['shadowcolor'],
            'shadowx': str(shadow_offset),
            'shadowy': str(shadow_offset)
        }
        filters.append("drawtext=" + ":".join(f"{k}={v}" for k, v in drawtext_options.items()))

    vf_string = ",".join(filters)
    command = ['ffmpeg', '-i', input_image_path, '-vf', vf_string]
    if overwrite: command.append('-y')
    command.append(output_image_path)

    print(f"主题: {chosen_theme}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎉 成功! 优化后的封面已保存到 '{output_image_path}'")
        return output_image_path
    except subprocess.CalledProcessError as e:
        print("FFMPEG 执行失败!")
        print(f"错误码: {e.returncode}")
        print("FFMPEG 输出 (stderr):")
        print(e.stderr)
        return None


def get_video_info(video_path: str):
    """
    使用 ffprobe 获取视频的 FPS 和总时长
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration,nb_frames",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]

        # 计算 FPS (处理 "60000/1001" 这种分数格式)
        fps_str = stream.get("r_frame_rate", "30/1")
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)

        # 获取总帧数 (nb_frames 有时为空，需要容错)
        nb_frames = stream.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            # 如果读不到总帧数，尝试用 时长 * fps 估算
            duration = float(stream.get("duration", 0))
            total_frames = int(duration * fps)

        return fps, total_frames
    except Exception as e:
        raise ValueError(f"无法获取视频信息，请检查 ffprobe 是否安装或视频路径: {e}")


def save_frames_around_timestamp_ffmpeg(
        video_path: str,
        timestamp,
        num_frames: int,
        output_dir: str,
        time_duration_s=None
) -> List[str]:
    """
    使用 FFmpeg 从视频中在给定时间戳前后各截取 num_frames 帧。
    逻辑尽量模拟 OpenCV 版本：如果所有目标文件都已存在，则跳过 FFmpeg 执行。
    """

    # 1. 准备数据
    ts_target_sec = float(timestamp)

    # 获取视频信息 (假设 get_video_info 返回 fps, total_frames)
    # 这里的 probe_video 和 get_video_info 需要你自己提供或实现
    # video_info = probe_video(video_path)
    fps, total_frames = get_video_info(video_path)

    if fps <= 0:
        raise ValueError("无效的 FPS")

    if time_duration_s:
        num_frames = int(fps * time_duration_s)

    # 2. 计算截取范围 (Index)
    target_idx = int(round(ts_target_sec * fps))
    start_idx = max(0, target_idx - num_frames)
    # 限制 end_idx，防止超出视频末尾
    end_idx = min(total_frames - 1, target_idx + num_frames)

    extract_count = end_idx - start_idx + 1

    if extract_count <= 0:
        return []

    # 3. 预先计算所有目标文件路径，检查是否需要执行
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    missing_files = False  # 标记是否缺少文件

    # 用于后续重命名的映射列表: [(frame_idx, output_path), ...]
    frame_map = []

    print(f"FPS: {fps}, 目标时间: {ts_target_sec}s, 截取范围: {start_idx} - {end_idx} {video_path}")

    for i in range(extract_count):
        current_idx = start_idx + i
        current_sec = current_idx / fps
        current_ms = int(current_sec * 1000)
        filename = f"frame_{current_ms}.png"
        out_path = os.path.join(output_dir, filename)

        frame_map.append((current_idx, out_path))
        saved_paths.append(out_path)

        if not os.path.exists(out_path):
            missing_files = True

    # --- 逻辑对齐点：如果所有文件都存在，直接返回，不运行 FFmpeg ---
    if not missing_files:
        # print("所有文件已存在，跳过提取。") # 可选日志
        return saved_paths

    # 4. 只有当确实缺文件时，才配置并运行 FFmpeg

    # FFmpeg start_time
    start_time_sec = start_idx / fps
    # print(f"FFmpeg Seek 时间点: {start_time_sec:.3f}s") # 调试用，为了对齐旧日志可不打

    temp_dir = os.path.join(output_dir, "temp_ffmpeg_out")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time_sec:.6f}",
        "-i", video_path,
        "-frames:v", str(extract_count),
        "-vsync", "0",
        "-q:v", "2",
        "-f", "image2",  # 显式指定格式
        os.path.join(temp_dir, "%05d.png")
    ]

    try:
        # 捕获输出以保持洁癖，出错再打印
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 获取指定时间戳周围图片执行出错: {e.stderr.decode()}")
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return []

    # 5. 重命名并移动文件
    temp_files = sorted(os.listdir(temp_dir))

    # 安全检查：FFmpeg 产出的数量可能少于预期（例如到了视频末尾）
    count_to_move = min(len(temp_files), len(frame_map))

    for i in range(count_to_move):
        temp_filename = temp_files[i]
        src_path = os.path.join(temp_dir, temp_filename)

        # 从之前的映射中取出目标路径
        _, final_path = frame_map[i]

        # 移动并覆盖 (shutil.move 在目标存在时行为取决于系统，建议先删后移或直接copy)
        if os.path.exists(final_path):
            os.remove(final_path)

        shutil.move(src_path, final_path)
        # print(f"已保存: {os.path.basename(final_path)}")

    # 清理
    shutil.rmtree(temp_dir)

    # 返回完整的路径列表（包含本来就存在的和新覆盖的）
    return saved_paths


def adjust_audio_duration(
        input_mp3_path: str,
        current_duration: float,
        target_duration: float,
        output_mp3_path: str,
        tolerance: float = 0.1
):
    """
    调整音频时长以接近目标时长，变速比例被限制在指定容差范围内。
    此函数非常健壮：在任何处理失败或无需处理的情况下，它会复制原始音频到输出路径，
    并返回原始时长，而不会抛出异常。

    Args:
        input_mp3_path (str): 输入的 MP3 文件路径。
        current_duration (float): 当前音频的实际时长（秒）。
        target_duration (float): 目标音频时长（秒）。
        output_mp3_path (str): 处理后输出的 MP3 文件路径。
        tolerance (float, optional): 速度调整的容忍差值。例如 0.1 表示速度倍率
                                     必须在 [0.9, 1.1] 之间。默认为 0.1。

    Returns:
        Optional[float]: 如果成功调整，返回处理后音频的实际时长（秒）。
                         如果执行了后备复制操作，返回原始音频时长。
                         如果输入文件不存在，无法复制，则返回 None。
    """

    # --- 统一的后备（Fallback）函数 ---
    def fallback_copy():
        """复制原始文件到目标地址，并返回原始时长。"""
        print("执行后备操作：复制原始文件。")
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_mp3_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            # 复制文件，copy2会同时复制元数据
            shutil.copy2(input_mp3_path, output_mp3_path)
            print(f"已将原始文件复制到: {output_mp3_path}")
            return current_duration
        except Exception as e:
            print(f"致命错误：后备复制操作失败: {e}")
            return None  # 连复制都失败了，返回None

    # --- 主逻辑开始 ---
    try:
        # 1. 输入验证：检查文件是否存在和时长是否有效
        if not os.path.exists(input_mp3_path):
            print(f"错误：输入文件未找到: {input_mp3_path}")
            return None  # 输入文件不存在，无法进行任何操作

        if current_duration <= 0 or target_duration <= 0:
            print("错误：当前时长和目标时长必须为正数。")
            return fallback_copy()

        # 2. 计算并钳制速度倍率
        ideal_speed_factor = current_duration / target_duration
        min_speed = 1.0 - tolerance
        max_speed = 1.0 + tolerance

        # 将理想速度倍率限制在 [min_speed, max_speed] 范围内
        final_speed_factor = max(min_speed, min(ideal_speed_factor, max_speed))

        # 3. 如果速度变化极小，直接复制，无需调用 FFmpeg
        if abs(final_speed_factor - 1.0) < 0.001:
            print("速度变化可忽略不计，无需处理。")
            return fallback_copy()


        print(f"目标时长: {target_duration:.2f}s, 当前时长: {current_duration:.2f}s  理想速度倍率: {ideal_speed_factor:.4f} -> 限制后最终倍率: {final_speed_factor:.4f}")

        # 4. 构建并执行 FFmpeg 命令
        command = [
            'ffmpeg', '-y', '-i', input_mp3_path,
            '-filter:a', f'atempo={final_speed_factor}',
            '-b:a', '192k', output_mp3_path
        ]

        result = subprocess.run(
            command, capture_output=True, text=True, check=False, encoding='utf-8'
        )

        # 5. 检查结果
        if result.returncode == 0:
            adjusted_duration = current_duration / final_speed_factor
            print(f"音频处理成功！最终时长约为: {adjusted_duration:.2f}s")
            return adjusted_duration
        else:
            print("错误：FFmpeg 执行失败。")
            print(f"FFmpeg 错误输出: {result.stderr.strip()}")
            return fallback_copy()

    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")
        return fallback_copy()


def has_audio(path):
    LOG_FILE = r'W:\project\python_project\auto_video\config\error_audio.txt'
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 解析 JSON
        streams = json.loads(result.stdout).get('streams', [])

        # 判断是否存在音频流
        audio_exists = any(stream['codec_type'] == 'audio' for stream in streams)

        # 如果没有音频，写入日志
        if not audio_exists:
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 使用 'a' (append) 模式打开文件，确保追加而不是覆盖
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{current_time}] Path: {path}\n")

        return audio_exists

    except Exception as e:
        # 简单的错误处理，防止 ffprobe 失败导致程序崩溃
        print(f"Error processing {path}: {e}")
        return False


def create_snapshot_from_video(video_path, start_time, end_time, snapshot_path):
    """
    为指定视频在 start_time 和 end_time 中间时间点创建一张截图，并利用文件名实现缓存。

    Args:
        video_path (str): 本地视频文件的路径。
        start_time (str): 开始时间，格式 "HH:MM:SS" (也可为可被 time_to_ms 解析的形式)。
        end_time (str): 结束时间，格式 "HH:MM:SS"。若为 "0"（或解析后为 0），则自动使用视频总时长。
        snapshot_path (str): 目标截图保存路径。

    Returns:
        tuple: 成功时 (截图文件路径, None)，失败时 (None, 错误信息字符串)
    """
    try:
        # 解析时间为毫秒（假定 time_to_ms 已存在并可用）
        start_ms = time_to_ms(start_time)
    except Exception as e:
        return None, f"解析 start_time 失败: {e}"
    try:
        total_ms = int(probe_duration(video_path) * 1000)
    except Exception as e:
        return None, f"获取视频时长失败: {e}"

    try:
        end_ms = time_to_ms(end_time)
    except Exception:
        # 若无法解析 end_time，则视为 0（即使用总时长）
        end_ms = 0

    # 如果 end_time 解析为 0，则使用视频总时长
    if end_ms == 0:
        end_ms = total_ms

    # 边界与合法性处理
    if start_ms < 0:
        start_ms = 0
    if end_ms < 0:
        end_ms = 0
    # 限定不超过总时长
    if start_ms > total_ms:
        start_ms = total_ms
    if end_ms > total_ms:
        end_ms = total_ms

    # 如果 start > end，交换它们
    if start_ms > end_ms:
        start_ms, end_ms = end_ms, start_ms

    # 取中间时间点（向下取整为毫秒整数）
    mid_ms = int((start_ms + end_ms) / 2)
    # 再次确保不超过总时长
    if mid_ms > total_ms:
        mid_ms = total_ms

    target_time_str = ms_to_time(mid_ms)

    # 2. 检查缓存是否存在
    if os.path.exists(snapshot_path):
        print(f"缓存命中，直接返回已存在的截图: {snapshot_path}")
        return snapshot_path, None

    # 3. 如果缓存不存在，则执行 ffmpeg 命令来生成截图
    print(f"缓存未命中，正在为 {video_path} 在 {target_time_str}（中点）生成新截图...")
    command = [
        'ffmpeg',
        '-ss', target_time_str,
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '10',
        '-y',
        '-loglevel', 'error',
        snapshot_path
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(f"截图成功生成: {snapshot_path}")
        return snapshot_path, None

    except FileNotFoundError:
        error_msg = "错误：ffmpeg 命令未找到。请确保 ffmpeg 已安装并配置在系统 PATH 中。"
        print(error_msg)
        return None, error_msg

    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg 执行失败: {e.stderr.strip()}"
        print(error_msg)
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)
        return None, error_msg




def extract_audio(input_video, output_audio=None):
    """
    从视频文件分离音频。
    - input_video: 视频文件路径
    - output_audio: 可选，输出音频路径（若不指定，默认使用输入文件名 + .m4a）
    返回输出文件路径（成功）或抛出 subprocess.CalledProcessError（失败）。
    """
    if output_audio is None:
        # 为了准确率，默认使用 .wav 格式
        base = os.path.splitext(input_video)[0]
        output_audio = base + '_to_asr.wav'
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vn',
        '-ac', '1',
        '-ar', '16000',
        '-acodec', 'pcm_s16le',
        output_audio
    ]

    try:
        # 执行转换
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {e.stderr.decode()}")
        raise e
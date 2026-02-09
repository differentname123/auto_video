# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/1/28 3:43
:last_date:
    2026/1/28 3:43
:description:
    基于 LaMa 的图片与视频去水印/修复工具 (深度优化版)
"""
import os
import time
import shutil
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from utils.inpating.lama_utils import convert_normalized_boxes_to_poly


# ==========================================
# 核心辅助函数：仅在内存中处理单帧修复
# ==========================================
def _inpaint_frame_in_memory(image_pil, box_coords_list, lama):
    """
    内存级修复函数，不涉及文件IO。
    """
    # 转换为 RGB (防止传入 RGBA 或 BGR)
    image = image_pil.convert("RGB")
    width, height = image.size

    # 遍历当前帧需要修复的所有框
    for box_coords in box_coords_list:
        # 1. 生成 Mask
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array([box_coords], dtype=np.int32)
        cv2.fillPoly(mask, pts=points, color=(255))

        # 膨胀 Mask 以覆盖边缘
        kernel = np.ones((10, 10), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        mask_image = Image.fromarray(dilated_mask).convert("L")

        # 2. 计算裁剪区域 (Optimization)
        x_min_coord = np.min(points[:, :, 0])
        y_min_coord = np.min(points[:, :, 1])
        x_max_coord = np.max(points[:, :, 0])
        y_max_coord = np.max(points[:, :, 1])

        margin = 50  # 外扩像素
        crop_x1 = max(0, int(x_min_coord - margin))
        crop_y1 = max(0, int(y_min_coord - margin))
        crop_x2 = min(width, int(x_max_coord + margin))
        crop_y2 = min(height, int(y_max_coord + margin))

        # 3. 裁剪
        cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped_mask = mask_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 4. 推理
        try:
            inpainted_crop = lama(cropped_image, cropped_mask)
            # 5. 回贴
            image.paste(inpainted_crop, (crop_x1, crop_y1))
        except Exception as e:
            print(f"LaMa推理出错: {e}，跳过该框")
            continue

    return image


# ==========================================
# 辅助函数：分段处理逻辑
# ==========================================

def _extract_segment_ffmpeg(video_path, start_time, duration, output_path):
    """
    使用 FFmpeg 快速提取视频片段（保留段）。
    为了保证与 OpenCV 输出的片段能顺利合并，这里进行快速转码到 mpeg4。
    """
    # -ss 在 -i 前面为快速定位（关键帧精度），在 -i 后面为精确解码。
    # 为了速度和兼容性，这里使用 re-encoding (mpeg4) 确保格式与 OpenCV 默认一致。
    # -q:v 2 表示高质量 (1-31, 越小质量越高)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-c:v", "mpeg4", "-q:v", "2",  # 匹配 cv2.VideoWriter_fourcc(*'mp4v')
        "-an",  # 去除音频，最后统一合并
        output_path
    ]
    # 如果片段很长，放在 -i 后面可能慢，但精确。放在前面快但不准。
    # 这里折中：对于非修复段，稍微的帧偏差通常可接受，但为了拼接严谨，建议重编码。

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _process_repair_segment_cv2(cap, start_frame, end_frame, fps, width, height, frames_to_repair, lama, output_path):
    """
    使用 OpenCV + Python 处理修复片段。
    """
    # 设置写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 定位到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 检查是否需要修复
        if current_frame in frames_to_repair:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = frames_to_repair[current_frame]
            try:
                repaired_pil = _inpaint_frame_in_memory(frame_pil, boxes, lama)
                frame_out = cv2.cvtColor(np.array(repaired_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_out)
            except Exception as e:
                print(f"帧 {current_frame} 修复失败: {e}")
                out.write(frame)
        else:
            # 在修复段内但没有标记框的帧（可能是连续段中的间隙），直接写入
            out.write(frame)

        current_frame += 1

    out.release()


# ==========================================
# 核心入口：视频区间修复函数 (优化版)
# ==========================================
def inpaint_video_intervals(video_path, output_path, repair_info_list, lama=None):
    """
    修复视频指定时间段的指定区域。
    采用分段处理策略：不修复的时间段直接调用 FFmpeg 截取，修复的时间段才进 Python 循环。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return

    # 1. 初始化模型
    if lama is None:
        print("初始化 LaMa 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lama = SimpleLama(device=device)

    # 2. 读取视频基础信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}, 时长: {video_duration:.2f}s")

    # 3. 解析修复信息，构建 timeline
    # frames_to_repair: { frame_index: [box_poly, ...] }
    frames_to_repair = {}

    # 标记哪些帧需要修复 (boolean array)
    needs_repair_mask = np.zeros(total_frames, dtype=bool)

    for info in repair_info_list:
        start_ms = info.get('start', 0)
        end_ms = info.get('end', 0)
        boxes = info.get('boxs', [])

        start_frame = int((start_ms / 1000.0) * fps)
        end_frame = int((end_ms / 1000.0) * fps)

        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        # 标记 Mask 和 字典
        if start_frame < end_frame:
            needs_repair_mask[start_frame:end_frame] = True
            for f_idx in range(start_frame, end_frame):
                if f_idx not in frames_to_repair:
                    frames_to_repair[f_idx] = []
                frames_to_repair[f_idx].extend(boxes)

    # 4. 生成处理片段 (Segments)
    # 策略：如果“不需要修复”的间隔太短（例如小于1秒），则为了避免频繁启动FFmpeg开销，依然归为修复段处理。
    min_skip_frames = int(fps * 2.0)  # 2秒内的空隙不切分，直接用Python跑

    segments = []  # List of {'type': 'repair'|'keep', 'start': f, 'end': f}

    is_repairing = False
    seg_start = 0

    # 使用简单的状态机合并片段
    # 为了简化，先生成纯粹的 True/False 序列，再合并短间隙

    # 步骤 4.1: 初始切分
    raw_segments = []
    if len(needs_repair_mask) > 0:
        current_type = needs_repair_mask[0]
        current_start = 0
        for i in range(1, total_frames):
            if needs_repair_mask[i] != current_type:
                raw_segments.append({'type': 'repair' if current_type else 'keep', 'start': current_start, 'end': i})
                current_type = needs_repair_mask[i]
                current_start = i
        raw_segments.append({'type': 'repair' if current_type else 'keep', 'start': current_start, 'end': total_frames})

    # 步骤 4.2: 合并短的 keep 片段到 repair 片段中，减少文件碎片
    merged_segments = []
    if not raw_segments:
        # 视频为空或逻辑错误
        merged_segments.append({'type': 'keep', 'start': 0, 'end': total_frames})
    else:
        current_seg = raw_segments[0]
        for i in range(1, len(raw_segments)):
            next_seg = raw_segments[i]

            # 如果当前是 Keep 且很短，或者下一个是 Keep 且很短，可以考虑合并（这里主要处理 Keep 很短的情况）
            # 逻辑：如果中间夹着一个很短的 Keep，直接把它变成 Repair 处理，避免频繁 IO
            if current_seg['type'] == 'keep' and (current_seg['end'] - current_seg['start'] < min_skip_frames):
                current_seg['type'] = 'repair'  # 标记为 Repair，这样 Python 会读取并原样写入

            # 尝试与下一个合并
            if current_seg['type'] == next_seg['type']:
                current_seg['end'] = next_seg['end']
            else:
                # 再次检查：如果刚刚把 keep 变成了 repair，现在可能可以和下一个 repair 合并
                # 但这里是简单处理：push current, start new
                merged_segments.append(current_seg)
                current_seg = next_seg
        merged_segments.append(current_seg)

    # 再次遍历合并连续的同类型片段（因为上面的逻辑可能遗漏合并）
    final_segments = []
    if merged_segments:
        curr = merged_segments[0]
        for i in range(1, len(merged_segments)):
            nxt = merged_segments[i]
            if curr['type'] == nxt['type']:
                curr['end'] = nxt['end']
            else:
                final_segments.append(curr)
                curr = nxt
        final_segments.append(curr)
    else:
        final_segments = [{'type': 'keep', 'start': 0, 'end': total_frames}]

    print(f"处理计划: 将视频分为 {len(final_segments)} 个片段进行处理")
    for seg in final_segments:
        print(
            f"  - [{seg['type'].upper()}] 帧范围: {seg['start']} -> {seg['end']} (时长: {(seg['end'] - seg['start']) / fps:.2f}s)")

    # 5. 执行处理
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_segments")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    segment_files = []
    start_proc_time = time.time()

    for i, seg in enumerate(final_segments):
        seg_file = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
        start_f = seg['start']
        end_f = seg['end']
        duration = (end_f - start_f) / fps
        start_t = start_f / fps

        if end_f <= start_f:
            continue

        print(f"正在处理片段 {i + 1}/{len(final_segments)} ({seg['type']})...")

        if seg['type'] == 'keep':
            # 调用 FFmpeg 截取
            _extract_segment_ffmpeg(video_path, start_t, duration, seg_file)
        else:
            # 调用 Python 修复
            _process_repair_segment_cv2(cap, start_f, end_f, fps, width, height, frames_to_repair, lama, seg_file)

        if os.path.exists(seg_file):
            segment_files.append(seg_file)
        else:
            print(f"警告: 片段文件生成失败 {seg_file}")

    cap.release()

    # 6. 合并片段 (FFmpeg concat demuxer)
    print("正在合并视频片段...")
    list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(list_path, "w", encoding='utf-8') as f:
        for seg_file in segment_files:
            # 写入绝对路径，防止 FFmpeg 读取路径出错
            abs_path = os.path.abspath(seg_file).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    temp_video_combined = output_path.replace(".mp4", "_temp_combined.mp4")

    # 合并视频流
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",  # 流复制，极快
        temp_video_combined
    ]
    subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 7. 混入原始音频
    # 因为片段合并后是无声的(部分keep可能有声，部分repair无声，混合容易出问题)，
    # 最稳妥是直接把原始视频的音频轨 map 到新视频上。
    print("正在合并原始音频...")
    final_merge_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_combined,  # 视频源
        "-i", video_path,  # 音频源
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",  # 使用原视频的音频
        "-shortest",  # 以最短流为准，防止尾部对齐问题
        output_path
    ]

    result = subprocess.run(final_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 清理
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(temp_video_combined):
        os.remove(temp_video_combined)

    if result.returncode != 0:
        print("FFmpeg 音频合并失败，可能是原视频无音频。")
        # 尝试直接重命名
        if os.path.exists(temp_video_combined):
            os.rename(temp_video_combined, output_path)

    print(f"全部完成，总耗时: {time.time() - start_proc_time:.2f}s，输出文件: {output_path}")


if __name__ == '__main__':
    # -----------------------
    # 测试用例
    # -----------------------

    # 1. 准备模型 (全局加载一次)
    print("预加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_lama = SimpleLama(device=device)

    # # 2. 定义测试参数
    # 请修改为你的实际路径
    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"
    output_video = video_file.replace(".mp4", "_inpainted_optimized.mp4")

    # 模拟输入数据：假设这个水印只在前5秒出现
    formatted_boxes = [[
        [389, 914],
        [1049, 914],
        [1049, 954],
        [389, 954]
    ]]

    repair_info = [
        {
            "start": 0,
            "end": 5000,  # 仅修复前5000毫秒
            "boxs": formatted_boxes
        }
    ]

    try:
        # 3. 调用修复
        inpaint_video_intervals(video_file, output_video, repair_info, lama=global_lama)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
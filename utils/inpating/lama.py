# -- coding: utf-8 --
"""
:description:
    基于 LaMa 的图片与视频去水印/修复工具 (深度优化版)
    功能：自动识别视频修复区间，使用多线程分离IO与计算，基于LaMa模型进行修复。
"""
import os
import time
import shutil
import subprocess
import threading
import json
import itertools
from typing import List, Dict, Tuple, Optional, Any
from queue import Queue

import cv2
import numpy as np
import torch
from PIL import Image

# 尝试导入 LaMa，如果不存在则提示
try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    SimpleLama = None
    print("Warning: simple_lama_inpainting not found. Mock mode or install it.")


# ==========================================
# 核心工具类
# ==========================================

class MaskCache:
    """
    Mask 缓存管理器
    策略：对于静态水印，坐标不变，直接复用生成的 Mask 和 裁剪坐标，避免重复计算 fillPoly 和 dilate。
    """

    def __init__(self):
        # Key: str(boxes_coords), Value: List[(crop_coords, mask_crop_pil)]
        self.cache = {}

    def get_processing_data(self, width: int, height: int, boxes: List[List[int]]) -> List[Tuple]:
        # 使用 JSON 序列化确保 Key 的唯一性和稳定性
        boxes_key = json.dumps(boxes)

        if boxes_key in self.cache:
            return self.cache[boxes_key]

        # --- 缓存未命中，执行计算 ---
        processing_items = []
        margin = 50
        kernel = np.ones((10, 10), np.uint8)

        for box_coords in boxes:
            # 1. 生成基础 Mask
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([box_coords], dtype=np.int32)
            cv2.fillPoly(mask, pts=points, color=255)

            # 2. 膨胀 Mask 以覆盖边缘
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # 3. 计算裁剪坐标 (Crop Coordinates)
            x_min = np.min(points[:, :, 0])
            y_min = np.min(points[:, :, 1])
            x_max = np.max(points[:, :, 0])
            y_max = np.max(points[:, :, 1])

            crop_x1 = max(0, int(x_min - margin))
            crop_y1 = max(0, int(y_min - margin))
            crop_x2 = min(width, int(x_max + margin))
            crop_y2 = min(height, int(y_max + margin))

            # 4. 裁剪 Mask 并转为 PIL L模式
            mask_crop = dilated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            mask_crop_pil = Image.fromarray(mask_crop).convert("L")

            processing_items.append(((crop_x1, crop_y1, crop_x2, crop_y2), mask_crop_pil))

        self.cache[boxes_key] = processing_items
        return processing_items


class VideoReader(threading.Thread):
    """多线程视频帧读取器"""

    def __init__(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, queue: Queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.queue = queue
        self._stop_event = threading.Event()

    def run(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        current_idx = self.start_frame

        while not self._stop_event.is_set() and current_idx < self.end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 提前转换颜色空间 (CPU操作)，减少主线程负担
            # 传输: (帧索引, RGB用于推理, BGR用于直接写入)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.queue.put((current_idx, frame_rgb, frame))
            current_idx += 1

        self.queue.put(None)  # 结束信号

    def stop(self):
        self._stop_event.set()


class VideoWriter(threading.Thread):
    """多线程视频帧写入器"""

    def __init__(self, out_writer: cv2.VideoWriter, queue: Queue):
        super().__init__(daemon=True)
        self.out = out_writer
        self.queue = queue

    def run(self):
        while True:
            frame_bgr = self.queue.get()
            if frame_bgr is None:
                self.queue.task_done()
                break
            self.out.write(frame_bgr)
            self.queue.task_done()


# ==========================================
# 辅助函数
# ==========================================

def _run_ffmpeg(cmd: List[str], description: str = "FFmpeg"):
    """执行 FFmpeg 命令并处理异常"""
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[{description}] Warning/Error: {result.stderr.decode('utf-8', errors='ignore')}")
        return result
    except Exception as e:
        print(f"[{description}] Execution failed: {e}")
        return None


def _extract_segment_ffmpeg(video_path: str, start_time: float, duration: float, output_path: str):
    """使用 FFmpeg 快速提取视频片段（转码为 mpeg4 以匹配 OpenCV 输出流）。"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-c:v", "mpeg4", "-q:v", "2",  # 保持与原逻辑一致的编码
        "-an",  # 去除音频，因为最后会统一合并音频
        output_path
    ]
    _run_ffmpeg(cmd, "Extract Segment")


def _process_repair_segment(cap: cv2.VideoCapture, start_frame: int, end_frame: int,
                            fps: float, width: int, height: int,
                            frames_to_repair: Dict[int, List], lama_model: Any, output_path: str):
    """
    处理需要修复的视频片段：读取 -> Mask -> LaMa Inpaint -> 写入
    """
    # 初始化写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 队列与线程 (限制队列大小以控制内存)
    input_queue = Queue(maxsize=16)
    output_queue = Queue(maxsize=16)

    reader = VideoReader(cap, start_frame, end_frame, input_queue)
    writer = VideoWriter(out, output_queue)
    mask_cache = MaskCache()

    reader.start()
    writer.start()

    try:
        while True:
            item = input_queue.get()
            if item is None:
                output_queue.put(None)
                break

            frame_idx, frame_rgb_np, frame_bgr_orig = item

            # 如果当前帧在需要修复的列表中
            if frame_idx in frames_to_repair:
                try:
                    boxes = frames_to_repair[frame_idx]
                    # 获取缓存的 Mask 数据
                    proc_data = mask_cache.get_processing_data(width, height, boxes)

                    frame_pil = Image.fromarray(frame_rgb_np)

                    # 遍历并修复
                    for (crop_coords, mask_crop_pil) in proc_data:
                        cx1, cy1, cx2, cy2 = crop_coords
                        cropped_image = frame_pil.crop((cx1, cy1, cx2, cy2))
                        inpainted_crop = lama_model(cropped_image, mask_crop_pil)
                        frame_pil.paste(inpainted_crop, (cx1, cy1))

                    # 转回 BGR 写入
                    result_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    output_queue.put(result_bgr)
                except Exception as e:
                    print(f"[Error] Frame {frame_idx}: {e}")
                    output_queue.put(frame_bgr_orig)  # 出错降级：直接写原图
            else:
                output_queue.put(frame_bgr_orig)

    finally:
        reader.join()
        writer.join()
        out.release()


# ==========================================
# 主逻辑
# ==========================================

def inpaint_video_intervals(video_path: str, output_path: str, repair_info_list: List[Dict], lama=None):
    """
    主函数：分析时间轴，分割视频，分别处理（保留/修复），最后合并。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. 初始化模型
    if lama is None:
        if SimpleLama is None:
            raise ImportError("SimpleLama module missing.")
        print("Initializing LaMa model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lama = SimpleLama(device=device)

    # 2. 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Info: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    # 3. 构建修复映射表 (Frame Index -> Boxes) 和 状态数组
    frames_to_repair = {}
    is_repair_frame = np.zeros(total_frames, dtype=bool)

    for info in repair_info_list:
        start_ms, end_ms = info.get('start', 0), info.get('end', 0)
        boxes = info.get('boxs', [])  # 兼容 boxs 拼写

        start_f = max(0, int((start_ms / 1000.0) * fps))
        end_f = min(total_frames, int((end_ms / 1000.0) * fps))

        if start_f < end_f:
            is_repair_frame[start_f:end_f] = True
            for f in range(start_f, end_f):
                frames_to_repair.setdefault(f, []).extend(boxes)

    # 4. 生成时间轴片段 (使用 itertools.groupby 简化逻辑)
    # 逻辑：将连续的 True/False 状态分组为片段
    # 并将过短的“保持”片段(keep)合并为“修复”片段，以减少IO切换开销
    min_skip_frames = int(fps * 2.0)
    segments = []

    # 第一次遍历：生成原始片段
    raw_segments = []
    current_frame = 0
    for is_repair, group in itertools.groupby(is_repair_frame):
        length = len(list(group))
        raw_segments.append({
            'type': 'repair' if is_repair else 'keep',
            'start': current_frame,
            'end': current_frame + length
        })
        current_frame += length

    # 第二次遍历：合并短片段
    if not raw_segments:
        segments = [{'type': 'keep', 'start': 0, 'end': total_frames}]
    else:
        # 初始段
        current_seg = raw_segments[0]
        for next_seg in raw_segments[1:]:
            # 如果当前是 keep 且非常短，将其转化为 repair (避免频繁开关ffmpeg/cv2)
            if current_seg['type'] == 'keep' and (current_seg['end'] - current_seg['start'] < min_skip_frames):
                current_seg['type'] = 'repair'

            # 尝试合并同类型片段
            if current_seg['type'] == next_seg['type']:
                current_seg['end'] = next_seg['end']
            else:
                segments.append(current_seg)
                current_seg = next_seg
        segments.append(current_seg)

    print(f"Plan: Video split into {len(segments)} segments.")

    # 5. 执行分段处理
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_segments_opt")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    segment_files = []
    start_time = time.time()

    for i, seg in enumerate(segments):
        seg_file = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
        sf, ef = seg['start'], seg['end']

        if ef <= sf: continue

        print(f"Processing segment {i + 1}/{len(segments)} [{seg['type'].upper()}]: Frames {sf}-{ef}")

        if seg['type'] == 'keep':
            _extract_segment_ffmpeg(video_path, sf / fps, (ef - sf) / fps, seg_file)
        else:
            _process_repair_segment(cap, sf, ef, fps, width, height, frames_to_repair, lama, seg_file)

        if os.path.exists(seg_file):
            segment_files.append(seg_file)

    cap.release()

    # 6. 合并视频与音频
    if segment_files:
        print("Concatenating segments...")
        list_path = os.path.join(temp_dir, "concat_list.txt")
        temp_video = output_path.replace(".mp4", "_no_audio.mp4")

        with open(list_path, "w", encoding='utf-8') as f:
            for sf in segment_files:
                f.write(f"file '{os.path.abspath(sf).replace(os.sep, '/')}'\n")

        # 合并视频流
        _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", temp_video
        ], "Concat Video")

        # 混入原始音频
        print("Muxing audio...")
        res = _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", output_path
        ], "Mux Audio")

        # 如果混流失败（例如原视频无音频），直接使用无音频版本
        if res and res.returncode != 0:
            print("Audio mux failed (no audio stream?), using video only.")
            if os.path.exists(temp_video):
                shutil.move(temp_video, output_path)
        elif os.path.exists(temp_video):
            os.remove(temp_video)

    # 清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"Done. Total time: {time.time() - start_time:.2f}s. Saved to: {output_path}")


if __name__ == '__main__':
    # -----------------------
    # 测试区域
    # -----------------------
    print("Loading Global Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 如果没安装 simple_lama_inpainting，这里会报错，需自行处理环境
    if SimpleLama:
        global_lama = SimpleLama(device=device)
    else:
        global_lama = None

    # 配置路径
    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"
    output_video = video_file.replace(".mp4", "_inpainted_optimized.mp4")

    # 配置修复参数
    # box结构: [x, y]
    formatted_boxes = [[
        [389, 914], [1049, 914], [1049, 954], [389, 954]
    ]]

    repair_info = [
        {"start": 0, "end": 5000, "boxs": formatted_boxes}
    ]

    if os.path.exists(video_file) and global_lama:
        try:
            inpaint_video_intervals(video_file, output_video, repair_info, lama=global_lama)
        except Exception as e:
            import traceback

            traceback.print_exc()
    else:
        print("Skipping test: Input file not found or model not loaded.")
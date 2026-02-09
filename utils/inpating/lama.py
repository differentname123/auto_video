# -- coding: utf-8 --
"""
:description:
    基于 LaMa 的图片与视频去水印/修复工具 (深度优化 + 极速合并版)

    性能与逻辑说明：
    1. [提取] 使用 mpeg4 编码，保证帧级精准度 (解决音画不同步)。
    2. [提取] 强制 tag 为 mp4v，与 OpenCV 输出完全一致。
    3. [合并] 回归 stream copy (-c copy)，避免全片重编码，恢复原始代码的极速体验。
    4. [核心] 保留 GPU Tensor 优化，推理速度极快。
"""
import os
import time
import shutil
import subprocess
import threading
import json
import itertools
from typing import List, Dict, Tuple, Any
from queue import Queue

import cv2
import numpy as np
import torch

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    SimpleLama = None
    print("Warning: simple_lama_inpainting not found.")


# ==========================================
# 核心工具类
# ==========================================

class MaskCache:
    def __init__(self, device):
        self.device = device
        self.cache = {}

    def get_processing_data(self, width: int, height: int, boxes: List[List[int]]) -> List[Tuple]:
        boxes_key = json.dumps(boxes)
        if boxes_key in self.cache:
            return self.cache[boxes_key]

        processing_items = []
        margin = 50
        kernel = np.ones((10, 10), np.uint8)

        for box_coords in boxes:
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([box_coords], dtype=np.int32)
            cv2.fillPoly(mask, pts=points, color=255)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            x_min = np.min(points[:, :, 0])
            y_min = np.min(points[:, :, 1])
            x_max = np.max(points[:, :, 0])
            y_max = np.max(points[:, :, 1])

            crop_x1 = max(0, int(x_min - margin))
            crop_y1 = max(0, int(y_min - margin))
            crop_x2 = min(width, int(x_max + margin))
            crop_y2 = min(height, int(y_max + margin))

            # 强制 8 对齐
            def align_to_8(v_min, v_max, max_limit):
                length = v_max - v_min
                remainder = length % 8
                if remainder != 0:
                    pad = 8 - remainder
                    if v_max + pad <= max_limit:
                        v_max += pad
                    elif v_min - pad >= 0:
                        v_min -= pad
                    else:
                        v_max -= remainder
                return v_min, v_max

            crop_x1, crop_x2 = align_to_8(crop_x1, crop_x2, width)
            crop_y1, crop_y2 = align_to_8(crop_y1, crop_y2, height)

            mask_crop = dilated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            mask_tensor = torch.from_numpy(mask_crop).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_tensor = (mask_tensor > 0.5).float()
            mask_tensor = mask_tensor.to(self.device)

            processing_items.append(((crop_x1, crop_y1, crop_x2, crop_y2), mask_tensor))

        self.cache[boxes_key] = processing_items
        return processing_items


class VideoReader(threading.Thread):
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
            self.queue.put((current_idx, frame))
            current_idx += 1
        self.queue.put(None)

    def stop(self):
        self._stop_event.set()


class VideoWriter(threading.Thread):
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
    try:
        # Windows下如果不加 creationflags，控制台可能会弹出黑框，可视情况添加
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[{description}] Warning/Error: {result.stderr.decode('utf-8', errors='ignore')}")
        return result
    except Exception as e:
        print(f"[{description}] Execution failed: {e}")
        return None


def _extract_segment_ffmpeg(video_path: str, start_time: float, duration: float, output_path: str):
    """
    [关键修改]
    1. 使用 mpeg4 编码 (-c:v mpeg4) 以确保帧级精准切割 (解决音画不同步)。
    2. 添加 -tag:v mp4v 强制标签，使其与 OpenCV 生成的视频流完全一致。
       这样最后合并时可以使用 -c copy，速度极快。
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-c:v", "mpeg4", "-q:v", "2", "-tag:v", "mp4v",
        "-map", "0:v",
        output_path
    ]
    _run_ffmpeg(cmd, "Extract Segment (Fast Compatible)")


def _process_repair_segment(cap: cv2.VideoCapture, start_frame: int, end_frame: int,
                            fps: float, width: int, height: int,
                            frames_to_repair: Dict[int, List], lama_obj: Any, output_path: str):
    device = lama_obj.device
    model = lama_obj.model

    # OpenCV 默认输出 mp4v，这与我们上面的 extract 对应
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    input_queue = Queue(maxsize=16)
    output_queue = Queue(maxsize=16)

    reader = VideoReader(cap, start_frame, end_frame, input_queue)
    writer = VideoWriter(out, output_queue)
    mask_cache = MaskCache(device)

    reader.start()
    writer.start()

    try:
        while True:
            item = input_queue.get()
            if item is None:
                output_queue.put(None)
                break

            frame_idx, frame_bgr_ref = item
            frame_bgr = frame_bgr_ref.copy()  # 安全拷贝

            if frame_idx in frames_to_repair:
                try:
                    boxes = frames_to_repair[frame_idx]
                    proc_data = mask_cache.get_processing_data(width, height, boxes)

                    for (crop_coords, mask_tensor) in proc_data:
                        cx1, cy1, cx2, cy2 = crop_coords
                        crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                        img_tensor = torch.from_numpy(crop_bgr).permute(2, 0, 1).float().div(255.0)
                        img_tensor = img_tensor.unsqueeze(0).to(device)
                        img_tensor_rgb = img_tensor[:, [2, 1, 0], :, :]

                        with torch.no_grad():
                            inpainted_tensor = model(img_tensor_rgb, mask_tensor)

                        inpainted_bgr = inpainted_tensor[:, [2, 1, 0], :, :]
                        inpainted_bgr = inpainted_bgr.clamp(0, 1).mul(255).byte()
                        inpainted_np = inpainted_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        frame_bgr[cy1:cy2, cx1:cx2] = inpainted_np

                    output_queue.put(frame_bgr)
                except Exception as e:
                    print(f"[Error] Frame {frame_idx}: {e}")
                    output_queue.put(frame_bgr_ref)
            else:
                output_queue.put(frame_bgr)

    finally:
        reader.join()
        writer.join()
        out.release()


# ==========================================
# 主逻辑
# ==========================================

def inpaint_video_intervals(video_path: str, output_path: str, repair_info_list: List[Dict], lama=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if lama is None:
        if SimpleLama is None:
            raise ImportError("SimpleLama module missing.")
        print("Initializing LaMa model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lama = SimpleLama(device=device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Info: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    frames_to_repair = {}
    is_repair_frame = np.zeros(total_frames, dtype=bool)

    for info in repair_info_list:
        start_ms, end_ms = info.get('start', 0), info.get('end', 0)
        boxes = info.get('boxs', [])
        start_f = max(0, int((start_ms / 1000.0) * fps))
        end_f = min(total_frames, int((end_ms / 1000.0) * fps))
        if start_f < end_f:
            is_repair_frame[start_f:end_f] = True
            for f in range(start_f, end_f):
                frames_to_repair.setdefault(f, []).extend(boxes)

    min_skip_frames = int(fps * 2.0)
    segments = []
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

    if not raw_segments:
        segments = [{'type': 'keep', 'start': 0, 'end': total_frames}]
    else:
        current_seg = raw_segments[0]
        for next_seg in raw_segments[1:]:
            if current_seg['type'] == 'keep' and (current_seg['end'] - current_seg['start'] < min_skip_frames):
                current_seg['type'] = 'repair'

            if current_seg['type'] == next_seg['type']:
                current_seg['end'] = next_seg['end']
            else:
                segments.append(current_seg)
                current_seg = next_seg
        segments.append(current_seg)

    print(f"Plan: Video split into {len(segments)} segments.")

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

    if segment_files:
        print("Concatenating segments (Stream Copy Mode)...")
        list_path = os.path.join(temp_dir, "concat_list.txt")
        temp_video = output_path.replace(".mp4", "_no_audio.mp4")

        with open(list_path, "w", encoding='utf-8') as f:
            for sf in segment_files:
                f.write(f"file '{os.path.abspath(sf).replace(os.sep, '/')}'\n")

        # [关键修改] 回归 -c copy
        # 因为所有片段都是 mpeg4/mp4v，这里直接流拷贝，速度极快（秒级）
        res = _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",  # <--- 恢复极速合并
            temp_video
        ], "Concat Video")

        print("Muxing audio...")
        # 混流音频也使用 copy
        res_mux = _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", output_path
        ], "Mux Audio")

        if res_mux and res_mux.returncode != 0:
            print("Audio mux failed, using video only.")
            if os.path.exists(temp_video):
                shutil.move(temp_video, output_path)
        elif os.path.exists(temp_video):
            os.remove(temp_video)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"Done. Total time: {time.time() - start_time:.2f}s. Saved to: {output_path}")


if __name__ == '__main__':
    # -----------------------
    # 测试区域
    # -----------------------
    print("Loading Global Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if SimpleLama:
        global_lama = SimpleLama(device=device)
    else:
        global_lama = None

    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"
    output_video = video_file.replace(".mp4", "_inpainted_fast.mp4")

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
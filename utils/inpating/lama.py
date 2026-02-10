# -- coding: utf-8 --
"""
:description:
    基于 LaMa 的图片与视频去水印/修复工具 (深度优化 + 极速合并 + 降分辨率加速版)

    性能与逻辑说明：
    1. [提取] 使用 mpeg4 编码，保证帧级精准度 (解决音画不同步)。
    2. [提取] 强制 tag 为 mp4v，与 OpenCV 输出完全一致。
    3. [合并] 回归 stream copy (-c copy)，避免全片重编码，恢复原始代码的极速体验。
    4. [核心] 保留 GPU Tensor 优化。
    5. [加速] 在推理前降低修复区域分辨率 (0.5x)，修复后再还原，大幅提升速度。
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
        # 定义默认缩放比例，虽然下面会动态计算，保留此变量以防影响 cache key 生成
        self.shrink_ratio = 0.25

    def get_processing_data(self, width: int, height: int, boxes: List[List[int]]) -> List[Tuple]:
        boxes_key = json.dumps(boxes) + f"_{self.shrink_ratio}"
        if boxes_key in self.cache:
            return self.cache[boxes_key]

        processing_items = []
        margin = 50
        kernel = np.ones((10, 10), np.uint8)

        for box_coords in boxes:
            # 1. 在原图尺寸上生成基础 Mask
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([box_coords], dtype=np.int32)
            cv2.fillPoly(mask, pts=points, color=255)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            x_min = np.min(points[:, :, 0])
            y_min = np.min(points[:, :, 1])
            x_max = np.max(points[:, :, 0])
            y_max = np.max(points[:, :, 1])

            # 原始裁剪坐标
            crop_x1 = max(0, int(x_min - margin))
            crop_y1 = max(0, int(y_min - margin))
            crop_x2 = min(width, int(x_max + margin))
            crop_y2 = min(height, int(y_max + margin))

            # --- 修改开始：计算缩放后的对齐尺寸 ---

            # 先确保原图裁剪区域稍微规整一点（偶数即可）
            if (crop_x2 - crop_x1) % 2 != 0: crop_x2 += 1
            if (crop_y2 - crop_y1) % 2 != 0: crop_y2 += 1

            # 获取原始裁剪区域的宽和高
            orig_w = crop_x2 - crop_x1
            orig_h = crop_y2 - crop_y1

            # [新增逻辑] 动态计算缩放比例：确保最长边不超过 512px
            long_side = max(orig_w, orig_h)
            if long_side > 512:
                dynamic_ratio = 512.0 / long_side
            else:
                # 如果原尺寸小于 512，则不进行缩放，保证最佳画质
                dynamic_ratio = 1.0
            print(f"Original crop size: ({orig_w}x{orig_h}), Dynamic ratio: {dynamic_ratio:.3f}")
            # 计算缩放后的目标宽高
            target_w = int(orig_w * dynamic_ratio)
            target_h = int(orig_h * dynamic_ratio)

            # 强制目标宽高必须是 8 的倍数 (LaMa 模型要求)
            def align_down_to_8(v):
                rem = v % 8
                if rem == 0: return v
                # 如果数值太小，至少保留8
                if v < 8: return 8
                return v - rem  # 向下取整比较安全

            target_w = align_down_to_8(target_w)
            target_h = align_down_to_8(target_h)

            # 截取原始 Mask
            mask_crop_orig = dilated_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            # 将 Mask 缩放到目标尺寸 (使用最近邻插值保持二值化特性)
            if mask_crop_orig.size > 0:
                mask_crop_small = cv2.resize(mask_crop_orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                # 异常保护
                mask_crop_small = np.zeros((target_h, target_w), dtype=np.uint8)

            # 转 Tensor
            mask_tensor = torch.from_numpy(mask_crop_small).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_tensor = (mask_tensor > 0.5).float()
            mask_tensor = mask_tensor.to(self.device)

            # 存储：(原始裁剪坐标, Mask Tensor, 目标小尺寸)
            processing_items.append(((crop_x1, crop_y1, crop_x2, crop_y2), mask_tensor, (target_w, target_h)))
            # --- 修改结束 ---

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

                    # proc_data 解包: ((x1,y1,x2,y2), mask_tensor, (target_w, target_h))
                    for (crop_coords, mask_tensor, target_size) in proc_data:
                        cx1, cy1, cx2, cy2 = crop_coords
                        target_w, target_h = target_size

                        # 1. 截取原分辨率图像
                        crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]

                        # 2. [加速关键] 缩小图像到 target_size
                        # 使用线性插值速度快，效果尚可
                        if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
                            continue  # 防止空切片报错

                        orig_h_crop, orig_w_crop = crop_bgr.shape[:2]

                        small_bgr = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                        # 3. 转 Tensor 并推理
                        img_tensor = torch.from_numpy(small_bgr).permute(2, 0, 1).float().div(255.0)
                        img_tensor = img_tensor.unsqueeze(0).to(device)
                        img_tensor_rgb = img_tensor[:, [2, 1, 0], :, :]  # BGR -> RGB

                        with torch.no_grad():
                            inpainted_tensor = model(img_tensor_rgb, mask_tensor)

                        # 4. 后处理
                        inpainted_bgr_small = inpainted_tensor[:, [2, 1, 0], :, :]  # RGB -> BGR
                        inpainted_bgr_small = inpainted_bgr_small.clamp(0, 1).mul(255).byte()
                        inpainted_np_small = inpainted_bgr_small.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # 5. [还原] 将修复后的小图放大回原始尺寸
                        # 使用线性插值还原
                        inpainted_np_orig = cv2.resize(inpainted_np_small, (orig_w_crop, orig_h_crop),
                                                       interpolation=cv2.INTER_LINEAR)

                        # 6. 贴回原图
                        frame_bgr[cy1:cy2, cx1:cx2] = inpainted_np_orig

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
        seg_start_time = time.time()
        seg_file = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
        sf, ef = seg['start'], seg['end']

        if ef <= sf: continue

        if seg['type'] == 'keep':
            _extract_segment_ffmpeg(video_path, sf / fps, (ef - sf) / fps, seg_file)
        else:
            _process_repair_segment(cap, sf, ef, fps, width, height, frames_to_repair, lama, seg_file)

        if os.path.exists(seg_file):
            segment_files.append(seg_file)
        print(
            f"Processing segment {i + 1}/{len(segments)} [{seg['type'].upper()}]: Frames {sf}-{ef} 耗时 {time.time() - seg_start_time:.2f}s")

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

    video_file = r"W:\project\python_project\auto_video\videos\material\7602198039888989481\7602198039888989481_static_cut.mp4"
    output_video = video_file.replace(".mp4", "_inpainted_fast.mp4")

    # box结构: [x, y]
    formatted_boxes = [[

            [
                423,
                1749
            ],
            [
                3408,
                1749
            ],
            [
                3408,
                2048
            ],
            [
                423,
                2048
            ]

    ]]

    repair_info = [{'boxs': [[[1093, 14], [1897, 14], [1897, 293], [1093, 293]]], 'end': 1033.3333333333335, 'start': 133.33333333333334}, {'boxs': [[[1035, 0], [1981, 0], [1981, 299], [1035, 299]]], 'end': 1933.3333333333333, 'start': 1033.3333333333335}]

    if os.path.exists(video_file) and global_lama:
        try:
            inpaint_video_intervals(video_file, output_video, repair_info, lama=global_lama)
        except Exception as e:
            import traceback

            traceback.print_exc()
    else:
        print("Skipping test: Input file not found or model not loaded.")
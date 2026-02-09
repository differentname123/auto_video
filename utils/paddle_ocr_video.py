import cv2
import numpy as np
import os
import uuid
import traceback
import time
import shutil

from utils.common_utils import save_json
# 假设这两个函数在 utils.paddle_ocr_fast 中
from utils.paddle_ocr_fast import run_fast_det_rec_ocr, _init_engine


def crop_polygon(img, points):
    """高效裁剪，利用切片特性"""
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    x, y = max(0, x), max(0, y)
    return img[y:y + h, x:x + w]


def is_similar_image(img1, img2, threshold=30):
    """相似度比对 (尺寸缩放 + 灰度差值)"""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 缩放至 64宽，保持比例，大幅加速比对
    h, w = g1.shape
    new_w = 64
    new_h = int(new_w * h / w) if w > 0 else 64

    s1 = cv2.resize(g1, (new_w, new_h))
    s2 = cv2.resize(g2, (new_w, new_h))

    return np.mean(cv2.absdiff(s1, s2)) < threshold


def video_ocr_processor(video_path, ocr_info, similarity_threshold=25):
    """
    高性能视频 OCR 处理器：扫描 -> 批量OCR -> 合并结果
    """
    t_start_all = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理视频: {os.path.basename(video_path)}")

    # 1. 初始化
    global_engine = _init_engine(use_gpu=True)
    video_dir = os.path.dirname(os.path.abspath(video_path))
    # 创建一个专属临时目录，避免文件混乱，处理完后可以直接删目录
    temp_dir = os.path.join(video_dir, f"temp_ocr_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_final_results = []

    try:
        # 按时间段处理
        for seg_idx, info in enumerate(ocr_info):
            boxes = info['boxs']
            start_ms, end_ms = info['start'], info['end']
            start_frame = int((start_ms / 1000) * fps)
            end_frame = int((end_ms / 1000) * fps)

            print(f"\n--- 处理片段 {seg_idx + 1}/{len(ocr_info)} [帧范围: {start_frame}-{end_frame}] ---")

            # --- 阶段 1: 扫描视频，生成任务 ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # timeline_tasks: 记录每一帧的状态
            # 格式: [ { "timestamp": ms, "box_actions": [action_code, value] }, ... ]
            # action_code: 0=OCR(value=img_path), 1=COPY(value=None)
            timeline_tasks = []

            # batch_ocr_tasks: 待OCR的文件列表 [(path, frame_local_idx, box_idx)]
            batch_ocr_paths = []
            path_map_key = {}  # 路径 -> (frame_idx_in_segment, box_idx) 的映射

            # 缓存上一帧的图片，用于对比
            last_crops = [None] * len(boxes)

            # 统计数据
            stats = {"total": 0, "ocr": 0, "skip": 0}

            for curr_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break

                stats["total"] += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                frame_task = {
                    "frame_index": curr_idx,
                    "timestamp": timestamp,
                    "box_actions": []
                }

                for box_i, points in enumerate(boxes):
                    crop = crop_polygon(frame, points)

                    if is_similar_image(crop, last_crops[box_i], threshold=similarity_threshold):
                        # 动作: 复用上一帧
                        frame_task["box_actions"].append((1, None))
                        stats["skip"] += 1
                    else:
                        # 动作: OCR
                        filename = f"f{curr_idx}_b{box_i}_{uuid.uuid4().hex[:6]}.jpg"
                        filepath = os.path.join(temp_dir, filename)

                        # 立即保存图片 (IO操作)
                        cv2.imwrite(filepath, crop)

                        frame_task["box_actions"].append((0, filepath))
                        batch_ocr_paths.append(filepath)

                        # 记录映射关系，方便后续填回
                        # 注意：normpath处理windows路径分隔符问题
                        path_map_key[os.path.normpath(filepath)] = (len(timeline_tasks), box_i)

                        last_crops[box_i] = crop  # 更新对比图
                        stats["ocr"] += 1

                timeline_tasks.append(frame_task)

            print(f"扫描完成。总帧数: {stats['total']} | 需OCR数量: {stats['ocr']} | 复用数量: {stats['skip']}")

            # --- 阶段 2: 统一 OCR ---
            ocr_results_map = {}  # {(frame_local_idx, box_idx): "text"}

            if batch_ocr_paths:
                t_ocr_start = time.time()
                print(f"正在执行批量 OCR ({len(batch_ocr_paths)} 张图片)...")

                try:
                    # 一次性调用，极快
                    ocr_response = run_fast_det_rec_ocr(batch_ocr_paths, engine=global_engine)

                    # 解析结果
                    for item in ocr_response.get('data', []):
                        p = os.path.normpath(item['file'])
                        text = item.get('text', '')
                        if p in path_map_key:
                            key = path_map_key[p]  # key is (frame_local_idx, box_idx)
                            ocr_results_map[key] = text

                    print(f"OCR 耗时: {time.time() - t_ocr_start:.2f}s")
                except Exception as e:
                    print(f"[Error] OCR 批量处理失败: {e}")
                    traceback.print_exc()

            # --- 阶段 3: 合并结果与时间轴回填 ---
            # 维护当前每一个box的最新文本
            current_texts = [""] * len(boxes)

            for f_local_idx, task in enumerate(timeline_tasks):
                final_frame_res = {
                    "frame_index": task["frame_index"],
                    "timestamp": task["timestamp"],
                    "ocr_data": {}
                }

                for box_i, (action_code, val) in enumerate(task["box_actions"]):
                    if action_code == 0:
                        # 是 OCR 任务，从结果 Map 中取值
                        text = ocr_results_map.get((f_local_idx, box_i), "")
                        current_texts[box_i] = text  # 更新记忆

                    # action_code == 1 (复用) 时，直接使用 current_texts 中的旧值

                    final_frame_res["ocr_data"][box_i] = current_texts[box_i]

                total_final_results.append(final_frame_res)

    finally:
        cap.release()
        # 彻底清理临时目录
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"清理临时文件完成。")
            except:
                pass

    t_end_all = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 全部完成。总耗时: {t_end_all - t_start_all:.2f}s")
    return total_final_results


if __name__ == "__main__":
    # 配置区
    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"

    # 示例框
    formatted_boxes = [[
        [389, 914], [1049, 914], [1049, 954], [389, 954]
    ]]

    ocr_info = [
        {"start": 0, "end": 86000, "boxs": formatted_boxes}
    ]

    try:
        results = video_ocr_processor(video_file, ocr_info)
        save_json("video_ocr_results.json", results)
    except Exception as e:
        print(f"主程序运行出错: {e}")
        traceback.print_exc()
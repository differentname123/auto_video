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


def get_image_signature(img, width=48):
    """
    生成图像指纹 (优化版)
    1. 使用 INTER_NEAREST 极速缩放
    2. 直接转灰度
    3. 返回用于比对的小型矩阵
    """
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    # 保持比例计算新高度，防止变形导致的误判
    new_h = int(width * h / w) if w > 0 else width

    # 性能优化点：使用 INTER_NEAREST (最近邻插值) 替代默认的线性插值，速度快 3-5 倍
    small = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray


def video_ocr_processor(video_path, ocr_info, similarity_threshold=10):
    """
    高性能视频 OCR 处理器：扫描 -> 批量OCR -> 合并结果
    优化：数学计算时间戳 + 图像指纹缓存机制
    """
    t_start_all = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理视频: {os.path.basename(video_path)}")

    # 1. 初始化
    global_engine = _init_engine(use_gpu=True)
    video_dir = os.path.dirname(os.path.abspath(video_path))
    temp_dir = os.path.join(video_dir, f"temp_ocr")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0  # 防止FPS获取失败导致除零错误

    total_final_results = []

    try:
        # 按时间段处理
        for seg_idx, info in enumerate(ocr_info):
            boxes = info['boxs']
            start_ms, end_ms = info['start'], info['end']

            # 严格计算帧范围
            start_frame = int(round((start_ms / 1000) * fps))
            end_frame = int(round((end_ms / 1000) * fps))

            print(f"\n--- 处理片段 {seg_idx + 1}/{len(ocr_info)} [帧范围: {start_frame}-{end_frame}] ---")

            # --- 阶段 1: 扫描视频，生成任务 ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            timeline_tasks = []
            batch_ocr_paths = []
            path_map_key = {}

            # 【核心优化】缓存上一帧的“指纹”（也就是缩放后的灰度图），而不是原图
            # 避免了每次比对时都要对上一帧重新 resize 和 cvtColor
            last_signatures = [None] * len(boxes)

            stats = {"total": 0, "ocr": 0, "skip": 0}
            start_time = time.time()

            # 使用循环计数器确保帧号准确
            for curr_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break

                stats["total"] += 1

                # 【修复问题1】使用数学公式计算精确时间戳，不再依赖 cap.get(POS_MSEC)
                timestamp = (curr_idx / fps) * 1000.0

                frame_task = {
                    "frame_index": curr_idx,
                    "timestamp": timestamp,
                    "box_actions": []
                }

                for box_i, points in enumerate(boxes):
                    crop = crop_polygon(frame, points)

                    # 【修复问题2】获取当前帧的指纹
                    curr_sig = get_image_signature(crop)
                    last_sig = last_signatures[box_i]

                    is_duplicate = False
                    # 快速比对逻辑
                    if last_sig is not None and curr_sig.shape == last_sig.shape:
                        # 此时已经是两个很小的灰度矩阵，absdiff 和 mean 极快
                        diff_score = np.mean(cv2.absdiff(curr_sig, last_sig))
                        if diff_score < similarity_threshold:
                            is_duplicate = True

                    if is_duplicate:
                        # 动作: 复用上一帧
                        frame_task["box_actions"].append((1, None))
                        stats["skip"] += 1
                    else:
                        # 动作: OCR

                        filename = f"frame{curr_idx}_box{box_i}.jpg"
                        filepath = os.path.join(temp_dir, filename)

                        # 只有判定不同时才写入磁盘
                        cv2.imwrite(filepath, crop)

                        frame_task["box_actions"].append((0, filepath))
                        batch_ocr_paths.append(filepath)

                        path_map_key[os.path.normpath(filepath)] = (len(timeline_tasks), box_i)

                        # 更新指纹缓存
                        last_signatures[box_i] = curr_sig
                        stats["ocr"] += 1

                timeline_tasks.append(frame_task)

            scan_cost = time.time() - start_time
            print(
                f"扫描完成。总帧数: {stats['total']} | 需OCR数量: {stats['ocr']} | 复用数量: {stats['skip']} | 耗时: {scan_cost:.2f}s (FPS: {stats['total'] / scan_cost:.1f})")

            # --- 阶段 2: 统一 OCR ---
            ocr_results_map = {}

            if batch_ocr_paths:
                t_ocr_start = time.time()
                try:
                    ocr_response = run_fast_det_rec_ocr(batch_ocr_paths, engine=global_engine)

                    for item in ocr_response.get('data', []):
                        p = os.path.normpath(item['file'])
                        text = item.get('text', '')
                        if p in path_map_key:
                            key = path_map_key[p]
                            ocr_results_map[key] = item

                    print(f"执行批量 OCR ({len(batch_ocr_paths)} 张图片)... OCR 耗时: {time.time() - t_ocr_start:.2f}s")
                except Exception as e:
                    print(f"[Error] OCR 批量处理失败: {e}")
                    traceback.print_exc()

            # --- 阶段 3: 合并结果与时间轴回填 (修复后) ---

            # [Fix] 初始化缓存字典，用于在跳过帧时存储和回填上一次的有效结果
            # 键是 box_i (框的索引)，值是该框最后一次 OCR 的结果数据
            last_valid_box_results = {}

            for f_local_idx, task in enumerate(timeline_tasks):
                final_frame_res = {
                    "frame_index": task["frame_index"],
                    "timestamp": task["timestamp"],
                    "ocr_data": {}
                }

                for box_i, (action_code, val) in enumerate(task["box_actions"]):
                    current_box_result = None

                    if action_code == 0:
                        # 0 代表执行了 OCR，从结果映射中获取
                        item = ocr_results_map.get((f_local_idx, box_i), {})
                        text = item.get('text', '')
                        box = item.get('box', [])

                        current_box_result = {
                            "text": text,
                            "box": box
                        }
                        # [Fix] 更新缓存，供下一帧复用
                        last_valid_box_results[box_i] = current_box_result

                    elif action_code == 1:
                        # 1 代表跳过（内容重复），应该复用上一帧的结果
                        # [Fix] 从缓存中读取
                        if box_i in last_valid_box_results:
                            current_box_result = last_valid_box_results[box_i]
                        else:
                            # 理论上不会发生，除非第一帧就是重复帧（逻辑上第一帧 last_sig 为 None，强制 OCR）
                            current_box_result = {"text": "", "box": []}

                    # 将结果填入当前帧数据
                    if current_box_result:
                        final_frame_res["ocr_data"][box_i] = current_box_result

                total_final_results.append(final_frame_res)

    finally:
        cap.release()
        if os.path.exists(temp_dir):
            try:
                pass
                # shutil.rmtree(temp_dir)
            except:
                pass

    t_end_all = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 全部完成。总耗时: {t_end_all - t_start_all:.2f}s")
    return total_final_results


if __name__ == "__main__":
    # 配置区
    video_file = r"W:\project\python_project\auto_video\videos\material\7602198039888989481\7602198039888989481_static_cut.mp4"

    # 示例框
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

    ocr_info = [
        {"start": 0, "end": 5000, "boxs": formatted_boxes}
    ]

    try:
        results = video_ocr_processor(video_file, ocr_info)
        # 简单打印前5个结果验证时间戳
        for res in results[:5]:
            print(f"Frame: {res['frame_index']}, Time: {res['timestamp']:.2f}ms")

        save_json("video_ocr_results.json", results)
    except Exception as e:
        print(f"主程序运行出错: {e}")
        traceback.print_exc()
import cv2
import numpy as np
import os
import uuid
import traceback
import time
import shutil

from utils.common_utils import save_json, read_json
from utils.inpating.lama import inpaint_video_intervals
from utils.paddle_ocr import adjust_subtitle_box
# 假设这两个函数在 utils.paddle_ocr_fast 中
from utils.paddle_ocr_fast import run_fast_det_rec_ocr, _init_engine
import json
import copy
from difflib import SequenceMatcher


def calculate_iou(box1, box2):
    """简单的辅助函数，用于计算两个框的重叠程度（可选，此处主要靠文字和时间修复）"""
    # 这里主要依赖文字相似度，暂不需要复杂的IoU计算，
    # 但保留此接口以便未来扩展空间位置校验。
    return 0


def get_text_similarity(str1, str2):
    """计算两个字符串的相似度"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def get_union_box(boxes):
    """计算一组框的最小包围框 (Union Box)"""
    if not boxes:
        return []

    # 展平所有坐标点
    all_x = []
    all_y = []
    for box in boxes:
        # box 格式通常为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        if not box: continue
        for point in box:
            all_x.append(point[0])
            all_y.append(point[1])

    if not all_x or not all_y:
        return []

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # 返回标准的4点矩形框
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]


def optimize_ocr_results(raw_data):
    # 1. 初始化：计算平均帧间隔，用于推算结束时间
    if len(raw_data) > 1:
        frame_duration = raw_data[1]['timestamp'] - raw_data[0]['timestamp']
    else:
        frame_duration = 33.33  # 默认 30fps

    # 提取所有出现的 key
    all_keys = set()
    for frame in raw_data:
        if 'ocr_data' in frame:
            for k in frame['ocr_data']:
                all_keys.add(k)

    # 将数据转为按 Key 索引的轨迹列表: tracks[key] = [ {timestamp, text, box}, ... ]
    # 这一步是为了方便对单个物体进行时序修复
    tracks = {k: [] for k in all_keys}

    # 填充轨迹，保持帧的顺序
    for i, frame in enumerate(raw_data):
        ts = frame['timestamp']
        ocr_data = frame.get('ocr_data', {})

        for k in all_keys:
            if k in ocr_data:
                tracks[k].append({
                    'index': i,
                    'timestamp': ts,
                    'text': ocr_data[k].get('text', ''),
                    'box': ocr_data[k].get('box', [])
                })
            else:
                # 这一帧没有这个key，填空
                tracks[k].append({
                    'index': i,
                    'timestamp': ts,
                    'text': '',
                    'box': []
                })

    # 2. 修复逻辑 (Repair)
    # 针对每个key的轨迹进行双向修复
    for k in tracks:
        track = tracks[k]
        n = len(track)

        # 多次遍历以处理连续的错误
        # 逻辑：如果当前帧是空的或者包含了子串，但前后有相同的“强”文字，则修复

        # 简单窗口修复（窗口大小可以根据实际情况调整，这里设为前后各看2帧）
        # 这里使用一个简单的逻辑：找到连续的非空文本块，然后填补中间的空隙

        # Pass 1: 填补短小的空隙 (Gap Filling) 和 修复相似文本
        # 比如: "A", "", "A" -> "A", "A", "A"
        # 比如: "ABC", "AB", "ABC" -> "ABC", "ABC", "ABC"

        for i in range(1, n - 1):
            prev_frame = track[i - 1]
            curr_frame = track[i]
            next_frame = track[i + 1]

            # 如果前后文字一致且非空
            if prev_frame['text'] and prev_frame['text'] == next_frame['text']:
                # 如果当前为空，或者当前文字是前后文字的子串/相似度高
                is_weak = (not curr_frame['text']) or \
                          (len(curr_frame['text']) < len(prev_frame['text']) and curr_frame['text'] in prev_frame[
                              'text']) or \
                          (get_text_similarity(curr_frame['text'], prev_frame['text']) > 0.6)

                if is_weak and curr_frame['text'] != prev_frame['text']:
                    curr_frame['text'] = prev_frame['text']
                    # 框也需要修复，暂时复用上一帧的框，稍后会重新计算聚合框
                    if not curr_frame['box']:
                        curr_frame['box'] = prev_frame['box']

        # Pass 2: 处理稍大一点的空隙 (比如中间空了2帧)
        # 这里的逻辑是寻找每一个非空文本的“段”，如果两个段文本相同且距离很近，合并它们
        # 简单起见，再进行一次更激进的扫描
        for i in range(1, n - 2):
            # 检查 pattern: Text, Empty, Empty, Text
            if track[i]['text'] == '' and track[i + 1]['text'] == '':
                prev_f = track[i - 1]
                next_f = track[i + 2]
                if prev_f['text'] and prev_f['text'] == next_f['text']:
                    track[i]['text'] = prev_f['text']
                    track[i]['box'] = prev_f['box']
                    track[i + 1]['text'] = prev_f['text']
                    track[i + 1]['box'] = prev_f['box']

    # 3. 生成独立的事件段 (Generate Segments) 并计算包围框
    # 格式: { 'key': '0', 'text': 'xxx', 'start': 0.0, 'end': 100.0, 'box': [...] }
    all_segments = []

    for k in tracks:
        track = tracks[k]
        if not track: continue

        current_segment = None

        for i, frame in enumerate(track):
            txt = frame['text']
            box = frame['box']
            ts = frame['timestamp']

            if not txt:
                # 遇到空文本，结束当前段
                if current_segment:
                    # 计算该段的 Union Box
                    current_segment['box'] = get_union_box(current_segment['raw_boxes'])
                    # 移除临时数据
                    del current_segment['raw_boxes']
                    # 设定结束时间 (当前帧的时间作为上一段的结束点，或者当前帧时间+duration)
                    # 通常认为上一帧结束时刻是当前帧开始时刻
                    current_segment['end'] = ts
                    all_segments.append(current_segment)
                    current_segment = None
            else:
                # 有文本
                if current_segment is None:
                    # 开启新段
                    current_segment = {
                        'key': k,
                        'text': txt,
                        'start': ts,
                        'end': ts + frame_duration,  # 暂定
                        'raw_boxes': [box] if box else []
                    }
                else:
                    # 检查文本是否变化
                    if txt == current_segment['text']:
                        # 文本相同，延续段
                        if box:
                            current_segment['raw_boxes'].append(box)
                        current_segment['end'] = ts + frame_duration
                    else:
                        # 文本变了，结束旧段，开启新段
                        current_segment['box'] = get_union_box(current_segment['raw_boxes'])
                        del current_segment['raw_boxes']
                        # 旧段结束时间是当前帧开始时间
                        current_segment['end'] = ts
                        all_segments.append(current_segment)

                        current_segment = {
                            'key': k,
                            'text': txt,
                            'start': ts,
                            'end': ts + frame_duration,
                            'raw_boxes': [box] if box else []
                        }

        # 循环结束处理最后的段
        if current_segment:
            current_segment['box'] = get_union_box(current_segment['raw_boxes'])
            del current_segment['raw_boxes']
            all_segments.append(current_segment)

    # 4. 时序切片 (Time Slicing)
    # 我们需要把不同key的重叠时间段压扁成无重叠的线性时间轴

    # 收集所有关键时间点（切点）
    cut_points = set()
    for seg in all_segments:
        cut_points.add(seg['start'])
        cut_points.add(seg['end'])

    sorted_points = sorted(list(cut_points))

    final_output = []

    # 遍历每一个微小的时间区间
    for i in range(len(sorted_points) - 1):
        t_start = sorted_points[i]
        t_end = sorted_points[i + 1]
        mid_time = (t_start + t_end) / 2  # 取中点判断区间归属

        # 找出这个区间内活跃的所有 segments
        active_segments = []
        for seg in all_segments:
            # 判断 segment 是否覆盖这个区间
            # 使用稍微宽松的判定防止浮点数精度问题: start <= mid < end
            if seg['start'] <= mid_time and seg['end'] > mid_time:
                active_segments.append(seg)

        if not active_segments:
            continue

        # 构建当前区间的 ocr_data
        current_ocr_data = {}
        for seg in active_segments:
            current_ocr_data[seg['key']] = {
                'text': seg['text'],
                'box': seg['box']
            }

        # 尝试合并：如果当前生成的 ocr_data 与 result 中最后一个条目的 ocr_data 完全一致，则只延长结束时间
        if final_output and final_output[-1]['ocr_data'] == current_ocr_data:
            final_output[-1]['end'] = t_end
        else:
            final_output.append({
                'start': t_start,
                'end': t_end,
                'ocr_data': current_ocr_data
            })

    return final_output


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


def video_ocr_processor(video_path, ocr_info, similarity_threshold=20):
    """
    高性能视频 OCR 处理器：扫描 -> 批量OCR -> 合并结果
    优化：数学计算时间戳 + 图像指纹缓存机制 + 坐标系还原
    """
    t_start_all = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理视频: {os.path.basename(video_path)}")

    # 1. 初始化
    global_engine = _init_engine(use_gpu=True)
    video_dir = os.path.dirname(os.path.abspath(video_path))
    temp_dir = os.path.join(video_dir, f"temp_ocr")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

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

            # 【新增逻辑 1】预先计算每个裁剪框在原图中的偏移量 (x, y)
            # 这样后续合并 OCR 结果时，可以将局部坐标还原为全局坐标
            box_offsets = []
            for b_points in boxes:
                # 必须与 crop_polygon 中的逻辑一致
                bx, by, bw, bh = cv2.boundingRect(np.array(b_points, dtype=np.int32))
                box_offsets.append((max(0, bx), max(0, by)))

            # 严格计算帧范围
            start_frame = int(round((start_ms / 1000) * fps))
            end_frame = int(round((end_ms / 1000) * fps))

            print(f"\n--- 处理片段 {seg_idx + 1}/{len(ocr_info)} [帧范围: {start_frame}-{end_frame}] ---")

            # --- 阶段 1: 扫描视频，生成任务 ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            timeline_tasks = []
            batch_ocr_paths = []
            path_map_key = {}

            # 缓存上一帧的“指纹”
            last_signatures = [None] * len(boxes)

            stats = {"total": 0, "ocr": 0, "skip": 0}
            start_time = time.time()

            # 使用循环计数器确保帧号准确
            for curr_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break

                stats["total"] += 1

                # 计算精确时间戳
                timestamp = (curr_idx / fps) * 1000.0

                frame_task = {
                    "frame_index": curr_idx,
                    "timestamp": timestamp,
                    "box_actions": []
                }

                for box_i, points in enumerate(boxes):
                    crop = crop_polygon(frame, points)

                    # 获取当前帧的指纹
                    curr_sig = get_image_signature(crop)
                    last_sig = last_signatures[box_i]

                    is_duplicate = False
                    # 快速比对逻辑
                    if last_sig is not None and curr_sig.shape == last_sig.shape:
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
                        # item['box'] 这里是相对于小图的坐标
                        if p in path_map_key:
                            key = path_map_key[p]
                            ocr_results_map[key] = item

                    print(f"执行批量 OCR ({len(batch_ocr_paths)} 张图片)... OCR 耗时: {time.time() - t_ocr_start:.2f}s")
                except Exception as e:
                    print(f"[Error] OCR 批量处理失败: {e}")
                    traceback.print_exc()

            # --- 阶段 3: 合并结果与时间轴回填 ---

            # 初始化缓存字典，用于在跳过帧时存储和回填上一次的有效结果
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
                        raw_box = item.get('box', [])

                        # 【新增逻辑 2】将局部坐标转换为全局坐标
                        global_box = []
                        if raw_box:
                            off_x, off_y = box_offsets[box_i]  # 获取该框的左上角偏移
                            for point in raw_box:
                                global_box.append([
                                    point[0] + off_x,
                                    point[1] + off_y
                                ])

                        current_box_result = {
                            "text": text,
                            "box": global_box
                        }

                        # 更新缓存，供下一帧复用
                        last_valid_box_results[box_i] = current_box_result

                    elif action_code == 1:
                        # 1 代表跳过（内容重复），应该复用上一帧的结果
                        if box_i in last_valid_box_results:
                            current_box_result = last_valid_box_results[box_i]
                        else:
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


def transform_ocr_to_repair_info(raw_data):
    """
    将原始 OCR 数据列表转换为 repair_info 格式
    """
    repair_info = []

    for item in raw_data:
        # 提取时间
        start_time = item.get('start')
        end_time = item.get('end')

        # 提取并格式化该时间段内的所有 box
        formatted_boxes = []
        ocr_data = item.get('ocr_data', {})

        # ocr_data 是一个字典 {'0': {...}, '1': {...}}，我们需要遍历它的值
        for key in ocr_data:
            obj = ocr_data[key]
            if 'box' in obj:
                raw_box = obj['box']
                # 原始数据是浮点数，目标格式通常是整数坐标
                # 这里将坐标四舍五入转为 int
                cleaned_box = [
                    [int(round(point[0])), int(round(point[1]))]
                    for point in raw_box
                ]
                formatted_boxes.append(cleaned_box)

        # 构造单个结果条目
        entry = {
            "start": start_time,
            "end": end_time,
            "boxs": formatted_boxes  # 注意目标格式使用的是 "boxs"
        }
        repair_info.append(entry)

    return repair_info

def inpaint_subtitle_box():
    """
    直接修复精确的字幕区域
    :return:
    """
    results = read_json("video_ocr_results.json")
    format_result = optimize_ocr_results(results)
    print()
    video_file = r"W:\project\python_project\auto_video\videos\material\7602198039888989481\7602198039888989481_static_cut.mp4"
    output_video = video_file.replace(".mp4", "_inpainted_fast.mp4")
    repair_info = transform_ocr_to_repair_info(format_result)
    repair_info = repair_info[:2]
    inpaint_video_intervals(video_file, output_video, repair_info)


if __name__ == "__main__":
    # inpaint_subtitle_box()



    # 配置区
    video_file = r"W:\project\python_project\auto_video\videos\material\7602198039888989481\7602198039888989481_static_cut.mp4"

    # 示例框

    raw_box = [
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
    ]


    top_left, bottom_right, _, _ = adjust_subtitle_box(video_file, raw_box, 0)

    # 根据top_left, bottom_right得到调整后的四个坐标点
    adjust_box = [
        [top_left[0], top_left[1]],
        [bottom_right[0], top_left[1]],
        [bottom_right[0], bottom_right[1]],
        [top_left[0], bottom_right[1]]
    ]

    formatted_boxes = [adjust_box]


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
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

    # 将数据转为按 Key 索引的轨迹列表
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
                tracks[k].append({
                    'index': i,
                    'timestamp': ts,
                    'text': '',
                    'box': []
                })

    # 2. 修复逻辑 (Repair) - 原始逻辑保留，处理中间的抖动
    for k in tracks:
        track = tracks[k]
        n = len(track)

        # Pass 1: 填补短小的空隙 (Gap Filling) 和 修复相似文本
        for i in range(1, n - 1):
            prev_frame = track[i - 1]
            curr_frame = track[i]
            next_frame = track[i + 1]

            if prev_frame['text'] and prev_frame['text'] == next_frame['text']:
                is_weak = (not curr_frame['text']) or \
                          (len(curr_frame['text']) < len(prev_frame['text']) and curr_frame['text'] in prev_frame[
                              'text']) or \
                          (get_text_similarity(curr_frame['text'], prev_frame['text']) > 0.6)

                if is_weak and curr_frame['text'] != prev_frame['text']:
                    curr_frame['text'] = prev_frame['text']
                    if not curr_frame['box']:
                        curr_frame['box'] = prev_frame['box']

        # Pass 2: 处理稍大一点的空隙
        for i in range(1, n - 2):
            if track[i]['text'] == '' and track[i + 1]['text'] == '':
                prev_f = track[i - 1]
                next_f = track[i + 2]
                if prev_f['text'] and prev_f['text'] == next_f['text']:
                    track[i]['text'] = prev_f['text']
                    track[i]['box'] = prev_f['box']
                    track[i + 1]['text'] = prev_f['text']
                    track[i + 1]['box'] = prev_f['box']

    # --- 修正后的辅助函数：兼容 4点坐标格式 [[x,y],...] ---
    def _filter_boxes_by_ratio(boxes, text):
        """
        过滤掉那些长宽比与文字长度严重不符的异常框。
        """
        if not boxes or len(boxes) < 2 or not text:
            return boxes

        try:
            # 1. 预计算每个框的几何信息 (宽、高、面积)
            box_stats = []
            valid_areas = []

            for b in boxes:
                w, h = 0, 0
                # 判断格式: 如果第一个元素是列表，说明是 [[x,y], [x,y]...] 格式
                if len(b) > 0 and isinstance(b[0], list):
                    xs = [p[0] for p in b]
                    ys = [p[1] for p in b]
                    if xs and ys:
                        w = max(xs) - min(xs)
                        h = max(ys) - min(ys)
                # 否则假设是 [x, y, w, h]
                elif len(b) >= 4:
                    w = b[2]
                    h = b[3]

                area = w * h
                box_stats.append({'box': b, 'w': w, 'h': h, 'area': area})
                if area > 0:
                    valid_areas.append(area)

            if not valid_areas:
                return boxes

            # 2. 检查面积差异是否显著
            min_area, max_area = min(valid_areas), max(valid_areas)

            # 差异小于20%则视为稳定，不处理
            if max_area == 0 or (max_area - min_area) / max_area <= 0.2:
                return boxes

            # 3. 启用筛选逻辑
            target_ratio = float(len(text))

            scored_boxes = []
            for item in box_stats:
                w, h = item['w'], item['h']
                if h <= 1:
                    ratio = 0
                else:
                    ratio = w / h

                diff = abs(ratio - target_ratio)
                item['diff'] = diff
                scored_boxes.append(item)

            # 4. 找出最接近理论值的“最佳差异”
            scored_boxes.sort(key=lambda x: x['diff'])
            best_diff = scored_boxes[0]['diff']

            # 5. 筛选：只保留差异值在容忍范围内的框
            # 阈值：best_diff + 1.0 (例如最佳是diff 0.5，那么保留diff 1.5以内的)
            filtered = [item['box'] for item in scored_boxes if item['diff'] <= best_diff + 1.0]

            return filtered if filtered else boxes

        except Exception:
            # 万一出错，返回原数据防止崩溃
            return boxes

    # 3. 生成独立的事件段 (Generate Segments) 并计算包围框
    all_segments_list = []  # 改名，避免歧义，这是一个扁平列表

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
                    valid_boxes = _filter_boxes_by_ratio(current_segment['raw_boxes'], current_segment['text'])
                    current_segment['box'] = get_union_box(valid_boxes)

                    del current_segment['raw_boxes']
                    current_segment['end'] = ts
                    all_segments_list.append(current_segment)
                    current_segment = None
            else:
                # 有文本
                if current_segment is None:
                    current_segment = {
                        'key': k,
                        'text': txt,
                        'start': ts,
                        'end': ts + frame_duration,
                        'raw_boxes': [box] if box else []
                    }
                else:
                    if txt == current_segment['text']:
                        # 文本相同，延续段
                        if box:
                            current_segment['raw_boxes'].append(box)
                        current_segment['end'] = ts + frame_duration
                    else:
                        # 文本变了，结束旧段
                        valid_boxes = _filter_boxes_by_ratio(current_segment['raw_boxes'], current_segment['text'])
                        current_segment['box'] = get_union_box(valid_boxes)

                        del current_segment['raw_boxes']
                        current_segment['end'] = ts
                        all_segments_list.append(current_segment)

                        current_segment = {
                            'key': k,
                            'text': txt,
                            'start': ts,
                            'end': ts + frame_duration,
                            'raw_boxes': [box] if box else []
                        }

        # 循环结束处理最后的段
        if current_segment:
            valid_boxes = _filter_boxes_by_ratio(current_segment['raw_boxes'], current_segment['text'])
            current_segment['box'] = get_union_box(valid_boxes)
            del current_segment['raw_boxes']
            all_segments_list.append(current_segment)

    # =========================================================================
    # [新增] 3.5. 深度合并碎片段 (Fix Start-up Noise & Fragmented Segments)
    # 解决 "TER..." 和 "..." 被切分为两段的问题
    # =========================================================================

    # 1. 先按 Key 分组
    segments_by_key = {}
    for seg in all_segments_list:
        k = seg['key']
        if k not in segments_by_key:
            segments_by_key[k] = []
        segments_by_key[k].append(seg)

    merged_segments = []

    for k in segments_by_key:
        segs = sorted(segments_by_key[k], key=lambda x: x['start'])
        if not segs: continue

        # 迭代合并
        curr = segs[0]
        for i in range(1, len(segs)):
            next_seg = segs[i]

            # 判断时间连续性 (允许 1-2 帧的微小误差，例如 100ms)
            time_gap = next_seg['start'] - curr['end']
            is_continuous = time_gap < 100.0

            if is_continuous:
                txt1 = curr['text']
                txt2 = next_seg['text']

                # 判断文本相似性或包含关系
                # 场景: "TER就把..." vs "就把..."
                similarity = get_text_similarity(txt1, txt2)
                is_subset = (txt1 in txt2) or (txt2 in txt1)

                # 如果连续且相似
                if similarity > 0.6 or is_subset:
                    # 决定保留哪个文本：保留持续时间更长的那个（通常是正确的）
                    dur1 = curr['end'] - curr['start']
                    dur2 = next_seg['end'] - next_seg['start']

                    if dur2 > dur1:
                        # 后面的段更稳定，curr 变为 next 的一部分
                        curr['text'] = next_seg['text']
                        curr['end'] = next_seg['end']
                        # 合并Box
                        curr['box'] = get_union_box([curr['box'], next_seg['box']])
                    else:
                        # 前面的段更稳定，next 变为 curr 的一部分
                        curr['end'] = next_seg['end']
                        curr['box'] = get_union_box([curr['box'], next_seg['box']])

                    # 继续下一轮，curr 保持不变（已经吸收了next）
                    continue

            # 无法合并，保存 curr，切换 next 为 curr
            merged_segments.append(curr)
            curr = next_seg

        merged_segments.append(curr)

    all_segments = merged_segments  # 替换原来的列表
    # =========================================================================

    # 4. 时序切片 (Time Slicing)
    cut_points = set()
    for seg in all_segments:
        cut_points.add(seg['start'])
        cut_points.add(seg['end'])

    sorted_points = sorted(list(cut_points))
    final_output = []

    for i in range(len(sorted_points) - 1):
        t_start = sorted_points[i]
        t_end = sorted_points[i + 1]
        mid_time = (t_start + t_end) / 2

        # 过滤极短的时间片 (可选，防止浮点数误差产生极细切片)
        if t_end - t_start < 1.0:
            continue

        active_segments = []
        for seg in all_segments:
            if seg['start'] <= mid_time and seg['end'] > mid_time:
                active_segments.append(seg)

        if not active_segments:
            continue

        current_ocr_data = {}
        for seg in active_segments:
            current_ocr_data[seg['key']] = {
                'text': seg['text'],
                'box': seg['box']
            }

        # 合并相邻且内容完全一致的切片结果
        if final_output and final_output[-1]['ocr_data'] == current_ocr_data:
            final_output[-1]['end'] = t_end
        else:
            final_output.append({
                'ocr_data': current_ocr_data,
                'start': int(t_start),
                'end': int(t_end),
            })

    return final_output


def crop_polygon(img, points):
    """高效裁剪，利用切片特性"""
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    x, y = max(0, x), max(0, y)
    return img[y:y + h, x:x + w]


def get_image_signature(img, width=48):
    """
    生成图像指纹 (优化版 - 聚焦中心区域)
    1. 裁切掉上下左右的边缘背景，只保留中间更有可能包含文字的区域进行比对。
       这能显著提高由于字幕框过大导致的微小文字变化识别率。
    2. 使用 INTER_NEAREST 极速缩放
    3. 直接转灰度
    """
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    # --- [新增逻辑] 聚焦中间区域 (Center Crop) ---
    # 只有当图片尺寸足够时才裁剪，防止极小图出错
    if h > 8 and w > 8:
        # 垂直方向：保留中间 50% (去掉顶部 25% 和底部 25%)
        # 字幕通常在框的垂直居中位置，上下边缘多为背景噪音
        y_start = int(h * 0.25)
        y_end = int(h * 0.75)

        # 水平方向：保留中间 70% (去掉左边 15% 和右边 15%)
        # 稍微保留宽一点，防止长字幕首尾差异被忽略，但去掉最边缘的背景
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)

        # 再次校验防止切空（对于极扁或极细的图）
        if y_end > y_start and x_end > x_start:
            img = img[y_start:y_end, x_start:x_end]

    # 获取裁切后的新尺寸
    h, w = img.shape[:2]

    # 保持比例计算新高度，防止变形导致的误判
    # 注意：虽然裁切改变了原图视野，但只要前后帧都按同样比例裁切，比对依然有效
    new_h = int(width * h / w) if w > 0 else width

    # 性能优化点：使用 INTER_NEAREST (最近邻插值) 替代默认的线性插值，速度快 3-5 倍
    small = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray


def video_ocr_processor(video_path, ocr_info, similarity_threshold=25):
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
    # repair_info = repair_info[:2]
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


    top_left, bottom_right, _, _ = adjust_subtitle_box(video_file, raw_box, 0.1)

    # 根据top_left, bottom_right得到调整后的四个坐标点
    adjust_box = [
        [top_left[0], top_left[1]],
        [bottom_right[0], top_left[1]],
        [bottom_right[0], bottom_right[1]],
        [top_left[0], bottom_right[1]]
    ]

    formatted_boxes = [adjust_box]


    ocr_info = [
        {"start": 00, "end": 450000, "boxs": formatted_boxes}
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
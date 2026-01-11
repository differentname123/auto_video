# ocr_engine.py
# -- coding: utf-8 --
import gc
import os
import sys
import threading
import time
import traceback
from typing import List

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from PIL import Image

from utils.common_utils import time_to_ms
from utils.paddle_ocr_base import run_subtitle_ocr
from utils.video_utils import get_video_dimensions


def analyze_and_filter_boxes(
        boxes: List[List[List[int]]],
        height_tolerance_ratio: float = 0.3,
        y_pos_tolerance_ratio: float = 0.25,
        z_score_threshold: float = 2.0
) -> List[List[List[int]]]:
    """
    【三阶段优化版】分析并过滤字幕框，增加下底位置筛选。

    Args:
        boxes (List[List[List[int]]]): 所有检测到的字幕框。
        height_tolerance_ratio (float): 第一阶段过滤中，框高与中位数的最大允许偏差比例。
        y_pos_tolerance_ratio (float): 第二阶段过滤中，框下底与中位下底的最大允许偏差比例
                                       (以中位数高度为基准)。
        z_score_threshold (float): 第三阶段过滤中，用于判断是否为异常值的Z分数阈值。

    Returns:
        List[List[List[int]]]: 过滤后的高质量字幕框列表。
    """
    if len(boxes) < 5:  # 数据太少，统计无意义，直接返回
        print("[过滤] 检测到的框数量过少，跳过复杂过滤。")
        return boxes

    # --- 阶段 1: 基于高度中位数的粗筛 ---
    heights = []
    for box in boxes:
        y_coords = [p[1] for p in box]
        heights.append(max(y_coords) - min(y_coords))

    median_height = np.median(heights)
    if median_height == 0: return []  # 避免后续除零错误

    height_diff_threshold = median_height * height_tolerance_ratio

    height_filtered_boxes = []
    # print(f"[过滤-阶段1: 高度] 以中位高度 {median_height:.2f} 为基准 (容忍度: {height_diff_threshold:.2f}px)。")
    for i, box in enumerate(boxes):
        if abs(heights[i] - median_height) <= height_diff_threshold:
            height_filtered_boxes.append(box)
        else:
            # print(f"  - 剔除高度异常框: 高度为 {heights[i]}, 与中位数差异过大。")
            pass
    # print(f"[过滤-阶段1: 高度] 完成，剩余 {len(height_filtered_boxes)} 个框进入下一阶段。")
    if len(height_filtered_boxes) < 3:
        return height_filtered_boxes

    # --- 阶段 2: 基于下底位置的中筛 (新增筛选逻辑) ---
    bottom_ys = [max(p[1] for p in box) for box in height_filtered_boxes]
    median_bottom_y = np.median(bottom_ys)

    # 位置容忍度以中位高度为基准，更具自适应性
    y_pos_diff_threshold = median_height * y_pos_tolerance_ratio

    position_filtered_boxes = []
    # print(f"[过滤-阶段2: 位置] 以中位下底 {median_bottom_y:.2f} 为基准 (容忍度: {y_pos_diff_threshold:.2f}px)。")
    for i, box in enumerate(height_filtered_boxes):
        if abs(bottom_ys[i] - median_bottom_y) <= y_pos_diff_threshold:
            position_filtered_boxes.append(box)
        else:
            # print(f"  - 剔除位置异常框: 下底为 {bottom_ys[i]} (期望: {median_bottom_y:.2f})，差异过大。")
            pass

    # print(f"[过滤-阶段2: 位置] 完成，剩余 {len(position_filtered_boxes)} 个框进入精筛。")
    if len(position_filtered_boxes) < 3:
        return position_filtered_boxes

    # --- 阶段 3: 对精筛后的框进行Z-score统计分析 ---
    properties = []
    for box in position_filtered_boxes:
        y_coords = [p[1] for p in box]
        min_y, max_y = min(y_coords), max(y_coords)
        properties.append({'height': max_y - min_y, 'center_y': min_y + (max_y - min_y) / 2})

    clean_heights = np.array([p['height'] for p in properties])
    clean_center_ys = np.array([p['center_y'] for p in properties])

    mean_height, std_height = np.mean(clean_heights), np.std(clean_heights)
    mean_center_y, std_center_y = np.mean(clean_center_ys), np.std(clean_center_ys)

    final_good_boxes = []
    # print(f"[过滤-阶段3: 精筛] 以稳定数据为基准进行Z-score分析。")
    for i, box in enumerate(position_filtered_boxes):
        prop = properties[i]
        # 计算 Z-score, 避免标准差为0时除零
        height_z = abs(prop['height'] - mean_height) / std_height if std_height > 0 else 0
        center_y_z = abs(prop['center_y'] - mean_center_y) / std_center_y if std_center_y > 0 else 0

        if height_z < z_score_threshold and center_y_z < z_score_threshold:
            final_good_boxes.append(box)
        else:
            # print(f"  - 剔除Z-score异常框 (高度Z: {height_z:.2f}, 中心Y Z: {center_y_z:.2f})")
            pass
    print(f"[过滤] 所有流程完成，最终保留 {len(final_good_boxes)} 个高质量字幕框。")
    return final_good_boxes


def draw_box_on_images(image_paths: List[str], box_coords: List[List[int]]):
    """
    将一个指定的包围框绘制到一组图片上并覆盖保存。
    """
    # 从[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]中获取左上角和右下角点
    pt1 = tuple(box_coords[0])  # (min_x, min_y)
    pt2 = tuple(box_coords[2])  # (max_x, max_y)
    color = (0, 255, 0)  # 绿色 (BGR)
    thickness = 2

    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            cv2.rectangle(img, pt1, pt2, color, thickness)
            cv2.imwrite(path, img)  # 覆盖保存
        except Exception as e:
            print(f"绘制 '{os.path.basename(path)}' 时出错: {e}")


def adjust_subtitle_box(video_path, final_box):
    """
    将字幕框左右边距至少设为视频宽度的 10%，
    但如果原框更宽就不再缩窄它。

    参数:
        video_path: 视频文件路径
        final_box: 原始字幕框，格式 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    返回:
        (top_left, bottom_right)：调整后的左上角和右下角坐标
    """
    # 1. 打开视频，获取分辨率
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2. 阈值：左右各保留 10%
    thresh_left  = int(width * 0.1)
    thresh_right = int(width * 0.9)

    # 3. 原始框的最小/最大 x，和最小/最大 y
    xs = [pt[0] for pt in final_box]
    ys = [pt[1] for pt in final_box]
    orig_x_min = min(xs)
    orig_x_max = max(xs)
    y_top      = min(ys)
    y_bottom   = max(ys)

    # 4. 如果原框比 阈值 宽度小则扩张，否则保留
    new_x_left  = min(orig_x_min, thresh_left)
    new_x_right = max(orig_x_max, thresh_right)

    # 5. 构造返回值
    top_left     = [new_x_left,  y_top]
    bottom_right = [new_x_right, y_bottom]

    return top_left, bottom_right, width, height
def is_short_author_voice(video_path, video_duration, merged_timerange_list):
    try:
        # 获取merge_intervals_list中总共的时间
        total_time = 0
        for timerange in merged_timerange_list:

            start_time = time_to_ms(timerange['startTime'])
            end_time = time_to_ms(timerange['endTime'])
            total_time += (end_time - start_time)
        # 计算占比
        proportion = total_time / video_duration
        intervals_count = len(merged_timerange_list)
        if proportion < 0.05 and intervals_count < 3:
            print(f"原作者声音较少，采用合理默认字幕区域。原作者声音占比: {proportion}, 原作者说话数量: {intervals_count} {video_path}")
            return True
    except Exception as e:
        traceback.print_exc()
        print(f"判断原作者声音时失败: {e}")
        return False
    return False

def gen_proper_box(video_path, video_duration, merged_timerange_list):
    try:
        is_short_author = is_short_author_voice(video_path, video_duration, merged_timerange_list)
        if is_short_author:
            video_width, video_height = get_video_dimensions(video_path)
            short_size = min(video_width, video_height)
            # --- 配置参数 ---
            # 左右边距 (例如 5% 的宽度)
            margin_x_ratio = 0.05
            # 上下安全边距 (例如 8% 的高度，防止字幕贴边)
            margin_y_ratio = 0.08
            # 预估字幕框高度 (例如 15% 的高度，足以容纳双行字幕)
            box_height_ratio = 0.15
            margin_x = int(short_size * margin_x_ratio)
            margin_y = int(short_size * margin_y_ratio)
            box_height = int(short_size * box_height_ratio)
            # --- 计算具体像素值 ---
            min_x = margin_x
            max_x = video_width - margin_x
            max_y = video_height - margin_y
            min_y = max_y - box_height # 或者 video_height - margin_y - box_height
            box_bottom = [
                [min_x, min_y],  # 左上 (min_x, min_y)
                [max_x, min_y],  # 右上 (max_x, min_y)
                [max_x, max_y],  # 右下 (max_x, max_y)
                [min_x, max_y]   # 左下 (min_x, max_y)
            ]
            return box_bottom
    except Exception as e:
        traceback.print_exc()
        print(f"生成合理字幕区域失败: {e}")
        return None
    return None


def find_overall_subtitle_box_target_number(
    video_path: str,
    merged_timerange_list: list[dict],
    num_samples: int = 10,
    output_dir = 'temp_dir',
    video_duration_ms: int = 0
):
    """
    主函数，找到包围视频字幕的最小框，并将框绘制到所有抽帧图片上。
    且保证抽取到 num_samples 帧（或所有符合区间的帧）。
    如果字幕框高度超过视频高度的10%，则返回 None。
    :param video_path: 视频文件路径
    :param merged_timerange_list: [{ "startTime": "00:00:00.205", "endTime": "00:00:09.060" }, ...]
    :param num_samples: 希望抽取的帧数
    :return: 最终包围框顶点列表 [[x1, y1], ..., [x4, y4]] 或 None
    """
    os.makedirs(output_dir, exist_ok=True)
    # --- 检查视频文件 ---
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到 '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return None

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or fps <= 0 or video_height <= 0:
        print("错误: 无法获取视频信息，抽帧失败。")
        cap.release()
        return None

    # --- 预处理：将所有时间区间转为毫秒，并对应到帧索引 ---
    valid_frames = set()
    for tr in merged_timerange_list:
        start_ms = time_to_ms(tr['startTime'])
        end_ms = time_to_ms(tr['endTime'])
        if end_ms <= start_ms:
            continue
        # 对应的帧索引区间
        start_idx = int(np.ceil(start_ms / 1000 * fps))
        end_idx   = int(np.floor(end_ms   / 1000 * fps))
        # 限制在 [0, total_frames-1]
        start_idx = max(0, start_idx)
        end_idx   = min(total_frames - 1, end_idx)
        valid_frames.update(range(start_idx, end_idx + 1))

    valid_frames = sorted(valid_frames)
    if not valid_frames:
        print("错误: 没有任何帧落在指定的时间区间内。")
        cap.release()
        return None

    # --- 从 valid_frames 中均匀抽取 num_samples 帧 ---
    num_to_pick = min(num_samples, len(valid_frames))
    step = len(valid_frames) / num_to_pick
    sample_indices = [valid_frames[int(i * step)] for i in range(num_to_pick)]

    # --- 阶段 1: 按 sample_indices 抽帧并保存 ---
    print(f"\n[阶段 1] 从 {len(valid_frames)} 个符合区间的帧中抽取 {len(sample_indices)} 帧...")
    saved_frame_paths = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {idx} 帧。")
            continue
        path = os.path.join(output_dir, f"frame_{idx}.jpg")
        cv2.imwrite(path, frame)
        saved_frame_paths.append(path)
    cap.release()
    # print(f"[阶段 1] 抽帧完成。共保存 {len(saved_frame_paths)} 帧图片。")

    if not saved_frame_paths:
        print("未能提取任何帧。")
        return None


    # --- 阶段 2: 对抽出的帧进行字幕检测 ---
    # print(f"\n[阶段 2] 开始检测 {len(saved_frame_paths)} 张图片的字幕框...")
    is_short_author = is_short_author_voice(video_path, video_duration_ms, merged_timerange_list)
    if is_short_author:
        result_json = run_subtitle_ocr(saved_frame_paths, crop_ratio=0.5)
    else:
        result_json = run_subtitle_ocr(saved_frame_paths, crop_ratio=0.6)

    detected_boxes = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]

    print(f"[阶段 2] 检测完成。共检测到 {len(detected_boxes)} 个字幕框。{video_path}")

    if not detected_boxes:
        print(f"未找到任何字幕框。{video_path}")
        return gen_proper_box(video_path, video_duration_ms, merged_timerange_list)


    # --- 阶段 3: 分析并计算最终包围框 ---
    # print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)

    if not good_boxes:
        print("\n[结果] 所有检测到的框都被过滤为异常值。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return gen_proper_box(video_path, video_duration_ms, merged_timerange_list)

    print(f"[阶段 3] 过滤后剩余 {len(good_boxes)} 个有效字幕框。")

    all_points = np.array([point for box in good_boxes for point in box])
    final_box = [
        [int(np.min(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # min_x, min_y
        [int(np.max(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # max_x, min_y
        [int(np.max(all_points[:, 0])), int(np.max(all_points[:, 1]))],  # max_x, max_y
        [int(np.min(all_points[:, 0])), int(np.max(all_points[:, 1]))]  # min_x, max_y
    ]
    print(f"[阶段 3] 计算出的最终包围框: {final_box}")
    return final_box

# ================= 测试调用 =================
if __name__ == "__main__":
    # 模拟路径 (请确保这些文件存在，或修改为你的真实路径进行测试)
    # 故意包含一个不存在的图片来测试健壮性
    test_images = [
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",
        r"C:\Users\zxh\Desktop\temp\a2.png",

    ]
    # 扫描W:\project\python_project\auto_video\videos\material\7590376286820814107\frame下面所有的png文件
    image_list = []
    for root, dirs, files in os.walk(r"W:\project\python_project\auto_video\videos\material\7590376286820814107\frame"):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(os.path.join(root, file))




    print("\n--- Starting Safe Batch OCR ---")
    # # 这一步绝对不会报错，只会返回 JSON
    for i in range(5):
        print(f"\n--- OCR Attempt {i + 1} ---")
        result_json = run_subtitle_ocr(image_list, crop_ratio=0.5)
        print(result_json)
        # 获取所有的box，保存到box_list
        box_list = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]


